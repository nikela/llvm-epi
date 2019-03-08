//===--- SymbolCollector.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolCollector.h"
#include "AST.h"
#include "CanonicalIncludes.h"
#include "CodeComplete.h"
#include "CodeCompletionStrings.h"
#include "ExpectedTypes.h"
#include "Logger.h"
#include "SourceCode.h"
#include "SymbolLocation.h"
#include "URI.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace clangd {
namespace {

/// If \p ND is a template specialization, returns the described template.
/// Otherwise, returns \p ND.
const NamedDecl &getTemplateOrThis(const NamedDecl &ND) {
  if (auto T = ND.getDescribedTemplate())
    return *T;
  return ND;
}

// Returns a URI of \p Path. Firstly, this makes the \p Path absolute using the
// current working directory of the given SourceManager if the Path is not an
// absolute path. If failed, this resolves relative paths against \p FallbackDir
// to get an absolute path. Then, this tries creating an URI for the absolute
// path with schemes specified in \p Opts. This returns an URI with the first
// working scheme, if there is any; otherwise, this returns None.
//
// The Path can be a path relative to the build directory, or retrieved from
// the SourceManager.
std::string toURI(const SourceManager &SM, llvm::StringRef Path,
                  const SymbolCollector::Options &Opts) {
  llvm::SmallString<128> AbsolutePath(Path);
  if (auto CanonPath =
          getCanonicalPath(SM.getFileManager().getFile(Path), SM)) {
    AbsolutePath = *CanonPath;
  }
  // We don't perform is_absolute check in an else branch because makeAbsolute
  // might return a relative path on some InMemoryFileSystems.
  if (!llvm::sys::path::is_absolute(AbsolutePath) && !Opts.FallbackDir.empty())
    llvm::sys::fs::make_absolute(Opts.FallbackDir, AbsolutePath);
  llvm::sys::path::remove_dots(AbsolutePath, /*remove_dot_dot=*/true);
  return URI::create(AbsolutePath).toString();
}

// All proto generated headers should start with this line.
static const char *PROTO_HEADER_COMMENT =
    "// Generated by the protocol buffer compiler.  DO NOT EDIT!";

// Checks whether the decl is a private symbol in a header generated by
// protobuf compiler.
// To identify whether a proto header is actually generated by proto compiler,
// we check whether it starts with PROTO_HEADER_COMMENT.
// FIXME: make filtering extensible when there are more use cases for symbol
// filters.
bool isPrivateProtoDecl(const NamedDecl &ND) {
  const auto &SM = ND.getASTContext().getSourceManager();
  auto Loc = findNameLoc(&ND);
  auto FileName = SM.getFilename(Loc);
  if (!FileName.endswith(".proto.h") && !FileName.endswith(".pb.h"))
    return false;
  auto FID = SM.getFileID(Loc);
  // Double check that this is an actual protobuf header.
  if (!SM.getBufferData(FID).startswith(PROTO_HEADER_COMMENT))
    return false;

  // ND without identifier can be operators.
  if (ND.getIdentifier() == nullptr)
    return false;
  auto Name = ND.getIdentifier()->getName();
  if (!Name.contains('_'))
    return false;
  // Nested proto entities (e.g. Message::Nested) have top-level decls
  // that shouldn't be used (Message_Nested). Ignore them completely.
  // The nested entities are dangling type aliases, we may want to reconsider
  // including them in the future.
  // For enum constants, SOME_ENUM_CONSTANT is not private and should be
  // indexed. Outer_INNER is private. This heuristic relies on naming style, it
  // will include OUTER_INNER and exclude some_enum_constant.
  // FIXME: the heuristic relies on naming style (i.e. no underscore in
  // user-defined names) and can be improved.
  return (ND.getKind() != Decl::EnumConstant) || llvm::any_of(Name, islower);
}

// We only collect #include paths for symbols that are suitable for global code
// completion, except for namespaces since #include path for a namespace is hard
// to define.
bool shouldCollectIncludePath(index::SymbolKind Kind) {
  using SK = index::SymbolKind;
  switch (Kind) {
  case SK::Macro:
  case SK::Enum:
  case SK::Struct:
  case SK::Class:
  case SK::Union:
  case SK::TypeAlias:
  case SK::Using:
  case SK::Function:
  case SK::Variable:
  case SK::EnumConstant:
    return true;
  default:
    return false;
  }
}

/// Gets a canonical include (URI of the header or <header>  or "header") for
/// header of \p Loc.
/// Returns None if fails to get include header for \p Loc.
llvm::Optional<std::string>
getIncludeHeader(llvm::StringRef QName, const SourceManager &SM,
                 SourceLocation Loc, const SymbolCollector::Options &Opts) {
  std::vector<std::string> Headers;
  // Collect the #include stack.
  while (true) {
    if (!Loc.isValid())
      break;
    auto FilePath = SM.getFilename(Loc);
    if (FilePath.empty())
      break;
    Headers.push_back(FilePath);
    if (SM.isInMainFile(Loc))
      break;
    Loc = SM.getIncludeLoc(SM.getFileID(Loc));
  }
  if (Headers.empty())
    return None;
  llvm::StringRef Header = Headers[0];
  if (Opts.Includes) {
    Header = Opts.Includes->mapHeader(Headers, QName);
    if (Header.startswith("<") || Header.startswith("\""))
      return Header.str();
  }
  return toURI(SM, Header, Opts);
}

// Return the symbol range of the token at \p TokLoc.
std::pair<SymbolLocation::Position, SymbolLocation::Position>
getTokenRange(SourceLocation TokLoc, const SourceManager &SM,
              const LangOptions &LangOpts) {
  auto CreatePosition = [&SM](SourceLocation Loc) {
    auto LSPLoc = sourceLocToPosition(SM, Loc);
    SymbolLocation::Position Pos;
    Pos.setLine(LSPLoc.line);
    Pos.setColumn(LSPLoc.character);
    return Pos;
  };

  auto TokenLength = clang::Lexer::MeasureTokenLength(TokLoc, SM, LangOpts);
  return {CreatePosition(TokLoc),
          CreatePosition(TokLoc.getLocWithOffset(TokenLength))};
}

bool shouldIndexFile(const SourceManager &SM, FileID FID,
                     const SymbolCollector::Options &Opts,
                     llvm::DenseMap<FileID, bool> *FilesToIndexCache) {
  if (!Opts.FileFilter)
    return true;
  auto I = FilesToIndexCache->try_emplace(FID);
  if (I.second)
    I.first->second = Opts.FileFilter(SM, FID);
  return I.first->second;
}

// Return the symbol location of the token at \p TokLoc.
llvm::Optional<SymbolLocation>
getTokenLocation(SourceLocation TokLoc, const SourceManager &SM,
                 const SymbolCollector::Options &Opts,
                 const clang::LangOptions &LangOpts,
                 std::string &FileURIStorage) {
  auto Path = SM.getFilename(TokLoc);
  if (Path.empty())
    return None;
  FileURIStorage = toURI(SM, Path, Opts);
  SymbolLocation Result;
  Result.FileURI = FileURIStorage.c_str();
  auto Range = getTokenRange(TokLoc, SM, LangOpts);
  Result.Start = Range.first;
  Result.End = Range.second;

  return Result;
}

// Checks whether \p ND is a definition of a TagDecl (class/struct/enum/union)
// in a header file, in which case clangd would prefer to use ND as a canonical
// declaration.
// FIXME: handle symbol types that are not TagDecl (e.g. functions), if using
// the first seen declaration as canonical declaration is not a good enough
// heuristic.
bool isPreferredDeclaration(const NamedDecl &ND, index::SymbolRoleSet Roles) {
  const auto &SM = ND.getASTContext().getSourceManager();
  return (Roles & static_cast<unsigned>(index::SymbolRole::Definition)) &&
         isa<TagDecl>(&ND) &&
         !SM.isWrittenInMainFile(SM.getExpansionLoc(ND.getLocation()));
}

RefKind toRefKind(index::SymbolRoleSet Roles) {
  return static_cast<RefKind>(static_cast<unsigned>(RefKind::All) & Roles);
}

template <class T> bool explicitTemplateSpecialization(const NamedDecl &ND) {
  if (const auto *TD = dyn_cast<T>(&ND))
    if (TD->getTemplateSpecializationKind() == TSK_ExplicitSpecialization)
      return true;
  return false;
}

} // namespace

SymbolCollector::SymbolCollector(Options Opts) : Opts(std::move(Opts)) {}

void SymbolCollector::initialize(ASTContext &Ctx) {
  ASTCtx = &Ctx;
  CompletionAllocator = std::make_shared<GlobalCodeCompletionAllocator>();
  CompletionTUInfo =
      llvm::make_unique<CodeCompletionTUInfo>(CompletionAllocator);
}

bool SymbolCollector::shouldCollectSymbol(const NamedDecl &ND,
                                          const ASTContext &ASTCtx,
                                          const Options &Opts,
                                          bool IsMainFileOnly) {
  if (ND.isImplicit())
    return false;
  // Skip anonymous declarations, e.g (anonymous enum/class/struct).
  if (ND.getDeclName().isEmpty())
    return false;

  // Skip main-file symbols if we are not collecting them.
  if (IsMainFileOnly && !Opts.CollectMainFileSymbols)
    return false;

  // Skip symbols in anonymous namespaces in header files.
  if (!IsMainFileOnly && ND.isInAnonymousNamespace())
    return false;

  // We want most things but not "local" symbols such as symbols inside
  // FunctionDecl, BlockDecl, ObjCMethodDecl and OMPDeclareReductionDecl.
  // FIXME: Need a matcher for ExportDecl in order to include symbols declared
  // within an export.
  const auto *DeclCtx = ND.getDeclContext();
  switch (DeclCtx->getDeclKind()) {
  case Decl::TranslationUnit:
  case Decl::Namespace:
  case Decl::LinkageSpec:
  case Decl::Enum:
  case Decl::ObjCProtocol:
  case Decl::ObjCInterface:
  case Decl::ObjCCategory:
  case Decl::ObjCCategoryImpl:
  case Decl::ObjCImplementation:
    break;
  default:
    // Record has a few derivations (e.g. CXXRecord, Class specialization), it's
    // easier to cast.
    if (!isa<RecordDecl>(DeclCtx))
      return false;
  }
  if (explicitTemplateSpecialization<FunctionDecl>(ND) ||
      explicitTemplateSpecialization<CXXRecordDecl>(ND) ||
      explicitTemplateSpecialization<VarDecl>(ND))
    return false;

  // Avoid indexing internal symbols in protobuf generated headers.
  if (isPrivateProtoDecl(ND))
    return false;
  return true;
}

// Always return true to continue indexing.
bool SymbolCollector::handleDeclOccurence(
    const Decl *D, index::SymbolRoleSet Roles,
    llvm::ArrayRef<index::SymbolRelation> Relations, SourceLocation Loc,
    index::IndexDataConsumer::ASTNodeInfo ASTNode) {
  assert(ASTCtx && PP.get() && "ASTContext and Preprocessor must be set.");
  assert(CompletionAllocator && CompletionTUInfo);
  assert(ASTNode.OrigD);
  // If OrigD is an declaration associated with a friend declaration and it's
  // not a definition, skip it. Note that OrigD is the occurrence that the
  // collector is currently visiting.
  if ((ASTNode.OrigD->getFriendObjectKind() !=
       Decl::FriendObjectKind::FOK_None) &&
      !(Roles & static_cast<unsigned>(index::SymbolRole::Definition)))
    return true;
  // Skip non-semantic references, we should start processing these when we
  // decide to implement renaming with index support.
  if ((Roles & static_cast<unsigned>(index::SymbolRole::NameReference)))
    return true;
  // A declaration created for a friend declaration should not be used as the
  // canonical declaration in the index. Use OrigD instead, unless we've already
  // picked a replacement for D
  if (D->getFriendObjectKind() != Decl::FriendObjectKind::FOK_None)
    D = CanonicalDecls.try_emplace(D, ASTNode.OrigD).first->second;
  const NamedDecl *ND = dyn_cast<NamedDecl>(D);
  if (!ND)
    return true;

  // Mark D as referenced if this is a reference coming from the main file.
  // D may not be an interesting symbol, but it's cheaper to check at the end.
  auto &SM = ASTCtx->getSourceManager();
  auto SpellingLoc = SM.getSpellingLoc(Loc);
  if (Opts.CountReferences &&
      (Roles & static_cast<unsigned>(index::SymbolRole::Reference)) &&
      SM.getFileID(SpellingLoc) == SM.getMainFileID())
    ReferencedDecls.insert(ND);

  bool CollectRef = static_cast<unsigned>(Opts.RefFilter) & Roles;
  bool IsOnlyRef =
      !(Roles & (static_cast<unsigned>(index::SymbolRole::Declaration) |
                 static_cast<unsigned>(index::SymbolRole::Definition)));

  if (IsOnlyRef && !CollectRef)
    return true;

  // ND is the canonical (i.e. first) declaration. If it's in the main file,
  // then no public declaration was visible, so assume it's main-file only.
  bool IsMainFileOnly = SM.isWrittenInMainFile(SM.getExpansionLoc(
    ND->getBeginLoc()));
  if (!shouldCollectSymbol(*ND, *ASTCtx, Opts, IsMainFileOnly))
    return true;
  // Do not store references to main-file symbols.
  if (CollectRef && !IsMainFileOnly && !isa<NamespaceDecl>(ND) &&
      (Opts.RefsInHeaders || SM.getFileID(SpellingLoc) == SM.getMainFileID()))
    DeclRefs[ND].emplace_back(SpellingLoc, Roles);
  // Don't continue indexing if this is a mere reference.
  if (IsOnlyRef)
    return true;

  auto ID = getSymbolID(ND);
  if (!ID)
    return true;

  // FIXME: ObjCPropertyDecl are not properly indexed here:
  // - ObjCPropertyDecl may have an OrigD of ObjCPropertyImplDecl, which is
  // not a NamedDecl.
  auto *OriginalDecl = dyn_cast<NamedDecl>(ASTNode.OrigD);
  if (!OriginalDecl)
    return true;

  const Symbol *BasicSymbol = Symbols.find(*ID);
  if (!BasicSymbol) // Regardless of role, ND is the canonical declaration.
    BasicSymbol = addDeclaration(*ND, std::move(*ID), IsMainFileOnly);
  else if (isPreferredDeclaration(*OriginalDecl, Roles))
    // If OriginalDecl is preferred, replace the existing canonical
    // declaration (e.g. a class forward declaration). There should be at most
    // one duplicate as we expect to see only one preferred declaration per
    // TU, because in practice they are definitions.
    BasicSymbol = addDeclaration(*OriginalDecl, std::move(*ID), IsMainFileOnly);

  if (Roles & static_cast<unsigned>(index::SymbolRole::Definition))
    addDefinition(*OriginalDecl, *BasicSymbol);
  return true;
}

bool SymbolCollector::handleMacroOccurence(const IdentifierInfo *Name,
                                           const MacroInfo *MI,
                                           index::SymbolRoleSet Roles,
                                           SourceLocation Loc) {
  if (!Opts.CollectMacro)
    return true;
  assert(PP.get());

  const auto &SM = PP->getSourceManager();
  auto DefLoc = MI->getDefinitionLoc();

  // Header guards are not interesting in index. Builtin macros don't have
  // useful locations and are not needed for code completions.
  if (MI->isUsedForHeaderGuard() || MI->isBuiltinMacro())
    return true;

  // Skip main-file symbols if we are not collecting them.
  bool IsMainFileSymbol = SM.isInMainFile(SM.getExpansionLoc(DefLoc));
  if (IsMainFileSymbol && !Opts.CollectMainFileSymbols)
    return false;

  // Also avoid storing predefined macros like __DBL_MIN__.
  if (SM.isWrittenInBuiltinFile(DefLoc))
    return true;

  // Mark the macro as referenced if this is a reference coming from the main
  // file. The macro may not be an interesting symbol, but it's cheaper to check
  // at the end.
  if (Opts.CountReferences &&
      (Roles & static_cast<unsigned>(index::SymbolRole::Reference)) &&
      SM.getFileID(SM.getSpellingLoc(Loc)) == SM.getMainFileID())
    ReferencedMacros.insert(Name);
  // Don't continue indexing if this is a mere reference.
  // FIXME: remove macro with ID if it is undefined.
  if (!(Roles & static_cast<unsigned>(index::SymbolRole::Declaration) ||
        Roles & static_cast<unsigned>(index::SymbolRole::Definition)))
    return true;

  auto ID = getSymbolID(*Name, MI, SM);
  if (!ID)
    return true;

  // Only collect one instance in case there are multiple.
  if (Symbols.find(*ID) != nullptr)
    return true;

  Symbol S;
  S.ID = std::move(*ID);
  S.Name = Name->getName();
  if (!IsMainFileSymbol) {
    S.Flags |= Symbol::IndexedForCodeCompletion;
    S.Flags |= Symbol::VisibleOutsideFile;
  }
  S.SymInfo = index::getSymbolInfoForMacro(*MI);
  std::string FileURI;
  // FIXME: use the result to filter out symbols.
  shouldIndexFile(SM, SM.getFileID(Loc), Opts, &FilesToIndexCache);
  if (auto DeclLoc =
          getTokenLocation(DefLoc, SM, Opts, PP->getLangOpts(), FileURI))
    S.CanonicalDeclaration = *DeclLoc;

  CodeCompletionResult SymbolCompletion(Name);
  const auto *CCS = SymbolCompletion.CreateCodeCompletionStringForMacro(
      *PP, *CompletionAllocator, *CompletionTUInfo);
  std::string Signature;
  std::string SnippetSuffix;
  getSignature(*CCS, &Signature, &SnippetSuffix);

  std::string Include;
  if (Opts.CollectIncludePath && shouldCollectIncludePath(S.SymInfo.Kind)) {
    if (auto Header = getIncludeHeader(Name->getName(), SM,
                                       SM.getExpansionLoc(DefLoc), Opts))
      Include = std::move(*Header);
  }
  S.Signature = Signature;
  S.CompletionSnippetSuffix = SnippetSuffix;
  if (!Include.empty())
    S.IncludeHeaders.emplace_back(Include, 1);

  Symbols.insert(S);
  return true;
}

void SymbolCollector::finish() {
  // At the end of the TU, add 1 to the refcount of all referenced symbols.
  auto IncRef = [this](const SymbolID &ID) {
    if (const auto *S = Symbols.find(ID)) {
      Symbol Inc = *S;
      ++Inc.References;
      Symbols.insert(Inc);
    }
  };
  for (const NamedDecl *ND : ReferencedDecls) {
    if (auto ID = getSymbolID(ND)) {
      IncRef(*ID);
    }
  }
  if (Opts.CollectMacro) {
    assert(PP);
    for (const IdentifierInfo *II : ReferencedMacros) {
      if (const auto *MI = PP->getMacroDefinition(II).getMacroInfo())
        if (auto ID = getSymbolID(*II, MI, PP->getSourceManager()))
          IncRef(*ID);
    }
  }

  const auto &SM = ASTCtx->getSourceManager();
  llvm::DenseMap<FileID, std::string> URICache;
  auto GetURI = [&](FileID FID) -> llvm::Optional<std::string> {
    auto Found = URICache.find(FID);
    if (Found == URICache.end()) {
      if (auto *FileEntry = SM.getFileEntryForID(FID)) {
        auto FileURI = toURI(SM, FileEntry->getName(), Opts);
        Found = URICache.insert({FID, FileURI}).first;
      } else {
        // Ignore cases where we can not find a corresponding file entry
        // for the loc, thoses are not interesting, e.g. symbols formed
        // via macro concatenation.
        return None;
      }
    }
    return Found->second;
  };

  if (auto MainFileURI = GetURI(SM.getMainFileID())) {
    for (const auto &It : DeclRefs) {
      if (auto ID = getSymbolID(It.first)) {
        for (const auto &LocAndRole : It.second) {
          auto FileID = SM.getFileID(LocAndRole.first);
          // FIXME: use the result to filter out references.
          shouldIndexFile(SM, FileID, Opts, &FilesToIndexCache);
          if (auto FileURI = GetURI(FileID)) {
            auto Range =
                getTokenRange(LocAndRole.first, SM, ASTCtx->getLangOpts());
            Ref R;
            R.Location.Start = Range.first;
            R.Location.End = Range.second;
            R.Location.FileURI = FileURI->c_str();
            R.Kind = toRefKind(LocAndRole.second);
            Refs.insert(*ID, R);
          }
        }
      }
    }
  }

  ReferencedDecls.clear();
  ReferencedMacros.clear();
  DeclRefs.clear();
  FilesToIndexCache.clear();
}

const Symbol *SymbolCollector::addDeclaration(const NamedDecl &ND,
                                              SymbolID ID,
                                              bool IsMainFileOnly) {
  auto &Ctx = ND.getASTContext();
  auto &SM = Ctx.getSourceManager();

  Symbol S;
  S.ID = std::move(ID);
  std::string QName = printQualifiedName(ND);
  std::tie(S.Scope, S.Name) = splitQualifiedName(QName);
  // FIXME: this returns foo:bar: for objective-C methods, we prefer only foo:
  // for consistency with CodeCompletionString and a clean name/signature split.

  // We collect main-file symbols, but do not use them for code completion.
  if (!IsMainFileOnly && isIndexedForCodeCompletion(ND, Ctx))
    S.Flags |= Symbol::IndexedForCodeCompletion;
  if (isImplementationDetail(&ND))
    S.Flags |= Symbol::ImplementationDetail;
  if (!IsMainFileOnly)
    S.Flags |= Symbol::VisibleOutsideFile;
  S.SymInfo = index::getSymbolInfo(&ND);
  std::string FileURI;
  auto Loc = findNameLoc(&ND);
  // FIXME: use the result to filter out symbols.
  shouldIndexFile(SM, SM.getFileID(Loc), Opts, &FilesToIndexCache);
  if (auto DeclLoc =
          getTokenLocation(Loc, SM, Opts, ASTCtx->getLangOpts(), FileURI))
    S.CanonicalDeclaration = *DeclLoc;

  S.Origin = Opts.Origin;
  if (ND.getAvailability() == AR_Deprecated)
    S.Flags |= Symbol::Deprecated;

  // Add completion info.
  // FIXME: we may want to choose a different redecl, or combine from several.
  assert(ASTCtx && PP.get() && "ASTContext and Preprocessor must be set.");
  // We use the primary template, as clang does during code completion.
  CodeCompletionResult SymbolCompletion(&getTemplateOrThis(ND), 0);
  const auto *CCS = SymbolCompletion.CreateCodeCompletionString(
      *ASTCtx, *PP, CodeCompletionContext::CCC_Symbol, *CompletionAllocator,
      *CompletionTUInfo,
      /*IncludeBriefComments*/ false);
  std::string Documentation =
      formatDocumentation(*CCS, getDocComment(Ctx, SymbolCompletion,
                                              /*CommentsFromHeaders=*/true));
  if (!(S.Flags & Symbol::IndexedForCodeCompletion)) {
    if (Opts.StoreAllDocumentation)
      S.Documentation = Documentation;
    Symbols.insert(S);
    return Symbols.find(S.ID);
  }
  S.Documentation = Documentation;
  std::string Signature;
  std::string SnippetSuffix;
  getSignature(*CCS, &Signature, &SnippetSuffix);
  S.Signature = Signature;
  S.CompletionSnippetSuffix = SnippetSuffix;
  std::string ReturnType = getReturnType(*CCS);
  S.ReturnType = ReturnType;

  std::string Include;
  if (Opts.CollectIncludePath && shouldCollectIncludePath(S.SymInfo.Kind)) {
    // Use the expansion location to get the #include header since this is
    // where the symbol is exposed.
    if (auto Header = getIncludeHeader(
            QName, SM, SM.getExpansionLoc(ND.getLocation()), Opts))
      Include = std::move(*Header);
  }
  if (!Include.empty())
    S.IncludeHeaders.emplace_back(Include, 1);

  llvm::Optional<OpaqueType> TypeStorage;
  if (S.Flags & Symbol::IndexedForCodeCompletion) {
    TypeStorage = OpaqueType::fromCompletionResult(*ASTCtx, SymbolCompletion);
    if (TypeStorage)
      S.Type = TypeStorage->raw();
  }

  Symbols.insert(S);
  return Symbols.find(S.ID);
}

void SymbolCollector::addDefinition(const NamedDecl &ND,
                                    const Symbol &DeclSym) {
  if (DeclSym.Definition)
    return;
  // If we saw some forward declaration, we end up copying the symbol.
  // This is not ideal, but avoids duplicating the "is this a definition" check
  // in clang::index. We should only see one definition.
  Symbol S = DeclSym;
  std::string FileURI;
  auto Loc = findNameLoc(&ND);
  const auto &SM = ND.getASTContext().getSourceManager();
  // FIXME: use the result to filter out symbols.
  shouldIndexFile(SM, SM.getFileID(Loc), Opts, &FilesToIndexCache);
  if (auto DefLoc =
          getTokenLocation(Loc, SM, Opts, ASTCtx->getLangOpts(), FileURI))
    S.Definition = *DefLoc;
  Symbols.insert(S);
}

} // namespace clangd
} // namespace clang
