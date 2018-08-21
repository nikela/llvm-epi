//===--- MemIndex.cpp - Dynamic in-memory symbol index. ----------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//

#include "MemIndex.h"
#include "../FuzzyMatch.h"
#include "../Logger.h"
#include <queue>

namespace clang {
namespace clangd {

void MemIndex::build(std::shared_ptr<std::vector<const Symbol *>> Syms) {
  llvm::DenseMap<SymbolID, const Symbol *> TempIndex;
  for (const Symbol *Sym : *Syms)
    TempIndex[Sym->ID] = Sym;

  // Swap out the old symbols and index.
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    Index = std::move(TempIndex);
    Symbols = std::move(Syms); // Relase old symbols.
  }
}

std::unique_ptr<SymbolIndex> MemIndex::build(SymbolSlab Slab) {
  auto Idx = llvm::make_unique<MemIndex>();
  Idx->build(getSymbolsFromSlab(std::move(Slab)));
  return std::move(Idx);
}

bool MemIndex::fuzzyFind(
    const FuzzyFindRequest &Req,
    llvm::function_ref<void(const Symbol &)> Callback) const {
  assert(!StringRef(Req.Query).contains("::") &&
         "There must be no :: in query.");

  std::priority_queue<std::pair<float, const Symbol *>> Top;
  FuzzyMatcher Filter(Req.Query);
  bool More = false;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    for (const auto Pair : Index) {
      const Symbol *Sym = Pair.second;

      // Exact match against all possible scopes.
      if (!Req.Scopes.empty() && !llvm::is_contained(Req.Scopes, Sym->Scope))
        continue;
      if (Req.RestrictForCodeCompletion && !Sym->IsIndexedForCodeCompletion)
        continue;

      if (auto Score = Filter.match(Sym->Name)) {
        Top.emplace(-*Score * quality(*Sym), Sym);
        if (Top.size() > Req.MaxCandidateCount) {
          More = true;
          Top.pop();
        }
      }
    }
    for (; !Top.empty(); Top.pop())
      Callback(*Top.top().second);
  }
  return More;
}

void MemIndex::lookup(const LookupRequest &Req,
                      llvm::function_ref<void(const Symbol &)> Callback) const {
  std::lock_guard<std::mutex> Lock(Mutex);
  for (const auto &ID : Req.IDs) {
    auto I = Index.find(ID);
    if (I != Index.end())
      Callback(*I->second);
  }
}

void MemIndex::findOccurrences(
    const OccurrencesRequest &Req,
    llvm::function_ref<void(const SymbolOccurrence &)> Callback) const {
  log("findOccurrences is not implemented.");
}

std::shared_ptr<std::vector<const Symbol *>>
getSymbolsFromSlab(SymbolSlab Slab) {
  struct Snapshot {
    SymbolSlab Slab;
    std::vector<const Symbol *> Pointers;
  };
  auto Snap = std::make_shared<Snapshot>();
  Snap->Slab = std::move(Slab);
  for (auto &Sym : Snap->Slab)
    Snap->Pointers.push_back(&Sym);
  return std::shared_ptr<std::vector<const Symbol *>>(std::move(Snap),
                                                      &Snap->Pointers);
}

} // namespace clangd
} // namespace clang
