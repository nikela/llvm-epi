//===-- Flang.cpp - Flang+LLVM ToolChain Implementations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "Flang.h"
#include "CommonArgs.h"
#include "Arch/RISCV.h"

#include "clang/Driver/Options.h"

#include <cassert>

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

/// Add -x lang to \p CmdArgs for \p Input.
static void addDashXForInput(const ArgList &Args, const InputInfo &Input,
                             ArgStringList &CmdArgs) {
  CmdArgs.push_back("-x");
  // Map the driver type to the frontend type.
  CmdArgs.push_back(types::getTypeName(Input.getType()));
}

void Flang::AddFortranDialectOptions(const ArgList &Args,
                                     ArgStringList &CmdArgs) const {
  Args.AddAllArgs(
      CmdArgs, {options::OPT_ffixed_form, options::OPT_ffree_form,
                options::OPT_ffixed_line_length_EQ, options::OPT_fopenmp,
                options::OPT_fopenacc, options::OPT_finput_charset_EQ,
                options::OPT_fimplicit_none, options::OPT_fno_implicit_none,
                options::OPT_fbackslash, options::OPT_fno_backslash,
                options::OPT_flogical_abbreviations,
                options::OPT_fno_logical_abbreviations,
                options::OPT_fxor_operator, options::OPT_fno_xor_operator,
                options::OPT_falternative_parameter_statement,
                options::OPT_fdefault_real_8, options::OPT_fdefault_integer_8,
                options::OPT_fdefault_double_8, options::OPT_flarge_sizes,
                options::OPT_fno_automatic});
}

void Flang::AddPreprocessingOptions(const ArgList &Args,
                                    ArgStringList &CmdArgs) const {
  Args.AddAllArgs(CmdArgs,
                  {options::OPT_P, options::OPT_D, options::OPT_U,
                   options::OPT_I, options::OPT_cpp, options::OPT_nocpp});
}

void Flang::AddOtherOptions(const ArgList &Args, ArgStringList &CmdArgs) const {
  Args.AddAllArgs(CmdArgs,
                  {options::OPT_module_dir, options::OPT_fdebug_module_writer,
                   options::OPT_fintrinsic_modules_path, options::OPT_pedantic,
                   options::OPT_std_EQ, options::OPT_W_Joined});
}

static bool shouldEnableVectorizerAtOLevel(const ArgList &Args, bool isSlpVec) {
  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    if (A->getOption().matches(options::OPT_O4) ||
        A->getOption().matches(options::OPT_Ofast))
      return true;

    if (A->getOption().matches(options::OPT_O0))
      return false;

    assert(A->getOption().matches(options::OPT_O) && "Must have a -O flag");

    // Vectorize -Os.
    StringRef S(A->getValue());
    if (S == "s")
      return true;

    // Don't vectorize -Oz, unless it's the slp vectorizer.
    if (S == "z")
      return isSlpVec;

    unsigned OptLevel = 0;
    if (S.getAsInteger(10, OptLevel))
      return false;

    return OptLevel > 1;
  }

  return false;
}

void Flang::AddCodeGenOptions(const ArgList &Args,
                              ArgStringList &CmdArgs) const {
  Args.AddAllArgs(CmdArgs, options::OPT_O);

  // Enable vectorization per default according to the optimization level
  // selected. For optimization levels that want vectorization we use the alias
  // option to simplify the hasFlag logic.
  bool EnableVec = shouldEnableVectorizerAtOLevel(Args, false);
  OptSpecifier VectorizeAliasOption =
      EnableVec ? options::OPT_O_Group : options::OPT_fvectorize;
  if (Args.hasFlag(options::OPT_fvectorize, VectorizeAliasOption,
                   options::OPT_fno_vectorize, EnableVec))
    CmdArgs.push_back("-vectorize-loops");

  // -fslp-vectorize is enabled based on the optimization level selected.
  bool EnableSLPVec = shouldEnableVectorizerAtOLevel(Args, true);
  OptSpecifier SLPVectAliasOption =
      EnableSLPVec ? options::OPT_O_Group : options::OPT_fslp_vectorize;
  if (Args.hasFlag(options::OPT_fslp_vectorize, SLPVectAliasOption,
                   options::OPT_fno_slp_vectorize, EnableSLPVec))
    CmdArgs.push_back("-vectorize-slp");

  Args.AddLastArg(CmdArgs, options::OPT_funroll_loops,
                  options::OPT_fno_unroll_loops);
}

static const char *RelocationModelName(llvm::Reloc::Model Model) {
  switch (Model) {
  case llvm::Reloc::Static:
    return "static";
  case llvm::Reloc::PIC_:
    return "pic";
  case llvm::Reloc::DynamicNoPIC:
    return "dynamic-no-pic";
  case llvm::Reloc::ROPI:
    return "ropi";
  case llvm::Reloc::RWPI:
    return "rwpi";
  case llvm::Reloc::ROPI_RWPI:
    return "ropi-rwpi";
  }
  llvm_unreachable("Unknown Reloc::Model kind");
}

void Flang::AddCodeModelOptions(const ArgList &Args,
                                ArgStringList &CmdArgs) const {
  // FIXME: Some of this stuff is copied from Clang.cpp.
  // Handle -fPIC et al -- the relocation-model affects the assembler
  // for some targets.
  llvm::Reloc::Model RelocationModel;
  unsigned PICLevel;
  bool IsPIE;
  std::tie(RelocationModel, PICLevel, IsPIE) =
      ParsePICArgs(getToolChain(), Args);

  if (PICLevel > 0) {
    CmdArgs.push_back("-pic-level");
    CmdArgs.push_back(PICLevel == 1 ? "1" : "2");
    if (IsPIE)
      CmdArgs.push_back("-pic-is-pie");
  }

  const char *RMName = RelocationModelName(RelocationModel);
  if (RMName) {
    CmdArgs.push_back("-mrelocation-model");
    CmdArgs.push_back(RMName);
  }
}

void Flang::AddRISCVTargetArgs(const ArgList &Args,
                               ArgStringList &CmdArgs) const {
  if (Args.hasArg(options::OPT_mepi)) {
    CmdArgs.push_back("-mepi");

    // We are only interested in scalable vectorization in EPI.
    CmdArgs.push_back("-mllvm");
    CmdArgs.push_back("--scalable-vectorization=only");

    // Predicates are preferred when vectorising in EPI.
    CmdArgs.push_back("-mllvm");
    if (Args.hasArg(options::OPT_mno_prefer_predicate_over_epilog))
      CmdArgs.push_back("--prefer-predicate-over-epilogue=scalar-epilogue");
    else
      CmdArgs.push_back(
          "--prefer-predicate-over-epilogue=predicate-dont-vectorize");

    CmdArgs.push_back("-mllvm");
    CmdArgs.push_back("-epi-pipeline");

    // In EPI we assume vectors of at least 64 bits, even if the spec
    // says >128. Our implementation predates this and we can't change
    // that now. Should impact only the vectorizer, mostly.
    CmdArgs.push_back("-mllvm");
    CmdArgs.push_back("-riscv-v-vector-bits-min=64");

    // IndVarSimplify will not expand the loop count because we do not have
    // an implementation of getCastInstrCost, a zext appears that is given
    // a cost of 1 and then this may exceed the default budget of 4. Raise
    // the budget to 8.
    CmdArgs.push_back("-mllvm");
    CmdArgs.push_back("-scev-cheap-expansion-budget=8");
  }
}

static void getTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                              const ArgList &Args, ArgStringList &CmdArgs,
                              bool ForAS, bool IsAux = false) {
  std::vector<StringRef> Features;
  switch (Triple.getArch()) {
  default:
    break;
  case llvm::Triple::riscv64:
    riscv::getRISCVTargetFeatures(D, Triple, Args, Features);
    break;
  }

  for (auto Feature : unifyTargetFeatures(Features)) {
    CmdArgs.push_back(IsAux ? "-aux-target-feature" : "-target-feature");
    CmdArgs.push_back(Feature.data());
  }
}

void Flang::RenderTargetOptions(const llvm::Triple &EffectiveTriple,
                                const ArgList &Args,
                                ArgStringList &CmdArgs) const {
  const ToolChain &TC = getToolChain();

  // Add the target features
  getTargetFeatures(TC.getDriver(), EffectiveTriple, Args, CmdArgs, false);

  // Add target specific flags.
  switch (TC.getArch()) {
  default:
    break;
  case llvm::Triple::riscv64:
    AddRISCVTargetArgs(Args, CmdArgs);
    break;
  }
}

void Flang::ConstructJob(Compilation &C, const JobAction &JA,
                         const InputInfo &Output, const InputInfoList &Inputs,
                         const ArgList &Args, const char *LinkingOutput) const {
  const auto &TC = getToolChain();
  const llvm::Triple &Triple = TC.getEffectiveTriple();
  const std::string &TripleStr = Triple.getTriple();

  ArgStringList CmdArgs;

  // Invoke ourselves in -fc1 mode.
  CmdArgs.push_back("-fc1");

  // Add the "effective" target triple.
  CmdArgs.push_back("-triple");
  CmdArgs.push_back(Args.MakeArgString(TripleStr));

  if (isa<PreprocessJobAction>(JA)) {
      CmdArgs.push_back("-E");
  } else if (isa<CompileJobAction>(JA) || isa<BackendJobAction>(JA)) {
    if (JA.getType() == types::TY_Nothing) {
      CmdArgs.push_back("-fsyntax-only");
    } else if (JA.getType() == types::TY_AST) {
      CmdArgs.push_back("-emit-ast");
    } else if (JA.getType() == types::TY_LLVM_IR ||
               JA.getType() == types::TY_LTO_IR) {
      CmdArgs.push_back("-emit-llvm");
    } else if (JA.getType() == types::TY_LLVM_BC ||
               JA.getType() == types::TY_LTO_BC) {
      CmdArgs.push_back("-emit-llvm-bc");
    } else if (JA.getType() == types::TY_PP_Asm) {
      CmdArgs.push_back("-S");
    } else if (JA.getType() == types::TY_LLVM_MLIR) {
      CmdArgs.push_back("-emit-mlir");
    } else {
      assert(false && "Unexpected output type!");
    }
  } else if (isa<AssembleJobAction>(JA)) {
    CmdArgs.push_back("-emit-obj");
  } else {
    assert(false && "Unexpected action class for Flang tool.");
  }

  const InputInfo &Input = Inputs[0];
  types::ID InputType = Input.getType();

  // Add preprocessing options like -I, -D, etc. if we are using the
  // preprocessor (i.e. skip when dealing with e.g. binary files).
  if (types::getPreprocessedType(InputType) != types::TY_INVALID)
    AddPreprocessingOptions(Args, CmdArgs);

  AddFortranDialectOptions(Args, CmdArgs);

  // Code generation options.
  AddCodeGenOptions(Args, CmdArgs);

  // Code model options.
  AddCodeModelOptions(Args, CmdArgs);

  // Add other compile options
  AddOtherOptions(Args, CmdArgs);

  // Forward -Xflang arguments to -fc1
  Args.AddAllArgValues(CmdArgs, options::OPT_Xflang);

  // Forward -mllvm options to the LLVM option parser. In practice, this means
  // forwarding to `-fc1` as that's where the LLVM parser is run.
  for (const Arg *A : Args.filtered(options::OPT_mllvm)) {
    A->claim();
    A->render(Args, CmdArgs);
  }

  for (const Arg *A : Args.filtered(options::OPT_mmlir)) {
    A->claim();
    A->render(Args, CmdArgs);
  }

  RenderTargetOptions(Triple, Args, CmdArgs);

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  assert(Input.isFilename() && "Invalid input.");

  addDashXForInput(Args, Input, CmdArgs);

  CmdArgs.push_back(Input.getFilename());

  const auto& D = C.getDriver();
  // TODO: Replace flang-new with flang once the new driver replaces the
  // throwaway driver
  const char *Exec = Args.MakeArgString(D.GetProgramPath("flang-new", TC));
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileUTF8(),
                                         Exec, CmdArgs, Inputs, Output));
}

Flang::Flang(const ToolChain &TC) : Tool("flang-new", "flang frontend", TC) {}

Flang::~Flang() {}
