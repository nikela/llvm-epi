//===-- Flang.cpp - Flang+LLVM ToolChain Implementations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Flang.h"
#include "Arch/AArch64.h"
#include "Arch/ARM.h"
#include "Arch/RISCV.h"
#include "Arch/X86.h"
#include "CommonArgs.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Types.h"
#include "llvm/Support/Host.h"

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

void Flang::addFortranDialectOptions(const ArgList &Args,
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

void Flang::addPreprocessingOptions(const ArgList &Args,
                                    ArgStringList &CmdArgs) const {
  Args.AddAllArgs(CmdArgs,
                  {options::OPT_P, options::OPT_D, options::OPT_U,
                   options::OPT_I, options::OPT_cpp, options::OPT_nocpp});
}

void Flang::addOtherOptions(const ArgList &Args, ArgStringList &CmdArgs) const {
  Args.AddAllArgs(CmdArgs,
                  {options::OPT_module_dir, options::OPT_fdebug_module_writer,
                   options::OPT_fintrinsic_modules_path, options::OPT_pedantic,
                   options::OPT_std_EQ, options::OPT_W_Joined,
                   options::OPT_fconvert_EQ, options::OPT_fpass_plugin_EQ});
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

static bool mustUseNonLeafFramePointerForTarget(const llvm::Triple &Triple) {
  switch (Triple.getArch()){
  default:
    return false;
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    // ARM Darwin targets require a frame pointer to be always present to aid
    // offline debugging via backtraces.
    return Triple.isOSDarwin();
  }
}

static bool useFramePointerForTargetByDefault(const ArgList &Args,
                                              const llvm::Triple &Triple) {
  if (Args.hasArg(options::OPT_pg) && !Args.hasArg(options::OPT_mfentry))
    return true;

  switch (Triple.getArch()) {
  case llvm::Triple::xcore:
  case llvm::Triple::wasm32:
  case llvm::Triple::wasm64:
  case llvm::Triple::msp430:
    // XCore never wants frame pointers, regardless of OS.
    // WebAssembly never wants frame pointers.
    return false;
  case llvm::Triple::ppc:
  case llvm::Triple::ppcle:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
  case llvm::Triple::riscv32:
  case llvm::Triple::riscv64:
  case llvm::Triple::amdgcn:
  case llvm::Triple::r600:
  case llvm::Triple::csky:
    return !areOptimizationsEnabled(Args);
  default:
    break;
  }

  if (Triple.isOSNetBSD()) {
    return !areOptimizationsEnabled(Args);
  }

  if (Triple.isOSLinux() || Triple.getOS() == llvm::Triple::CloudABI ||
      Triple.isOSHurd()) {
    switch (Triple.getArch()) {
    // Don't use a frame pointer on linux if optimizing for certain targets.
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
      if (Triple.isAndroid())
        return true;
      LLVM_FALLTHROUGH;
    case llvm::Triple::mips64:
    case llvm::Triple::mips64el:
    case llvm::Triple::mips:
    case llvm::Triple::mipsel:
    case llvm::Triple::systemz:
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      return !areOptimizationsEnabled(Args);
    default:
      return true;
    }
  }

  if (Triple.isOSWindows()) {
    switch (Triple.getArch()) {
    case llvm::Triple::x86:
      return !areOptimizationsEnabled(Args);
    case llvm::Triple::x86_64:
      return Triple.isOSBinFormatMachO();
    case llvm::Triple::arm:
    case llvm::Triple::thumb:
      // Windows on ARM builds with FPO disabled to aid fast stack walking
      return true;
    default:
      // All other supported Windows ISAs use xdata unwind information, so frame
      // pointers are not generally useful.
      return false;
    }
  }

  return true;
}

static CodeGenOptions::FramePointerKind
getFramePointerKind(const ArgList &Args, const llvm::Triple &Triple) {
  // We have 4 states:
  //
  //  00) leaf retained, non-leaf retained
  //  01) leaf retained, non-leaf omitted (this is invalid)
  //  10) leaf omitted, non-leaf retained
  //      (what -momit-leaf-frame-pointer was designed for)
  //  11) leaf omitted, non-leaf omitted
  //
  //  "omit" options taking precedence over "no-omit" options is the only way
  //  to make 3 valid states representable
  Arg *A = Args.getLastArg(options::OPT_fomit_frame_pointer,
                           options::OPT_fno_omit_frame_pointer);
  bool OmitFP = A && A->getOption().matches(options::OPT_fomit_frame_pointer);
  bool NoOmitFP =
      A && A->getOption().matches(options::OPT_fno_omit_frame_pointer);
  bool OmitLeafFP =
      Args.hasFlag(options::OPT_momit_leaf_frame_pointer,
                   options::OPT_mno_omit_leaf_frame_pointer,
                   Triple.isAArch64() || Triple.isPS() || Triple.isVE());
  if (NoOmitFP || mustUseNonLeafFramePointerForTarget(Triple) ||
      (!OmitFP && useFramePointerForTargetByDefault(Args, Triple))) {
    if (OmitLeafFP)
      return CodeGenOptions::FramePointerKind::NonLeaf;
    return CodeGenOptions::FramePointerKind::All;
  }
  return CodeGenOptions::FramePointerKind::None;
}


void Flang::AddCodeModelOptions(const ArgList &Args,
                                ArgStringList &CmdArgs,
                                const llvm::Triple &Triple) const {
  CodeGenOptions::FramePointerKind FPKeepKind =
      getFramePointerKind(Args, Triple);
  const char *FPKeepKindStr = nullptr;
  switch (FPKeepKind) {
  case CodeGenOptions::FramePointerKind::None:
    FPKeepKindStr = "-mframe-pointer=none";
    break;
  case CodeGenOptions::FramePointerKind::NonLeaf:
    FPKeepKindStr = "-mframe-pointer=non-leaf";
    break;
  case CodeGenOptions::FramePointerKind::All:
    FPKeepKindStr = "-mframe-pointer=all";
    break;
  }
  assert(FPKeepKindStr && "unknown FramePointerKind");
  CmdArgs.push_back(FPKeepKindStr);
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

namespace {
void RenderAArch64ABI(const llvm::Triple &Triple, const ArgList &Args,
                      ArgStringList &CmdArgs) {
  const char *ABIName = nullptr;
  if (Arg *A = Args.getLastArg(options::OPT_mabi_EQ))
    ABIName = A->getValue();
  else if (Triple.isOSDarwin())
    ABIName = "darwinpcs";
  else
    ABIName = "aapcs";

  CmdArgs.push_back("-target-abi");
  CmdArgs.push_back(ABIName);
}
}

static void CollectARMPACBTIOptions(const ToolChain &TC, const ArgList &Args,
                                    ArgStringList &CmdArgs, bool isAArch64) {
  const Arg *A = isAArch64
                     ? Args.getLastArg(options::OPT_msign_return_address_EQ,
                                       options::OPT_mbranch_protection_EQ)
                     : Args.getLastArg(options::OPT_mbranch_protection_EQ);
  if (!A)
    return;

  const Driver &D = TC.getDriver();
  const llvm::Triple &Triple = TC.getEffectiveTriple();
  if (!(isAArch64 || (Triple.isArmT32() && Triple.isArmMClass())))
    D.Diag(diag::warn_incompatible_branch_protection_option)
        << Triple.getArchName();

  StringRef Scope, Key;
  bool IndirectBranches;

  if (A->getOption().matches(options::OPT_msign_return_address_EQ)) {
    Scope = A->getValue();
    if (Scope != "none" && Scope != "non-leaf" && Scope != "all")
      D.Diag(diag::err_drv_unsupported_option_argument)
          << A->getOption().getName() << Scope;
    Key = "a_key";
    IndirectBranches = false;
  } else {
    StringRef DiagMsg;
    llvm::ARM::ParsedBranchProtection PBP;
    if (!llvm::ARM::parseBranchProtection(A->getValue(), PBP, DiagMsg))
      D.Diag(diag::err_drv_unsupported_option_argument)
          << A->getOption().getName() << DiagMsg;
    if (!isAArch64 && PBP.Key == "b_key")
      D.Diag(diag::warn_unsupported_branch_protection)
          << "b-key" << A->getAsString(Args);
    Scope = PBP.Scope;
    Key = PBP.Key;
    IndirectBranches = PBP.BranchTargetEnforcement;
  }

  CmdArgs.push_back(
      Args.MakeArgString(Twine("-msign-return-address=") + Scope));
  if (!Scope.equals("none"))
    CmdArgs.push_back(
        Args.MakeArgString(Twine("-msign-return-address-key=") + Key));
  if (IndirectBranches)
    CmdArgs.push_back("-mbranch-target-enforce");
}

extern void AddAAPCSVolatileBitfieldArgs(const ArgList &Args,
                                         ArgStringList &CmdArgs);

namespace {
void AddUnalignedAccessWarning(ArgStringList &CmdArgs) {
  auto StrictAlignIter =
      std::find_if(CmdArgs.rbegin(), CmdArgs.rend(), [](StringRef Arg) {
        return Arg == "+strict-align" || Arg == "-strict-align";
      });
  if (StrictAlignIter != CmdArgs.rend() &&
      StringRef(*StrictAlignIter) == "+strict-align")
    CmdArgs.push_back("-Wunaligned-access");
}
}

void Flang::AddAArch64TargetArgs(const ArgList &Args,
                                 ArgStringList &CmdArgs) const {
  const llvm::Triple &Triple = getToolChain().getEffectiveTriple();

  if (!Args.hasFlag(options::OPT_mred_zone, options::OPT_mno_red_zone, true) ||
      Args.hasArg(options::OPT_mkernel) ||
      Args.hasArg(options::OPT_fapple_kext))
    CmdArgs.push_back("-disable-red-zone");

  if (!Args.hasFlag(options::OPT_mimplicit_float,
                    options::OPT_mno_implicit_float, true))
    CmdArgs.push_back("-no-implicit-float");

  RenderAArch64ABI(Triple, Args, CmdArgs);

  // Forward the -mglobal-merge option for explicit control over the pass.
  if (Arg *A = Args.getLastArg(options::OPT_mglobal_merge,
                               options::OPT_mno_global_merge)) {
    CmdArgs.push_back("-mllvm");
    if (A->getOption().matches(options::OPT_mno_global_merge))
      CmdArgs.push_back("-aarch64-enable-global-merge=false");
    else
      CmdArgs.push_back("-aarch64-enable-global-merge=true");
  }

  // Enable/disable return address signing and indirect branch targets.
  CollectARMPACBTIOptions(getToolChain(), Args, CmdArgs, true /*isAArch64*/);

  // Handle -msve_vector_bits=<bits>
  if (Arg *A = Args.getLastArg(options::OPT_msve_vector_bits_EQ)) {
    StringRef Val = A->getValue();
    const Driver &D = getToolChain().getDriver();
    if (Val.equals("128") || Val.equals("256") || Val.equals("512") ||
        Val.equals("1024") || Val.equals("2048") || Val.equals("128+") ||
        Val.equals("256+") || Val.equals("512+") || Val.equals("1024+") ||
        Val.equals("2048+")) {
      unsigned Bits = 0;
      if (Val.endswith("+"))
        Val = Val.substr(0, Val.size() - 1);
      else {
        bool Invalid = Val.getAsInteger(10, Bits); (void)Invalid;
        assert(!Invalid && "Failed to parse value");
        CmdArgs.push_back(
            Args.MakeArgString("-mvscale-max=" + llvm::Twine(Bits / 128)));
      }

      bool Invalid = Val.getAsInteger(10, Bits); (void)Invalid;
      assert(!Invalid && "Failed to parse value");
      CmdArgs.push_back(
          Args.MakeArgString("-mvscale-min=" + llvm::Twine(Bits / 128)));
    // Silently drop requests for vector-length agnostic code as it's implied.
    } else if (!Val.equals("scalable"))
      // Handle the unsupported values passed to msve-vector-bits.
      D.Diag(diag::err_drv_unsupported_option_argument)
          << A->getOption().getName() << Val;
  }

  AddAAPCSVolatileBitfieldArgs(Args, CmdArgs);

  if (const Arg *A = Args.getLastArg(clang::driver::options::OPT_mtune_EQ)) {
    StringRef Name = A->getValue();

    std::string TuneCPU;
    if (Name == "native")
      TuneCPU = std::string(llvm::sys::getHostCPUName());
    else
      TuneCPU = std::string(Name);

    if (!TuneCPU.empty()) {
      CmdArgs.push_back("-tune-cpu");
      CmdArgs.push_back(Args.MakeArgString(TuneCPU));
    }
  }

  AddUnalignedAccessWarning(CmdArgs);
}

void Flang::AddX86TargetArgs(const ArgList &Args,
                             ArgStringList &CmdArgs) const {
  const Driver &D = getToolChain().getDriver();
  addX86AlignBranchArgs(D, Args, CmdArgs, /*IsLTO=*/false);

  if (!Args.hasFlag(options::OPT_mred_zone, options::OPT_mno_red_zone, true) ||
      Args.hasArg(options::OPT_mkernel) ||
      Args.hasArg(options::OPT_fapple_kext))
    CmdArgs.push_back("-disable-red-zone");

  if (!Args.hasFlag(options::OPT_mtls_direct_seg_refs,
                    options::OPT_mno_tls_direct_seg_refs, true))
    CmdArgs.push_back("-mno-tls-direct-seg-refs");

  // Default to avoid implicit floating-point for kernel/kext code, but allow
  // that to be overridden with -mno-soft-float.
  bool NoImplicitFloat = (Args.hasArg(options::OPT_mkernel) ||
                          Args.hasArg(options::OPT_fapple_kext));
  if (Arg *A = Args.getLastArg(
          options::OPT_msoft_float, options::OPT_mno_soft_float,
          options::OPT_mimplicit_float, options::OPT_mno_implicit_float)) {
    const Option &O = A->getOption();
    NoImplicitFloat = (O.matches(options::OPT_mno_implicit_float) ||
                       O.matches(options::OPT_msoft_float));
  }
  if (NoImplicitFloat)
    CmdArgs.push_back("-no-implicit-float");

  if (Arg *A = Args.getLastArg(options::OPT_masm_EQ)) {
    StringRef Value = A->getValue();
    if (Value == "intel" || Value == "att") {
      CmdArgs.push_back("-mllvm");
      CmdArgs.push_back(Args.MakeArgString("-x86-asm-syntax=" + Value));
      CmdArgs.push_back(Args.MakeArgString("-inline-asm=" + Value));
    } else {
      D.Diag(diag::err_drv_unsupported_option_argument)
          << A->getOption().getName() << Value;
    }
  } else if (D.IsCLMode()) {
    CmdArgs.push_back("-mllvm");
    CmdArgs.push_back("-x86-asm-syntax=intel");
  }

  if (Arg *A = Args.getLastArg(options::OPT_mskip_rax_setup,
                               options::OPT_mno_skip_rax_setup))
    if (A->getOption().matches(options::OPT_mskip_rax_setup))
      CmdArgs.push_back(Args.MakeArgString("-mskip-rax-setup"));

  // Set flags to support MCU ABI.
  if (Args.hasFlag(options::OPT_miamcu, options::OPT_mno_iamcu, false)) {
    CmdArgs.push_back("-mfloat-abi");
    CmdArgs.push_back("soft");
    CmdArgs.push_back("-mstack-alignment=4");
  }

  // Handle -mtune.

  // Default to "generic" unless -march is present or targetting the PS4/PS5.
  std::string TuneCPU;
  if (!Args.hasArg(clang::driver::options::OPT_march_EQ) &&
      !getToolChain().getTriple().isPS())
    TuneCPU = "generic";

  // Override based on -mtune.
  if (const Arg *A = Args.getLastArg(clang::driver::options::OPT_mtune_EQ)) {
    StringRef Name = A->getValue();

    if (Name == "native") {
      Name = llvm::sys::getHostCPUName();
      if (!Name.empty())
        TuneCPU = std::string(Name);
    } else
      TuneCPU = std::string(Name);
  }

  if (!TuneCPU.empty()) {
    CmdArgs.push_back("-tune-cpu");
    CmdArgs.push_back(Args.MakeArgString(TuneCPU));
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
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_32:
  case llvm::Triple::aarch64_be:
    AddAArch64TargetArgs(Args, CmdArgs);
    break;
  case llvm::Triple::riscv64:
    AddRISCVTargetArgs(Args, CmdArgs);
    break;
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    AddX86TargetArgs(Args, CmdArgs);
    break;
  }
}

void Flang::addPicOptions(const ArgList &Args, ArgStringList &CmdArgs) const {
  // ParsePICArgs parses -fPIC/-fPIE and their variants and returns a tuple of
  // (RelocationModel, PICLevel, IsPIE).
  llvm::Reloc::Model RelocationModel;
  unsigned PICLevel;
  bool IsPIE;
  std::tie(RelocationModel, PICLevel, IsPIE) =
      ParsePICArgs(getToolChain(), Args);

  if (auto *RMName = RelocationModelName(RelocationModel)) {
    CmdArgs.push_back("-mrelocation-model");
    CmdArgs.push_back(RMName);
  }
  if (PICLevel > 0) {
    CmdArgs.push_back("-pic-level");
    CmdArgs.push_back(PICLevel == 1 ? "1" : "2");
    if (IsPIE)
      CmdArgs.push_back("-pic-is-pie");
  }
}

void Flang::addTargetOptions(const ArgList &Args,
                             ArgStringList &CmdArgs) const {
  const ToolChain &TC = getToolChain();
  const llvm::Triple &Triple = TC.getEffectiveTriple();
  const Driver &D = TC.getDriver();

  std::string CPU = getCPUName(D, Args, Triple);
  if (!CPU.empty()) {
    CmdArgs.push_back("-target-cpu");
    CmdArgs.push_back(Args.MakeArgString(CPU));
  }

  // Add the target features.
  switch (TC.getArch()) {
  default:
    break;
  case llvm::Triple::aarch64:
    [[fallthrough]];
  case llvm::Triple::x86_64:
    getTargetFeatures(D, Triple, Args, CmdArgs, /*ForAs*/ false);
    break;
  }

  // TODO: Add target specific flags, ABI, mtune option etc.
}

static void addFloatingPointOptions(const Driver &D, const ArgList &Args,
                                    ArgStringList &CmdArgs) {
  StringRef FPContract;
  bool HonorINFs = true;
  bool HonorNaNs = true;
  bool ApproxFunc = false;
  bool SignedZeros = true;
  bool AssociativeMath = false;
  bool ReciprocalMath = false;

  if (const Arg *A = Args.getLastArg(options::OPT_ffp_contract)) {
    const StringRef Val = A->getValue();
    if (Val == "fast" || Val == "off") {
      FPContract = Val;
    } else if (Val == "on") {
      // Warn instead of error because users might have makefiles written for
      // gfortran (which accepts -ffp-contract=on)
      D.Diag(diag::warn_drv_unsupported_option_for_flang)
          << Val << A->getOption().getName() << "off";
      FPContract = "off";
    } else
      // Clang's "fast-honor-pragmas" option is not supported because it is
      // non-standard
      D.Diag(diag::err_drv_unsupported_option_argument)
          << A->getSpelling() << Val;
  }

  for (const Arg *A : Args) {
    auto optId = A->getOption().getID();
    switch (optId) {
    // if this isn't an FP option, skip the claim below
    default:
      continue;

    case options::OPT_fhonor_infinities:
      HonorINFs = true;
      break;
    case options::OPT_fno_honor_infinities:
      HonorINFs = false;
      break;
    case options::OPT_fhonor_nans:
      HonorNaNs = true;
      break;
    case options::OPT_fno_honor_nans:
      HonorNaNs = false;
      break;
    case options::OPT_fapprox_func:
      ApproxFunc = true;
      break;
    case options::OPT_fno_approx_func:
      ApproxFunc = false;
      break;
    case options::OPT_fsigned_zeros:
      SignedZeros = true;
      break;
    case options::OPT_fno_signed_zeros:
      SignedZeros = false;
      break;
    case options::OPT_fassociative_math:
      AssociativeMath = true;
      break;
    case options::OPT_fno_associative_math:
      AssociativeMath = false;
      break;
    case options::OPT_freciprocal_math:
      ReciprocalMath = true;
      break;
    case options::OPT_fno_reciprocal_math:
      ReciprocalMath = false;
      break;
    case options::OPT_Ofast:
      [[fallthrough]];
    case options::OPT_ffast_math:
      HonorINFs = false;
      HonorNaNs = false;
      AssociativeMath = true;
      ReciprocalMath = true;
      ApproxFunc = true;
      SignedZeros = false;
      FPContract = "fast";
      break;
    case options::OPT_fno_fast_math:
      HonorINFs = true;
      HonorNaNs = true;
      AssociativeMath = false;
      ReciprocalMath = false;
      ApproxFunc = false;
      SignedZeros = true;
      // -fno-fast-math should undo -ffast-math so I return FPContract to the
      // default. It is important to check it is "fast" (the default) so that
      // --ffp-contract=off -fno-fast-math --> -ffp-contract=off
      if (FPContract == "fast")
        FPContract = "";
      break;
    }

    // If we handled this option claim it
    A->claim();
  }

  if (!HonorINFs && !HonorNaNs && AssociativeMath && ReciprocalMath &&
      ApproxFunc && !SignedZeros &&
      (FPContract == "fast" || FPContract == "")) {
    CmdArgs.push_back("-ffast-math");
    return;
  }

  if (!FPContract.empty())
    CmdArgs.push_back(Args.MakeArgString("-ffp-contract=" + FPContract));

  if (!HonorINFs)
    CmdArgs.push_back("-menable-no-infs");

  if (!HonorNaNs)
    CmdArgs.push_back("-menable-no-nans");

  if (ApproxFunc)
    CmdArgs.push_back("-fapprox-func");

  if (!SignedZeros)
    CmdArgs.push_back("-fno-signed-zeros");

  if (AssociativeMath && !SignedZeros)
    CmdArgs.push_back("-mreassociate");

  if (ReciprocalMath)
    CmdArgs.push_back("-freciprocal-math");
}

void Flang::ConstructJob(Compilation &C, const JobAction &JA,
                         const InputInfo &Output, const InputInfoList &Inputs,
                         const ArgList &Args, const char *LinkingOutput) const {
  const auto &TC = getToolChain();
  const llvm::Triple &Triple = TC.getEffectiveTriple();
  const std::string &TripleStr = Triple.getTriple();

  const Driver &D = TC.getDriver();
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
    addPreprocessingOptions(Args, CmdArgs);

  addFortranDialectOptions(Args, CmdArgs);

  // Code generation options.
  AddCodeGenOptions(Args, CmdArgs);

  // Code model options.
  AddCodeModelOptions(Args, CmdArgs, Triple);

  // Color diagnostics are parsed by the driver directly from argv and later
  // re-parsed to construct this job; claim any possible color diagnostic here
  // to avoid warn_drv_unused_argument.
  Args.getLastArg(options::OPT_fcolor_diagnostics,
                  options::OPT_fno_color_diagnostics);
  if (D.getDiags().getDiagnosticOptions().ShowColors)
    CmdArgs.push_back("-fcolor-diagnostics");

  // -fPIC and related options.
  addPicOptions(Args, CmdArgs);

  // Floating point related options
  addFloatingPointOptions(D, Args, CmdArgs);

  // Add target args, features, etc.
  addTargetOptions(Args, CmdArgs);

  // Add other compile options
  addOtherOptions(Args, CmdArgs);

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

  // Optimization level for CodeGen.
  if (const Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    if (A->getOption().matches(options::OPT_O4)) {
      CmdArgs.push_back("-O3");
      D.Diag(diag::warn_O4_is_O3);
    } else if (A->getOption().matches(options::OPT_Ofast)) {
      CmdArgs.push_back("-O3");
    } else {
      A->render(Args, CmdArgs);
    }
  }

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  assert(Input.isFilename() && "Invalid input.");

  addDashXForInput(Args, Input, CmdArgs);

  CmdArgs.push_back(Input.getFilename());

  // TODO: Replace flang-new with flang once the new driver replaces the
  // throwaway driver
  const char *Exec = Args.MakeArgString(D.GetProgramPath("flang-new", TC));
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileUTF8(),
                                         Exec, CmdArgs, Inputs, Output));
}

Flang::Flang(const ToolChain &TC) : Tool("flang-new", "flang frontend", TC) {}

Flang::~Flang() {}
