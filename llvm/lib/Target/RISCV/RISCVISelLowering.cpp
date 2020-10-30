//===-- RISCVISelLowering.cpp - RISCV DAG Lowering Implementation  --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that RISCV uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "RISCVISelLowering.h"
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "Utils/RISCVMatInt.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/IntrinsicsEPI.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-lower"

STATISTIC(NumTailCalls, "Number of tail calls");

RISCVTargetLowering::RISCVTargetLowering(const TargetMachine &TM,
                                         const RISCVSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {

  if (Subtarget.isRV32E())
    report_fatal_error("Codegen not yet implemented for RV32E");

  RISCVABI::ABI ABI = Subtarget.getTargetABI();
  assert(ABI != RISCVABI::ABI_Unknown && "Improperly initialised target ABI");

  if ((ABI == RISCVABI::ABI_ILP32F || ABI == RISCVABI::ABI_LP64F) &&
      !Subtarget.hasStdExtF()) {
    errs() << "Hard-float 'f' ABI can't be used for a target that "
                "doesn't support the F instruction set extension (ignoring "
                          "target-abi)\n";
    ABI = Subtarget.is64Bit() ? RISCVABI::ABI_LP64 : RISCVABI::ABI_ILP32;
  } else if ((ABI == RISCVABI::ABI_ILP32D || ABI == RISCVABI::ABI_LP64D) &&
             !Subtarget.hasStdExtD()) {
    errs() << "Hard-float 'd' ABI can't be used for a target that "
              "doesn't support the D instruction set extension (ignoring "
              "target-abi)\n";
    ABI = Subtarget.is64Bit() ? RISCVABI::ABI_LP64 : RISCVABI::ABI_ILP32;
  }

  switch (ABI) {
  default:
    report_fatal_error("Don't know how to lower this ABI");
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64:
  case RISCVABI::ABI_LP64F:
  case RISCVABI::ABI_LP64D:
    break;
  }

  MVT XLenVT = Subtarget.getXLenVT();

  // Set up the register classes.
  addRegisterClass(XLenVT, &RISCV::GPRRegClass);

  if (Subtarget.hasStdExtF())
    addRegisterClass(MVT::f32, &RISCV::FPR32RegClass);
  if (Subtarget.hasStdExtD())
    addRegisterClass(MVT::f64, &RISCV::FPR64RegClass);

  if (Subtarget.hasStdExtV()) {
    addRegisterClass(MVT::nxv1i1, &RISCV::VRRegClass);
    addRegisterClass(MVT::nxv2i1, &RISCV::VRRegClass);
    addRegisterClass(MVT::nxv4i1, &RISCV::VRRegClass);
    addRegisterClass(MVT::nxv8i1, &RISCV::VRRegClass);
    addRegisterClass(MVT::nxv16i1, &RISCV::VRRegClass);
    addRegisterClass(MVT::nxv32i1, &RISCV::VRRegClass);
    addRegisterClass(MVT::nxv64i1, &RISCV::VRRegClass);

    //addRegisterClass(MVT::nxv1i8, &RISCV::VRRegClass); // FIXME illegal type
    //addRegisterClass(MVT::nxv2i8, &RISCV::VRRegClass); // FIXME illegal type
    //addRegisterClass(MVT::nxv4i8, &RISCV::VRRegClass); // FIXME illegal type
    addRegisterClass(MVT::nxv8i8, &RISCV::VRRegClass);
    addRegisterClass(MVT::nxv16i8, &RISCV::VRM2RegClass);
    addRegisterClass(MVT::nxv32i8, &RISCV::VRM4RegClass);
    addRegisterClass(MVT::nxv64i8, &RISCV::VRM8RegClass);

    //addRegisterClass(MVT::nxv1i16, &RISCV::VRRegClass); // FIXME illegal type
    //addRegisterClass(MVT::nxv2i16, &RISCV::VRRegClass); // FIXME illegal type
    addRegisterClass(MVT::nxv4i16, &RISCV::VRRegClass);
    addRegisterClass(MVT::nxv8i16, &RISCV::VRM2RegClass);
    addRegisterClass(MVT::nxv16i16, &RISCV::VRM4RegClass);
    addRegisterClass(MVT::nxv32i16, &RISCV::VRM8RegClass);

    //addRegisterClass(MVT::nxv1i32, &RISCV::VRRegClass); // FIXME illegal type
    addRegisterClass(MVT::nxv2i32, &RISCV::VRRegClass);
    addRegisterClass(MVT::nxv4i32, &RISCV::VRM2RegClass);
    addRegisterClass(MVT::nxv8i32, &RISCV::VRM4RegClass);
    addRegisterClass(MVT::nxv16i32, &RISCV::VRM8RegClass);

    addRegisterClass(MVT::nxv1i64, &RISCV::VRRegClass);
    addRegisterClass(MVT::nxv2i64, &RISCV::VRM2RegClass);
    addRegisterClass(MVT::nxv4i64, &RISCV::VRM4RegClass);
    addRegisterClass(MVT::nxv8i64, &RISCV::VRM8RegClass);

    //addRegisterClass(MVT::nxv1f32, &RISCV::VRRegClass); // FIXME illegal type
    addRegisterClass(MVT::nxv2f32, &RISCV::VRRegClass);
    addRegisterClass(MVT::nxv4f32, &RISCV::VRM2RegClass);
    addRegisterClass(MVT::nxv8f32, &RISCV::VRM4RegClass);
    addRegisterClass(MVT::nxv16f32, &RISCV::VRM8RegClass);

    addRegisterClass(MVT::nxv1f64, &RISCV::VRRegClass);
    addRegisterClass(MVT::nxv2f64, &RISCV::VRM2RegClass);
    addRegisterClass(MVT::nxv4f64, &RISCV::VRM4RegClass);
    addRegisterClass(MVT::nxv8f64, &RISCV::VRM8RegClass);

    setBooleanVectorContents(ZeroOrOneBooleanContent);

    for (auto VT : MVT::integer_scalable_vector_valuetypes()) {
      setOperationAction(ISD::SPLAT_VECTOR, VT, Legal);
      setOperationAction(ISD::VECTOR_SHUFFLE, VT, Custom);
    }
    for (auto VT : MVT::fp_scalable_vector_valuetypes()) {
      setOperationAction(ISD::SPLAT_VECTOR, VT, Legal);
      setOperationAction(ISD::VECTOR_SHUFFLE, VT, Custom);
    }
  }

  // Compute derived properties from the register classes.
  computeRegisterProperties(STI.getRegisterInfo());

  setStackPointerRegisterToSaveRestore(RISCV::X2);

  for (auto N : {ISD::EXTLOAD, ISD::SEXTLOAD, ISD::ZEXTLOAD})
    setLoadExtAction(N, XLenVT, MVT::i1, Promote);

  // TODO: add all necessary setOperationAction calls.
  setOperationAction(ISD::DYNAMIC_STACKALLOC, XLenVT, Expand);

  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BR_CC, XLenVT, Expand);
  setOperationAction(ISD::SELECT, XLenVT, Custom);
  setOperationAction(ISD::SELECT_CC, XLenVT, Expand);

  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);

  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VAARG, MVT::Other, Expand);
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);

  for (auto VT : {MVT::i1, MVT::i8, MVT::i16})
    setOperationAction(ISD::SIGN_EXTEND_INREG, VT, Expand);

  if (Subtarget.is64Bit()) {
    setOperationAction(ISD::ADD, MVT::i32, Custom);
    setOperationAction(ISD::SUB, MVT::i32, Custom);
    setOperationAction(ISD::SHL, MVT::i32, Custom);
    setOperationAction(ISD::SRA, MVT::i32, Custom);
    setOperationAction(ISD::SRL, MVT::i32, Custom);
  }

  if (!Subtarget.hasStdExtM()) {
    setOperationAction(ISD::MUL, XLenVT, Expand);
    setOperationAction(ISD::MULHS, XLenVT, Expand);
    setOperationAction(ISD::MULHU, XLenVT, Expand);
    setOperationAction(ISD::SDIV, XLenVT, Expand);
    setOperationAction(ISD::UDIV, XLenVT, Expand);
    setOperationAction(ISD::SREM, XLenVT, Expand);
    setOperationAction(ISD::UREM, XLenVT, Expand);
  }

  if (Subtarget.is64Bit() && Subtarget.hasStdExtM()) {
    setOperationAction(ISD::MUL, MVT::i32, Custom);
    setOperationAction(ISD::SDIV, MVT::i32, Custom);
    setOperationAction(ISD::UDIV, MVT::i32, Custom);
    setOperationAction(ISD::UREM, MVT::i32, Custom);
  }

  setOperationAction(ISD::SDIVREM, XLenVT, Expand);
  setOperationAction(ISD::UDIVREM, XLenVT, Expand);
  setOperationAction(ISD::SMUL_LOHI, XLenVT, Expand);
  setOperationAction(ISD::UMUL_LOHI, XLenVT, Expand);

  setOperationAction(ISD::SHL_PARTS, XLenVT, Custom);
  setOperationAction(ISD::SRL_PARTS, XLenVT, Custom);
  setOperationAction(ISD::SRA_PARTS, XLenVT, Custom);

  if (!(Subtarget.hasStdExtZbb() || Subtarget.hasStdExtZbp())) {
    setOperationAction(ISD::ROTL, XLenVT, Expand);
    setOperationAction(ISD::ROTR, XLenVT, Expand);
  }

  if (!Subtarget.hasStdExtZbp())
    setOperationAction(ISD::BSWAP, XLenVT, Expand);

  if (!Subtarget.hasStdExtZbb()) {
    setOperationAction(ISD::CTTZ, XLenVT, Expand);
    setOperationAction(ISD::CTLZ, XLenVT, Expand);
    setOperationAction(ISD::CTPOP, XLenVT, Expand);
  }

  if (Subtarget.hasStdExtZbp())
    setOperationAction(ISD::BITREVERSE, XLenVT, Legal);

  if (Subtarget.hasStdExtZbt()) {
    setOperationAction(ISD::FSHL, XLenVT, Legal);
    setOperationAction(ISD::FSHR, XLenVT, Legal);
  }

  ISD::CondCode FPCCToExtend[] = {
      ISD::SETOGT, ISD::SETOGE, ISD::SETONE, ISD::SETUEQ, ISD::SETUGT,
      ISD::SETUGE, ISD::SETULT, ISD::SETULE, ISD::SETUNE, ISD::SETGT,
      ISD::SETGE,  ISD::SETNE};

  ISD::NodeType FPOpToExtend[] = {
      ISD::FSIN, ISD::FCOS, ISD::FSINCOS, ISD::FPOW, ISD::FREM, ISD::FP16_TO_FP,
      ISD::FP_TO_FP16};

  if (Subtarget.hasStdExtF()) {
    setOperationAction(ISD::FMINNUM, MVT::f32, Legal);
    setOperationAction(ISD::FMAXNUM, MVT::f32, Legal);
    for (auto CC : FPCCToExtend)
      setCondCodeAction(CC, MVT::f32, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::f32, Expand);
    setOperationAction(ISD::SELECT, MVT::f32, Custom);
    setOperationAction(ISD::BR_CC, MVT::f32, Expand);
    for (auto Op : FPOpToExtend)
      setOperationAction(Op, MVT::f32, Expand);
    setLoadExtAction(ISD::EXTLOAD, MVT::f32, MVT::f16, Expand);
    setTruncStoreAction(MVT::f32, MVT::f16, Expand);
  }

  if (Subtarget.hasStdExtF() && Subtarget.is64Bit())
    setOperationAction(ISD::BITCAST, MVT::i32, Custom);

  if (Subtarget.hasStdExtD()) {
    setOperationAction(ISD::FMINNUM, MVT::f64, Legal);
    setOperationAction(ISD::FMAXNUM, MVT::f64, Legal);
    for (auto CC : FPCCToExtend)
      setCondCodeAction(CC, MVT::f64, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::f64, Expand);
    setOperationAction(ISD::SELECT, MVT::f64, Custom);
    setOperationAction(ISD::BR_CC, MVT::f64, Expand);
    setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f32, Expand);
    setTruncStoreAction(MVT::f64, MVT::f32, Expand);
    for (auto Op : FPOpToExtend)
      setOperationAction(Op, MVT::f64, Expand);
    setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f16, Expand);
    setTruncStoreAction(MVT::f64, MVT::f16, Expand);
  }

  if (Subtarget.is64Bit() &&
      !(Subtarget.hasStdExtD() || Subtarget.hasStdExtF())) {
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
    setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
    setOperationAction(ISD::STRICT_FP_TO_UINT, MVT::i32, Custom);
    setOperationAction(ISD::STRICT_FP_TO_SINT, MVT::i32, Custom);
  }

  setOperationAction(ISD::GlobalAddress, XLenVT, Custom);
  setOperationAction(ISD::BlockAddress, XLenVT, Custom);
  setOperationAction(ISD::ConstantPool, XLenVT, Custom);

  setOperationAction(ISD::GlobalTLSAddress, XLenVT, Custom);

  // TODO: On M-mode only targets, the cycle[h] CSR may not be present.
  // Unfortunately this can't be determined just from the ISA naming string.
  setOperationAction(ISD::READCYCLECOUNTER, MVT::i64,
                     Subtarget.is64Bit() ? Legal : Custom);

  setOperationAction(ISD::TRAP, MVT::Other, Legal);
  setOperationAction(ISD::DEBUGTRAP, MVT::Other, Legal);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);

  if (Subtarget.hasStdExtA()) {
    setMaxAtomicSizeInBitsSupported(Subtarget.getXLen());
    setMinCmpXchgSizeInBits(32);
  } else {
    setMaxAtomicSizeInBitsSupported(0);
  }

  setBooleanContents(ZeroOrOneBooleanContent);

  // Function alignments.
  const Align FunctionAlignment(Subtarget.hasStdExtC() ? 2 : 4);
  setMinFunctionAlignment(FunctionAlignment);
  setPrefFunctionAlignment(FunctionAlignment);

  // Effectively disable jump table generation.
  setMinimumJumpTableEntries(INT_MAX);

  // Jumps are expensive, compared to logic
  setJumpIsExpensive();

  // We can use any register for comparisons
  setHasMultipleConditionRegisters();

  if (Subtarget.hasStdExtV()) {
    // EPI & VPred intrinsics may have illegal operands/results
    for (auto VT : {MVT::i1, MVT::i8, MVT::i16, MVT::i32, MVT::nxv1i32}) {
      setOperationAction(ISD::INTRINSIC_WO_CHAIN, VT, Custom);
    }

    // VPred intrinsics may have illegal operands/results
    for (auto VT : {MVT::i32}) {
      setOperationAction(ISD::INTRINSIC_W_CHAIN, VT, Custom);
      setOperationAction(ISD::INTRINSIC_VOID, VT, Custom);
    }

    // Some tuple operations are chained and need custom lowering.
    setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Other, Custom);
    setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);

    // Custom-legalize this node for illegal result types or mask operands.
    for (auto VT : {MVT::i8, MVT::i16, MVT::i32, MVT::nxv1i1, MVT::nxv2i1,
                    MVT::nxv4i1, MVT::nxv8i1, MVT::nxv16i1, MVT::nxv32i1,
                    MVT::nxv64i1}) {
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);
    }

    // Custom-legalize this node for scalable vectors.
    for (auto VT : {MVT::nxv1i64, MVT::nxv2i32, MVT::nxv4i16}) {
      setOperationAction(ISD::SIGN_EXTEND_INREG, VT, Custom);
    }

    // Custom-legalize these nodes for scalable vectors.
    for (auto VT : {MVT::nxv8i8, MVT::nxv16i8, MVT::nxv32i8, MVT::nxv64i8,
                    MVT::nxv4i16, MVT::nxv8i16, MVT::nxv16i16, MVT::nxv32i16,
                    MVT::nxv2i32, MVT::nxv4i32, MVT::nxv8i32, MVT::nxv16i32,
                    MVT::nxv1i64, MVT::nxv2i64, MVT::nxv4i64, MVT::nxv8i64,
                    MVT::nxv2f32, MVT::nxv4f32, MVT::nxv8f32, MVT::nxv16f32,
                    MVT::nxv1f64, MVT::nxv2f64, MVT::nxv4f64, MVT::nxv8f64}) {
      setOperationAction(ISD::MGATHER, VT, Custom);
      setOperationAction(ISD::MSCATTER, VT, Custom);
      setOperationAction(ISD::SELECT, VT, Custom);
      setOperationAction(ISD::SIGN_EXTEND, VT, Custom);
      setOperationAction(ISD::ZERO_EXTEND, VT, Custom);
      setOperationAction(ISD::TRUNCATE, VT, Custom);
    }

    // Register libcalls for fp EXP functions.
    setLibcallName(RTLIB::EXP_NXV1F64, "__epi_exp_nxv1f64");
    setLibcallName(RTLIB::EXP_NXV2F64, "__epi_exp_nxv2f64");
    setLibcallName(RTLIB::EXP_NXV4F64, "__epi_exp_nxv4f64");
    setLibcallName(RTLIB::EXP_NXV8F64, "__epi_exp_nxv8f64");
    setLibcallName(RTLIB::EXP_NXV2F32, "__epi_exp_nxv2f32");
    setLibcallName(RTLIB::EXP_NXV4F32, "__epi_exp_nxv4f32");
    setLibcallName(RTLIB::EXP_NXV8F32, "__epi_exp_nxv8f32");
    setLibcallName(RTLIB::EXP_NXV16F32, "__epi_exp_nxv16f32");

    // Custom-legalize these nodes for fp scalable vectors.
    for (auto VT : {MVT::nxv2f32, MVT::nxv4f32, MVT::nxv8f32, MVT::nxv16f32,
                    MVT::nxv1f64, MVT::nxv2f64, MVT::nxv4f64, MVT::nxv8f64}) {
      setOperationAction(ISD::FEXP, VT, Custom);
    }

    // Vector integer reductions.
    for (auto VT : {MVT::nxv8i8, MVT::nxv16i8, MVT::nxv32i8, MVT::nxv64i8,
                    MVT::nxv4i16, MVT::nxv8i16, MVT::nxv16i16, MVT::nxv32i16,
                    MVT::nxv2i32, MVT::nxv4i32, MVT::nxv8i32, MVT::nxv16i32,
                    MVT::nxv1i64, MVT::nxv2i64, MVT::nxv4i64, MVT::nxv8i64}) {
      setOperationAction(ISD::VECREDUCE_ADD, VT, Legal);
      setOperationAction(ISD::VECREDUCE_AND, VT, Legal);
      setOperationAction(ISD::VECREDUCE_OR, VT, Legal);
      setOperationAction(ISD::VECREDUCE_XOR, VT, Legal);
      setOperationAction(ISD::VECREDUCE_SMAX, VT, Legal);
      setOperationAction(ISD::VECREDUCE_SMIN, VT, Legal);
      setOperationAction(ISD::VECREDUCE_UMAX, VT, Legal);
      setOperationAction(ISD::VECREDUCE_UMIN, VT, Legal);
    }

    // Vector fp reductions.
    for (auto VT : {MVT::nxv2f32, MVT::nxv4f32, MVT::nxv8f32, MVT::nxv16f32,
                    MVT::nxv1f64, MVT::nxv2f64, MVT::nxv4f64, MVT::nxv8f64}) {
      setOperationAction(ISD::VECREDUCE_SEQ_FADD, VT, Legal);
      setOperationAction(ISD::VECREDUCE_FADD, VT, Legal);
      setOperationAction(ISD::VECREDUCE_FMUL, VT, Legal);
      setOperationAction(ISD::VECREDUCE_FMAX, VT, Legal);
      setOperationAction(ISD::VECREDUCE_FMIN, VT, Legal);
    }

    // Vector fp min/max operations.
    for (auto VT : {MVT::nxv2f32, MVT::nxv4f32, MVT::nxv8f32, MVT::nxv16f32,
                    MVT::nxv1f64, MVT::nxv2f64, MVT::nxv4f64, MVT::nxv8f64}) {
      setOperationAction(ISD::FMINNUM, VT, Legal);
      setOperationAction(ISD::FMAXNUM, VT, Legal);
    }
  }
}

EVT RISCVTargetLowering::getSetCCResultType(const DataLayout &DL, LLVMContext &,
                                            EVT VT) const {
  if (!VT.isVector())
    return getPointerTy(DL);
  if (Subtarget.hasStdExtV())
    return MVT::getVectorVT(MVT::i1, VT.getVectorElementCount());
  else
    return VT.changeVectorElementTypeToInteger();
}

bool RISCVTargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
                                             const CallInst &I,
                                             MachineFunction &MF,
                                             unsigned Intrinsic) const {
  auto &DL = I.getModule()->getDataLayout();
  switch (Intrinsic) {
  default:
    return false;
  case Intrinsic::riscv_masked_atomicrmw_xchg_i32:
  case Intrinsic::riscv_masked_atomicrmw_add_i32:
  case Intrinsic::riscv_masked_atomicrmw_sub_i32:
  case Intrinsic::riscv_masked_atomicrmw_nand_i32:
  case Intrinsic::riscv_masked_atomicrmw_max_i32:
  case Intrinsic::riscv_masked_atomicrmw_min_i32:
  case Intrinsic::riscv_masked_atomicrmw_umax_i32:
  case Intrinsic::riscv_masked_atomicrmw_umin_i32:
  case Intrinsic::riscv_masked_cmpxchg_i32: {
    PointerType *PtrTy = cast<PointerType>(I.getArgOperand(0)->getType());
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getVT(PtrTy->getElementType());
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.align = Align(4);
    Info.flags = MachineMemOperand::MOLoad | MachineMemOperand::MOStore |
                 MachineMemOperand::MOVolatile;
    return true;
  }
  case Intrinsic::epi_vload:
  case Intrinsic::epi_vload_strided:
  case Intrinsic::epi_vload_indexed: {
    PointerType *PtrTy = cast<PointerType>(I.getArgOperand(0)->getType());
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getVT(PtrTy->getElementType());
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.align = MaybeAlign(DL.getABITypeAlignment(PtrTy->getElementType()));
    Info.flags = MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::epi_vload_mask:
  case Intrinsic::epi_vload_strided_mask:
  case Intrinsic::epi_vload_indexed_mask: {
    PointerType *PtrTy = cast<PointerType>(I.getArgOperand(1)->getType());
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getVT(PtrTy->getElementType());
    Info.ptrVal = I.getArgOperand(1);
    Info.offset = 0;
    Info.align = MaybeAlign(DL.getABITypeAlignment(PtrTy->getElementType()));
    Info.flags = MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::epi_vstore:
  case Intrinsic::epi_vstore_strided:
  case Intrinsic::epi_vstore_indexed:
  case Intrinsic::epi_vstore_mask:
  case Intrinsic::epi_vstore_strided_mask:
  case Intrinsic::epi_vstore_indexed_mask: {
    PointerType *PtrTy = cast<PointerType>(I.getArgOperand(1)->getType());
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::getVT(PtrTy->getElementType());
    Info.ptrVal = I.getArgOperand(1);
    Info.offset = 0;
    Info.align = MaybeAlign(DL.getABITypeAlignment(PtrTy->getElementType()));
    Info.flags = MachineMemOperand::MOStore;
    return true;
  }
  }
}

bool RISCVTargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                                const AddrMode &AM, Type *Ty,
                                                unsigned AS,
                                                Instruction *I) const {
  // No global is ever allowed as a base.
  if (AM.BaseGV)
    return false;

  // Require a 12-bit signed offset.
  if (!isInt<12>(AM.BaseOffs))
    return false;

  switch (AM.Scale) {
  case 0: // "r+i" or just "i", depending on HasBaseReg.
    break;
  case 1:
    if (!AM.HasBaseReg) // allow "r+i".
      break;
    return false; // disallow "r+r" or "r+r+i".
  default:
    return false;
  }

  return true;
}

bool RISCVTargetLowering::isLegalICmpImmediate(int64_t Imm) const {
  return isInt<12>(Imm);
}

bool RISCVTargetLowering::isLegalAddImmediate(int64_t Imm) const {
  return isInt<12>(Imm);
}

// On RV32, 64-bit integers are split into their high and low parts and held
// in two different registers, so the trunc is free since the low register can
// just be used.
bool RISCVTargetLowering::isTruncateFree(Type *SrcTy, Type *DstTy) const {
  if (Subtarget.is64Bit() || !SrcTy->isIntegerTy() || !DstTy->isIntegerTy())
    return false;
  unsigned SrcBits = SrcTy->getPrimitiveSizeInBits();
  unsigned DestBits = DstTy->getPrimitiveSizeInBits();
  return (SrcBits == 64 && DestBits == 32);
}

bool RISCVTargetLowering::isTruncateFree(EVT SrcVT, EVT DstVT) const {
  if (Subtarget.is64Bit() || SrcVT.isVector() || DstVT.isVector() ||
      !SrcVT.isInteger() || !DstVT.isInteger())
    return false;
  unsigned SrcBits = SrcVT.getSizeInBits();
  unsigned DestBits = DstVT.getSizeInBits();
  return (SrcBits == 64 && DestBits == 32);
}

bool RISCVTargetLowering::isZExtFree(SDValue Val, EVT VT2) const {
  // Zexts are free if they can be combined with a load.
  if (auto *LD = dyn_cast<LoadSDNode>(Val)) {
    EVT MemVT = LD->getMemoryVT();
    if ((MemVT == MVT::i8 || MemVT == MVT::i16 ||
         (Subtarget.is64Bit() && MemVT == MVT::i32)) &&
        (LD->getExtensionType() == ISD::NON_EXTLOAD ||
         LD->getExtensionType() == ISD::ZEXTLOAD))
      return true;
  }

  return TargetLowering::isZExtFree(Val, VT2);
}

bool RISCVTargetLowering::isSExtCheaperThanZExt(EVT SrcVT, EVT DstVT) const {
  return Subtarget.is64Bit() && SrcVT == MVT::i32 && DstVT == MVT::i64;
}

bool RISCVTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT,
                                       bool ForCodeSize) const {
  if (VT == MVT::f32 && !Subtarget.hasStdExtF())
    return false;
  if (VT == MVT::f64 && !Subtarget.hasStdExtD())
    return false;
  if (Imm.isNegZero())
    return false;
  return Imm.isZero();
}

bool RISCVTargetLowering::hasBitPreservingFPLogic(EVT VT) const {
  return (VT == MVT::f32 && Subtarget.hasStdExtF()) ||
         (VT == MVT::f64 && Subtarget.hasStdExtD());
}

// Changes the condition code and swaps operands if necessary, so the SetCC
// operation matches one of the comparisons supported directly in the RISC-V
// ISA.
static void normaliseSetCC(SDValue &LHS, SDValue &RHS, ISD::CondCode &CC) {
  switch (CC) {
  default:
    break;
  case ISD::SETGT:
  case ISD::SETLE:
  case ISD::SETUGT:
  case ISD::SETULE:
    CC = ISD::getSetCCSwappedOperands(CC);
    std::swap(LHS, RHS);
    break;
  }
}

// Return the RISC-V branch opcode that matches the given DAG integer
// condition code. The CondCode must be one of those supported by the RISC-V
// ISA (see normaliseSetCC).
static unsigned getBranchOpcodeForIntCondCode(ISD::CondCode CC) {
  switch (CC) {
  default:
    llvm_unreachable("Unsupported CondCode");
  case ISD::SETEQ:
    return RISCV::BEQ;
  case ISD::SETNE:
    return RISCV::BNE;
  case ISD::SETLT:
    return RISCV::BLT;
  case ISD::SETGE:
    return RISCV::BGE;
  case ISD::SETULT:
    return RISCV::BLTU;
  case ISD::SETUGE:
    return RISCV::BGEU;
  }
}

SDValue RISCVTargetLowering::lowerVECTOR_SHUFFLE(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(Op.getNode());

  if (!SVN->isSplat())
    return SDValue();

  // FIXME - Use splat index!
  if (SVN->getSplatIndex() != 0)
    return SDValue();

  // Recover the scalar value.
  SDValue SplatVector = Op.getOperand(0);
  // FIXME - Look other targets what they do with suffles, it should be
  // possible to use vmv/vfmv or vrgather.
  if (SplatVector.getOpcode() != ISD::INSERT_VECTOR_ELT)
    return SDValue();
  if (SplatVector.getOperand(0).getOpcode() != ISD::UNDEF)
    return SDValue();
  SDValue ScalarValue = SplatVector.getOperand(1);
  auto *ConstantValue = dyn_cast<ConstantSDNode>(SplatVector.getOperand(2));
  if (!ConstantValue || ConstantValue->getZExtValue() != 0)
    return SDValue();

  return DAG.getNode(ISD::SPLAT_VECTOR, DL, VT, ScalarValue);
}

SDValue RISCVTargetLowering::lowerExtend(SDValue Op, SelectionDAG &DAG,
                                         int Opcode) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();

  // Skip masks.
  if (SrcVT.getVectorElementType() == MVT::i1)
    return Op;

  EVT ResultVT = SrcVT;
  SDValue Result = Src;
  do {
    ResultVT = ResultVT.widenIntegerVectorElementType(*DAG.getContext());
    Result = DAG.getNode(Opcode, DL, ResultVT, Result);
  } while (ResultVT != VT);

  return Result;
}

SDValue RISCVTargetLowering::lowerSIGN_EXTEND(SDValue Op,
                                              SelectionDAG &DAG) const {
  return lowerExtend(Op, DAG, RISCVISD::SIGN_EXTEND_VECTOR);
}

SDValue RISCVTargetLowering::lowerZERO_EXTEND(SDValue Op,
                                              SelectionDAG &DAG) const {
  return lowerExtend(Op, DAG, RISCVISD::ZERO_EXTEND_VECTOR);
}

SDValue RISCVTargetLowering::lowerTRUNCATE(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();

  // Skip masks.
  if (VT.getVectorElementType() == MVT::i1)
    return Op;

  auto HalfIntegerVectorElementType = [&DAG](EVT PrevVT) {
    EVT EltVT = PrevVT.getVectorElementType();
    assert(EltVT.getSizeInBits() % 2 == 0 && "Invalid bit size");
    EltVT = EVT::getIntegerVT(*DAG.getContext(), EltVT.getSizeInBits() / 2);
    return EVT::getVectorVT(*DAG.getContext(), EltVT,
                            PrevVT.getVectorElementCount());
  };

  EVT ResultVT = SrcVT;
  SDValue Result = Src;
  do {
    ResultVT = HalfIntegerVectorElementType(ResultVT);
    Result = DAG.getNode(RISCVISD::TRUNCATE_VECTOR, DL, ResultVT, Result);
  } while (ResultVT != VT);

  return Result;
}

SDValue RISCVTargetLowering::lowerSIGN_EXTEND_INREG(SDValue Op,
                                                    SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  VTSDNode *SrcVTNode = cast<VTSDNode>(Op.getOperand(1));
  assert(SrcVTNode != nullptr && "Unexpected SDNode");
  EVT SrcVT = SrcVTNode->getVT();
  MVT::SimpleValueType SimpleSrcVT = SrcVT.getSimpleVT().SimpleTy;

  assert((SimpleSrcVT == MVT::nxv1i8 || SimpleSrcVT == MVT::nxv1i16 ||
          SimpleSrcVT == MVT::nxv1i32 || SimpleSrcVT == MVT::nxv2i8 ||
          SimpleSrcVT == MVT::nxv2i16 || SimpleSrcVT == MVT::nxv4i8) &&
         "Unexpected type to extend");

  unsigned ResTyBits = VT.getScalarSizeInBits();
  unsigned OpTyBits = SrcVT.getScalarSizeInBits();

  assert(ResTyBits > OpTyBits);

  // Compute the number of bits to sign-extend.
  SDValue ExtendBits = DAG.getConstant(ResTyBits - OpTyBits, DL, MVT::i64);

  SDValue SextInreg = DAG.getNode(RISCVISD::SIGN_EXTEND_BITS_INREG, DL, VT,
                                  Op.getOperand(0), ExtendBits);

  return SextInreg;
}

SDValue RISCVTargetLowering::lowerMGATHER(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  assert(VT.isScalableVector() && "Unexpected type");

  SDValue Indices = Op.getOperand(4);
  EVT IndexVT = Indices.getValueType();
  assert(IndexVT == VT.changeVectorElementTypeToInteger() &&
         "Unexpected type for indices");

  SDValue Offsets;
  if (Op.getConstantOperandVal(5) == 1) {
    Offsets = Indices;
  } else {
    SDValue VSLLOperands[] = {
        DAG.getTargetConstant(Intrinsic::epi_vsll, DL, MVT::i64), Indices,
        DAG.getConstant(Log2_64(Op.getConstantOperandVal(5)), DL,
                        MVT::i64),           // log2(Scale).
        DAG.getRegister(RISCV::X0, MVT::i64) // VLMAX.
    };
    // FIXME: This may overflow. RVV-0.9 provides mechanisms to decouple the
    // index width from the data element width, solving this problem.
    Offsets = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, IndexVT, VSLLOperands);
  }

  SDValue VLXEOperands[] = {
      Op.getOperand(0), // Chain.
      DAG.getTargetConstant(Intrinsic::epi_vload_indexed_mask, DL, MVT::i64),
      Op.getOperand(1), // Merge.
      Op.getOperand(3), // Base address.
      Offsets,
      Op.getOperand(2),                    // Mask.
      DAG.getRegister(RISCV::X0, MVT::i64) // VLMAX.
  };

  return DAG.getNode(ISD::INTRINSIC_W_CHAIN, DL, Op->getVTList(), VLXEOperands);
}

SDValue RISCVTargetLowering::lowerMSCATTER(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getOperand(1).getValueType();
  assert(VT.isScalableVector() && "Unexpected type");

  SDValue Indices = Op.getOperand(4);
  EVT IndexVT = Indices.getValueType();
  assert(IndexVT == VT.changeVectorElementTypeToInteger() &&
         "Unexpected type for indices");

  SDValue Offsets;
  if (Op.getConstantOperandVal(5) == 1) {
    Offsets = Indices;
  } else {
    SDValue VSLLOperands[] = {
        DAG.getTargetConstant(Intrinsic::epi_vsll, DL, MVT::i64), Indices,
        DAG.getConstant(Log2_64(Op.getConstantOperandVal(5)), DL,
                        MVT::i64),           // log2(Scale).
        DAG.getRegister(RISCV::X0, MVT::i64) // VLMAX.
    };
    // FIXME: This may overflow. RVV-0.9 provides mechanisms to decouple the
    // index width from the data element width, solving this problem.
    Offsets = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, IndexVT, VSLLOperands);
  }

  // Val, OutChain = GATHER (InChain, PassThru, Mask, BasePtr, Index, Scale)
  //      OutChain = SCATTER(InChain, Value,    Mask, BasePtr, Index, Scale)
  SDValue VSXEOperands[] = {
      Op.getOperand(0), // Chain.
      DAG.getTargetConstant(Intrinsic::epi_vstore_indexed_mask, DL, MVT::i64),
      Op.getOperand(1), // Data
      Op.getOperand(3), // Base address.
      Offsets,
      Op.getOperand(2),                    // Mask.
      DAG.getRegister(RISCV::X0, MVT::i64) // VLMAX.
  };

  return DAG.getNode(ISD::INTRINSIC_VOID, DL, Op->getVTList(), VSXEOperands);
}

SDValue RISCVTargetLowering::lowerEXTRACT_VECTOR_ELT(SDValue Op,
                                                     SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VectorVT = Op.getOperand(0).getValueType();

  if (VectorVT.getVectorElementType() != MVT::i1)
    return SDValue();

  // Extend logical vector type before extracting an element:
  //
  // Before:
  //   (i64 (extract_vector_elt (nxv1i1), i64))
  // After:
  //   (i64 (extract_vector_elt (nxv1i64 (any_extend (nxv1i1))), i64))
  //
  // Before:
  //   (i64 (extract_vector_elt (nxv2i1), i64))
  // After:
  //   (i64 (extract_vector_elt (nxv2i32 (any_extend (nxv2i1))), i64))
  //
  // For LMUL>1 logical types we use LMUL>1 integer types (as opposed to LMUL=1
  // sub-byte types. Eg. nxv16i8 instead of nxv8i8 with two i4 elements in each
  // i8).
  // Before:
  //   (i64 (extract_vector_elt (nxv16i1), i64))
  // After:
  //   (i64 (extract_vector_elt (nxv16i8 (any_extend (nxv16i1))), i64))

  MVT ExtVectorEltVT =
      VectorVT.getVectorMinNumElements() <= 8
          ? MVT::getIntegerVT(64 / VectorVT.getVectorMinNumElements())
          : MVT::i8; // FIXME: ELEN=64 hardcoded.
  EVT ExtVectorVT = VectorVT.changeVectorElementType(ExtVectorEltVT);

  unsigned Opcode = ExtVectorEltVT == MVT::i64 ? ISD::EXTRACT_VECTOR_ELT
                                               : RISCVISD::EXTRACT_VECTOR_ELT;

  SDValue ExtOp0 =
      DAG.getNode(ISD::ANY_EXTEND, DL, ExtVectorVT, Op.getOperand(0));

  MVT ScalarResVT = Op.getValueType().getSimpleVT(); // Type already legalized.
  SDValue ExtractElt =
      DAG.getNode(Opcode, DL, ScalarResVT, ExtOp0, Op.getOperand(1));

  return DAG.getNode(ISD::AssertZext, DL, ScalarResVT, ExtractElt,
                     DAG.getValueType(MVT::i1));
}

SDValue RISCVTargetLowering::lowerFEXP(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  RTLIB::Libcall LC;
  switch (VT.getSimpleVT().SimpleTy) {
  default:
    llvm_unreachable("Unexpected VT");
  case MVT::nxv1f64:
    LC = RTLIB::EXP_NXV1F64;
    break;
  case MVT::nxv2f64:
    LC = RTLIB::EXP_NXV2F64;
    break;
  case MVT::nxv4f64:
    LC = RTLIB::EXP_NXV4F64;
    break;
  case MVT::nxv8f64:
    LC = RTLIB::EXP_NXV8F64;
    break;
  case MVT::nxv2f32:
    LC = RTLIB::EXP_NXV2F32;
    break;
  case MVT::nxv4f32:
    LC = RTLIB::EXP_NXV4F32;
    break;
  case MVT::nxv8f32:
    LC = RTLIB::EXP_NXV8F32;
    break;
  case MVT::nxv16f32:
    LC = RTLIB::EXP_NXV16F32;
    break;
  }

  MakeLibCallOptions CallOptions;
  SDValue Chain;
  SDValue Result;
  std::tie(Result, Chain) = makeLibCall(DAG, LC, Op.getValueType(),
                                        Op.getOperand(0), CallOptions, DL);
  return Result;
}

void RISCVTargetLowering::LowerOperationWrapper(
    SDNode *N, SmallVectorImpl<SDValue> &Results, SelectionDAG &DAG) const {
  SDValue Result = LowerOperation(SDValue(N, 0), DAG);

  if (Result)
    for (unsigned I = 0, E = Result->getNumValues(); I != E; ++I)
      Results.push_back(Result.getValue(I));
}

SDValue RISCVTargetLowering::LowerOperation(SDValue Op,
                                            SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    report_fatal_error("unimplemented operand");
  case ISD::GlobalAddress:
    return lowerGlobalAddress(Op, DAG);
  case ISD::BlockAddress:
    return lowerBlockAddress(Op, DAG);
  case ISD::ConstantPool:
    return lowerConstantPool(Op, DAG);
  case ISD::GlobalTLSAddress:
    return lowerGlobalTLSAddress(Op, DAG);
  case ISD::SELECT:
    return lowerSELECT(Op, DAG);
  case ISD::VASTART:
    return lowerVASTART(Op, DAG);
  case ISD::FRAMEADDR:
    return lowerFRAMEADDR(Op, DAG);
  case ISD::RETURNADDR:
    return lowerRETURNADDR(Op, DAG);
  case ISD::SHL_PARTS:
    return lowerShiftLeftParts(Op, DAG);
  case ISD::SRA_PARTS:
    return lowerShiftRightParts(Op, DAG, true);
  case ISD::SRL_PARTS:
    return lowerShiftRightParts(Op, DAG, false);
  case ISD::BITCAST: {
    assert(Subtarget.is64Bit() && Subtarget.hasStdExtF() &&
           "Unexpected custom legalisation");
    SDLoc DL(Op);
    SDValue Op0 = Op.getOperand(0);
    if (Op.getValueType() != MVT::f32 || Op0.getValueType() != MVT::i32)
      return SDValue();
    SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op0);
    SDValue FPConv = DAG.getNode(RISCVISD::FMV_W_X_RV64, DL, MVT::f32, NewOp0);
    return FPConv;
  }
  case ISD::INTRINSIC_WO_CHAIN:
    return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::INTRINSIC_W_CHAIN:
    return LowerINTRINSIC_W_CHAIN(Op, DAG);
  case ISD::INTRINSIC_VOID:
    return LowerINTRINSIC_VOID(Op, DAG);
  case ISD::VECTOR_SHUFFLE:
    return lowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::SIGN_EXTEND:
    return lowerSIGN_EXTEND(Op, DAG);
  case ISD::ZERO_EXTEND:
    return lowerZERO_EXTEND(Op, DAG);
  case ISD::TRUNCATE:
    return lowerTRUNCATE(Op, DAG);
  case ISD::SIGN_EXTEND_INREG:
    return lowerSIGN_EXTEND_INREG(Op, DAG);
  case ISD::MGATHER:
    return lowerMGATHER(Op, DAG);
  case ISD::MSCATTER:
    return lowerMSCATTER(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT:
    return lowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::FEXP:
    return lowerFEXP(Op, DAG);
  }
}

static SDValue getTargetNode(GlobalAddressSDNode *N, SDLoc DL, EVT Ty,
                             SelectionDAG &DAG, unsigned Flags) {
  return DAG.getTargetGlobalAddress(N->getGlobal(), DL, Ty, 0, Flags);
}

static SDValue getTargetNode(BlockAddressSDNode *N, SDLoc DL, EVT Ty,
                             SelectionDAG &DAG, unsigned Flags) {
  return DAG.getTargetBlockAddress(N->getBlockAddress(), Ty, N->getOffset(),
                                   Flags);
}

static SDValue getTargetNode(ConstantPoolSDNode *N, SDLoc DL, EVT Ty,
                             SelectionDAG &DAG, unsigned Flags) {
  return DAG.getTargetConstantPool(N->getConstVal(), Ty, N->getAlign(),
                                   N->getOffset(), Flags);
}

template <class NodeTy>
SDValue RISCVTargetLowering::getAddr(NodeTy *N, SelectionDAG &DAG,
                                     bool IsLocal) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());

  if (isPositionIndependent()) {
    SDValue Addr = getTargetNode(N, DL, Ty, DAG, 0);
    if (IsLocal)
      // Use PC-relative addressing to access the symbol. This generates the
      // pattern (PseudoLLA sym), which expands to (addi (auipc %pcrel_hi(sym))
      // %pcrel_lo(auipc)).
      return SDValue(DAG.getMachineNode(RISCV::PseudoLLA, DL, Ty, Addr), 0);

    // Use PC-relative addressing to access the GOT for this symbol, then load
    // the address from the GOT. This generates the pattern (PseudoLA sym),
    // which expands to (ld (addi (auipc %got_pcrel_hi(sym)) %pcrel_lo(auipc))).
    return SDValue(DAG.getMachineNode(RISCV::PseudoLA, DL, Ty, Addr), 0);
  }

  switch (getTargetMachine().getCodeModel()) {
  default:
    report_fatal_error("Unsupported code model for lowering");
  case CodeModel::Small: {
    // Generate a sequence for accessing addresses within the first 2 GiB of
    // address space. This generates the pattern (addi (lui %hi(sym)) %lo(sym)).
    SDValue AddrHi = getTargetNode(N, DL, Ty, DAG, RISCVII::MO_HI);
    SDValue AddrLo = getTargetNode(N, DL, Ty, DAG, RISCVII::MO_LO);
    SDValue MNHi = SDValue(DAG.getMachineNode(RISCV::LUI, DL, Ty, AddrHi), 0);
    return SDValue(DAG.getMachineNode(RISCV::ADDI, DL, Ty, MNHi, AddrLo), 0);
  }
  case CodeModel::Medium: {
    // Generate a sequence for accessing addresses within any 2GiB range within
    // the address space. This generates the pattern (PseudoLLA sym), which
    // expands to (addi (auipc %pcrel_hi(sym)) %pcrel_lo(auipc)).
    SDValue Addr = getTargetNode(N, DL, Ty, DAG, 0);
    return SDValue(DAG.getMachineNode(RISCV::PseudoLLA, DL, Ty, Addr), 0);
  }
  }
}

SDValue RISCVTargetLowering::lowerGlobalAddress(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  int64_t Offset = N->getOffset();
  MVT XLenVT = Subtarget.getXLenVT();

  const GlobalValue *GV = N->getGlobal();
  bool IsLocal = getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV);
  SDValue Addr = getAddr(N, DAG, IsLocal);

  // In order to maximise the opportunity for common subexpression elimination,
  // emit a separate ADD node for the global address offset instead of folding
  // it in the global address node. Later peephole optimisations may choose to
  // fold it back in when profitable.
  if (Offset != 0)
    return DAG.getNode(ISD::ADD, DL, Ty, Addr,
                       DAG.getConstant(Offset, DL, XLenVT));
  return Addr;
}

SDValue RISCVTargetLowering::lowerBlockAddress(SDValue Op,
                                               SelectionDAG &DAG) const {
  BlockAddressSDNode *N = cast<BlockAddressSDNode>(Op);

  return getAddr(N, DAG);
}

SDValue RISCVTargetLowering::lowerConstantPool(SDValue Op,
                                               SelectionDAG &DAG) const {
  ConstantPoolSDNode *N = cast<ConstantPoolSDNode>(Op);

  return getAddr(N, DAG);
}

SDValue RISCVTargetLowering::getStaticTLSAddr(GlobalAddressSDNode *N,
                                              SelectionDAG &DAG,
                                              bool UseGOT) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());
  const GlobalValue *GV = N->getGlobal();
  MVT XLenVT = Subtarget.getXLenVT();

  if (UseGOT) {
    // Use PC-relative addressing to access the GOT for this TLS symbol, then
    // load the address from the GOT and add the thread pointer. This generates
    // the pattern (PseudoLA_TLS_IE sym), which expands to
    // (ld (auipc %tls_ie_pcrel_hi(sym)) %pcrel_lo(auipc)).
    SDValue Addr = DAG.getTargetGlobalAddress(GV, DL, Ty, 0, 0);
    SDValue Load =
        SDValue(DAG.getMachineNode(RISCV::PseudoLA_TLS_IE, DL, Ty, Addr), 0);

    // Add the thread pointer.
    SDValue TPReg = DAG.getRegister(RISCV::X4, XLenVT);
    return DAG.getNode(ISD::ADD, DL, Ty, Load, TPReg);
  }

  // Generate a sequence for accessing the address relative to the thread
  // pointer, with the appropriate adjustment for the thread pointer offset.
  // This generates the pattern
  // (add (add_tprel (lui %tprel_hi(sym)) tp %tprel_add(sym)) %tprel_lo(sym))
  SDValue AddrHi =
      DAG.getTargetGlobalAddress(GV, DL, Ty, 0, RISCVII::MO_TPREL_HI);
  SDValue AddrAdd =
      DAG.getTargetGlobalAddress(GV, DL, Ty, 0, RISCVII::MO_TPREL_ADD);
  SDValue AddrLo =
      DAG.getTargetGlobalAddress(GV, DL, Ty, 0, RISCVII::MO_TPREL_LO);

  SDValue MNHi = SDValue(DAG.getMachineNode(RISCV::LUI, DL, Ty, AddrHi), 0);
  SDValue TPReg = DAG.getRegister(RISCV::X4, XLenVT);
  SDValue MNAdd = SDValue(
      DAG.getMachineNode(RISCV::PseudoAddTPRel, DL, Ty, MNHi, TPReg, AddrAdd),
      0);
  return SDValue(DAG.getMachineNode(RISCV::ADDI, DL, Ty, MNAdd, AddrLo), 0);
}

SDValue RISCVTargetLowering::getDynamicTLSAddr(GlobalAddressSDNode *N,
                                               SelectionDAG &DAG) const {
  SDLoc DL(N);
  EVT Ty = getPointerTy(DAG.getDataLayout());
  IntegerType *CallTy = Type::getIntNTy(*DAG.getContext(), Ty.getSizeInBits());
  const GlobalValue *GV = N->getGlobal();

  // Use a PC-relative addressing mode to access the global dynamic GOT address.
  // This generates the pattern (PseudoLA_TLS_GD sym), which expands to
  // (addi (auipc %tls_gd_pcrel_hi(sym)) %pcrel_lo(auipc)).
  SDValue Addr = DAG.getTargetGlobalAddress(GV, DL, Ty, 0, 0);
  SDValue Load =
      SDValue(DAG.getMachineNode(RISCV::PseudoLA_TLS_GD, DL, Ty, Addr), 0);

  // Prepare argument list to generate call.
  ArgListTy Args;
  ArgListEntry Entry;
  Entry.Node = Load;
  Entry.Ty = CallTy;
  Args.push_back(Entry);

  // Setup call to __tls_get_addr.
  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(DL)
      .setChain(DAG.getEntryNode())
      .setLibCallee(CallingConv::C, CallTy,
                    DAG.getExternalSymbol("__tls_get_addr", Ty),
                    std::move(Args));

  return LowerCallTo(CLI).first;
}

SDValue RISCVTargetLowering::lowerGlobalTLSAddress(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  int64_t Offset = N->getOffset();
  MVT XLenVT = Subtarget.getXLenVT();

  TLSModel::Model Model = getTargetMachine().getTLSModel(N->getGlobal());

  SDValue Addr;
  switch (Model) {
  case TLSModel::LocalExec:
    Addr = getStaticTLSAddr(N, DAG, /*UseGOT=*/false);
    break;
  case TLSModel::InitialExec:
    Addr = getStaticTLSAddr(N, DAG, /*UseGOT=*/true);
    break;
  case TLSModel::LocalDynamic:
  case TLSModel::GeneralDynamic:
    Addr = getDynamicTLSAddr(N, DAG);
    break;
  }

  // In order to maximise the opportunity for common subexpression elimination,
  // emit a separate ADD node for the global address offset instead of folding
  // it in the global address node. Later peephole optimisations may choose to
  // fold it back in when profitable.
  if (Offset != 0)
    return DAG.getNode(ISD::ADD, DL, Ty, Addr,
                       DAG.getConstant(Offset, DL, XLenVT));
  return Addr;
}

SDValue RISCVTargetLowering::lowerSELECT(SDValue Op, SelectionDAG &DAG) const {
  SDValue CondV = Op.getOperand(0);
  SDValue TrueV = Op.getOperand(1);
  SDValue FalseV = Op.getOperand(2);
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  MVT XLenVT = Subtarget.getXLenVT();

  if (VT.isScalableVector()) {
    // This sets all bits of a VR to the value of the condition bit. It sets more
    // bits than necessary but those can be safely ignored.
    // E.g. (nxv64i1:copy (nxv8i8:splat (i8:sext (i1:trunc (i64:CondV)))))
    SDValue Trunc = DAG.getNode(ISD::TRUNCATE, DL, MVT::i1, CondV);
    SDValue Sext = DAG.getNode(ISD::SIGN_EXTEND, DL, MVT::i8, Trunc);
    SDValue VecCondV = DAG.getNode(ISD::SPLAT_VECTOR, DL, MVT::nxv8i8, Sext);

    SDValue RC = DAG.getTargetConstant(RISCV::VRRegClass.getID(), DL, MVT::i64);
    SDValue CastVecCondV = SDValue(
        DAG.getMachineNode(TargetOpcode::COPY_TO_REGCLASS, DL,
                           VT.changeVectorElementType(MVT::i1), VecCondV, RC),
        0);
    return DAG.getNode(ISD::VSELECT, DL, VT, {CastVecCondV, TrueV, FalseV});
  }

  // If the result type is XLenVT and CondV is the output of a SETCC node
  // which also operated on XLenVT inputs, then merge the SETCC node into the
  // lowered RISCVISD::SELECT_CC to take advantage of the integer
  // compare+branch instructions. i.e.:
  // (select (setcc lhs, rhs, cc), truev, falsev)
  // -> (riscvisd::select_cc lhs, rhs, cc, truev, falsev)
  if (VT.getSimpleVT() == XLenVT && CondV.getOpcode() == ISD::SETCC &&
      CondV.getOperand(0).getSimpleValueType() == XLenVT) {
    SDValue LHS = CondV.getOperand(0);
    SDValue RHS = CondV.getOperand(1);
    auto CC = cast<CondCodeSDNode>(CondV.getOperand(2));
    ISD::CondCode CCVal = CC->get();

    normaliseSetCC(LHS, RHS, CCVal);

    SDValue TargetCC = DAG.getConstant(CCVal, DL, XLenVT);
    SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
    SDValue Ops[] = {LHS, RHS, TargetCC, TrueV, FalseV};
    return DAG.getNode(RISCVISD::SELECT_CC, DL, VTs, Ops);
  }

  // Otherwise:
  // (select condv, truev, falsev)
  // -> (riscvisd::select_cc condv, zero, setne, truev, falsev)
  SDValue Zero = DAG.getConstant(0, DL, XLenVT);
  SDValue SetNE = DAG.getConstant(ISD::SETNE, DL, XLenVT);

  SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
  SDValue Ops[] = {CondV, Zero, SetNE, TrueV, FalseV};

  return DAG.getNode(RISCVISD::SELECT_CC, DL, VTs, Ops);
}

SDValue RISCVTargetLowering::lowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  RISCVMachineFunctionInfo *FuncInfo = MF.getInfo<RISCVMachineFunctionInfo>();

  SDLoc DL(Op);
  SDValue FI = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(),
                                 getPointerTy(MF.getDataLayout()));

  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), DL, FI, Op.getOperand(1),
                      MachinePointerInfo(SV));
}

SDValue RISCVTargetLowering::lowerFRAMEADDR(SDValue Op,
                                            SelectionDAG &DAG) const {
  const RISCVRegisterInfo &RI = *Subtarget.getRegisterInfo();
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MFI.setFrameAddressIsTaken(true);
  Register FrameReg = RI.getFrameRegister(MF);
  int XLenInBytes = Subtarget.getXLen() / 8;

  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  SDValue FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), DL, FrameReg, VT);
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  while (Depth--) {
    int Offset = -(XLenInBytes * 2);
    SDValue Ptr = DAG.getNode(ISD::ADD, DL, VT, FrameAddr,
                              DAG.getIntPtrConstant(Offset, DL));
    FrameAddr =
        DAG.getLoad(VT, DL, DAG.getEntryNode(), Ptr, MachinePointerInfo());
  }
  return FrameAddr;
}

SDValue RISCVTargetLowering::lowerRETURNADDR(SDValue Op,
                                             SelectionDAG &DAG) const {
  const RISCVRegisterInfo &RI = *Subtarget.getRegisterInfo();
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MFI.setReturnAddressIsTaken(true);
  MVT XLenVT = Subtarget.getXLenVT();
  int XLenInBytes = Subtarget.getXLen() / 8;

  if (verifyReturnAddressArgumentIsConstant(Op, DAG))
    return SDValue();

  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  if (Depth) {
    int Off = -XLenInBytes;
    SDValue FrameAddr = lowerFRAMEADDR(Op, DAG);
    SDValue Offset = DAG.getConstant(Off, DL, VT);
    return DAG.getLoad(VT, DL, DAG.getEntryNode(),
                       DAG.getNode(ISD::ADD, DL, VT, FrameAddr, Offset),
                       MachinePointerInfo());
  }

  // Return the value of the return address register, marking it an implicit
  // live-in.
  Register Reg = MF.addLiveIn(RI.getRARegister(), getRegClassFor(XLenVT));
  return DAG.getCopyFromReg(DAG.getEntryNode(), DL, Reg, XLenVT);
}

SDValue RISCVTargetLowering::lowerShiftLeftParts(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  EVT VT = Lo.getValueType();

  // if Shamt-XLEN < 0: // Shamt < XLEN
  //   Lo = Lo << Shamt
  //   Hi = (Hi << Shamt) | ((Lo >>u 1) >>u (XLEN-1 - Shamt))
  // else:
  //   Lo = 0
  //   Hi = Lo << (Shamt-XLEN)

  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue One = DAG.getConstant(1, DL, VT);
  SDValue MinusXLen = DAG.getConstant(-(int)Subtarget.getXLen(), DL, VT);
  SDValue XLenMinus1 = DAG.getConstant(Subtarget.getXLen() - 1, DL, VT);
  SDValue ShamtMinusXLen = DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusXLen);
  SDValue XLenMinus1Shamt = DAG.getNode(ISD::SUB, DL, VT, XLenMinus1, Shamt);

  SDValue LoTrue = DAG.getNode(ISD::SHL, DL, VT, Lo, Shamt);
  SDValue ShiftRight1Lo = DAG.getNode(ISD::SRL, DL, VT, Lo, One);
  SDValue ShiftRightLo =
      DAG.getNode(ISD::SRL, DL, VT, ShiftRight1Lo, XLenMinus1Shamt);
  SDValue ShiftLeftHi = DAG.getNode(ISD::SHL, DL, VT, Hi, Shamt);
  SDValue HiTrue = DAG.getNode(ISD::OR, DL, VT, ShiftLeftHi, ShiftRightLo);
  SDValue HiFalse = DAG.getNode(ISD::SHL, DL, VT, Lo, ShamtMinusXLen);

  SDValue CC = DAG.getSetCC(DL, VT, ShamtMinusXLen, Zero, ISD::SETLT);

  Lo = DAG.getNode(ISD::SELECT, DL, VT, CC, LoTrue, Zero);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, CC, HiTrue, HiFalse);

  SDValue Parts[2] = {Lo, Hi};
  return DAG.getMergeValues(Parts, DL);
}

SDValue RISCVTargetLowering::lowerShiftRightParts(SDValue Op, SelectionDAG &DAG,
                                                  bool IsSRA) const {
  SDLoc DL(Op);
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  EVT VT = Lo.getValueType();

  // SRA expansion:
  //   if Shamt-XLEN < 0: // Shamt < XLEN
  //     Lo = (Lo >>u Shamt) | ((Hi << 1) << (XLEN-1 - Shamt))
  //     Hi = Hi >>s Shamt
  //   else:
  //     Lo = Hi >>s (Shamt-XLEN);
  //     Hi = Hi >>s (XLEN-1)
  //
  // SRL expansion:
  //   if Shamt-XLEN < 0: // Shamt < XLEN
  //     Lo = (Lo >>u Shamt) | ((Hi << 1) << (XLEN-1 - Shamt))
  //     Hi = Hi >>u Shamt
  //   else:
  //     Lo = Hi >>u (Shamt-XLEN);
  //     Hi = 0;

  unsigned ShiftRightOp = IsSRA ? ISD::SRA : ISD::SRL;

  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue One = DAG.getConstant(1, DL, VT);
  SDValue MinusXLen = DAG.getConstant(-(int)Subtarget.getXLen(), DL, VT);
  SDValue XLenMinus1 = DAG.getConstant(Subtarget.getXLen() - 1, DL, VT);
  SDValue ShamtMinusXLen = DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusXLen);
  SDValue XLenMinus1Shamt = DAG.getNode(ISD::SUB, DL, VT, XLenMinus1, Shamt);

  SDValue ShiftRightLo = DAG.getNode(ISD::SRL, DL, VT, Lo, Shamt);
  SDValue ShiftLeftHi1 = DAG.getNode(ISD::SHL, DL, VT, Hi, One);
  SDValue ShiftLeftHi =
      DAG.getNode(ISD::SHL, DL, VT, ShiftLeftHi1, XLenMinus1Shamt);
  SDValue LoTrue = DAG.getNode(ISD::OR, DL, VT, ShiftRightLo, ShiftLeftHi);
  SDValue HiTrue = DAG.getNode(ShiftRightOp, DL, VT, Hi, Shamt);
  SDValue LoFalse = DAG.getNode(ShiftRightOp, DL, VT, Hi, ShamtMinusXLen);
  SDValue HiFalse =
      IsSRA ? DAG.getNode(ISD::SRA, DL, VT, Hi, XLenMinus1) : Zero;

  SDValue CC = DAG.getSetCC(DL, VT, ShamtMinusXLen, Zero, ISD::SETLT);

  Lo = DAG.getNode(ISD::SELECT, DL, VT, CC, LoTrue, LoFalse);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, CC, HiTrue, HiFalse);

  SDValue Parts[2] = {Lo, Hi};
  return DAG.getMergeValues(Parts, DL);
}

static SDValue LowerVPUnorderedFCmp(unsigned EPIIntNo, const SDValue &Op1,
                                    const SDValue &Op2, const SDValue &EVL,
                                    EVT VT, SelectionDAG &DAG,
                                    const SDLoc &DL) {
  // Note: Implementing masked intrinsic as unmasked (this is correct given
  // that masked-off elements are undef and the operation is only applied
  // to ordered elements).
  //
  // %0 = vmfeq.vv %a, %a
  // %1 = vmfeq.vv %b, %b
  // %2 = vmand.mm %0, %1
  // v0 = %2
  // %3 = vm<fcmp>.vv %a, %b, v0.t
  // %result = vmornot.mm %3, %2

  assert(EVL.getValueType() == MVT::i32 && "Unexpected operand");
  SDValue AnyExtEVL = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, EVL);

  SDValue FeqOp1Operands[] = {
      DAG.getTargetConstant(Intrinsic::epi_vmfeq, DL, MVT::i64), Op1, Op1,
      AnyExtEVL
  };
  SDValue FeqOp2Operands[] = {
      DAG.getTargetConstant(Intrinsic::epi_vmfeq, DL, MVT::i64), Op2, Op2,
      AnyExtEVL
  };
  SDValue FeqOp1 = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT, FeqOp1Operands);
  SDValue FeqOp2 = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT, FeqOp2Operands);

  SDValue AndOperands[] = {
      DAG.getTargetConstant(Intrinsic::epi_vmand, DL, MVT::i64), FeqOp1, FeqOp2,
      AnyExtEVL
  };
  SDValue And = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT, AndOperands);

  SDValue FCmpOperands[] = {
      DAG.getTargetConstant(EPIIntNo, DL, MVT::i64),
      DAG.getNode(ISD::UNDEF, DL, VT), // Merge.
      Op1, Op2, And, AnyExtEVL
  };
  SDValue FCmp = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT, FCmpOperands);

  SDValue OrNotOperands[] = {
      DAG.getTargetConstant(Intrinsic::epi_vmornot, DL, MVT::i64), FCmp, And,
      AnyExtEVL
  };
  return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, VT, OrNotOperands);
}

static bool IsSplatOfOne(const SDValue &MaskOp) {
  ConstantSDNode *C;
  return MaskOp.getOpcode() == ISD::SPLAT_VECTOR &&
         (C = dyn_cast<ConstantSDNode>(MaskOp.getOperand(0))) &&
         C->getZExtValue() == 1;
}

static SDValue LowerVPIntrinsicConversion(SDValue Op, SelectionDAG &DAG) {
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  SDLoc DL(Op);

  EVT DstType = Op.getValueType();
  uint64_t DstTypeSize = DstType.getScalarSizeInBits();
  SDValue SrcOp = Op.getOperand(1);
  EVT SrcType = SrcOp.getValueType();
  uint64_t SrcTypeSize = SrcType.getScalarSizeInBits();

  assert(isPowerOf2_64(DstTypeSize) && isPowerOf2_64(SrcTypeSize) &&
         "Types must be powers of two");
  int Ratio =
      std::max(DstTypeSize, SrcTypeSize) / std::min(DstTypeSize, SrcTypeSize);

  unsigned MaskOpNo;
  unsigned EVLOpNo;
  bool IsMasked;
  unsigned EPIIntNo;

  if (Ratio == 1) {
    switch (IntNo) {
    default:
      llvm_unreachable("Unexpected intrinsic");
      break;
    case Intrinsic::vp_sitofp:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo =
          IsMasked ? Intrinsic::epi_vfcvt_f_x_mask : Intrinsic::epi_vfcvt_f_x;
      break;
    case Intrinsic::vp_uitofp:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo =
          IsMasked ? Intrinsic::epi_vfcvt_f_xu_mask : Intrinsic::epi_vfcvt_f_xu;
      break;
    case Intrinsic::vp_fptosi:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo =
          IsMasked ? Intrinsic::epi_vfcvt_x_f_mask : Intrinsic::epi_vfcvt_x_f;
      break;
    case Intrinsic::vp_fptoui:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo =
          IsMasked ? Intrinsic::epi_vfcvt_xu_f_mask : Intrinsic::epi_vfcvt_xu_f;
      break;
    }
  } else if (Ratio == 2 && DstTypeSize > SrcTypeSize) {
    switch (IntNo) {
    default:
      llvm_unreachable("Unexpected intrinsic");
      break;
    case Intrinsic::vp_sitofp:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo =
          IsMasked ? Intrinsic::epi_vfwcvt_f_x_mask : Intrinsic::epi_vfwcvt_f_x;
      break;
    case Intrinsic::vp_uitofp:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo = IsMasked ? Intrinsic::epi_vfwcvt_f_xu_mask
                          : Intrinsic::epi_vfwcvt_f_xu;
      break;
    case Intrinsic::vp_fpext:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo = IsMasked ? Intrinsic::epi_vfwcvt_f_f_mask
                          : Intrinsic::epi_vfwcvt_f_f;
      break;
    case Intrinsic::vp_fptosi:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo =
          IsMasked ? Intrinsic::epi_vfwcvt_x_f_mask : Intrinsic::epi_vfwcvt_x_f;
      break;
    case Intrinsic::vp_fptoui:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo =
          IsMasked ? Intrinsic::epi_vfwcvt_xu_f_mask : Intrinsic::epi_vfwcvt_xu_f;
      break;
    case Intrinsic::vp_sext:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo = IsMasked ? Intrinsic::epi_vwadd_mask
                          : Intrinsic::epi_vwadd;
      break;
    case Intrinsic::vp_zext:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo = IsMasked ? Intrinsic::epi_vwaddu_mask
                          : Intrinsic::epi_vwaddu;
      break;
    }
  } else if (Ratio == 2 && DstTypeSize < SrcTypeSize) {
    switch (IntNo) {
    default:
      llvm_unreachable("Unexpected intrinsic");
      break;
    case Intrinsic::vp_sitofp:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo =
          IsMasked ? Intrinsic::epi_vfncvt_f_x_mask : Intrinsic::epi_vfncvt_f_x;
      break;
    case Intrinsic::vp_uitofp:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo = IsMasked ? Intrinsic::epi_vfncvt_f_xu_mask
                          : Intrinsic::epi_vfncvt_f_xu;
      break;
    case Intrinsic::vp_fptrunc:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo = IsMasked ? Intrinsic::epi_vfncvt_f_f_mask
                          : Intrinsic::epi_vfncvt_f_f;
      break;
    case Intrinsic::vp_fptosi:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo =
          IsMasked ? Intrinsic::epi_vfncvt_x_f_mask : Intrinsic::epi_vfncvt_x_f;
      break;
    case Intrinsic::vp_fptoui:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo =
          IsMasked ? Intrinsic::epi_vfncvt_xu_f_mask : Intrinsic::epi_vfncvt_xu_f;
      break;
    case Intrinsic::vp_trunc:
      MaskOpNo = 2;
      EVLOpNo = 3;
      IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
      EPIIntNo = IsMasked ? Intrinsic::epi_vnsrl_mask
                          : Intrinsic::epi_vnsrl;
      break;
    }
  }

  // Straightforward cases.
  if (Ratio == 1 || Ratio == 2) {
    std::vector<SDValue> Operands;
    Operands.reserve(3 + IsMasked * 2);

    Operands.push_back(DAG.getTargetConstant(EPIIntNo, DL, MVT::i64));

    if (IsMasked)
      Operands.push_back(
          DAG.getNode(ISD::UNDEF, DL, Op.getValueType())); // Merge.

    Operands.push_back(SrcOp);

    // Special case because there is no unary narrowing instruction.
    if (IntNo == Intrinsic::vp_trunc || IntNo == Intrinsic::vp_zext ||
        IntNo == Intrinsic::vp_sext) {
      Operands.push_back(DAG.getTargetConstant(0, DL, MVT::i64));
    }

    if (IsMasked)
      Operands.push_back(Op.getOperand(MaskOpNo)); // Mask.

    assert(Op.getOperand(EVLOpNo).getValueType() == MVT::i32 &&
           "Unexpected operand");
    Operands.push_back(DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64,
                                   Op.getOperand(EVLOpNo))); // EVL.

    SDValue Result = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, Op.getValueType(),
                       Operands);

    return Result;
  }

  // Ideas to implement this? Use a narrowing/widening operation and then
  // the required number of truncations/extensions.
  report_fatal_error("FP conversions not mappable to "
                     "widenings/narrowings not implemented yet");
  return SDValue();
}

static SDValue LowerVPINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) {
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  SDLoc DL(Op);

  // Many instructions allow commuting the second operand with the first one.
  // This is beneficial when we can use a scalar in the second operand as way
  // to fold a vector splat.
  auto GetCanonicalCommutativePerm = [&](SmallVector<unsigned, 3> VOpsPerm) {
    if (VOpsPerm.size() < 2)
      return VOpsPerm;

    SDValue Operand0 = Op.getOperand(VOpsPerm[0]);
    SDValue Operand1 = Op.getOperand(VOpsPerm[1]);

    if (Operand0.getOpcode() == ISD::SPLAT_VECTOR &&
        Operand1.getOpcode() != ISD::SPLAT_VECTOR) {
      SmallVector<unsigned, 3> CanonicalVOpsPerm = {VOpsPerm[1], VOpsPerm[0]};

      for (unsigned i = 2; i < VOpsPerm.size(); ++i) {
        CanonicalVOpsPerm.push_back(VOpsPerm[i]);
      }

      return CanonicalVOpsPerm;
    }

    return VOpsPerm;
  };

  SmallVector<unsigned, 3> VOpsPerm;
  unsigned ScalarOpNo = 0;
  unsigned MaskOpNo;
  unsigned EVLOpNo;
  bool IsMasked;
  unsigned EPIIntNo;
  bool IsLogical = Op.getValueType().isVector() &&
                   Op.getValueType().getVectorElementType() == MVT::i1;
  switch (IntNo) {
  default:
    llvm_unreachable("Unexpected intrinsic");
  case Intrinsic::vp_add:
    VOpsPerm = GetCanonicalCommutativePerm({1, 2});
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo = IsMasked ? Intrinsic::epi_vadd_mask : Intrinsic::epi_vadd;
    break;
  case Intrinsic::vp_sub:
    VOpsPerm = {1, 2};
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo = IsMasked ? Intrinsic::epi_vsub_mask : Intrinsic::epi_vsub;
    break;
  case Intrinsic::vp_mul:
    VOpsPerm = GetCanonicalCommutativePerm({1, 2});
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo = IsMasked ? Intrinsic::epi_vmul_mask : Intrinsic::epi_vmul;
    break;
  case Intrinsic::vp_sdiv:
    VOpsPerm = {1, 2};
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo = IsMasked ? Intrinsic::epi_vdiv_mask : Intrinsic::epi_vdiv;
    break;
  case Intrinsic::vp_srem:
    VOpsPerm = {1, 2};
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo = IsMasked ? Intrinsic::epi_vrem_mask : Intrinsic::epi_vrem;
    break;
  case Intrinsic::vp_udiv:
    VOpsPerm = {1, 2};
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo = IsMasked ? Intrinsic::epi_vdivu_mask : Intrinsic::epi_vdivu;
    break;
  case Intrinsic::vp_urem:
    VOpsPerm = {1, 2};
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo = IsMasked ? Intrinsic::epi_vremu_mask : Intrinsic::epi_vremu;
    break;
  case Intrinsic::vp_and:
    VOpsPerm = GetCanonicalCommutativePerm({1, 2});
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    if (IsLogical) {
      ScalarOpNo = 0;
      EPIIntNo = Intrinsic::epi_vmand;
      if (IsMasked)
        report_fatal_error("Unimplemented masked logical AND operation");
    } else
      EPIIntNo = IsMasked ? Intrinsic::epi_vand_mask : Intrinsic::epi_vand;
    break;
  case Intrinsic::vp_or:
    VOpsPerm = GetCanonicalCommutativePerm({1, 2});
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    if (IsLogical) {
      ScalarOpNo = 0;
      EPIIntNo = Intrinsic::epi_vmor;
      if (IsMasked)
        report_fatal_error("Unimplemented masked logical OR operation");
    } else
      EPIIntNo = IsMasked ? Intrinsic::epi_vor_mask : Intrinsic::epi_vor;
    break;
  case Intrinsic::vp_xor:
    VOpsPerm = GetCanonicalCommutativePerm({1, 2});
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    if (IsLogical) {
      ScalarOpNo = 0;
      EPIIntNo = Intrinsic::epi_vmxor;
      if (IsMasked)
        report_fatal_error("Unimplemented masked logical XOR operation");
    } else
      EPIIntNo = IsMasked ? Intrinsic::epi_vxor_mask : Intrinsic::epi_vxor;
    break;
  case Intrinsic::vp_ashr:
    VOpsPerm = {1, 2};
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo = IsMasked ? Intrinsic::epi_vsra_mask : Intrinsic::epi_vsra;
    break;
  case Intrinsic::vp_lshr:
    VOpsPerm = {1, 2};
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo = IsMasked ? Intrinsic::epi_vsrl_mask : Intrinsic::epi_vsrl;
    break;
  case Intrinsic::vp_shl:
    VOpsPerm = {1, 2};
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo = IsMasked ? Intrinsic::epi_vsll_mask : Intrinsic::epi_vsll;
    break;
  case Intrinsic::vp_fadd:
    VOpsPerm = {1, 2};
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo = IsMasked ? Intrinsic::epi_vfadd_mask : Intrinsic::epi_vfadd;
    break;
  case Intrinsic::vp_fsub:
    VOpsPerm = {1, 2};
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo = IsMasked ? Intrinsic::epi_vfsub_mask : Intrinsic::epi_vfsub;
    break;
  case Intrinsic::vp_fmul:
    VOpsPerm = GetCanonicalCommutativePerm({1, 2});
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo = IsMasked ? Intrinsic::epi_vfmul_mask : Intrinsic::epi_vfmul;
    break;
  case Intrinsic::vp_fdiv:
    VOpsPerm = {1, 2};
    ScalarOpNo = 2;
    MaskOpNo = 3;
    EVLOpNo = 4;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo = IsMasked ? Intrinsic::epi_vfdiv_mask : Intrinsic::epi_vfdiv;
    break;
  case Intrinsic::vp_frem:
    // FIXME Needs to be expanded.
    report_fatal_error("Unimplemented intrinsic vp_frem");
    break;
  case Intrinsic::vp_fma:
    VOpsPerm = GetCanonicalCommutativePerm({1, 2, 3});
    ScalarOpNo = 2;
    MaskOpNo = 4;
    EVLOpNo = 5;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo = IsMasked ? Intrinsic::epi_vfmadd_mask : Intrinsic::epi_vfmadd;
    break;
  case Intrinsic::vp_fneg:
    VOpsPerm = {1, 1};
    MaskOpNo = 2;
    EVLOpNo = 3;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));
    EPIIntNo =
        IsMasked ? Intrinsic::epi_vfsgnjn_mask : Intrinsic::epi_vfsgnjn;
    break;
  case Intrinsic::vp_icmp: {
    VOpsPerm = {1, 2};
    ScalarOpNo = 2;
    MaskOpNo = 4;
    EVLOpNo = 5;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));

    unsigned Cmp = cast<ConstantSDNode>(Op.getOperand(3))->getZExtValue();
    switch (Cmp) {
    case CmpInst::ICMP_EQ:
      VOpsPerm = GetCanonicalCommutativePerm({1, 2});
      EPIIntNo = IsMasked ? Intrinsic::epi_vmseq_mask : Intrinsic::epi_vmseq;
      break;
    case CmpInst::ICMP_NE:
      VOpsPerm = GetCanonicalCommutativePerm({1, 2});
      EPIIntNo = IsMasked ? Intrinsic::epi_vmsne_mask : Intrinsic::epi_vmsne;
      break;
    case CmpInst::ICMP_UGT: {
      SDValue RHS = Op.getOperand(2);
      if (RHS.getOpcode() == ISD::SPLAT_VECTOR) {
        EPIIntNo =
            IsMasked ? Intrinsic::epi_vmsgtu_mask : Intrinsic::epi_vmsgtu_mask;
      } else {
        VOpsPerm = {2, 1};
        EPIIntNo =
            IsMasked ? Intrinsic::epi_vmsltu_mask : Intrinsic::epi_vmsltu;
      }
      break;
    }
    case CmpInst::ICMP_UGE:
      // Note: The ISA does not provide vmsgeu.vx to fold a scalar.
      VOpsPerm = {2, 1};
      EPIIntNo = IsMasked ? Intrinsic::epi_vmsleu_mask : Intrinsic::epi_vmsleu;
      break;
    case CmpInst::ICMP_ULT: {
      SDValue LHS = Op.getOperand(1);
      if (LHS.getOpcode() == ISD::SPLAT_VECTOR) {
        VOpsPerm = {2, 1};
        EPIIntNo =
            IsMasked ? Intrinsic::epi_vmsgtu_mask : Intrinsic::epi_vmsgtu;
      } else {
        EPIIntNo =
            IsMasked ? Intrinsic::epi_vmsltu_mask : Intrinsic::epi_vmsltu;
      }
      break;
    }
    case CmpInst::ICMP_ULE:
      // Note: The ISA does not provide vmsgeu.vx so we can't flip the operands
      // to fold a scalar.
      EPIIntNo = IsMasked ? Intrinsic::epi_vmsleu_mask : Intrinsic::epi_vmsleu;
      break;
    case CmpInst::ICMP_SGT: {
      SDValue RHS = Op.getOperand(2);
      if (RHS.getOpcode() == ISD::SPLAT_VECTOR) {
        EPIIntNo =
            IsMasked ? Intrinsic::epi_vmsgt_mask : Intrinsic::epi_vmsgt_mask;
      } else {
        VOpsPerm = {2, 1};
        EPIIntNo = IsMasked ? Intrinsic::epi_vmslt_mask : Intrinsic::epi_vmslt;
      }
      break;
    }
    case CmpInst::ICMP_SGE:
      // Note: The ISA does not provide vmsge.vx to fold a scalar.
      VOpsPerm = {2, 1};
      EPIIntNo = IsMasked ? Intrinsic::epi_vmsle_mask : Intrinsic::epi_vmsle;
      break;
    case CmpInst::ICMP_SLT:
      EPIIntNo = IsMasked ? Intrinsic::epi_vmslt_mask : Intrinsic::epi_vmslt;
      break;
    case CmpInst::ICMP_SLE:
      // Note: The ISA does not provide vmsge.vx so we can't flip the operands
      // to fold a scalar.
      EPIIntNo = IsMasked ? Intrinsic::epi_vmsle_mask : Intrinsic::epi_vmsle;
      break;
    }
    break;
  }
  case Intrinsic::vp_fcmp: {
    VOpsPerm = {1, 2};
    ScalarOpNo = 2;
    MaskOpNo = 4;
    EVLOpNo = 5;
    IsMasked = !IsSplatOfOne(Op.getOperand(MaskOpNo));

    unsigned FCmp = cast<ConstantSDNode>(Op.getOperand(3))->getZExtValue();
    switch (FCmp) {
    case FCmpInst::FCMP_FALSE:
      // FIXME Lower to vmclr.
      report_fatal_error("Unimplemented case FCMP_FALSE for intrinsic vp_fcmp");
      break;
    case FCmpInst::FCMP_OEQ:
      VOpsPerm = GetCanonicalCommutativePerm({1, 2});
      EPIIntNo = IsMasked ? Intrinsic::epi_vmfeq_mask : Intrinsic::epi_vmfeq;
      break;
    case FCmpInst::FCMP_OGT: {
      SDValue RHS = Op.getOperand(2);
      if (RHS.getOpcode() == ISD::SPLAT_VECTOR) {
        EPIIntNo = IsMasked ? Intrinsic::epi_vmfgt_mask : Intrinsic::epi_vmfgt;
      } else {
        VOpsPerm = {2, 1};
        EPIIntNo = IsMasked ? Intrinsic::epi_vmflt_mask : Intrinsic::epi_vmflt;
      }
      break;
    }
    case FCmpInst::FCMP_OGE: {
      SDValue RHS = Op.getOperand(2);
      if (RHS.getOpcode() == ISD::SPLAT_VECTOR) {
        EPIIntNo = IsMasked ? Intrinsic::epi_vmfge_mask : Intrinsic::epi_vmfge;
      } else {
        VOpsPerm = {2, 1};
        EPIIntNo = IsMasked ? Intrinsic::epi_vmfle_mask : Intrinsic::epi_vmfle;
      }
      break;
    }
    case FCmpInst::FCMP_OLT: {
      SDValue LHS = Op.getOperand(1);
      if (LHS.getOpcode() == ISD::SPLAT_VECTOR) {
        VOpsPerm = {2, 1};
        EPIIntNo = IsMasked ? Intrinsic::epi_vmfgt_mask : Intrinsic::epi_vmfgt;
      } else {
        EPIIntNo = IsMasked ? Intrinsic::epi_vmflt_mask : Intrinsic::epi_vmflt;
      }
      break;
    }
    case FCmpInst::FCMP_OLE: {
      SDValue LHS = Op.getOperand(1);
      if (LHS.getOpcode() == ISD::SPLAT_VECTOR) {
        VOpsPerm = {2, 1};
        EPIIntNo = IsMasked ? Intrinsic::epi_vmfge_mask : Intrinsic::epi_vmfge;
      } else {
        EPIIntNo = IsMasked ? Intrinsic::epi_vmfle_mask : Intrinsic::epi_vmfle;
      }
      break;
    }
    case FCmpInst::FCMP_ONE:
      report_fatal_error("Unimplemented case FCMP_ONE for intrinsic vp_fcmp");
      break;
    case FCmpInst::FCMP_ORD:
      report_fatal_error("Unimplemented case FCMP_ORD for intrinsic vp_fcmp");
      break;
    case FCmpInst::FCMP_UEQ:
      report_fatal_error("Unimplemented case FCMP_UEQ for intrinsic vp_fcmp");
      break;
      // FIXME: Fold scalar operands also in unordered comparisons.
    case FCmpInst::FCMP_UGT:
      return LowerVPUnorderedFCmp(Intrinsic::epi_vmfgt_mask, Op.getOperand(1),
                                  Op.getOperand(2), Op.getOperand(EVLOpNo),
                                  Op.getValueType(), DAG, DL);
    case FCmpInst::FCMP_UGE:
      return LowerVPUnorderedFCmp(Intrinsic::epi_vmfge_mask, Op.getOperand(1),
                                  Op.getOperand(2), Op.getOperand(EVLOpNo),
                                  Op.getValueType(), DAG, DL);
    case FCmpInst::FCMP_ULT:
      return LowerVPUnorderedFCmp(Intrinsic::epi_vmflt_mask, Op.getOperand(1),
                                  Op.getOperand(2), Op.getOperand(EVLOpNo),
                                  Op.getValueType(), DAG, DL);
    case FCmpInst::FCMP_ULE:
      return LowerVPUnorderedFCmp(Intrinsic::epi_vmfle_mask, Op.getOperand(1),
                                  Op.getOperand(2), Op.getOperand(EVLOpNo),
                                  Op.getValueType(), DAG, DL);
    case FCmpInst::FCMP_UNE:
      VOpsPerm = GetCanonicalCommutativePerm({1, 2});
      EPIIntNo = IsMasked ? Intrinsic::epi_vmfne_mask : Intrinsic::epi_vmfne;
      break;
    case FCmpInst::FCMP_UNO:
      report_fatal_error("Unimplemented case FCMP_UNO for intrinsic vp_fcmp");
      break;
    case FCmpInst::FCMP_TRUE:
      // FIXME Lower to vmset.
      report_fatal_error("Unimplemented case FCMP_TRUE for intrinsic vp_fcmp");
      break;
    }
    break;
  }
  case Intrinsic::vp_select: {
    VOpsPerm = {3, 2, 1};
    ScalarOpNo = 2;
    MaskOpNo = -1;
    EVLOpNo = 4;
    IsMasked = false;

    const EVT &ElementType = Op.getValueType().getVectorElementType();
    if (ElementType.isFloatingPoint()) {
      EPIIntNo = Intrinsic::epi_vfmerge;
      break;
    } else if (ElementType != MVT::i1) {
      EPIIntNo = Intrinsic::epi_vmerge;
      break;
    }

    // llvm.vp.select applied to mask types.
    // vp.select computes a masked merge from two values. This can be naively
    // computed doing: (a & mask) | (b & ~mask)
    // However, the bithack described in
    // https://graphics.stanford.edu/~seander/bithacks.html#MaskedMerge shows
    // how it can be optimized to: b ^ ((b ^ a) & mask)

    assert(Op.getOperand(EVLOpNo).getValueType() == MVT::i32 &&
           "Unexpected operand");
    SDValue EVL =
        DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op.getOperand(EVLOpNo));

    const SDValue &OpA = Op.getOperand(2);
    const SDValue &OpB = Op.getOperand(3);
    const SDValue &Mask = Op.getOperand(1);

    SDValue BXorA =
        DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, Op.getValueType(),
                    {DAG.getTargetConstant(Intrinsic::epi_vmxor, DL, MVT::i64),
                     OpB, OpA, EVL});

    SDValue XorAndMask =
        DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, Op.getValueType(),
                    {DAG.getTargetConstant(Intrinsic::epi_vmand, DL, MVT::i64),
                     BXorA, Mask, EVL});

    return DAG.getNode(
        ISD::INTRINSIC_WO_CHAIN, DL, Op.getValueType(),
        {DAG.getTargetConstant(Intrinsic::epi_vmxor, DL, MVT::i64), OpB,
         XorAndMask, EVL});
  }
  case Intrinsic::vp_bitcast: {
    assert(Op.getValueType().getSizeInBits() ==
               Op.getOperand(1).getValueType().getSizeInBits() &&
           "Unable to bitcast values of unmatching sizes");
    return Op.getOperand(1);
  }
  }

  std::vector<SDValue> Operands;
  Operands.reserve(2 + VOpsPerm.size() + IsMasked * 2);

  Operands.push_back(DAG.getTargetConstant(EPIIntNo, DL, MVT::i64));

  if (IsMasked && IntNo != Intrinsic::vp_fma)
    Operands.push_back(
        DAG.getNode(ISD::UNDEF, DL, Op.getValueType())); // Merge.

  for (auto VOpI = VOpsPerm.begin(), VOpE = VOpsPerm.end(), VOpStart = VOpI;
       VOpI != VOpE; VOpI++) {
    SDValue Operand = Op.getOperand(*VOpI);
    // +1 because we skip the IntrinsicID
    unsigned OpIdx = (VOpI - VOpStart) + 1;
    if ((OpIdx == ScalarOpNo) && (Operand.getOpcode() == ISD::SPLAT_VECTOR))
      Operand = Operand.getOperand(0);
    Operands.push_back(Operand);
  }

  if (IsMasked)
    Operands.push_back(Op.getOperand(MaskOpNo)); // Mask.

  assert(Op.getOperand(EVLOpNo).getValueType() == MVT::i32 &&
         "Unexpected operand");
  Operands.push_back(DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64,
                                 Op.getOperand(EVLOpNo))); // EVL.

  return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, Op.getValueType(), Operands);
}

SDValue RISCVTargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                     SelectionDAG &DAG) const {
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  SDLoc DL(Op);

  if (Subtarget.hasStdExtV()) {
    // Some EPI intrinsics may claim that they want an integer operand to be
    // extended.
    if (const RISCVEPIIntrinsicsTable::EPIIntrinsicInfo *EII =
            RISCVEPIIntrinsicsTable::getEPIIntrinsicInfo(IntNo)) {
      if (EII->ExtendedOperand) {
        assert(EII->ExtendedOperand < Op.getNumOperands());
        std::vector<SDValue> Operands(Op->op_begin(), Op->op_end());
        SDValue &ScalarOp = Operands[EII->ExtendedOperand];
        if (ScalarOp.getValueType() == MVT::i32 ||
            ScalarOp.getValueType() == MVT::i16 ||
            ScalarOp.getValueType() == MVT::i8) {
          ScalarOp = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, ScalarOp);
          return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, Op.getValueType(),
                             Operands);
        }
      }
    }
  }

  switch (IntNo) {
  default:
    return SDValue();    // Don't custom lower most intrinsics.
  case Intrinsic::thread_pointer: {
    EVT PtrVT = getPointerTy(DAG.getDataLayout());
    return DAG.getRegister(RISCV::X4, PtrVT);
  }
  case Intrinsic::epi_vzip2:
  case Intrinsic::epi_vunzip2:
  case Intrinsic::epi_vtrn: {
    SDVTList VTList = Op->getVTList();
    assert(VTList.NumVTs == 2);
    EVT VT = VTList.VTs[0];

    unsigned TupleOpcode;
    switch (IntNo) {
    default:
      llvm_unreachable("Invalid opcode");
      break;
    case Intrinsic::epi_vzip2:
      TupleOpcode = RISCVISD::VZIP2;
      break;
    case Intrinsic::epi_vunzip2:
      TupleOpcode = RISCVISD::VUNZIP2;
      break;
    case Intrinsic::epi_vtrn:
      TupleOpcode = RISCVISD::VTRN;
      break;
    }

    SDValue TupleNode =
        DAG.getNode(TupleOpcode, DL, MVT::Untyped, Op->getOperand(1),
                    Op->getOperand(2), Op->getOperand(3));
    SDValue SubRegFirst = DAG.getTargetConstant(RISCV::vtuple2_0, DL, MVT::i32);
    MachineSDNode *FirstNode = DAG.getMachineNode(
        TargetOpcode::EXTRACT_SUBREG, DL, VT, TupleNode, SubRegFirst);

    SDValue SubRegSecond = DAG.getTargetConstant(RISCV::vtuple2_1, DL, MVT::i32);
    MachineSDNode *SecondNode = DAG.getMachineNode(
        TargetOpcode::EXTRACT_SUBREG, DL, VT, TupleNode, SubRegSecond);

    SDValue ExtractedOps[] = {SDValue(FirstNode, 0), SDValue(SecondNode, 0)};
    return DAG.getNode(ISD::MERGE_VALUES, DL, VTList, ExtractedOps);
  }
  case Intrinsic::vp_add:
  case Intrinsic::vp_sub:
  case Intrinsic::vp_mul:
  case Intrinsic::vp_sdiv:
  case Intrinsic::vp_srem:
  case Intrinsic::vp_udiv:
  case Intrinsic::vp_urem:
  case Intrinsic::vp_and:
  case Intrinsic::vp_or:
  case Intrinsic::vp_xor:
  case Intrinsic::vp_ashr:
  case Intrinsic::vp_lshr:
  case Intrinsic::vp_shl:
  case Intrinsic::vp_fadd:
  case Intrinsic::vp_fsub:
  case Intrinsic::vp_fmul:
  case Intrinsic::vp_fdiv:
  case Intrinsic::vp_frem:
  case Intrinsic::vp_fma:
  case Intrinsic::vp_fneg:
  case Intrinsic::vp_icmp:
  case Intrinsic::vp_fcmp:
  case Intrinsic::vp_select:
  case Intrinsic::vp_bitcast:
    return LowerVPINTRINSIC_WO_CHAIN(Op, DAG);
  case Intrinsic::vp_sitofp:
  case Intrinsic::vp_uitofp:
  case Intrinsic::vp_fptosi:
  case Intrinsic::vp_fptoui:
  case Intrinsic::vp_fpext:
  case Intrinsic::vp_fptrunc:
  case Intrinsic::vp_trunc:
  case Intrinsic::vp_zext:
  case Intrinsic::vp_sext:
    return LowerVPIntrinsicConversion(Op, DAG);
  }
}

// FIXME: This does not handle fractional LMUL.
static std::pair<int64_t, int64_t> getSewLMul(MVT VT) {
  switch (VT.SimpleTy) {
  default:
    llvm_unreachable("Unexpected type");
    // LMUL=1
  case MVT::nxv1i64:
  case MVT::nxv1f64:
    return {64, 1};
  case MVT::nxv2i32:
  case MVT::nxv2f32:
    return {32, 1};
  case MVT::nxv4i16:
  case MVT::nxv4f16:
    return {16, 1};
  case MVT::nxv8i8:
    return {8, 1};
    // LMUL=2
  case MVT::nxv2i64:
  case MVT::nxv2f64:
    return {64, 2};
  case MVT::nxv4i32:
  case MVT::nxv4f32:
    return {32, 2};
  case MVT::nxv8i16:
  case MVT::nxv8f16:
    return {16, 2};
  case MVT::nxv16i8:
    return {8, 2};
    // LMUL=4
  case MVT::nxv4i64:
  case MVT::nxv4f64:
    return {64, 4};
  case MVT::nxv8i32:
  case MVT::nxv8f32:
    return {32, 4};
  case MVT::nxv16i16:
  case MVT::nxv16f16:
    return {16, 4};
  case MVT::nxv32i8:
    return {8, 4};
    // LMUL=8
  case MVT::nxv8i64:
  case MVT::nxv8f64:
    return {64, 8};
  case MVT::nxv16i32:
  case MVT::nxv16f32:
    return {32, 8};
  case MVT::nxv32i16:
  case MVT::nxv32f16:
    return {16, 8};
  case MVT::nxv64i8:
    return {8, 8};
  }
}

// Decomposes a vector of addresses into a base address plus a vector of
// offsets.
static void GetBaseAddressAndOffsets(const SDValue &Addresses, EVT OffsetsVT,
                                     SDLoc DL, SelectionDAG &DAG,
                                     SDValue &BaseAddr, SDValue &Offsets) {
  unsigned Opcode = Addresses.getOpcode();
  if (Opcode == ISD::SPLAT_VECTOR) {
    // Addresses is a splat vector. Set the BaseAddr as the splatted value
    // and the offsets to zero.
    BaseAddr = Addresses.getOperand(0);
    Offsets =
        DAG.getNode(ISD::SPLAT_VECTOR, DL, OffsetsVT,
                    DAG.getConstant(0, DL, OffsetsVT.getVectorElementType()));
    return;
  }

  if (Opcode == ISD::ADD) {
    // Addresses is either (add a, (splat b)) or (add (splat a), b). Compute the
    // base address as int the previous case and use the addend as offsets.
    SDValue Op0 = Addresses.getOperand(0);
    SDValue Op1 = Addresses.getOperand(1);
    if (Op0.getOpcode() == ISD::SPLAT_VECTOR ||
        Op1.getOpcode() == ISD::SPLAT_VECTOR) {
      if (Op0.getOpcode() == ISD::SPLAT_VECTOR) {
        BaseAddr = Op0.getOperand(0);
        Offsets = Op1;
      } else {
        BaseAddr = Op1.getOperand(0);
        Offsets = Op0;
      }
      assert(OffsetsVT == Offsets.getValueType() &&
             "Unexpected type for the offsets vector");
      return;
    }
  }

  // Fallback to setting the base address to zero and the offsets to the
  // Addresses vector.
  assert(OffsetsVT == Addresses.getValueType() &&
         "Unexpected type for the offsets vector");
  BaseAddr = DAG.getConstant(0, DL, MVT::i64);
  Offsets = Addresses;
}

static SDValue lowerVLSEG(SDValue Op, SelectionDAG &DAG,
                          const RISCVSubtarget &Subtarget, unsigned OpCode,
                          const ArrayRef<unsigned> SubRegisters) {
  SDLoc DL(Op);
  SDVTList VTList = Op->getVTList();
  SDVTList VTs = DAG.getVTList(MVT::Untyped, MVT::Other);
  EVT VT = VTList.VTs[0];

  int64_t LMUL;
  int64_t SEWBits;
  std::tie(SEWBits, LMUL) = getSewLMul(VT.getSimpleVT());

  MVT XLenVT = Subtarget.getXLenVT();
  SDValue SEW = DAG.getTargetConstant(SEWBits, DL, XLenVT);

  SmallVector<SDValue, 4> Operands;
  for (unsigned I = 0, E = Op->getNumOperands(); I != E; I++) {
    // Skip Intrinsic ID
    if (I == 1)
      continue;
    Operands.push_back(Op->getOperand(I));
  }
  Operands.push_back(SEW);

  SDValue TupleNode = DAG.getNode(OpCode, DL, VTs, Operands);

  SmallVector<SDValue, 4> ExtractedOps;
  for (unsigned SubReg : SubRegisters) {
    SDValue SubRegIdx = DAG.getTargetConstant(SubReg, DL, MVT::i32);
    MachineSDNode *SubRegNode = DAG.getMachineNode(
        TargetOpcode::EXTRACT_SUBREG, DL, VT, TupleNode, SubRegIdx);
    ExtractedOps.push_back(SDValue(SubRegNode, 0));
  }

  ExtractedOps.push_back(SDValue(TupleNode.getNode(), 1));
  return DAG.getNode(ISD::MERGE_VALUES, DL, VTList, ExtractedOps);
}

SDValue RISCVTargetLowering::LowerINTRINSIC_W_CHAIN(SDValue Op,
                                                    SelectionDAG &DAG) const {
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  SDLoc DL(Op);
  switch (IntNo) {
    // By default we do not lower any intrinsic.
  default:
    break;
  case Intrinsic::vp_load: {
    assert(Op.getOperand(5).getValueType() == MVT::i32 && "Unexpected operand");

    std::vector<SDValue> Operands;
    const SDValue &MaskOp = Op.getOperand(4);
    ConstantSDNode *C;
    if (MaskOp.getOpcode() == ISD::SPLAT_VECTOR &&
        (C = dyn_cast<ConstantSDNode>(MaskOp.getOperand(0))) &&
        C->getZExtValue() == 1)
      // Unmasked.
      Operands = {
          Op.getOperand(0),                                          // Chain.
          DAG.getTargetConstant(Intrinsic::epi_vload, DL, MVT::i64),
          Op.getOperand(2),                                          // Address.
          // FIXME Alignment ignored.
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64,
                      Op.getOperand(5))                              // EVL.
      };
    else
      Operands = {
          Op.getOperand(0),                                    // Chain.
          DAG.getTargetConstant(Intrinsic::epi_vload_mask, DL,
                                MVT::i64),
          DAG.getNode(ISD::UNDEF, DL, Op.getValueType()),      // Merge.
          Op.getOperand(2),                                    // Address.
          // FIXME Alignment ignored.
          Op.getOperand(4),                                    // Mask.
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64,
                      Op.getOperand(5))                        // EVL.
      };

    SDValue Result =
        DAG.getNode(ISD::INTRINSIC_W_CHAIN, DL, Op->getVTList(), Operands);
    return DAG.getMergeValues({Result, Result.getValue(1)}, DL);
  }
  case Intrinsic::vp_gather: {
    EVT VT = Op.getValueType();
    EVT OffsetsVT = VT.changeVectorElementTypeToInteger();

    SDValue BaseAddr;
    SDValue Offsets;
    SDValue Addresses = Op.getOperand(2);
    GetBaseAddressAndOffsets(Addresses, OffsetsVT, DL, DAG, BaseAddr, Offsets);

    SDValue VL = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op.getOperand(5));

    // FIXME Address alignment operand (3) ignored.
    SDValue VLXEOperands[] = {
        Op.getOperand(0), // Chain.
        DAG.getTargetConstant(Intrinsic::epi_vload_indexed_mask, DL, MVT::i64),
        DAG.getNode(ISD::UNDEF, DL, VT), // Merge.
        BaseAddr,
        Offsets,
        Op.getOperand(4), // Mask.
        VL
    };
    SDValue Result =
        DAG.getNode(ISD::INTRINSIC_W_CHAIN, DL, Op->getVTList(), VLXEOperands);

    return Result;
  }
  case Intrinsic::epi_vlseg2:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLSEG2,
                      {RISCV::vtuple2_0, RISCV::vtuple2_1});
  case Intrinsic::epi_vlseg3:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLSEG3,
                      {RISCV::vtuple3_0, RISCV::vtuple3_1, RISCV::vtuple3_2});
  case Intrinsic::epi_vlseg4:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLSEG4,
                      {RISCV::vtuple4_0, RISCV::vtuple4_1, RISCV::vtuple4_2,
                       RISCV::vtuple4_3});
  case Intrinsic::epi_vlseg5:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLSEG5,
                      {RISCV::vtuple5_0, RISCV::vtuple5_1, RISCV::vtuple5_2,
                       RISCV::vtuple5_3, RISCV::vtuple5_4});
  case Intrinsic::epi_vlseg6:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLSEG6,
                      {RISCV::vtuple6_0, RISCV::vtuple6_1, RISCV::vtuple6_2,
                       RISCV::vtuple6_3, RISCV::vtuple6_4, RISCV::vtuple6_5});
  case Intrinsic::epi_vlseg7:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLSEG7,
                      {RISCV::vtuple7_0, RISCV::vtuple7_1, RISCV::vtuple7_2,
                       RISCV::vtuple7_3, RISCV::vtuple7_4, RISCV::vtuple7_5,
                       RISCV::vtuple7_6});
  case Intrinsic::epi_vlseg8:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLSEG8,
                      {RISCV::vtuple8_0, RISCV::vtuple8_1, RISCV::vtuple8_2,
                       RISCV::vtuple8_3, RISCV::vtuple8_4, RISCV::vtuple8_5,
                       RISCV::vtuple8_6, RISCV::vtuple8_7});
  case Intrinsic::epi_vlseg2_strided:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLSSEG2,
                      {RISCV::vtuple2_0, RISCV::vtuple2_1});
  case Intrinsic::epi_vlseg3_strided:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLSSEG3,
                      {RISCV::vtuple3_0, RISCV::vtuple3_1, RISCV::vtuple3_2});
  case Intrinsic::epi_vlseg4_strided:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLSSEG4,
                      {RISCV::vtuple4_0, RISCV::vtuple4_1, RISCV::vtuple4_2,
                       RISCV::vtuple4_3});
  case Intrinsic::epi_vlseg5_strided:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLSSEG5,
                      {RISCV::vtuple5_0, RISCV::vtuple5_1, RISCV::vtuple5_2,
                       RISCV::vtuple5_3, RISCV::vtuple5_4});
  case Intrinsic::epi_vlseg6_strided:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLSSEG6,
                      {RISCV::vtuple6_0, RISCV::vtuple6_1, RISCV::vtuple6_2,
                       RISCV::vtuple6_3, RISCV::vtuple6_4, RISCV::vtuple6_5});
  case Intrinsic::epi_vlseg7_strided:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLSSEG7,
                      {RISCV::vtuple7_0, RISCV::vtuple7_1, RISCV::vtuple7_2,
                       RISCV::vtuple7_3, RISCV::vtuple7_4, RISCV::vtuple7_5,
                       RISCV::vtuple7_6});
  case Intrinsic::epi_vlseg8_strided:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLSSEG8,
                      {RISCV::vtuple8_0, RISCV::vtuple8_1, RISCV::vtuple8_2,
                       RISCV::vtuple8_3, RISCV::vtuple8_4, RISCV::vtuple8_5,
                       RISCV::vtuple8_6, RISCV::vtuple8_7});
  case Intrinsic::epi_vlseg2_indexed:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLXSEG2,
                      {RISCV::vtuple2_0, RISCV::vtuple2_1});
  case Intrinsic::epi_vlseg3_indexed:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLXSEG3,
                      {RISCV::vtuple3_0, RISCV::vtuple3_1, RISCV::vtuple3_2});
  case Intrinsic::epi_vlseg4_indexed:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLXSEG4,
                      {RISCV::vtuple4_0, RISCV::vtuple4_1, RISCV::vtuple4_2,
                       RISCV::vtuple4_3});
  case Intrinsic::epi_vlseg5_indexed:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLXSEG5,
                      {RISCV::vtuple5_0, RISCV::vtuple5_1, RISCV::vtuple5_2,
                       RISCV::vtuple5_3, RISCV::vtuple5_4});
  case Intrinsic::epi_vlseg6_indexed:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLXSEG6,
                      {RISCV::vtuple6_0, RISCV::vtuple6_1, RISCV::vtuple6_2,
                       RISCV::vtuple6_3, RISCV::vtuple6_4, RISCV::vtuple6_5});
  case Intrinsic::epi_vlseg7_indexed:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLXSEG7,
                      {RISCV::vtuple7_0, RISCV::vtuple7_1, RISCV::vtuple7_2,
                       RISCV::vtuple7_3, RISCV::vtuple7_4, RISCV::vtuple7_5,
                       RISCV::vtuple7_6});
  case Intrinsic::epi_vlseg8_indexed:
    return lowerVLSEG(Op, DAG, Subtarget, RISCVISD::VLXSEG8,
                      {RISCV::vtuple8_0, RISCV::vtuple8_1, RISCV::vtuple8_2,
                       RISCV::vtuple8_3, RISCV::vtuple8_4, RISCV::vtuple8_5,
                       RISCV::vtuple8_6, RISCV::vtuple8_7});
  }

  return SDValue();
}

static SDValue lowerVSSEG(SDValue Op, SelectionDAG &DAG,
                          const RISCVSubtarget &Subtarget, unsigned TupleSize,
                          unsigned Opcode, unsigned BuildOpcode) {
  SDLoc DL(Op);
  EVT VT = Op->getOperand(2).getValueType();
  int64_t LMUL;
  int64_t SEWBits;
  std::tie(SEWBits, LMUL) = getSewLMul(VT.getSimpleVT());

  MVT XLenVT = Subtarget.getXLenVT();
  SDValue SEW = DAG.getTargetConstant(SEWBits, DL, XLenVT);

  // Because the type is MVT:Untyped we can't actually use INSERT_SUBREG
  // so we use a pseudo instruction that we will expand later into proper
  // INSERT_SUBREGs using the right register class.
  SmallVector<SDValue, 4> TupleOperands;
  for (unsigned I = 0; I < TupleSize; I++) {
    TupleOperands.push_back(Op->getOperand(2 + I));
  }

  MachineSDNode *Tuple =
      DAG.getMachineNode(BuildOpcode, DL, MVT::Untyped, TupleOperands);

  SmallVector<SDValue, 4> Operands;
  Operands.push_back(/* Chain */ Op->getOperand(0));
  Operands.push_back(SDValue(Tuple, 0));
  for (unsigned I = 2 + TupleSize, E = Op.getNumOperands(); I != E; I++) {
    Operands.push_back(Op->getOperand(I));
  }
  Operands.push_back(SEW);

  SDVTList VTs = DAG.getVTList(MVT::Other);
  return DAG.getNode(Opcode, DL, VTs, Operands);
}

SDValue RISCVTargetLowering::LowerINTRINSIC_VOID(SDValue Op,
                                                 SelectionDAG &DAG) const {
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  SDLoc DL(Op);
  switch (IntNo) {
    // By default we do not lower any intrinsic.
  default:
    break;
  case Intrinsic::vp_store: {
    assert(Op.getOperand(6).getValueType() == MVT::i32 && "Unexpected operand");

    std::vector<SDValue> Operands;
    const SDValue &MaskOp = Op.getOperand(5);
    ConstantSDNode *C;
    if (MaskOp.getOpcode() == ISD::SPLAT_VECTOR &&
        (C = dyn_cast<ConstantSDNode>(MaskOp.getOperand(0))) &&
        C->getZExtValue() == 1)
      // Unmasked.
      Operands = {
          Op.getOperand(0),                                     // Chain.
          DAG.getTargetConstant(Intrinsic::epi_vstore, DL,
                                MVT::i64),
          Op.getOperand(2),                                     // Value.
          Op.getOperand(3),                                     // Address.
          // FIXME Alignment ignored.
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64,
                      Op.getOperand(6)),                        // EVL.
      };
    else
      Operands = {
          Op.getOperand(0),                                     // Chain.
          DAG.getTargetConstant(Intrinsic::epi_vstore_mask, DL,
                                MVT::i64),
          Op.getOperand(2),                                     // Value.
          Op.getOperand(3),                                     // Address.
          // FIXME Alignment ignored.
          MaskOp,                                               // Mask.
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64,
                      Op.getOperand(6)),                        // EVL.
      };

    return DAG.getNode(ISD::INTRINSIC_VOID, DL, Op->getVTList(), Operands);
    break;
  }
  case Intrinsic::epi_vsseg2:
    return lowerVSSEG(Op, DAG, Subtarget, 2, RISCVISD::VSSEG2,
                      RISCV::PseudoVBuildVRM1T2);
  case Intrinsic::epi_vsseg3:
    return lowerVSSEG(Op, DAG, Subtarget, 3, RISCVISD::VSSEG3,
                      RISCV::PseudoVBuildVRM1T3);
  case Intrinsic::epi_vsseg4:
    return lowerVSSEG(Op, DAG, Subtarget, 4, RISCVISD::VSSEG4,
                      RISCV::PseudoVBuildVRM1T4);
  case Intrinsic::epi_vsseg5:
    return lowerVSSEG(Op, DAG, Subtarget, 5, RISCVISD::VSSEG5,
                      RISCV::PseudoVBuildVRM1T5);
  case Intrinsic::epi_vsseg6:
    return lowerVSSEG(Op, DAG, Subtarget, 6, RISCVISD::VSSEG6,
                      RISCV::PseudoVBuildVRM1T6);
  case Intrinsic::epi_vsseg7:
    return lowerVSSEG(Op, DAG, Subtarget, 7, RISCVISD::VSSEG7,
                      RISCV::PseudoVBuildVRM1T7);
  case Intrinsic::epi_vsseg8:
    return lowerVSSEG(Op, DAG, Subtarget, 8, RISCVISD::VSSEG8,
                      RISCV::PseudoVBuildVRM1T8);
  case Intrinsic::epi_vsseg2_strided:
    return lowerVSSEG(Op, DAG, Subtarget, 2, RISCVISD::VSSSEG2,
                      RISCV::PseudoVBuildVRM1T2);
  case Intrinsic::epi_vsseg3_strided:
    return lowerVSSEG(Op, DAG, Subtarget, 3, RISCVISD::VSSSEG3,
                      RISCV::PseudoVBuildVRM1T3);
  case Intrinsic::epi_vsseg4_strided:
    return lowerVSSEG(Op, DAG, Subtarget, 4, RISCVISD::VSSSEG4,
                      RISCV::PseudoVBuildVRM1T4);
  case Intrinsic::epi_vsseg5_strided:
    return lowerVSSEG(Op, DAG, Subtarget, 5, RISCVISD::VSSSEG5,
                      RISCV::PseudoVBuildVRM1T5);
  case Intrinsic::epi_vsseg6_strided:
    return lowerVSSEG(Op, DAG, Subtarget, 6, RISCVISD::VSSSEG6,
                      RISCV::PseudoVBuildVRM1T6);
  case Intrinsic::epi_vsseg7_strided:
    return lowerVSSEG(Op, DAG, Subtarget, 7, RISCVISD::VSSSEG7,
                      RISCV::PseudoVBuildVRM1T7);
  case Intrinsic::epi_vsseg8_strided:
    return lowerVSSEG(Op, DAG, Subtarget, 8, RISCVISD::VSSSEG8,
                      RISCV::PseudoVBuildVRM1T8);
  case Intrinsic::epi_vsseg2_indexed:
    return lowerVSSEG(Op, DAG, Subtarget, 2, RISCVISD::VSXSEG2,
                      RISCV::PseudoVBuildVRM1T2);
  case Intrinsic::epi_vsseg3_indexed:
    return lowerVSSEG(Op, DAG, Subtarget, 3, RISCVISD::VSXSEG3,
                      RISCV::PseudoVBuildVRM1T3);
  case Intrinsic::epi_vsseg4_indexed:
    return lowerVSSEG(Op, DAG, Subtarget, 4, RISCVISD::VSXSEG4,
                      RISCV::PseudoVBuildVRM1T4);
  case Intrinsic::epi_vsseg5_indexed:
    return lowerVSSEG(Op, DAG, Subtarget, 5, RISCVISD::VSXSEG5,
                      RISCV::PseudoVBuildVRM1T5);
  case Intrinsic::epi_vsseg6_indexed:
    return lowerVSSEG(Op, DAG, Subtarget, 6, RISCVISD::VSXSEG6,
                      RISCV::PseudoVBuildVRM1T6);
  case Intrinsic::epi_vsseg7_indexed:
    return lowerVSSEG(Op, DAG, Subtarget, 7, RISCVISD::VSXSEG7,
                      RISCV::PseudoVBuildVRM1T7);
  case Intrinsic::epi_vsseg8_indexed:
    return lowerVSSEG(Op, DAG, Subtarget, 8, RISCVISD::VSXSEG8,
                      RISCV::PseudoVBuildVRM1T8);
  case Intrinsic::vp_scatter: {
    SDValue Data = Op.getOperand(2);
    EVT OffsetsVT = Data.getValueType().changeVectorElementTypeToInteger();

    SDValue BaseAddr;
    SDValue Offsets;
    SDValue Addresses = Op.getOperand(3);
    GetBaseAddressAndOffsets(Addresses, OffsetsVT, DL, DAG, BaseAddr, Offsets);

    SDValue VL = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op.getOperand(6));

    // FIXME Address alignment operand (4) ignored.
    SDValue VSXEOperands[] = {
        Op.getOperand(0), // Chain.
        DAG.getTargetConstant(Intrinsic::epi_vstore_indexed_mask, DL, MVT::i64),
        Data,
        BaseAddr,
        Offsets,
        Op.getOperand(5), // Mask.
        VL
    };
    SDValue Result =
        DAG.getNode(ISD::INTRINSIC_VOID, DL, Op->getVTList(), VSXEOperands);

    return Result;
  }
  }

  return SDValue();
}

// Returns the opcode of the target-specific SDNode that implements the 32-bit
// form of the given Opcode.
static RISCVISD::NodeType getRISCVWOpcode(unsigned Opcode) {
  switch (Opcode) {
  default:
    llvm_unreachable("Unexpected opcode");
  case ISD::SHL:
    return RISCVISD::SLLW;
  case ISD::SRA:
    return RISCVISD::SRAW;
  case ISD::SRL:
    return RISCVISD::SRLW;
  case ISD::SDIV:
    return RISCVISD::DIVW;
  case ISD::UDIV:
    return RISCVISD::DIVUW;
  case ISD::UREM:
    return RISCVISD::REMUW;
  }
}

// Converts the given 32-bit operation to a target-specific SelectionDAG node.
// Because i32 isn't a legal type for RV64, these operations would otherwise
// be promoted to i64, making it difficult to select the SLLW/DIVUW/.../*W
// later one because the fact the operation was originally of type i32 is
// lost.
static SDValue customLegalizeToWOp(SDNode *N, SelectionDAG &DAG) {
  SDLoc DL(N);
  RISCVISD::NodeType WOpcode = getRISCVWOpcode(N->getOpcode());
  SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));
  SDValue NewOp1 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
  SDValue NewRes = DAG.getNode(WOpcode, DL, MVT::i64, NewOp0, NewOp1);
  // ReplaceNodeResults requires we maintain the same type for the return value.
  return DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, NewRes);
}

// Converts the given 32-bit operation to a i64 operation with signed extension
// semantic to reduce the signed extension instructions.
static SDValue customLegalizeToWOpWithSExt(SDNode *N, SelectionDAG &DAG) {
  SDLoc DL(N);
  SDValue NewOp0 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(0));
  SDValue NewOp1 = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, N->getOperand(1));
  SDValue NewWOp = DAG.getNode(N->getOpcode(), DL, MVT::i64, NewOp0, NewOp1);
  SDValue NewRes = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i64, NewWOp,
                               DAG.getValueType(MVT::i32));
  return DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, NewRes);
}

void RISCVTargetLowering::ReplaceNodeResults(SDNode *N,
                                             SmallVectorImpl<SDValue> &Results,
                                             SelectionDAG &DAG) const {
  SDLoc DL(N);
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Don't know how to custom type legalize this operation!");
  case ISD::STRICT_FP_TO_SINT:
  case ISD::STRICT_FP_TO_UINT:
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT: {
    bool IsStrict = N->isStrictFPOpcode();
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    SDValue Op0 = IsStrict ? N->getOperand(1) : N->getOperand(0);
    RTLIB::Libcall LC;
    if (N->getOpcode() == ISD::FP_TO_SINT ||
        N->getOpcode() == ISD::STRICT_FP_TO_SINT)
      LC = RTLIB::getFPTOSINT(Op0.getValueType(), N->getValueType(0));
    else
      LC = RTLIB::getFPTOUINT(Op0.getValueType(), N->getValueType(0));
    MakeLibCallOptions CallOptions;
    EVT OpVT = Op0.getValueType();
    CallOptions.setTypeListBeforeSoften(OpVT, N->getValueType(0), true);
    SDValue Chain = IsStrict ? N->getOperand(0) : SDValue();
    SDValue Result;
    std::tie(Result, Chain) =
        makeLibCall(DAG, LC, N->getValueType(0), Op0, CallOptions, DL, Chain);
    Results.push_back(Result);
    if (IsStrict)
      Results.push_back(Chain);
    break;
  }
  case ISD::READCYCLECOUNTER: {
    assert(!Subtarget.is64Bit() &&
           "READCYCLECOUNTER only has custom type legalization on riscv32");

    SDVTList VTs = DAG.getVTList(MVT::i32, MVT::i32, MVT::Other);
    SDValue RCW =
        DAG.getNode(RISCVISD::READ_CYCLE_WIDE, DL, VTs, N->getOperand(0));

    Results.push_back(
        DAG.getNode(ISD::BUILD_PAIR, DL, MVT::i64, RCW, RCW.getValue(1)));
    Results.push_back(RCW.getValue(2));
    break;
  }
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    if (N->getOperand(1).getOpcode() == ISD::Constant)
      return;
    Results.push_back(customLegalizeToWOpWithSExt(N, DAG));
    break;
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           "Unexpected custom legalisation");
    if (N->getOperand(1).getOpcode() == ISD::Constant)
      return;
    Results.push_back(customLegalizeToWOp(N, DAG));
    break;
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::UREM:
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           Subtarget.hasStdExtM() && "Unexpected custom legalisation");
    if (N->getOperand(0).getOpcode() == ISD::Constant ||
        N->getOperand(1).getOpcode() == ISD::Constant)
      return;
    Results.push_back(customLegalizeToWOp(N, DAG));
    break;
  case ISD::BITCAST: {
    assert(N->getValueType(0) == MVT::i32 && Subtarget.is64Bit() &&
           Subtarget.hasStdExtF() && "Unexpected custom legalisation");
    SDValue Op0 = N->getOperand(0);
    if (Op0.getValueType() != MVT::f32)
      return;
    SDValue FPConv =
        DAG.getNode(RISCVISD::FMV_X_ANYEXTW_RV64, DL, MVT::i64, Op0);
    Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, FPConv));
    break;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntNo = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
    switch (IntNo) {
    default:
      llvm_unreachable(
          "Don't know how to custom type legalize this intrinsic!");
    case Intrinsic::vscale: {
      EVT Ty = N->getValueType(0);
      switch (Ty.getSimpleVT().SimpleTy) {
      default:
        llvm_unreachable("Unexpected result type to legalize");
      case MVT::i32:
      case MVT::i16:
      case MVT::i8:
        SDValue Promoted = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i64,
                                       DAG.getConstant(IntNo, DL, MVT::i64));
        SDValue Trunc = DAG.getNode(ISD::TRUNCATE, DL, Ty, Promoted);
        Results.push_back(Trunc);
        break;
      }
      break;
    }
    case Intrinsic::epi_vmv_x_s: {
      EVT Ty = N->getValueType(0);
      MVT::SimpleValueType SimpleVT = Ty.getSimpleVT().SimpleTy;
      assert(SimpleVT == MVT::i8 || SimpleVT == MVT::i16 ||
             SimpleVT == MVT::i32);

      SDValue Extract64 =
          DAG.getNode(RISCVISD::VMV_X_S, DL, MVT::i64, N->getOperand(1));
      SDValue Trunc = DAG.getNode(ISD::TRUNCATE, DL, Ty, Extract64);
      Results.push_back(Trunc);

      break;
    }
    case Intrinsic::experimental_vector_stepvector: {
      EVT Ty = N->getValueType(0);
      switch (Ty.getSimpleVT().SimpleTy) {
      default:
        llvm_unreachable("Unexpected result type to legalize");
      case MVT::nxv1i32:
        SDValue Promoted =
            DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::nxv1i64,
                        DAG.getConstant(IntNo, DL, MVT::i64));
        SDValue Trunc = DAG.getNode(ISD::TRUNCATE, DL, Ty, Promoted);
        Results.push_back(Trunc);
        break;
      }
      break;
    }
    }
    break;
  }
  case ISD::EXTRACT_VECTOR_ELT: {
    EVT Ty = N->getValueType(0);
    MVT::SimpleValueType SimpleVT = Ty.getSimpleVT().SimpleTy;
    assert(SimpleVT == MVT::i8 || SimpleVT == MVT::i16 || SimpleVT == MVT::i32);

    SDValue Extract64 = DAG.getNode(RISCVISD::EXTRACT_VECTOR_ELT, DL, MVT::i64,
                                    N->getOperand(0), N->getOperand(1));
    SDValue Trunc = DAG.getNode(ISD::TRUNCATE, DL, Ty, Extract64);
    Results.push_back(Trunc);

    break;
  }
  }
}

SDValue RISCVTargetLowering::PerformDAGCombine(SDNode *N,
                                               DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;

  switch (N->getOpcode()) {
  default:
    break;
  case RISCVISD::SplitF64: {
    SDValue Op0 = N->getOperand(0);
    // If the input to SplitF64 is just BuildPairF64 then the operation is
    // redundant. Instead, use BuildPairF64's operands directly.
    if (Op0->getOpcode() == RISCVISD::BuildPairF64)
      return DCI.CombineTo(N, Op0.getOperand(0), Op0.getOperand(1));

    SDLoc DL(N);

    // It's cheaper to materialise two 32-bit integers than to load a double
    // from the constant pool and transfer it to integer registers through the
    // stack.
    if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Op0)) {
      APInt V = C->getValueAPF().bitcastToAPInt();
      SDValue Lo = DAG.getConstant(V.trunc(32), DL, MVT::i32);
      SDValue Hi = DAG.getConstant(V.lshr(32).trunc(32), DL, MVT::i32);
      return DCI.CombineTo(N, Lo, Hi);
    }

    // This is a target-specific version of a DAGCombine performed in
    // DAGCombiner::visitBITCAST. It performs the equivalent of:
    // fold (bitconvert (fneg x)) -> (xor (bitconvert x), signbit)
    // fold (bitconvert (fabs x)) -> (and (bitconvert x), (not signbit))
    if (!(Op0.getOpcode() == ISD::FNEG || Op0.getOpcode() == ISD::FABS) ||
        !Op0.getNode()->hasOneUse())
      break;
    SDValue NewSplitF64 =
        DAG.getNode(RISCVISD::SplitF64, DL, DAG.getVTList(MVT::i32, MVT::i32),
                    Op0.getOperand(0));
    SDValue Lo = NewSplitF64.getValue(0);
    SDValue Hi = NewSplitF64.getValue(1);
    APInt SignBit = APInt::getSignMask(32);
    if (Op0.getOpcode() == ISD::FNEG) {
      SDValue NewHi = DAG.getNode(ISD::XOR, DL, MVT::i32, Hi,
                                  DAG.getConstant(SignBit, DL, MVT::i32));
      return DCI.CombineTo(N, Lo, NewHi);
    }
    assert(Op0.getOpcode() == ISD::FABS);
    SDValue NewHi = DAG.getNode(ISD::AND, DL, MVT::i32, Hi,
                                DAG.getConstant(~SignBit, DL, MVT::i32));
    return DCI.CombineTo(N, Lo, NewHi);
  }
  case RISCVISD::SLLW:
  case RISCVISD::SRAW:
  case RISCVISD::SRLW: {
    // Only the lower 32 bits of LHS and lower 5 bits of RHS are read.
    SDValue LHS = N->getOperand(0);
    SDValue RHS = N->getOperand(1);
    APInt LHSMask = APInt::getLowBitsSet(LHS.getValueSizeInBits(), 32);
    APInt RHSMask = APInt::getLowBitsSet(RHS.getValueSizeInBits(), 5);
    if ((SimplifyDemandedBits(N->getOperand(0), LHSMask, DCI)) ||
        (SimplifyDemandedBits(N->getOperand(1), RHSMask, DCI)))
      return SDValue();
    break;
  }
  case RISCVISD::FMV_X_ANYEXTW_RV64: {
    SDLoc DL(N);
    SDValue Op0 = N->getOperand(0);
    // If the input to FMV_X_ANYEXTW_RV64 is just FMV_W_X_RV64 then the
    // conversion is unnecessary and can be replaced with an ANY_EXTEND
    // of the FMV_W_X_RV64 operand.
    if (Op0->getOpcode() == RISCVISD::FMV_W_X_RV64) {
      SDValue AExtOp =
          DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, Op0.getOperand(0));
      return DCI.CombineTo(N, AExtOp);
    }

    // This is a target-specific version of a DAGCombine performed in
    // DAGCombiner::visitBITCAST. It performs the equivalent of:
    // fold (bitconvert (fneg x)) -> (xor (bitconvert x), signbit)
    // fold (bitconvert (fabs x)) -> (and (bitconvert x), (not signbit))
    if (!(Op0.getOpcode() == ISD::FNEG || Op0.getOpcode() == ISD::FABS) ||
        !Op0.getNode()->hasOneUse())
      break;
    SDValue NewFMV = DAG.getNode(RISCVISD::FMV_X_ANYEXTW_RV64, DL, MVT::i64,
                                 Op0.getOperand(0));
    APInt SignBit = APInt::getSignMask(32).sext(64);
    if (Op0.getOpcode() == ISD::FNEG) {
      return DCI.CombineTo(N,
                           DAG.getNode(ISD::XOR, DL, MVT::i64, NewFMV,
                                       DAG.getConstant(SignBit, DL, MVT::i64)));
    }
    assert(Op0.getOpcode() == ISD::FABS);
    return DCI.CombineTo(N,
                         DAG.getNode(ISD::AND, DL, MVT::i64, NewFMV,
                                     DAG.getConstant(~SignBit, DL, MVT::i64)));
  }
  }

  return SDValue();
}

bool RISCVTargetLowering::isDesirableToCommuteWithShift(
    const SDNode *N, CombineLevel Level) const {
  // The following folds are only desirable if `(OP _, c1 << c2)` can be
  // materialised in fewer instructions than `(OP _, c1)`:
  //
  //   (shl (add x, c1), c2) -> (add (shl x, c2), c1 << c2)
  //   (shl (or x, c1), c2) -> (or (shl x, c2), c1 << c2)
  SDValue N0 = N->getOperand(0);
  EVT Ty = N0.getValueType();
  if (Ty.isScalarInteger() &&
      (N0.getOpcode() == ISD::ADD || N0.getOpcode() == ISD::OR)) {
    auto *C1 = dyn_cast<ConstantSDNode>(N0->getOperand(1));
    auto *C2 = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (C1 && C2) {
      APInt C1Int = C1->getAPIntValue();
      APInt ShiftedC1Int = C1Int << C2->getAPIntValue();

      // We can materialise `c1 << c2` into an add immediate, so it's "free",
      // and the combine should happen, to potentially allow further combines
      // later.
      if (ShiftedC1Int.getMinSignedBits() <= 64 &&
          isLegalAddImmediate(ShiftedC1Int.getSExtValue()))
        return true;

      // We can materialise `c1` in an add immediate, so it's "free", and the
      // combine should be prevented.
      if (C1Int.getMinSignedBits() <= 64 &&
          isLegalAddImmediate(C1Int.getSExtValue()))
        return false;

      // Neither constant will fit into an immediate, so find materialisation
      // costs.
      int C1Cost = RISCVMatInt::getIntMatCost(C1Int, Ty.getSizeInBits(),
                                              Subtarget.is64Bit());
      int ShiftedC1Cost = RISCVMatInt::getIntMatCost(
          ShiftedC1Int, Ty.getSizeInBits(), Subtarget.is64Bit());

      // Materialising `c1` is cheaper than materialising `c1 << c2`, so the
      // combine should be prevented.
      if (C1Cost < ShiftedC1Cost)
        return false;
    }
  }
  return true;
}

unsigned RISCVTargetLowering::ComputeNumSignBitsForTargetNode(
    SDValue Op, const APInt &DemandedElts, const SelectionDAG &DAG,
    unsigned Depth) const {
  switch (Op.getOpcode()) {
  default:
    break;
  case RISCVISD::SLLW:
  case RISCVISD::SRAW:
  case RISCVISD::SRLW:
  case RISCVISD::DIVW:
  case RISCVISD::DIVUW:
  case RISCVISD::REMUW:
    // TODO: As the result is sign-extended, this is conservatively correct. A
    // more precise answer could be calculated for SRAW depending on known
    // bits in the shift amount.
    return 33;
  case RISCVISD::VMV_X_S:
  case RISCVISD::EXTRACT_VECTOR_ELT:
    unsigned XLen = DAG.getDataLayout().getLargestLegalIntTypeSizeInBits();
    // The number of sign bits of the scalar result is computed by obtaining the
    // element type of the input vector operand, substracting its width from the
    // XLEN, and then adding one (sign bit within the element type).
    return XLen -
           Op->getOperand(0)
               .getValueType()
               .getVectorElementType()
               .getSizeInBits() +
           1;
  }

  return 1;
}

static MachineBasicBlock *emitReadCycleWidePseudo(MachineInstr &MI,
                                                  MachineBasicBlock *BB) {
  assert(MI.getOpcode() == RISCV::ReadCycleWide && "Unexpected instruction");

  // To read the 64-bit cycle CSR on a 32-bit target, we read the two halves.
  // Should the count have wrapped while it was being read, we need to try
  // again.
  // ...
  // read:
  // rdcycleh x3 # load high word of cycle
  // rdcycle  x2 # load low word of cycle
  // rdcycleh x4 # load high word of cycle
  // bne x3, x4, read # check if high word reads match, otherwise try again
  // ...

  MachineFunction &MF = *BB->getParent();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = ++BB->getIterator();

  MachineBasicBlock *LoopMBB = MF.CreateMachineBasicBlock(LLVM_BB);
  MF.insert(It, LoopMBB);

  MachineBasicBlock *DoneMBB = MF.CreateMachineBasicBlock(LLVM_BB);
  MF.insert(It, DoneMBB);

  // Transfer the remainder of BB and its successor edges to DoneMBB.
  DoneMBB->splice(DoneMBB->begin(), BB,
                  std::next(MachineBasicBlock::iterator(MI)), BB->end());
  DoneMBB->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(LoopMBB);

  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  Register ReadAgainReg = RegInfo.createVirtualRegister(&RISCV::GPRRegClass);
  Register LoReg = MI.getOperand(0).getReg();
  Register HiReg = MI.getOperand(1).getReg();
  DebugLoc DL = MI.getDebugLoc();

  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  BuildMI(LoopMBB, DL, TII->get(RISCV::CSRRS), HiReg)
      .addImm(RISCVSysReg::lookupSysRegByName("CYCLEH")->Encoding)
      .addReg(RISCV::X0);
  BuildMI(LoopMBB, DL, TII->get(RISCV::CSRRS), LoReg)
      .addImm(RISCVSysReg::lookupSysRegByName("CYCLE")->Encoding)
      .addReg(RISCV::X0);
  BuildMI(LoopMBB, DL, TII->get(RISCV::CSRRS), ReadAgainReg)
      .addImm(RISCVSysReg::lookupSysRegByName("CYCLEH")->Encoding)
      .addReg(RISCV::X0);

  BuildMI(LoopMBB, DL, TII->get(RISCV::BNE))
      .addReg(HiReg)
      .addReg(ReadAgainReg)
      .addMBB(LoopMBB);

  LoopMBB->addSuccessor(LoopMBB);
  LoopMBB->addSuccessor(DoneMBB);

  MI.eraseFromParent();

  return DoneMBB;
}

static MachineBasicBlock *emitSplitF64Pseudo(MachineInstr &MI,
                                             MachineBasicBlock *BB) {
  assert(MI.getOpcode() == RISCV::SplitF64Pseudo && "Unexpected instruction");

  MachineFunction &MF = *BB->getParent();
  DebugLoc DL = MI.getDebugLoc();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();
  Register LoReg = MI.getOperand(0).getReg();
  Register HiReg = MI.getOperand(1).getReg();
  Register SrcReg = MI.getOperand(2).getReg();
  const TargetRegisterClass *SrcRC = &RISCV::FPR64RegClass;
  int FI = MF.getInfo<RISCVMachineFunctionInfo>()->getMoveF64FrameIndex(MF);

  TII.storeRegToStackSlot(*BB, MI, SrcReg, MI.getOperand(2).isKill(), FI, SrcRC,
                          RI);
  MachineMemOperand *MMO =
      MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(MF, FI),
                              MachineMemOperand::MOLoad, 8, Align(8));
  BuildMI(*BB, MI, DL, TII.get(RISCV::LW), LoReg)
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMO);
  BuildMI(*BB, MI, DL, TII.get(RISCV::LW), HiReg)
      .addFrameIndex(FI)
      .addImm(4)
      .addMemOperand(MMO);
  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static MachineBasicBlock *emitBuildPairF64Pseudo(MachineInstr &MI,
                                                 MachineBasicBlock *BB) {
  assert(MI.getOpcode() == RISCV::BuildPairF64Pseudo &&
         "Unexpected instruction");

  MachineFunction &MF = *BB->getParent();
  DebugLoc DL = MI.getDebugLoc();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();
  Register DstReg = MI.getOperand(0).getReg();
  Register LoReg = MI.getOperand(1).getReg();
  Register HiReg = MI.getOperand(2).getReg();
  const TargetRegisterClass *DstRC = &RISCV::FPR64RegClass;
  int FI = MF.getInfo<RISCVMachineFunctionInfo>()->getMoveF64FrameIndex(MF);

  MachineMemOperand *MMO =
      MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(MF, FI),
                              MachineMemOperand::MOStore, 8, Align(8));
  BuildMI(*BB, MI, DL, TII.get(RISCV::SW))
      .addReg(LoReg, getKillRegState(MI.getOperand(1).isKill()))
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMO);
  BuildMI(*BB, MI, DL, TII.get(RISCV::SW))
      .addReg(HiReg, getKillRegState(MI.getOperand(2).isKill()))
      .addFrameIndex(FI)
      .addImm(4)
      .addMemOperand(MMO);
  TII.loadRegFromStackSlot(*BB, MI, DstReg, FI, DstRC, RI);
  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static bool isSelectPseudo(MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    return false;
  case RISCV::Select_GPR_Using_CC_GPR:
  case RISCV::Select_FPR32_Using_CC_GPR:
  case RISCV::Select_FPR64_Using_CC_GPR:
    return true;
  }
}

static MachineBasicBlock *emitSelectPseudo(MachineInstr &MI,
                                           MachineBasicBlock *BB) {
  // To "insert" Select_* instructions, we actually have to insert the triangle
  // control-flow pattern.  The incoming instructions know the destination vreg
  // to set, the condition code register to branch on, the true/false values to
  // select between, and the condcode to use to select the appropriate branch.
  //
  // We produce the following control flow:
  //     HeadMBB
  //     |  \
  //     |  IfFalseMBB
  //     | /
  //    TailMBB
  //
  // When we find a sequence of selects we attempt to optimize their emission
  // by sharing the control flow. Currently we only handle cases where we have
  // multiple selects with the exact same condition (same LHS, RHS and CC).
  // The selects may be interleaved with other instructions if the other
  // instructions meet some requirements we deem safe:
  // - They are debug instructions. Otherwise,
  // - They do not have side-effects, do not access memory and their inputs do
  //   not depend on the results of the select pseudo-instructions.
  // The TrueV/FalseV operands of the selects cannot depend on the result of
  // previous selects in the sequence.
  // These conditions could be further relaxed. See the X86 target for a
  // related approach and more information.
  Register LHS = MI.getOperand(1).getReg();
  Register RHS = MI.getOperand(2).getReg();
  auto CC = static_cast<ISD::CondCode>(MI.getOperand(3).getImm());

  SmallVector<MachineInstr *, 4> SelectDebugValues;
  SmallSet<Register, 4> SelectDests;
  SelectDests.insert(MI.getOperand(0).getReg());

  MachineInstr *LastSelectPseudo = &MI;

  for (auto E = BB->end(), SequenceMBBI = MachineBasicBlock::iterator(MI);
       SequenceMBBI != E; ++SequenceMBBI) {
    if (SequenceMBBI->isDebugInstr())
      continue;
    else if (isSelectPseudo(*SequenceMBBI)) {
      if (SequenceMBBI->getOperand(1).getReg() != LHS ||
          SequenceMBBI->getOperand(2).getReg() != RHS ||
          SequenceMBBI->getOperand(3).getImm() != CC ||
          SelectDests.count(SequenceMBBI->getOperand(4).getReg()) ||
          SelectDests.count(SequenceMBBI->getOperand(5).getReg()))
        break;
      LastSelectPseudo = &*SequenceMBBI;
      SequenceMBBI->collectDebugValues(SelectDebugValues);
      SelectDests.insert(SequenceMBBI->getOperand(0).getReg());
    } else {
      if (SequenceMBBI->hasUnmodeledSideEffects() ||
          SequenceMBBI->mayLoadOrStore())
        break;
      if (llvm::any_of(SequenceMBBI->operands(), [&](MachineOperand &MO) {
            return MO.isReg() && MO.isUse() && SelectDests.count(MO.getReg());
          }))
        break;
    }
  }

  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  DebugLoc DL = MI.getDebugLoc();
  MachineFunction::iterator I = ++BB->getIterator();

  MachineBasicBlock *HeadMBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *TailMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *IfFalseMBB = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(I, IfFalseMBB);
  F->insert(I, TailMBB);

  // Transfer debug instructions associated with the selects to TailMBB.
  for (MachineInstr *DebugInstr : SelectDebugValues) {
    TailMBB->push_back(DebugInstr->removeFromParent());
  }

  // Move all instructions after the sequence to TailMBB.
  TailMBB->splice(TailMBB->end(), HeadMBB,
                  std::next(LastSelectPseudo->getIterator()), HeadMBB->end());
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi nodes for the selects.
  TailMBB->transferSuccessorsAndUpdatePHIs(HeadMBB);
  // Set the successors for HeadMBB.
  HeadMBB->addSuccessor(IfFalseMBB);
  HeadMBB->addSuccessor(TailMBB);

  // Insert appropriate branch.
  unsigned Opcode = getBranchOpcodeForIntCondCode(CC);

  BuildMI(HeadMBB, DL, TII.get(Opcode)).addReg(LHS).addReg(RHS).addMBB(TailMBB);

  // IfFalseMBB just falls through to TailMBB.
  IfFalseMBB->addSuccessor(TailMBB);

  // Create PHIs for all of the select pseudo-instructions.
  auto SelectMBBI = MI.getIterator();
  auto SelectEnd = std::next(LastSelectPseudo->getIterator());
  auto InsertionPoint = TailMBB->begin();
  while (SelectMBBI != SelectEnd) {
    auto Next = std::next(SelectMBBI);
    if (isSelectPseudo(*SelectMBBI)) {
      // %Result = phi [ %TrueValue, HeadMBB ], [ %FalseValue, IfFalseMBB ]
      BuildMI(*TailMBB, InsertionPoint, SelectMBBI->getDebugLoc(),
              TII.get(RISCV::PHI), SelectMBBI->getOperand(0).getReg())
          .addReg(SelectMBBI->getOperand(4).getReg())
          .addMBB(HeadMBB)
          .addReg(SelectMBBI->getOperand(5).getReg())
          .addMBB(IfFalseMBB);
      SelectMBBI->eraseFromParent();
    }
    SelectMBBI = Next;
  }

  F->getProperties().reset(MachineFunctionProperties::Property::NoPHIs);
  return TailMBB;
}

static MachineBasicBlock *addEPISetVL(MachineInstr &MI, MachineBasicBlock *BB,
                                      int VLIndex, unsigned SEWIndex,
                                      unsigned VLMul) {
  MachineFunction &MF = *BB->getParent();
  DebugLoc DL = MI.getDebugLoc();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();

  unsigned SEW = MI.getOperand(SEWIndex).getImm();
  RISCVEPIVectorMultiplier::VectorMultiplier Multiplier;

  switch (VLMul) {
  default:
    llvm_unreachable("Unexpected VLMul");
  case 1:
    Multiplier = RISCVEPIVectorMultiplier::VMul1;
    break;
  case 2:
    Multiplier = RISCVEPIVectorMultiplier::VMul2;
    break;
  case 4:
    Multiplier = RISCVEPIVectorMultiplier::VMul4;
    break;
  case 8:
    Multiplier = RISCVEPIVectorMultiplier::VMul8;
  }

  RISCVEPIVectorElementWidth::VectorElementWidth ElementWidth;
  switch (SEW) {
  default:
    llvm_unreachable("Unexpected SEW for instruction");
  case 8:
    ElementWidth = RISCVEPIVectorElementWidth::ElementWidth8;
    break;
  case 16:
    ElementWidth = RISCVEPIVectorElementWidth::ElementWidth16;
    break;
  case 32:
    ElementWidth = RISCVEPIVectorElementWidth::ElementWidth32;
    break;
  case 64:
    ElementWidth = RISCVEPIVectorElementWidth::ElementWidth64;
    break;
  case 128:
    ElementWidth = RISCVEPIVectorElementWidth::ElementWidth128;
  }

  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Note: VL and VTYPE are alive here.
  MachineInstrBuilder MIB = BuildMI(*BB, MI, DL, TII.get(RISCV::PseudoVSETVLI));

  if (VLIndex >= 0) {
    // rs1 != X0.
    unsigned DestReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    MIB.addReg(DestReg, RegState::Define | RegState::Dead);
    MIB.addReg(MI.getOperand(VLIndex).getReg());
  } else {
    // No VL operator in the pseudo, do not modify VL (rd = X0, rs1 = X0).
    MIB.addReg(RISCV::X0, RegState::Define | RegState::Dead);
    MIB.addReg(RISCV::X0, RegState::Kill);
  }

  MIB.addImm((ElementWidth << 2) | Multiplier);

  // Remove (now) redundant operands from pseudo
  MI.getOperand(SEWIndex).setImm(-1);
  if (VLIndex >= 0) {
    MI.getOperand(VLIndex).setReg(RISCV::NoRegister);
    MI.getOperand(VLIndex).setIsKill(false);
  }

  return BB;
}

static MachineBasicBlock *emitComputeVSCALE(MachineInstr &MI,
                                            MachineBasicBlock *BB) {
  MachineFunction &MF = *BB->getParent();
  DebugLoc DL = MI.getDebugLoc();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();

  Register DestReg = MI.getOperand(0).getReg();

  // VSCALE can be computed as VLMAX of ELEN, given that the scaling factor for
  // ELEN is '1'.
  MachineInstr &I =
      *BuildMI(*BB, MI, DL, TII.get(RISCV::PseudoVSETVLI), DestReg)
           .addReg(RISCV::X0)
           // FIXME - ELEN hardcoded to SEW=64.
           .addImm(/* e64,m1 */ 3 << 2);
  // Set VTYPE and VL as dead.
  I.getOperand(3).setIsDead();
  I.getOperand(4).setIsDead();

  // The pseudo instruction is gone now.
  MI.eraseFromParent();
  return BB;
}

static MachineBasicBlock *emitComputeVMSET(MachineInstr &MI,
                                           MachineBasicBlock *BB) {
  MachineFunction &MF = *BB->getParent();
  DebugLoc DL = MI.getDebugLoc();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();

  unsigned VLMul;
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unexpected instruction");
  case RISCV::PseudoVMSET_M1:
    VLMul = 1;
    break;
  case RISCV::PseudoVMSET_M2:
    VLMul = 2;
    break;
  case RISCV::PseudoVMSET_M4:
    VLMul = 4;
    break;
  case RISCV::PseudoVMSET_M8:
    VLMul = 8;
    break;
  }

  Register DestReg = MI.getOperand(0).getReg();
  unsigned SEW = MI.getOperand(2).getImm();

  MachineInstr *NewMI =
      BuildMI(*BB, MI, DL, TII.get(RISCV::PseudoVMXNOR_MM_M1), DestReg)
          .addReg(DestReg, RegState::Undef)
          .addReg(DestReg, RegState::Undef)
          .addReg(MI.getOperand(1).getReg())
          .addImm(SEW);

  // The pseudo instruction is gone now.
  MI.eraseFromParent();

  return addEPISetVL(*NewMI, BB, /* VLIndex */ 3, /* SEWIndex */ 4, VLMul);
}

static MachineBasicBlock *emitComputeVMCLR(MachineInstr &MI,
                                           MachineBasicBlock *BB) {
  MachineFunction &MF = *BB->getParent();
  DebugLoc DL = MI.getDebugLoc();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();

  unsigned VLMul;
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unexpected instruction");
  case RISCV::PseudoVMCLR_M1:
    VLMul = 1;
    break;
  case RISCV::PseudoVMCLR_M2:
    VLMul = 2;
    break;
  case RISCV::PseudoVMCLR_M4:
    VLMul = 4;
    break;
  case RISCV::PseudoVMCLR_M8:
    VLMul = 8;
    break;
  }

  Register DestReg = MI.getOperand(0).getReg();
  unsigned SEW = MI.getOperand(2).getImm();

  MachineInstr *NewMI =
      BuildMI(*BB, MI, DL, TII.get(RISCV::PseudoVMXOR_MM_M1), DestReg)
          .addReg(DestReg, RegState::Undef)
          .addReg(DestReg, RegState::Undef)
          .addReg(MI.getOperand(1).getReg())
          .addImm(SEW);

  // The pseudo instruction is gone now.
  MI.eraseFromParent();

  return addEPISetVL(*NewMI, BB, /* VLIndex */ 3, /* SEWIndex */ 4, VLMul);
}

static MachineBasicBlock *emitImplicitVRM1Tuple(MachineInstr &MI,
                                              MachineBasicBlock *BB) {
  MachineFunction &MF = *BB->getParent();
  DebugLoc DL = MI.getDebugLoc();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();

  Register DestReg = MI.getOperand(0).getReg();
  BuildMI(*BB, MI, DL, TII.get(RISCV::IMPLICIT_DEF), DestReg);

  // The pseudo instruction is gone now.
  MI.eraseFromParent();
  return BB;
}

static MachineBasicBlock *
emitVBuildVRM1Tuple(MachineInstr &MI, MachineBasicBlock *BB,
                    const ArrayRef<unsigned> SubRegisters,
                    const TargetRegisterClass *RC) {
  MachineFunction &MF = *BB->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  DebugLoc DL = MI.getDebugLoc();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();

  assert(SubRegisters.size() > 1);
  Register DestReg = MI.getOperand(0).getReg();

  Register TmpSrc = MRI.createVirtualRegister(RC);
  BuildMI(*BB, MI, DL, TII.get(RISCV::IMPLICIT_DEF), TmpSrc);
  for (unsigned I = 0, E = SubRegisters.size(); I != E; I++) {
    Register TmpDest = I == E - 1 ? DestReg : MRI.createVirtualRegister(RC);
    BuildMI(*BB, MI, DL, TII.get(RISCV::INSERT_SUBREG), TmpDest)
        .addReg(TmpSrc)
        .addReg(MI.getOperand(I + 1).getReg())
        .addImm(SubRegisters[I]);
    TmpSrc = TmpDest;
  }

  // The pseudo instruction is gone now.
  MI.eraseFromParent();
  return BB;
}

MachineBasicBlock *
RISCVTargetLowering::EmitInstrWithCustomInserter(MachineInstr &MI,
                                                 MachineBasicBlock *BB) const {
  if (const RISCVEPIPseudosTable::EPIPseudoInfo *EPI =
          RISCVEPIPseudosTable::getEPIPseudoInfo(MI.getOpcode())) {
    int VLIndex = EPI->getVLIndex();
    int SEWIndex = EPI->getSEWIndex();

    // SEWIndex must be >= 0
    assert(SEWIndex >= 0);

    return addEPISetVL(MI, BB, VLIndex, SEWIndex, EPI->VLMul);
  }

  // Other EPI pseudo-instructions.
  switch (MI.getOpcode()) {
  default:
    break;
  case RISCV::PseudoVSCALE:
    return emitComputeVSCALE(MI, BB);
  case RISCV::PseudoVMSET_M1:
  case RISCV::PseudoVMSET_M2:
  case RISCV::PseudoVMSET_M4:
  case RISCV::PseudoVMSET_M8:
    return emitComputeVMSET(MI, BB);
  case RISCV::PseudoVMCLR_M1:
  case RISCV::PseudoVMCLR_M2:
  case RISCV::PseudoVMCLR_M4:
  case RISCV::PseudoVMCLR_M8:
    return emitComputeVMCLR(MI, BB);
  case RISCV::PseudoImplicitVRM1T2:
  case RISCV::PseudoImplicitVRM1T3:
  case RISCV::PseudoImplicitVRM1T4:
  case RISCV::PseudoImplicitVRM1T5:
  case RISCV::PseudoImplicitVRM1T6:
  case RISCV::PseudoImplicitVRM1T7:
  case RISCV::PseudoImplicitVRM1T8:
    return emitImplicitVRM1Tuple(MI, BB);
  case RISCV::PseudoVBuildVRM1T2:
    return emitVBuildVRM1Tuple(MI, BB, {RISCV::vtuple2_0, RISCV::vtuple2_1},
                               &RISCV::VRM1T2RegClass);
  case RISCV::PseudoVBuildVRM1T3:
    return emitVBuildVRM1Tuple(
        MI, BB, {RISCV::vtuple3_0, RISCV::vtuple3_1, RISCV::vtuple3_2},
        &RISCV::VRM1T3RegClass);
  case RISCV::PseudoVBuildVRM1T4:
    return emitVBuildVRM1Tuple(MI, BB,
                               {RISCV::vtuple4_0, RISCV::vtuple4_1,
                                RISCV::vtuple4_2, RISCV::vtuple4_3},
                               &RISCV::VRM1T4RegClass);
  case RISCV::PseudoVBuildVRM1T5:
    return emitVBuildVRM1Tuple(MI, BB,
                               {RISCV::vtuple5_0, RISCV::vtuple5_1,
                                RISCV::vtuple5_2, RISCV::vtuple5_3,
                                RISCV::vtuple5_4},
                               &RISCV::VRM1T5RegClass);
  case RISCV::PseudoVBuildVRM1T6:
    return emitVBuildVRM1Tuple(MI, BB,
                               {RISCV::vtuple6_0, RISCV::vtuple6_1,
                                RISCV::vtuple6_2, RISCV::vtuple6_3,
                                RISCV::vtuple6_4, RISCV::vtuple6_5},
                               &RISCV::VRM1T6RegClass);
  case RISCV::PseudoVBuildVRM1T7:
    return emitVBuildVRM1Tuple(
        MI, BB,
        {RISCV::vtuple7_0, RISCV::vtuple7_1, RISCV::vtuple7_2, RISCV::vtuple7_3,
         RISCV::vtuple7_4, RISCV::vtuple7_5, RISCV::vtuple7_6},
        &RISCV::VRM1T7RegClass);
  case RISCV::PseudoVBuildVRM1T8:
    return emitVBuildVRM1Tuple(MI, BB,
                               {RISCV::vtuple8_0, RISCV::vtuple8_1,
                                RISCV::vtuple8_2, RISCV::vtuple8_3,
                                RISCV::vtuple8_4, RISCV::vtuple8_5,
                                RISCV::vtuple8_6, RISCV::vtuple8_7},
                               &RISCV::VRM1T8RegClass);
  }

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unexpected instr type to insert");
  case RISCV::ReadCycleWide:
    assert(!Subtarget.is64Bit() &&
           "ReadCycleWrite is only to be used on riscv32");
    return emitReadCycleWidePseudo(MI, BB);
  case RISCV::Select_GPR_Using_CC_GPR:
  case RISCV::Select_FPR32_Using_CC_GPR:
  case RISCV::Select_FPR64_Using_CC_GPR:
    return emitSelectPseudo(MI, BB);
  case RISCV::BuildPairF64Pseudo:
    return emitBuildPairF64Pseudo(MI, BB);
  case RISCV::SplitF64Pseudo:
    return emitSplitF64Pseudo(MI, BB);
  }
}

// Calling Convention Implementation.
// The expectations for frontend ABI lowering vary from target to target.
// Ideally, an LLVM frontend would be able to avoid worrying about many ABI
// details, but this is a longer term goal. For now, we simply try to keep the
// role of the frontend as simple and well-defined as possible. The rules can
// be summarised as:
// * Never split up large scalar arguments. We handle them here.
// * If a hardfloat calling convention is being used, and the struct may be
// passed in a pair of registers (fp+fp, int+fp), and both registers are
// available, then pass as two separate arguments. If either the GPRs or FPRs
// are exhausted, then pass according to the rule below.
// * If a struct could never be passed in registers or directly in a stack
// slot (as it is larger than 2*XLEN and the floating point rules don't
// apply), then pass it using a pointer with the byval attribute.
// * If a struct is less than 2*XLEN, then coerce to either a two-element
// word-sized array or a 2*XLEN scalar (depending on alignment).
// * The frontend can determine whether a struct is returned by reference or
// not based on its size and fields. If it will be returned by reference, the
// frontend must modify the prototype so a pointer with the sret annotation is
// passed as the first argument. This is not necessary for large scalar
// returns.
// * Struct return values and varargs should be coerced to structs containing
// register-size fields in the same situations they would be for fixed
// arguments.

static const MCPhysReg ArgGPRs[] = {
  RISCV::X10, RISCV::X11, RISCV::X12, RISCV::X13,
  RISCV::X14, RISCV::X15, RISCV::X16, RISCV::X17
};
static const MCPhysReg ArgFPR32s[] = {
  RISCV::F10_F, RISCV::F11_F, RISCV::F12_F, RISCV::F13_F,
  RISCV::F14_F, RISCV::F15_F, RISCV::F16_F, RISCV::F17_F
};
static const MCPhysReg ArgFPR64s[] = {
  RISCV::F10_D, RISCV::F11_D, RISCV::F12_D, RISCV::F13_D,
  RISCV::F14_D, RISCV::F15_D, RISCV::F16_D, RISCV::F17_D
};

static const MCPhysReg ArgVRs[] = {RISCV::V16, RISCV::V17, RISCV::V18,
                                    RISCV::V19, RISCV::V20, RISCV::V21,
                                    RISCV::V22, RISCV::V23};
static const MCPhysReg ArgVRM2s[] = {
    RISCV::V16M2,
    RISCV::V18M2,
    RISCV::V20M2,
    RISCV::V22M2,
};
static const MCPhysReg ArgVRM4s[] = {RISCV::V16M4, RISCV::V20M4};
static const MCPhysReg ArgVRM8s[] = {RISCV::V16M8};

// Pass a 2*XLEN argument that has been split into two XLEN values through
// registers or the stack as necessary.
static bool CC_RISCVAssign2XLen(unsigned XLen, CCState &State, CCValAssign VA1,
                                ISD::ArgFlagsTy ArgFlags1, unsigned ValNo2,
                                MVT ValVT2, MVT LocVT2,
                                ISD::ArgFlagsTy ArgFlags2) {
  unsigned XLenInBytes = XLen / 8;
  if (Register Reg = State.AllocateReg(ArgGPRs)) {
    // At least one half can be passed via register.
    State.addLoc(CCValAssign::getReg(VA1.getValNo(), VA1.getValVT(), Reg,
                                     VA1.getLocVT(), CCValAssign::Full));
  } else {
    // Both halves must be passed on the stack, with proper alignment.
    Align StackAlign =
        std::max(Align(XLenInBytes), ArgFlags1.getNonZeroOrigAlign());
    State.addLoc(
        CCValAssign::getMem(VA1.getValNo(), VA1.getValVT(),
                            State.AllocateStack(XLenInBytes, StackAlign),
                            VA1.getLocVT(), CCValAssign::Full));
    State.addLoc(CCValAssign::getMem(
        ValNo2, ValVT2, State.AllocateStack(XLenInBytes, Align(XLenInBytes)),
        LocVT2, CCValAssign::Full));
    return false;
  }

  if (Register Reg = State.AllocateReg(ArgGPRs)) {
    // The second half can also be passed via register.
    State.addLoc(
        CCValAssign::getReg(ValNo2, ValVT2, Reg, LocVT2, CCValAssign::Full));
  } else {
    // The second half is passed via the stack, without additional alignment.
    State.addLoc(CCValAssign::getMem(
        ValNo2, ValVT2, State.AllocateStack(XLenInBytes, Align(XLenInBytes)),
        LocVT2, CCValAssign::Full));
  }

  return false;
}

// Implements the RISC-V calling convention. Returns true upon failure.
static bool CC_RISCV(const DataLayout &DL, RISCVABI::ABI ABI, unsigned ValNo,
                     MVT ValVT, MVT LocVT, CCValAssign::LocInfo LocInfo,
                     ISD::ArgFlagsTy ArgFlags, CCState &State, bool IsFixed,
                     bool IsRet, Type *OrigTy, const RISCVTargetLowering *TLI,
                     Optional<unsigned> FirstMaskArgument) {
  unsigned XLen = DL.getLargestLegalIntTypeSizeInBits();
  assert(XLen == 32 || XLen == 64);
  MVT XLenVT = XLen == 32 ? MVT::i32 : MVT::i64;

  // Any return value split in to more than two values can't be returned
  // directly.
  if (IsRet && ValNo > 1)
    return true;

  // UseGPRForF32 if targeting one of the soft-float ABIs, if passing a
  // variadic argument, or if no F32 argument registers are available.
  bool UseGPRForF32 = true;
  // UseGPRForF64 if targeting soft-float ABIs or an FLEN=32 ABI, if passing a
  // variadic argument, or if no F64 argument registers are available.
  bool UseGPRForF64 = true;

  switch (ABI) {
  default:
    llvm_unreachable("Unexpected ABI");
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_LP64:
    break;
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_LP64F:
    UseGPRForF32 = !IsFixed;
    break;
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64D:
    UseGPRForF32 = !IsFixed;
    UseGPRForF64 = !IsFixed;
    break;
  }

  if (State.getFirstUnallocated(ArgFPR32s) == array_lengthof(ArgFPR32s))
    UseGPRForF32 = true;
  if (State.getFirstUnallocated(ArgFPR64s) == array_lengthof(ArgFPR64s))
    UseGPRForF64 = true;

  // From this point on, rely on UseGPRForF32, UseGPRForF64 and similar local
  // variables rather than directly checking against the target ABI.

  if (UseGPRForF32 && ValVT == MVT::f32) {
    LocVT = XLenVT;
    LocInfo = CCValAssign::BCvt;
  } else if (UseGPRForF64 && XLen == 64 && ValVT == MVT::f64) {
    LocVT = MVT::i64;
    LocInfo = CCValAssign::BCvt;
  }

  // If this is a variadic argument, the RISC-V calling convention requires
  // that it is assigned an 'even' or 'aligned' register if it has 8-byte
  // alignment (RV32) or 16-byte alignment (RV64). An aligned register should
  // be used regardless of whether the original argument was split during
  // legalisation or not. The argument will not be passed by registers if the
  // original type is larger than 2*XLEN, so the register alignment rule does
  // not apply.
  unsigned TwoXLenInBytes = (2 * XLen) / 8;
  if (!IsFixed && ArgFlags.getNonZeroOrigAlign() == TwoXLenInBytes &&
      DL.getTypeAllocSize(OrigTy) == TwoXLenInBytes) {
    unsigned RegIdx = State.getFirstUnallocated(ArgGPRs);
    // Skip 'odd' register if necessary.
    if (RegIdx != array_lengthof(ArgGPRs) && RegIdx % 2 == 1)
      State.AllocateReg(ArgGPRs);
  }

  SmallVectorImpl<CCValAssign> &PendingLocs = State.getPendingLocs();
  SmallVectorImpl<ISD::ArgFlagsTy> &PendingArgFlags =
      State.getPendingArgFlags();

  assert(PendingLocs.size() == PendingArgFlags.size() &&
         "PendingLocs and PendingArgFlags out of sync");

  // Handle passing f64 on RV32D with a soft float ABI or when floating point
  // registers are exhausted.
  if (UseGPRForF64 && XLen == 32 && ValVT == MVT::f64) {
    assert(!ArgFlags.isSplit() && PendingLocs.empty() &&
           "Can't lower f64 if it is split");
    // Depending on available argument GPRS, f64 may be passed in a pair of
    // GPRs, split between a GPR and the stack, or passed completely on the
    // stack. LowerCall/LowerFormalArguments/LowerReturn must recognise these
    // cases.
    Register Reg = State.AllocateReg(ArgGPRs);
    LocVT = MVT::i32;
    if (!Reg) {
      unsigned StackOffset = State.AllocateStack(8, Align(8));
      State.addLoc(
          CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
      return false;
    }
    if (!State.AllocateReg(ArgGPRs))
      State.AllocateStack(4, Align(4));
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    return false;
  }

  // Split arguments might be passed indirectly, so keep track of the pending
  // values.
  if (ArgFlags.isSplit() || !PendingLocs.empty()) {
    LocVT = XLenVT;
    LocInfo = CCValAssign::Indirect;
    PendingLocs.push_back(
        CCValAssign::getPending(ValNo, ValVT, LocVT, LocInfo));
    PendingArgFlags.push_back(ArgFlags);
    if (!ArgFlags.isSplitEnd()) {
      return false;
    }
  }

  // If the split argument only had two elements, it should be passed directly
  // in registers or on the stack.
  if (ArgFlags.isSplitEnd() && PendingLocs.size() <= 2) {
    assert(PendingLocs.size() == 2 && "Unexpected PendingLocs.size()");
    // Apply the normal calling convention rules to the first half of the
    // split argument.
    CCValAssign VA = PendingLocs[0];
    ISD::ArgFlagsTy AF = PendingArgFlags[0];
    PendingLocs.clear();
    PendingArgFlags.clear();
    return CC_RISCVAssign2XLen(XLen, State, VA, AF, ValNo, ValVT, LocVT,
                               ArgFlags);
  }

  // Allocate to a register if possible, or else a stack slot.
  Register Reg;
  if (ValVT == MVT::f32 && !UseGPRForF32)
    Reg = State.AllocateReg(ArgFPR32s, ArgFPR64s);
  else if (ValVT == MVT::f64 && !UseGPRForF64)
    Reg = State.AllocateReg(ArgFPR64s, ArgFPR32s);
  else if (ValVT.isScalableVector()) {
    const TargetRegisterClass *RC = TLI->getRegClassFor(ValVT);
    if (RC->hasSuperClassEq(&RISCV::VRRegClass)) {
      if (FirstMaskArgument.hasValue() &&
          ValNo == FirstMaskArgument.getValue()) {
        Reg = RISCV::V0;
      } else {
        Reg = State.AllocateReg(ArgVRs);
      }
    } else if (RC->hasSuperClassEq(&RISCV::VRM2RegClass)) {
      Reg = State.AllocateReg(ArgVRM2s);
    } else if (RC->hasSuperClassEq(&RISCV::VRM4RegClass)) {
      Reg = State.AllocateReg(ArgVRM4s);
    } else if (RC->hasSuperClassEq(&RISCV::VRM8RegClass)) {
      Reg = State.AllocateReg(ArgVRM8s);
    } else {
      llvm_unreachable("Unhandled class register for ValueType");
    }
    if (!Reg) {
      LocInfo = CCValAssign::Indirect;
      // Try using a GPR to pass the address
      Reg = State.AllocateReg(ArgGPRs);
      LocVT = XLenVT;
    }
  } else
    Reg = State.AllocateReg(ArgGPRs);
  unsigned StackOffset =
      Reg ? 0 : State.AllocateStack(XLen / 8, Align(XLen / 8));

  // If we reach this point and PendingLocs is non-empty, we must be at the
  // end of a split argument that must be passed indirectly.
  if (!PendingLocs.empty()) {
    assert(ArgFlags.isSplitEnd() && "Expected ArgFlags.isSplitEnd()");
    assert(PendingLocs.size() > 2 && "Unexpected PendingLocs.size()");

    for (auto &It : PendingLocs) {
      if (Reg)
        It.convertToReg(Reg);
      else
        It.convertToMem(StackOffset);
      State.addLoc(It);
    }
    PendingLocs.clear();
    PendingArgFlags.clear();
    return false;
  }

  assert((!UseGPRForF32 || !UseGPRForF64 ||
          (TLI->getSubtarget().hasStdExtV() && ValVT.isScalableVector()) ||
          LocVT == XLenVT) &&
         "Expected an XLenVT at this stage");

  if (Reg) {
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    return false;
  }

  // When an f32 or f64 is passed on the stack, no bit-conversion is needed.
  if (ValVT == MVT::f32 || ValVT == MVT::f64) {
    LocVT = ValVT;
    LocInfo = CCValAssign::Full;
  }
  State.addLoc(CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
  return false;
}

template <typename ArgTy>
static void PreAssignMask(const ArgTy &Args,
                          Optional<unsigned> &FirstMaskArgument,
                          CCState &CCInfo) {
  unsigned NumArgs = Args.size();
  for (unsigned i = 0; i != NumArgs; ++i) {
    MVT ArgVT = Args[i].VT;
    if (!ArgVT.isScalableVector() ||
        ArgVT.getVectorElementType().SimpleTy != MVT::i1)
      continue;

    FirstMaskArgument = i;
    CCInfo.AllocateReg(RISCV::V0);
    break;
  }
}

void RISCVTargetLowering::analyzeInputArgs(
    MachineFunction &MF, CCState &CCInfo,
    const SmallVectorImpl<ISD::InputArg> &Ins, bool IsRet) const {
  unsigned NumArgs = Ins.size();
  FunctionType *FType = MF.getFunction().getFunctionType();

  Optional<unsigned> FirstMaskArgument;
  if (Subtarget.hasStdExtV()) {
    PreAssignMask(Ins, FirstMaskArgument, CCInfo);
  }

  for (unsigned i = 0; i != NumArgs; ++i) {
    MVT ArgVT = Ins[i].VT;
    ISD::ArgFlagsTy ArgFlags = Ins[i].Flags;

    Type *ArgTy = nullptr;
    if (IsRet)
      ArgTy = FType->getReturnType();
    else if (Ins[i].isOrigArg())
      ArgTy = FType->getParamType(Ins[i].getOrigArgIndex());

    RISCVABI::ABI ABI = MF.getSubtarget<RISCVSubtarget>().getTargetABI();
    if (CC_RISCV(MF.getDataLayout(), ABI, i, ArgVT, ArgVT, CCValAssign::Full,
                 ArgFlags, CCInfo, /*IsFixed=*/true, IsRet, ArgTy, this,
                 FirstMaskArgument)) {
      LLVM_DEBUG(dbgs() << "InputArg #" << i << " has unhandled type "
                        << EVT(ArgVT).getEVTString() << '\n');
      llvm_unreachable(nullptr);
    }
  }
}

void RISCVTargetLowering::analyzeOutputArgs(
    MachineFunction &MF, CCState &CCInfo,
    const SmallVectorImpl<ISD::OutputArg> &Outs, bool IsRet,
    CallLoweringInfo *CLI) const {
  unsigned NumArgs = Outs.size();

  Optional<unsigned> FirstMaskArgument;
  if (Subtarget.hasStdExtV()) {
    PreAssignMask(Outs, FirstMaskArgument, CCInfo);
  }

  for (unsigned i = 0; i != NumArgs; i++) {
    MVT ArgVT = Outs[i].VT;
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    Type *OrigTy = CLI ? CLI->getArgs()[Outs[i].OrigArgIndex].Ty : nullptr;

    RISCVABI::ABI ABI = MF.getSubtarget<RISCVSubtarget>().getTargetABI();
    if (CC_RISCV(MF.getDataLayout(), ABI, i, ArgVT, ArgVT, CCValAssign::Full,
                 ArgFlags, CCInfo, Outs[i].IsFixed, IsRet, OrigTy, this,
                 FirstMaskArgument)) {
      LLVM_DEBUG(dbgs() << "OutputArg #" << i << " has unhandled type "
                        << EVT(ArgVT).getEVTString() << "\n");
      llvm_unreachable(nullptr);
    }
  }
}

// Convert Val to a ValVT. Should not be called for CCValAssign::Indirect
// values.
static SDValue convertLocVTToValVT(SelectionDAG &DAG, SDValue Val,
                                   const CCValAssign &VA, const SDLoc &DL) {
  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Full:
    break;
  case CCValAssign::BCvt:
    if (VA.getLocVT() == MVT::i64 && VA.getValVT() == MVT::f32) {
      Val = DAG.getNode(RISCVISD::FMV_W_X_RV64, DL, MVT::f32, Val);
      break;
    }
    Val = DAG.getNode(ISD::BITCAST, DL, VA.getValVT(), Val);
    break;
  }
  return Val;
}

// The caller is responsible for loading the full value if the argument is
// passed with CCValAssign::Indirect.
static SDValue unpackFromRegLoc(SelectionDAG &DAG, SDValue Chain,
                                const CCValAssign &VA, const SDLoc &DL,
                                const RISCVTargetLowering *TLI) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  EVT LocVT = VA.getLocVT();
  SDValue Val;
  const TargetRegisterClass *RC;

  if (LocVT.getSimpleVT().isScalableVector()) {
    RC = TLI->getRegClassFor(LocVT.getSimpleVT());
  } else {
    switch (LocVT.getSimpleVT().SimpleTy) {
    default:
      llvm_unreachable("Unexpected register type");
    case MVT::i32:
    case MVT::i64:
      RC = &RISCV::GPRRegClass;
      break;
    case MVT::f32:
      RC = &RISCV::FPR32RegClass;
      break;
    case MVT::f64:
      RC = &RISCV::FPR64RegClass;
      break;
    }
  }

  Register VReg = RegInfo.createVirtualRegister(RC);
  RegInfo.addLiveIn(VA.getLocReg(), VReg);
  Val = DAG.getCopyFromReg(Chain, DL, VReg, LocVT);

  if (VA.getLocInfo() == CCValAssign::Indirect)
    return Val;

  return convertLocVTToValVT(DAG, Val, VA, DL);
}

static SDValue convertValVTToLocVT(SelectionDAG &DAG, SDValue Val,
                                   const CCValAssign &VA, const SDLoc &DL) {
  EVT LocVT = VA.getLocVT();

  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Full:
    break;
  case CCValAssign::BCvt:
    if (VA.getLocVT() == MVT::i64 && VA.getValVT() == MVT::f32) {
      Val = DAG.getNode(RISCVISD::FMV_X_ANYEXTW_RV64, DL, MVT::i64, Val);
      break;
    }
    Val = DAG.getNode(ISD::BITCAST, DL, LocVT, Val);
    break;
  }
  return Val;
}

// The caller is responsible for loading the full value if the argument is
// passed with CCValAssign::Indirect.
static SDValue unpackFromMemLoc(SelectionDAG &DAG, SDValue Chain,
                                const CCValAssign &VA, const SDLoc &DL) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  EVT LocVT = VA.getLocVT();
  EVT ValVT = VA.getValVT();
  EVT PtrVT = MVT::getIntegerVT(DAG.getDataLayout().getPointerSizeInBits(0));
  int FI = MFI.CreateFixedObject(ValVT.getSizeInBits().getKnownMinSize() / 8,
                                 VA.getLocMemOffset(), /*Immutable=*/true);
  SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
  SDValue Val;

  ISD::LoadExtType ExtType;
  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unexpected CCValAssign::LocInfo");
  case CCValAssign::Indirect:
    if (ValVT.isScalableVector()) {
      // Indirect load of the vector value
      SDValue Ptr = DAG.getLoad(
          LocVT, DL, Chain, FIN,
          MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI));
      return Ptr;
    }
    LLVM_FALLTHROUGH;
  case CCValAssign::Full:
  case CCValAssign::BCvt:
    ExtType = ISD::NON_EXTLOAD;
    break;
  }
  Val = DAG.getExtLoad(
      ExtType, DL, LocVT, Chain, FIN,
      MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI), ValVT);
  return Val;
}

static SDValue unpackF64OnRV32DSoftABI(SelectionDAG &DAG, SDValue Chain,
                                       const CCValAssign &VA, const SDLoc &DL) {
  assert(VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64 &&
         "Unexpected VA");
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();

  if (VA.isMemLoc()) {
    // f64 is passed on the stack.
    int FI = MFI.CreateFixedObject(8, VA.getLocMemOffset(), /*Immutable=*/true);
    SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
    return DAG.getLoad(MVT::f64, DL, Chain, FIN,
                       MachinePointerInfo::getFixedStack(MF, FI));
  }

  assert(VA.isRegLoc() && "Expected register VA assignment");

  Register LoVReg = RegInfo.createVirtualRegister(&RISCV::GPRRegClass);
  RegInfo.addLiveIn(VA.getLocReg(), LoVReg);
  SDValue Lo = DAG.getCopyFromReg(Chain, DL, LoVReg, MVT::i32);
  SDValue Hi;
  if (VA.getLocReg() == RISCV::X17) {
    // Second half of f64 is passed on the stack.
    int FI = MFI.CreateFixedObject(4, 0, /*Immutable=*/true);
    SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
    Hi = DAG.getLoad(MVT::i32, DL, Chain, FIN,
                     MachinePointerInfo::getFixedStack(MF, FI));
  } else {
    // Second half of f64 is passed in another GPR.
    Register HiVReg = RegInfo.createVirtualRegister(&RISCV::GPRRegClass);
    RegInfo.addLiveIn(VA.getLocReg() + 1, HiVReg);
    Hi = DAG.getCopyFromReg(Chain, DL, HiVReg, MVT::i32);
  }
  return DAG.getNode(RISCVISD::BuildPairF64, DL, MVT::f64, Lo, Hi);
}

// FastCC has less than 1% performance improvement for some particular
// benchmark. But theoretically, it may has benenfit for some cases.
static bool CC_RISCV_FastCC(unsigned ValNo, MVT ValVT, MVT LocVT,
                            CCValAssign::LocInfo LocInfo,
                            ISD::ArgFlagsTy ArgFlags, CCState &State) {

  if (LocVT == MVT::i32 || LocVT == MVT::i64) {
    // X5 and X6 might be used for save-restore libcall.
    static const MCPhysReg GPRList[] = {
        RISCV::X10, RISCV::X11, RISCV::X12, RISCV::X13, RISCV::X14,
        RISCV::X15, RISCV::X16, RISCV::X17, RISCV::X7,  RISCV::X28,
        RISCV::X29, RISCV::X30, RISCV::X31};
    if (unsigned Reg = State.AllocateReg(GPRList)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f32) {
    static const MCPhysReg FPR32List[] = {
        RISCV::F10_F, RISCV::F11_F, RISCV::F12_F, RISCV::F13_F, RISCV::F14_F,
        RISCV::F15_F, RISCV::F16_F, RISCV::F17_F, RISCV::F0_F,  RISCV::F1_F,
        RISCV::F2_F,  RISCV::F3_F,  RISCV::F4_F,  RISCV::F5_F,  RISCV::F6_F,
        RISCV::F7_F,  RISCV::F28_F, RISCV::F29_F, RISCV::F30_F, RISCV::F31_F};
    if (unsigned Reg = State.AllocateReg(FPR32List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f64) {
    static const MCPhysReg FPR64List[] = {
        RISCV::F10_D, RISCV::F11_D, RISCV::F12_D, RISCV::F13_D, RISCV::F14_D,
        RISCV::F15_D, RISCV::F16_D, RISCV::F17_D, RISCV::F0_D,  RISCV::F1_D,
        RISCV::F2_D,  RISCV::F3_D,  RISCV::F4_D,  RISCV::F5_D,  RISCV::F6_D,
        RISCV::F7_D,  RISCV::F28_D, RISCV::F29_D, RISCV::F30_D, RISCV::F31_D};
    if (unsigned Reg = State.AllocateReg(FPR64List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::i32 || LocVT == MVT::f32) {
    unsigned Offset4 = State.AllocateStack(4, Align(4));
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset4, LocVT, LocInfo));
    return false;
  }

  if (LocVT == MVT::i64 || LocVT == MVT::f64) {
    unsigned Offset5 = State.AllocateStack(8, Align(8));
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset5, LocVT, LocInfo));
    return false;
  }

  return true; // CC didn't match.
}

// Transform physical registers into virtual registers.
SDValue RISCVTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {

  switch (CallConv) {
  default:
    report_fatal_error("Unsupported calling convention");
  case CallingConv::C:
  case CallingConv::Fast:
    break;
  }

  MachineFunction &MF = DAG.getMachineFunction();

  const Function &Func = MF.getFunction();
  if (Func.hasFnAttribute("interrupt")) {
    if (!Func.arg_empty())
      report_fatal_error(
        "Functions with the interrupt attribute cannot have arguments!");

    StringRef Kind =
      MF.getFunction().getFnAttribute("interrupt").getValueAsString();

    if (!(Kind == "user" || Kind == "supervisor" || Kind == "machine"))
      report_fatal_error(
        "Function interrupt attribute argument not supported!");
  }

  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  MVT XLenVT = Subtarget.getXLenVT();
  unsigned XLenInBytes = Subtarget.getXLen() / 8;
  // Used with vargs to acumulate store chains.
  std::vector<SDValue> OutChains;

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  // We do not want to use fastcc when returning scalable vectors because
  // we want to have the CC for them in a single place in the code for now.
  bool HasInsScalableVectors =
      std::any_of(Ins.begin(), Ins.end(),
                  [](ISD::InputArg In) { return In.VT.isScalableVector(); });

  if (CallConv == CallingConv::Fast && !HasInsScalableVectors)
    CCInfo.AnalyzeFormalArguments(Ins, CC_RISCV_FastCC);
  else
    analyzeInputArgs(MF, CCInfo, Ins, /*IsRet=*/false);

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue ArgValue;
    // Passing f64 on RV32D with a soft float ABI must be handled as a special
    // case.
    if (VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64)
      ArgValue = unpackF64OnRV32DSoftABI(DAG, Chain, VA, DL);
    else if (VA.isRegLoc())
      ArgValue = unpackFromRegLoc(DAG, Chain, VA, DL, this);
    else
      ArgValue = unpackFromMemLoc(DAG, Chain, VA, DL);

    if (VA.getLocInfo() == CCValAssign::Indirect) {
      // If the original argument was split and passed by reference (e.g. i128
      // on RV32), we need to load all parts of it here (using the same
      // address).
      InVals.push_back(DAG.getLoad(VA.getValVT(), DL, Chain, ArgValue,
                                   MachinePointerInfo()));
      unsigned ArgIndex = Ins[i].OrigArgIndex;
      assert(Ins[i].PartOffset == 0);
      while (i + 1 != e && Ins[i + 1].OrigArgIndex == ArgIndex) {
        CCValAssign &PartVA = ArgLocs[i + 1];
        unsigned PartOffset = Ins[i + 1].PartOffset;
        SDValue Address = DAG.getNode(ISD::ADD, DL, PtrVT, ArgValue,
                                      DAG.getIntPtrConstant(PartOffset, DL));
        InVals.push_back(DAG.getLoad(PartVA.getValVT(), DL, Chain, Address,
                                     MachinePointerInfo()));
        ++i;
      }
      continue;
    }
    InVals.push_back(ArgValue);
  }

  if (IsVarArg) {
    ArrayRef<MCPhysReg> ArgRegs = makeArrayRef(ArgGPRs);
    unsigned Idx = CCInfo.getFirstUnallocated(ArgRegs);
    const TargetRegisterClass *RC = &RISCV::GPRRegClass;
    MachineFrameInfo &MFI = MF.getFrameInfo();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();
    RISCVMachineFunctionInfo *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();

    // Offset of the first variable argument from stack pointer, and size of
    // the vararg save area. For now, the varargs save area is either zero or
    // large enough to hold a0-a7.
    int VaArgOffset, VarArgsSaveSize;

    // If all registers are allocated, then all varargs must be passed on the
    // stack and we don't need to save any argregs.
    if (ArgRegs.size() == Idx) {
      VaArgOffset = CCInfo.getNextStackOffset();
      VarArgsSaveSize = 0;
    } else {
      VarArgsSaveSize = XLenInBytes * (ArgRegs.size() - Idx);
      VaArgOffset = -VarArgsSaveSize;
    }

    // Record the frame index of the first variable argument
    // which is a value necessary to VASTART.
    int FI = MFI.CreateFixedObject(XLenInBytes, VaArgOffset, true);
    RVFI->setVarArgsFrameIndex(FI);

    // If saving an odd number of registers then create an extra stack slot to
    // ensure that the frame pointer is 2*XLEN-aligned, which in turn ensures
    // offsets to even-numbered registered remain 2*XLEN-aligned.
    if (Idx % 2) {
      MFI.CreateFixedObject(XLenInBytes, VaArgOffset - (int)XLenInBytes, true);
      VarArgsSaveSize += XLenInBytes;
    }

    // Copy the integer registers that may have been used for passing varargs
    // to the vararg save area.
    for (unsigned I = Idx; I < ArgRegs.size();
         ++I, VaArgOffset += XLenInBytes) {
      const Register Reg = RegInfo.createVirtualRegister(RC);
      RegInfo.addLiveIn(ArgRegs[I], Reg);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, XLenVT);
      FI = MFI.CreateFixedObject(XLenInBytes, VaArgOffset, true);
      SDValue PtrOff = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      SDValue Store = DAG.getStore(Chain, DL, ArgValue, PtrOff,
                                   MachinePointerInfo::getFixedStack(MF, FI));
      cast<StoreSDNode>(Store.getNode())
          ->getMemOperand()
          ->setValue((Value *)nullptr);
      OutChains.push_back(Store);
    }
    RVFI->setVarArgsSaveSize(VarArgsSaveSize);
  }

  // All stores are grouped in one node to allow the matching between
  // the size of Ins and InVals. This only happens for vararg functions.
  if (!OutChains.empty()) {
    OutChains.push_back(Chain);
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, OutChains);
  }

  return Chain;
}

/// isEligibleForTailCallOptimization - Check whether the call is eligible
/// for tail call optimization.
/// Note: This is modelled after ARM's IsEligibleForTailCallOptimization.
bool RISCVTargetLowering::isEligibleForTailCallOptimization(
    CCState &CCInfo, CallLoweringInfo &CLI, MachineFunction &MF,
    const SmallVector<CCValAssign, 16> &ArgLocs) const {

  auto &Callee = CLI.Callee;
  auto CalleeCC = CLI.CallConv;
  auto &Outs = CLI.Outs;
  auto &Caller = MF.getFunction();
  auto CallerCC = Caller.getCallingConv();

  // Exception-handling functions need a special set of instructions to
  // indicate a return to the hardware. Tail-calling another function would
  // probably break this.
  // TODO: The "interrupt" attribute isn't currently defined by RISC-V. This
  // should be expanded as new function attributes are introduced.
  if (Caller.hasFnAttribute("interrupt"))
    return false;

  // Do not tail call opt if the stack is used to pass parameters.
  if (CCInfo.getNextStackOffset() != 0)
    return false;

  // Do not tail call opt if any parameters need to be passed indirectly.
  // Since long doubles (fp128) and i128 are larger than 2*XLEN, they are
  // passed indirectly. So the address of the value will be passed in a
  // register, or if not available, then the address is put on the stack. In
  // order to pass indirectly, space on the stack often needs to be allocated
  // in order to store the value. In this case the CCInfo.getNextStackOffset()
  // != 0 check is not enough and we need to check if any CCValAssign ArgsLocs
  // are passed CCValAssign::Indirect.
  for (auto &VA : ArgLocs)
    if (VA.getLocInfo() == CCValAssign::Indirect)
      return false;

  // Do not tail call opt if either caller or callee uses struct return
  // semantics.
  auto IsCallerStructRet = Caller.hasStructRetAttr();
  auto IsCalleeStructRet = Outs.empty() ? false : Outs[0].Flags.isSRet();
  if (IsCallerStructRet || IsCalleeStructRet)
    return false;

  // Externally-defined functions with weak linkage should not be
  // tail-called. The behaviour of branch instructions in this situation (as
  // used for tail calls) is implementation-defined, so we cannot rely on the
  // linker replacing the tail call with a return.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = G->getGlobal();
    if (GV->hasExternalWeakLinkage())
      return false;
  }

  // The callee has to preserve all registers the caller needs to preserve.
  const RISCVRegisterInfo *TRI = Subtarget.getRegisterInfo();
  const uint32_t *CallerPreserved = TRI->getCallPreservedMask(MF, CallerCC);
  if (CalleeCC != CallerCC) {
    const uint32_t *CalleePreserved = TRI->getCallPreservedMask(MF, CalleeCC);
    if (!TRI->regmaskSubsetEqual(CallerPreserved, CalleePreserved))
      return false;
  }

  // Byval parameters hand the function a pointer directly into the stack area
  // we want to reuse during a tail call. Working around this *is* possible
  // but less efficient and uglier in LowerCall.
  for (auto &Arg : Outs)
    if (Arg.Flags.isByVal())
      return false;

  return true;
}

// Lower a call to a callseq_start + CALL + callseq_end chain, and add input
// and output parameter nodes.
SDValue RISCVTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                       SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &DL = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  MVT XLenVT = Subtarget.getXLenVT();

  MachineFunction &MF = DAG.getMachineFunction();

  // Analyze the operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState ArgCCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  // We do not want to use fastcc when returning scalable vectors because
  // we want to have the CC for them in a single place in the code for now.
  bool ReturningScalableVectors =
      std::any_of(Outs.begin(), Outs.end(),
                  [](ISD::OutputArg Out) { return Out.VT.isScalableVector(); });

  if (CallConv == CallingConv::Fast && !ReturningScalableVectors)
    ArgCCInfo.AnalyzeCallOperands(Outs, CC_RISCV_FastCC);
  else
    analyzeOutputArgs(MF, ArgCCInfo, Outs, /*IsRet=*/false, &CLI);

  // Check if it's really possible to do a tail call.
  if (IsTailCall)
    IsTailCall = isEligibleForTailCallOptimization(ArgCCInfo, CLI, MF, ArgLocs);

  if (IsTailCall)
    ++NumTailCalls;
  else if (CLI.CB && CLI.CB->isMustTailCall())
    report_fatal_error("failed to perform tail call elimination on a call "
                       "site marked musttail");

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = ArgCCInfo.getNextStackOffset();

  // Create local copies for byval args
  SmallVector<SDValue, 8> ByValArgs;
  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    ISD::ArgFlagsTy Flags = Outs[i].Flags;
    if (!Flags.isByVal())
      continue;

    SDValue Arg = OutVals[i];
    unsigned Size = Flags.getByValSize();
    Align Alignment = Flags.getNonZeroByValAlign();

    int FI =
        MF.getFrameInfo().CreateStackObject(Size, Alignment, /*isSS=*/false);
    SDValue FIPtr = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
    SDValue SizeNode = DAG.getConstant(Size, DL, XLenVT);

    Chain = DAG.getMemcpy(Chain, DL, FIPtr, Arg, SizeNode, Alignment,
                          /*IsVolatile=*/false,
                          /*AlwaysInline=*/false, IsTailCall,
                          MachinePointerInfo(), MachinePointerInfo());
    ByValArgs.push_back(FIPtr);
  }

  if (!IsTailCall)
    Chain = DAG.getCALLSEQ_START(Chain, NumBytes, 0, CLI.DL);

  // Copy argument values to their designated locations.
  SmallVector<std::pair<Register, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  SDValue StackPtr;
  for (unsigned i = 0, j = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue ArgValue = OutVals[i];
    ISD::ArgFlagsTy Flags = Outs[i].Flags;

    // Handle passing f64 on RV32D with a soft float or hard float single ABI as
    // a special case.
    bool IsF64OnRV32DSoftABI =
        VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64;
    if (IsF64OnRV32DSoftABI && VA.isRegLoc()) {
      SDValue SplitF64 = DAG.getNode(
          RISCVISD::SplitF64, DL, DAG.getVTList(MVT::i32, MVT::i32), ArgValue);
      SDValue Lo = SplitF64.getValue(0);
      SDValue Hi = SplitF64.getValue(1);

      Register RegLo = VA.getLocReg();
      RegsToPass.push_back(std::make_pair(RegLo, Lo));

      if (RegLo == RISCV::X17) {
        // Second half of f64 is passed on the stack.
        // Work out the address of the stack slot.
        if (!StackPtr.getNode())
          StackPtr = DAG.getCopyFromReg(Chain, DL, RISCV::X2, PtrVT);
        // Emit the store.
        MemOpChains.push_back(
            DAG.getStore(Chain, DL, Hi, StackPtr, MachinePointerInfo()));
      } else {
        // Second half of f64 is passed in another GPR.
        assert(RegLo < RISCV::X31 && "Invalid register pair");
        Register RegHigh = RegLo + 1;
        RegsToPass.push_back(std::make_pair(RegHigh, Hi));
      }
      continue;
    }

    // IsF64OnRV32DSoftABI && VA.isMemLoc() is handled below in the same way
    // as any other MemLoc.

    // Promote the value if needed.
    // For now, only handle fully promoted and indirect arguments.
    if (VA.getLocInfo() == CCValAssign::Indirect) {
      if (VA.getValVT().isScalableVector()) {
        // Create a stack slot for the EPI register.
        SDValue SpillSlot = DAG.CreateStackTemporary(Outs[i].ArgVT);
        int FI = cast<FrameIndexSDNode>(SpillSlot)->getIndex();

        RISCVMachineFunctionInfo *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
        // Let know FrameLowering that we're spilling vector registers.
        RVFI->setHasSpilledVR();
        // Mark this spill as a vector spill.
        MF.getFrameInfo().setStackID(FI, TargetStackID::EPIVector);

        // We load an XLenVT from the spill slot because RISCVFrameLowering.cpp
        // will replace this slot from a vector type to an XLenVT.
        SDValue Ptr = DAG.getLoad(XLenVT, DL, Chain, SpillSlot,
                                  MachinePointerInfo::getFixedStack(MF, FI));

        MemOpChains.push_back(
            DAG.getStore(Chain, DL, ArgValue, Ptr, MachinePointerInfo()));
        ArgValue = Ptr;
      } else {
        // Store the argument in a stack slot and pass its address.
        SDValue SpillSlot = DAG.CreateStackTemporary(Outs[i].ArgVT);
        int FI = cast<FrameIndexSDNode>(SpillSlot)->getIndex();
        MemOpChains.push_back(
            DAG.getStore(Chain, DL, ArgValue, SpillSlot,
                         MachinePointerInfo::getFixedStack(MF, FI)));
        // If the original argument was split (e.g. i128), we need
        // to store all parts of it here (and pass just one address).
        unsigned ArgIndex = Outs[i].OrigArgIndex;
        assert(Outs[i].PartOffset == 0);
        while (i + 1 != e && Outs[i + 1].OrigArgIndex == ArgIndex) {
          SDValue PartValue = OutVals[i + 1];
          unsigned PartOffset = Outs[i + 1].PartOffset;
          SDValue Address = DAG.getNode(ISD::ADD, DL, PtrVT, SpillSlot,
                                        DAG.getIntPtrConstant(PartOffset, DL));
          MemOpChains.push_back(
              DAG.getStore(Chain, DL, PartValue, Address,
                           MachinePointerInfo::getFixedStack(MF, FI)));
          ++i;
        }
        ArgValue = SpillSlot;
      }
    } else {
      ArgValue = convertValVTToLocVT(DAG, ArgValue, VA, DL);
    }

    // Use local copy if it is a byval arg.
    if (Flags.isByVal())
      ArgValue = ByValArgs[j++];

    if (VA.isRegLoc()) {
      // Queue up the argument copies and emit them at the end.
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), ArgValue));
    } else {
      assert(VA.isMemLoc() && "Argument not register or memory");
      assert(!IsTailCall && "Tail call not allowed if stack is used "
                            "for passing parameters");

      // Work out the address of the stack slot.
      if (!StackPtr.getNode())
        StackPtr = DAG.getCopyFromReg(Chain, DL, RISCV::X2, PtrVT);
      SDValue Address =
          DAG.getNode(ISD::ADD, DL, PtrVT, StackPtr,
                      DAG.getIntPtrConstant(VA.getLocMemOffset(), DL));

      // Emit the store.
      MemOpChains.push_back(
          DAG.getStore(Chain, DL, ArgValue, Address, MachinePointerInfo()));
    }
  }

  // Join the stores, which are independent of one another.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

  SDValue Glue;

  // Build a sequence of copy-to-reg nodes, chained and glued together.
  for (auto &Reg : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, DL, Reg.first, Reg.second, Glue);
    Glue = Chain.getValue(1);
  }

  // Validate that none of the argument registers have been marked as
  // reserved, if so report an error. Do the same for the return address if this
  // is not a tailcall.
  validateCCReservedRegs(RegsToPass, MF);
  if (!IsTailCall &&
      MF.getSubtarget<RISCVSubtarget>().isRegisterReservedByUser(RISCV::X1))
    MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported{
        MF.getFunction(),
        "Return address register required, but has been reserved."});

  // If the callee is a GlobalAddress/ExternalSymbol node, turn it into a
  // TargetGlobalAddress/TargetExternalSymbol node so that legalize won't
  // split it and then direct call can be matched by PseudoCALL.
  if (GlobalAddressSDNode *S = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = S->getGlobal();

    unsigned OpFlags = RISCVII::MO_CALL;
    if (!getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV))
      OpFlags = RISCVII::MO_PLT;

    Callee = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, OpFlags);
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    unsigned OpFlags = RISCVII::MO_CALL;

    if (!getTargetMachine().shouldAssumeDSOLocal(*MF.getFunction().getParent(),
                                                 nullptr))
      OpFlags = RISCVII::MO_PLT;

    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), PtrVT, OpFlags);
  }

  // The first call operand is the chain and the second is the target address.
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (auto &Reg : RegsToPass)
    Ops.push_back(DAG.getRegister(Reg.first, Reg.second.getValueType()));

  if (!IsTailCall) {
    // Add a register mask operand representing the call-preserved registers.
    const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();
    const uint32_t *Mask = TRI->getCallPreservedMask(MF, CallConv);
    assert(Mask && "Missing call preserved mask for calling convention");
    Ops.push_back(DAG.getRegisterMask(Mask));
  }

  // Glue the call to the argument copies, if any.
  if (Glue.getNode())
    Ops.push_back(Glue);

  // Emit the call.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  if (IsTailCall) {
    MF.getFrameInfo().setHasTailCall();
    return DAG.getNode(RISCVISD::TAIL, DL, NodeTys, Ops);
  }

  Chain = DAG.getNode(RISCVISD::CALL, DL, NodeTys, Ops);
  DAG.addNoMergeSiteInfo(Chain.getNode(), CLI.NoMerge);
  Glue = Chain.getValue(1);

  // Mark the end of the call, which is glued to the call itself.
  Chain = DAG.getCALLSEQ_END(Chain,
                             DAG.getConstant(NumBytes, DL, PtrVT, true),
                             DAG.getConstant(0, DL, PtrVT, true),
                             Glue, DL);
  Glue = Chain.getValue(1);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, RVLocs, *DAG.getContext());
  analyzeInputArgs(MF, RetCCInfo, Ins, /*IsRet=*/true);

  // Copy all of the result registers out of their specified physreg.
  for (auto &VA : RVLocs) {
    // Copy the value out
    SDValue RetValue =
        DAG.getCopyFromReg(Chain, DL, VA.getLocReg(), VA.getLocVT(), Glue);
    // Glue the RetValue to the end of the call sequence
    Chain = RetValue.getValue(1);
    Glue = RetValue.getValue(2);

    if (VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64) {
      assert(VA.getLocReg() == ArgGPRs[0] && "Unexpected reg assignment");
      SDValue RetValue2 =
          DAG.getCopyFromReg(Chain, DL, ArgGPRs[1], MVT::i32, Glue);
      Chain = RetValue2.getValue(1);
      Glue = RetValue2.getValue(2);
      RetValue = DAG.getNode(RISCVISD::BuildPairF64, DL, MVT::f64, RetValue,
                             RetValue2);
    }

    RetValue = convertLocVTToValVT(DAG, RetValue, VA, DL);

    InVals.push_back(RetValue);
  }

  return Chain;
}

bool RISCVTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, Context);

  Optional<unsigned> FirstMaskArgument;
  if (Subtarget.hasStdExtV()) {
    PreAssignMask(Outs, FirstMaskArgument, CCInfo);
  }

  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    MVT VT = Outs[i].VT;
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    RISCVABI::ABI ABI = MF.getSubtarget<RISCVSubtarget>().getTargetABI();
    if (CC_RISCV(MF.getDataLayout(), ABI, i, VT, VT, CCValAssign::Full,
                 ArgFlags, CCInfo, /*IsFixed=*/true, /*IsRet=*/true, nullptr,
                 this, FirstMaskArgument))
      return false;
  }
  return true;
}

SDValue
RISCVTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                 bool IsVarArg,
                                 const SmallVectorImpl<ISD::OutputArg> &Outs,
                                 const SmallVectorImpl<SDValue> &OutVals,
                                 const SDLoc &DL, SelectionDAG &DAG) const {
  const MachineFunction &MF = DAG.getMachineFunction();
  const RISCVSubtarget &STI = MF.getSubtarget<RISCVSubtarget>();

  // Stores the assignment of the return value to a location.
  SmallVector<CCValAssign, 16> RVLocs;

  // Info about the registers and stack slot.
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  analyzeOutputArgs(DAG.getMachineFunction(), CCInfo, Outs, /*IsRet=*/true,
                    nullptr);

  SDValue Glue;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  // Copy the result values into the output registers.
  for (unsigned i = 0, e = RVLocs.size(); i < e; ++i) {
    SDValue Val = OutVals[i];
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    if (VA.getLocVT() == MVT::i32 && VA.getValVT() == MVT::f64) {
      // Handle returning f64 on RV32D with a soft float ABI.
      assert(VA.isRegLoc() && "Expected return via registers");
      SDValue SplitF64 = DAG.getNode(RISCVISD::SplitF64, DL,
                                     DAG.getVTList(MVT::i32, MVT::i32), Val);
      SDValue Lo = SplitF64.getValue(0);
      SDValue Hi = SplitF64.getValue(1);
      Register RegLo = VA.getLocReg();
      assert(RegLo < RISCV::X31 && "Invalid register pair");
      Register RegHi = RegLo + 1;

      if (STI.isRegisterReservedByUser(RegLo) ||
          STI.isRegisterReservedByUser(RegHi))
        MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported{
            MF.getFunction(),
            "Return value register required, but has been reserved."});

      Chain = DAG.getCopyToReg(Chain, DL, RegLo, Lo, Glue);
      Glue = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(RegLo, MVT::i32));
      Chain = DAG.getCopyToReg(Chain, DL, RegHi, Hi, Glue);
      Glue = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(RegHi, MVT::i32));
    } else {
      // Handle a 'normal' return.
      Val = convertValVTToLocVT(DAG, Val, VA, DL);
      Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Val, Glue);

      if (STI.isRegisterReservedByUser(VA.getLocReg()))
        MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported{
            MF.getFunction(),
            "Return value register required, but has been reserved."});

      // Guarantee that all emitted copies are stuck together.
      Glue = Chain.getValue(1);
      RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
    }
  }

  RetOps[0] = Chain; // Update chain.

  // Add the glue node if we have it.
  if (Glue.getNode()) {
    RetOps.push_back(Glue);
  }

  // Interrupt service routines use different return instructions.
  const Function &Func = DAG.getMachineFunction().getFunction();
  if (Func.hasFnAttribute("interrupt")) {
    if (!Func.getReturnType()->isVoidTy())
      report_fatal_error(
          "Functions with the interrupt attribute must have void return type!");

    MachineFunction &MF = DAG.getMachineFunction();
    StringRef Kind =
      MF.getFunction().getFnAttribute("interrupt").getValueAsString();

    unsigned RetOpc;
    if (Kind == "user")
      RetOpc = RISCVISD::URET_FLAG;
    else if (Kind == "supervisor")
      RetOpc = RISCVISD::SRET_FLAG;
    else
      RetOpc = RISCVISD::MRET_FLAG;

    return DAG.getNode(RetOpc, DL, MVT::Other, RetOps);
  }

  return DAG.getNode(RISCVISD::RET_FLAG, DL, MVT::Other, RetOps);
}

void RISCVTargetLowering::validateCCReservedRegs(
    const SmallVectorImpl<std::pair<llvm::Register, llvm::SDValue>> &Regs,
    MachineFunction &MF) const {
  const Function &F = MF.getFunction();
  const RISCVSubtarget &STI = MF.getSubtarget<RISCVSubtarget>();

  if (std::any_of(std::begin(Regs), std::end(Regs), [&STI](auto Reg) {
        return STI.isRegisterReservedByUser(Reg.first);
      }))
    F.getContext().diagnose(DiagnosticInfoUnsupported{
        F, "Argument register required, but has been reserved."});
}

bool RISCVTargetLowering::mayBeEmittedAsTailCall(const CallInst *CI) const {
  return CI->isTailCall();
}

const char *RISCVTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch ((RISCVISD::NodeType)Opcode) {
  case RISCVISD::FIRST_NUMBER:
    break;
  case RISCVISD::RET_FLAG:
    return "RISCVISD::RET_FLAG";
  case RISCVISD::URET_FLAG:
    return "RISCVISD::URET_FLAG";
  case RISCVISD::SRET_FLAG:
    return "RISCVISD::SRET_FLAG";
  case RISCVISD::MRET_FLAG:
    return "RISCVISD::MRET_FLAG";
  case RISCVISD::CALL:
    return "RISCVISD::CALL";
  case RISCVISD::SELECT_CC:
    return "RISCVISD::SELECT_CC";
  case RISCVISD::BuildPairF64:
    return "RISCVISD::BuildPairF64";
  case RISCVISD::SplitF64:
    return "RISCVISD::SplitF64";
  case RISCVISD::TAIL:
    return "RISCVISD::TAIL";
  case RISCVISD::SLLW:
    return "RISCVISD::SLLW";
  case RISCVISD::SRAW:
    return "RISCVISD::SRAW";
  case RISCVISD::SRLW:
    return "RISCVISD::SRLW";
  case RISCVISD::DIVW:
    return "RISCVISD::DIVW";
  case RISCVISD::DIVUW:
    return "RISCVISD::DIVUW";
  case RISCVISD::REMUW:
    return "RISCVISD::REMUW";
  case RISCVISD::FMV_W_X_RV64:
    return "RISCVISD::FMV_W_X_RV64";
  case RISCVISD::FMV_X_ANYEXTW_RV64:
    return "RISCVISD::FMV_X_ANYEXTW_RV64";
  case RISCVISD::READ_CYCLE_WIDE:
    return "RISCVISD::READ_CYCLE_WIDE";
  case RISCVISD::VMV_X_S:
    return "RISCVISD::VMV_X_S";
  case RISCVISD::EXTRACT_VECTOR_ELT:
    return "RISCVISD::EXTRACT_VECTOR_ELT";
  case RISCVISD::SIGN_EXTEND_VECTOR:
    return "RISCVISD::SIGN_EXTEND_VECTOR";
  case RISCVISD::ZERO_EXTEND_VECTOR:
    return "RISCVISD::ZERO_EXTEND_VECTOR";
  case RISCVISD::TRUNCATE_VECTOR:
    return "RISCVISD::TRUNCATE_VECTOR";
  case RISCVISD::SIGN_EXTEND_BITS_INREG:
    return "RISCVISD::SIGN_EXTEND_BITS_INREG";
#define TUPLE_NODE(X)  \
  case X##2: return #X"2"; \
  case X##3: return #X"3"; \
  case X##4: return #X"4"; \
  case X##5: return #X"5"; \
  case X##6: return #X"6"; \
  case X##7: return #X"7"; \
  case X##8: return #X"8";
  TUPLE_NODE(RISCVISD::VLSEG)
  TUPLE_NODE(RISCVISD::VSSEG)
  TUPLE_NODE(RISCVISD::VLSSEG)
  TUPLE_NODE(RISCVISD::VSSSEG)
  TUPLE_NODE(RISCVISD::VLXSEG)
  TUPLE_NODE(RISCVISD::VSXSEG)
#undef TUPLE_NODE
  case RISCVISD::VZIP2:
    return "RISCVISD::VZIP2";
  case RISCVISD::VUNZIP2:
    return "RISCVISD::VUNZIP2";
  case RISCVISD::VTRN:
    return "RISCVISD::VTRN";
  }
  return nullptr;
}

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
RISCVTargetLowering::ConstraintType
RISCVTargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:
      break;
    case 'f':
    case 'v':
      return C_RegisterClass;
    case 'I':
    case 'J':
    case 'K':
      return C_Immediate;
    case 'A':
      return C_Memory;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass *>
RISCVTargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                                  StringRef Constraint,
                                                  MVT VT) const {
  // First, see if this is a constraint that directly corresponds to a
  // RISCV register class.
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      return std::make_pair(0U, &RISCV::GPRRegClass);
    case 'f':
      if (Subtarget.hasStdExtF() && VT == MVT::f32)
        return std::make_pair(0U, &RISCV::FPR32RegClass);
      if (Subtarget.hasStdExtD() && VT == MVT::f64)
        return std::make_pair(0U, &RISCV::FPR64RegClass);
      break;
    case 'v':
      for (const auto *RC : {&RISCV::VRRegClass, &RISCV::VRM2RegClass,
                             &RISCV::VRM4RegClass, &RISCV::VRM8RegClass}) {
        if (TRI->isTypeLegalForClass(*RC, VT.SimpleTy))
          return std::make_pair(0U, RC);
      }
      break;
    default:
      break;
    }
  }

  // Clang will correctly decode the usage of register name aliases into their
  // official names. However, other frontends like `rustc` do not. This allows
  // users of these frontends to use the ABI names for registers in LLVM-style
  // register constraints.
  Register XRegFromAlias = StringSwitch<Register>(Constraint.lower())
                               .Case("{zero}", RISCV::X0)
                               .Case("{ra}", RISCV::X1)
                               .Case("{sp}", RISCV::X2)
                               .Case("{gp}", RISCV::X3)
                               .Case("{tp}", RISCV::X4)
                               .Case("{t0}", RISCV::X5)
                               .Case("{t1}", RISCV::X6)
                               .Case("{t2}", RISCV::X7)
                               .Cases("{s0}", "{fp}", RISCV::X8)
                               .Case("{s1}", RISCV::X9)
                               .Case("{a0}", RISCV::X10)
                               .Case("{a1}", RISCV::X11)
                               .Case("{a2}", RISCV::X12)
                               .Case("{a3}", RISCV::X13)
                               .Case("{a4}", RISCV::X14)
                               .Case("{a5}", RISCV::X15)
                               .Case("{a6}", RISCV::X16)
                               .Case("{a7}", RISCV::X17)
                               .Case("{s2}", RISCV::X18)
                               .Case("{s3}", RISCV::X19)
                               .Case("{s4}", RISCV::X20)
                               .Case("{s5}", RISCV::X21)
                               .Case("{s6}", RISCV::X22)
                               .Case("{s7}", RISCV::X23)
                               .Case("{s8}", RISCV::X24)
                               .Case("{s9}", RISCV::X25)
                               .Case("{s10}", RISCV::X26)
                               .Case("{s11}", RISCV::X27)
                               .Case("{t3}", RISCV::X28)
                               .Case("{t4}", RISCV::X29)
                               .Case("{t5}", RISCV::X30)
                               .Case("{t6}", RISCV::X31)
                               .Default(RISCV::NoRegister);
  if (XRegFromAlias != RISCV::NoRegister)
    return std::make_pair(XRegFromAlias, &RISCV::GPRRegClass);

  // Since TargetLowering::getRegForInlineAsmConstraint uses the name of the
  // TableGen record rather than the AsmName to choose registers for InlineAsm
  // constraints, plus we want to match those names to the widest floating point
  // register type available, manually select floating point registers here.
  //
  // The second case is the ABI name of the register, so that frontends can also
  // use the ABI names in register constraint lists.
  if (Subtarget.hasStdExtF() || Subtarget.hasStdExtD()) {
    std::pair<Register, Register> FReg =
        StringSwitch<std::pair<Register, Register>>(Constraint.lower())
            .Cases("{f0}", "{ft0}", {RISCV::F0_F, RISCV::F0_D})
            .Cases("{f1}", "{ft1}", {RISCV::F1_F, RISCV::F1_D})
            .Cases("{f2}", "{ft2}", {RISCV::F2_F, RISCV::F2_D})
            .Cases("{f3}", "{ft3}", {RISCV::F3_F, RISCV::F3_D})
            .Cases("{f4}", "{ft4}", {RISCV::F4_F, RISCV::F4_D})
            .Cases("{f5}", "{ft5}", {RISCV::F5_F, RISCV::F5_D})
            .Cases("{f6}", "{ft6}", {RISCV::F6_F, RISCV::F6_D})
            .Cases("{f7}", "{ft7}", {RISCV::F7_F, RISCV::F7_D})
            .Cases("{f8}", "{fs0}", {RISCV::F8_F, RISCV::F8_D})
            .Cases("{f9}", "{fs1}", {RISCV::F9_F, RISCV::F9_D})
            .Cases("{f10}", "{fa0}", {RISCV::F10_F, RISCV::F10_D})
            .Cases("{f11}", "{fa1}", {RISCV::F11_F, RISCV::F11_D})
            .Cases("{f12}", "{fa2}", {RISCV::F12_F, RISCV::F12_D})
            .Cases("{f13}", "{fa3}", {RISCV::F13_F, RISCV::F13_D})
            .Cases("{f14}", "{fa4}", {RISCV::F14_F, RISCV::F14_D})
            .Cases("{f15}", "{fa5}", {RISCV::F15_F, RISCV::F15_D})
            .Cases("{f16}", "{fa6}", {RISCV::F16_F, RISCV::F16_D})
            .Cases("{f17}", "{fa7}", {RISCV::F17_F, RISCV::F17_D})
            .Cases("{f18}", "{fs2}", {RISCV::F18_F, RISCV::F18_D})
            .Cases("{f19}", "{fs3}", {RISCV::F19_F, RISCV::F19_D})
            .Cases("{f20}", "{fs4}", {RISCV::F20_F, RISCV::F20_D})
            .Cases("{f21}", "{fs5}", {RISCV::F21_F, RISCV::F21_D})
            .Cases("{f22}", "{fs6}", {RISCV::F22_F, RISCV::F22_D})
            .Cases("{f23}", "{fs7}", {RISCV::F23_F, RISCV::F23_D})
            .Cases("{f24}", "{fs8}", {RISCV::F24_F, RISCV::F24_D})
            .Cases("{f25}", "{fs9}", {RISCV::F25_F, RISCV::F25_D})
            .Cases("{f26}", "{fs10}", {RISCV::F26_F, RISCV::F26_D})
            .Cases("{f27}", "{fs11}", {RISCV::F27_F, RISCV::F27_D})
            .Cases("{f28}", "{ft8}", {RISCV::F28_F, RISCV::F28_D})
            .Cases("{f29}", "{ft9}", {RISCV::F29_F, RISCV::F29_D})
            .Cases("{f30}", "{ft10}", {RISCV::F30_F, RISCV::F30_D})
            .Cases("{f31}", "{ft11}", {RISCV::F31_F, RISCV::F31_D})
            .Default({RISCV::NoRegister, RISCV::NoRegister});
    if (FReg.first != RISCV::NoRegister)
      return Subtarget.hasStdExtD()
                 ? std::make_pair(FReg.second, &RISCV::FPR64RegClass)
                 : std::make_pair(FReg.first, &RISCV::FPR32RegClass);
  }

  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

unsigned
RISCVTargetLowering::getInlineAsmMemConstraint(StringRef ConstraintCode) const {
  // Currently only support length 1 constraints.
  if (ConstraintCode.size() == 1) {
    switch (ConstraintCode[0]) {
    case 'A':
      return InlineAsm::Constraint_A;
    default:
      break;
    }
  }

  return TargetLowering::getInlineAsmMemConstraint(ConstraintCode);
}

void RISCVTargetLowering::LowerAsmOperandForConstraint(
    SDValue Op, std::string &Constraint, std::vector<SDValue> &Ops,
    SelectionDAG &DAG) const {
  // Currently only support length 1 constraints.
  if (Constraint.length() == 1) {
    switch (Constraint[0]) {
    case 'I':
      // Validate & create a 12-bit signed immediate operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op)) {
        uint64_t CVal = C->getSExtValue();
        if (isInt<12>(CVal))
          Ops.push_back(
              DAG.getTargetConstant(CVal, SDLoc(Op), Subtarget.getXLenVT()));
      }
      return;
    case 'J':
      // Validate & create an integer zero operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op))
        if (C->getZExtValue() == 0)
          Ops.push_back(
              DAG.getTargetConstant(0, SDLoc(Op), Subtarget.getXLenVT()));
      return;
    case 'K':
      // Validate & create a 5-bit unsigned immediate operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op)) {
        uint64_t CVal = C->getZExtValue();
        if (isUInt<5>(CVal))
          Ops.push_back(
              DAG.getTargetConstant(CVal, SDLoc(Op), Subtarget.getXLenVT()));
      }
      return;
    default:
      break;
    }
  }
  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

Instruction *RISCVTargetLowering::emitLeadingFence(IRBuilder<> &Builder,
                                                   Instruction *Inst,
                                                   AtomicOrdering Ord) const {
  if (isa<LoadInst>(Inst) && Ord == AtomicOrdering::SequentiallyConsistent)
    return Builder.CreateFence(Ord);
  if (isa<StoreInst>(Inst) && isReleaseOrStronger(Ord))
    return Builder.CreateFence(AtomicOrdering::Release);
  return nullptr;
}

Instruction *RISCVTargetLowering::emitTrailingFence(IRBuilder<> &Builder,
                                                    Instruction *Inst,
                                                    AtomicOrdering Ord) const {
  if (isa<LoadInst>(Inst) && isAcquireOrStronger(Ord))
    return Builder.CreateFence(AtomicOrdering::Acquire);
  return nullptr;
}

TargetLowering::AtomicExpansionKind
RISCVTargetLowering::shouldExpandAtomicRMWInIR(AtomicRMWInst *AI) const {
  // atomicrmw {fadd,fsub} must be expanded to use compare-exchange, as floating
  // point operations can't be used in an lr/sc sequence without breaking the
  // forward-progress guarantee.
  if (AI->isFloatingPointOperation())
    return AtomicExpansionKind::CmpXChg;

  unsigned Size = AI->getType()->getPrimitiveSizeInBits();
  if (Size == 8 || Size == 16)
    return AtomicExpansionKind::MaskedIntrinsic;
  return AtomicExpansionKind::None;
}

static Intrinsic::ID
getIntrinsicForMaskedAtomicRMWBinOp(unsigned XLen, AtomicRMWInst::BinOp BinOp) {
  if (XLen == 32) {
    switch (BinOp) {
    default:
      llvm_unreachable("Unexpected AtomicRMW BinOp");
    case AtomicRMWInst::Xchg:
      return Intrinsic::riscv_masked_atomicrmw_xchg_i32;
    case AtomicRMWInst::Add:
      return Intrinsic::riscv_masked_atomicrmw_add_i32;
    case AtomicRMWInst::Sub:
      return Intrinsic::riscv_masked_atomicrmw_sub_i32;
    case AtomicRMWInst::Nand:
      return Intrinsic::riscv_masked_atomicrmw_nand_i32;
    case AtomicRMWInst::Max:
      return Intrinsic::riscv_masked_atomicrmw_max_i32;
    case AtomicRMWInst::Min:
      return Intrinsic::riscv_masked_atomicrmw_min_i32;
    case AtomicRMWInst::UMax:
      return Intrinsic::riscv_masked_atomicrmw_umax_i32;
    case AtomicRMWInst::UMin:
      return Intrinsic::riscv_masked_atomicrmw_umin_i32;
    }
  }

  if (XLen == 64) {
    switch (BinOp) {
    default:
      llvm_unreachable("Unexpected AtomicRMW BinOp");
    case AtomicRMWInst::Xchg:
      return Intrinsic::riscv_masked_atomicrmw_xchg_i64;
    case AtomicRMWInst::Add:
      return Intrinsic::riscv_masked_atomicrmw_add_i64;
    case AtomicRMWInst::Sub:
      return Intrinsic::riscv_masked_atomicrmw_sub_i64;
    case AtomicRMWInst::Nand:
      return Intrinsic::riscv_masked_atomicrmw_nand_i64;
    case AtomicRMWInst::Max:
      return Intrinsic::riscv_masked_atomicrmw_max_i64;
    case AtomicRMWInst::Min:
      return Intrinsic::riscv_masked_atomicrmw_min_i64;
    case AtomicRMWInst::UMax:
      return Intrinsic::riscv_masked_atomicrmw_umax_i64;
    case AtomicRMWInst::UMin:
      return Intrinsic::riscv_masked_atomicrmw_umin_i64;
    }
  }

  llvm_unreachable("Unexpected XLen\n");
}

Value *RISCVTargetLowering::emitMaskedAtomicRMWIntrinsic(
    IRBuilder<> &Builder, AtomicRMWInst *AI, Value *AlignedAddr, Value *Incr,
    Value *Mask, Value *ShiftAmt, AtomicOrdering Ord) const {
  unsigned XLen = Subtarget.getXLen();
  Value *Ordering =
      Builder.getIntN(XLen, static_cast<uint64_t>(AI->getOrdering()));
  Type *Tys[] = {AlignedAddr->getType()};
  Function *LrwOpScwLoop = Intrinsic::getDeclaration(
      AI->getModule(),
      getIntrinsicForMaskedAtomicRMWBinOp(XLen, AI->getOperation()), Tys);

  if (XLen == 64) {
    Incr = Builder.CreateSExt(Incr, Builder.getInt64Ty());
    Mask = Builder.CreateSExt(Mask, Builder.getInt64Ty());
    ShiftAmt = Builder.CreateSExt(ShiftAmt, Builder.getInt64Ty());
  }

  Value *Result;

  // Must pass the shift amount needed to sign extend the loaded value prior
  // to performing a signed comparison for min/max. ShiftAmt is the number of
  // bits to shift the value into position. Pass XLen-ShiftAmt-ValWidth, which
  // is the number of bits to left+right shift the value in order to
  // sign-extend.
  if (AI->getOperation() == AtomicRMWInst::Min ||
      AI->getOperation() == AtomicRMWInst::Max) {
    const DataLayout &DL = AI->getModule()->getDataLayout();
    unsigned ValWidth =
        DL.getTypeStoreSizeInBits(AI->getValOperand()->getType());
    Value *SextShamt =
        Builder.CreateSub(Builder.getIntN(XLen, XLen - ValWidth), ShiftAmt);
    Result = Builder.CreateCall(LrwOpScwLoop,
                                {AlignedAddr, Incr, Mask, SextShamt, Ordering});
  } else {
    Result =
        Builder.CreateCall(LrwOpScwLoop, {AlignedAddr, Incr, Mask, Ordering});
  }

  if (XLen == 64)
    Result = Builder.CreateTrunc(Result, Builder.getInt32Ty());
  return Result;
}

TargetLowering::AtomicExpansionKind
RISCVTargetLowering::shouldExpandAtomicCmpXchgInIR(
    AtomicCmpXchgInst *CI) const {
  unsigned Size = CI->getCompareOperand()->getType()->getPrimitiveSizeInBits();
  if (Size == 8 || Size == 16)
    return AtomicExpansionKind::MaskedIntrinsic;
  return AtomicExpansionKind::None;
}

Value *RISCVTargetLowering::emitMaskedAtomicCmpXchgIntrinsic(
    IRBuilder<> &Builder, AtomicCmpXchgInst *CI, Value *AlignedAddr,
    Value *CmpVal, Value *NewVal, Value *Mask, AtomicOrdering Ord) const {
  unsigned XLen = Subtarget.getXLen();
  Value *Ordering = Builder.getIntN(XLen, static_cast<uint64_t>(Ord));
  Intrinsic::ID CmpXchgIntrID = Intrinsic::riscv_masked_cmpxchg_i32;
  if (XLen == 64) {
    CmpVal = Builder.CreateSExt(CmpVal, Builder.getInt64Ty());
    NewVal = Builder.CreateSExt(NewVal, Builder.getInt64Ty());
    Mask = Builder.CreateSExt(Mask, Builder.getInt64Ty());
    CmpXchgIntrID = Intrinsic::riscv_masked_cmpxchg_i64;
  }
  Type *Tys[] = {AlignedAddr->getType()};
  Function *MaskedCmpXchg =
      Intrinsic::getDeclaration(CI->getModule(), CmpXchgIntrID, Tys);
  Value *Result = Builder.CreateCall(
      MaskedCmpXchg, {AlignedAddr, CmpVal, NewVal, Mask, Ordering});
  if (XLen == 64)
    Result = Builder.CreateTrunc(Result, Builder.getInt32Ty());
  return Result;
}

bool RISCVTargetLowering::isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                                     EVT VT) const {
  VT = VT.getScalarType();

  if (!VT.isSimple())
    return false;

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f32:
  case MVT::f64:
    return true;
  default:
    break;
  }

  return false;
}

Register RISCVTargetLowering::getExceptionPointerRegister(
    const Constant *PersonalityFn) const {
  return RISCV::X10;
}

Register RISCVTargetLowering::getExceptionSelectorRegister(
    const Constant *PersonalityFn) const {
  return RISCV::X11;
}

bool RISCVTargetLowering::allowsMisalignedMemoryAccesses(
    EVT E, unsigned AddrSpace, unsigned Align, MachineMemOperand::Flags Flags,
    bool *Fast) const {
  if (!E.isScalableVector())
    return false;

  // Scalable vectors enforce only the alignment of the element type.
  // There is no reason to think these should be any slower.
  if (Fast)
    *Fast = true;

  EVT ElementType = E.getVectorElementType();
  return Align >= ElementType.getStoreSize();
}

bool RISCVTargetLowering::shouldExtendTypeInLibCall(EVT Type) const {
  // Return false to suppress the unnecessary extensions if the LibCall
  // arguments or return value is f32 type for LP64 ABI.
  RISCVABI::ABI ABI = Subtarget.getTargetABI();
  if (ABI == RISCVABI::ABI_LP64 && (Type == MVT::f32))
    return false;

  return true;
}

bool RISCVTargetLowering::decomposeMulByConstant(LLVMContext &Context, EVT VT,
                                                 SDValue C) const {
  // Check integral scalar types.
  if (VT.isScalarInteger()) {
    // Do not perform the transformation on riscv32 with the M extension.
    if (!Subtarget.is64Bit() && Subtarget.hasStdExtM())
      return false;
    if (auto *ConstNode = dyn_cast<ConstantSDNode>(C.getNode())) {
      if (ConstNode->getAPIntValue().getBitWidth() > 8 * sizeof(int64_t))
        return false;
      int64_t Imm = ConstNode->getSExtValue();
      if (isPowerOf2_64(Imm + 1) || isPowerOf2_64(Imm - 1) ||
          isPowerOf2_64(1 - Imm) || isPowerOf2_64(-1 - Imm))
        return true;
    }
  }

  return false;
}

#define GET_REGISTER_MATCHER
#include "RISCVGenAsmMatcher.inc"

Register
RISCVTargetLowering::getRegisterByName(const char *RegName, LLT VT,
                                       const MachineFunction &MF) const {
  Register Reg = MatchRegisterAltName(RegName);
  if (Reg == RISCV::NoRegister)
    Reg = MatchRegisterName(RegName);
  if (Reg == RISCV::NoRegister)
    report_fatal_error(
        Twine("Invalid register name \"" + StringRef(RegName) + "\"."));
  BitVector ReservedRegs = Subtarget.getRegisterInfo()->getReservedRegs(MF);
  if (!ReservedRegs.test(Reg) && !Subtarget.isRegisterReservedByUser(Reg))
    report_fatal_error(Twine("Trying to obtain non-reserved register \"" +
                             StringRef(RegName) + "\"."));
  return Reg;
}

bool RISCVTargetLowering::shouldSinkOperands(
    Instruction *I, SmallVectorImpl<Use *> &Ops) const {
  if (!isa<ScalableVectorType>(I->getType()))
    return false;

  // Sinking broadcasts is always beneficial because it avoids keeping
  // vector registers alive and often they can be folded into the operand.
  for (unsigned OpI = 0, E = I->getNumOperands(); OpI < E; OpI++) {
    Use &U = I->getOperandUse(OpI);
    if (auto *SI = dyn_cast<ShuffleVectorInst>(&U)) {
      if (SI->isZeroEltSplat()) {
        Ops.push_back(&SI->getOperandUse(0));
        Ops.push_back(&U);
      }
    } else if (auto *II = dyn_cast<IntrinsicInst>(&U)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::epi_vmv_v_x:
      case Intrinsic::epi_vfmv_v_f: {
        Ops.push_back(&U);
        break;
      }
      default:
        break;
      }
    }
    if (!Ops.empty())
      return true;
  }

  return false;
}
