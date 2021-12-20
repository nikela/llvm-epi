//===-- RISCVTargetTransformInfo.cpp - RISC-V specific TTI ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVTargetTransformInfo.h"
#include "MCTargetDesc/RISCVMatInt.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "riscvtti"

InstructionCost RISCVTTIImpl::getIntImmCost(const APInt &Imm, Type *Ty,
                                            TTI::TargetCostKind CostKind) {
  assert(Ty->isIntegerTy() &&
         "getIntImmCost can only estimate cost of materialising integers");

  // We have a Zero register, so 0 is always free.
  if (Imm == 0)
    return TTI::TCC_Free;

  // Otherwise, we check how many instructions it will take to materialise.
  const DataLayout &DL = getDataLayout();
  return RISCVMatInt::getIntMatCost(Imm, DL.getTypeSizeInBits(Ty),
                                    getST()->getFeatureBits());
}

InstructionCost RISCVTTIImpl::getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                                const APInt &Imm, Type *Ty,
                                                TTI::TargetCostKind CostKind,
                                                Instruction *Inst) {
  assert(Ty->isIntegerTy() &&
         "getIntImmCost can only estimate cost of materialising integers");

  // We have a Zero register, so 0 is always free.
  if (Imm == 0)
    return TTI::TCC_Free;

  // Some instructions in RISC-V can take a 12-bit immediate. Some of these are
  // commutative, in others the immediate comes from a specific argument index.
  bool Takes12BitImm = false;
  unsigned ImmArgIdx = ~0U;

  switch (Opcode) {
  case Instruction::GetElementPtr:
    // Never hoist any arguments to a GetElementPtr. CodeGenPrepare will
    // split up large offsets in GEP into better parts than ConstantHoisting
    // can.
    return TTI::TCC_Free;
  case Instruction::And:
    // zext.h
    if (Imm == UINT64_C(0xffff) && ST->hasStdExtZbb())
      return TTI::TCC_Free;
    // zext.w
    if (Imm == UINT64_C(0xffffffff) && ST->hasStdExtZbb())
      return TTI::TCC_Free;
    LLVM_FALLTHROUGH;
  case Instruction::Add:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Mul:
    Takes12BitImm = true;
    break;
  case Instruction::Sub:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    Takes12BitImm = true;
    ImmArgIdx = 1;
    break;
  default:
    break;
  }

  if (Takes12BitImm) {
    // Check immediate is the correct argument...
    if (Instruction::isCommutative(Opcode) || Idx == ImmArgIdx) {
      // ... and fits into the 12-bit immediate.
      if (Imm.getMinSignedBits() <= 64 &&
          getTLI()->isLegalAddImmediate(Imm.getSExtValue())) {
        return TTI::TCC_Free;
      }
    }

    // Otherwise, use the full materialisation cost.
    return getIntImmCost(Imm, Ty, CostKind);
  }

  // By default, prevent hoisting.
  return TTI::TCC_Free;
}

InstructionCost
RISCVTTIImpl::getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx,
                                  const APInt &Imm, Type *Ty,
                                  TTI::TargetCostKind CostKind) {
  // Prevent hoisting in unknown cases.
  return TTI::TCC_Free;
}

unsigned RISCVTTIImpl::getNumberOfRegisters(unsigned ClassID) const {
  if (ClassID == 1 && ST->hasStdExtV())
    // Although there are 32 vector registers, v0 is special in that it is the
    // only register that can be used to hold a mask. We conservatively return
    // 31 as the number of usable vector registers.
    return 31;
  else if (ClassID == 0)
    // Similarly for scalar registers, x0(zero), x1(ra) and x2(sp) are special
    // and we return 29 usable registers.
    return 29;
  else
    return 0;
}

unsigned RISCVTTIImpl::getMaxElementWidth() const {
  // Returns ELEN. This is the value for which k-scale-factor would be one.
  // Current EPI implementation plans this to be 64. 
  return 64;
}

bool RISCVTTIImpl::preferPredicatedVectorOps() const {
  return ST->hasEPI();
}

bool RISCVTTIImpl::isLegalMaskedLoadStore(Type *DataType) const {
  if (!ST->hasStdExtV())
    return false;
  Type *ScalarTy = DataType->getScalarType();
  return (ScalarTy->isFloatTy() || ScalarTy->isDoubleTy() ||
          ScalarTy->isIntegerTy(8) || ScalarTy->isIntegerTy(16) ||
          ScalarTy->isIntegerTy(32) || ScalarTy->isIntegerTy(64));
}

bool RISCVTTIImpl::isLegalMaskedLoad(Type *DataType,
                                     MaybeAlign Alignment) const {
  return isLegalMaskedLoadStore(DataType);
}

bool RISCVTTIImpl::isLegalMaskedStore(Type *DataType,
                                      MaybeAlign Alignment) const {
  return isLegalMaskedLoadStore(DataType);
}

bool RISCVTTIImpl::isLegalMaskedGather(Type *DataType,
                                       MaybeAlign Alignment) const {
  return isLegalMaskedLoadStore(DataType);
}

bool RISCVTTIImpl::isLegalMaskedScatter(Type *DataType,
                                        MaybeAlign Alignment) const {
  return isLegalMaskedLoadStore(DataType);
}

InstructionCost RISCVTTIImpl::getVectorInstrCost(unsigned Opcode, Type *Val,
                                          unsigned Index) {
  // FIXME: Implement a more precise cost computation model.
  // For now this function is simply a wrapper over the base implementation
  // (i.e. return the legalization cost of the scalar type of the vector
  // elements). It is the simplest reasonable assumption that does not break
  // existing calls to this function, including for FixedVectorTypes.
  return BaseT::getVectorInstrCost(Opcode, Val, Index);
}

InstructionCost RISCVTTIImpl::getShuffleCost(TTI::ShuffleKind Kind,
                                             VectorType *Tp, ArrayRef<int> Mask,
                                             int Index, VectorType *SubTp) {
  if (isa<ScalableVectorType>(Tp) &&
      (!SubTp || isa<ScalableVectorType>(SubTp))) {
    switch (Kind) {
    case TTI::SK_Broadcast:
      return getBroadcastShuffleOverhead(cast<ScalableVectorType>(Tp));
    case TTI::SK_Select:
    case TTI::SK_Reverse:
    case TTI::SK_Transpose:
    case TTI::SK_PermuteSingleSrc:
    case TTI::SK_PermuteTwoSrc:
      return getPermuteShuffleOverhead(cast<ScalableVectorType>(Tp));
    case TTI::SK_ExtractSubvector:
      return getExtractSubvectorOverhead(cast<ScalableVectorType>(Tp), Index,
                                         cast<ScalableVectorType>(SubTp));
    case TTI::SK_InsertSubvector:
      return getInsertSubvectorOverhead(cast<ScalableVectorType>(Tp), Index,
                                        cast<ScalableVectorType>(SubTp));
    }
  }
  return BaseT::getShuffleCost(Kind, Tp, Mask, Index, SubTp);
}

/// Estimate the overhead of scalarizing an instructions unique
/// non-constant operands. The types of the arguments are ordinarily
/// scalar, in which case the costs are multiplied with VF.
InstructionCost
RISCVTTIImpl::getOperandsScalarizationOverhead(ArrayRef<const Value *> Args,
                                               ArrayRef<Type *> Tys) {
  return BaseT::getOperandsScalarizationOverhead(Args, Tys);
}

InstructionCost RISCVTTIImpl::getScalarizationOverhead(
    VectorType *InTy, const APInt &DemandedElts, bool Insert, bool Extract) {
  // FIXME: a bitfield is not a reasonable abstraction for talking about
  // which elements are needed from a scalable vector.
  // For scalable vectors DemenadedElts currently represent
  // ElementCount.getKnownMinValue() number of elements.

  unsigned NumELts = InTy->getElementCount().getKnownMinValue();
  assert(DemandedElts.getBitWidth() == NumELts && "Vector size mismatch");

  InstructionCost MinCost = 0;

  for (unsigned i = 0, e = NumELts; i < e; ++i) {
    if (!DemandedElts[i])
      continue;
    if (Insert)
      MinCost += getVectorInstrCost(Instruction::InsertElement, InTy, i);
    if (Extract)
      MinCost += getVectorInstrCost(Instruction::ExtractElement, InTy, i);
  }

  return *MinCost.getValue();
}

InstructionCost RISCVTTIImpl::getCastInstrCost(unsigned Opcode, Type *Dst,
                                               Type *Src,
                                               TTI::CastContextHint CCH,
                                               TTI::TargetCostKind CostKind,
                                               const Instruction *I) {
  if (!isa<ScalableVectorType>(Dst) || !isa<ScalableVectorType>(Src))
    return BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);

  unsigned LegalizationFactor = 1;
  if (!isTypeLegal(Dst))
    LegalizationFactor = 2;
  if (!isTypeLegal(Src))
    LegalizationFactor *= 2;

  EVT DstVT = getTLI()->getValueType(DL, Dst);
  EVT SrcVT = getTLI()->getValueType(DL, Src);

  // Truncating a mask is cheap (vmsne.vi)
  if (Dst->getScalarSizeInBits() == 1)
    return LegalizationFactor;

  // Extending to a mask should be cheap (vmv.v with mask)
  if (Src->getScalarSizeInBits() == 1)
    return LegalizationFactor;

  int BitRatio =
      std::max(DstVT.getScalarSizeInBits(), SrcVT.getScalarSizeInBits()) /
      std::min(DstVT.getScalarSizeInBits(), SrcVT.getScalarSizeInBits());

  // This case can be done with a single instruction.
  if (BitRatio <= 2)
    return LegalizationFactor;

  // This costs log2(BitRatio) because we need to do several conversions.
  return LegalizationFactor * Log2_32(BitRatio);
}

bool RISCVTTIImpl::shouldMaximizeVectorBandwidth() const {
  return ST->hasStdExtV();
}

ElementCount RISCVTTIImpl::getMinimumVF(unsigned ElemWidth,
                                        bool IsScalable) const {
  return ST->hasStdExtV() && IsScalable
             ? ElementCount::get(
                   std::max<unsigned>(1, getMinVectorRegisterBitWidth() /
                                             ElemWidth),
                   IsScalable)
             : ElementCount::getNull();
}

InstructionCost RISCVTTIImpl::getRegUsageForType(Type *Ty) {
  if (!ST->hasStdExtV()) {
    return BaseT::getRegUsageForType(Ty);
  }

  // FIXME: May need some thought for fixed vectors.
  VectorType *VTy = cast<VectorType>(Ty);
  Type *ETy = VTy->getElementType();
  // Size in bits of this vector type.
  unsigned VectorSizeBits =
      ETy->getScalarSizeInBits() * VTy->getElementCount().getKnownMinValue();

  unsigned RegisterBitSize = getMinVectorRegisterBitWidth();
  return std::max<unsigned>(1, VectorSizeBits / RegisterBitSize);
}

std::pair<ElementCount, ElementCount>
RISCVTTIImpl::getFeasibleMaxVFRange(TargetTransformInfo::RegisterKind K,
                                    unsigned SmallestType, unsigned WidestType,
                                    unsigned MaxSafeRegisterWidth,
                                    unsigned RegWidthFactor,
                                    bool IsScalable) const {
  // check for SEW <= ELEN in the base ISA
  if (WidestType > getMaxElementWidth() || SmallestType > getMaxElementWidth())
    return {ElementCount::getNull(), ElementCount::getNull()};

  // Smallest SEW supported = 8. For 1 bit wide Type, clip to 8 bit to get a
  // valid range of VFs.
  SmallestType = std::max<unsigned>(8, SmallestType);
  WidestType = std::max<unsigned>(8, WidestType);
  unsigned WidestRegister = std::min<unsigned>(
      getMinVectorRegisterBitWidth() * RegWidthFactor,
      MaxSafeRegisterWidth);
  unsigned SmallestRegister =
      std::min(getMinVectorRegisterBitWidth(), MaxSafeRegisterWidth);

  unsigned LowerBoundVFKnownMin =
      std::max<unsigned>(1, PowerOf2Floor(SmallestRegister / SmallestType));
  ElementCount LowerBoundVF =
      ElementCount::get(LowerBoundVFKnownMin, IsScalable);

  unsigned UpperBoundVFKnownMin =
      std::min<unsigned>(64, PowerOf2Floor(WidestRegister / WidestType));
  ElementCount UpperBoundVF =
      ElementCount::get(UpperBoundVFKnownMin, IsScalable);

  return {LowerBoundVF, UpperBoundVF};
}

InstructionCost RISCVTTIImpl::getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                                 Type *CondTy,
                                                 CmpInst::Predicate VecPred,
                                                 TTI::TargetCostKind CostKind,
                                                 const Instruction *I) {
  if (ValTy && isa<ScalableVectorType>(ValTy) && !isTypeLegal(ValTy))
    return InstructionCost::getInvalid();

  if (CondTy && isa<ScalableVectorType>(CondTy) && !isTypeLegal(CondTy))
    return InstructionCost::getInvalid();

  // Apparently the base cannot handle some scalable cases, so let's stop it
  // here for now.
  if (ValTy && isa<ScalableVectorType>(ValTy) && CondTy &&
      isa<ScalableVectorType>(CondTy))
    return 1;

  return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind, I);
}

TargetTransformInfo::PopcntSupportKind
RISCVTTIImpl::getPopcntSupport(unsigned TyWidth) {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  return ST->hasStdExtZbb() ? TTI::PSK_FastHardware : TTI::PSK_Software;
}

bool RISCVTTIImpl::shouldExpandReduction(const IntrinsicInst *II) const {
  // Currently, the ExpandReductions pass can't expand scalable-vector
  // reductions, but we still request expansion as RVV doesn't support certain
  // reductions and the SelectionDAG can't legalize them either.
  switch (II->getIntrinsicID()) {
  default:
    return false;
  // These reductions have no equivalent in RVV
  case Intrinsic::vector_reduce_mul:
  case Intrinsic::vector_reduce_fmul:
    return true;
  }
}

Optional<unsigned> RISCVTTIImpl::getMaxVScale() const {
  // There is no assumption of the maximum vector length in V specification.
  // We use the value specified by users as the maximum vector length.
  // This function will use the assumed maximum vector length to get the
  // maximum vscale for LoopVectorizer.
  // If users do not specify the maximum vector length, we have no way to
  // know whether the LoopVectorizer is safe to do or not.
  // We only consider to use single vector register (LMUL = 1) to vectorize.
  unsigned MaxVectorSizeInBits = ST->getMaxRVVVectorSizeInBits();
  if (ST->hasVInstructions() && MaxVectorSizeInBits != 0)
    return MaxVectorSizeInBits / RISCV::RVVBitsPerBlock;
  return BaseT::getMaxVScale();
}

InstructionCost
RISCVTTIImpl::getArithmeticReductionCost(unsigned Opcode, VectorType *ValTy,
                                         Optional<FastMathFlags> FMF,
                                         TTI::TargetCostKind CostKind) {
  if (!isa<ScalableVectorType>(ValTy))
    return BaseT::getArithmeticReductionCost(Opcode, ValTy, FMF, CostKind);

  std::pair<InstructionCost, MVT> LT = TLI->getTypeLegalizationCost(DL, ValTy);
  InstructionCost LegalizationCost = 0;
  if (LT.first > 1) {
    Type *LegalVTy = EVT(LT.second).getTypeForEVT(ValTy->getContext());
    LegalizationCost = getArithmeticInstrCost(Opcode, LegalVTy, CostKind);
    LegalizationCost *= LT.first - 1;
  }

  // Add the final reduction cost for the legal horizontal reduction
  switch (Opcode) {
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::ICmp:
  case Instruction::FCmp:
  case Instruction::FAdd:
  case Instruction::Add:
    return LegalizationCost + 2;
  default:
    return InstructionCost::getInvalid();
  }
}

// Taken from AArch64.
InstructionCost
RISCVTTIImpl::getMinMaxReductionCost(VectorType *Ty, VectorType *CondTy,
                                     bool IsUnsigned,
                                     TTI::TargetCostKind CostKind) {
  if (!isa<ScalableVectorType>(Ty))
    return BaseT::getMinMaxReductionCost(Ty, CondTy, IsUnsigned, CostKind);

  assert((isa<ScalableVectorType>(Ty) && isa<ScalableVectorType>(CondTy)) &&
         "Both vectors need to be scalable");

  std::pair<InstructionCost, MVT> LT = TLI->getTypeLegalizationCost(DL, Ty);
  InstructionCost LegalizationCost = 0;
  if (LT.first > 1) {
    Type *LegalVTy = EVT(LT.second).getTypeForEVT(Ty->getContext());
    unsigned CmpOpcode =
        Ty->isFPOrFPVectorTy() ? Instruction::FCmp : Instruction::ICmp;
    LegalizationCost =
        getCmpSelInstrCost(CmpOpcode, LegalVTy, LegalVTy,
                           CmpInst::BAD_ICMP_PREDICATE, CostKind) +
        getCmpSelInstrCost(Instruction::Select, LegalVTy, LegalVTy,
                           CmpInst::BAD_ICMP_PREDICATE, CostKind);
    LegalizationCost *= LT.first - 1;
  }

  return LegalizationCost + /*Cost of horizontal reduction*/ 2;
}

InstructionCost
RISCVTTIImpl::getMaskedMemoryOpCost(unsigned Opcode, Type *Src, Align Alignment,
                                    unsigned AddressSpace,
                                    TTI::TargetCostKind CostKind) {
  if (!isa<ScalableVectorType>(Src))
    return BaseT::getMaskedMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                        CostKind);

  return TLI->getTypeLegalizationCost(DL, Src).first;
}

InstructionCost RISCVTTIImpl::getGatherScatterOpCost(
    unsigned Opcode, Type *DataTy, const Value *Ptr, bool VariableMask,
    Align Alignment, TTI::TargetCostKind CostKind, const Instruction *I) {
  // We can do gather/scatter using a single instruction.
  // FIXME: The actual cost is likely to be higher than that.
  if (isa<ScalableVectorType>(DataTy))
    return 1;

  if (CostKind != TTI::TCK_RecipThroughput)
    return BaseT::getGatherScatterOpCost(Opcode, DataTy, Ptr, VariableMask,
                                         Alignment, CostKind, I);

  if ((Opcode == Instruction::Load &&
       !isLegalMaskedGather(DataTy, Align(Alignment))) ||
      (Opcode == Instruction::Store &&
       !isLegalMaskedScatter(DataTy, Align(Alignment))))
    return BaseT::getGatherScatterOpCost(Opcode, DataTy, Ptr, VariableMask,
                                         Alignment, CostKind, I);

  if (!isa<FixedVectorType>(DataTy))
    return BaseT::getGatherScatterOpCost(Opcode, DataTy, Ptr, VariableMask,
                                         Alignment, CostKind, I);

  auto *VTy = cast<FixedVectorType>(DataTy);
  unsigned NumLoads = VTy->getNumElements();
  InstructionCost MemOpCost =
      getMemoryOpCost(Opcode, VTy->getElementType(), Alignment, 0, CostKind, I);
  return NumLoads * MemOpCost;
}

InstructionCost
RISCVTTIImpl::getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                    TTI::TargetCostKind CostKind) {
  // Taken from AArch64.
  auto *RetTy = ICA.getReturnType();
  switch (ICA.getID()) {
  case Intrinsic::experimental_stepvector: {
    InstructionCost Cost = 1; // Cost of the `index' instruction
    auto LT = TLI->getTypeLegalizationCost(DL, RetTy);
    // Legalisation of illegal vectors involves an `index' instruction plus
    // (LT.first - 1) vector adds.
    if (LT.first > 1) {
      Type *LegalVTy = EVT(LT.second).getTypeForEVT(RetTy->getContext());
      InstructionCost AddCost =
          getArithmeticInstrCost(Instruction::Add, LegalVTy, CostKind);
      Cost += AddCost * (LT.first - 1);
    }
    return Cost;
  }
  case Intrinsic::nearbyint: {
    if (isa<ScalableVectorType>(RetTy))
      return InstructionCost::getInvalid();
    break;
  }
  // This is not ideal but untill all VP intrinsics are in upstream we can't use
  // the IsVPIntrinsic getter, so build the list manually from
  // IntrinsicEnums.inc.
#define VP_INTRINSIC_LIST                                                      \
  VP_INTRINSIC(vp_add)                                                         \
  VP_INTRINSIC(vp_and)                                                         \
  VP_INTRINSIC(vp_ashr)                                                        \
  VP_INTRINSIC(vp_fadd)                                                        \
  VP_INTRINSIC(vp_fcmp)                                                        \
  VP_INTRINSIC(vp_fdiv)                                                        \
  VP_INTRINSIC(vp_fma)                                                         \
  VP_INTRINSIC(vp_fmul)                                                        \
  VP_INTRINSIC(vp_fneg)                                                        \
  VP_INTRINSIC(vp_fpext)                                                       \
  VP_INTRINSIC(vp_fptosi)                                                      \
  VP_INTRINSIC(vp_fptoui)                                                      \
  VP_INTRINSIC(vp_fptrunc)                                                     \
  VP_INTRINSIC(vp_frem)                                                        \
  VP_INTRINSIC(vp_fsub)                                                        \
  VP_INTRINSIC(vp_gather)                                                      \
  VP_INTRINSIC(vp_icmp)                                                        \
  VP_INTRINSIC(vp_inttoptr)                                                    \
  VP_INTRINSIC(vp_load)                                                        \
  VP_INTRINSIC(vp_lshr)                                                        \
  VP_INTRINSIC(vp_mul)                                                         \
  VP_INTRINSIC(vp_or)                                                          \
  VP_INTRINSIC(vp_ptrtoint)                                                    \
  VP_INTRINSIC(vp_scatter)                                                     \
  VP_INTRINSIC(vp_sdiv)                                                        \
  VP_INTRINSIC(vp_select)                                                      \
  VP_INTRINSIC(vp_sext)                                                        \
  VP_INTRINSIC(vp_shl)                                                         \
  VP_INTRINSIC(vp_sitofp)                                                      \
  VP_INTRINSIC(vp_srem)                                                        \
  VP_INTRINSIC(vp_store)                                                       \
  VP_INTRINSIC(vp_sub)                                                         \
  VP_INTRINSIC(vp_trunc)                                                       \
  VP_INTRINSIC(vp_udiv)                                                        \
  VP_INTRINSIC(vp_uitofp)                                                      \
  VP_INTRINSIC(vp_urem)                                                        \
  VP_INTRINSIC(vp_xor)                                                         \
  VP_INTRINSIC(vp_zext)                                                        \
  VP_INTRINSIC(experimental_vp_strided_load)                                                \
  VP_INTRINSIC(experimental_vp_strided_store)                                               \
  VP_INTRINSIC(experimental_vp_splice)                                         \
  VP_INTRINSIC(experimental_vp_reverse)
#define VP_INTRINSIC(name) case Intrinsic::name:
  VP_INTRINSIC_LIST
#undef VP_INTRINSIC
    return 1;
  default:
    break;
  }

  return BaseT::getIntrinsicInstrCost(ICA, CostKind);
}

void RISCVTTIImpl::getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                                           TTI::UnrollingPreferences &UP,
                                           OptimizationRemarkEmitter *ORE) {
  // TODO: More tuning on benchmarks and metrics with changes as needed
  //       would apply to all settings below to enable performance.

  // Support explicit targets enabled for SiFive with the unrolling preferences
  // below
  bool UseDefaultPreferences = true;
  if (ST->getTuneCPU().contains("sifive-e76") ||
      ST->getTuneCPU().contains("sifive-s76") ||
      ST->getTuneCPU().contains("sifive-u74") ||
      ST->getTuneCPU().contains("sifive-7"))
    UseDefaultPreferences = false;

  if (UseDefaultPreferences)
    return BasicTTIImplBase::getUnrollingPreferences(L, SE, UP, ORE);

  // Enable Upper bound unrolling universally, not dependant upon the conditions
  // below.
  UP.UpperBound = true;

  // Disable loop unrolling for Oz and Os.
  UP.OptSizeThreshold = 0;
  UP.PartialOptSizeThreshold = 0;
  if (L->getHeader()->getParent()->hasOptSize())
    return;

  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  LLVM_DEBUG(dbgs() << "Loop has:\n"
                    << "Blocks: " << L->getNumBlocks() << "\n"
                    << "Exit blocks: " << ExitingBlocks.size() << "\n");

  // Only allow another exit other than the latch. This acts as an early exit
  // as it mirrors the profitability calculation of the runtime unroller.
  if (ExitingBlocks.size() > 2)
    return;

  // Limit the CFG of the loop body for targets with a branch predictor.
  // Allowing 4 blocks permits if-then-else diamonds in the body.
  if (L->getNumBlocks() > 4)
    return;

  // Don't unroll vectorized loops, including the remainder loop
  if (getBooleanLoopAttribute(L, "llvm.loop.isvectorized"))
    return;

  // Scan the loop: don't unroll loops with calls as this could prevent
  // inlining.
  InstructionCost Cost = 0;
  for (auto *BB : L->getBlocks()) {
    for (auto &I : *BB) {
      // Initial setting - Don't unroll loops containing vectorized
      // instructions.
      if (I.getType()->isVectorTy())
        return;

      if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
        if (const Function *F = cast<CallBase>(I).getCalledFunction()) {
          if (!isLoweredToCall(F))
            continue;
        }
        return;
      }

      SmallVector<const Value *> Operands(I.operand_values());
      Cost +=
          getUserCost(&I, Operands, TargetTransformInfo::TCK_SizeAndLatency);
    }
  }

  LLVM_DEBUG(dbgs() << "Cost of loop: " << Cost << "\n");

  UP.Partial = true;
  UP.Runtime = true;
  UP.UnrollRemainder = true;
  UP.UnrollAndJam = true;
  UP.UnrollAndJamInnerLoopThreshold = 60;

  // Force unrolling small loops can be very useful because of the branch
  // taken cost of the backedge.
  if (Cost < 12)
    UP.Force = true;
}

void RISCVTTIImpl::getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                                         TTI::PeelingPreferences &PP) {
  BaseT::getPeelingPreferences(L, SE, PP);
}
