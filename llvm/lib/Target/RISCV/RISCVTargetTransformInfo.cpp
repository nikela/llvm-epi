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
#include "llvm/Support/MathExtras.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "riscvtti"

int RISCVTTIImpl::getIntImmCost(const APInt &Imm, Type *Ty,
                                TTI::TargetCostKind CostKind) {
  assert(Ty->isIntegerTy() &&
         "getIntImmCost can only estimate cost of materialising integers");

  // We have a Zero register, so 0 is always free.
  if (Imm == 0)
    return TTI::TCC_Free;

  // Otherwise, we check how many instructions it will take to materialise.
  const DataLayout &DL = getDataLayout();
  return RISCVMatInt::getIntMatCost(Imm, DL.getTypeSizeInBits(Ty),
                                    getST()->is64Bit());
}

int RISCVTTIImpl::getIntImmCostInst(unsigned Opcode, unsigned Idx,
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
  case Instruction::Add:
  case Instruction::And:
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

int RISCVTTIImpl::getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx,
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

bool RISCVTTIImpl::useScalableVectorType() const {
  return ST->hasStdExtV();
}

bool RISCVTTIImpl::preferPredicatedVectorOps() const {
  return (useScalableVectorType() && true);
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

unsigned RISCVTTIImpl::getVectorInstrCost(unsigned Opcode, Type *Val,
                                          unsigned Index) {
  // FIXME: Implement a more precise cost computation model.
  // For now this function is simply a wrapper over the base implementation
  // (i.e. return the legalization cost of the scalar type of the vector
  // elements). It is the simplest reasonable assumption that does not break
  // existing calls to this function, including for FixedVectorTypes.
  return BaseT::getVectorInstrCost(Opcode, Val, Index);
}

unsigned RISCVTTIImpl::getShuffleCost(TTI::ShuffleKind Kind, VectorType *Tp,
                                      ArrayRef<int> Mask, int Index, VectorType *SubTp) {
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
unsigned
RISCVTTIImpl::getOperandsScalarizationOverhead(ArrayRef<const Value *> Args,
                                               ArrayRef<Type *> Tys) {
  return BaseT::getOperandsScalarizationOverhead(Args, Tys);
}

unsigned RISCVTTIImpl::getScalarizationOverhead(VectorType *InTy,
                                                const APInt &DemandedElts,
                                                bool Insert, bool Extract) {
  // FIXME: a bitfield is not a reasonable abstraction for talking about
  // which elements are needed from a scalable vector.
  // For scalable vectors DemenadedElts currently represent
  // ElementCount.getKnownMinValue() number of elements.

  unsigned NumELts = InTy->getElementCount().getKnownMinValue();
  assert(DemandedElts.getBitWidth() == NumELts && "Vector size mismatch");

  unsigned MinCost = 0;

  for (unsigned i = 0, e = NumELts; i < e; ++i) {
    if (!DemandedElts[i])
      continue;
    if (Insert)
      MinCost += getVectorInstrCost(Instruction::InsertElement, InTy, i);
    if (Extract)
      MinCost += getVectorInstrCost(Instruction::ExtractElement, InTy, i);
  }

  return MinCost;
}

unsigned RISCVTTIImpl::getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
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

bool RISCVTTIImpl::shouldMaximizeVectorBandwidth(bool OptSize) const {
  return (ST->hasStdExtV() && true);
}

unsigned RISCVTTIImpl::getMinVectorRegisterBitWidth() const {
  // Actual min vector register bitwidth is <vscale x ELEN>.
  // getMaxElementWidth() simply return ELEN.
  return ST->hasStdExtV() ? getMaxElementWidth() : 0;
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

unsigned RISCVTTIImpl::getVectorRegisterUsage(
    TargetTransformInfo::RegisterKind K, unsigned VFKnownMin,
    unsigned ElementTypeSize, unsigned SafeDepDist) const {
  // FIXME: For the time being we assume dependency distance is always safe.
  // Once we have dependency distance computations for scalable vectors, we need
  // to figure out its relationship with register group usage;
  unsigned RegisterWidth = getMinVectorRegisterBitWidth();
  return std::max<unsigned>(1, VFKnownMin * ElementTypeSize / RegisterWidth);
}

std::pair<ElementCount, ElementCount>
RISCVTTIImpl::getFeasibleMaxVFRange(TargetTransformInfo::RegisterKind K,
                                    unsigned SmallestType, unsigned WidestType,
                                    unsigned MaxSafeRegisterWidth,
                                    unsigned RegWidthFactor) const {
  // check for SEW <= ELEN in the base ISA
  assert(WidestType <= getMaxElementWidth() &&
         "Vector element type larger than the maximum supported type.");
  // Smallest SEW supported = 8. For 1 bit wide Type, clip to 8 bit to get a
  // valid range of VFs.
  SmallestType = std::max<unsigned>(8, SmallestType);
  WidestType = std::max<unsigned>(8, WidestType);
  unsigned WidestRegister = std::min<unsigned>(
      getMinVectorRegisterBitWidth() * RegWidthFactor,
      MaxSafeRegisterWidth);
  unsigned SmallestRegister =
      std::min(getMinVectorRegisterBitWidth(), MaxSafeRegisterWidth);
  bool IsScalable = useScalableVectorType();

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

int RISCVTTIImpl::getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                                     CmpInst::Predicate VecPred,
                                     TTI::TargetCostKind CostKind,
                                     const Instruction *I) {
  // FIXME: For the time being we only consider the case when the ValTy or
  // CondTy is illegal and return an artificially high cost. For other cases we
  // default to the base implementation.
  if (ValTy && ValTy->isVectorTy() && !isTypeLegal(ValTy))
    return HighCost;

  if (CondTy && CondTy->isVectorTy() && !isTypeLegal(CondTy))
    return HighCost;

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
  // The fmin and fmax intrinsics are not currently supported due to a
  // discrepancy between the LLVM semantics and the RVV 0.10 ISA behaviour with
  // regards to signaling NaNs: the vector fmin/fmax reduction intrinsics match
  // the behaviour minnum/maxnum intrinsics, whereas the vfredmin/vfredmax
  // instructions match the vfmin/vfmax instructions which match the equivalent
  // scalar fmin/fmax instructions as defined in 2.2 F/D/Q extension (see
  // https://bugs.llvm.org/show_bug.cgi?id=27363).
  // This behaviour is likely fixed in version 2.3 of the RISC-V F/D/Q
  // extension, where fmin/fmax behave like minnum/maxnum, but until then the
  // intrinsics are left unsupported.
  case Intrinsic::vector_reduce_fmax:
  case Intrinsic::vector_reduce_fmin:
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
  if (ST->hasStdExtV() && MaxVectorSizeInBits != 0)
    return MaxVectorSizeInBits / RISCV::RVVBitsPerBlock;
  return BaseT::getMaxVScale();
}

int RISCVTTIImpl::getArithmeticReductionCost(unsigned Opcode, VectorType *ValTy,
                                             bool IsPairwiseForm,
                                             TTI::TargetCostKind CostKind) {
  if (!isa<ScalableVectorType>(ValTy))
    return BaseT::getArithmeticReductionCost(Opcode, ValTy, IsPairwiseForm,
                                             CostKind);

  // Following what AArch64 does here.
  if (IsPairwiseForm)
    return BaseT::getArithmeticReductionCost(Opcode, ValTy, IsPairwiseForm,
                                             CostKind);

  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, ValTy);
  int LegalizationCost = 0;
  if (LT.first > 1) {
    Type *LegalVTy = EVT(LT.second).getTypeForEVT(ValTy->getContext());
    LegalizationCost = getArithmeticInstrCost(Opcode, LegalVTy, CostKind);
    LegalizationCost *= LT.first - 1;
  }

  // Update to InstructionCost when ready.
  constexpr int InfiniteCost = 1024;
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
    // TODO: Replace for invalid when InstructionCost is used.
    return InfiniteCost;
  }
}

// Taken from AArch64.
int RISCVTTIImpl::getMinMaxReductionCost(VectorType *Ty, VectorType *CondTy,
                                         bool IsPairwise, bool IsUnsigned,
                                         TTI::TargetCostKind CostKind) {
  if (!isa<ScalableVectorType>(Ty))
    return BaseT::getMinMaxReductionCost(Ty, CondTy, IsPairwise, IsUnsigned,
                                         CostKind);

  assert((isa<ScalableVectorType>(Ty) && isa<ScalableVectorType>(CondTy)) &&
         "Both vectors need to be scalable");

  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Ty);
  int LegalizationCost = 0;
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

unsigned RISCVTTIImpl::getGatherScatterOpCost(
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
  unsigned MemOpCost =
      getMemoryOpCost(Opcode, VTy->getElementType(), Alignment, 0, CostKind, I);
  return NumLoads * MemOpCost;
}
