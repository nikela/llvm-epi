//===-- RISCVTargetTransformInfo.cpp - RISC-V specific TTI ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVTargetTransformInfo.h"
#include "Utils/RISCVMatInt.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"
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
  if (ClassID == 1 && ST->hasStdExtV()) {
    return 32;
  }
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

bool RISCVTTIImpl::useReductionIntrinsic(unsigned Opcode, Type *Ty,
                                         TTI::ReductionFlags Flags) const {
  assert(isa<VectorType>(Ty) && "Expected Ty to be a vector type");
  if (!useScalableVectorType())
    return false;
  switch (Opcode) {
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::ICmp:
  case Instruction::FCmp:
  case Instruction::FAdd:
  case Instruction::Add:
    return true;
  case Instruction::Mul:
  case Instruction::FMul:
    return false;
  default:
    llvm_unreachable("Unhandled reduction opcode");
  }
  return false;
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
  return BaseT::getShuffleCost(Kind, Tp, Index, SubTp);
}

/// Estimate the overhead of scalarizing an instructions unique
/// non-constant operands. The types of the arguments are ordinarily
/// scalar, in which case the costs are multiplied with VF.
unsigned
RISCVTTIImpl::getOperandsScalarizationOverhead(ArrayRef<const Value *> Args,
                                               unsigned MinNumElts) {
  unsigned MinCost = 0;
  SmallPtrSet<const Value *, 4> UniqueOperands;
  for (const Value *A : Args) {
    if (!isa<Constant>(A) && UniqueOperands.insert(A).second) {
      auto *VecTy = dyn_cast<VectorType>(A->getType());
      if (VecTy) {
        // If A is a vector operand, VF should correspond to A.
        assert(MinNumElts == cast<ScalableVectorType>(VecTy)
                                 ->getElementCount()
                                 .getKnownMinValue() &&
               "Vector argument does not match VF");
      } else
        VecTy = ScalableVectorType::get(A->getType(), MinNumElts);

      MinCost += getScalarizationOverhead(VecTy, false, true);
    }
  }
  return MinCost;
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

/// Helper wrapper for the DemandedElts variant of getScalarizationOverhead.
unsigned RISCVTTIImpl::getScalarizationOverhead(VectorType *InTy, bool Insert,
                                                bool Extract) {
  // FIXME: DemandedElts represents active lanes using the number of elements.
  // For scalable vectors it represents min number of elements (vscale = 1).
  // This works fine as long as the cost model is based on the same model of
  // vscale = 1. Once the cost model is changed to represent scalability, we
  // would need a different ADT capable of representing scalable number of
  // elements.
  APInt MinDemandedElts =
      APInt::getAllOnesValue(InTy->getElementCount().getKnownMinValue());
  return getScalarizationOverhead(InTy, MinDemandedElts, Insert, Extract);
}

unsigned RISCVTTIImpl::getScalarizationOverhead(VectorType *InTy,
                                                ArrayRef<const Value *> Args) {
  unsigned Cost = 0;

  Cost += getScalarizationOverhead(InTy, true, false);
  if (!Args.empty())
    Cost += getOperandsScalarizationOverhead(
        Args, InTy->getElementCount().getKnownMinValue());
  else
    // When no information on arguments is provided, we add the cost
    // associated with one argument as a heuristic.
    Cost += getScalarizationOverhead(InTy, false, true);

  return Cost;
}

unsigned RISCVTTIImpl::getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                                        TTI::CastContextHint CCH,
                                        TTI::TargetCostKind CostKind,
                                        const Instruction *I) {
  if (!isa<ScalableVectorType>(Dst) || !isa<ScalableVectorType>(Src))
    return BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);

  unsigned LegalizationFactor = 1;
  if (!isTypeLegal(Dst))
    LegalizationFactor *= 2;
  if (!isTypeLegal(Src))
    LegalizationFactor *= 2;

  // Truncating a mask is cheap (vmsne.vi)
  if (Dst->getScalarSizeInBits() == 1)
    return 1;

  // Extending to a mask should be cheap (vmv.v with mask)
  if (Src->getScalarSizeInBits() == 1)
    return 1;

  int BitRatio =
      std::max(Dst->getScalarSizeInBits(), Src->getScalarSizeInBits()) /
      std::min(Dst->getScalarSizeInBits(), Src->getScalarSizeInBits());

  // This case can be done with a single instruction.
  if (BitRatio <= 2)
    return 1;

  // This costs log2(BitRatio) because we need to do several conversions.
  return Log2_32(BitRatio);
}
