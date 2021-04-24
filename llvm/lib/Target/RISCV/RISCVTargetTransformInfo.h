//===- RISCVTargetTransformInfo.h - RISC-V specific TTI ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines a TargetTransformInfo::Concept conforming object specific
/// to the RISC-V target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_RISCV_RISCVTARGETTRANSFORMINFO_H

#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"

namespace llvm {

class RISCVTTIImpl : public BasicTTIImplBase<RISCVTTIImpl> {
  using BaseT = BasicTTIImplBase<RISCVTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const RISCVSubtarget *ST;
  const RISCVTargetLowering *TLI;

  const RISCVSubtarget *getST() const { return ST; }
  const RISCVTargetLowering *getTLI() const { return TLI; }

  // FIXME:This is just a temporary way to signal that the cost of an
  // instruction is too high to consider. When we have a more complete cost
  // object that has inbuilt mechanism to indicate an infinite/saturated cost,
  // use that. (For the same reason, at the moment we are limiting this const
  // value only to RISCV.)
  const int HighCost = 1 << 10;

  bool isLegalMaskedLoadStore(Type *DataType) const;

  /// Estimate a cost of Broadcast as an extract and sequence of insert
  /// operations.
  InstructionCost getBroadcastShuffleOverhead(ScalableVectorType *VTy) {
    InstructionCost MinCost = 0;
    // Broadcast cost is equal to the cost of extracting the zero'th element
    // plus the cost of inserting it into every element of the result vector.
    // FIXME: For scalable vectors for now we compute the MinCost based on Min
    // number of elements but this does not represent the correct cost. This
    // would be fixed once the cost model has support for scalable vectors.
    MinCost += getVectorInstrCost(Instruction::ExtractElement, VTy, 0);

    for (int i = 0, e = VTy->getElementCount().getKnownMinValue(); i < e; ++i) {
      MinCost += getVectorInstrCost(Instruction::InsertElement, VTy, i);
    }
    return MinCost;
  }

  /// Estimate a cost of shuffle as a sequence of extract and insert
  /// operations.
  InstructionCost getPermuteShuffleOverhead(ScalableVectorType *VTy) {
    InstructionCost MinCost = 0;
    // Shuffle cost is equal to the cost of extracting element from its argument
    // plus the cost of inserting them onto the result vector.

    // e.g. for a fixed vector <4 x float> has a mask of <0,5,2,7> i.e we need
    // to extract from index 0 of first vector, index 1 of second vector,index 2
    // of first vector and finally index 3 of second vector and insert them at
    // index <0,1,2,3> of result vector.
    // FIXME: For scalable vectors for now we compute the MinCost based on Min
    // number of elements but this does not represent the correct cost. This
    // would be fixed once the cost model has support for scalable vectors.
    for (int i = 0, e = VTy->getElementCount().getKnownMinValue(); i < e; ++i) {
      MinCost += getVectorInstrCost(Instruction::InsertElement, VTy, i);
      MinCost += getVectorInstrCost(Instruction::ExtractElement, VTy, i);
    }
    return MinCost;
  }

  /// Estimate a cost of subvector extraction as a sequence of extract and
  /// insert operations.
  InstructionCost getExtractSubvectorOverhead(ScalableVectorType *VTy, int Index,
                                       ScalableVectorType *SubVTy) {
    assert(VTy && SubVTy && "Can only extract subvectors from vectors");
    // FIXME: We cannot assert index bounds of SubVTy at compile time.

    unsigned NumSubElts = SubVTy->getElementCount().getKnownMinValue();
    InstructionCost MinCost = 0;
    // Subvector extraction cost is equal to the cost of extracting element
    // from the source type plus the cost of inserting them into the result
    // vector type.
    // FIXME: For scalable vectors for now we compute the MinCost based on Min
    // number of elements but this does not represent the correct cost. This
    // would be fixed once the cost model has support for scalable vectors.
    for (unsigned i = 0; i != NumSubElts; ++i) {
      MinCost +=
          getVectorInstrCost(Instruction::ExtractElement, VTy, i + Index);
      MinCost += getVectorInstrCost(Instruction::InsertElement, SubVTy, i);
    }
    return MinCost;
  }

  /// Estimate a cost of subvector insertion as a sequence of extract and
  /// insert operations.
  InstructionCost getInsertSubvectorOverhead(ScalableVectorType *VTy, int Index,
                                             ScalableVectorType *SubVTy) {
    assert(VTy && SubVTy && "Can only insert subvectors into vectors");
    // FIXME: We cannot assert index bounds of SubVTy at compile time.

    unsigned NumSubElts = SubVTy->getElementCount().getKnownMinValue();
    InstructionCost MinCost = 0;
    // Subvector insertion cost is equal to the cost of extracting element
    // from the source type plus the cost of inserting them into the result
    // vector type.
    // FIXME: For scalable vectors for now we compute the MinCost based on Min
    // number of elements but this does not represent the correct cost. This
    // would be fixed once the cost model has support for scalable vectors.
    for (unsigned i = 0; i != NumSubElts; ++i) {
      MinCost += getVectorInstrCost(Instruction::ExtractElement, SubVTy, i);
      MinCost += getVectorInstrCost(Instruction::InsertElement, VTy, i + Index);
    }
    return MinCost;
  }

public:
  explicit RISCVTTIImpl(const RISCVTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  InstructionCost getIntImmCost(const APInt &Imm, Type *Ty,
                                TTI::TargetCostKind CostKind);
  InstructionCost getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                    const APInt &Imm, Type *Ty,
                                    TTI::TargetCostKind CostKind,
                                    Instruction *Inst = nullptr);
  InstructionCost getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx,
                                      const APInt &Imm, Type *Ty,
                                      TTI::TargetCostKind CostKind);

  unsigned getNumberOfRegisters(unsigned ClassID) const;
  unsigned getMaxElementWidth() const;
  bool useScalableVectorType() const;
  bool preferPredicatedVectorOps() const;
  bool isLegalMaskedLoad(Type *DataType, MaybeAlign Alignment) const;
  bool isLegalMaskedStore(Type *DataType, MaybeAlign Alignment) const;
  bool isLegalMaskedGather(Type *DataType, MaybeAlign Alignment) const;
  bool isLegalMaskedScatter(Type *DataType, MaybeAlign Alignment) const;
  InstructionCost getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index);
  InstructionCost getShuffleCost(TTI::ShuffleKind Kind, VectorType *Tp,
                                 ArrayRef<int> Mask, int Index,
                                 VectorType *SubTp);
  unsigned getOperandsScalarizationOverhead(ArrayRef<const Value *> Args,
                                            ArrayRef<Type *> Tys);
  unsigned getScalarizationOverhead(VectorType *InTy,
                                           const APInt &DemandedElts,
                                           bool Insert, bool Extract);
  InstructionCost getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                                   TTI::CastContextHint CCH,
                                   TTI::TargetCostKind CostKind,
                                   const Instruction *I = nullptr);
  bool shouldMaximizeVectorBandwidth() const;
  unsigned getMinVectorRegisterBitWidth() const;
  ElementCount getMinimumVF(unsigned ElemWidth, bool IsScalable) const;
  unsigned getVectorRegisterUsage(TargetTransformInfo::RegisterKind K,
                                  unsigned VFKnownMin, unsigned ElementTypeSize,
                                  unsigned SafeDepDist) const;
  std::pair<ElementCount, ElementCount>
  getFeasibleMaxVFRange(TargetTransformInfo::RegisterKind K,
                        unsigned SmallestType, unsigned WidestType,
                        unsigned MaxSafeRegisterWidth = -1U,
                        unsigned RegWidthFactor = 1) const;
  InstructionCost
  getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy = nullptr,
                     CmpInst::Predicate VecPred = CmpInst::BAD_ICMP_PREDICATE,
                     TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
                     const Instruction *I = nullptr);

  TargetTransformInfo::PopcntSupportKind getPopcntSupport(unsigned TyWidth);

  bool shouldExpandReduction(const IntrinsicInst *II) const;
  bool supportsScalableVectors() const { return ST->hasStdExtV(); }
  Optional<unsigned> getMaxVScale() const;
  InstructionCost getArithmeticReductionCost(unsigned Opcode, VectorType *ValTy,
                                             bool IsPairwiseForm,
                                             TTI::TargetCostKind CostKind);
  InstructionCost getMinMaxReductionCost(VectorType *Ty, VectorType *CondTy,
                                         bool IsPairwise, bool IsUnsigned,
                                         TTI::TargetCostKind CostKind);
  TypeSize getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const {
    switch (K) {
    case TargetTransformInfo::RGK_Scalar:
      return TypeSize::getFixed(ST->getXLen());
    case TargetTransformInfo::RGK_FixedWidthVector:
      return TypeSize::getFixed(
          ST->hasStdExtV() ? ST->getMinRVVVectorSizeInBits() : 0);
    case TargetTransformInfo::RGK_ScalableVector:
      return TypeSize::getScalable(
          ST->hasStdExtV() ? ST->getMinRVVVectorSizeInBits() : 0);
    }

    llvm_unreachable("Unsupported register kind");
  }

  InstructionCost getGatherScatterOpCost(unsigned Opcode, Type *DataTy,
                                         const Value *Ptr, bool VariableMask,
                                         Align Alignment,
                                         TTI::TargetCostKind CostKind,
                                         const Instruction *I);

  bool isLegalElementTypeForRVV(Type *ScalarTy) {
    if (ScalarTy->isPointerTy())
      return true;

    if (ScalarTy->isIntegerTy(8) || ScalarTy->isIntegerTy(16) ||
        ScalarTy->isIntegerTy(32) || ScalarTy->isIntegerTy(64))
      return true;

    if (ScalarTy->isHalfTy())
      return ST->hasStdExtZfh();
    if (ScalarTy->isFloatTy())
      return ST->hasStdExtF();
    if (ScalarTy->isDoubleTy())
      return ST->hasStdExtD();

    return false;
  }

  bool isLegalMaskedLoadStore(Type *DataType, Align Alignment) {
    if (!ST->hasStdExtV())
      return false;

    // Only support fixed vectors if we know the minimum vector size.
    if (isa<FixedVectorType>(DataType) && ST->getMinRVVVectorSizeInBits() == 0)
      return false;

    return isLegalElementTypeForRVV(DataType->getScalarType());
  }

  bool isLegalMaskedLoad(Type *DataType, Align Alignment) {
    return isLegalMaskedLoadStore(DataType, Alignment);
  }
  bool isLegalMaskedStore(Type *DataType, Align Alignment) {
    return isLegalMaskedLoadStore(DataType, Alignment);
  }

  bool isLegalMaskedGatherScatter(Type *DataType, Align Alignment) {
    if (!ST->hasStdExtV())
      return false;

    // Only support fixed vectors if we know the minimum vector size.
    if (isa<FixedVectorType>(DataType) && ST->getMinRVVVectorSizeInBits() == 0)
      return false;

    return isLegalElementTypeForRVV(DataType->getScalarType());
  }

  bool isLegalMaskedGather(Type *DataType, Align Alignment) {
    return isLegalMaskedGatherScatter(DataType, Alignment);
  }
  bool isLegalMaskedScatter(Type *DataType, Align Alignment) {
    return isLegalMaskedGatherScatter(DataType, Alignment);
  }

  bool isLegalToVectorizeReduction(RecurrenceDescriptor RdxDesc,
                                   ElementCount VF) const;

  InstructionCost getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                        TTI::TargetCostKind CostKind);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVTARGETTRANSFORMINFO_H
