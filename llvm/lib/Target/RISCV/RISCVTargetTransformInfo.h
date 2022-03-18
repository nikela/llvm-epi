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
#include "llvm/Analysis/IVDescriptors.h"
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

  unsigned getMaxElementWidth() const;
  bool preferPredicatedVectorOps() const;
  bool isLegalMaskedLoad(Type *DataType, MaybeAlign Alignment) const;
  bool isLegalMaskedStore(Type *DataType, MaybeAlign Alignment) const;
  bool isLegalMaskedGather(Type *DataType, MaybeAlign Alignment) const;
  bool isLegalMaskedScatter(Type *DataType, MaybeAlign Alignment) const;
  InstructionCost getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index);
  InstructionCost getOperandsScalarizationOverhead(ArrayRef<const Value *> Args,
                                                   ArrayRef<Type *> Tys);
  InstructionCost getScalarizationOverhead(VectorType *InTy,
                                           const APInt &DemandedElts,
                                           bool Insert, bool Extract);
  InstructionCost getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                                   TTI::CastContextHint CCH,
                                   TTI::TargetCostKind CostKind,
                                   const Instruction *I = nullptr);
  bool shouldMaximizeVectorBandwidth() const;
  ElementCount getMinimumVF(unsigned ElemWidth, bool IsScalable) const;
  unsigned getVectorRegisterUsage(TargetTransformInfo::RegisterKind K,
                                  unsigned VFKnownMin, unsigned ElementTypeSize,
                                  unsigned SafeDepDist) const;
  std::pair<ElementCount, ElementCount>
  getFeasibleMaxVFRange(TargetTransformInfo::RegisterKind K,
                        unsigned SmallestType, unsigned WidestType,
                        unsigned MaxSafeRegisterWidth = -1U,
                        unsigned RegWidthFactor = 1,
                        bool IsScalable = false) const;
  InstructionCost
  getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy = nullptr,
                     CmpInst::Predicate VecPred = CmpInst::BAD_ICMP_PREDICATE,
                     TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
                     const Instruction *I = nullptr);

  TargetTransformInfo::PopcntSupportKind getPopcntSupport(unsigned TyWidth);

  bool shouldExpandReduction(const IntrinsicInst *II) const;
  bool supportsScalableVectors() const { return ST->hasVInstructions(); }
  Optional<unsigned> getMaxVScale() const;

  TypeSize getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const;

  InstructionCost getRegUsageForType(Type *Ty);

  InstructionCost getMemoryOpCost(unsigned Opcode, Type *Ty,
                                  MaybeAlign Alignment, unsigned AddressSpace,
                                  TTI::TargetCostKind CostKind,
                                  const Instruction *I = nullptr);

  InstructionCost getMaskedMemoryOpCost(unsigned Opcode, Type *Src,
                                        Align Alignment, unsigned AddressSpace,
                                        TTI::TargetCostKind CostKind);

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP,
                               OptimizationRemarkEmitter *ORE);

  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP);

  unsigned getMinVectorRegisterBitWidth() const {
    return ST->useRVVForFixedLengthVectors() ? 16 : 0;
  }

  InstructionCost getSpliceCost(VectorType *Tp, int Index);
  InstructionCost getShuffleCost(TTI::ShuffleKind Kind, VectorType *Tp,
                                 ArrayRef<int> Mask, int Index,
                                 VectorType *SubTp);

  InstructionCost getGatherScatterOpCost(unsigned Opcode, Type *DataTy,
                                         const Value *Ptr, bool VariableMask,
                                         Align Alignment,
                                         TTI::TargetCostKind CostKind,
                                         const Instruction *I);

  InstructionCost getMinMaxReductionCost(VectorType *Ty, VectorType *CondTy,
                                         bool IsUnsigned,
                                         TTI::TargetCostKind CostKind);

  InstructionCost getArithmeticReductionCost(unsigned Opcode, VectorType *Ty,
                                             Optional<FastMathFlags> FMF,
                                             TTI::TargetCostKind CostKind);

  bool isLegalMaskedLoadStore(Type *DataType, Align Alignment) {
    if (!ST->hasVInstructions())
      return false;

    // Only support fixed vectors if we know the minimum vector size.
    if (isa<FixedVectorType>(DataType) && ST->getMinRVVVectorSizeInBits() == 0)
      return false;

    // Don't allow elements larger than the ELEN.
    // FIXME: How to limit for scalable vectors?
    if (isa<FixedVectorType>(DataType) &&
        DataType->getScalarSizeInBits() > ST->getMaxELENForFixedLengthVectors())
      return false;

    if (Alignment <
        DL.getTypeStoreSize(DataType->getScalarType()).getFixedSize())
      return false;

    return TLI->isLegalElementTypeForRVV(DataType->getScalarType());
  }

  bool isLegalMaskedLoad(Type *DataType, Align Alignment) {
    return isLegalMaskedLoadStore(DataType, Alignment);
  }
  bool isLegalMaskedStore(Type *DataType, Align Alignment) {
    return isLegalMaskedLoadStore(DataType, Alignment);
  }

  bool isLegalMaskedGatherScatter(Type *DataType, Align Alignment) {
    if (!ST->hasVInstructions())
      return false;

    // Only support fixed vectors if we know the minimum vector size.
    if (isa<FixedVectorType>(DataType) && ST->getMinRVVVectorSizeInBits() == 0)
      return false;

    // Don't allow elements larger than the ELEN.
    // FIXME: How to limit for scalable vectors?
    if (isa<FixedVectorType>(DataType) &&
        DataType->getScalarSizeInBits() > ST->getMaxELENForFixedLengthVectors())
      return false;

    if (Alignment <
        DL.getTypeStoreSize(DataType->getScalarType()).getFixedSize())
      return false;

    return TLI->isLegalElementTypeForRVV(DataType->getScalarType());
  }

  bool isLegalMaskedGather(Type *DataType, Align Alignment) {
    return isLegalMaskedGatherScatter(DataType, Alignment);
  }
  bool isLegalMaskedScatter(Type *DataType, Align Alignment) {
    return isLegalMaskedGatherScatter(DataType, Alignment);
  }

  InstructionCost getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                        TTI::TargetCostKind CostKind);

  TargetTransformInfo::VPLegalization
  getVPLegalizationStrategy(const VPIntrinsic &PI) const {
    // FIXME: we may want to be more selective.
    if (ST->hasVInstructions())
      return {/* EVL */ TargetTransformInfo::VPLegalization::Legal,
              /* Op */ TargetTransformInfo::VPLegalization::Legal};

    return BaseT::getVPLegalizationStrategy(PI);
  }

  bool isLegalToVectorizeReduction(const RecurrenceDescriptor &RdxDesc,
                                   ElementCount VF) const {
    if (!ST->hasVInstructions())
      return false;

    if (!VF.isScalable())
      return true;

    Type *Ty = RdxDesc.getRecurrenceType();
    if (!TLI->isLegalElementTypeForRVV(Ty))
      return false;

    switch (RdxDesc.getRecurrenceKind()) {
    case RecurKind::Add:
    case RecurKind::FAdd:
    case RecurKind::And:
    case RecurKind::Or:
    case RecurKind::Xor:
    case RecurKind::SMin:
    case RecurKind::SMax:
    case RecurKind::UMin:
    case RecurKind::UMax:
    case RecurKind::FMin:
    case RecurKind::FMax:
      return true;
    default:
      return false;
    }
  }

  unsigned getMaxInterleaveFactor(unsigned VF) {
    // If the loop will not be vectorized, don't interleave the loop.
    // Let regular unroll to unroll the loop.
    return VF == 1 ? 1 : ST->getMaxInterleaveFactor();
  }

  unsigned getNumberOfRegistersEPI(unsigned ClassID) const {
    if (ClassID == 1 && ST->hasVInstructions())
      // Although there are 32 vector registers, v0 is special in that it is
      // the only register that can be used to hold a mask. We conservatively
      // return 31 as the number of usable vector registers.
      return 31;
    else if (ClassID == 0)
      // Similarly for scalar registers, x0(zero), x1(ra) and x2(sp) are
      // special and we return 29 usable registers.
      return 29;
    else
      return 0;
  }

  // TODO: We should define RISC-V's own register classes.
  //       e.g. register class for FPR.
  unsigned getNumberOfRegisters(unsigned ClassID) const {
    if (ST->hasEPI())
      return getNumberOfRegistersEPI(ClassID);

    bool Vector = (ClassID == 1);
    if (Vector) {
      if (ST->hasVInstructions())
        return 32;
      return 0;
    }
    // 31 = 32 GPR - x0 (zero register)
    // FIXME: Should we exclude fixed registers like SP, TP or GP?
    return 31;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVTARGETTRANSFORMINFO_H
