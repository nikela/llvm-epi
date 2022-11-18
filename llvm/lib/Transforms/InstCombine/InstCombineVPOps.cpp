//===- InstCombineVPOps.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the visitVP* functions.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"

#define DEBUG_TYPE "instcombine"

using namespace llvm;
using namespace llvm::PatternMatch;

Value *createIntExtOrTruncVP(VectorBuilder &VB, Value *From, Type *ToTy,
                             bool IsSigned) {
  Type *FromTy = From->getType();
  assert(FromTy->isIntOrIntVectorTy() && ToTy->isIntOrIntVectorTy() &&
         "Invalid integer cast");
  if (FromTy == ToTy)
    return From;

  unsigned SrcBits = FromTy->getScalarSizeInBits();
  unsigned DstBits = ToTy->getScalarSizeInBits();
  assert(SrcBits != DstBits);
  Instruction::CastOps Opcode =
      SrcBits > DstBits ? Instruction::Trunc
                        : (IsSigned ? Instruction::SExt : Instruction::ZExt);

  return VB.createVectorInstruction(Opcode, ToTy, {From});
}

Value *InstCombinerImpl::emitGEPOffsetVP(GetElementPtrInst &GEP,
                                         VectorBuilder &VB) {
  // Get the right integer type needed to represent a pointer.
  Type *IntPtrTy = Builder.getIntPtrTy(DL);

  // Build a mask for high order bits.
  unsigned IntPtrWidth = IntPtrTy->getIntegerBitWidth();
  uint64_t PtrSizeMask =
      std::numeric_limits<uint64_t>::max() >> (64 - IntPtrWidth);

  auto *IntIdxTy = VectorType::get(IntPtrTy, cast<VectorType>(GEP.getType()));
  Value *Result = nullptr;
  gep_type_iterator GTI = gep_type_begin(GEP);
  for (User::op_iterator I = GEP.op_begin() + 1, E = GEP.op_end(); I != E;
       ++I, ++GTI) {
    Value *Op = *I;
    uint64_t Size = DL.getTypeAllocSize(GTI.getIndexedType()) & PtrSizeMask;
    Value *Offset = nullptr;
    if (auto *OpC = dyn_cast<Constant>(Op)) {
      if (OpC->isZeroValue())
        continue;

      // Handle a struct index, which adds its field offset to the pointer.
      if (StructType *STy = GTI.getStructTypeOrNull()) {
        uint64_t OpValue = OpC->getUniqueInteger().getZExtValue();
        Size = DL.getStructLayout(STy)->getElementOffset(OpValue);
        if (!Size)
          continue;

        Offset = ConstantInt::get(IntIdxTy, Size);
      }
    }
    if (!Offset) {
      // Splat the index if needed.
      if (!Op->getType()->isVectorTy())
        Op = Builder.CreateVectorSplat(IntIdxTy->getElementCount(), Op);
      // Convert to correct type.
      if (Op->getType() != IntIdxTy)
        Op = createIntExtOrTruncVP(VB, Op, IntIdxTy, /*IsSigned*/ true);
      if (Size != 1)
        // We'll let instcombine(mul) convert this to a shl if possible.
        Op = VB.createVectorInstruction(Instruction::Mul, IntIdxTy,
                                        {Op, ConstantInt::get(IntIdxTy, Size)});

      Offset = Op;
    }

    if (Result)
      Result = VB.createVectorInstruction(Instruction::Add, IntIdxTy,
                                          {Result, Offset});
    else
      Result = Offset;
  }

  return Result ? Result : ConstantInt::get(IntIdxTy, 0);
}

Instruction *InstCombinerImpl::visitVPInst(VPIntrinsic *VPI) {
  switch (VPI->getIntrinsicID()) {
  default:
    break;
  case Intrinsic::vp_mul:
    return visitVPMul(VPI);
  case Intrinsic::vp_select:
    return visitVPSelect(VPI);
  }
  return nullptr;
}

Instruction *
InstCombinerImpl::visitVPGatherScatterOnlyGEP(GetElementPtrInst &GEP) {
  if (!GEP.getType()->isVectorTy())
    return nullptr;

  // Check if the GEP is a viable candidate:
  // - Check that all uses of this GEP are the pointer operand of either a
  // vp.gather or vp.scatter intrinsic.
  // - Check that all users have the same mask and the same VL.
  Value *Mask = nullptr;
  Value *VL = nullptr;
  for (auto &GEPUse : GEP.uses()) {
    User *GEPUser = GEPUse.getUser();
    auto *VPI = dyn_cast<VPIntrinsic>(GEPUser);
    if (!VPI)
      return nullptr;
    if (VPI->getIntrinsicID() != Intrinsic::vp_gather &&
        VPI->getIntrinsicID() != Intrinsic::vp_scatter)
      return nullptr;
    // The GEP must be the pointer operand of the gather/scatter.
    if (GEPUse != VPI->getMemoryPointerParam())
      return nullptr;
    if (!Mask && !VL) {
      Mask = VPI->getMaskParam();
      VL = VPI->getVectorLengthParam();
    } else {
      if (VPI->getMaskParam() != Mask || VPI->getVectorLengthParam() != VL)
        return nullptr;
    }
  }

  // Replace the GEP with VP instrinsics:
  // - vp.ptrtoint
  // - vp.add to calculate new vector of pointers
  // - vp.inttoptr
  // N.B.: indices are correctly scaled to represent the number of bytes when
  // calculating the offset in VPEmitGEPOffset().
  VectorBuilder VB(Builder);
  VB.setMask(ConstantInt::getAllOnesValue(Mask->getType()));
  VB.setEVL(VL);
  auto *GEPType = cast<VectorType>(GEP.getType());
  Value *PtrOp = GEP.getPointerOperand();
  Value *Offset = emitGEPOffsetVP(GEP, VB);
  auto *PtrSizeVecTy = VectorType::get(Builder.getIntPtrTy(DL), GEPType);
  if (!PtrOp->getType()->isVectorTy())
    PtrOp = Builder.CreateVectorSplat(GEPType->getElementCount(), PtrOp);
  if (!Offset->getType()->isVectorTy())
    Offset = Builder.CreateVectorSplat(GEPType->getElementCount(), Offset);
  if (Offset->getType() != PtrSizeVecTy)
    Offset = createIntExtOrTruncVP(VB, Offset, PtrSizeVecTy, /*IsSigned*/ true);

  Value *PtrToInt =
      VB.createVectorInstruction(Instruction::PtrToInt, PtrSizeVecTy, {PtrOp});
  Value *Add = VB.createVectorInstruction(Instruction::Add, PtrSizeVecTy,
                                          {PtrToInt, Offset});
  Value *IntToPtr =
      VB.createVectorInstruction(Instruction::IntToPtr, GEPType, {Add});

  return replaceInstUsesWith(GEP, IntToPtr);
}

Instruction *InstCombinerImpl::visitVPMul(VPIntrinsic *VPMul) {
  auto *RetTy = cast<ScalableVectorType>(VPMul->getType());
  Value *Op0 = VPMul->getOperand(0), *Op1 = VPMul->getOperand(1);
  if (isa<Constant>(Op0) && !isa<Constant>(Op1)) {
    // FIXME: we should have canonicalised this earlier.
    std::swap(Op0, Op1);
  }

  if (auto *CI = dyn_cast_or_null<ConstantInt>(getSplatValue(Op1))) {
    if (auto *Log2CI = ConstantExpr::getExactLogBase2(CI)) {
      auto *NewOp1 =
          Builder.CreateVectorSplat(RetTy->getElementCount(), Log2CI);
      VectorBuilder VB(Builder);
      VB.setMask(VPMul->getMaskParam());
      VB.setEVL(VPMul->getVectorLengthParam());
      Value *Shl =
          VB.createVectorInstruction(Instruction::Shl, RetTy, {Op0, NewOp1});
      return replaceInstUsesWith(*VPMul, Shl);
    }
  }

  return nullptr;
}

Instruction *InstCombinerImpl::visitVPSelect(VPIntrinsic *VPSelect) {
  Type *RetType = VPSelect->getType();
  Value *CondVal = VPSelect->getArgOperand(0);
  Value *TrueVal = VPSelect->getArgOperand(1);
  Value *FalseVal = VPSelect->getArgOperand(2);
  Value *VL = VPSelect->getVectorLengthParam();

  // If true and false values are the same, no need for the select.
  if (TrueVal == FalseVal)
    return replaceInstUsesWith(*VPSelect, TrueVal);

  // No need for a select when the cond is an allzeros or allones vector.
  if (match(CondVal, m_One()))
    return replaceInstUsesWith(*VPSelect, TrueVal);
  if (match(CondVal, m_Zero()))
    return replaceInstUsesWith(*VPSelect, FalseVal);

  // Merge two selects with the same condition value.
  if (auto *PrevVPSelect = dyn_cast<VPIntrinsic>(FalseVal))
    if (PrevVPSelect->getIntrinsicID() == Intrinsic::vp_select &&
        PrevVPSelect->getArgOperand(0) == CondVal &&
        PrevVPSelect->getVectorLengthParam() == VL)
      return replaceOperand(*VPSelect, 2, PrevVPSelect->getArgOperand(2));

  if (RetType->isIntOrIntVectorTy(1) &&
      CondVal->getType() == TrueVal->getType()) {
    VectorBuilder VB(Builder);
    VB.setMask(ConstantInt::getAllOnesValue(CondVal->getType()));
    VB.setEVL(VL);

    // If TrueVal is an allones vector, transform the select in an or.
    if (match(TrueVal, m_One())) {
      Value *Or = VB.createVectorInstruction(Instruction::Or, RetType,
                                             {CondVal, FalseVal});
      return replaceInstUsesWith(*VPSelect, Or);
    }

    // If FalseVal is an allzeros vector, transform the select in an and.
    if (match(FalseVal, m_Zero())) {
      Value *And = VB.createVectorInstruction(Instruction::And, RetType,
                                              {CondVal, TrueVal});
      return replaceInstUsesWith(*VPSelect, And);
    }
  }

  return nullptr;
}
