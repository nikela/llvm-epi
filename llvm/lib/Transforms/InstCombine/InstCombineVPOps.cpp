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

#define DEBUG_TYPE "instcombine"

using namespace llvm;
using namespace llvm::PatternMatch;

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
