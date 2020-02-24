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

int RISCVTTIImpl::getIntImmCostInst(unsigned Opcode, unsigned Idx, const APInt &Imm,
                                Type *Ty, TTI::TargetCostKind CostKind) {
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
  case Instruction::FMul:
  case Instruction::Add:
  case Instruction::Mul:
    return true;
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
