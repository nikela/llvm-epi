//===- EPIFoldBroadcast.cpp - Fold broadcasted scalar operands as scalars ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that folds the following patterns
//
//    %v1 = ...
//    %v2 = <vscale x ...> @llvm.int.epi.vmv.v.x(%scalar, %gvl)
//    %v3 = <vscale x ...> @llvm.int.epi.v<op>(%v1, %v2, %gvl)
//
//    %v1 = ...
//    %v2 = <vscale x ...> @llvm.int.epi.vfmv.v.f(%scalar, %gvl)
//    %v3 = <vscale x ...> @llvm.int.epi.v<op>(%v1, %v2, %gvl)
//
// into
//
//    %v1 = ...
//    %v3 = <vscale x ...> @llvm.int.epi.v<op>(%v1, %scalar, %gvl)
//
// This is beneficial because many intrinsics map to instructions that
// that can extend the scalar in place. This way we avoid a broadcast
// instruction.
//
// TODO:
// - Some operations are commutable and may have the broadcasted value in the
// first operand rather than the second. It should be possible to swap the
// operands to make the fold effective.
// - This analysis could be extended to phis. The uses of a phi whose all
// incoming values are broadcasts with the same gvl is eligible for folding.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsEPI.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "epi-fold-broadcast"

STATISTIC(NumFolds, "Number of EPI intrinsics with a scalar operand folded");
STATISTIC(NumBroadcastsRemoved, "Number of EPI broadcast intrinsics removed");

static cl::opt<bool>
    DisableFolding("no-epi-broadcast-folding", cl::init(false), cl::Hidden,
                   cl::desc("Disable folding of broadcast intrinsics"));

namespace {

class EPIFoldBroadcast : public FunctionPass {
private:
  void initRun();
  void determineFoldableUses(Instruction *Broadcast, Value *Scalar);
  bool foldBroadcasts(Function &F);

public:
  static char ID; // Pass identification, replacement for typeid

  EPIFoldBroadcast() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  struct FoldInfo {
    Value *Scalar;
    Instruction *Broadcast;
    Value *User;
  };

  SmallVector<FoldInfo, 16> FoldableUses;
};

} // namespace

char EPIFoldBroadcast::ID = 0;

INITIALIZE_PASS(EPIFoldBroadcast, DEBUG_TYPE, "EPI Fold Broadcast Intrinsics",
                false, false)
namespace llvm {

FunctionPass *createEPIFoldBroadcastPass() { return new EPIFoldBroadcast(); }

} // end of namespace llvm

void EPIFoldBroadcast::determineFoldableUses(Instruction *Broadcast,
                                             Value *Scalar) {
  // A broadcast has foldable uses if it looks like this
  //    %v1 = ...
  //    %v2 = <vscale x ...> @llvm.int.epi.vmv.v.x(%scalar, %gvl)
  //    %v3 = <vscale x ...> @llvm.int.epi.v<op>(%v1, %v2, %gvl)
  //
  //    %v1 = ...
  //    %v2 = <vscale x ...> @llvm.int.epi.vfmv.v.f(%scalar, %gvl)
  //    %v3 = <vscale x ...> @llvm.int.epi.v<op>(%v1, %v2, %gvl)
  LLVM_DEBUG(dbgs() << "Determining uses of broadcasted value ");
  LLVM_DEBUG(Broadcast->dump());
  for (auto &U : Broadcast->uses()) {
    User *R = U.getUser();
    // TODO: Extend this algorithm through phis
    if (auto *CBUser = dyn_cast<CallBase>(R)) {
      LLVM_DEBUG(dbgs() << "This is used in a call\n");
      LLVM_DEBUG(R->dump());
      Intrinsic::ID II = CBUser->getIntrinsicID();
      if (II == Intrinsic::not_intrinsic)
        continue;

      // Is this an EPI intrinsic with an extended operand and a GVL?
      const RISCVEPIIntrinsicsTable::EPIIntrinsicInfo *EII =
          RISCVEPIIntrinsicsTable::getEPIIntrinsicInfo(II);
      if (!EII || !EII->ExtendedOperand || !EII->GVLOperand)
        continue;

      assert(EII->ExtendedOperand > 0);
      unsigned ExtendedOp = EII->ExtendedOperand - 1;
      // Check that the broadcasted value is used in the extended operand.
      if (CBUser->getDataOperandNo(&U) != ExtendedOp)
        continue;
      // Note: we used to check that the GVL in both cases matches but
      // this seems unnecessary because:
      //  - if the values are the same, then exactly the same set
      //  of produced elements is consumed.
      //  - if the consumer uses a smaller gvl than the one used by the
      //  producer, then the consumer consumes a subset of the elements
      //  produced.
      //  - if the consumer uses a larger gvl than the one used by the
      //  producer, then the consumer will use more elements than those
      //  produced. Because we currently only fold unmasked broadcasts, this
      //  triggers undefined behaviour because elements beyond gvl are undef.

      // If the broadcast is used in other arguments than the extended operand,
      // this transformation is not beneficial.
      bool HasOtherBroadcastUses = false;
      for (unsigned Arg = 0, E = CBUser->arg_size();
           Arg < E && !HasOtherBroadcastUses; Arg++) {
        // This is the extended operand. Skip it.
        if (Arg == ExtendedOp)
          continue;

        HasOtherBroadcastUses = (CBUser->getArgOperand(Arg) == Broadcast);
      }

      if (HasOtherBroadcastUses)
        continue;

      LLVM_DEBUG(dbgs() << "Found a foldable use");
      LLVM_DEBUG(CBUser->dump());
      FoldInfo FI;
      FI.Scalar = Scalar;
      FI.Broadcast = Broadcast;
      FI.User = R;
      FoldableUses.push_back(FI);
    }
  }
}

bool EPIFoldBroadcast::foldBroadcasts(Function &F) {
  bool Changed = false;

  for (auto &FI : FoldableUses) {
    auto *CBUser = dyn_cast<CallBase>(FI.User);
    assert(CBUser && "This must be a call");

    Intrinsic::ID II = CBUser->getIntrinsicID();
    assert(II != Intrinsic::not_intrinsic &&
           "This must be a call to an intrinsic");
    const RISCVEPIIntrinsicsTable::EPIIntrinsicInfo *EII =
        RISCVEPIIntrinsicsTable::getEPIIntrinsicInfo(II);
    assert(EII && "This must be a known EPI intrinsic");

    FunctionType *FTy = CBUser->getCalledFunction()->getFunctionType();

    Type *ScalarTy = FI.Scalar->getType();

    LLVM_DEBUG(CBUser->print(dbgs()); dbgs() << "\n");

    SmallVector<Type *, 4> NewIntrinsicTypes;
    SmallVector<Value *, 4> NewOps;
    switch ((RISCVEPIIntrinsicsTable::EPIIntrClassID)EII->ClassID) {
    case RISCVEPIIntrinsicsTable::EPICIDBinary:
      LLVM_DEBUG(dbgs() << "Binary intrinsic\n");
      NewIntrinsicTypes = {FTy->getReturnType(), ScalarTy};
      NewOps = {CBUser->getArgOperand(0), FI.Scalar, CBUser->getArgOperand(2)};
      break;
    case RISCVEPIIntrinsicsTable::EPICIDBinaryMask:
      LLVM_DEBUG(dbgs() << "Binary intrinsic with mask\n");
      NewIntrinsicTypes = {FTy->getReturnType(), ScalarTy,
                           FTy->getParamType(3)};
      NewOps = {CBUser->getArgOperand(0), CBUser->getArgOperand(1), FI.Scalar,
                CBUser->getArgOperand(3), CBUser->getArgOperand(4)};
      break;
    case RISCVEPIIntrinsicsTable::EPICIDBinaryMaskIn:
      LLVM_DEBUG(dbgs() << "Binary intrinsic with mask input (without merge)\n");
      NewIntrinsicTypes = {FTy->getReturnType(), ScalarTy,
                           FTy->getParamType(2)};
      NewOps = {CBUser->getArgOperand(0), FI.Scalar, CBUser->getArgOperand(2),
                CBUser->getArgOperand(3)};
      break;
    case RISCVEPIIntrinsicsTable::EPICIDTernary:
      LLVM_DEBUG(dbgs() << "Ternary intrinsic\n");
      NewIntrinsicTypes = {FTy->getReturnType(), ScalarTy};
      NewOps = {CBUser->getArgOperand(0), FI.Scalar, CBUser->getArgOperand(2),
                CBUser->getArgOperand(3)};
      break;
    case RISCVEPIIntrinsicsTable::EPICIDTernaryMask:
      LLVM_DEBUG(dbgs() << "Ternary intrinsic with mask\n");
      NewIntrinsicTypes = {FTy->getReturnType(), ScalarTy,
                           FTy->getParamType(3)};
      NewOps = {CBUser->getArgOperand(0), FI.Scalar, CBUser->getArgOperand(2),
                CBUser->getArgOperand(3), CBUser->getArgOperand(4)};
      break;
    case RISCVEPIIntrinsicsTable::EPICIDBinaryWide:
      LLVM_DEBUG(dbgs() << "Binary intrinsic widening/narrowing\n");
      NewIntrinsicTypes = {FTy->getReturnType(), FTy->getParamType(0), ScalarTy};
      NewOps = {CBUser->getArgOperand(0), FI.Scalar, CBUser->getArgOperand(2)};
      break;
    case RISCVEPIIntrinsicsTable::EPICIDBinaryWideMask:
      LLVM_DEBUG(dbgs() << "Binary intrinsic widening/narrowing with mask\n");
      NewIntrinsicTypes = {FTy->getReturnType(), FTy->getParamType(1), ScalarTy,
                           FTy->getParamType(3)};
      NewOps = {CBUser->getArgOperand(0), CBUser->getArgOperand(1), FI.Scalar,
                CBUser->getArgOperand(3), CBUser->getArgOperand(4)};
      break;
    default:
      // We don't handle this class of intrinsics yet.
      LLVM_DEBUG(dbgs() << "Intrinsic not handled yet\n");
      continue;
      break;
    }
    Function *NewIntrinsic = Intrinsic::getDeclaration(
        F.getParent(), (Intrinsic::ID)EII->IntrinsicID, NewIntrinsicTypes);

    Instruction *CurrentCall = cast<Instruction>(FI.User);
    IRBuilder<> IRB(CurrentCall);
    CallInst *NewCall = IRB.CreateCall(NewIntrinsic, NewOps, "");

    LLVM_DEBUG(dbgs() << "Replacing ");
    LLVM_DEBUG(FI.User->dump());
    LLVM_DEBUG(dbgs() << "with ");
    LLVM_DEBUG(NewCall->dump());

    CurrentCall->replaceAllUsesWith(NewCall);
    CurrentCall->eraseFromParent();
    NumFolds++;

    if (isInstructionTriviallyDead(FI.Broadcast)) {
      LLVM_DEBUG(
          dbgs() << "Removing broadcast that has become trivially dead ");
      LLVM_DEBUG(FI.Broadcast->dump());
      FI.Broadcast->eraseFromParent();
      NumBroadcastsRemoved++;
    }

    Changed = true;
  }

  return Changed;
}

void EPIFoldBroadcast::initRun() { FoldableUses.clear(); }

bool EPIFoldBroadcast::runOnFunction(Function &F) {
  if (skipFunction(F) || DisableFolding)
    return false;

  initRun();

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        if (CB->getIntrinsicID() == Intrinsic::epi_vmv_v_x ||
            CB->getIntrinsicID() == Intrinsic::epi_vfmv_v_f) {
          determineFoldableUses(CB, CB->getArgOperand(0));
        }
      }
    }
  }

  return foldBroadcasts(F);
}
