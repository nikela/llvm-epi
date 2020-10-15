//===- EPIFMAContraction.cpp - Contract FMA-like VPRed --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the wrong way and place to do this, but we made our lives a bit
// difficult by using intrinsics for many things.
//
// This pass only does the minimal contractions we care so far.
//
// Part of the code has been inspired by AArch64/SVEIntrinsicOpts
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "epi-fma-contraction"

static cl::opt<bool>
    DisableFMAContraction("no-epi-fma-contraction", cl::init(false), cl::Hidden,
                          cl::desc("Disable FMA contraction in EPI"));

#define PASS_NAME "EPI FMA Contraction"

namespace {

class EPIFMAContraction : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid

  EPIFMAContraction() : ModulePass(ID) { }

  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  StringRef getPassName() const override { return PASS_NAME; }

private:
  bool optimizeFunctions(SmallSetVector<Function *, 4> &Functions);
  bool optimizeIntrinsic(Instruction *I);

  bool tryFMADDContraction(IntrinsicInst *I, bool IsSub);
};

} // namespace

char EPIFMAContraction::ID = 0;
INITIALIZE_PASS(EPIFMAContraction, DEBUG_TYPE, PASS_NAME, false, false)

namespace llvm {

ModulePass *createEPIFMAContractionPass() { return new EPIFMAContraction(); }

} // end of namespace llvm

void EPIFMAContraction::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.setPreservesCFG();
}

bool EPIFMAContraction::tryFMADDContraction(IntrinsicInst *I, bool IsSub) {
  if (!I->getFastMathFlags().allowContract())
    return false;
  Value *X = nullptr;
  Value *Y = nullptr;
  Value *Z = nullptr;

  LLVMContext &Ctx = I->getContext();
  IRBuilder<> Builder(Ctx);
  Builder.SetInsertPoint(I);

  auto ComputeMatch = [&](int FirstOp, int SecondOp) {
    if (0) {
    } else if (match(I->getOperand(FirstOp),
                     m_Intrinsic<Intrinsic::vp_fmul>(
                         m_Intrinsic<Intrinsic::vp_fneg>(m_Value(X)),
                         m_Intrinsic<Intrinsic::vp_fneg>(m_Value(Y))))) {
      // (± (* (- x) (- y)) z) -> FMA(X, Y, Z)
      Z = I->getOperand(SecondOp);
    } else if (match(I->getOperand(FirstOp),
                     m_Intrinsic<Intrinsic::vp_fmul>(
                         m_Intrinsic<Intrinsic::vp_fneg>(m_Value()),
                         m_Value(Y)))) {
      // (± (* (- x) y) z) -> FMA(-X, Y, Z)
      X = cast<IntrinsicInst>(I->getOperand(FirstOp))->getOperand(0);
      Z = I->getOperand(SecondOp);
    } else if (match(I->getOperand(FirstOp),
                     m_Intrinsic<Intrinsic::vp_fmul>(
                         m_Value(Y),
                         m_Intrinsic<Intrinsic::vp_fneg>(m_Value())))) {
      // (± (* y (- x)) z) -> FMA(-X, Y, Z)
      X = cast<IntrinsicInst>(I->getOperand(FirstOp))->getOperand(1);
      Z = I->getOperand(SecondOp);
    } else if (match(I->getOperand(FirstOp),
                     m_Intrinsic<Intrinsic::vp_fmul>(m_Value(X), m_Value(Y)))) {
      // (± (* x y) z) - FMA(X, Y, Z)
      Z = I->getOperand(SecondOp);
    } else {
      return false;
    }

    return true;
  };

  auto ComputeMatchSub = [&](int FirstOp, int SecondOp) {
    assert(FirstOp == 1 && SecondOp == 0 && "Invalid ops");
    if (0) {
    } else if (match(I->getOperand(FirstOp),
                     m_Intrinsic<Intrinsic::vp_fmul>(
                         m_Intrinsic<Intrinsic::vp_fneg>(m_Value()),
                         m_Intrinsic<Intrinsic::vp_fneg>(m_Value(Y))))) {
      // (- z (* (-y) (-x))) -> FMA(-X, Y, Z)
      X = cast<IntrinsicInst>(I->getOperand(FirstOp))->getOperand(0);
      Z = I->getOperand(SecondOp);
      IsSub = false;
    } else if (match(I->getOperand(FirstOp),
                     m_Intrinsic<Intrinsic::vp_fmul>(
                         m_Value(Y),
                         m_Intrinsic<Intrinsic::vp_fneg>(m_Value(X))))) {
      // (- z (* y (-x))) -> FMA(X, Y, Z)
      Z = I->getOperand(SecondOp);
      IsSub = false;
    } else if (match(I->getOperand(FirstOp),
                     m_Intrinsic<Intrinsic::vp_fmul>(
                         m_Intrinsic<Intrinsic::vp_fneg>(m_Value(X)),
                         m_Value(Y)))) {
      // (- z (* (-x) y)) -> FMA(X, Y, Z)
      Z = I->getOperand(SecondOp);
      IsSub = false;
    } else if (match(I->getOperand(FirstOp),
                     m_Intrinsic<Intrinsic::vp_fmul>(m_Value(X), m_Value(Y)))) {
      // (- z (* x y)) -> FMA(-X, Y, Z)
      Z = I->getOperand(SecondOp);
      X = Builder.CreateIntrinsic(
          Intrinsic::vp_fneg, {X->getType()},
          {X, I->getOperand(3), I->getOperand(4), I->getOperand(5)}, nullptr,
          "vp.fneg");
      IsSub = false;
    } else {
      return false;
    }

    return true;
  };

  if (!ComputeMatch(0, 1) && (IsSub || !ComputeMatch(1, 0))
      && (!IsSub || !ComputeMatchSub(1, 0)))
    return false;

  assert(X && Y && Z && "Invalid match");

  if (IsSub) {
    Z = Builder.CreateIntrinsic(
        Intrinsic::vp_fneg, {Z->getType()},
        {Z, I->getOperand(3), I->getOperand(4), I->getOperand(5)}, nullptr,
        "vp.fneg");
  }

  Value *NewFMA = Builder.CreateIntrinsic(
      Intrinsic::vp_fma, {I->getOperand(0)->getType()},
      {X, Y, Z, I->getOperand(2), I->getOperand(3), I->getOperand(4),
       I->getOperand(5)},
      nullptr, IsSub ? "vp.contracted.fmsub" : "vp.contracted.fmadd");
  I->replaceAllUsesWith(NewFMA);
  I->eraseFromParent();

  return true;
}

bool EPIFMAContraction::optimizeIntrinsic(Instruction *I) {
  IntrinsicInst *IntrI = dyn_cast<IntrinsicInst>(I);
  if (!IntrI)
    return false;

  switch (IntrI->getIntrinsicID()) {
  case Intrinsic::vp_fadd:
    return tryFMADDContraction(IntrI, /* IsSub */ false);
  case Intrinsic::vp_fsub:
    return tryFMADDContraction(IntrI, /* IsSub */ true);
  default:
    return false;
  }
}

bool EPIFMAContraction::optimizeFunctions(
    SmallSetVector<Function *, 4> &Functions) {
  bool Changed = false;
  for (auto *F : Functions) {
    DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>(*F).getDomTree();

    // Traverse the DT with an rpo walk so we see defs before uses, allowing
    // simplification to be done incrementally.
    BasicBlock *Root = DT->getRoot();
    ReversePostOrderTraversal<BasicBlock *> RPOT(Root);
    for (auto *BB : RPOT)
      for (Instruction &I : make_early_inc_range(*BB))
        Changed |= optimizeIntrinsic(&I);
  }
  return Changed;
}

bool EPIFMAContraction::runOnModule(Module &M) {
  if (DisableFMAContraction)
    return false;

  bool Changed = false;
  SmallSetVector<Function *, 4> Functions;

  for (auto &F : M.getFunctionList()) {
    if (!F.isDeclaration())
      continue;

    switch (F.getIntrinsicID()) {
    case Intrinsic::vp_fadd:
    case Intrinsic::vp_fsub:
      for (auto I = F.user_begin(), E = F.user_end(); I != E;) {
        auto *Inst = dyn_cast<Instruction>(*I++);
        Functions.insert(Inst->getFunction());
      }
      break;
    default:
      break;
    }
  }

  if (!Functions.empty())
    Changed |= optimizeFunctions(Functions);

  return Changed;
}
