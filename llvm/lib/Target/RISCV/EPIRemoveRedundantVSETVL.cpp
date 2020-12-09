//===- EPIRemoveRedundantVSETVL.cpp - Remove redundant VSETVL instructions ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that removes the 'vsetvl' instructions
// that are known to have no effect, and thus are redundant.
//
// In particular, given a pair of 'vsetvli' instructions that specify the same
// SEW and VLMUL and have no instruction modifying the VL inbetween, the
// later can be safely removed in the following scenarios:
//
// - If we detect that the value passed as the requested vector length (AVL) is
//   found in the same virtual register in both instructions.
//
// - If we detect that the value passed as the requested vector length (AVL)
//   for the later 'vsetvli' instruction is actually defined by the prior
//   'vsetvli' (i.e. it is a granted vector length (GVL)).
//
// - If we detect a VLMAX definition that is only consumed by a non-VLMAX uses,
//   use, if possible, the AVL at the point of use.
//
// Currently, this phase requires SSA form, and the analysis is limited within
// a basic block.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVTargetMachine.h"
#include "RISCVRegisterInfo.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "epi-remove-redundant-vsetvl"

static cl::opt<bool>
    DisableRemoveVSETVL("no-epi-remove-redundant-vsetvl", cl::init(false),
                        cl::Hidden,
                        cl::desc("Disable removing redundant vsetvl"));

static cl::opt<bool>
    DisableVLBackPropagation("epi-disable-vl-backpropagation", cl::init(false),
                        cl::Hidden,
                        cl::desc("Disable VL backpropagation"));

namespace {

class EPIRemoveRedundantVSETVL : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid

  EPIRemoveRedundantVSETVL() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &F) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  // This pass modifies the program, but does not modify the CFG
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveVariables>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
};

char EPIRemoveRedundantVSETVL::ID = 0;

// This class holds information related to the operands of a VSETVLI
// instruction. It provides mechanisms to compare such instructions.
struct VSETVLInfo {
  Register AVLReg;
  unsigned SEW;
  unsigned VLMul;
  bool Nontemporal;

  VSETVLInfo(Register AVLReg, unsigned SEW, unsigned VLMul, bool Nontemporal)
      : AVLReg(AVLReg), SEW(SEW), VLMul(VLMul), Nontemporal(Nontemporal) {}

  VSETVLInfo(const MachineInstr &MI) {
    assert(MI.getOpcode() == RISCV::PseudoVSETVLI);

    const MachineOperand &AVLOp = MI.getOperand(1);
    assert(AVLOp.isReg());

    AVLReg = AVLOp.getReg();

    const MachineOperand &VTypeIOp = MI.getOperand(2);
    assert(VTypeIOp.isImm());

    unsigned VTypeI = VTypeIOp.getImm();

    unsigned SEWBits = (VTypeI >> 2) & 0x7;
    unsigned VMulBits = VTypeI & 0x3;

    Nontemporal = (VTypeI >> 9) & 0x1;

    SEW = (1 << SEWBits) * 8;
    VLMul = 1 << VMulBits;
  }

  bool sameOpKind(const VSETVLInfo &other) const {
    return this->Nontemporal == other.Nontemporal;
  }

  // A VSETVLI instruction has a 'more restrictive VType' if it entails a
  // smaller VLMAX. This is independent of the AVL operand.
  bool hasMoreRestrictiveVType(const VSETVLInfo &other) const {
    return sameOpKind(other) && (SEW / VLMul) > (other.SEW / other.VLMul);
  }

  bool hasMoreRestrictiveOrEqualVType(const VSETVLInfo &other) const {
    return sameOpKind(other) && ((SEW / VLMul) >= (other.SEW / other.VLMul));
  }

  bool computesSameGVL(const VSETVLInfo &other) const {
    return sameOpKind(other) && (AVLReg == other.AVLReg &&
                                 (SEW / VLMul) == (other.SEW / other.VLMul));
  }

  bool operator==(const VSETVLInfo &other) const {
    return sameOpKind(other) &&
           (AVLReg == other.AVLReg && SEW == other.SEW && VLMul == other.VLMul);
  }

  bool operator!=(const VSETVLInfo &other) const {
    return !(*this == other);
  }
};

bool removeDeadVSETVLInstructions(MachineBasicBlock &MBB,
                                  const MachineRegisterInfo &MRI) {
  bool IsMBBModified = false;

  for (MachineBasicBlock::instr_iterator II = MBB.instr_begin(),
                                         IIEnd = MBB.instr_end();
       II != IIEnd;) {
    MachineInstr *MI(&*II++);

    bool RemovedLastUse;
    do {
      RemovedLastUse = false;

      if (MI->getOpcode() != RISCV::PseudoVSETVLI)
        continue;

      assert(MI->getNumExplicitOperands() == 3);
      assert(MI->getNumOperands() == 5);

      MachineOperand GVLOp = MI->getOperand(0);
      GVLOp.clearParent();
      assert(GVLOp.isReg());

      MachineOperand AVLOp = MI->getOperand(1);
      AVLOp.clearParent();
      assert(AVLOp.isReg());

      MachineOperand ImplVLOp = MI->getOperand(3);
      ImplVLOp.clearParent();
      MachineOperand ImplVTypeOp = MI->getOperand(4);
      ImplVTypeOp.clearParent();

      assert(ImplVLOp.isImplicit() && ImplVLOp.isReg());
      assert(ImplVTypeOp.isImplicit() && ImplVTypeOp.isReg());

      if (GVLOp.isDead() && ImplVLOp.isDead() && ImplVTypeOp.isDead()) {
        LLVM_DEBUG(dbgs() << "Erase trivially dead VSETVLI instruction:\n";
                   MI->dump(); dbgs() << "\n");
        MI->eraseFromParent();
        IsMBBModified = true;

        if (Register::isVirtualRegister(AVLOp.getReg())) {
          // Check if by removing this instruction another def can be set dead.
          MachineInstr *DefMI = MRI.getUniqueVRegDef(AVLOp.getReg());
          assert(DefMI != nullptr && "Expected MachineInstr defining AVLOp");
          MachineOperand *DefMO = DefMI->findRegisterDefOperand(AVLOp.getReg());
          assert(DefMO != nullptr && "Expected MachineOperand defining AVLOp");

          if (MRI.use_nodbg_empty(AVLOp.getReg())) {
            DefMO->setIsDead();
            RemovedLastUse = true;
            MI = DefMI;
          }
        } else {
          // Instruction does not modify VL or sets it to VLMAX.
          assert(AVLOp.getReg() == RISCV::X0);
          assert(Register::isVirtualRegister(GVLOp.getReg()) ||
                 GVLOp.getReg() == RISCV::X0);
        }
      }
    } while (RemovedLastUse);
  }

  return IsMBBModified;
}

bool forwardCompatibleAVLToGVLUses(const MachineRegisterInfo &MRI,
                                   const MachineInstr &OriginalMI,
                                   const Register &GVLReg,
                                   const Register &AVLReg) {
  bool Modified = false;
  assert(MRI.hasOneDef(GVLReg));
  for (MachineRegisterInfo::use_nodbg_iterator UI = MRI.use_nodbg_begin(GVLReg),
                                               UIEnd = MRI.use_nodbg_end();
       UI != UIEnd;) {
    MachineOperand &Use(*UI++);
    assert(Use.getParent() != nullptr);
    const MachineInstr &UseInstr = *Use.getParent();

    if (UseInstr.getOpcode() != RISCV::PseudoVSETVLI)
      continue;

    // Ensure use is AVL operand
    assert(UseInstr.getOperandNo(&Use) == 1);

    if (!VSETVLInfo(OriginalMI).hasMoreRestrictiveVType(VSETVLInfo(UseInstr))) {
      LLVM_DEBUG(dbgs() << "Forward AVL from VSETVLI instruction:\n";
                 OriginalMI.dump(); dbgs() << "to VSETVLI instruction:\n";
                 UseInstr.dump(); dbgs() << "\n");
      Use.setReg(AVLReg);
      Modified = true;

      // Stop forwarding AVL when we find a more restrictive VSETVLI. Otherwise
      // the outcome GVL can be greater than in the original code. Eg.
      // gvl  = vsetvli avl,  e32, m1
      // gvl2 = vsetvli gvl,  e16, m1
      // gvl3 = vsetvli gvl2, e64, m1 // Can't forward AVL past this instruction
      //        vsetvli gvl3, e32, m1
      continue;
    }

    const MachineOperand &NewGVLOp = UseInstr.getOperand(0);
    assert(NewGVLOp.isReg());
    if (!NewGVLOp.isDead())
      Modified |= forwardCompatibleAVLToGVLUses(MRI, OriginalMI,
                                                NewGVLOp.getReg(), AVLReg);
  }

  // Update liveness.
  if (MRI.use_nodbg_empty(GVLReg)) {
    assert(Register::isVirtualRegister(GVLReg));
    MachineRegisterInfo::def_iterator GVLOpIt = MRI.def_begin(GVLReg);
    assert(GVLOpIt != MRI.def_end());
    MachineOperand &GVLOp = *GVLOpIt;
    GVLOp.setIsDead();
  }

  return Modified;
}

bool forwardCompatibleAVL(MachineBasicBlock &MBB,
                          const MachineRegisterInfo &MRI) {
  bool IsMBBModified = false;
  for (MachineBasicBlock::instr_iterator II = MBB.instr_begin(),
                                         IIEnd = MBB.instr_end();
       II != IIEnd;) {
    MachineInstr &MI(*II++);

    if (MI.getOpcode() != RISCV::PseudoVSETVLI)
      continue;

    assert(MI.getNumExplicitOperands() == 3);
    assert(MI.getNumOperands() == 5);

    const MachineOperand &GVLOp = MI.getOperand(0);
    assert(GVLOp.isReg());

    const MachineOperand &AVLOp = MI.getOperand(1);
    assert(AVLOp.isReg());

    // Skip instruction if it does not modify VL.
    if (GVLOp.getReg() == RISCV::X0) {
      assert(AVLOp.getReg() == RISCV::X0 &&
             "Unexpected VSETVLI instruction (AVL != X0 while GVL == X0)");
      continue;
    }

    IsMBBModified |=
        forwardCompatibleAVLToGVLUses(MRI, MI, GVLOp.getReg(), AVLOp.getReg());
  }

  return IsMBBModified;
}

// Class used to compare VSETVLInfo keys based on the outcome GVL. In its terms,
// if a pair of (AVL, VType) tuples compute the same GVL, they are considered
// equal.
struct SameGVLKeyInfo {
  using DenseMapInfoReg = DenseMapInfo<Register>;
  using DenseMapInfoUnsigned = DenseMapInfo<unsigned>;

  static inline VSETVLInfo getEmptyKey() {
    return {DenseMapInfoReg::getEmptyKey(), DenseMapInfoUnsigned::getEmptyKey(),
            DenseMapInfoUnsigned::getEmptyKey(),
            !!DenseMapInfoUnsigned::getEmptyKey()};
  }

  static inline VSETVLInfo getTombstoneKey() {
    return {DenseMapInfoReg::getTombstoneKey(),
            DenseMapInfoUnsigned::getTombstoneKey(),
            DenseMapInfoUnsigned::getTombstoneKey(),
            !!DenseMapInfoUnsigned::getTombstoneKey()};
  }

  static unsigned getHashValue(const VSETVLInfo &Val) {
    return DenseMapInfoReg::getHashValue(Val.AVLReg) ^
           (DenseMapInfoUnsigned::getHashValue(Val.SEW / Val.VLMul) << 1) ^
           (DenseMapInfoUnsigned::getHashValue(Val.Nontemporal) << 2);
  }

  static bool isEqual(const VSETVLInfo &LHS, const VSETVLInfo &RHS) {
    return LHS.computesSameGVL(RHS);
  }
};

bool forwardCompatibleGVL(MachineBasicBlock &MBB,
                          const MachineRegisterInfo &MRI) {

  // Map VSETVLInfo (representing VSETVLI input parameters) to the
  // corresponding computed GVL GPR.
  typedef DenseMap<VSETVLInfo, Register, SameGVLKeyInfo> VSETVLInfoMap_t;
  VSETVLInfoMap_t VSETVLInfoMap;

  bool IsMBBModified = false;
  for (MachineBasicBlock::instr_iterator II = MBB.instr_begin(),
                                         IIEnd = MBB.instr_end();
       II != IIEnd;) {
    MachineInstr &MI(*II++);

    if (MI.getOpcode() != RISCV::PseudoVSETVLI)
      continue;

    assert(MI.getNumExplicitOperands() == 3);
    assert(MI.getNumOperands() == 5);

    MachineOperand &GVLOp = MI.getOperand(0);
    assert(GVLOp.isReg());

    MachineOperand &AVLOp = MI.getOperand(1);
    assert(AVLOp.isReg());

    const MachineOperand &ImplVLOp = MI.getOperand(3);
    const MachineOperand &ImplVTypeOp = MI.getOperand(4);

    assert(ImplVLOp.isImplicit() && ImplVLOp.isReg());
    assert(ImplVTypeOp.isImplicit() && ImplVTypeOp.isReg());

    assert(ImplVTypeOp.isDead() == ImplVLOp.isDead());

    VSETVLInfo VI = VSETVLInfo(MI);
    // Find the VSETVLI instruction up in the AVL - GVL chain that actually
    // determines the GVL. That is, the VSETVLI with the most restrictive VType
    // (and thus the VSETVLI that produces the smallest GVL).
    if (Register::isVirtualRegister(AVLOp.getReg())) {
      MachineInstr *ParentMI = MRI.getUniqueVRegDef(AVLOp.getReg());
      assert(ParentMI != nullptr);
      // Given that forwardCompatibleGVL is run after forwardCompatibleAVL, we
      // should find the most restrictive VSETVLI instruction within one jump in
      // the AVL - GVL chain. Otherwise we would need a loop here.
      if (ParentMI->getOpcode() == RISCV::PseudoVSETVLI &&
          !VI.hasMoreRestrictiveVType(VSETVLInfo(*ParentMI))) {
        // Ensure is GVL op.
        assert(ParentMI->getOperand(0).isReg() &&
               ParentMI->getOperand(0).getReg() == AVLOp.getReg());

        VI = VSETVLInfo(*ParentMI);
      }
    } else {
      assert(AVLOp.getReg() == RISCV::X0);
      // Skip instruction if it does not modify VL.
      if (GVLOp.getReg() == RISCV::X0)
        continue;

      // Instruction computes VLMAX.
      assert(Register::isVirtualRegister(GVLOp.getReg()));
    }

    VSETVLInfoMap_t::const_iterator I = VSETVLInfoMap.find(VI);
    if (I == VSETVLInfoMap.end()) {
      VSETVLInfoMap[VI] = GVLOp.getReg();
    } else if (ImplVTypeOp.isDead()) {
      Register PrevGVLReg = I->second;
      // Replace all uses.
      if (!MRI.use_nodbg_empty(GVLOp.getReg())) {
        assert(!GVLOp.isDead());
        for (MachineRegisterInfo::use_nodbg_iterator
                 UI = MRI.use_nodbg_begin(GVLOp.getReg()),
                 UIEnd = MRI.use_nodbg_end();
             UI != UIEnd;) {
          MachineOperand &Use(*UI++);
          Use.setReg(PrevGVLReg);
          IsMBBModified = true;
        }

        assert(Register::isVirtualRegister(PrevGVLReg));
        assert(MRI.hasOneDef(PrevGVLReg));
        MachineOperand &PrevGVLOp = *MRI.def_begin(PrevGVLReg);
        LLVM_DEBUG(dbgs() << "Forward GVL from VSETVLI instruction:\n";
                   PrevGVLOp.getParent()->dump();
                   dbgs() << "to VSETVLI instruction:\n"; MI.dump();
                   dbgs() << "\n");
        if (PrevGVLOp.isDead()) {
          // Now it has become alive.
          PrevGVLOp.setIsDead(false);
        }
        // No uses left, thus dead.
        GVLOp.setIsDead();
      }
    }
  }

  return IsMBBModified;
}

// FIXME: Adapted from GlobalISel/CombinerHelper.cpp.
bool isPredecessor(const MachineInstr &DefMI, const MachineInstr &UseMI) {
  assert(!DefMI.isDebugInstr() && !UseMI.isDebugInstr() &&
         "shouldn't consider debug uses");
  assert(DefMI.getParent() == UseMI.getParent());
  if (&DefMI == &UseMI)
    return false;

  // Loop through the basic block until we find one of the instructions.
  MachineBasicBlock::const_iterator I = DefMI.getParent()->begin();
  for (; &*I != &DefMI && &*I != &UseMI; ++I)
    ;
  return &*I == &DefMI;
}

// FIXME: remove some of the repetition here.
bool backpropagateVLMax(MachineBasicBlock &MBB, MachineRegisterInfo &MRI,
                        const RISCVInstrInfo *TII) {
  MachineInstr *CurrentVL = nullptr;
  // Maps each instruction to its defining PseudoVSETVLI (if any).
  llvm::DenseMap<MachineInstr *, MachineInstr *> RegionVL;
  for (MachineBasicBlock::instr_iterator II = MBB.instr_begin(),
                                         IIEnd = MBB.instr_end();
       II != IIEnd;) {
    MachineInstr &MI(*II++);

    if (MI.getOpcode() == RISCV::PseudoVSETVLI) {
      // This should not change VL so let's assume we cant't tell much about it.
      if (MI.getOperand(0).getReg() == RISCV::X0 &&
          MI.getOperand(1).getReg() == RISCV::X0)
        CurrentVL = nullptr;
      else
        CurrentVL = &MI;
      continue;
    }

    // Check if the current intruction defines VL (e.g. 'vsetvl', (but not
    // 'vsetvli')). If it does, we should not remove a subsequent 'vsetvli',
    // even when its vtype matches the reference 'vsetvli's. To force this
    // we clear 'CurrentVL'.
    for (auto const &Def : MI.defs()) {
      assert(Def.isReg());
      if (Def.getReg() == RISCV::VL) {
        CurrentVL = nullptr;
      }
    }

    // Implicit defs are not included in MachineInstruction::defs()
    for (auto const &ImplOp : MI.implicit_operands()) {
      if (ImplOp.isReg() && (ImplOp.getReg() == RISCV::VL) && ImplOp.isDef()) {
        CurrentVL = nullptr;
      }
    }

    // VL may be changed within functions, we can't reuse defs through calls
    if (MI.isCall()) {
      CurrentVL = nullptr;
    }

    RegionVL.insert(std::make_pair(&MI, CurrentVL));
  }

  auto DefinesVector =
      [&MRI](const MachineInstr &MI) -> llvm::Optional<Register> {
    for (auto const &DefOp : MI.defs()) {
      if (!DefOp.isReg())
        continue;
      Register Def = DefOp.getReg();
      if (!Register::isVirtualRegister(Def))
        continue;
      const TargetRegisterClass *RC = MRI.getRegClass(Def);
      if (!RC->hasSuperClassEq(&RISCV::VRRegClass) &&
          !RC->hasSuperClassEq(&RISCV::VRM2RegClass) &&
          !RC->hasSuperClassEq(&RISCV::VRM4RegClass) &&
          !RC->hasSuperClassEq(&RISCV::VRM8RegClass))
        continue;
      return Def;
    }
    return NoneType();
  };

  bool Changed = false;
  for (MachineBasicBlock::reverse_instr_iterator II = MBB.instr_rbegin(),
                                                 IIEnd = MBB.instr_rend();
       II != IIEnd;) {
    MachineInstr &MI(*II++);

    LLVM_DEBUG(dbgs() << "(1) Looking at: "; MI.dump(););

    llvm::Optional<Register> Def = DefinesVector(MI);
    if (!Def.hasValue())
      continue;

    LLVM_DEBUG(dbgs() << "(2) Looking at: "; MI.dump(););

    CurrentVL = RegionVL[&MI];
    if (!CurrentVL)
      continue;

    LLVM_DEBUG(dbgs() << "(3) Looking at: "; MI.dump(););

    // Check that we are using VLMAX.
    VSETVLInfo CurrentVLInfo(*CurrentVL);
    if (CurrentVLInfo.AVLReg != RISCV::X0) {
      continue;
    }
    LLVM_DEBUG(dbgs() << "(3.1) Looking at: "; MI.dump(););

    // Check for every use they are all the same
    MachineInstr *RefUseMIVL = nullptr;
    for (auto &UseMI : MRI.use_nodbg_instructions(*Def)) {
      MachineInstr *UseMIVL = RegionVL[&UseMI];
      if (!UseMIVL) {
        RefUseMIVL = nullptr;
        break;
      }

      LLVM_DEBUG(dbgs() << "(4) Looking at: "; MI.dump();
                 dbgs() << "    Used by: "; UseMI.dump());

      VSETVLInfo UseMIVLInfo(*UseMIVL);
      if (UseMIVLInfo.AVLReg == RISCV::X0) {
        RefUseMIVL = nullptr;
        break;
      }

      LLVM_DEBUG(dbgs() << "(5) Looking at: "; MI.dump();
                 dbgs() << "    Used by: "; UseMI.dump());

      // If the VL of the use computes less or equal number of VL we can use
      // this.
      if (!UseMIVLInfo.hasMoreRestrictiveOrEqualVType(CurrentVLInfo)) {
        RefUseMIVL = nullptr;
        break;
      }

      LLVM_DEBUG(dbgs() << "(6) Looking at: "; MI.dump();
                 dbgs() << "    Used by: "; UseMI.dump());

      // Ok, so far we know that this vector is used in an instruction that is
      // not under a X0 and will definitely use less elements (or the same) as
      // its AVLReg.

      // Now check AVLReg is actually defined before this instruction.
      MachineInstr *DefAVL = MRI.getUniqueVRegDef(UseMIVLInfo.AVLReg);
      assert(DefAVL && "Definition for AVL not found?");
      LLVM_DEBUG(dbgs() << "(7) Looking at: "; MI.dump();
                 dbgs() << "    Used by: "; UseMI.dump();
                 dbgs() << "    AVL constrained: "; DefAVL->dump());
      if (DefAVL->getParent() != MI.getParent()) {
        LLVM_DEBUG(dbgs() << "(7.1) Not in the same basic block\n");
        RefUseMIVL = nullptr;
        break;
      }
      if (!isPredecessor(*DefAVL, MI)) {
        LLVM_DEBUG(dbgs() << "(7.2) Not a predecessor\n");
        RefUseMIVL = nullptr;
        break;
      }

      if (!RefUseMIVL) {
        RefUseMIVL = UseMIVL;
      } else {
        // Check it matches the current one.
        if (VSETVLInfo(*RefUseMIVL) != UseMIVLInfo) {
          LLVM_DEBUG(dbgs() << "(7.3) Does not match current VL predecessor\n");
          RefUseMIVL = nullptr;
          break;
        }
      }
    }

    if (!RefUseMIVL) {
      LLVM_DEBUG(dbgs() << "(7.4) Giving up\n");
      continue;
    }

    LLVM_DEBUG(dbgs() << "(8) Constraining: "; MI.dump(););

    // When we run this pass, CurrentVL is the instruction preceding the current
    // one so we can remove it. If we can't, then things will still be correct
    // but will look a bit off.
    if (&*II == CurrentVL) {
      LLVM_DEBUG(dbgs() << "(8.1) Removing Useless VSETVLI: "; MI.dump(););
      II++;
      CurrentVL->eraseFromParent();
    }

    LLVM_DEBUG(dbgs() << "(8.2) Adding new VSETVLI "; MI.dump(););
    // All is good, so it should be possible to introduce a VSETVLI right here
    // using AVLReg.
    BuildMI(*MI.getParent(), &MI, MI.getDebugLoc(),
            TII->get(RISCV::PseudoVSETVLI))
        .addReg(RISCV::X0, RegState::Define | RegState::Dead)
        .add(RefUseMIVL->getOperand(1))
        .add(RefUseMIVL->getOperand(2));

    // Now update RegionVL to reflect the new change so it can be used by
    // earlier instructions.
    RegionVL[&MI] = RefUseMIVL;
    Changed = true;
  }

  return Changed;
}

bool removeDuplicateVSETVLI(MachineBasicBlock &MBB) {
  bool IsMBBModified = false;

  MachineInstr *RefInstr = nullptr;
  for (MachineBasicBlock::instr_iterator II = MBB.instr_begin(),
                                         IIEnd = MBB.instr_end();
       II != IIEnd;) {
    MachineInstr &MI(*II++);

    if (MI.getOpcode() != RISCV::PseudoVSETVLI) {
      // Check if the current intruction defines VL (e.g. 'vsetvl', (but not
      // 'vsetvli')). If it does, we should not remove a subsequent 'vsetvli',
      // even when its vtype matches the reference 'vsetvli's. To force this
      // we clear 'RefInstr'.
      for (auto const &Def : MI.defs()) {
        assert(Def.isReg());
        if (Def.getReg() == RISCV::VL) {
          RefInstr = nullptr;
          continue;
        }
      }

      // Implicit defs are not included in MachineInstruction::defs()
      for (auto const &ImplOp : MI.implicit_operands()) {
        if (ImplOp.isReg() && (ImplOp.getReg() == RISCV::VL) &&
            ImplOp.isDef()) {
          RefInstr = nullptr;
          continue;
        }
      }

      // VL may be changed within functions, we can't reuse defs through calls
      if (MI.isCall()) {
        RefInstr = nullptr;
        continue;
      }

      continue;
    }

    assert(MI.getNumExplicitOperands() == 3);
    assert(MI.getNumOperands() == 5);

    if (RefInstr == nullptr) {
      RefInstr = &MI;
      continue;
    }

    assert(&MI != RefInstr);

    const MachineOperand &GVLOp = MI.getOperand(0);
    assert(GVLOp.isReg());

    if (GVLOp.isDead() && VSETVLInfo(MI) == VSETVLInfo(*RefInstr)) {
      LLVM_DEBUG(dbgs() << "Remove duplicate VSETVLI instruction:\n"; MI.dump();
                 dbgs() << "in favour of:\n"; RefInstr->dump(); dbgs() << "\n");
      MachineOperand &RefInstrImplVLOp = RefInstr->getOperand(3);
      MachineOperand &RefInstrImplVTypeOp = RefInstr->getOperand(4);

      assert(RefInstrImplVLOp.isImplicit() && RefInstrImplVLOp.isReg());
      assert(RefInstrImplVTypeOp.isImplicit() && RefInstrImplVTypeOp.isReg());
      assert(RefInstrImplVTypeOp.isDead() == RefInstrImplVLOp.isDead());

      const MachineOperand &ImplVLOp = MI.getOperand(3);
      const MachineOperand &ImplVTypeOp = MI.getOperand(4);

      assert(ImplVLOp.isImplicit() && ImplVLOp.isReg());
      assert(ImplVTypeOp.isImplicit() && ImplVTypeOp.isReg());
      assert(ImplVTypeOp.isDead() == ImplVLOp.isDead());

      if (RefInstrImplVTypeOp.isDead() && !ImplVTypeOp.isDead()) {
        // Now it has become alive.
        RefInstrImplVTypeOp.setIsDead(false);
        RefInstrImplVLOp.setIsDead(false);
      }

      IsMBBModified = true;

      MI.eraseFromParent();
      // MI has been removed, do not update RefInstr.
      continue;
    }

    RefInstr = &MI;
  }

  return IsMBBModified;
}

bool EPIRemoveRedundantVSETVL::runOnMachineFunction(MachineFunction &F) {
  if (skipFunction(F.getFunction()) || DisableRemoveVSETVL)
    return false;

  LLVM_DEBUG(
      dbgs() << "********** Begin remove redundant VSETVLI phase on function '"
             << F.getName() << "' **********\n\n");

  const RISCVInstrInfo *TII =
      static_cast<const RISCVInstrInfo *>(F.getSubtarget().getInstrInfo());
  MachineRegisterInfo &MRI = F.getRegInfo();
  assert(MRI.isSSA());
  assert(MRI.tracksLiveness());

  bool IsFunctionModified = false;

  for (MachineBasicBlock &MBB : F) {
    bool ForwardedAVL = forwardCompatibleAVL(MBB, MRI);
    if (ForwardedAVL) {
      LLVM_DEBUG(dbgs() << "--- BB dump after forwardCompatibleAVL ---";
                 MBB.dump(); dbgs() << "\n");
    }

    bool ForwardedGVL = forwardCompatibleGVL(MBB, MRI);
    if (ForwardedGVL) {
      LLVM_DEBUG(dbgs() << "--- BB dump after forwardCompatibleGVL ---";
                 MBB.dump(); dbgs() << "\n");
    }

    bool BackPropagateVLMax = false;
    if (!DisableVLBackPropagation) {
      LLVM_DEBUG(dbgs() << "-- Before backpropagation\n";);
      BackPropagateVLMax = backpropagateVLMax(MBB, MRI, TII);
      LLVM_DEBUG(dbgs() << "-- After backpropagation\n\n";);
      if (BackPropagateVLMax) {
        LLVM_DEBUG(dbgs() << "--- BB dump after backpropagateVLMax ---";
                   MBB.dump(); dbgs() << "\n");
      }
    }

    bool RemovedDuplicates = removeDuplicateVSETVLI(MBB);
    if (RemovedDuplicates) {
      LLVM_DEBUG(dbgs() << "--- BB dump after removeDuplicateVSETVLI ---";
                 MBB.dump(); dbgs() << "\n");
    }

    IsFunctionModified |=
        ForwardedAVL || ForwardedGVL || BackPropagateVLMax || RemovedDuplicates;
  }

  LiveVariables LV;
  LV.runOnMachineFunction(F);

  for (MachineBasicBlock &MBB : F) {
    bool RemovedDeadInstrs = removeDeadVSETVLInstructions(MBB, MRI);
    if (RemovedDeadInstrs) {
      LLVM_DEBUG(dbgs() << "--- BB dump after removeDeadVSETVLInstructions ---";
                 MBB.dump(); dbgs() << "\n");
      IsFunctionModified = true;
    }
  }

  LLVM_DEBUG(
      dbgs() << "*********** End remove redundant VSETVLI phase on function '"
             << F.getName() << "' ***********\n");

  return IsFunctionModified;
}

} // namespace

INITIALIZE_PASS(EPIRemoveRedundantVSETVL, "epi-remove-redundant-vsetvl",
                "EPI Remove Redundant VSETVL pass", false, false)
namespace llvm {

FunctionPass *createEPIRemoveRedundantVSETVLPass() {
  return new EPIRemoveRedundantVSETVL();
}

} // end of namespace llvm
