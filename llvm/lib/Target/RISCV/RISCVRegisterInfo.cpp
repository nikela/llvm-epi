//===-- RISCVRegisterInfo.cpp - RISCV Register Information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the RISCV implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "RISCVRegisterInfo.h"
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_REGINFO_TARGET_DESC
#include "RISCVGenRegisterInfo.inc"

using namespace llvm;

static_assert(RISCV::X1 == RISCV::X0 + 1, "Register list not consecutive");
static_assert(RISCV::X31 == RISCV::X0 + 31, "Register list not consecutive");
static_assert(RISCV::F1_F == RISCV::F0_F + 1, "Register list not consecutive");
static_assert(RISCV::F31_F == RISCV::F0_F + 31,
              "Register list not consecutive");
static_assert(RISCV::F1_D == RISCV::F0_D + 1, "Register list not consecutive");
static_assert(RISCV::F31_D == RISCV::F0_D + 31,
              "Register list not consecutive");
static_assert(RISCV::V1 == RISCV::V0 + 1, "Register list not consecutive");
static_assert(RISCV::V31 == RISCV::V0 + 31, "Register list not consecutive");

RISCVRegisterInfo::RISCVRegisterInfo(unsigned HwMode)
    : RISCVGenRegisterInfo(RISCV::X1, /*DwarfFlavour*/0, /*EHFlavor*/0,
                           /*PC*/0, HwMode) {}

const MCPhysReg *
RISCVRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  auto &Subtarget = MF->getSubtarget<RISCVSubtarget>();
  if (MF->getFunction().hasFnAttribute("interrupt")) {
    if (Subtarget.hasStdExtD())
      return CSR_XLEN_F64_Interrupt_SaveList;
    if (Subtarget.hasStdExtF())
      return CSR_XLEN_F32_Interrupt_SaveList;
    return CSR_Interrupt_SaveList;
  }

  switch (Subtarget.getTargetABI()) {
  default:
    llvm_unreachable("Unrecognized ABI");
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_LP64:
    return CSR_ILP32_LP64_SaveList;
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_LP64F:
    return CSR_ILP32F_LP64F_SaveList;
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64D:
    return CSR_ILP32D_LP64D_SaveList;
  }
}

BitVector RISCVRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  const RISCVSubtarget &Subtarget = MF.getSubtarget<RISCVSubtarget>();
  const RISCVFrameLowering *TFI = getFrameLowering(MF);
  BitVector Reserved(getNumRegs());

  // Mark any registers requested to be reserved as such
  for (size_t Reg = 0; Reg < getNumRegs(); Reg++) {
    if (MF.getSubtarget<RISCVSubtarget>().isRegisterReservedByUser(Reg))
      markSuperRegs(Reserved, Reg);
  }

  // Use markSuperRegs to ensure any register aliases are also reserved
  markSuperRegs(Reserved, RISCV::X0); // zero
  markSuperRegs(Reserved, RISCV::X2); // sp
  markSuperRegs(Reserved, RISCV::X3); // gp
  markSuperRegs(Reserved, RISCV::X4); // tp
  // When using EPI we need to reserve FP tentatively just in case there is a
  // spill.  Unfortunately we know there are spills _after_ the Register
  // Allocator has queried the reserved registers.
  //
  // TODO: Add a pass that undoes this.
  if (TFI->hasFP(MF) || Subtarget.hasStdExtV())
    markSuperRegs(Reserved, RISCV::X8); // fp
  // Reserve the base register if we need to realign the stack and allocate
  // variable-sized objects at runtime.
  if (TFI->hasBP(MF))
    markSuperRegs(Reserved, RISCVABI::getBPReg()); // bp

  // EPI registers
  markSuperRegs(Reserved, RISCV::VL);
  markSuperRegs(Reserved, RISCV::VTYPE);

  assert(checkAllSuperRegsMarked(Reserved));
  return Reserved;
}

bool RISCVRegisterInfo::isAsmClobberable(const MachineFunction &MF,
                                         MCRegister PhysReg) const {
  return !MF.getSubtarget<RISCVSubtarget>().isRegisterReservedByUser(PhysReg);
}

bool RISCVRegisterInfo::isConstantPhysReg(MCRegister PhysReg) const {
  return PhysReg == RISCV::X0;
}

const uint32_t *RISCVRegisterInfo::getNoPreservedMask() const {
  return CSR_NoRegs_RegMask;
}

// Frame indexes representing locations of CSRs which are given a fixed location
// by save/restore libcalls.
static const std::map<unsigned, int> FixedCSRFIMap = {
  {/*ra*/  RISCV::X1,   -1},
  {/*s0*/  RISCV::X8,   -2},
  {/*s1*/  RISCV::X9,   -3},
  {/*s2*/  RISCV::X18,  -4},
  {/*s3*/  RISCV::X19,  -5},
  {/*s4*/  RISCV::X20,  -6},
  {/*s5*/  RISCV::X21,  -7},
  {/*s6*/  RISCV::X22,  -8},
  {/*s7*/  RISCV::X23,  -9},
  {/*s8*/  RISCV::X24,  -10},
  {/*s9*/  RISCV::X25,  -11},
  {/*s10*/ RISCV::X26,  -12},
  {/*s11*/ RISCV::X27,  -13}
};

bool RISCVRegisterInfo::hasReservedSpillSlot(const MachineFunction &MF,
                                             Register Reg,
                                             int &FrameIdx) const {
  const auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  if (!RVFI->useSaveRestoreLibCalls())
    return false;

  auto FII = FixedCSRFIMap.find(Reg);
  if (FII == FixedCSRFIMap.end())
    return false;

  FrameIdx = FII->second;
  return true;
}

static Register computeVRSpillReloadInstructions(
    MachineBasicBlock::iterator II, const Register &VReg,
    const Register &HandleReg, unsigned LMUL, bool IsReload, bool KillVReg,
    Register VLenBReg = 0, bool KillHandle = false) {
  MachineBasicBlock &MBB = *II->getParent();
  MachineFunction &MF = *MBB.getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterInfo &RI = *MF.getSubtarget().getRegisterInfo();
  const RISCVInstrInfo &TII = *MF.getSubtarget<RISCVSubtarget>().getInstrInfo();
  DebugLoc DL = II->getDebugLoc();

  if (LMUL == 1) {
    if (IsReload)
      BuildMI(MBB, II, DL, TII.get(RISCV::VL1R_V), VReg)
          .addReg(HandleReg, getKillRegState(KillHandle || !VLenBReg));
    else
      BuildMI(MBB, II, DL, TII.get(RISCV::VS1R_V))
          .addReg(VReg, getKillRegState(KillVReg))
          .addReg(HandleReg, getKillRegState(KillHandle || !VLenBReg));

    return HandleReg;
  }

  Register VRegEven, VRegOdd;
  switch (LMUL) {
  default:
    llvm_unreachable("Unexpected LMUL value");
  case 2:
    VRegEven = RI.getSubReg(VReg, RISCV::sub_vrm2);
    VRegOdd = RI.getSubReg(VReg, RISCV::sub_vrm2_hi);
    break;
  case 4:
    VRegEven = RI.getSubReg(VReg, RISCV::sub_vrm4);
    VRegOdd = RI.getSubReg(VReg, RISCV::sub_vrm4_hi);
    break;
  case 8:
    VRegEven = RI.getSubReg(VReg, RISCV::sub_vrm8);
    VRegOdd = RI.getSubReg(VReg, RISCV::sub_vrm8_hi);
    break;
  }

  // Compute VLENB if it hasn't been computed.
  if (!VLenBReg) {
    VLenBReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    BuildMI(MBB, II, DL, TII.get(RISCV::PseudoReadVLENB), VLenBReg);

    // The handle should only be killed in the VS1R/VL1R for the last register
    // in the register group. This boolean is forwarded through the odd
    // recursive calls only.
    // Since VLENB hasn't been computed this must be the toplevel recursive
    // call.
    KillHandle = true;
  }

  // Recursive call on the even subregister. The handle corresponding to the
  // latest spill/reload in the recursion is returned as a result.
  Register LastHandleReg = computeVRSpillReloadInstructions(
      II, VRegEven, HandleReg, LMUL / 2, IsReload, KillVReg, VLenBReg);

  Register HandleRegOdd = MRI.createVirtualRegister(&RISCV::GPRRegClass);
  BuildMI(MBB, II, DL, TII.get(RISCV::ADD), HandleRegOdd)
      .addReg(LastHandleReg, RegState::Kill)
      .addReg(VLenBReg);

  // Recursive call on the odd subregister. Return latest spill/reload handle.
  return computeVRSpillReloadInstructions(II, VRegOdd, HandleRegOdd, LMUL / 2,
                                          IsReload, KillVReg, VLenBReg,
                                          KillHandle);
}

void RISCVRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                            int SPAdj, unsigned FIOperandNum,
                                            RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected non-zero SPAdj value");

  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const RISCVInstrInfo *TII = MF.getSubtarget<RISCVSubtarget>().getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  Register FrameReg;
  int Offset =
      getFrameLowering(MF)->getFrameIndexReference(MF, FrameIndex, FrameReg);

  bool NeedsIndirectAddressing = false;
  bool OffsetFits = false;
  int OffsetIndex = -1;
  // FIXME: Improve this to make it more robust.
  switch (MI.getOpcode()) {
  case RISCV::PseudoVLE_V_M1:
  case RISCV::PseudoVLE_V_M2:
  case RISCV::PseudoVLE_V_M4:
  case RISCV::PseudoVLE_V_M8:
  case RISCV::PseudoVSE_V_M1:
  case RISCV::PseudoVSE_V_M2:
  case RISCV::PseudoVSE_V_M4:
  case RISCV::PseudoVSE_V_M8:
    // FIXME: Are the following two needed now?
  case RISCV::VLE_V:
  case RISCV::VSE_V:
    // The following two are handled later in this function
  case RISCV::PseudoVSPILL_M1:
  case RISCV::PseudoVRELOAD_M1:
  case RISCV::PseudoVSPILL_M2:
  case RISCV::PseudoVRELOAD_M2:
  case RISCV::PseudoVSPILL_M4:
  case RISCV::PseudoVRELOAD_M4:
  case RISCV::PseudoVSPILL_M8:
  case RISCV::PseudoVRELOAD_M8:
    NeedsIndirectAddressing =
        MFI.getStackID(FrameIndex) == TargetStackID::EPIVector;
    break;
  default:
    OffsetIndex = FIOperandNum + 1;
    Offset += MI.getOperand(OffsetIndex).getImm();
    OffsetFits = isInt<12>(Offset);
    break;
  }

  if (!isInt<32>(Offset)) {
    report_fatal_error(
        "Frame offsets outside of the signed 32-bit range not supported");
  }

  MachineBasicBlock &MBB = *MI.getParent();

  if (NeedsIndirectAddressing) {
    assert(OffsetIndex == -1 && "There must not be offset");
    assert(MFI.getStackID(FrameIndex) == TargetStackID::EPIVector &&
           "Unexpected stack ID");

    MachineOperand SlotAddr = MI.getOperand(FIOperandNum);

    // TODO: Consider using loadRegFromStackSlot but this has to be before
    // replacing the FI above.
    unsigned LoadHandleOpcode =
        getRegSizeInBits(RISCV::GPRRegClass) == 32 ? RISCV::LW : RISCV::LD;
    Register HandleReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    MachineInstr *LoadHandle =
        BuildMI(MBB, II, DL, TII->get(LoadHandleOpcode), HandleReg)
            .add(SlotAddr)
            .addImm(0);

    // Handle vector spills here
    if (MI.getOpcode() == RISCV::PseudoVSPILL_M1 ||
        MI.getOpcode() == RISCV::PseudoVRELOAD_M1 ||
        MI.getOpcode() == RISCV::PseudoVSPILL_M2 ||
        MI.getOpcode() == RISCV::PseudoVRELOAD_M2 ||
        MI.getOpcode() == RISCV::PseudoVSPILL_M4 ||
        MI.getOpcode() == RISCV::PseudoVRELOAD_M4 ||
        MI.getOpcode() == RISCV::PseudoVSPILL_M8 ||
        MI.getOpcode() == RISCV::PseudoVRELOAD_M8) {

      // Make sure we spill/reload all the bits using whole register
      // instructions.
      MachineOperand &OpReg = MI.getOperand(0);
      bool IsReload;
      unsigned LMUL;
      switch (MI.getOpcode()) {
      default:
        llvm_unreachable("Unexpected instruction");
      case RISCV::PseudoVSPILL_M1:
        IsReload = false; LMUL = 1; break;
      case RISCV::PseudoVRELOAD_M1:
        IsReload = true;  LMUL = 1; break;
      case RISCV::PseudoVSPILL_M2:
        IsReload = false; LMUL = 2; break;
      case RISCV::PseudoVRELOAD_M2:
        IsReload = true;  LMUL = 2; break;
      case RISCV::PseudoVSPILL_M4:
        IsReload = false; LMUL = 4; break;
      case RISCV::PseudoVRELOAD_M4:
        IsReload = true;  LMUL = 4; break;
      case RISCV::PseudoVSPILL_M8:
        IsReload = false; LMUL = 8; break;
      case RISCV::PseudoVRELOAD_M8:
        IsReload = true;  LMUL = 8; break;
      }
      computeVRSpillReloadInstructions(II, OpReg.getReg(), HandleReg, LMUL,
                                       IsReload, /* KillVReg */ OpReg.isKill());

      // Remove the pseudo
      MI.eraseFromParent();
    } else {
      // Use the handle as address
      MI.getOperand(FIOperandNum)
          .ChangeToRegister(HandleReg, false, false, /* isKill */ true);
    }

    // Now remove the FI of the handle load
    return eliminateFrameIndex(LoadHandle, /* SPAdj */ 0, 1, RS);
  }

  bool FrameRegIsKill = false;

  if (!OffsetFits) {
    assert(isInt<32>(Offset) && "Int32 expected");
    // The offset won't fit in an immediate, so use a scratch register instead
    // Modify Offset and FrameReg appropriately
    Register ScratchReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    TII->movImm(MBB, II, DL, ScratchReg, Offset);
    BuildMI(MBB, II, DL, TII->get(RISCV::ADD), ScratchReg)
        .addReg(FrameReg)
        .addReg(ScratchReg, RegState::Kill);
    Offset = 0;
    FrameReg = ScratchReg;
    FrameRegIsKill = true;
  }

  MI.getOperand(FIOperandNum)
      .ChangeToRegister(FrameReg, false, false, FrameRegIsKill);
  if (OffsetIndex >= 0) {
    MI.getOperand(OffsetIndex).ChangeToImmediate(Offset);
  }
}

Register RISCVRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = getFrameLowering(MF);
  return TFI->hasFP(MF) ? RISCV::X8 : RISCV::X2;
}

const uint32_t *
RISCVRegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                        CallingConv::ID /*CC*/) const {
  const RISCVSubtarget &Subtarget = MF.getSubtarget<RISCVSubtarget>();
  if (MF.getFunction().hasFnAttribute("interrupt")) {
    if (Subtarget.hasStdExtD())
      return CSR_XLEN_F64_Interrupt_RegMask;
    if (Subtarget.hasStdExtF())
      return CSR_XLEN_F32_Interrupt_RegMask;
    return CSR_Interrupt_RegMask;
  }

  switch (Subtarget.getTargetABI()) {
  default:
    llvm_unreachable("Unrecognized ABI");
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_LP64:
    return CSR_ILP32_LP64_RegMask;
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_LP64F:
    return CSR_ILP32F_LP64F_RegMask;
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64D:
    return CSR_ILP32D_LP64D_RegMask;
  }
}

bool RISCVRegisterInfo::hasBasePointer(const MachineFunction &MF) const {
  // We use a BP when all of the following are true:
  // - the stack needs realignment (due to overaligned local objects)
  // - the stack has VLAs
  // Note that when we need a BP the conditions also imply a FP.
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  return needsStackRealignment(MF) &&
         (MFI.hasVarSizedObjects() || RVFI->hasSpilledVR());
}
