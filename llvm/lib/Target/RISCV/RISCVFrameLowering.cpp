//===-- RISCVFrameLowering.cpp - RISCV Frame Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the RISCV implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "RISCVFrameLowering.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/MC/MCDwarf.h"

#define DEBUG_TYPE "prologepilog"

using namespace llvm;

// For now we use x18, a.k.a s2, as pointer to shadow call stack.
// User should explicitly set -ffixed-x18 and not use x18 in their asm.
static void emitSCSPrologue(MachineFunction &MF, MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            const DebugLoc &DL) {
  if (!MF.getFunction().hasFnAttribute(Attribute::ShadowCallStack))
    return;

  const auto &STI = MF.getSubtarget<RISCVSubtarget>();
  Register RAReg = STI.getRegisterInfo()->getRARegister();

  // Do not save RA to the SCS if it's not saved to the regular stack,
  // i.e. RA is not at risk of being overwritten.
  std::vector<CalleeSavedInfo> &CSI = MF.getFrameInfo().getCalleeSavedInfo();
  if (std::none_of(CSI.begin(), CSI.end(),
                   [&](CalleeSavedInfo &CSR) { return CSR.getReg() == RAReg; }))
    return;

  Register SCSPReg = RISCVABI::getSCSPReg();

  auto &Ctx = MF.getFunction().getContext();
  if (!STI.isRegisterReservedByUser(SCSPReg)) {
    Ctx.diagnose(DiagnosticInfoUnsupported{
        MF.getFunction(), "x18 not reserved by user for Shadow Call Stack."});
    return;
  }

  const auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  if (RVFI->useSaveRestoreLibCalls(MF)) {
    Ctx.diagnose(DiagnosticInfoUnsupported{
        MF.getFunction(),
        "Shadow Call Stack cannot be combined with Save/Restore LibCalls."});
    return;
  }

  const RISCVInstrInfo *TII = STI.getInstrInfo();
  bool IsRV64 = STI.hasFeature(RISCV::Feature64Bit);
  int64_t SlotSize = STI.getXLen() / 8;
  // Store return address to shadow call stack
  // s[w|d]  ra, 0(s2)
  // addi    s2, s2, [4|8]
  BuildMI(MBB, MI, DL, TII->get(IsRV64 ? RISCV::SD : RISCV::SW))
      .addReg(RAReg)
      .addReg(SCSPReg)
      .addImm(0);
  BuildMI(MBB, MI, DL, TII->get(RISCV::ADDI))
      .addReg(SCSPReg, RegState::Define)
      .addReg(SCSPReg)
      .addImm(SlotSize);
}

static void emitSCSEpilogue(MachineFunction &MF, MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            const DebugLoc &DL) {
  if (!MF.getFunction().hasFnAttribute(Attribute::ShadowCallStack))
    return;

  const auto &STI = MF.getSubtarget<RISCVSubtarget>();
  Register RAReg = STI.getRegisterInfo()->getRARegister();

  // See emitSCSPrologue() above.
  std::vector<CalleeSavedInfo> &CSI = MF.getFrameInfo().getCalleeSavedInfo();
  if (std::none_of(CSI.begin(), CSI.end(),
                   [&](CalleeSavedInfo &CSR) { return CSR.getReg() == RAReg; }))
    return;

  Register SCSPReg = RISCVABI::getSCSPReg();

  auto &Ctx = MF.getFunction().getContext();
  if (!STI.isRegisterReservedByUser(SCSPReg)) {
    Ctx.diagnose(DiagnosticInfoUnsupported{
        MF.getFunction(), "x18 not reserved by user for Shadow Call Stack."});
    return;
  }

  const auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  if (RVFI->useSaveRestoreLibCalls(MF)) {
    Ctx.diagnose(DiagnosticInfoUnsupported{
        MF.getFunction(),
        "Shadow Call Stack cannot be combined with Save/Restore LibCalls."});
    return;
  }

  const RISCVInstrInfo *TII = STI.getInstrInfo();
  bool IsRV64 = STI.hasFeature(RISCV::Feature64Bit);
  int64_t SlotSize = STI.getXLen() / 8;
  // Load return address from shadow call stack
  // l[w|d]  ra, -[4|8](s2)
  // addi    s2, s2, -[4|8]
  BuildMI(MBB, MI, DL, TII->get(IsRV64 ? RISCV::LD : RISCV::LW))
      .addReg(RAReg, RegState::Define)
      .addReg(SCSPReg)
      .addImm(-SlotSize);
  BuildMI(MBB, MI, DL, TII->get(RISCV::ADDI))
      .addReg(SCSPReg, RegState::Define)
      .addReg(SCSPReg)
      .addImm(-SlotSize);
}

// Get the ID of the libcall used for spilling and restoring callee saved
// registers. The ID is representative of the number of registers saved or
// restored by the libcall, except it is zero-indexed - ID 0 corresponds to a
// single register.
static int getLibCallID(const MachineFunction &MF,
                        const std::vector<CalleeSavedInfo> &CSI) {
  const auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();

  if (CSI.empty() || !RVFI->useSaveRestoreLibCalls(MF))
    return -1;

  Register MaxReg = RISCV::NoRegister;
  for (auto &CS : CSI)
    // RISCVRegisterInfo::hasReservedSpillSlot assigns negative frame indexes to
    // registers which can be saved by libcall.
    if (CS.getFrameIdx() < 0)
      MaxReg = std::max(MaxReg.id(), CS.getReg().id());

  if (MaxReg == RISCV::NoRegister)
    return -1;

  switch (MaxReg) {
  default:
    llvm_unreachable("Something has gone wrong!");
  case /*s11*/ RISCV::X27: return 12;
  case /*s10*/ RISCV::X26: return 11;
  case /*s9*/  RISCV::X25: return 10;
  case /*s8*/  RISCV::X24: return 9;
  case /*s7*/  RISCV::X23: return 8;
  case /*s6*/  RISCV::X22: return 7;
  case /*s5*/  RISCV::X21: return 6;
  case /*s4*/  RISCV::X20: return 5;
  case /*s3*/  RISCV::X19: return 4;
  case /*s2*/  RISCV::X18: return 3;
  case /*s1*/  RISCV::X9:  return 2;
  case /*s0*/  RISCV::X8:  return 1;
  case /*ra*/  RISCV::X1:  return 0;
  }
}

// Get the name of the libcall used for spilling callee saved registers.
// If this function will not use save/restore libcalls, then return a nullptr.
static const char *
getSpillLibCallName(const MachineFunction &MF,
                    const std::vector<CalleeSavedInfo> &CSI) {
  static const char *const SpillLibCalls[] = {
    "__riscv_save_0",
    "__riscv_save_1",
    "__riscv_save_2",
    "__riscv_save_3",
    "__riscv_save_4",
    "__riscv_save_5",
    "__riscv_save_6",
    "__riscv_save_7",
    "__riscv_save_8",
    "__riscv_save_9",
    "__riscv_save_10",
    "__riscv_save_11",
    "__riscv_save_12"
  };

  int LibCallID = getLibCallID(MF, CSI);
  if (LibCallID == -1)
    return nullptr;
  return SpillLibCalls[LibCallID];
}

// Get the name of the libcall used for restoring callee saved registers.
// If this function will not use save/restore libcalls, then return a nullptr.
static const char *
getRestoreLibCallName(const MachineFunction &MF,
                      const std::vector<CalleeSavedInfo> &CSI) {
  static const char *const RestoreLibCalls[] = {
    "__riscv_restore_0",
    "__riscv_restore_1",
    "__riscv_restore_2",
    "__riscv_restore_3",
    "__riscv_restore_4",
    "__riscv_restore_5",
    "__riscv_restore_6",
    "__riscv_restore_7",
    "__riscv_restore_8",
    "__riscv_restore_9",
    "__riscv_restore_10",
    "__riscv_restore_11",
    "__riscv_restore_12"
  };

  int LibCallID = getLibCallID(MF, CSI);
  if (LibCallID == -1)
    return nullptr;
  return RestoreLibCalls[LibCallID];
}

bool RISCVFrameLowering::hasFP(const MachineFunction &MF) const {
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();

  // We use a FP when any of the following is true:
  // - we are told to have one
  // - the stack needs realignment (due to overaligned local objects)
  // - the stack has VLAs
  // - the function has to spill VR vectors
  // - the function uses @llvm.frameaddress
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MF.getTarget().Options.DisableFramePointerElim(MF) ||
         RegInfo->needsStackRealignment(MF) || MFI.hasVarSizedObjects() ||
         RVFI->hasSpilledVR() || MFI.isFrameAddressTaken();
}

bool RISCVFrameLowering::hasBP(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();

  return MFI.hasVarSizedObjects() && TRI->needsStackRealignment(MF);
}

void RISCVFrameLowering::determineFrameLayout(MachineFunction &MF) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();

  // Get the number of bytes to allocate from the FrameInfo.
  uint64_t FrameSize = MFI.getStackSize();

  // Account all VR_SPILL taking the size of a pointer.
  for (int FI = MFI.getObjectIndexBegin(), EFI = MFI.getObjectIndexEnd();
       FI < EFI; FI++) {
    uint8_t StackID = MFI.getStackID(FI);
    if (StackID == TargetStackID::Default)
      continue;
    if (MFI.isDeadObjectIndex(FI))
      continue;

    switch (StackID) {
    case TargetStackID::ScalableVector:
      FrameSize =
          alignTo(FrameSize, RegInfo->getSpillAlignment(RISCV::GPRRegClass));
      FrameSize += RegInfo->getSpillSize(RISCV::GPRRegClass);
      MFI.setObjectOffset(FI, -FrameSize);
      break;
    default:
      llvm_unreachable("Unexpected StackID");
    }
  }

#ifndef NDEBUG
  for (int FI = MFI.getObjectIndexBegin(), EFI = MFI.getObjectIndexEnd();
       FI < EFI; FI++) {
    // Skip those already printed in PrologEpilogEmitter
    if (MFI.getStackID(FI) == TargetStackID::Default)
      continue;
    if (MFI.isDeadObjectIndex(FI))
      continue;
    assert(MFI.getStackID(FI) == TargetStackID::ScalableVector &&
           "Unexpected Stack ID!");
    LLVM_DEBUG(dbgs() << "alloc FI(" << FI << ") at SP["
                      << MFI.getObjectOffset(FI) << "] StackID: VR_SPILL\n");
  }
#endif

  // Get the alignment.
  Align StackAlign = getStackAlign();

  // Set Max Call Frame Size
  uint64_t MaxCallSize = alignTo(MFI.getMaxCallFrameSize(), StackAlign);
  MFI.setMaxCallFrameSize(MaxCallSize);

  // Make sure the frame is aligned.
  FrameSize = alignTo(FrameSize, StackAlign);

  // Update frame info.
  MFI.setStackSize(FrameSize);
}

void RISCVFrameLowering::adjustReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   const DebugLoc &DL, Register DestReg,
                                   Register SrcReg, int64_t Val,
                                   MachineInstr::MIFlag Flag) const {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const RISCVInstrInfo *TII = STI.getInstrInfo();

  if (DestReg == SrcReg && Val == 0)
    return;

  if (isInt<12>(Val)) {
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADDI), DestReg)
        .addReg(SrcReg)
        .addImm(Val)
        .setMIFlag(Flag);
  } else {
    unsigned Opc = RISCV::ADD;
    bool isSub = Val < 0;
    if (isSub) {
      Val = -Val;
      Opc = RISCV::SUB;
    }

    Register ScratchReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    TII->movImm(MBB, MBBI, DL, ScratchReg, Val, Flag);
    BuildMI(MBB, MBBI, DL, TII->get(Opc), DestReg)
        .addReg(SrcReg)
        .addReg(ScratchReg, RegState::Kill)
        .setMIFlag(Flag);
  }
}

// Returns the register used to hold the frame pointer.
static Register getFPReg(const RISCVSubtarget &STI) { return RISCV::X8; }

// Returns the register used to hold the base pointer.
static Register getBPReg(const RISCVSubtarget &STI) { return RISCV::X9; }

// Returns the register used to hold the stack pointer.
static Register getSPReg(const RISCVSubtarget &STI) { return RISCV::X2; }

static SmallVector<CalleeSavedInfo, 8>
getNonLibcallCSI(const std::vector<CalleeSavedInfo> &CSI) {
  SmallVector<CalleeSavedInfo, 8> NonLibcallCSI;

  for (auto &CS : CSI)
    if (CS.getFrameIdx() >= 0)
      NonLibcallCSI.push_back(CS);

  return NonLibcallCSI;
}

void RISCVFrameLowering::alignSP(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 const DebugLoc &DL, int64_t Alignment) const {
  assert(isPowerOf2_64(Alignment) && "Alignment must be a power of 2");

  const RISCVInstrInfo *TII = STI.getInstrInfo();

  if (isInt<12>(Alignment)) {
    //  ANDI SP, SP, -Alignment
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::ANDI), getSPReg(STI))
        .addReg(getSPReg(STI))
        .addImm(-Alignment)
        .setMIFlag(MachineInstr::FrameSetup);
  } else if (isInt<32>(Alignment)) {
    // Use shifts to avoid using a virtual register
    // SRLI SP, SP, Log2(Alignment)
    // SLLI SP, SP, Log2(Alignment)
    uint64_t Log2Alignment = Log2_64(Alignment);
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::SRLI), getSPReg(STI))
        .addReg(getSPReg(STI))
        .addImm(Log2Alignment)
        .setMIFlag(MachineInstr::FrameSetup);
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::SLLI), getSPReg(STI))
        .addReg(getSPReg(STI))
        .addImm(Log2Alignment)
        .setMIFlag(MachineInstr::FrameSetup);
  } else {
    report_fatal_error(
        "adjustReg cannot yet handle stack realignment >32 bits");
  }
}

bool RISCVFrameLowering::isSupportedStackID(TargetStackID::Value ID) const {
  switch (ID) {
  case TargetStackID::Default:
  case TargetStackID::ScalableVector:
    return true;
  case TargetStackID::NoAlloc:
  case TargetStackID::SGPRSpill:
    return false;
  }
  llvm_unreachable("Invalid TargetStackID::Value");
}

void RISCVFrameLowering::emitPrologue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  const RISCVRegisterInfo *RI = STI.getRegisterInfo();
  const RISCVInstrInfo *TII = STI.getInstrInfo();
  MachineBasicBlock::iterator MBBI = MBB.begin();

  Register FPReg = getFPReg(STI);
  Register SPReg = getSPReg(STI);
  Register BPReg = RISCVABI::getBPReg();

  // Debug location must be unknown since the first debug location is used
  // to determine the end of the prologue.
  DebugLoc DL;

  // All calls are tail calls in GHC calling conv, and functions have no
  // prologue/epilogue.
  if (MF.getFunction().getCallingConv() == CallingConv::GHC)
    return;

  // Emit prologue for shadow call stack.
  emitSCSPrologue(MF, MBB, MBBI, DL);

  // Since spillCalleeSavedRegisters may have inserted a libcall, skip past
  // any instructions marked as FrameSetup
  while (MBBI != MBB.end() && MBBI->getFlag(MachineInstr::FrameSetup))
    ++MBBI;

  // Determine the correct frame layout
  determineFrameLayout(MF);

  // If libcalls are used to spill and restore callee-saved registers, the frame
  // has two sections; the opaque section managed by the libcalls, and the
  // section managed by MachineFrameInfo which can also hold callee saved
  // registers in fixed stack slots, both of which have negative frame indices.
  // This gets even more complicated when incoming arguments are passed via the
  // stack, as these too have negative frame indices. An example is detailed
  // below:
  //
  //  | incoming arg | <- FI[-3]
  //  | libcallspill |
  //  | calleespill  | <- FI[-2]
  //  | calleespill  | <- FI[-1]
  //  | this_frame   | <- FI[0]
  //
  // For negative frame indices, the offset from the frame pointer will differ
  // depending on which of these groups the frame index applies to.
  // The following calculates the correct offset knowing the number of callee
  // saved registers spilt by the two methods.
  if (int LibCallRegs = getLibCallID(MF, MFI.getCalleeSavedInfo()) + 1) {
    // Calculate the size of the frame managed by the libcall. The libcalls are
    // implemented such that the stack will always be 16 byte aligned.
    unsigned LibCallFrameSize = alignTo((STI.getXLen() / 8) * LibCallRegs, 16);
    RVFI->setLibCallStackSize(LibCallFrameSize);
  }

  // FIXME (note copied from Lanai): This appears to be overallocating.  Needs
  // investigation. Get the number of bytes to allocate from the FrameInfo.
  uint64_t StackSize = MFI.getStackSize();
  uint64_t RealStackSize = StackSize + RVFI->getLibCallStackSize();

  // Early exit if there is no need to allocate on the stack
  if (RealStackSize == 0 && !MFI.adjustsStack())
    return;

  // If the stack pointer has been marked as reserved, then produce an error if
  // the frame requires stack allocation
  if (STI.isRegisterReservedByUser(SPReg))
    MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported{
        MF.getFunction(), "Stack pointer required, but has been reserved."});

  uint64_t FirstSPAdjustAmount = getFirstSPAdjustAmount(MF);
  // Split the SP adjustment to reduce the offsets of callee saved spill.
  if (FirstSPAdjustAmount) {
    StackSize = FirstSPAdjustAmount;
    RealStackSize = FirstSPAdjustAmount;
  }

  // Allocate space on the stack if necessary.
  adjustReg(MBB, MBBI, DL, SPReg, SPReg, -StackSize, MachineInstr::FrameSetup);

  // Emit ".cfi_def_cfa_offset RealStackSize"
  unsigned CFIIndex = MF.addFrameInst(
      MCCFIInstruction::cfiDefCfaOffset(nullptr, RealStackSize));
  BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(CFIIndex)
      .setMIFlag(MachineInstr::FrameSetup);

  const auto &CSI = MFI.getCalleeSavedInfo();

  // The frame pointer is callee-saved, and code has been generated for us to
  // save it to the stack. We need to skip over the storing of callee-saved
  // registers as the frame pointer must be modified after it has been saved
  // to the stack, not before.
  // FIXME: assumes exactly one instruction is used to save each callee-saved
  // register.
  // EPI registers break this assumption but they are to be handled after we
  // adjust the FP.
  int InsnToSkip = getNonLibcallCSI(CSI).size();
  for (auto &CS : CSI) {
    if (RISCV::VRRegClass.contains(CS.getReg()))
      InsnToSkip--;
  }
  std::advance(MBBI, InsnToSkip);

  // Iterate over list of callee-saved registers and emit .cfi_offset
  // directives.
  for (const auto &Entry : CSI) {
    int FrameIdx = Entry.getFrameIdx();
    int64_t Offset;
    // Offsets for objects with fixed locations (IE: those saved by libcall) are
    // simply calculated from the frame index.
    if (FrameIdx < 0)
      Offset = FrameIdx * (int64_t) STI.getXLen() / 8;
    else
      Offset = MFI.getObjectOffset(Entry.getFrameIdx()) -
               RVFI->getLibCallStackSize();
    Register Reg = Entry.getReg();
    // We don't have sensible DWARF for VRs yet
    if (RISCV::VRRegClass.contains(Reg))
      continue;
    unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::createOffset(
        nullptr, RI->getDwarfRegNum(Reg, true), Offset));
    BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex)
        .setMIFlag(MachineInstr::FrameSetup);
  }

  // Generate new FP.
  if (hasFP(MF)) {
    if (STI.isRegisterReservedByUser(FPReg))
      MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported{
          MF.getFunction(), "Frame pointer required, but has been reserved."});

    adjustReg(MBB, MBBI, DL, FPReg, SPReg,
              RealStackSize - RVFI->getVarArgsSaveSize(),
              MachineInstr::FrameSetup);

    // Emit ".cfi_def_cfa $fp, RVFI->getVarArgsSaveSize()"
    unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::cfiDefCfa(
        nullptr, RI->getDwarfRegNum(FPReg, true), RVFI->getVarArgsSaveSize()));
    BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex)
        .setMIFlag(MachineInstr::FrameSetup);
  }

  // Emit the second SP adjustment after saving callee saved registers.
  if (FirstSPAdjustAmount) {
    uint64_t SecondSPAdjustAmount = MFI.getStackSize() - FirstSPAdjustAmount;
    assert(SecondSPAdjustAmount > 0 &&
           "SecondSPAdjustAmount should be greater than zero");
    adjustReg(MBB, MBBI, DL, SPReg, SPReg, -SecondSPAdjustAmount,
              MachineInstr::FrameSetup);

    // If we are using a frame-pointer, and thus emitted ".cfi_def_cfa fp, 0",
    // don't emit an sp-based .cfi_def_cfa_offset
    if (!hasFP(MF)) {
      // Emit ".cfi_def_cfa_offset StackSize"
      unsigned CFIIndex = MF.addFrameInst(
          MCCFIInstruction::cfiDefCfaOffset(nullptr, MFI.getStackSize()));
      BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex);
    }
  }

  if (hasFP(MF)) {
    // Realign Stack
    const RISCVRegisterInfo *RI = STI.getRegisterInfo();
    if (RI->needsStackRealignment(MF)) {
      Align MaxAlignment = MFI.getMaxAlign();

      const RISCVInstrInfo *TII = STI.getInstrInfo();
      if (isInt<12>(-(int)MaxAlignment.value())) {
        BuildMI(MBB, MBBI, DL, TII->get(RISCV::ANDI), SPReg)
            .addReg(SPReg)
            .addImm(-(int)MaxAlignment.value());
      } else {
        unsigned ShiftAmount = Log2(MaxAlignment);
        Register VR =
            MF.getRegInfo().createVirtualRegister(&RISCV::GPRRegClass);
        BuildMI(MBB, MBBI, DL, TII->get(RISCV::SRLI), VR)
            .addReg(SPReg)
            .addImm(ShiftAmount);
        BuildMI(MBB, MBBI, DL, TII->get(RISCV::SLLI), SPReg)
            .addReg(VR)
            .addImm(ShiftAmount);
      }
      // FP will be used to restore the frame in the epilogue, so we need
      // another base register BP to record SP after re-alignment. SP will
      // track the current stack after allocating variable sized objects.
      if (hasBP(MF)) {
        // move BP, SP
        BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADDI), BPReg)
            .addReg(SPReg)
            .addImm(0);
      }
    }
  }

  prepareStorageSpilledVR(MF, MBB, MBBI, MFI, MF.getRegInfo(), *TII, DL);
}

void RISCVFrameLowering::prepareStorageSpilledVR(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MBBI, const MachineFrameInfo &MFI,
    MachineRegisterInfo &MRI, const TargetInstrInfo &TII,
    const DebugLoc &DL) const {
  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  if (!RVFI->hasSpilledVR())
    return;

  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();

  // FIXME: We're presuming in advance that this is all about VRs
  unsigned SizeOfVector = MRI.createVirtualRegister(&RISCV::GPRRegClass);
  BuildMI(MBB, MBBI, DL, TII.get(RISCV::PseudoEPIReadVLENB), SizeOfVector);

  // Classify the FIs.
  std::array<llvm::SmallVector<int, 4>, 4> BinSizes;

  for (int FI = MFI.getObjectIndexBegin(), EFI = MFI.getObjectIndexEnd();
       FI < EFI; FI++) {
    int8_t StackID = MFI.getStackID(FI);
    if (StackID == TargetStackID::Default)
      continue;
    if (MFI.isDeadObjectIndex(FI))
      continue;
    assert(StackID == TargetStackID::ScalableVector && "Unexpected StackID");

    int64_t ObjectSize = MFI.getObjectSize(FI);

    unsigned ShiftAmount;
    // Mask objects may be logically smaller than the spill size of the VR
    // class.
    if (ObjectSize <= RegInfo->getSpillSize(RISCV::VRRegClass))
      ShiftAmount = 0;
    else if (ObjectSize == RegInfo->getSpillSize(RISCV::VRM2RegClass))
      ShiftAmount = 1;
    else if (ObjectSize == RegInfo->getSpillSize(RISCV::VRM4RegClass))
      ShiftAmount = 2;
    else if (ObjectSize == RegInfo->getSpillSize(RISCV::VRM8RegClass))
      ShiftAmount = 3;
    else
      llvm_unreachable("Unexpected object size");

    BinSizes[ShiftAmount].push_back(FI);
  }

  // Do the actual allocation.
  unsigned SPReg = getSPReg(STI);

  for (const auto &Bin : BinSizes) {
    if (Bin.empty())
      continue;

    unsigned FactorRegister = 0;
    int ShiftAmount = &Bin - &BinSizes[0];
    assert(0 <= ShiftAmount && ShiftAmount <= 3 && "Invalid shift amount!");
    if (ShiftAmount > 0) {
      FactorRegister = MRI.createVirtualRegister(&RISCV::GPRRegClass);
      BuildMI(MBB, MBBI, DL, TII.get(RISCV::SLLI), FactorRegister)
          .addReg(SizeOfVector)
          .addImm(ShiftAmount);
    }

    for (const int &FI : Bin) {
      if (ShiftAmount > 0) {
        assert(FactorRegister && "Invalid register!");
        bool LastUse = &FI == &Bin.back();
        BuildMI(MBB, MBBI, DL, TII.get(RISCV::SUB), SPReg)
            .addReg(SPReg)
            .addReg(FactorRegister, LastUse ? RegState::Kill : 0);
      } else
        BuildMI(MBB, MBBI, DL, TII.get(RISCV::SUB), SPReg)
            .addReg(SPReg)
            .addReg(SizeOfVector);

      // FIXME: We used to align the stack here but given that the default
      // minimum alignment is already 16 bytes, we'd only need to keep it
      // aligned if VLEN < 128 bits. We presume VLEN >= 128 bit in every
      // implementation.
      unsigned StoreOpcode = RegInfo->getSpillSize(RISCV::GPRRegClass) == 4
                                 ? RISCV::SW
                                 : RISCV::SD;
      BuildMI(MBB, MBBI, DL, TII.get(StoreOpcode))
          .addReg(SPReg)
          .addFrameIndex(FI)
          .addImm(0);
    }
  }
}

void RISCVFrameLowering::emitEpilogue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
  const RISCVRegisterInfo *RI = STI.getRegisterInfo();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  Register FPReg = getFPReg(STI);
  Register SPReg = getSPReg(STI);

  // All calls are tail calls in GHC calling conv, and functions have no
  // prologue/epilogue.
  if (MF.getFunction().getCallingConv() == CallingConv::GHC)
    return;

  // Get the insert location for the epilogue. If there were no terminators in
  // the block, get the last instruction.
  MachineBasicBlock::iterator MBBI = MBB.end();
  DebugLoc DL;
  if (!MBB.empty()) {
    MBBI = MBB.getFirstTerminator();
    if (MBBI == MBB.end())
      MBBI = MBB.getLastNonDebugInstr();
    DL = MBBI->getDebugLoc();

    // If this is not a terminator, the actual insert location should be after the
    // last instruction.
    if (!MBBI->isTerminator())
      MBBI = std::next(MBBI);

    // If callee-saved registers are saved via libcall, place stack adjustment
    // before this call.
    while (MBBI != MBB.begin() &&
           std::prev(MBBI)->getFlag(MachineInstr::FrameDestroy))
      --MBBI;
  }

  const auto &CSI = getNonLibcallCSI(MFI.getCalleeSavedInfo());

  // Skip to before the restores of callee-saved registers
  // FIXME: assumes exactly one instruction is used to restore each
  // callee-saved register.
  auto LastFrameDestroy = MBBI;
  if (!CSI.empty()) {
    // Ignore the VRs as we did in the prologue.
    int InsnToSkip = CSI.size();
    for (auto &CS : CSI) {
      if (RISCV::VRRegClass.contains(CS.getReg()))
        InsnToSkip--;
    }
    LastFrameDestroy = std::prev(MBBI, InsnToSkip);
  }

  uint64_t StackSize = MFI.getStackSize();
  uint64_t RealStackSize = StackSize + RVFI->getLibCallStackSize();
  uint64_t FPOffset = RealStackSize - RVFI->getVarArgsSaveSize();

  // Restore the stack pointer using the value of the frame pointer. Only
  // necessary if the stack pointer was modified, meaning the stack size is
  // unknown.
  if (RI->needsStackRealignment(MF) || MFI.hasVarSizedObjects() ||
      RVFI->hasSpilledVR()) {
    assert(hasFP(MF) && "frame pointer should not have been eliminated");
    adjustReg(MBB, LastFrameDestroy, DL, SPReg, FPReg, -FPOffset,
              MachineInstr::FrameDestroy);
  }

  uint64_t FirstSPAdjustAmount = getFirstSPAdjustAmount(MF);
  if (FirstSPAdjustAmount) {
    uint64_t SecondSPAdjustAmount = MFI.getStackSize() - FirstSPAdjustAmount;
    assert(SecondSPAdjustAmount > 0 &&
           "SecondSPAdjustAmount should be greater than zero");

    adjustReg(MBB, LastFrameDestroy, DL, SPReg, SPReg, SecondSPAdjustAmount,
              MachineInstr::FrameDestroy);
  }

  if (FirstSPAdjustAmount)
    StackSize = FirstSPAdjustAmount;

  // Deallocate stack
  adjustReg(MBB, MBBI, DL, SPReg, SPReg, StackSize, MachineInstr::FrameDestroy);

  // Emit epilogue for shadow call stack.
  emitSCSEpilogue(MF, MBB, MBBI, DL);
}

StackOffset
RISCVFrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                           Register &FrameReg) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const RISCVRegisterInfo *RI =
      MF.getSubtarget<RISCVSubtarget>().getRegisterInfo();
  const auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();

  // Callee-saved registers in the default stack should be referenced relative
  // to the stack pointer (positive offset), otherwise use the frame pointer
  // (negative offset) unless the offset from FP is not known at compile time,
  // as it happens when we have to align the stack or we have variably sized
  // data.
  const auto &CSI = MFI.getCalleeSavedInfo();
  int MinCSFI = 0;
  int MaxCSFI = -1;

  int Offset = MFI.getObjectOffset(FI) - getOffsetOfLocalArea() +
               MFI.getOffsetAdjustment();

  uint64_t FirstSPAdjustAmount = getFirstSPAdjustAmount(MF);

  if (CSI.size()) {
    MinCSFI = CSI[0].getFrameIdx();
    MaxCSFI = CSI[CSI.size() - 1].getFrameIdx();
  }
  bool IsCSR = FI >= MinCSFI && FI <= MaxCSFI;

  if (IsCSR && MFI.getStackID(FI) == 0) {
    // Only CSRs in the default stack can be accessed using SP
    FrameReg = getSPReg(STI);
    if (FirstSPAdjustAmount)
      Offset += FirstSPAdjustAmount;
    else
      Offset += MFI.getStackSize();
  } else if (RI->needsStackRealignment(MF) && !MFI.isFixedObjectIndex(FI)) {
    // If the stack was realigned, the frame pointer is set in order to allow
    // SP to be restored, so we need another base register to record the stack
    // after realignment.
    if (hasBP(MF))
      FrameReg = RISCVABI::getBPReg();
    else
      FrameReg = RISCV::X2;
    Offset += MFI.getStackSize();
    if (FI < 0)
      Offset += RVFI->getLibCallStackSize();
  } else {
    FrameReg = RI->getFrameRegister(MF);
    if (hasFP(MF)) {
      Offset += RVFI->getVarArgsSaveSize();
      if (FI >= 0)
        Offset -= RVFI->getLibCallStackSize();
    } else {
      Offset += MFI.getStackSize();
      if (FI < 0)
        Offset += RVFI->getLibCallStackSize();
    }
  }
  return StackOffset::getFixed(Offset);
}

void RISCVFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                              BitVector &SavedRegs,
                                              RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
  const RISCVRegisterInfo *RI =
      MF.getSubtarget<RISCVSubtarget>().getRegisterInfo();

  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  const TargetRegisterClass *RC = &RISCV::GPRRegClass;
  if (RVFI->hasSpilledVR()) {
    // We conservatively add one extra emergency slots if we have seen PseudoVSPILL
    // or PseudoVRELOAD because we may need it for vlenb.
    int RegScavFI = MFI.CreateStackObject(RegInfo->getSpillSize(*RC),
                                          RegInfo->getSpillAlign(*RC), false);
    RS->addScavengingFrameIndex(RegScavFI);
  }

  // Go through all Stackslots coming from a VR alloca and make them VR_SPILL.
  // We need to this early otherwise we may be answering hasFP false.
  for (int FI = MFI.getObjectIndexBegin(), EFI = MFI.getObjectIndexEnd();
       FI < EFI; FI++) {
    // Get the (LLVM IR) allocation instruction
    const AllocaInst *Alloca = MFI.getObjectAllocation(FI);

    if (!Alloca)
      continue;

    if (const ScalableVectorType *VT = dyn_cast<const ScalableVectorType>(
            Alloca->getType()->getElementType())) {
      MFI.setStackID(FI, TargetStackID::ScalableVector);
      RVFI->setHasSpilledVR();
    } else if (const StructType *Sty = dyn_cast<const StructType>(
                   Alloca->getType()->getElementType())) {
      // Also check structures with scalable inside. We use them only for
      // tuples.
      for (const Type *ElTy : Sty->elements()) {
        if (isa<const ScalableVectorType>(ElTy)) {
          MFI.setStackID(FI, TargetStackID::ScalableVector);
          RVFI->setHasSpilledVR();
          break;
        }
      }
    }
  }

  // Unconditionally spill RA and FP only if the function uses a frame
  // pointer.
  if (hasFP(MF)) {
    SavedRegs.set(RISCV::X1);
    SavedRegs.set(getFPReg(STI));
  }
  if (RI->hasBasePointer(MF)) {
    SavedRegs.set(getBPReg(STI));
  }
  // Mark BP as used if function has dedicated base pointer.
  if (hasBP(MF))
    SavedRegs.set(RISCVABI::getBPReg());

  // If interrupt is enabled and there are calls in the handler,
  // unconditionally save all Caller-saved registers and
  // all FP registers, regardless whether they are used.
  if (MF.getFunction().hasFnAttribute("interrupt") && MFI.hasCalls()) {

    static const MCPhysReg CSRegs[] = { RISCV::X1,      /* ra */
      RISCV::X5, RISCV::X6, RISCV::X7,                  /* t0-t2 */
      RISCV::X10, RISCV::X11,                           /* a0-a1, a2-a7 */
      RISCV::X12, RISCV::X13, RISCV::X14, RISCV::X15, RISCV::X16, RISCV::X17,
      RISCV::X28, RISCV::X29, RISCV::X30, RISCV::X31, 0 /* t3-t6 */
    };

    for (unsigned i = 0; CSRegs[i]; ++i)
      SavedRegs.set(CSRegs[i]);

    if (MF.getSubtarget<RISCVSubtarget>().hasStdExtF()) {

      // If interrupt is enabled, this list contains all FP registers.
      const MCPhysReg * Regs = MF.getRegInfo().getCalleeSavedRegs();

      for (unsigned i = 0; Regs[i]; ++i)
        if (RISCV::FPR16RegClass.contains(Regs[i]) ||
            RISCV::FPR32RegClass.contains(Regs[i]) ||
            RISCV::FPR64RegClass.contains(Regs[i]))
          SavedRegs.set(Regs[i]);
    }
  }
}

void RISCVFrameLowering::processFunctionBeforeFrameFinalized(
    MachineFunction &MF, RegScavenger *RS) const {
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  const TargetRegisterClass *RC = &RISCV::GPRRegClass;

  // estimateStackSize has been observed to under-estimate the final stack
  // size, so give ourselves wiggle-room by checking for stack size
  // representable an 11-bit signed field rather than 12-bits.
  // FIXME: It may be possible to craft a function with a small stack that
  // still needs an emergency spill slot for branch relaxation. This case
  // would currently be missed.
  // EPI: frames that store vectors on the stack usually need large offsets
  // so make sure there is an emergency spill for them in case computing
  // them needs an extra register.
  if (!isInt<11>(MFI.estimateStackSize(MF)) || RVFI->hasSpilledVR()) {
    int RegScavFI = MFI.CreateStackObject(RegInfo->getSpillSize(*RC),
                                          RegInfo->getSpillAlign(*RC), false);
    RS->addScavengingFrameIndex(RegScavFI);
  }
}

// Not preserve stack space within prologue for outgoing variables when the
// function contains variable size objects and let eliminateCallFramePseudoInstr
// preserve stack space for it.
bool RISCVFrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  return !(MF.getFrameInfo().hasVarSizedObjects() || RVFI->hasSpilledVR());
}

// Eliminate ADJCALLSTACKDOWN, ADJCALLSTACKUP pseudo instructions.
MachineBasicBlock::iterator RISCVFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MI) const {
  Register SPReg = RISCV::X2;
  DebugLoc DL = MI->getDebugLoc();

  if (!hasReservedCallFrame(MF)) {
    // If space has not been reserved for a call frame, ADJCALLSTACKDOWN and
    // ADJCALLSTACKUP must be converted to instructions manipulating the stack
    // pointer. This is necessary when there is a variable length stack
    // allocation (e.g. alloca), which means it's not possible to allocate
    // space for outgoing arguments from within the function prologue.
    int64_t Amount = MI->getOperand(0).getImm();

    if (Amount != 0) {
      // Ensure the stack remains aligned after adjustment.
      Amount = alignSPAdjust(Amount);

      if (MI->getOpcode() == RISCV::ADJCALLSTACKDOWN)
        Amount = -Amount;

      adjustReg(MBB, MI, DL, SPReg, SPReg, Amount, MachineInstr::NoFlags);
    }
  }

  return MBB.erase(MI);
}

// We would like to split the SP adjustment to reduce prologue/epilogue
// as following instructions. In this way, the offset of the callee saved
// register could fit in a single store.
//   add     sp,sp,-2032
//   sw      ra,2028(sp)
//   sw      s0,2024(sp)
//   sw      s1,2020(sp)
//   sw      s3,2012(sp)
//   sw      s4,2008(sp)
//   add     sp,sp,-64
uint64_t
RISCVFrameLowering::getFirstSPAdjustAmount(const MachineFunction &MF) const {
  const auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  uint64_t StackSize = MFI.getStackSize();

  // Disable SplitSPAdjust if save-restore libcall used. The callee saved
  // registers will be pushed by the save-restore libcalls, so we don't have to
  // split the SP adjustment in this case.
  if (RVFI->getLibCallStackSize())
    return 0;

  // Return the FirstSPAdjustAmount if the StackSize can not fit in signed
  // 12-bit and there exists a callee saved register need to be pushed.
  if (!isInt<12>(StackSize) && (CSI.size() > 0)) {
    // FirstSPAdjustAmount is choosed as (2048 - StackAlign)
    // because 2048 will cause sp = sp + 2048 in epilogue split into
    // multi-instructions. The offset smaller than 2048 can fit in signle
    // load/store instruction and we have to stick with the stack alignment.
    // 2048 is 16-byte alignment. The stack alignment for RV32 and RV64 is 16,
    // for RV32E is 4. So (2048 - StackAlign) will satisfy the stack alignment.
    return 2048 - getStackAlign().value();
  }
  return 0;
}

bool RISCVFrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return true;

  MachineFunction *MF = MBB.getParent();
  const TargetInstrInfo &TII = *MF->getSubtarget().getInstrInfo();
  DebugLoc DL;
  if (MI != MBB.end() && !MI->isDebugInstr())
    DL = MI->getDebugLoc();

  const char *SpillLibCall = getSpillLibCallName(*MF, CSI);
  if (SpillLibCall) {
    // Add spill libcall via non-callee-saved register t0.
    BuildMI(MBB, MI, DL, TII.get(RISCV::PseudoCALLReg), RISCV::X5)
        .addExternalSymbol(SpillLibCall, RISCVII::MO_CALL)
        .setMIFlag(MachineInstr::FrameSetup);

    // Add registers spilled in libcall as liveins.
    for (auto &CS : CSI)
      MBB.addLiveIn(CS.getReg());
  }

  // Manually spill values not spilled by libcall.
  const auto &NonLibcallCSI = getNonLibcallCSI(CSI);
  for (auto &CS : NonLibcallCSI) {
    // Insert the spill to the stack frame.
    Register Reg = CS.getReg();
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII.storeRegToStackSlot(MBB, MI, Reg, true, CS.getFrameIdx(), RC, TRI);
  }

  return true;
}

bool RISCVFrameLowering::restoreCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return true;

  MachineFunction *MF = MBB.getParent();
  const TargetInstrInfo &TII = *MF->getSubtarget().getInstrInfo();
  DebugLoc DL;
  if (MI != MBB.end() && !MI->isDebugInstr())
    DL = MI->getDebugLoc();

  // Manually restore values not restored by libcall. Insert in reverse order.
  // loadRegFromStackSlot can insert multiple instructions.
  const auto &NonLibcallCSI = getNonLibcallCSI(CSI);
  for (auto &CS : reverse(NonLibcallCSI)) {
    Register Reg = CS.getReg();
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII.loadRegFromStackSlot(MBB, MI, Reg, CS.getFrameIdx(), RC, TRI);
    assert(MI != MBB.begin() && "loadRegFromStackSlot didn't insert any code!");
  }

  const char *RestoreLibCall = getRestoreLibCallName(*MF, CSI);
  if (RestoreLibCall) {
    // Add restore libcall via tail call.
    MachineBasicBlock::iterator NewMI =
        BuildMI(MBB, MI, DL, TII.get(RISCV::PseudoTAIL))
            .addExternalSymbol(RestoreLibCall, RISCVII::MO_CALL)
            .setMIFlag(MachineInstr::FrameDestroy);

    // Remove trailing returns, since the terminator is now a tail call to the
    // restore function.
    if (MI != MBB.end() && MI->getOpcode() == RISCV::PseudoRET) {
      NewMI->copyImplicitOps(*MF, *MI);
      MI->eraseFromParent();
    }
  }

  return true;
}

bool RISCVFrameLowering::canUseAsPrologue(const MachineBasicBlock &MBB) const {
  MachineBasicBlock *TmpMBB = const_cast<MachineBasicBlock *>(&MBB);
  const MachineFunction *MF = MBB.getParent();
  const auto *RVFI = MF->getInfo<RISCVMachineFunctionInfo>();

  if (!RVFI->useSaveRestoreLibCalls(*MF))
    return true;

  // Inserting a call to a __riscv_save libcall requires the use of the register
  // t0 (X5) to hold the return address. Therefore if this register is already
  // used we can't insert the call.

  RegScavenger RS;
  RS.enterBasicBlock(*TmpMBB);
  return !RS.isRegUsed(RISCV::X5);
}

bool RISCVFrameLowering::canUseAsEpilogue(const MachineBasicBlock &MBB) const {
  const MachineFunction *MF = MBB.getParent();
  MachineBasicBlock *TmpMBB = const_cast<MachineBasicBlock *>(&MBB);
  const auto *RVFI = MF->getInfo<RISCVMachineFunctionInfo>();

  if (!RVFI->useSaveRestoreLibCalls(*MF))
    return true;

  // Using the __riscv_restore libcalls to restore CSRs requires a tail call.
  // This means if we still need to continue executing code within this function
  // the restore cannot take place in this basic block.

  if (MBB.succ_size() > 1)
    return false;

  MachineBasicBlock *SuccMBB =
      MBB.succ_empty() ? TmpMBB->getFallThrough() : *MBB.succ_begin();

  // Doing a tail call should be safe if there are no successors, because either
  // we have a returning block or the end of the block is unreachable, so the
  // restore will be eliminated regardless.
  if (!SuccMBB)
    return true;

  // The successor can only contain a return, since we would effectively be
  // replacing the successor with our own tail return at the end of our block.
  return SuccMBB->isReturnBlock() && SuccMBB->size() == 1;
}
