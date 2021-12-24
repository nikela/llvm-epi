//==- RISCVSchedAvispado.td - Avispado Scheduling Defs -------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// The following definitions describe the simpler per-operand machine model.
// This works with MachineScheduler. See MCSchedule.h for details.

// Avispado machine model for scheduling and other instruction cost heuristics.
def AvispadoModel : SchedMachineModel {
  let MicroOpBufferSize = 8; // This is a decoupled design,
                             // so the instruction buffer is large.
                             // FIXME: This is for the VPU but what about the
                             // scalar core.
  let IssueWidth = 1;        // One instruction per cycle can be dispatched.
  let LoadLatency = 3;
  let MispredictPenalty = 3;
  let UnsupportedFeatures = [HasStdExtZbb, HasStdExtZvlsseg];

  // We should be able to remove this when all the instructions
  // have scheduling information.
  let CompleteModel = 0;
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available.

// Modeling each pipeline as a ProcResource using the BufferSize = 0 since
// Avispado is in-order.

let BufferSize = 0 in {
def AvispadoUnitALU        : ProcResource<1>; // Int ALU
def AvispadoUnitIMul       : ProcResource<1>; // Int Multiply
def AvispadoUnitMem        : ProcResource<1>; // Load/Store
def AvispadoUnitB          : ProcResource<1>; // Branch

def AvispadoUnitFPALU      : ProcResource<1>; // FP ALU
}

// FIXME: This was taken from Rocket so perhaps doesn't make sense for Avispado.
let BufferSize = 1 in {
def AvispadoUnitIDiv       : ProcResource<1>; // Int Division
def AvispadoUnitFPDivSqrt  : ProcResource<1>; // FP Divide/Sqrt
}

//===----------------------------------------------------------------------===//

let SchedModel = AvispadoModel in {

// Branching
def : WriteRes<WriteJmp, [AvispadoUnitB]>;
def : WriteRes<WriteJal, [AvispadoUnitB]>;
def : WriteRes<WriteJalr, [AvispadoUnitB]>;
def : WriteRes<WriteJmpReg, [AvispadoUnitB]>;

// Integer arithmetic and logic
def : WriteRes<WriteIALU32, [AvispadoUnitALU]>;
def : WriteRes<WriteIALU, [AvispadoUnitALU]>;
def : WriteRes<WriteShiftReg32, [AvispadoUnitALU]>;
def : WriteRes<WriteShiftImm32, [AvispadoUnitALU]>;
def : WriteRes<WriteShiftReg, [AvispadoUnitALU]>;
def : WriteRes<WriteShiftImm, [AvispadoUnitALU]>;

// Integer multiplication
let Latency = 4 in {
def : WriteRes<WriteIMul, [AvispadoUnitIMul]>;
def : WriteRes<WriteIMul32, [AvispadoUnitIMul]>;
}

// Integer division
// Worst case latency is used.
def : WriteRes<WriteIDiv32, [AvispadoUnitIDiv]> {
  let Latency = 34;
  let ResourceCycles = [34];
}
def : WriteRes<WriteIDiv, [AvispadoUnitIDiv]> {
  let Latency = 33;
  let ResourceCycles = [33];
}

// Memory
def : WriteRes<WriteSTB, [AvispadoUnitMem]>;
def : WriteRes<WriteSTH, [AvispadoUnitMem]>;
def : WriteRes<WriteSTW, [AvispadoUnitMem]>;
def : WriteRes<WriteSTD, [AvispadoUnitMem]>;
def : WriteRes<WriteFST32, [AvispadoUnitMem]>;
def : WriteRes<WriteFST64, [AvispadoUnitMem]>;

let Latency = 3 in {
def : WriteRes<WriteLDB, [AvispadoUnitMem]>;
def : WriteRes<WriteLDH, [AvispadoUnitMem]>;
}

let Latency = 2 in {
def : WriteRes<WriteLDW, [AvispadoUnitMem]>;
def : WriteRes<WriteLDWU, [AvispadoUnitMem]>;
def : WriteRes<WriteLDD, [AvispadoUnitMem]>;
def : WriteRes<WriteFLD32, [AvispadoUnitMem]>;
def : WriteRes<WriteFLD64, [AvispadoUnitMem]>;

// Atomic memory
def : WriteRes<WriteAtomicW, [AvispadoUnitMem]>;
def : WriteRes<WriteAtomicD, [AvispadoUnitMem]>;

def : WriteRes<WriteAtomicLDW, [AvispadoUnitMem]>;
def : WriteRes<WriteAtomicLDD, [AvispadoUnitMem]>;
}

def : WriteRes<WriteAtomicSTW, [AvispadoUnitMem]>;
def : WriteRes<WriteAtomicSTD, [AvispadoUnitMem]>;

// Single precision.
let Latency = 4 in {
def : WriteRes<WriteFALU32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFSGNJ32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMinMax32, [AvispadoUnitFPALU]>;
}

// Double precision
let Latency = 6 in {
def : WriteRes<WriteFALU64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFSGNJ64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMinMax64, [AvispadoUnitFPALU]>;
}

// Conversions
let Latency = 2 in {
def : WriteRes<WriteFCvtI32ToF32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtI32ToF64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtI64ToF32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtI64ToF64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtF32ToI32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtF32ToI64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtF64ToI32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtF64ToI64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtF32ToF64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCvtF64ToF32, [AvispadoUnitFPALU]>;

def : WriteRes<WriteFClass32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFClass64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCmp32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFCmp64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMovF32ToI32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMovI32ToF32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMovF64ToI64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMovI64ToF64, [AvispadoUnitFPALU]>;
}

// FP multiplication
let Latency = 5 in {
def : WriteRes<WriteFMul32, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMA32, [AvispadoUnitFPALU]>;
}

let Latency = 7 in {
def : WriteRes<WriteFMul64, [AvispadoUnitFPALU]>;
def : WriteRes<WriteFMA64, [AvispadoUnitFPALU]>;
}

// FP division
// FP division unit on Avispado is not pipelined, so set resource cycles to latency.
let Latency = 20, ResourceCycles = [20] in {
def : WriteRes<WriteFDiv32, [AvispadoUnitFPDivSqrt]>;
def : WriteRes<WriteFDiv64, [AvispadoUnitFPDivSqrt]>;
}

// FP square root unit on Avispado is not pipelined, so set resource cycles to latency.
def : WriteRes<WriteFSqrt32, [AvispadoUnitFPDivSqrt]> { let Latency = 20;
                                                      let ResourceCycles = [20]; }
def : WriteRes<WriteFSqrt64, [AvispadoUnitFPDivSqrt]> { let Latency = 25;
                                                      let ResourceCycles = [25]; }

// Others
def : WriteRes<WriteCSR, []>;
def : WriteRes<WriteNop, []>;

def : InstRW<[WriteIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//
// Bypass and advance
def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;
def : ReadAdvance<ReadShiftReg, 0>;
def : ReadAdvance<ReadShiftImm, 0>;
def : ReadAdvance<ReadShiftReg32, 0>;
def : ReadAdvance<ReadShiftImm32, 0>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : ReadAdvance<ReadAtomicWA, 0>;
def : ReadAdvance<ReadAtomicWD, 0>;
def : ReadAdvance<ReadAtomicDA, 0>;
def : ReadAdvance<ReadAtomicDD, 0>;
def : ReadAdvance<ReadAtomicLDW, 0>;
def : ReadAdvance<ReadAtomicLDD, 0>;
def : ReadAdvance<ReadAtomicSTW, 0>;
def : ReadAdvance<ReadAtomicSTD, 0>;
def : ReadAdvance<ReadFMemBase, 0>;
def : ReadAdvance<ReadFALU32, 0>;
def : ReadAdvance<ReadFALU64, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA64, 0>;
def : ReadAdvance<ReadFDiv32, 0>;
def : ReadAdvance<ReadFDiv64, 0>;
def : ReadAdvance<ReadFSqrt32, 0>;
def : ReadAdvance<ReadFSqrt64, 0>;
def : ReadAdvance<ReadFCmp32, 0>;
def : ReadAdvance<ReadFCmp64, 0>;
def : ReadAdvance<ReadFSGNJ32, 0>;
def : ReadAdvance<ReadFSGNJ64, 0>;
def : ReadAdvance<ReadFMinMax32, 0>;
def : ReadAdvance<ReadFMinMax64, 0>;
def : ReadAdvance<ReadFCvtF32ToI32, 0>;
def : ReadAdvance<ReadFCvtF32ToI64, 0>;
def : ReadAdvance<ReadFCvtF64ToI32, 0>;
def : ReadAdvance<ReadFCvtF64ToI64, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI32ToF64, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF64, 0>;
def : ReadAdvance<ReadFCvtF32ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF32, 0>;
def : ReadAdvance<ReadFMovF32ToI32, 0>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : ReadAdvance<ReadFMovF64ToI64, 0>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFClass64, 0>;

}

// V-extension in Avispado
def AvispadoVSETVL          : ProcResource<1>;  // Vector configuration

let BufferSize = 8 in {
  def AvispadoVPUIArith      : ProcResource<1>; // Arithmetic VPU Integer ALU
  def AvispadoVPUFArith      : ProcResource<1>; // Arithmetic VPU FP ALU
  def AvispadoVPUMemLoad    : ProcResource<1>;  // VPU Load
  def AvispadoVPUMemStore   : ProcResource<1>;  // VPU Store
}

let SchedModel = AvispadoModel in {

// let Latency = 1 in
// {
//     def : WriteRes<WriteVL, [AvispadoVSETVL]>;
//     def : WriteRes<WriteVTYPE, [AvispadoVSETVL]>;
// }

let Latency = 16, ResourceCycles = [8] in
{
  def : WriteRes<WriteVLDM, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDE8, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDE16, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDE32, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDE64, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDS8, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDS16, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDS32, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDS64, [AvispadoVPUMemLoad]>;

  def : WriteRes<WriteVLDUX8, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDUX16, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDUX32, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDUX64, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDOX8, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDOX16, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDOX32, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDOX64, [AvispadoVPUMemLoad]>;

  def : WriteRes<WriteVLDFF8, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDFF16, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDFF32, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLDFF64, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD1R8, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD1R16, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD1R32, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD1R64, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD2R8, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD2R16, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD2R32, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD2R64, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD4R8, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD4R16, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD4R32, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD4R64, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD8R8, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD8R16, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD8R32, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVLD8R64, [AvispadoVPUMemLoad]>;
}

let Latency = 16, ResourceCycles = [16] in
{
  def : WriteRes<WriteVSTUX8, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVSTUX16, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVSTUX32, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVSTUX64, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVSTOX8, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVSTOX16, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVSTOX32, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVSTOX64, [AvispadoVPUMemStore]>;
  
  def : WriteRes<WriteVSTM, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVSTE8, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVSTE16, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVSTE32, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVSTE64, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVSTS8, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVSTS16, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVSTS32, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVSTS64, [AvispadoVPUMemStore]>;
  
  def : WriteRes<WriteVST1R, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVST2R, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVST4R, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVST8R, [AvispadoVPUMemStore]>;
}

let Latency = 8, ResourceCycles = [8] in
{
  def : WriteRes<WriteVIALUV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIALUX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIALUI, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIWALUV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIWALUX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIWALUI, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVExtV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVICALUV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVICALUX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVICALUI, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVShiftV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVShiftX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVShiftI, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVNShiftV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVNShiftX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVNShiftI, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVICmpV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVICmpX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVICmpI, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIMulV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIMulX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIDivV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIDivX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIWMulV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIWMulX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIMulAddV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIMulAddX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIWMulAddV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIWMulAddX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIMergeV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIMergeX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIMergeI, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIMovV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIMovX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIMovI, [AvispadoVPUIArith]>;

  // Fixed point is integer arithmetic in disguise.
  def : WriteRes<WriteVSALUV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVSALUX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVSALUI, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVAALUV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVAALUX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVSMulV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVSMulX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVSShiftV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVSShiftX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVSShiftI, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVNClipV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVNClipX, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVNClipI, [AvispadoVPUIArith]>;

  def : WriteRes<WriteVIRedV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVIWRedV, [AvispadoVPUIArith]>;

  def : WriteRes<WriteVMALUV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVMPopV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVMFFSV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVMSFSV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVMIotV, [AvispadoVPUIArith]>;
  def : WriteRes<WriteVMIdxV, [AvispadoVPUIArith]>;
}

let Latency = 10, ResourceCycles = [10] in
{
  def : WriteRes<WriteVFALUV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFALUF, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFWALUV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFWALUF, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFMulV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFMulF, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFDivV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFDivF, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFWMulV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFWMulF, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFMulAddV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFMulAddF, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFWMulAddV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFWMulAddF, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFSqrtV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFRecpV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFCmpV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFCmpF, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFSgnjV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFSgnjF, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFClassV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFMergeV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFMovV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFCvtIToFV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFCvtFToIV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFCvtFToFV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFWCvtIToFV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFWCvtFToIV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFWCvtFToFV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFNCvtIToFV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFNCvtFToIV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFNCvtFToFV, [AvispadoVPUFArith]>;

  def : WriteRes<WriteVFRedV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFRedOV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFWRedV, [AvispadoVPUFArith]>;
  def : WriteRes<WriteVFWRedOV, [AvispadoVPUFArith]>;
}

// FIXME
def : WriteRes<WriteVIMovVX, []>;
def : WriteRes<WriteVIMovXV, []>;
def : WriteRes<WriteVFMovVF, []>;
def : WriteRes<WriteVFMovFV, []>;
def : WriteRes<WriteVISlideX, []>;
def : WriteRes<WriteVISlideI, []>;
def : WriteRes<WriteVISlide1X, []>;
def : WriteRes<WriteVFSlide1F, []>;
def : WriteRes<WriteVGatherV, []>;
def : WriteRes<WriteVGatherX, []>;
def : WriteRes<WriteVGatherI, []>;
def : WriteRes<WriteVCompressV, []>;
def : WriteRes<WriteVMov1V, []>;
def : WriteRes<WriteVMov2V, []>;
def : WriteRes<WriteVMov4V, []>;
def : WriteRes<WriteVMov8V, []>;

// Bypasses

// 7. Vector Loads and Stores
def : ReadAdvance<ReadVLDX, 0>;
def : ReadAdvance<ReadVSTX, 0>;
def : ReadAdvance<ReadVSTE8V, 0>;
def : ReadAdvance<ReadVSTE16V, 0>;
def : ReadAdvance<ReadVSTE32V, 0>;
def : ReadAdvance<ReadVSTE64V, 0>;
def : ReadAdvance<ReadVSTM, 0>;
def : ReadAdvance<ReadVLDSX, 0>;
def : ReadAdvance<ReadVSTSX, 0>;
def : ReadAdvance<ReadVSTS8V, 0>;
def : ReadAdvance<ReadVSTS16V, 0>;
def : ReadAdvance<ReadVSTS32V, 0>;
def : ReadAdvance<ReadVSTS64V, 0>;
def : ReadAdvance<ReadVLDUXV, 0>;
def : ReadAdvance<ReadVLDOXV, 0>;
def : ReadAdvance<ReadVSTUXV, 0>;
def : ReadAdvance<ReadVSTUX8, 0>;
def : ReadAdvance<ReadVSTUX16, 0>;
def : ReadAdvance<ReadVSTUX32, 0>;
def : ReadAdvance<ReadVSTUX64, 0>;
def : ReadAdvance<ReadVSTUX8V, 0>;
def : ReadAdvance<ReadVSTUX16V, 0>;
def : ReadAdvance<ReadVSTUX32V, 0>;
def : ReadAdvance<ReadVSTUX64V, 0>;
def : ReadAdvance<ReadVSTOX8, 0>;
def : ReadAdvance<ReadVSTOX16, 0>;
def : ReadAdvance<ReadVSTOX32, 0>;
def : ReadAdvance<ReadVSTOX64, 0>;
def : ReadAdvance<ReadVSTOXV, 0>;
def : ReadAdvance<ReadVSTOX8V, 0>;
def : ReadAdvance<ReadVSTOX16V, 0>;
def : ReadAdvance<ReadVSTOX32V, 0>;
def : ReadAdvance<ReadVSTOX64V, 0>;
def : ReadAdvance<ReadVST1R, 0>;
def : ReadAdvance<ReadVST2R, 0>;
def : ReadAdvance<ReadVST4R, 0>;
def : ReadAdvance<ReadVST8R, 0>;

// 12. Vector Integer Arithmetic Instructions
def : ReadAdvance<ReadVIALUV, 0>;
def : ReadAdvance<ReadVIALUX, 0>;
def : ReadAdvance<ReadVIWALUV, 0>;
def : ReadAdvance<ReadVIWALUX, 0>;
def : ReadAdvance<ReadVExtV, 0>;
def : ReadAdvance<ReadVIALUCV, 0>;
def : ReadAdvance<ReadVIALUCX, 0>;
def : ReadAdvance<ReadVShiftV, 0>;
def : ReadAdvance<ReadVShiftX, 0>;
def : ReadAdvance<ReadVNShiftV, 0>;
def : ReadAdvance<ReadVNShiftX, 0>;
def : ReadAdvance<ReadVICmpV, 0>;
def : ReadAdvance<ReadVICmpX, 0>;
def : ReadAdvance<ReadVIMulV, 0>;
def : ReadAdvance<ReadVIMulX, 0>;
def : ReadAdvance<ReadVIDivV, 0>;
def : ReadAdvance<ReadVIDivX, 0>;
def : ReadAdvance<ReadVIWMulV, 0>;
def : ReadAdvance<ReadVIWMulX, 0>;
def : ReadAdvance<ReadVIMulAddV, 0>;
def : ReadAdvance<ReadVIMulAddX, 0>;
def : ReadAdvance<ReadVIWMulAddV, 0>;
def : ReadAdvance<ReadVIWMulAddX, 0>;
def : ReadAdvance<ReadVIMergeV, 0>;
def : ReadAdvance<ReadVIMergeX, 0>;
def : ReadAdvance<ReadVIMovV, 0>;
def : ReadAdvance<ReadVIMovX, 0>;

// 13. Vector Fixed-Point Arithmetic Instructions
def : ReadAdvance<ReadVSALUV, 0>;
def : ReadAdvance<ReadVSALUX, 0>;
def : ReadAdvance<ReadVAALUV, 0>;
def : ReadAdvance<ReadVAALUX, 0>;
def : ReadAdvance<ReadVSMulV, 0>;
def : ReadAdvance<ReadVSMulX, 0>;
def : ReadAdvance<ReadVSShiftV, 0>;
def : ReadAdvance<ReadVSShiftX, 0>;
def : ReadAdvance<ReadVNClipV, 0>;
def : ReadAdvance<ReadVNClipX, 0>;

// 14. Vector Floating-Point Instructions
def : ReadAdvance<ReadVFALUV, 0>;
def : ReadAdvance<ReadVFALUF, 0>;
def : ReadAdvance<ReadVFWALUV, 0>;
def : ReadAdvance<ReadVFWALUF, 0>;
def : ReadAdvance<ReadVFMulV, 0>;
def : ReadAdvance<ReadVFMulF, 0>;
def : ReadAdvance<ReadVFDivV, 0>;
def : ReadAdvance<ReadVFDivF, 0>;
def : ReadAdvance<ReadVFWMulV, 0>;
def : ReadAdvance<ReadVFWMulF, 0>;
def : ReadAdvance<ReadVFMulAddV, 0>;
def : ReadAdvance<ReadVFMulAddF, 0>;
def : ReadAdvance<ReadVFWMulAddV, 0>;
def : ReadAdvance<ReadVFWMulAddF, 0>;
def : ReadAdvance<ReadVFSqrtV, 0>;
def : ReadAdvance<ReadVFRecpV, 0>;
def : ReadAdvance<ReadVFCmpV, 0>;
def : ReadAdvance<ReadVFCmpF, 0>;
def : ReadAdvance<ReadVFSgnjV, 0>;
def : ReadAdvance<ReadVFSgnjF, 0>;
def : ReadAdvance<ReadVFClassV, 0>;
def : ReadAdvance<ReadVFMergeV, 0>;
def : ReadAdvance<ReadVFMergeF, 0>;
def : ReadAdvance<ReadVFMovF, 0>;
def : ReadAdvance<ReadVFCvtIToFV, 0>;
def : ReadAdvance<ReadVFCvtFToIV, 0>;
def : ReadAdvance<ReadVFWCvtIToFV, 0>;
def : ReadAdvance<ReadVFWCvtFToIV, 0>;
def : ReadAdvance<ReadVFWCvtFToFV, 0>;
def : ReadAdvance<ReadVFNCvtIToFV, 0>;
def : ReadAdvance<ReadVFNCvtFToIV, 0>;
def : ReadAdvance<ReadVFNCvtFToFV, 0>;

// 15. Vector Reduction Operations
def : ReadAdvance<ReadVIRedV, 0>;
def : ReadAdvance<ReadVIRedV0, 0>;
def : ReadAdvance<ReadVIWRedV, 0>;
def : ReadAdvance<ReadVIWRedV0, 0>;
def : ReadAdvance<ReadVFRedV, 0>;
def : ReadAdvance<ReadVFRedV0, 0>;
def : ReadAdvance<ReadVFRedOV, 0>;
def : ReadAdvance<ReadVFRedOV0, 0>;
def : ReadAdvance<ReadVFWRedV, 0>;
def : ReadAdvance<ReadVFWRedV0, 0>;
def : ReadAdvance<ReadVFWRedOV, 0>;
def : ReadAdvance<ReadVFWRedOV0, 0>;

// 16. Vector Mask Instructions
def : ReadAdvance<ReadVMALUV, 0>;
def : ReadAdvance<ReadVMPopV, 0>;
def : ReadAdvance<ReadVMFFSV, 0>;
def : ReadAdvance<ReadVMSFSV, 0>;
def : ReadAdvance<ReadVMIotV, 0>;

// 17. Vector Permutation Instructions
def : ReadAdvance<ReadVIMovVX, 0>;
def : ReadAdvance<ReadVIMovXV, 0>;
def : ReadAdvance<ReadVIMovXX, 0>;
def : ReadAdvance<ReadVFMovVF, 0>;
def : ReadAdvance<ReadVFMovFV, 0>;
def : ReadAdvance<ReadVFMovFX, 0>;
def : ReadAdvance<ReadVISlideV, 0>;
def : ReadAdvance<ReadVISlideX, 0>;
def : ReadAdvance<ReadVFSlideV, 0>;
def : ReadAdvance<ReadVFSlideF, 0>;
def : ReadAdvance<ReadVGatherV, 0>;
def : ReadAdvance<ReadVGatherX, 0>;
def : ReadAdvance<ReadVCompressV, 0>;
def : ReadAdvance<ReadVMov1V, 0>;
def : ReadAdvance<ReadVMov2V, 0>;
def : ReadAdvance<ReadVMov4V, 0>;
def : ReadAdvance<ReadVMov8V, 0>;

// Others
def : ReadAdvance<ReadVMask, 0>;

// Unsupported extensions
defm : UnsupportedSchedZba;
defm : UnsupportedSchedZbb;
defm : UnsupportedSchedZfh;

}
