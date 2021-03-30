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
  let UnsupportedFeatures = [HasStdExtZbb, HasStdExtZvamo, HasStdExtZvlsseg];

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
  def AvispadoVPUArith      : ProcResource<1>; // Arithmetic VPU ALU
  def AvispadoVPUMemLoad    : ProcResource<1>;  // VPU Load
  def AvispadoVPUMemStore   : ProcResource<1>;  // VPU Store
}

let SchedModel = AvispadoModel in {

let Latency = 1 in
{
    def : WriteRes<WriteVL, [AvispadoVSETVL]>;
    def : WriteRes<WriteVTYPE, [AvispadoVSETVL]>;
}

let Latency = 16, ResourceCycles = [8] in
{
  def : WriteRes<WriteVPULoad, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVPULoadStrided, [AvispadoVPUMemLoad]>;
  def : WriteRes<WriteVPULoadIndexed, [AvispadoVPUMemLoad]>;
}

let Latency = 16, ResourceCycles = [16] in
{
  def : WriteRes<WriteVPUStore, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVPUStoreStrided, [AvispadoVPUMemStore]>;
  def : WriteRes<WriteVPUStoreIndexed, [AvispadoVPUMemStore]>;

  def : WriteRes<WriteVPUScalarIALU, [AvispadoVPUArith]>;
  def : WriteRes<WriteVPUScalarFALU, [AvispadoVPUArith]>;
}

def : WriteRes<WriteVPUIALU, [AvispadoVPUArith]> {
  let Latency = 8;
  let ResourceCycles = [8];
}

def : WriteRes<WriteVPUFALU, [AvispadoVPUArith]> {
  let Latency = 10;
  let ResourceCycles = [10];
}

// By-passes and advanced (if any)
def : ReadAdvance<ReadVL, 0>;
def : ReadAdvance<ReadVTYPE, 0>;
def : ReadAdvance<ReadVPUScalarIALU, 0>;
def : ReadAdvance<ReadVPUScalarFALU, 0>;
def : ReadAdvance<ReadVPUIALU, 0>;
def : ReadAdvance<ReadVPUFALU, 0>;
def : ReadAdvance<ReadVPUMem, 0>;
def : ReadAdvance<ReadVPUStoreData,       0>;

// Unsupported extensions
defm : UnsupportedSchedZba;
defm : UnsupportedSchedZbb;

// Zfh is unsupported
let Unsupported = true in {
def : WriteRes<WriteFALU16, []>;
def : WriteRes<WriteFClass16, []>;
def : WriteRes<WriteFCvtF16ToF64, []>;
def : WriteRes<WriteFCvtF64ToF16, []>;
def : WriteRes<WriteFCvtI64ToF16, []>;
def : WriteRes<WriteFCvtF32ToF16, []>;
def : WriteRes<WriteFCvtI32ToF16, []>;
def : WriteRes<WriteFCvtF16ToI64, []>;
def : WriteRes<WriteFCvtF16ToF32, []>;
def : WriteRes<WriteFCvtF16ToI32, []>;
def : WriteRes<WriteFDiv16, []>;
def : WriteRes<WriteFCmp16, []>;
def : WriteRes<WriteFLD16, []>;
def : WriteRes<WriteFMA16, []>;
def : WriteRes<WriteFMinMax16, []>;
def : WriteRes<WriteFMul16, []>;
def : WriteRes<WriteFMovI16ToF16, []>;
def : WriteRes<WriteFMovF16ToI16, []>;
def : WriteRes<WriteFSGNJ16, []>;
def : WriteRes<WriteFST16, []>;
def : WriteRes<WriteFSqrt16, []>;

def : ReadAdvance<ReadFALU16, 0>;
def : ReadAdvance<ReadFClass16, 0>;
def : ReadAdvance<ReadFCvtF16ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF16, 0>;
def : ReadAdvance<ReadFCvtI64ToF16, 0>;
def : ReadAdvance<ReadFCvtF32ToF16, 0>;
def : ReadAdvance<ReadFCvtI32ToF16, 0>;
def : ReadAdvance<ReadFCvtF16ToI64, 0>;
def : ReadAdvance<ReadFCvtF16ToF32, 0>;
def : ReadAdvance<ReadFCvtF16ToI32, 0>;
def : ReadAdvance<ReadFDiv16, 0>;
def : ReadAdvance<ReadFCmp16, 0>;
def : ReadAdvance<ReadFMA16, 0>;
def : ReadAdvance<ReadFMinMax16, 0>;
def : ReadAdvance<ReadFMul16, 0>;
def : ReadAdvance<ReadFMovI16ToF16, 0>;
def : ReadAdvance<ReadFMovF16ToI16, 0>;
def : ReadAdvance<ReadFSGNJ16, 0>;
def : ReadAdvance<ReadFSqrt16, 0>;
} // Unsupported = true

}
