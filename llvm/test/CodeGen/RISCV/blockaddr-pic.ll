; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple riscv32 -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV32-NOPIC
; RUN: llc -mtriple riscv64 -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV64-NOPIC
; RUN: llc -mtriple riscv32 -relocation-model=pic -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV32-PIC
; RUN: llc -mtriple riscv64 -relocation-model=pic -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV64-PIC

define signext i32 @foo(i32 signext %w) nounwind {
; RV32-NOPIC-LABEL: foo:
; RV32-NOPIC:       # %bb.0: # %entry
; RV32-NOPIC-NEXT:    addi sp, sp, -16
; RV32-NOPIC-NEXT:    lui a1, %hi(.Ltmp0)
; RV32-NOPIC-NEXT:    addi a1, a1, %lo(.Ltmp0)
; RV32-NOPIC-NEXT:    addi a2, zero, 101
; RV32-NOPIC-NEXT:    sw a1, 8(sp)
; RV32-NOPIC-NEXT:    blt a0, a2, .LBB0_3
; RV32-NOPIC-NEXT:  # %bb.1: # %if.then
; RV32-NOPIC-NEXT:    lw a0, 8(sp)
; RV32-NOPIC-NEXT:    jr a0
; RV32-NOPIC-NEXT:  .Ltmp0: # Block address taken
; RV32-NOPIC-NEXT:  .LBB0_2: # %return
; RV32-NOPIC-NEXT:    addi a0, zero, 4
; RV32-NOPIC-NEXT:    addi sp, sp, 16
; RV32-NOPIC-NEXT:    ret
; RV32-NOPIC-NEXT:  .LBB0_3: # %return.clone
; RV32-NOPIC-NEXT:    addi a0, zero, 3
; RV32-NOPIC-NEXT:    addi sp, sp, 16
; RV32-NOPIC-NEXT:    ret
;
; RV64-NOPIC-LABEL: foo:
; RV64-NOPIC:       # %bb.0: # %entry
; RV64-NOPIC-NEXT:    addi sp, sp, -16
; RV64-NOPIC-NEXT:    lui a1, %hi(.Ltmp0)
; RV64-NOPIC-NEXT:    addi a1, a1, %lo(.Ltmp0)
; RV64-NOPIC-NEXT:    addi a2, zero, 101
; RV64-NOPIC-NEXT:    sd a1, 8(sp)
; RV64-NOPIC-NEXT:    blt a0, a2, .LBB0_3
; RV64-NOPIC-NEXT:  # %bb.1: # %if.then
; RV64-NOPIC-NEXT:    ld a0, 8(sp)
; RV64-NOPIC-NEXT:    jr a0
; RV64-NOPIC-NEXT:  .Ltmp0: # Block address taken
; RV64-NOPIC-NEXT:  .LBB0_2: # %return
; RV64-NOPIC-NEXT:    addi a0, zero, 4
; RV64-NOPIC-NEXT:    addi sp, sp, 16
; RV64-NOPIC-NEXT:    ret
; RV64-NOPIC-NEXT:  .LBB0_3: # %return.clone
; RV64-NOPIC-NEXT:    addi a0, zero, 3
; RV64-NOPIC-NEXT:    addi sp, sp, 16
; RV64-NOPIC-NEXT:    ret
;
; RV32-PIC-LABEL: foo:
; RV32-PIC:       # %bb.0: # %entry
; RV32-PIC-NEXT:    addi sp, sp, -16
; RV32-PIC-NEXT:  .LBB0_4: # %entry
; RV32-PIC-NEXT:    # Label of block must be emitted
; RV32-PIC-NEXT:    auipc a1, %pcrel_hi(.Ltmp0)
; RV32-PIC-NEXT:    addi a1, a1, %pcrel_lo(.LBB0_4)
; RV32-PIC-NEXT:    addi a2, zero, 101
; RV32-PIC-NEXT:    sw a1, 8(sp)
; RV32-PIC-NEXT:    blt a0, a2, .LBB0_3
; RV32-PIC-NEXT:  # %bb.1: # %if.then
; RV32-PIC-NEXT:    lw a0, 8(sp)
; RV32-PIC-NEXT:    jr a0
; RV32-PIC-NEXT:  .Ltmp0: # Block address taken
; RV32-PIC-NEXT:  .LBB0_2: # %return
; RV32-PIC-NEXT:    addi a0, zero, 4
; RV32-PIC-NEXT:    addi sp, sp, 16
; RV32-PIC-NEXT:    ret
; RV32-PIC-NEXT:  .LBB0_3: # %return.clone
; RV32-PIC-NEXT:    addi a0, zero, 3
; RV32-PIC-NEXT:    addi sp, sp, 16
; RV32-PIC-NEXT:    ret
;
; RV64-PIC-LABEL: foo:
; RV64-PIC:       # %bb.0: # %entry
; RV64-PIC-NEXT:    addi sp, sp, -16
; RV64-PIC-NEXT:  .LBB0_4: # %entry
; RV64-PIC-NEXT:    # Label of block must be emitted
; RV64-PIC-NEXT:    auipc a1, %pcrel_hi(.Ltmp0)
; RV64-PIC-NEXT:    addi a1, a1, %pcrel_lo(.LBB0_4)
; RV64-PIC-NEXT:    addi a2, zero, 101
; RV64-PIC-NEXT:    sd a1, 8(sp)
; RV64-PIC-NEXT:    blt a0, a2, .LBB0_3
; RV64-PIC-NEXT:  # %bb.1: # %if.then
; RV64-PIC-NEXT:    ld a0, 8(sp)
; RV64-PIC-NEXT:    jr a0
; RV64-PIC-NEXT:  .Ltmp0: # Block address taken
; RV64-PIC-NEXT:  .LBB0_2: # %return
; RV64-PIC-NEXT:    addi a0, zero, 4
; RV64-PIC-NEXT:    addi sp, sp, 16
; RV64-PIC-NEXT:    ret
; RV64-PIC-NEXT:  .LBB0_3: # %return.clone
; RV64-PIC-NEXT:    addi a0, zero, 3
; RV64-PIC-NEXT:    addi sp, sp, 16
; RV64-PIC-NEXT:    ret

entry:
  %x = alloca i8*, align 8
  store i8* blockaddress(@foo, %test_block), i8** %x, align 8
  %cmp = icmp sgt i32 %w, 100
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %addr = load i8*, i8** %x, align 8
  br label %indirectgoto

if.end:
  br label %return

test_block:
  br label %return

return:
  %retval = phi i32 [ 3, %if.end ], [ 4, %test_block ]
  ret i32 %retval

indirectgoto:
  %indirect.goto.dest = phi i8* [ %addr, %if.then ]
  indirectbr i8* %addr, [ label %test_block ]
}

