; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=riscv32 -mattr=+m | FileCheck %s --check-prefixes=RISCV32
; RUN: llc < %s -mtriple=riscv64 -mattr=+m | FileCheck %s --check-prefixes=RISCV64

define { i128, i8 } @muloti_test(i128 %l, i128 %r) nounwind {
; RISCV32-LABEL: muloti_test:
; RISCV32:       # %bb.0: # %start
; RISCV32-NEXT:    addi sp, sp, -80
; RISCV32-NEXT:    sw ra, 76(sp)
; RISCV32-NEXT:    sw s1, 72(sp)
; RISCV32-NEXT:    sw s2, 68(sp)
; RISCV32-NEXT:    sw s3, 64(sp)
; RISCV32-NEXT:    sw s4, 60(sp)
; RISCV32-NEXT:    sw s5, 56(sp)
; RISCV32-NEXT:    sw s6, 52(sp)
; RISCV32-NEXT:    sw s7, 48(sp)
; RISCV32-NEXT:    mv s3, a2
; RISCV32-NEXT:    mv s1, a1
; RISCV32-NEXT:    mv s2, a0
; RISCV32-NEXT:    sw zero, 12(sp)
; RISCV32-NEXT:    sw zero, 8(sp)
; RISCV32-NEXT:    sw zero, 28(sp)
; RISCV32-NEXT:    sw zero, 24(sp)
; RISCV32-NEXT:    lw s4, 4(a2)
; RISCV32-NEXT:    sw s4, 4(sp)
; RISCV32-NEXT:    lw s6, 0(a2)
; RISCV32-NEXT:    sw s6, 0(sp)
; RISCV32-NEXT:    lw s5, 4(a1)
; RISCV32-NEXT:    sw s5, 20(sp)
; RISCV32-NEXT:    lw s7, 0(a1)
; RISCV32-NEXT:    sw s7, 16(sp)
; RISCV32-NEXT:    addi a0, sp, 32
; RISCV32-NEXT:    addi a1, sp, 16
; RISCV32-NEXT:    mv a2, sp
; RISCV32-NEXT:    call __multi3
; RISCV32-NEXT:    lw t4, 12(s1)
; RISCV32-NEXT:    lw a1, 8(s1)
; RISCV32-NEXT:    mul a2, s4, a1
; RISCV32-NEXT:    mul a3, t4, s6
; RISCV32-NEXT:    add a7, a3, a2
; RISCV32-NEXT:    lw a2, 12(s3)
; RISCV32-NEXT:    lw a3, 8(s3)
; RISCV32-NEXT:    mul a5, s5, a3
; RISCV32-NEXT:    mul s1, a2, s7
; RISCV32-NEXT:    add a5, s1, a5
; RISCV32-NEXT:    mul s1, a3, s7
; RISCV32-NEXT:    mul a4, a1, s6
; RISCV32-NEXT:    add s1, a4, s1
; RISCV32-NEXT:    sltu a4, s1, a4
; RISCV32-NEXT:    mulhu a6, a3, s7
; RISCV32-NEXT:    add t1, a6, a5
; RISCV32-NEXT:    mulhu t2, a1, s6
; RISCV32-NEXT:    add t3, t2, a7
; RISCV32-NEXT:    add a5, t3, t1
; RISCV32-NEXT:    add a5, a5, a4
; RISCV32-NEXT:    lw a4, 44(sp)
; RISCV32-NEXT:    add a5, a4, a5
; RISCV32-NEXT:    lw a0, 40(sp)
; RISCV32-NEXT:    add a7, a0, s1
; RISCV32-NEXT:    sltu t0, a7, a0
; RISCV32-NEXT:    add s1, a5, t0
; RISCV32-NEXT:    beq s1, a4, .LBB0_2
; RISCV32-NEXT:  # %bb.1: # %start
; RISCV32-NEXT:    sltu t0, s1, a4
; RISCV32-NEXT:  .LBB0_2: # %start
; RISCV32-NEXT:    snez a0, s4
; RISCV32-NEXT:    snez a4, t4
; RISCV32-NEXT:    and a0, a4, a0
; RISCV32-NEXT:    snez a4, s5
; RISCV32-NEXT:    snez a5, a2
; RISCV32-NEXT:    and a4, a5, a4
; RISCV32-NEXT:    mulhu a5, a2, s7
; RISCV32-NEXT:    snez a5, a5
; RISCV32-NEXT:    or a4, a4, a5
; RISCV32-NEXT:    mulhu a5, t4, s6
; RISCV32-NEXT:    snez a5, a5
; RISCV32-NEXT:    or a0, a0, a5
; RISCV32-NEXT:    sltu t2, t3, t2
; RISCV32-NEXT:    mulhu a5, s4, a1
; RISCV32-NEXT:    snez a5, a5
; RISCV32-NEXT:    or t3, a0, a5
; RISCV32-NEXT:    sltu a5, t1, a6
; RISCV32-NEXT:    mulhu a0, s5, a3
; RISCV32-NEXT:    snez a0, a0
; RISCV32-NEXT:    or a0, a4, a0
; RISCV32-NEXT:    lw a4, 36(sp)
; RISCV32-NEXT:    sw a4, 4(s2)
; RISCV32-NEXT:    lw a4, 32(sp)
; RISCV32-NEXT:    sw a4, 0(s2)
; RISCV32-NEXT:    sw a7, 8(s2)
; RISCV32-NEXT:    sw s1, 12(s2)
; RISCV32-NEXT:    or a0, a0, a5
; RISCV32-NEXT:    or a4, t3, t2
; RISCV32-NEXT:    or a1, a1, t4
; RISCV32-NEXT:    or a2, a3, a2
; RISCV32-NEXT:    snez a2, a2
; RISCV32-NEXT:    snez a1, a1
; RISCV32-NEXT:    and a1, a1, a2
; RISCV32-NEXT:    or a1, a1, a4
; RISCV32-NEXT:    or a0, a1, a0
; RISCV32-NEXT:    or a0, a0, t0
; RISCV32-NEXT:    andi a0, a0, 1
; RISCV32-NEXT:    sb a0, 16(s2)
; RISCV32-NEXT:    lw s7, 48(sp)
; RISCV32-NEXT:    lw s6, 52(sp)
; RISCV32-NEXT:    lw s5, 56(sp)
; RISCV32-NEXT:    lw s4, 60(sp)
; RISCV32-NEXT:    lw s3, 64(sp)
; RISCV32-NEXT:    lw s2, 68(sp)
; RISCV32-NEXT:    lw s1, 72(sp)
; RISCV32-NEXT:    lw ra, 76(sp)
; RISCV32-NEXT:    addi sp, sp, 80
; RISCV32-NEXT:    ret
;
; RISCV64-LABEL: muloti_test:
; RISCV64:       # %bb.0: # %start
; RISCV64-NEXT:    mul a6, a4, a1
; RISCV64-NEXT:    mul a5, a2, a3
; RISCV64-NEXT:    add a6, a5, a6
; RISCV64-NEXT:    mul a5, a1, a3
; RISCV64-NEXT:    sd a5, 0(a0)
; RISCV64-NEXT:    mulhu a7, a1, a3
; RISCV64-NEXT:    add a5, a7, a6
; RISCV64-NEXT:    sd a5, 8(a0)
; RISCV64-NEXT:    sltu a6, a5, a7
; RISCV64-NEXT:    snez a7, a4
; RISCV64-NEXT:    snez a5, a2
; RISCV64-NEXT:    and a5, a5, a7
; RISCV64-NEXT:    mulhu a2, a2, a3
; RISCV64-NEXT:    snez a2, a2
; RISCV64-NEXT:    or a2, a5, a2
; RISCV64-NEXT:    mulhu a1, a4, a1
; RISCV64-NEXT:    snez a1, a1
; RISCV64-NEXT:    or a1, a2, a1
; RISCV64-NEXT:    or a1, a1, a6
; RISCV64-NEXT:    sb a1, 16(a0)
; RISCV64-NEXT:    ret
start:
  %0 = tail call { i128, i1 } @llvm.umul.with.overflow.i128(i128 %l, i128 %r) #2
  %1 = extractvalue { i128, i1 } %0, 0
  %2 = extractvalue { i128, i1 } %0, 1
  %3 = zext i1 %2 to i8
  %4 = insertvalue { i128, i8 } undef, i128 %1, 0
  %5 = insertvalue { i128, i8 } %4, i8 %3, 1
  ret { i128, i8 } %5
}

declare { i128, i1 } @llvm.umul.with.overflow.i128(i128, i128) nounwind
