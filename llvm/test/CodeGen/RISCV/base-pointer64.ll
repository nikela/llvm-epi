; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 < %s | FileCheck %s

define dso_local void @foo(i32 signext %n) nounwind {
; CHECK-LABEL: foo:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addi sp, sp, -64
; CHECK-NEXT:    sd ra, 56(sp)
; CHECK-NEXT:    sd s0, 48(sp)
; CHECK-NEXT:    sd s1, 40(sp)
; CHECK-NEXT:    addi s0, sp, 64
; CHECK-NEXT:    andi sp, sp, -32
; CHECK-NEXT:    mv s1, sp
; CHECK-NEXT:    slli a0, a0, 32
; CHECK-NEXT:    srli a0, a0, 32
; CHECK-NEXT:    slli a0, a0, 2
; CHECK-NEXT:    addi a0, a0, 15
; CHECK-NEXT:    addi a1, zero, 1
; CHECK-NEXT:    slli a1, a1, 35
; CHECK-NEXT:    addi a1, a1, -16
; CHECK-NEXT:    and a0, a0, a1
; CHECK-NEXT:    sub a0, sp, a0
; CHECK-NEXT:    mv sp, a0
; CHECK-NEXT:    lw a1, 4(a0)
; CHECK-NEXT:    addi a1, a1, 1
; CHECK-NEXT:    sw a1, 4(a0)
; CHECK-NEXT:    lw a0, 32(s1)
; CHECK-NEXT:    addi a0, a0, 1
; CHECK-NEXT:    sw a0, 32(s1)
; CHECK-NEXT:    lw a0, 28(s1)
; CHECK-NEXT:    addi a0, a0, 1
; CHECK-NEXT:    sw a0, 28(s1)
; CHECK-NEXT:    addi sp, s0, -64
; CHECK-NEXT:    ld s1, 40(sp)
; CHECK-NEXT:    ld s0, 48(sp)
; CHECK-NEXT:    ld ra, 56(sp)
; CHECK-NEXT:    addi sp, sp, 64
; CHECK-NEXT:    ret
entry:
  %w = alloca i32, align 32
  %y = alloca i32, align 4
  %0 = zext i32 %n to i64
  %vla = alloca i32, i64 %0, align 4
  %arrayidx = getelementptr inbounds i32, i32* %vla, i64 1
  %1 = load volatile i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %1, 1
  store volatile i32 %inc, i32* %arrayidx, align 4
  %w.0.w.0. = load volatile i32, i32* %w, align 32
  %inc1 = add nsw i32 %w.0.w.0., 1
  store volatile i32 %inc1, i32* %w, align 32
  %y.0.y.0. = load volatile i32, i32* %y, align 4
  %inc2 = add nsw i32 %y.0.y.0., 1
  store volatile i32 %inc2, i32* %y, align 4
  ret void
}
