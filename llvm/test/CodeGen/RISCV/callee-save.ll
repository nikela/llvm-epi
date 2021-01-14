; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -verify-machineinstrs -mtriple=riscv64 \
; RUN:     -target-abi lp64f -mattr=+f < %s | \
; RUN:     FileCheck --check-prefix=LP64F %s

; RUN: llc -verify-machineinstrs -mtriple=riscv64 \
; RUN:     -target-abi lp64d -mattr=+f,+d < %s | \
; RUN:     FileCheck --check-prefix=LP64D %s


@var = global float 0.0

define void @foo() nounwind {
  ; Create lots of live variables to exhaust the supply of
  ; caller-saved registers
; LP64F-LABEL: foo:
; LP64F:       # %bb.0:
; LP64F-NEXT:    addi sp, sp, -48
; LP64F-NEXT:    fsw fs0, 44(sp)
; LP64F-NEXT:    fsw fs1, 40(sp)
; LP64F-NEXT:    fsw fs2, 36(sp)
; LP64F-NEXT:    fsw fs3, 32(sp)
; LP64F-NEXT:    fsw fs4, 28(sp)
; LP64F-NEXT:    fsw fs5, 24(sp)
; LP64F-NEXT:    fsw fs6, 20(sp)
; LP64F-NEXT:    fsw fs7, 16(sp)
; LP64F-NEXT:    fsw fs8, 12(sp)
; LP64F-NEXT:    fsw fs9, 8(sp)
; LP64F-NEXT:    fsw fs10, 4(sp)
; LP64F-NEXT:    fsw fs11, 0(sp)
; LP64F:         flw fs11, 0(sp)
; LP64F-NEXT:    flw fs10, 4(sp)
; LP64F-NEXT:    flw fs9, 8(sp)
; LP64F-NEXT:    flw fs8, 12(sp)
; LP64F-NEXT:    flw fs7, 16(sp)
; LP64F-NEXT:    flw fs6, 20(sp)
; LP64F-NEXT:    flw fs5, 24(sp)
; LP64F-NEXT:    flw fs4, 28(sp)
; LP64F-NEXT:    flw fs3, 32(sp)
; LP64F-NEXT:    flw fs2, 36(sp)
; LP64F-NEXT:    flw fs1, 40(sp)
; LP64F-NEXT:    flw fs0, 44(sp)
; LP64F-NEXT:    addi sp, sp, 48
; LP64F-NEXT:    ret
;
; LP64D-LABEL: foo:
; LP64D:       # %bb.0:
; LP64D-NEXT:    addi sp, sp, -96
; LP64D-NEXT:    fsd fs0, 88(sp)
; LP64D-NEXT:    fsd fs1, 80(sp)
; LP64D-NEXT:    fsd fs2, 72(sp)
; LP64D-NEXT:    fsd fs3, 64(sp)
; LP64D-NEXT:    fsd fs4, 56(sp)
; LP64D-NEXT:    fsd fs5, 48(sp)
; LP64D-NEXT:    fsd fs6, 40(sp)
; LP64D-NEXT:    fsd fs7, 32(sp)
; LP64D-NEXT:    fsd fs8, 24(sp)
; LP64D-NEXT:    fsd fs9, 16(sp)
; LP64D-NEXT:    fsd fs10, 8(sp)
; LP64D-NEXT:    fsd fs11, 0(sp)
; LP64D:         fld fs11, 0(sp)
; LP64D-NEXT:    fld fs10, 8(sp)
; LP64D-NEXT:    fld fs9, 16(sp)
; LP64D-NEXT:    fld fs8, 24(sp)
; LP64D-NEXT:    fld fs7, 32(sp)
; LP64D-NEXT:    fld fs6, 40(sp)
; LP64D-NEXT:    fld fs5, 48(sp)
; LP64D-NEXT:    fld fs4, 56(sp)
; LP64D-NEXT:    fld fs3, 64(sp)
; LP64D-NEXT:    fld fs2, 72(sp)
; LP64D-NEXT:    fld fs1, 80(sp)
; LP64D-NEXT:    fld fs0, 88(sp)
; LP64D-NEXT:    addi sp, sp, 96
; LP64D-NEXT:    ret
  %val1 = load volatile float, float* @var
  %val2 = load volatile float, float* @var
  %val3 = load volatile float, float* @var
  %val4 = load volatile float, float* @var
  %val5 = load volatile float, float* @var
  %val6 = load volatile float, float* @var
  %val7 = load volatile float, float* @var
  %val8 = load volatile float, float* @var
  %val9 = load volatile float, float* @var
  %val10 = load volatile float, float* @var
  %val11 = load volatile float, float* @var
  %val12 = load volatile float, float* @var
  %val13 = load volatile float, float* @var
  %val14 = load volatile float, float* @var
  %val15 = load volatile float, float* @var
  %val16 = load volatile float, float* @var
  %val17 = load volatile float, float* @var
  %val18 = load volatile float, float* @var
  %val19 = load volatile float, float* @var
  %val20 = load volatile float, float* @var
  %val21 = load volatile float, float* @var
  %val22 = load volatile float, float* @var
  %val23 = load volatile float, float* @var
  %val24 = load volatile float, float* @var
  %val25 = load volatile float, float* @var
  %val26 = load volatile float, float* @var
  %val27 = load volatile float, float* @var
  %val28 = load volatile float, float* @var
  %val29 = load volatile float, float* @var
  %val30 = load volatile float, float* @var
  %val31 = load volatile float, float* @var
  %val32 = load volatile float, float* @var

  store volatile float %val1, float* @var
  store volatile float %val2, float* @var
  store volatile float %val3, float* @var
  store volatile float %val4, float* @var
  store volatile float %val5, float* @var
  store volatile float %val6, float* @var
  store volatile float %val7, float* @var
  store volatile float %val8, float* @var
  store volatile float %val9, float* @var
  store volatile float %val10, float* @var
  store volatile float %val11, float* @var
  store volatile float %val12, float* @var
  store volatile float %val13, float* @var
  store volatile float %val14, float* @var
  store volatile float %val15, float* @var
  store volatile float %val16, float* @var
  store volatile float %val17, float* @var
  store volatile float %val18, float* @var
  store volatile float %val19, float* @var
  store volatile float %val20, float* @var
  store volatile float %val21, float* @var
  store volatile float %val22, float* @var
  store volatile float %val23, float* @var
  store volatile float %val24, float* @var
  store volatile float %val25, float* @var
  store volatile float %val26, float* @var
  store volatile float %val27, float* @var
  store volatile float %val28, float* @var
  store volatile float %val29, float* @var
  store volatile float %val30, float* @var
  store volatile float %val31, float* @var
  store volatile float %val32, float* @var

  ret void
}
