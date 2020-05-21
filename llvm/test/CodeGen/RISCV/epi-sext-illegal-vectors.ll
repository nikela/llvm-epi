; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+experimental-v -verify-machineinstrs < %s \
; RUN:    | FileCheck %s

; NOTE: This test checks that illegal vector types (that would correspond to
; LMUL < 1) are properly legalized to the 'LMUL = 1' type that matches the
; number of elements, and that the required sign extension instructions are
; emitted.

;                       1xi8
;                1xi16  2xi8  "LMUL < 1"
;         1xi32  2xi16  4xi8
; --------------------------
;  1xi64  2xi32  4xi16  8xi8   LMUL = 1
;  2xi64  4xi32  8xi16 16xi8   LMUL = 2
;  4xi64  8xi32 16xi16 32xi8   LMUL = 4
;  8xi64 16xi32 32xi16 64xi8   LMUL = 8
; --------------------------
; 16xi64 32xi32 64xi16
; 32xi64 64xi32               "LMUL > 8"
; 64xi64

define <vscale x 1 x i64> @sext_nxv1i8(<vscale x 1 x i8> %v)
; CHECK-LABEL: sext_nxv1i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi a0, zero, 56
; CHECK-NEXT:    vsetvli a1, zero, e64,m1
; CHECK-NEXT:    vsll.vx v1, v16, a0
; CHECK-NEXT:    vsra.vx v16, v1, a0
; CHECK-NEXT:    ret
{
  %sv = sext <vscale x 1 x i8> %v to <vscale x 1 x i64>
  ret <vscale x 1 x i64> %sv
}
define <vscale x 2 x i32> @sext_nxv2i8(<vscale x 2 x i8> %v)
; CHECK-LABEL: sext_nxv2i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, zero, e32,m1
; CHECK-NEXT:    vsll.vi v1, v16, 24
; CHECK-NEXT:    vsra.vi v16, v1, 24
; CHECK-NEXT:    ret
{
  %sv = sext <vscale x 2 x i8> %v to <vscale x 2 x i32>
  ret <vscale x 2 x i32> %sv
}
define <vscale x 4 x i16> @sext_nxv4i8(<vscale x 4 x i8> %v)
; CHECK-LABEL: sext_nxv4i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, zero, e16,m1
; CHECK-NEXT:    vsll.vi v1, v16, 8
; CHECK-NEXT:    vsra.vi v16, v1, 8
; CHECK-NEXT:    ret
{
  %sv = sext <vscale x 4 x i8> %v to <vscale x 4 x i16>
  ret <vscale x 4 x i16> %sv
}


define <vscale x 1 x i64> @sext_nxv1i16(<vscale x 1 x i16> %v)
; CHECK-LABEL: sext_nxv1i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi a0, zero, 48
; CHECK-NEXT:    vsetvli a1, zero, e64,m1
; CHECK-NEXT:    vsll.vx v1, v16, a0
; CHECK-NEXT:    vsra.vx v16, v1, a0
; CHECK-NEXT:    ret
{
  %sv = sext <vscale x 1 x i16> %v to <vscale x 1 x i64>
  ret <vscale x 1 x i64> %sv
}
define <vscale x 2 x i32> @sext_nxv2i16(<vscale x 2 x i16> %v)
; CHECK-LABEL: sext_nxv2i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, zero, e32,m1
; CHECK-NEXT:    vsll.vi v1, v16, 16
; CHECK-NEXT:    vsra.vi v16, v1, 16
; CHECK-NEXT:    ret
{
  %sv = sext <vscale x 2 x i16> %v to <vscale x 2 x i32>
  ret <vscale x 2 x i32> %sv
}


define <vscale x 1 x i64> @sext_nxv1i32(<vscale x 1 x i32> %v)
; CHECK-LABEL: sext_nxv1i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi a0, zero, 32
; CHECK-NEXT:    vsetvli a1, zero, e64,m1
; CHECK-NEXT:    vsll.vx v1, v16, a0
; CHECK-NEXT:    vsra.vx v16, v1, a0
; CHECK-NEXT:    ret
{
  %sv = sext <vscale x 1 x i32> %v to <vscale x 1 x i64>
  ret <vscale x 1 x i64> %sv
}
