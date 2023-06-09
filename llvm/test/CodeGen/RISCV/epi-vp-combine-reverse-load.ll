; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+f,+v -verify-machineinstrs < %s | FileCheck %s

define <vscale x 2 x float> @test_reverse_load_combiner(<vscale x 2 x float>* %ptr, i32 zeroext %evl) {
; CHECK-LABEL: test_reverse_load_combiner:
; CHECK:       # %bb.0:
; CHECK-NEXT:    slli a2, a1, 2
; CHECK-NEXT:    add a0, a2, a0
; CHECK-NEXT:    addi a0, a0, -4
; CHECK-NEXT:    li a2, -4
; CHECK-NEXT:    vsetvli zero, a1, e32, m1, ta, ma
; CHECK-NEXT:    vlse32.v v8, (a0), a2
; CHECK-NEXT:    ret
  %head = insertelement <vscale x 2 x i1> undef, i1 1, i32 0
  %allones = shufflevector <vscale x 2 x i1> %head, <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer

  %load = call <vscale x 2 x float> @llvm.vp.load.nxv2f32.p0nxv2f32(<vscale x 2 x float>* %ptr, <vscale x 2 x i1> %allones, i32 %evl)
  %rev = call <vscale x 2 x float> @llvm.experimental.vp.reverse.nxv2f32(<vscale x 2 x float> %load, <vscale x 2 x i1> %allones, i32 %evl)
  ret <vscale x 2 x float> %rev
}

declare <vscale x 2 x float> @llvm.vp.load.nxv2f32.p0nxv2f32(<vscale x 2 x float>* nocapture, <vscale x 2 x i1>, i32)
declare <vscale x 2 x float> @llvm.experimental.vp.reverse.nxv2f32(<vscale x 2 x float>, <vscale x 2 x i1>, i32)
