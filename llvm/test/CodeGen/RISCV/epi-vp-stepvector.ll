; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+v -verify-machineinstrs < %s | FileCheck %s

define <vscale x 8 x i64> @test_experimental_vp_stepvector_nxv8i64(<vscale x 8 x i1> %mask, i32 zeroext %evl) {
; CHECK-LABEL: test_experimental_vp_stepvector_nxv8i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e64, m8, ta, ma
; CHECK-NEXT:    vid.v v8, v0.t
; CHECK-NEXT:    ret
  %stepvector = call <vscale x 8 x i64> @llvm.experimental.vp.stepvector.nxv8i64(<vscale x 8 x i1> %mask, i32 %evl)
  ret <vscale x 8 x i64> %stepvector
}

define <vscale x 4 x i64> @test_experimental_vp_stepvector_nxv4i64(<vscale x 4 x i1> %mask, i32 zeroext %evl) {
; CHECK-LABEL: test_experimental_vp_stepvector_nxv4i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e64, m4, ta, ma
; CHECK-NEXT:    vid.v v8, v0.t
; CHECK-NEXT:    ret
  %stepvector = call <vscale x 4 x i64> @llvm.experimental.vp.stepvector.nxv4i64(<vscale x 4 x i1> %mask, i32 %evl)
  ret <vscale x 4 x i64> %stepvector
}

define <vscale x 2 x i64> @test_experimental_vp_stepvector_nxv2i64(<vscale x 2 x i1> %mask, i32 zeroext %evl) {
; CHECK-LABEL: test_experimental_vp_stepvector_nxv2i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e64, m2, ta, ma
; CHECK-NEXT:    vid.v v8, v0.t
; CHECK-NEXT:    ret
  %stepvector = call <vscale x 2 x i64> @llvm.experimental.vp.stepvector.nxv2i64(<vscale x 2 x i1> %mask, i32 %evl)
  ret <vscale x 2 x i64> %stepvector
}

define <vscale x 1 x i64> @test_experimental_vp_stepvector_nxv1i64(<vscale x 1 x i1> %mask, i32 zeroext %evl) {
; CHECK-LABEL: test_experimental_vp_stepvector_nxv1i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-NEXT:    vid.v v8, v0.t
; CHECK-NEXT:    ret
  %stepvector = call <vscale x 1 x i64> @llvm.experimental.vp.stepvector.nxv1i64(<vscale x 1 x i1> %mask, i32 %evl)
  ret <vscale x 1 x i64> %stepvector
}

define <vscale x 2 x i32> @test_experimental_vp_stepvector_nxv2i32(<vscale x 2 x i1> %mask, i32 zeroext %evl) {
; CHECK-LABEL: test_experimental_vp_stepvector_nxv2i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vid.v v8, v0.t
; CHECK-NEXT:    ret
  %stepvector = call <vscale x 2 x i32> @llvm.experimental.vp.stepvector.nxv2i32(<vscale x 2 x i1> %mask, i32 %evl)
  ret <vscale x 2 x i32> %stepvector
}

define <vscale x 4 x i16> @test_experimental_vp_stepvector_nxv4i16(<vscale x 4 x i1> %mask, i32 zeroext %evl) {
; CHECK-LABEL: test_experimental_vp_stepvector_nxv4i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, m1, ta, ma
; CHECK-NEXT:    vid.v v8, v0.t
; CHECK-NEXT:    ret
  %stepvector = call <vscale x 4 x i16> @llvm.experimental.vp.stepvector.nxv4i16(<vscale x 4 x i1> %mask, i32 %evl)
  ret <vscale x 4 x i16> %stepvector
}

define <vscale x 8 x i8> @test_experimental_vp_stepvector_nxv8i8(<vscale x 8 x i1> %mask, i32 zeroext %evl) {
; CHECK-LABEL: test_experimental_vp_stepvector_nxv8i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e8, m1, ta, ma
; CHECK-NEXT:    vid.v v8, v0.t
; CHECK-NEXT:    ret
  %stepvector = call <vscale x 8 x i8> @llvm.experimental.vp.stepvector.nxv8i8(<vscale x 8 x i1> %mask, i32 %evl)
  ret <vscale x 8 x i8> %stepvector
}

declare <vscale x 8 x i64> @llvm.experimental.vp.stepvector.nxv8i64(<vscale x 8 x i1>, i32)
declare <vscale x 4 x i64> @llvm.experimental.vp.stepvector.nxv4i64(<vscale x 4 x i1>, i32)
declare <vscale x 2 x i64> @llvm.experimental.vp.stepvector.nxv2i64(<vscale x 2 x i1>, i32)
declare <vscale x 1 x i64> @llvm.experimental.vp.stepvector.nxv1i64(<vscale x 1 x i1>, i32)
declare <vscale x 2 x i32> @llvm.experimental.vp.stepvector.nxv2i32(<vscale x 2 x i1>, i32)
declare <vscale x 4 x i16> @llvm.experimental.vp.stepvector.nxv4i16(<vscale x 4 x i1>, i32)
declare <vscale x 8 x i8> @llvm.experimental.vp.stepvector.nxv8i8(<vscale x 8 x i1>, i32)
