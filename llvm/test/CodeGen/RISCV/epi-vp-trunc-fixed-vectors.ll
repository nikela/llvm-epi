; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+m,+f,+d,+c,+v -riscv-v-vector-bits-min=128 \
; RUN:     < %s | FileCheck %s

define <4 x i32> @test_vp_trunc_v4i32_v4i64(<4 x i64> %a, i32 zeroext %gvl) {
; CHECK-LABEL: test_vp_trunc_v4i32_v4i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vnsrl.wi v10, v8, 0
; CHECK-NEXT:    vmv.v.v v8, v10
; CHECK-NEXT:    ret
  %b = call <4 x i32> @llvm.vp.trunc.v4i32.v4i64(<4 x i64> %a,
            <4 x i1> shufflevector (<4 x i1> insertelement (<4 x i1> undef, i1 true, i32 0), <4 x i1> undef, <4 x i32> zeroinitializer),
            i32 %gvl)
  ret <4 x i32> %b
}

define <4 x i32> @test_vp_trunc_v4i32_v4i64_mask(<4 x i64> %a, <4 x i1> %mask, i32 zeroext %gvl) {
; CHECK-LABEL: test_vp_trunc_v4i32_v4i64_mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vnsrl.wi v10, v8, 0, v0.t
; CHECK-NEXT:    vmv.v.v v8, v10
; CHECK-NEXT:    ret
  %b = call <4 x i32> @llvm.vp.trunc.v4i32.v4i64(
               <4 x i64> %a,
               <4 x i1> %mask,
               i32 %gvl)
  ret <4 x i32> %b
}

define <16 x i8> @test_vp_trunc_v16i8_v16i32(<16 x i32> %a, i32 zeroext %evl) {
; CHECK-LABEL: test_vp_trunc_v16i8_v16i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, m2, ta, ma
; CHECK-NEXT:    vnsrl.wi v12, v8, 0
; CHECK-NEXT:    vsetvli zero, zero, e8, m1, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v12, 0
; CHECK-NEXT:    ret
    %m.first = insertelement <16 x i1> undef, i1 1, i32 0
    %m.splat = shufflevector <16 x i1> %m.first, <16 x i1> undef, <16 x i32> zeroinitializer
    %x = call <16 x i8> @llvm.vp.trunc.v16i8.v16i32(<16 x i32> %a, <16 x i1> %m.splat, i32 %evl)
    ret <16 x i8> %x
}

define <16 x i8> @test_vp_trunc_v16i8_v16i64(<16 x i64> %a, i32 zeroext %evl) {
; CHECK-LABEL: test_vp_trunc_v16i8_v16i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m4, ta, ma
; CHECK-NEXT:    vnsrl.wi v16, v8, 0
; CHECK-NEXT:    vsetvli zero, zero, e16, m2, ta, ma
; CHECK-NEXT:    vnsrl.wi v10, v16, 0
; CHECK-NEXT:    vsetvli zero, zero, e8, m1, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v10, 0
; CHECK-NEXT:    ret
    %m.first = insertelement <16 x i1> undef, i1 1, i32 0
    %m.splat = shufflevector <16 x i1> %m.first, <16 x i1> undef, <16 x i32> zeroinitializer
    %x = call <16 x i8> @llvm.vp.trunc.v16i8.v16i64(<16 x i64> %a, <16 x i1> %m.splat, i32 %evl)
    ret <16 x i8> %x
}

define <8 x i16> @test_vp_trunc_v8i16_v8i64(<8 x i64> %a, i32 zeroext %evl) {
; CHECK-LABEL: test_vp_trunc_v8i16_v8i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m2, ta, ma
; CHECK-NEXT:    vnsrl.wi v12, v8, 0
; CHECK-NEXT:    vsetvli zero, zero, e16, m1, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v12, 0
; CHECK-NEXT:    ret
    %m.first = insertelement <8 x i1> undef, i1 1, i32 0
    %m.splat = shufflevector <8 x i1> %m.first, <8 x i1> undef, <8 x i32> zeroinitializer
    %x = call <8 x i16> @llvm.vp.trunc.v8i16.v8i64(<8 x i64> %a, <8 x i1> %m.splat, i32 %evl)
    ret <8 x i16> %x
}

declare <16 x i8> @llvm.vp.trunc.v16i8.v16i32(<16 x i32> %op, <16 x i1> %mask, i32 %evl)
declare <16 x i8> @llvm.vp.trunc.v16i8.v16i64(<16 x i64> %op, <16 x i1> %mask, i32 %evl)
declare <8 x i16> @llvm.vp.trunc.v8i16.v8i64(<8 x i64> %op, <8 x i1> %mask, i32 %evl)
declare <4 x i32> @llvm.vp.trunc.v4i32.v4i64(<4 x i64>, <4 x i1>, i32)
