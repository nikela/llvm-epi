; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+m,+f,+d,+c,+v -riscv-v-vector-bits-min=128 \
; RUN:     < %s | FileCheck %s

define <2 x i32> @fptoui.i32.f32(<2 x float> %a, i32 zeroext %gvl)
; CHECK-LABEL: fptoui.i32.f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, mf2, ta, ma
; CHECK-NEXT:    vfcvt.rtz.xu.f.v v8, v8
; CHECK-NEXT:    ret
{
  %b = call <2 x i32> @llvm.vp.fptoui.v2i32.v2f32(<2 x float> %a,
            <2 x i1> shufflevector (<2 x i1> insertelement (<2 x i1> undef, i1 true, i32 0), <2 x i1> undef, <2 x i32> zeroinitializer),
            i32 %gvl)
  ret <2 x i32> %b
}

define <2 x i32> @fptoui.i32.f32.mask(<2 x float> %a, <2 x i1> %mask, i32 zeroext %gvl)
; CHECK-LABEL: fptoui.i32.f32.mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, mf2, ta, ma
; CHECK-NEXT:    vfcvt.rtz.xu.f.v v8, v8, v0.t
; CHECK-NEXT:    ret
{
  %b = call <2 x i32> @llvm.vp.fptoui.v2i32.v2f32(
               <2 x float> %a,
               <2 x i1> %mask,
               i32 %gvl)
  ret <2 x i32> %b
}

declare <2 x i32> @llvm.vp.fptoui.v2i32.v2f32(<2 x float> %a, <2 x i1> %mask, i32 %gvl)

define <2 x i64> @fptoui.i64.f32(<2 x float> %a, i32 zeroext %gvl)
; CHECK-LABEL: fptoui.i64.f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, mf2, ta, ma
; CHECK-NEXT:    vfwcvt.rtz.xu.f.v v9, v8
; CHECK-NEXT:    vmv1r.v v8, v9
; CHECK-NEXT:    ret
{
  %b = call <2 x i64> @llvm.vp.fptoui.v2i64.v2f32(<2 x float> %a,
            <2 x i1> shufflevector (<2 x i1> insertelement (<2 x i1> undef, i1 true, i32 0), <2 x i1> undef, <2 x i32> zeroinitializer),
            i32 %gvl)
  ret <2 x i64> %b
}

define <2 x i64> @fptoui.i64.f32.mask(<2 x float> %a, <2 x i1> %mask, i32 zeroext %gvl)
; CHECK-LABEL: fptoui.i64.f32.mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, mf2, ta, ma
; CHECK-NEXT:    vfwcvt.rtz.xu.f.v v9, v8, v0.t
; CHECK-NEXT:    vmv1r.v v8, v9
; CHECK-NEXT:    ret
{
  %b = call <2 x i64> @llvm.vp.fptoui.v2i64.v2f32(
               <2 x float> %a,
               <2 x i1> %mask,
               i32 %gvl)
  ret <2 x i64> %b
}

declare <2 x i64> @llvm.vp.fptoui.v2i64.v2f32(<2 x float> %a, <2 x i1> %mask, i32 %gvl)

define <2 x i32> @fptoui.i32.f64(<2 x double> %a, i32 zeroext %gvl)
; CHECK-LABEL: fptoui.i32.f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, mf2, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v9, v8
; CHECK-NEXT:    vmv1r.v v8, v9
; CHECK-NEXT:    ret
{
  %b = call <2 x i32> @llvm.vp.fptoui.v2i32.v2f64(<2 x double> %a,
            <2 x i1> shufflevector (<2 x i1> insertelement (<2 x i1> undef, i1 true, i32 0), <2 x i1> undef, <2 x i32> zeroinitializer),
            i32 %gvl)
  ret <2 x i32> %b
}

define <2 x i32> @fptoui.i32.f64.mask(<2 x double> %a, <2 x i1> %mask, i32 zeroext %gvl)
; CHECK-LABEL: fptoui.i32.f64.mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, mf2, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v9, v8, v0.t
; CHECK-NEXT:    vmv1r.v v8, v9
; CHECK-NEXT:    ret
{
  %b = call <2 x i32> @llvm.vp.fptoui.v2i32.v2f64(
               <2 x double> %a,
               <2 x i1> %mask,
               i32 %gvl)
  ret <2 x i32> %b
}

declare <2 x i32> @llvm.vp.fptoui.v2i32.v2f64(<2 x double> %a, <2 x i1> %mask, i32 %gvl)

define <4 x i8> @test_vp_fptoui_v4i8_v4f32(<4 x float> %a, i32 zeroext %evl) {
; CHECK-LABEL: test_vp_fptoui_v4i8_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v9, v8
; CHECK-NEXT:    vsetvli zero, zero, e8, mf4, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v9, 0
; CHECK-NEXT:    ret
    %m.first = insertelement <4 x i1> undef, i1 1, i32 0
    %m.splat = shufflevector <4 x i1> %m.first, <4 x i1> undef, <4 x i32> zeroinitializer
    %x = call <4 x i8> @llvm.vp.fptoui.v4i8.v4f32(<4 x float> %a, <4 x i1> %m.splat, i32 %evl)
    ret <4 x i8> %x
}

define <4 x i8> @test_vp_fptoui_v4i8_v4f32_mask(<4 x float> %a, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: test_vp_fptoui_v4i8_v4f32_mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e16, mf2, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v9, v8, v0.t
; CHECK-NEXT:    vsetvli zero, zero, e8, mf4, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v9, 0, v0.t
; CHECK-NEXT:    ret
    %x = call <4 x i8> @llvm.vp.fptoui.v4i8.v4f32(<4 x float> %a, <4 x i1> %m, i32 %evl)
    ret <4 x i8> %x
}

define <4 x i8> @test_vp_fptoui_v4i8_v4f64(<4 x double> %a, i32 zeroext %evl) {
; CHECK-LABEL: test_vp_fptoui_v4i8_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v10, v8
; CHECK-NEXT:    vsetvli zero, zero, e16, mf2, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v10, 0
; CHECK-NEXT:    vsetvli zero, zero, e8, mf4, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v8, 0
; CHECK-NEXT:    ret
    %m.first = insertelement <4 x i1> undef, i1 1, i32 0
    %m.splat = shufflevector <4 x i1> %m.first, <4 x i1> undef, <4 x i32> zeroinitializer
    %x = call <4 x i8> @llvm.vp.fptoui.v4i8.v4f64(<4 x double> %a, <4 x i1> %m.splat, i32 %evl)
    ret <4 x i8> %x
}

define <4 x i8> @test_vp_fptoui_v4i8_v4f64_mask(<4 x double> %a, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: test_vp_fptoui_v4i8_v4f64_mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v10, v8, v0.t
; CHECK-NEXT:    vsetvli zero, zero, e16, mf2, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v10, 0, v0.t
; CHECK-NEXT:    vsetvli zero, zero, e8, mf4, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v8, 0, v0.t
; CHECK-NEXT:    ret
    %x = call <4 x i8> @llvm.vp.fptoui.v4i8.v4f64(<4 x double> %a, <4 x i1> %m, i32 %evl)
    ret <4 x i8> %x
}

define <4 x i16> @test_vp_fptoui_v4i16_v4f64(<4 x double> %a, i32 zeroext %evl) {
; CHECK-LABEL: test_vp_fptoui_v4i16_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v10, v8
; CHECK-NEXT:    vsetvli zero, zero, e16, mf2, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v10, 0
; CHECK-NEXT:    ret
    %m.first = insertelement <4 x i1> undef, i1 1, i32 0
    %m.splat = shufflevector <4 x i1> %m.first, <4 x i1> undef, <4 x i32> zeroinitializer
    %x = call <4 x i16> @llvm.vp.fptoui.v4i16.v4f64(<4 x double> %a, <4 x i1> %m.splat, i32 %evl)
    ret <4 x i16> %x
}

define <4 x i16> @test_vp_fptoui_v4i16_v4f64_mask(<4 x double> %a, <4 x i1> %m, i32 zeroext %evl) {
; CHECK-LABEL: test_vp_fptoui_v4i16_v4f64_mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfncvt.rtz.xu.f.w v10, v8, v0.t
; CHECK-NEXT:    vsetvli zero, zero, e16, mf2, ta, ma
; CHECK-NEXT:    vnsrl.wi v8, v10, 0, v0.t
; CHECK-NEXT:    ret
    %x = call <4 x i16> @llvm.vp.fptoui.v4i16.v4f64(<4 x double> %a, <4 x i1> %m, i32 %evl)
    ret <4 x i16> %x
}

declare <4 x i8> @llvm.vp.fptoui.v4i8.v4f32(<4 x float> %op, <4 x i1> %mask, i32 %evl)
declare <4 x i8> @llvm.vp.fptoui.v4i8.v4f64(<4 x double> %op, <4 x i1> %mask, i32 %evl)
declare <4 x i16> @llvm.vp.fptoui.v4i16.v4f64(<4 x double> %op, <4 x i1> %mask, i32 %evl)
