; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+m,+f,+d,+c,+v < %s -epi-pipeline | \
; RUN:     FileCheck %s

define <vscale x 2 x float> @fptrunc.f32.f64(<vscale x 2 x double> %a, i32 zeroext %gvl)
; CHECK-LABEL: fptrunc.f32.f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfncvt.f.f.w v10, v8
; CHECK-NEXT:    vmv.v.v v8, v10
; CHECK-NEXT:    ret
{
  %b = call <vscale x 2 x float> @llvm.vp.fptrunc.nxv2f32.nxv2f64(<vscale x 2 x double> %a,
            <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> undef, i1 true, i32 0), <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer),
            i32 %gvl)
  ret <vscale x 2 x float> %b
}

define <vscale x 2 x float> @fptrunc.f32.f64.mask(<vscale x 2 x double> %a, <vscale x 2 x i1> %mask, i32 zeroext %gvl)
; CHECK-LABEL: fptrunc.f32.f64.mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vfncvt.f.f.w v10, v8, v0.t
; CHECK-NEXT:    vmv.v.v v8, v10
; CHECK-NEXT:    ret
{
  %b = call <vscale x 2 x float> @llvm.vp.fptrunc.nxv2f32.nxv2f64(
               <vscale x 2 x double> %a,
               <vscale x 2 x i1> %mask,
               i32 %gvl)
  ret <vscale x 2 x float> %b
}

declare <vscale x 2 x float> @llvm.vp.fptrunc.nxv2f32.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, i32) #3
