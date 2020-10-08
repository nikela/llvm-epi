; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+m,+f,+d,+c,+experimental-v < %s | \
; RUN:     FileCheck %s

define <vscale x 2 x i32> @fptosi.i32.f32(<vscale x 2 x float> %a, i32 %gvl)
; CHECK-LABEL: fptosi.i32.f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, a0, e32,m1,tu,mu
; CHECK-NEXT:    vfcvt.x.f.v v16, v16
; CHECK-NEXT:    ret
{
  %b = call <vscale x 2 x i32> @llvm.vp.fptosi.nxv2i32.nxv2f32(<vscale x 2 x float> %a,
            metadata !"fpexcept.ignore",
            <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> undef, i1 true, i32 0), <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer),
            i32 %gvl)
  ret <vscale x 2 x i32> %b
}

define <vscale x 2 x i32> @fptosi.i32.f32.mask(<vscale x 2 x float> %a, <vscale x 2 x i1> %mask, i32 %gvl)
; CHECK-LABEL: fptosi.i32.f32.mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, a0, e32,m1,tu,mu
; CHECK-NEXT:    vfcvt.x.f.v v16, v16, v0.t
; CHECK-NEXT:    ret
{
  %b = call <vscale x 2 x i32> @llvm.vp.fptosi.nxv2i32.nxv2f32(
               <vscale x 2 x float> %a,
               metadata !"fpexcept.ignore",
               <vscale x 2 x i1> %mask,
               i32 %gvl)
  ret <vscale x 2 x i32> %b
}

declare <vscale x 2 x i32> @llvm.vp.fptosi.nxv2i32.nxv2f32(<vscale x 2 x float> %a, metadata, <vscale x 2 x i1> %mask, i32 %gvl)

define <vscale x 2 x i64> @fptosi.i64.f32(<vscale x 2 x float> %a, i32 %gvl)
; CHECK-LABEL: fptosi.i64.f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, a0, e32,m1,tu,mu
; CHECK-NEXT:    vfwcvt.x.f.v v2, v16
; CHECK-NEXT:    vmv2r.v v16, v2
; CHECK-NEXT:    ret
{
  %b = call <vscale x 2 x i64> @llvm.vp.fptosi.nxv2i64.nxv2f32(<vscale x 2 x float> %a,
            metadata !"fpexcept.ignore",
            <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> undef, i1 true, i32 0), <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer),
            i32 %gvl)
  ret <vscale x 2 x i64> %b
}

define <vscale x 2 x i64> @fptosi.i64.f32.mask(<vscale x 2 x float> %a, <vscale x 2 x i1> %mask, i32 %gvl)
; CHECK-LABEL: fptosi.i64.f32.mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, a0, e32,m1,tu,mu
; CHECK-NEXT:    vfwcvt.x.f.v v2, v16, v0.t
; CHECK-NEXT:    vmv2r.v v16, v2
; CHECK-NEXT:    ret
{
  %b = call <vscale x 2 x i64> @llvm.vp.fptosi.nxv2i64.nxv2f32(
               <vscale x 2 x float> %a,
               metadata !"fpexcept.ignore",
               <vscale x 2 x i1> %mask,
               i32 %gvl)
  ret <vscale x 2 x i64> %b
}

declare <vscale x 2 x i64> @llvm.vp.fptosi.nxv2i64.nxv2f32(<vscale x 2 x float> %a, metadata, <vscale x 2 x i1> %mask, i32 %gvl)

define <vscale x 2 x i32> @fptosi.i32.f64(<vscale x 2 x double> %a, i32 %gvl)
; CHECK-LABEL: fptosi.i32.f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, a0, e32,m1,tu,mu
; CHECK-NEXT:    vfncvt.x.f.w v1, v16
; CHECK-NEXT:    vmv1r.v v16, v1
; CHECK-NEXT:    ret
{
  %b = call <vscale x 2 x i32> @llvm.vp.fptosi.nxv2i32.nxv2f64(<vscale x 2 x double> %a,
            metadata !"fpexcept.ignore",
            <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> undef, i1 true, i32 0), <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer),
            i32 %gvl)
  ret <vscale x 2 x i32> %b
}

define <vscale x 2 x i32> @fptosi.i32.f64.mask(<vscale x 2 x double> %a, <vscale x 2 x i1> %mask, i32 %gvl)
; CHECK-LABEL: fptosi.i32.f64.mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, a0, e32,m1,tu,mu
; CHECK-NEXT:    vfncvt.x.f.w v1, v16, v0.t
; CHECK-NEXT:    vmv1r.v v16, v1
; CHECK-NEXT:    ret
{
  %b = call <vscale x 2 x i32> @llvm.vp.fptosi.nxv2i32.nxv2f64(
               <vscale x 2 x double> %a,
               metadata !"fpexcept.ignore",
               <vscale x 2 x i1> %mask,
               i32 %gvl)
  ret <vscale x 2 x i32> %b
}

declare <vscale x 2 x i32> @llvm.vp.fptosi.nxv2i32.nxv2f64(<vscale x 2 x double> %a, metadata, <vscale x 2 x i1> %mask, i32 %gvl)
