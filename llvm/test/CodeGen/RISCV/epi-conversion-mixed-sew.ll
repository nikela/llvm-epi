; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+v,+a,+f,+d,+c,+m -o - %s \
; RUN:    -epi-pipeline | FileCheck %s

; Note: This test uses the vector calling convention which is subject to change.

; Widening float->uint
declare <vscale x 2 x i64> @llvm.epi.vfwcvt.xu.f.nxv2i64.nxv2f32(<vscale x 2 x float>, i64);

define <vscale x 2 x i64> @test_widen_float_to_uint(<vscale x 2 x float> %parm0, i64 %gvl) nounwind {
; CHECK-LABEL: test_widen_float_to_uint:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfwcvt.xu.f.v v10, v8
; CHECK-NEXT:    vmv2r.v v8, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x i64> @llvm.epi.vfwcvt.xu.f.nxv2i64.nxv2f32(
    <vscale x 2 x float> %parm0,
    i64 %gvl)
  ret <vscale x 2 x i64> %a
}

; Widening float->int
declare <vscale x 2 x i64> @llvm.epi.vfwcvt.x.f.nxv2i64.nxv2f32(<vscale x 2 x float>, i64);

define <vscale x 2 x i64> @test_widen_float_to_int(<vscale x 2 x float> %parm0, i64 %gvl) nounwind {
; CHECK-LABEL: test_widen_float_to_int:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfwcvt.x.f.v v10, v8
; CHECK-NEXT:    vmv2r.v v8, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x i64> @llvm.epi.vfwcvt.x.f.nxv2i64.nxv2f32(
    <vscale x 2 x float> %parm0,
    i64 %gvl)
  ret <vscale x 2 x i64> %a
}

; Widening uint->float
declare <vscale x 2 x double> @llvm.epi.vfwcvt.f.xu.nxv2f64.nxv2i32( <vscale x 2 x i32>, i64);

define <vscale x 2 x double> @test_widen_uint_to_float(<vscale x 2 x i32> %parm0, i64 %gvl) nounwind {
; CHECK-LABEL: test_widen_uint_to_float:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfwcvt.f.xu.v v10, v8
; CHECK-NEXT:    vmv2r.v v8, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x double> @llvm.epi.vfwcvt.f.xu.nxv2f64.nxv2i32(
    <vscale x 2 x i32> %parm0,
    i64 %gvl)

  ret <vscale x 2 x double> %a
}

; Widening int->float
declare <vscale x 2 x double> @llvm.epi.vfwcvt.f.x.nxv2f64.nxv2i32( <vscale x 2 x i32>, i64);

define <vscale x 2 x double> @test_widen_int_to_float(<vscale x 2 x i32> %parm0, i64 %gvl) nounwind {
; CHECK-LABEL: test_widen_int_to_float:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfwcvt.f.x.v v10, v8
; CHECK-NEXT:    vmv2r.v v8, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x double> @llvm.epi.vfwcvt.f.x.nxv2f64.nxv2i32(
    <vscale x 2 x i32> %parm0,
    i64 %gvl)

  ret <vscale x 2 x double> %a
}

; Widening float->float
declare <vscale x 2 x double> @llvm.epi.vfwcvt.f.f.nxv2f64.nxv2f32( <vscale x 2 x float>, i64);

define <vscale x 2 x double> @test_widen_float_to_float(<vscale x 2 x float> %parm0, i64 %gvl) nounwind {
; CHECK-LABEL: test_widen_float_to_float:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfwcvt.f.f.v v10, v8
; CHECK-NEXT:    vmv2r.v v8, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x double> @llvm.epi.vfwcvt.f.f.nxv2f64.nxv2f32(
    <vscale x 2 x float> %parm0,
    i64 %gvl)

  ret <vscale x 2 x double> %a
}

; Widening int->uint
declare <vscale x 2 x i64> @llvm.epi.vwcvtu.x.x.nxv2i64.nxv2i32( <vscale x 2 x i32>, i64);

define <vscale x 2 x i64> @test_widen_int_to_uint(<vscale x 2 x i32> %parm0, i64 %gvl) nounwind {
; CHECK-LABEL: test_widen_int_to_uint:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vwcvtu.x.x.v v10, v8
; CHECK-NEXT:    vmv2r.v v8, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x i64> @llvm.epi.vwcvtu.x.x.nxv2i64.nxv2i32(
    <vscale x 2 x i32> %parm0,
    i64 %gvl)

  ret <vscale x 2 x i64> %a
}

; Widening int->int
declare <vscale x 2 x i64> @llvm.epi.vwcvt.x.x.nxv2i64.nxv2i32( <vscale x 2 x i32>, i64);

define <vscale x 2 x i64> @test_widen_int_to_int(<vscale x 2 x i32> %parm0, i64 %gvl) nounwind {
; CHECK-LABEL: test_widen_int_to_int:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vwcvt.x.x.v v10, v8
; CHECK-NEXT:    vmv2r.v v8, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x i64> @llvm.epi.vwcvt.x.x.nxv2i64.nxv2i32(
    <vscale x 2 x i32> %parm0,
    i64 %gvl)

  ret <vscale x 2 x i64> %a
}

; Narrowing float->uint
declare <vscale x 2 x i32> @llvm.epi.vfncvt.xu.f.nxv2i32.nxv2f64(<vscale x 2 x double>, i64);

define <vscale x 2 x i32> @test_narrow_float_to_uint(<vscale x 2 x double> %parm0, i64 %gvl) nounwind {
; CHECK-LABEL: test_narrow_float_to_uint:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfncvt.xu.f.w v10, v8
; CHECK-NEXT:    vmv1r.v v8, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x i32> @llvm.epi.vfncvt.xu.f.nxv2i32.nxv2f64(
    <vscale x 2 x double> %parm0,
    i64 %gvl)
  ret <vscale x 2 x i32> %a
}

; Narrowing float->int
declare <vscale x 2 x i32> @llvm.epi.vfncvt.x.f.nxv2i32.nxv2f64(<vscale x 2 x double>, i64);

define <vscale x 2 x i32> @test_narrow_float_to_int(<vscale x 2 x double> %parm0, i64 %gvl) nounwind {
; CHECK-LABEL: test_narrow_float_to_int:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfncvt.x.f.w v10, v8
; CHECK-NEXT:    vmv1r.v v8, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x i32> @llvm.epi.vfncvt.x.f.nxv2i32.nxv2f64(
    <vscale x 2 x double> %parm0,
    i64 %gvl)
  ret <vscale x 2 x i32> %a
}

; Narrowing uint->float
declare <vscale x 2 x float> @llvm.epi.vfncvt.f.xu.nxv2f32.nxv2i64( <vscale x 2 x i64>, i64);

define <vscale x 2 x float> @test_narrow_uint_to_float(<vscale x 2 x i64> %parm0, i64 %gvl) nounwind {
; CHECK-LABEL: test_narrow_uint_to_float:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfncvt.f.xu.w v10, v8
; CHECK-NEXT:    vmv1r.v v8, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x float> @llvm.epi.vfncvt.f.xu.nxv2f32.nxv2i64(
    <vscale x 2 x i64> %parm0,
    i64 %gvl)

  ret <vscale x 2 x float> %a
}

; Narrowing int->float
declare <vscale x 2 x float> @llvm.epi.vfncvt.f.x.nxv2f32.nxv2i64( <vscale x 2 x i64>, i64);

define <vscale x 2 x float> @test_narrow_int_to_float(<vscale x 2 x i64> %parm0, i64 %gvl) nounwind {
; CHECK-LABEL: test_narrow_int_to_float:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfncvt.f.x.w v10, v8
; CHECK-NEXT:    vmv1r.v v8, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x float> @llvm.epi.vfncvt.f.x.nxv2f32.nxv2i64(
    <vscale x 2 x i64> %parm0,
    i64 %gvl)

  ret <vscale x 2 x float> %a
}

; Narrowing float->float
declare <vscale x 2 x float> @llvm.epi.vfncvt.f.f.nxv2f32.nxv2f64( <vscale x 2 x double>, i64);

define <vscale x 2 x float> @test_narrow_float_to_float(<vscale x 2 x double> %parm0, i64 %gvl) nounwind {
; CHECK-LABEL: test_narrow_float_to_float:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vfncvt.f.f.w v10, v8
; CHECK-NEXT:    vmv1r.v v8, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x float> @llvm.epi.vfncvt.f.f.nxv2f32.nxv2f64(
    <vscale x 2 x double> %parm0,
    i64 %gvl)

  ret <vscale x 2 x float> %a
}

; Narrowing int->int
declare <vscale x 2 x i32> @llvm.epi.vncvt.x.x.nxv2i32.nxv2i64( <vscale x 2 x i64>, i64);

define <vscale x 2 x i32> @test_narrow_int_to_int(<vscale x 2 x i64> %parm0, i64 %gvl) nounwind {
; CHECK-LABEL: test_narrow_int_to_int:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, mu
; CHECK-NEXT:    vncvt.x.x.w v10, v8
; CHECK-NEXT:    vmv1r.v v8, v10
; CHECK-NEXT:    ret
entry:
  %a = call <vscale x 2 x i32> @llvm.epi.vncvt.x.x.nxv2i32.nxv2i64(
    <vscale x 2 x i64> %parm0,
    i64 %gvl)

  ret <vscale x 2 x i32> %a
}
