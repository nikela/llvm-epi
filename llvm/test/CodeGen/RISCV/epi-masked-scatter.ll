; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple riscv64 -mattr=+f,+d,+v -verify-machineinstrs < %s \
; RUN:    -epi-pipeline | FileCheck %s

define void @nxv1i64(<vscale x 1 x i64> %data, i64* %ptr, <vscale x 1 x i64> %indices, <vscale x 1 x i1> %mask) {
; CHECK-LABEL: nxv1i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e64, m1, ta, ma
; CHECK-NEXT:    vsll.vi v9, v9, 3
; CHECK-NEXT:    vsoxei64.v v8, (a0), v9, v0.t
; CHECK-NEXT:    ret
  %1 = getelementptr i64, i64* %ptr, <vscale x 1 x i64> %indices
  call void @llvm.masked.scatter.nxv1i64.nxv1p0i64(<vscale x 1 x i64> %data, <vscale x 1 x i64*> %1, i32 8, <vscale x 1 x i1> %mask)
  ret void
}

; FIXME: Enable when scatter of element types <i64 is supported.

;define void @nxv2f32(<vscale x 2 x float> %data, float* %ptr, <vscale x 2 x i32> %indices, <vscale x 2 x i1> %mask) {
;  %1 = getelementptr float, float* %ptr, <vscale x 2 x i32> %indices
;  call void @llvm.masked.scatter.nxv2f32.nxv2p0f32(<vscale x 2 x float> %data, <vscale x 2 x float*> %1, i32 4, <vscale x 2 x i1> %mask)
;  ret void
;}

;define void @nxv16i8(<vscale x 16 x i8> %data, i8* %ptr, <vscale x 16 x i8> %indices, <vscale x 16 x i1> %mask) {
;  %1 = getelementptr i8, i8* %ptr, <vscale x 16 x i8> %indices
;  call void @llvm.masked.scatter.nxv16i8.nxv16p0i8(<vscale x 16 x i8> %data, <vscale x 16 x i8*> %1, i32 1, <vscale x 16 x i1> %mask)
;  ret void
;}

;define void @nxv16i16(<vscale x 16 x i16> %data, i16* %ptr, <vscale x 16 x i16> %indices, <vscale x 16 x i1> %mask) {
;  %1 = getelementptr i16, i16* %ptr, <vscale x 16 x i16> %indices
;  call void @llvm.masked.scatter.nxv16i16.nxv16p0i16(<vscale x 16 x i16> %data, <vscale x 16 x i16*> %1, i32 2, <vscale x 16 x i1> %mask)
;  ret void
;}

define void @nxv8f64(<vscale x 8 x double> %data, double* %ptr, <vscale x 8 x i64> %indices, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: nxv8f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e64, m8, ta, ma
; CHECK-NEXT:    vsll.vi v16, v16, 3
; CHECK-NEXT:    vsoxei64.v v8, (a0), v16, v0.t
; CHECK-NEXT:    ret
  %1 = getelementptr double, double* %ptr, <vscale x 8 x i64> %indices
  call void @llvm.masked.scatter.nxv8f64.nxv8p0f64(<vscale x 8 x double> %data, <vscale x 8 x double*> %1, i32 8, <vscale x 8 x i1> %mask)
  ret void
}

define void @nxv2i32(<vscale x 2 x i32> %data, i32* %ptr, <vscale x 2 x i64> %indices, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: nxv2i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e64, m2, ta, ma
; CHECK-NEXT:    vsll.vi v10, v10, 2
; CHECK-NEXT:    vsetvli zero, zero, e32, m1, ta, ma
; CHECK-NEXT:    vsoxei64.v v8, (a0), v10, v0.t
; CHECK-NEXT:    ret
  %1 = getelementptr i32, i32* %ptr, <vscale x 2 x i64> %indices
  call void @llvm.masked.scatter.nxv2i32.nxv2p0i32(<vscale x 2 x i32> %data, <vscale x 2 x i32*> %1, i32 4, <vscale x 2 x i1> %mask)
  ret void
}

define void @nxv2i32_full(<vscale x 2 x i32> %data, <vscale x 2 x i32*> %ptr, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: nxv2i32_full:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, zero, e32, m1, ta, ma
; CHECK-NEXT:    vsoxei64.v v8, (zero), v10, v0.t
; CHECK-NEXT:    ret
  call void @llvm.masked.scatter.nxv2i32.nxv2p0i32(<vscale x 2 x i32> %data, <vscale x 2 x i32*> %ptr, i32 4, <vscale x 2 x i1> %mask)
  ret void
}

; LMUL=1
declare void @llvm.masked.scatter.nxv1i64.nxv1p0i64(<vscale x 1 x i64>, <vscale x 1 x i64*>, i32, <vscale x 1 x i1>)
declare void @llvm.masked.scatter.nxv2i32.nxv2p0i32(<vscale x 2 x i32>, <vscale x 2 x i32*>, i32, <vscale x 2 x i1>)
declare void @llvm.masked.scatter.nxv2f32.nxv2p0f32(<vscale x 2 x float>, <vscale x 2 x float*>, i32, <vscale x 2 x i1>)

; LMUL>1
declare void @llvm.masked.scatter.nxv16i8.nxv16p0i8(<vscale x 16 x i8>, <vscale x 16 x i8*>, i32, <vscale x 16 x i1>)
declare void @llvm.masked.scatter.nxv16i16.nxv16p0i16(<vscale x 16 x i16>, <vscale x 16 x i16*>, i32, <vscale x 16 x i1>)
declare void @llvm.masked.scatter.nxv8f64.nxv8p0f64(<vscale x 8 x double>, <vscale x 8 x double*>, i32, <vscale x 8 x i1>)
