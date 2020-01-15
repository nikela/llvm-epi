; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+m,+f,+d,+a,+c,+v -verify-machineinstrs < %s \
; RUN:    | FileCheck %s

declare void @llvm.epi.vstore.v2f32(<vscale x 2 x float>, <vscale x 2 x float>* nocapture, i64)
declare <vscale x 2 x i32> @llvm.epi.vload.v2i32(<vscale x 2 x i32>* nocapture, i64)
declare void @llvm.epi.vstore.v1f64(<vscale x 1 x double>, <vscale x 1 x double>* nocapture, i64)

declare void @llvm.epi.vstore.v4f32(<vscale x 4 x float>, <vscale x 4 x float>* nocapture, i64)
declare <vscale x 4 x i32> @llvm.epi.vload.v4i32(<vscale x 4 x i32>* nocapture, i64)

define void @foo_1(i64 %gvl, <vscale x 2 x i32>* %src, <vscale x 1 x double>* %dst) nounwind {
; CHECK-LABEL: foo_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a3, a0, e32,m1
; CHECK-NEXT:    vle.v v0, (a1)
; CHECK-NEXT:    vsetvli a0, a0, e64,m1
; CHECK-NEXT:    vse.v v0, (a2)
; CHECK-NEXT:    ret
  %a = call <vscale x 2 x i32> @llvm.epi.vload.v2i32(<vscale x 2 x i32> *%src, i64 %gvl)
  %b = bitcast <vscale x 2 x i32> %a to <vscale x 1 x double>
  call void @llvm.epi.vstore.v1f64(<vscale x 1 x double> %b, <vscale x 1 x double>* %dst, i64 %gvl)
  ret void
}


define void @foo_2(i64 %gvl, <vscale x 2 x i32>* %src, <vscale x 2 x float>* %dst) nounwind {
; CHECK-LABEL: foo_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, a0, e32,m1
; CHECK-NEXT:    vle.v v0, (a1)
; CHECK-NEXT:    vse.v v0, (a2)
; CHECK-NEXT:    ret
  %a = call <vscale x 2 x i32> @llvm.epi.vload.v2i32(<vscale x 2 x i32> *%src, i64 %gvl)
  %b = bitcast <vscale x 2 x i32> %a to <vscale x 2 x float>
  call void @llvm.epi.vstore.v2f32(<vscale x 2 x float> %b, <vscale x 2 x float>* %dst, i64 %gvl)
  ret void
}

define void @foo_3(i64 %gvl, <vscale x 4 x i32>* %src, <vscale x 4 x float>* %dst) nounwind {
; CHECK-LABEL: foo_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, a0, e32,m2
; CHECK-NEXT:    vle.v v0, (a1)
; CHECK-NEXT:    vse.v v0, (a2)
; CHECK-NEXT:    ret
  %a = call <vscale x 4 x i32> @llvm.epi.vload.v4i32(<vscale x 4 x i32> *%src, i64 %gvl)
  %b = bitcast <vscale x 4 x i32> %a to <vscale x 4 x float>
  call void @llvm.epi.vstore.v4f32(<vscale x 4 x float> %b, <vscale x 4 x float>* %dst, i64 %gvl)
  ret void
}

; Use LLVM IR load/store
; Note that LLVM has updated the store to be of the same type as the load
; one. This is correct though a bit surprising.
define void @foo_5(<vscale x 2 x i32>* %src, <vscale x 1 x double>* %dst) nounwind {
; CHECK-LABEL: foo_5:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a2, zero, e32,m1
; CHECK-NEXT:    vle.v v0, (a0)
; CHECK-NEXT:    vse.v v0, (a1)
; CHECK-NEXT:    ret
  %a = load <vscale x 2 x i32>, <vscale x 2 x i32> *%src
  %b = bitcast <vscale x 2 x i32> %a to <vscale x 1 x double>
  store <vscale x 1 x double> %b, <vscale x 1 x double> *%dst
  ret void
}
