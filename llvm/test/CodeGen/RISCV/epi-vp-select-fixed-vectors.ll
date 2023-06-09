; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+f,+d,+v -verify-machineinstrs \
; RUN:    -riscv-v-vector-bits-min=128 < %s | FileCheck %s

@scratch = global i8 0, align 16

define void @test_vp_select_int_v4i32(<4 x i32> %a, <4 x i32> %b, <4 x i1> %m, i32 zeroext %n) nounwind {
; CHECK-LABEL: test_vp_select_int_v4i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lui a1, %hi(scratch)
; CHECK-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vmerge.vvm v8, v9, v8, v0
; CHECK-NEXT:    vsetivli zero, 4, e32, m1, ta, ma
; CHECK-NEXT:    vse32.v v8, (a1)
; CHECK-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <4 x i32>*

  %select = call <4 x i32> @llvm.vp.select.v4i32(<4 x i1> %m, <4 x i32> %a, <4 x i32> %b, i32 %n)
  store <4 x i32> %select, <4 x i32>* %store_addr

  ret void
}

define void @test_vp_select_int_v2i64(<2 x i64> %a, <2 x i64> %b, <2 x i1> %m, i32 zeroext %n) nounwind {
; CHECK-LABEL: test_vp_select_int_v2i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lui a1, %hi(scratch)
; CHECK-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-NEXT:    vmerge.vvm v8, v9, v8, v0
; CHECK-NEXT:    vsetivli zero, 2, e64, m1, ta, ma
; CHECK-NEXT:    vse64.v v8, (a1)
; CHECK-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <2 x i64>*

  %select = call <2 x i64> @llvm.vp.select.v2i64(<2 x i1> %m, <2 x i64> %a, <2 x i64> %b, i32 %n)
  store <2 x i64> %select, <2 x i64>* %store_addr

  ret void
}

define void @test_vp_select_fp_v4f32(<4 x float> %a, <4 x float> %b, <4 x i1> %m, i32 zeroext %n) nounwind {
; CHECK-LABEL: test_vp_select_fp_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lui a1, %hi(scratch)
; CHECK-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:    vmerge.vvm v8, v9, v8, v0
; CHECK-NEXT:    vsetivli zero, 4, e32, m1, ta, ma
; CHECK-NEXT:    vse32.v v8, (a1)
; CHECK-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <4 x float>*

  %select = call <4 x float> @llvm.vp.select.v4f32(<4 x i1> %m, <4 x float> %a, <4 x float> %b, i32 %n)
  store <4 x float> %select, <4 x float>* %store_addr

  ret void
}

define void @test_vp_select_fp_v2f64(<2 x double> %a, <2 x double> %b, <2 x i1> %m, i32 zeroext %n) nounwind {
; CHECK-LABEL: test_vp_select_fp_v2f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lui a1, %hi(scratch)
; CHECK-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-NEXT:    vmerge.vvm v8, v9, v8, v0
; CHECK-NEXT:    vsetivli zero, 2, e64, m1, ta, ma
; CHECK-NEXT:    vse64.v v8, (a1)
; CHECK-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <2 x double>*

  %select = call <2 x double> @llvm.vp.select.v2f64(<2 x i1> %m, <2 x double> %a, <2 x double> %b, i32 %n)
  store <2 x double> %select, <2 x double>* %store_addr

  ret void
}

; FIXME: not implemented yet
;define void @test_vp_select_mask_v2i1(<2 x i1> %a, <2 x i1> %b, <2 x i1> %m, i32 zeroext %n) nounwind {
;  %store_addr = bitcast i8* @scratch to <2 x i1>*
;
;  %select = call <2 x i1> @llvm.vp.select.v2i1(<2 x i1> %m, <2 x i1> %a, <2 x i1> %b, i32 2)
;  store <2 x i1> %select, <2 x i1>* %store_addr
;
;  ret void
;}

; store
declare void @llvm.vp.store.v4i32(<4 x i32>, <4 x i32>*, i32, <4 x i1>, i32)
declare void @llvm.vp.store.v2i64(<2 x i64>, <2 x i64>*, i32, <2 x i1>, i32)

; select
declare <4 x i32> @llvm.vp.select.v4i32(<4 x i1>, <4 x i32>, <4 x i32>, i32)
declare <2 x i64> @llvm.vp.select.v2i64(<2 x i1>, <2 x i64>, <2 x i64>, i32)

declare <4 x float> @llvm.vp.select.v4f32(<4 x i1>, <4 x float>, <4 x float>, i32)
declare <2 x double> @llvm.vp.select.v2f64(<2 x i1>, <2 x double>, <2 x double>, i32)

; FIXME: not implemented yet
;declare <2 x i1> @llvm.vp.select.v2i1(<2 x i1>, <2 x i1>, <2 x i1>, i32)
