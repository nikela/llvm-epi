; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+m,+v -verify-machineinstrs -O0 \
; RUN:    < %s -epi-pipeline | FileCheck --check-prefix=CHECK-O0 %s
; RUN: llc -mtriple=riscv64 -mattr=+m,+v -verify-machineinstrs -O2 \
; RUN:    < %s -epi-pipeline | FileCheck --check-prefix=CHECK-O2 %s

; NOTE: using volatile in order to avoid instruction selection optimizations.

@scratch = global i8 0, align 16

define void @test_vp_icmp(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, i32 %n) nounwind {
; CHECK-O0-LABEL: test_vp_icmp:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    mv a1, a0
; CHECK-O0-NEXT:    # kill: def $x10 killed $x11
; CHECK-O0-NEXT:    lui a0, %hi(scratch)
; CHECK-O0-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O0-NEXT:    slli a1, a1, 32
; CHECK-O0-NEXT:    srli a1, a1, 32
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmseq.vv v10, v8, v9
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmsne.vv v10, v8, v9
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmsltu.vv v10, v9, v8
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmsleu.vv v10, v9, v8
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmsltu.vv v10, v8, v9
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmsleu.vv v10, v8, v9
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmslt.vv v10, v9, v8
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmsle.vv v10, v9, v8
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmslt.vv v10, v8, v9
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m1, ta, ma
; CHECK-O0-NEXT:    vmsle.vv v8, v8, v9
; CHECK-O0-NEXT:    vsetvli a1, zero, e8, mf8, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_icmp:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    slli a0, a0, 32
; CHECK-O2-NEXT:    srli a0, a0, 32
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmseq.vv v10, v8, v9
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmsne.vv v10, v8, v9
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmsltu.vv v10, v9, v8
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmsleu.vv v10, v9, v8
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmsltu.vv v10, v8, v9
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmsleu.vv v10, v8, v9
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmslt.vv v10, v9, v8
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmsle.vv v10, v9, v8
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmslt.vv v10, v8, v9
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m1, ta, ma
; CHECK-O2-NEXT:    vmsle.vv v8, v8, v9
; CHECK-O2-NEXT:    vsetvli a0, zero, e8, mf8, ta, ma
; CHECK-O2-NEXT:    vsm.v v8, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 1 x i1>*

  %head = insertelement <vscale x 1 x i1> undef, i1 1, i32 0
  %allones = shufflevector <vscale x 1 x i1> %head, <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer

  %eq = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, metadata !"eq", <vscale x 1 x i1> %allones, i32 %n)
  store volatile <vscale x 1 x i1> %eq, <vscale x 1 x i1>* %store_addr

  %ne = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, metadata !"ne", <vscale x 1 x i1> %allones, i32 %n)
  store volatile <vscale x 1 x i1> %ne, <vscale x 1 x i1>* %store_addr

  %ugt = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, metadata !"ugt", <vscale x 1 x i1> %allones, i32 %n)
  store volatile <vscale x 1 x i1> %ugt, <vscale x 1 x i1>* %store_addr

  %uge = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, metadata !"uge", <vscale x 1 x i1> %allones, i32 %n)
  store volatile <vscale x 1 x i1> %uge, <vscale x 1 x i1>* %store_addr

  %ult = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, metadata !"ult", <vscale x 1 x i1> %allones, i32 %n)
  store volatile <vscale x 1 x i1> %ult, <vscale x 1 x i1>* %store_addr

  %ule = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, metadata !"ule", <vscale x 1 x i1> %allones, i32 %n)
  store volatile <vscale x 1 x i1> %ule, <vscale x 1 x i1>* %store_addr

  %sgt = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, metadata !"sgt", <vscale x 1 x i1> %allones, i32 %n)
  store volatile <vscale x 1 x i1> %sgt, <vscale x 1 x i1>* %store_addr

  %sge = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, metadata !"sge", <vscale x 1 x i1> %allones, i32 %n)
  store volatile <vscale x 1 x i1> %sge, <vscale x 1 x i1>* %store_addr

  %slt = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, metadata !"slt", <vscale x 1 x i1> %allones, i32 %n)
  store volatile <vscale x 1 x i1> %slt, <vscale x 1 x i1>* %store_addr

  %sle = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, metadata !"sle", <vscale x 1 x i1> %allones, i32 %n)
  store volatile <vscale x 1 x i1> %sle, <vscale x 1 x i1>* %store_addr

  ret void
}

define void @test_vp_icmp_2(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, i32 %n) nounwind {
; CHECK-O0-LABEL: test_vp_icmp_2:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    mv a1, a0
; CHECK-O0-NEXT:    # kill: def $x10 killed $x11
; CHECK-O0-NEXT:    lui a0, %hi(scratch)
; CHECK-O0-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O0-NEXT:    slli a1, a1, 32
; CHECK-O0-NEXT:    srli a1, a1, 32
; CHECK-O0-NEXT:    vsetvli zero, a1, e32, m1, ta, ma
; CHECK-O0-NEXT:    vmseq.vv v10, v8, v9
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e32, m1, ta, ma
; CHECK-O0-NEXT:    vmsne.vv v10, v8, v9
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e32, m1, ta, ma
; CHECK-O0-NEXT:    vmsltu.vv v10, v9, v8
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e32, m1, ta, ma
; CHECK-O0-NEXT:    vmsleu.vv v10, v9, v8
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e32, m1, ta, ma
; CHECK-O0-NEXT:    vmsltu.vv v10, v8, v9
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e32, m1, ta, ma
; CHECK-O0-NEXT:    vmsleu.vv v10, v8, v9
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e32, m1, ta, ma
; CHECK-O0-NEXT:    vmslt.vv v10, v9, v8
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e32, m1, ta, ma
; CHECK-O0-NEXT:    vmsle.vv v10, v9, v8
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e32, m1, ta, ma
; CHECK-O0-NEXT:    vmslt.vv v10, v8, v9
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v10, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e32, m1, ta, ma
; CHECK-O0-NEXT:    vmsle.vv v8, v8, v9
; CHECK-O0-NEXT:    vsetvli a1, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_icmp_2:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    slli a0, a0, 32
; CHECK-O2-NEXT:    srli a0, a0, 32
; CHECK-O2-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-O2-NEXT:    vmseq.vv v10, v8, v9
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-O2-NEXT:    vmsne.vv v10, v8, v9
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-O2-NEXT:    vmsltu.vv v10, v9, v8
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-O2-NEXT:    vmsleu.vv v10, v9, v8
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-O2-NEXT:    vmsltu.vv v10, v8, v9
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-O2-NEXT:    vmsleu.vv v10, v8, v9
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-O2-NEXT:    vmslt.vv v10, v9, v8
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-O2-NEXT:    vmsle.vv v10, v9, v8
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-O2-NEXT:    vmslt.vv v10, v8, v9
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v10, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e32, m1, ta, ma
; CHECK-O2-NEXT:    vmsle.vv v8, v8, v9
; CHECK-O2-NEXT:    vsetvli a0, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v8, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 2 x i1>*

  %head = insertelement <vscale x 2 x i1> undef, i1 1, i32 0
  %allones = shufflevector <vscale x 2 x i1> %head, <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer

  %eq = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, metadata !"eq", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %eq, <vscale x 2 x i1>* %store_addr

  %ne = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, metadata !"ne", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %ne, <vscale x 2 x i1>* %store_addr

  %ugt = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, metadata !"ugt", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %ugt, <vscale x 2 x i1>* %store_addr

  %uge = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, metadata !"uge", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %uge, <vscale x 2 x i1>* %store_addr

  %ult = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, metadata !"ult", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %ult, <vscale x 2 x i1>* %store_addr

  %ule = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, metadata !"ule", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %ule, <vscale x 2 x i1>* %store_addr

  %sgt = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, metadata !"sgt", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %sgt, <vscale x 2 x i1>* %store_addr

  %sge = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, metadata !"sge", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %sge, <vscale x 2 x i1>* %store_addr

  %slt = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, metadata !"slt", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %slt, <vscale x 2 x i1>* %store_addr

  %sle = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, metadata !"sle", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %sle, <vscale x 2 x i1>* %store_addr

  ret void
}

define void @test_vp_icmp_3(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, i32 %n) nounwind {
; CHECK-O0-LABEL: test_vp_icmp_3:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    mv a1, a0
; CHECK-O0-NEXT:    vmv2r.v v12, v10
; CHECK-O0-NEXT:    vmv2r.v v10, v8
; CHECK-O0-NEXT:    # kill: def $x10 killed $x11
; CHECK-O0-NEXT:    lui a0, %hi(scratch)
; CHECK-O0-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O0-NEXT:    slli a1, a1, 32
; CHECK-O0-NEXT:    srli a1, a1, 32
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m2, ta, ma
; CHECK-O0-NEXT:    vmseq.vv v8, v10, v12
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m2, ta, ma
; CHECK-O0-NEXT:    vmsne.vv v8, v10, v12
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m2, ta, ma
; CHECK-O0-NEXT:    vmsltu.vv v8, v12, v10
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m2, ta, ma
; CHECK-O0-NEXT:    vmsleu.vv v8, v12, v10
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m2, ta, ma
; CHECK-O0-NEXT:    vmsltu.vv v8, v10, v12
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m2, ta, ma
; CHECK-O0-NEXT:    vmsleu.vv v8, v10, v12
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m2, ta, ma
; CHECK-O0-NEXT:    vmslt.vv v8, v12, v10
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m2, ta, ma
; CHECK-O0-NEXT:    vmsle.vv v8, v12, v10
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m2, ta, ma
; CHECK-O0-NEXT:    vmslt.vv v8, v10, v12
; CHECK-O0-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    vsetvli zero, a1, e64, m2, ta, ma
; CHECK-O0-NEXT:    vmsle.vv v8, v10, v12
; CHECK-O0-NEXT:    vsetvli a1, zero, e8, mf4, ta, ma
; CHECK-O0-NEXT:    vsm.v v8, (a0)
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_icmp_3:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    slli a0, a0, 32
; CHECK-O2-NEXT:    srli a0, a0, 32
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m2, ta, ma
; CHECK-O2-NEXT:    vmseq.vv v12, v8, v10
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v12, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m2, ta, ma
; CHECK-O2-NEXT:    vmsne.vv v12, v8, v10
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v12, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m2, ta, ma
; CHECK-O2-NEXT:    vmsltu.vv v12, v10, v8
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v12, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m2, ta, ma
; CHECK-O2-NEXT:    vmsleu.vv v12, v10, v8
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v12, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m2, ta, ma
; CHECK-O2-NEXT:    vmsltu.vv v12, v8, v10
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v12, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m2, ta, ma
; CHECK-O2-NEXT:    vmsleu.vv v12, v8, v10
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v12, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m2, ta, ma
; CHECK-O2-NEXT:    vmslt.vv v12, v10, v8
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v12, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m2, ta, ma
; CHECK-O2-NEXT:    vmsle.vv v12, v10, v8
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v12, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m2, ta, ma
; CHECK-O2-NEXT:    vmslt.vv v12, v8, v10
; CHECK-O2-NEXT:    vsetvli a2, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v12, (a1)
; CHECK-O2-NEXT:    vsetvli zero, a0, e64, m2, ta, ma
; CHECK-O2-NEXT:    vmsle.vv v12, v8, v10
; CHECK-O2-NEXT:    vsetvli a0, zero, e8, mf4, ta, ma
; CHECK-O2-NEXT:    vsm.v v12, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 2 x i1>*

  %head = insertelement <vscale x 2 x i1> undef, i1 1, i32 0
  %allones = shufflevector <vscale x 2 x i1> %head, <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer

  %eq = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, metadata !"eq", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %eq, <vscale x 2 x i1>* %store_addr

  %ne = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, metadata !"ne", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %ne, <vscale x 2 x i1>* %store_addr

  %ugt = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, metadata !"ugt", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %ugt, <vscale x 2 x i1>* %store_addr

  %uge = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, metadata !"uge", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %uge, <vscale x 2 x i1>* %store_addr

  %ult = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, metadata !"ult", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %ult, <vscale x 2 x i1>* %store_addr

  %ule = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, metadata !"ule", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %ule, <vscale x 2 x i1>* %store_addr

  %sgt = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, metadata !"sgt", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %sgt, <vscale x 2 x i1>* %store_addr

  %sge = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, metadata !"sge", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %sge, <vscale x 2 x i1>* %store_addr

  %slt = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, metadata !"slt", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %slt, <vscale x 2 x i1>* %store_addr

  %sle = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, metadata !"sle", <vscale x 2 x i1> %allones, i32 %n)
  store volatile <vscale x 2 x i1> %sle, <vscale x 2 x i1>* %store_addr

  ret void
}

; store
declare void @llvm.vp.store.nxv1i1(<vscale x 1 x i1>, <vscale x 1 x i1>*, i1, <vscale x 1 x i1>, i32)
declare void @llvm.vp.store.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>*, i1, <vscale x 2 x i1>, i32)

; icmp
declare <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>, metadata, <vscale x 1 x i1>, i32)
declare <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, metadata, <vscale x 2 x i1>, i32)
declare <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, metadata, <vscale x 2 x i1>, i32)
