; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+experimental-v -verify-machineinstrs -O0 \
; RUN:    < %s | FileCheck --check-prefix=CHECK-O0 %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-v -verify-machineinstrs -O2 \
; RUN:    < %s | FileCheck --check-prefix=CHECK-O2 %s

@scratch = global i8 0, align 16

define void @test_vp_icmp(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, <vscale x 1 x i1> %m, i32 %n) nounwind {
; CHECK-O0-LABEL: test_vp_icmp:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    addi sp, sp, -48
; CHECK-O0-NEXT:    sd ra, 40(sp)
; CHECK-O0-NEXT:    sd s0, 32(sp)
; CHECK-O0-NEXT:    addi s0, sp, 48
; CHECK-O0-NEXT:    rdvlenb a1
; CHECK-O0-NEXT:    sub sp, sp, a1
; CHECK-O0-NEXT:    sd sp, -40(s0)
; CHECK-O0-NEXT:    # kill: def $x11 killed $x10
; CHECK-O0-NEXT:    lui a1, %hi(scratch)
; CHECK-O0-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vs1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmseq.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmsne.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmsleu.vv v1, v17, v16, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmsltu.vv v1, v17, v16, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmsltu.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmsleu.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmsle.vv v1, v17, v16, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmslt.vv v1, v17, v16, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmslt.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a0, a0, e64,m1
; CHECK-O0-NEXT:    vmsle.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a0, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    addi sp, s0, -48
; CHECK-O0-NEXT:    ld s0, 32(sp)
; CHECK-O0-NEXT:    ld ra, 40(sp)
; CHECK-O0-NEXT:    addi sp, sp, 48
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_icmp:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmseq.vv v1, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmsne.vv v1, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmsleu.vv v1, v17, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmsltu.vv v1, v17, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmsltu.vv v1, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmsleu.vv v1, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmsle.vv v1, v17, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmslt.vv v1, v17, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmslt.vv v1, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a0, a0, e64,m1
; CHECK-O2-NEXT:    vmsle.vv v1, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a0, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 1 x i64>*

  %eq = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, i8 32, <vscale x 1 x i1> %m, i32 %n)
  %zext_eq = zext <vscale x 1 x i1> %eq to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_eq, <vscale x 1 x i64>* %store_addr

  %ne = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, i8 33, <vscale x 1 x i1> %m, i32 %n)
  %zext_ne = zext <vscale x 1 x i1> %ne to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_ne, <vscale x 1 x i64>* %store_addr

  %ugt = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, i8 34, <vscale x 1 x i1> %m, i32 %n)
  %zext_ugt = zext <vscale x 1 x i1> %ugt to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_ugt, <vscale x 1 x i64>* %store_addr

  %uge = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, i8 35, <vscale x 1 x i1> %m, i32 %n)
  %zext_uge = zext <vscale x 1 x i1> %uge to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_uge, <vscale x 1 x i64>* %store_addr

  %ult = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, i8 36, <vscale x 1 x i1> %m, i32 %n)
  %zext_ult = zext <vscale x 1 x i1> %ult to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_ult, <vscale x 1 x i64>* %store_addr

  %ule = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, i8 37, <vscale x 1 x i1> %m, i32 %n)
  %zext_ule = zext <vscale x 1 x i1> %ule to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_ule, <vscale x 1 x i64>* %store_addr

  %sgt = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, i8 38, <vscale x 1 x i1> %m, i32 %n)
  %zext_sgt = zext <vscale x 1 x i1> %sgt to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_sgt, <vscale x 1 x i64>* %store_addr

  %sge = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, i8 39, <vscale x 1 x i1> %m, i32 %n)
  %zext_sge = zext <vscale x 1 x i1> %sge to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_sge, <vscale x 1 x i64>* %store_addr

  %slt = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, i8 40, <vscale x 1 x i1> %m, i32 %n)
  %zext_slt = zext <vscale x 1 x i1> %slt to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_slt, <vscale x 1 x i64>* %store_addr

  %sle = call <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b, i8 41, <vscale x 1 x i1> %m, i32 %n)
  %zext_sle = zext <vscale x 1 x i1> %sle to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_sle, <vscale x 1 x i64>* %store_addr

  ret void
}

define void @test_vp_icmp_2(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, <vscale x 2 x i1> %m, i32 %n) nounwind {
; CHECK-O0-LABEL: test_vp_icmp_2:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    addi sp, sp, -48
; CHECK-O0-NEXT:    sd ra, 40(sp)
; CHECK-O0-NEXT:    sd s0, 32(sp)
; CHECK-O0-NEXT:    addi s0, sp, 48
; CHECK-O0-NEXT:    rdvlenb a1
; CHECK-O0-NEXT:    sub sp, sp, a1
; CHECK-O0-NEXT:    sd sp, -40(s0)
; CHECK-O0-NEXT:    # kill: def $x11 killed $x10
; CHECK-O0-NEXT:    lui a1, %hi(scratch)
; CHECK-O0-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vs1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmseq.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmsne.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmsleu.vv v1, v17, v16, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmsltu.vv v1, v17, v16, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmsltu.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmsleu.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmsle.vv v1, v17, v16, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmslt.vv v1, v17, v16, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmslt.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a0, a0, e32,m1
; CHECK-O0-NEXT:    vmsle.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a0, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    addi sp, s0, -48
; CHECK-O0-NEXT:    ld s0, 32(sp)
; CHECK-O0-NEXT:    ld ra, 40(sp)
; CHECK-O0-NEXT:    addi sp, sp, 48
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_icmp_2:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmseq.vv v1, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmsne.vv v1, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmsleu.vv v1, v17, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmsltu.vv v1, v17, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmsltu.vv v1, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmsleu.vv v1, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmsle.vv v1, v17, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmslt.vv v1, v17, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmslt.vv v1, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    vsetvli a0, a0, e32,m1
; CHECK-O2-NEXT:    vmsle.vv v1, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a0, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 2 x i32>*

  %eq = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, i8 32, <vscale x 2 x i1> %m, i32 %n)
  %zext_eq = zext <vscale x 2 x i1> %eq to <vscale x 2 x i32>
  store <vscale x 2 x i32> %zext_eq, <vscale x 2 x i32>* %store_addr

  %ne = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, i8 33, <vscale x 2 x i1> %m, i32 %n)
  %zext_ne = zext <vscale x 2 x i1> %ne to <vscale x 2 x i32>
  store <vscale x 2 x i32> %zext_ne, <vscale x 2 x i32>* %store_addr

  %ugt = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, i8 34, <vscale x 2 x i1> %m, i32 %n)
  %zext_ugt = zext <vscale x 2 x i1> %ugt to <vscale x 2 x i32>
  store <vscale x 2 x i32> %zext_ugt, <vscale x 2 x i32>* %store_addr

  %uge = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, i8 35, <vscale x 2 x i1> %m, i32 %n)
  %zext_uge = zext <vscale x 2 x i1> %uge to <vscale x 2 x i32>
  store <vscale x 2 x i32> %zext_uge, <vscale x 2 x i32>* %store_addr

  %ult = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, i8 36, <vscale x 2 x i1> %m, i32 %n)
  %zext_ult = zext <vscale x 2 x i1> %ult to <vscale x 2 x i32>
  store <vscale x 2 x i32> %zext_ult, <vscale x 2 x i32>* %store_addr

  %ule = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, i8 37, <vscale x 2 x i1> %m, i32 %n)
  %zext_ule = zext <vscale x 2 x i1> %ule to <vscale x 2 x i32>
  store <vscale x 2 x i32> %zext_ule, <vscale x 2 x i32>* %store_addr

  %sgt = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, i8 38, <vscale x 2 x i1> %m, i32 %n)
  %zext_sgt = zext <vscale x 2 x i1> %sgt to <vscale x 2 x i32>
  store <vscale x 2 x i32> %zext_sgt, <vscale x 2 x i32>* %store_addr

  %sge = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, i8 39, <vscale x 2 x i1> %m, i32 %n)
  %zext_sge = zext <vscale x 2 x i1> %sge to <vscale x 2 x i32>
  store <vscale x 2 x i32> %zext_sge, <vscale x 2 x i32>* %store_addr

  %slt = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, i8 40, <vscale x 2 x i1> %m, i32 %n)
  %zext_slt = zext <vscale x 2 x i1> %slt to <vscale x 2 x i32>
  store <vscale x 2 x i32> %zext_slt, <vscale x 2 x i32>* %store_addr

  %sle = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b, i8 41, <vscale x 2 x i1> %m, i32 %n)
  %zext_sle = zext <vscale x 2 x i1> %sle to <vscale x 2 x i32>
  store <vscale x 2 x i32> %zext_sle, <vscale x 2 x i32>* %store_addr

  ret void
}

define void @test_vp_icmp_3(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i1> %m, i32 %n) nounwind {
; CHECK-O0-LABEL: test_vp_icmp_3:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    addi sp, sp, -48
; CHECK-O0-NEXT:    sd ra, 40(sp)
; CHECK-O0-NEXT:    sd s0, 32(sp)
; CHECK-O0-NEXT:    addi s0, sp, 48
; CHECK-O0-NEXT:    rdvlenb a1
; CHECK-O0-NEXT:    sub sp, sp, a1
; CHECK-O0-NEXT:    sd sp, -40(s0)
; CHECK-O0-NEXT:    # kill: def $x11 killed $x10
; CHECK-O0-NEXT:    lui a1, %hi(scratch)
; CHECK-O0-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vs1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmseq.vv v1, v16, v18, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmv.v.i v2, 0
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmsne.vv v1, v16, v18, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmsleu.vv v1, v18, v16, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmsltu.vv v1, v18, v16, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmsltu.vv v1, v16, v18, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmsleu.vv v1, v16, v18, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmsle.vv v1, v18, v16, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmslt.vv v1, v18, v16, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmslt.vv v1, v16, v18, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    vsetvli a0, a0, e64,m2
; CHECK-O0-NEXT:    vmsle.vv v1, v16, v18, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a0, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    addi sp, s0, -48
; CHECK-O0-NEXT:    ld s0, 32(sp)
; CHECK-O0-NEXT:    ld ra, 40(sp)
; CHECK-O0-NEXT:    addi sp, sp, 48
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_icmp_3:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    vmv1r.v v1, v0
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmseq.vv v2, v16, v18, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmv.v.i v4, 0
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmsne.vv v2, v16, v18, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmsleu.vv v2, v18, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmsltu.vv v2, v18, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmsltu.vv v2, v16, v18, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmsleu.vv v2, v16, v18, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmsle.vv v2, v18, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmslt.vv v2, v18, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmslt.vv v2, v16, v18, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a0, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmsle.vv v1, v16, v18, v0.t
; CHECK-O2-NEXT:    vsetvli a0, zero, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 2 x i64>*

  %eq = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, i8 32, <vscale x 2 x i1> %m, i32 %n)
  %zext_eq = zext <vscale x 2 x i1> %eq to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_eq, <vscale x 2 x i64>* %store_addr

  %ne = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, i8 33, <vscale x 2 x i1> %m, i32 %n)
  %zext_ne = zext <vscale x 2 x i1> %ne to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_ne, <vscale x 2 x i64>* %store_addr

  %ugt = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, i8 34, <vscale x 2 x i1> %m, i32 %n)
  %zext_ugt = zext <vscale x 2 x i1> %ugt to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_ugt, <vscale x 2 x i64>* %store_addr

  %uge = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, i8 35, <vscale x 2 x i1> %m, i32 %n)
  %zext_uge = zext <vscale x 2 x i1> %uge to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_uge, <vscale x 2 x i64>* %store_addr

  %ult = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, i8 36, <vscale x 2 x i1> %m, i32 %n)
  %zext_ult = zext <vscale x 2 x i1> %ult to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_ult, <vscale x 2 x i64>* %store_addr

  %ule = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, i8 37, <vscale x 2 x i1> %m, i32 %n)
  %zext_ule = zext <vscale x 2 x i1> %ule to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_ule, <vscale x 2 x i64>* %store_addr

  %sgt = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, i8 38, <vscale x 2 x i1> %m, i32 %n)
  %zext_sgt = zext <vscale x 2 x i1> %sgt to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_sgt, <vscale x 2 x i64>* %store_addr

  %sge = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, i8 39, <vscale x 2 x i1> %m, i32 %n)
  %zext_sge = zext <vscale x 2 x i1> %sge to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_sge, <vscale x 2 x i64>* %store_addr

  %slt = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, i8 40, <vscale x 2 x i1> %m, i32 %n)
  %zext_slt = zext <vscale x 2 x i1> %slt to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_slt, <vscale x 2 x i64>* %store_addr

  %sle = call <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b, i8 41, <vscale x 2 x i1> %m, i32 %n)
  %zext_sle = zext <vscale x 2 x i1> %sle to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_sle, <vscale x 2 x i64>* %store_addr

  ret void
}

; store
declare void @llvm.vp.store.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>*, i32, <vscale x 1 x i1>, i32)
declare void @llvm.vp.store.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>*, i32, <vscale x 2 x i1>, i32)
declare void @llvm.vp.store.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>*, i32, <vscale x 2 x i1>, i32)

; icmp
declare <vscale x 1 x i1> @llvm.vp.icmp.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>, i8 immarg, <vscale x 1 x i1>, i32)
declare <vscale x 2 x i1> @llvm.vp.icmp.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, i8 immarg, <vscale x 2 x i1>, i32)
declare <vscale x 2 x i1> @llvm.vp.icmp.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, i8 immarg, <vscale x 2 x i1>, i32)
