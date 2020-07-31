; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+experimental-v -verify-machineinstrs -O0 \
; RUN:    < %s | FileCheck --check-prefix=CHECK-O0 %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-v -verify-machineinstrs -O2 \
; RUN:    < %s | FileCheck --check-prefix=CHECK-O2 %s

@scratch = global i8 0, align 16

define void @test_vp_fcmp(<vscale x 1 x double> %a, <vscale x 1 x double> %b, <vscale x 1 x i1> %m, i32 %n) nounwind {
; CHECK-O0-LABEL: test_vp_fcmp:
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
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmfeq.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmflt.vv v1, v17, v16, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmfle.vv v1, v17, v16, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmflt.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmfle.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmfeq.vv v1, v17, v17
; CHECK-O0-NEXT:    # implicit-def: $v2
; CHECK-O0-NEXT:    vmfeq.vv v2, v16, v16
; CHECK-O0-NEXT:    vmand.mm v1, v2, v1
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    # implicit-def: $v2
; CHECK-O0-NEXT:    vmflt.vv v2, v17, v16, v0.t
; CHECK-O0-NEXT:    vmornot.mm v2, v2, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v2, (a1)
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    # implicit-def: $v2
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmfle.vv v2, v17, v16, v0.t
; CHECK-O0-NEXT:    vmornot.mm v2, v2, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v2, (a1)
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    # implicit-def: $v2
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmflt.vv v2, v16, v17, v0.t
; CHECK-O0-NEXT:    vmornot.mm v2, v2, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v2, (a1)
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    # implicit-def: $v2
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O0-NEXT:    vmfle.vv v2, v16, v17, v0.t
; CHECK-O0-NEXT:    vmornot.mm v1, v2, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a0, a0, e64,m1
; CHECK-O0-NEXT:    vmfne.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a0, zero, e8,m1
; CHECK-O0-NEXT:    vse.v v1, (a1)
; CHECK-O0-NEXT:    addi sp, s0, -48
; CHECK-O0-NEXT:    ld s0, 32(sp)
; CHECK-O0-NEXT:    ld ra, 40(sp)
; CHECK-O0-NEXT:    addi sp, sp, 48
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vp_fcmp:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    vmv1r.v v1, v0
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmfeq.vv v2, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmflt.vv v2, v17, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmfle.vv v2, v17, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmflt.vv v2, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmfle.vv v2, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmfeq.vv v2, v17, v17
; CHECK-O2-NEXT:    vmfeq.vv v3, v16, v16
; CHECK-O2-NEXT:    vmand.mm v0, v3, v2
; CHECK-O2-NEXT:    vmflt.vv v2, v17, v16, v0.t
; CHECK-O2-NEXT:    vmornot.mm v2, v2, v0
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmfle.vv v2, v17, v16, v0.t
; CHECK-O2-NEXT:    vmornot.mm v2, v2, v0
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmflt.vv v2, v16, v17, v0.t
; CHECK-O2-NEXT:    vmornot.mm v2, v2, v0
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m1
; CHECK-O2-NEXT:    vmfle.vv v2, v16, v17, v0.t
; CHECK-O2-NEXT:    vmornot.mm v2, v2, v0
; CHECK-O2-NEXT:    vsetvli a2, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a0, a0, e64,m1
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmfne.vv v1, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a0, zero, e8,m1
; CHECK-O2-NEXT:    vse.v v1, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 1 x i64>*

  ;%false = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 0, <vscale x 1 x i1> %m, i32 %n)
  ;%zext_false = zext <vscale x 1 x i1> %false to <vscale x 1 x i64>
  ;store <vscale x 1 x i64> %zext_false, <vscale x 1 x i64>* %store_addr

  %oeq = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 1, <vscale x 1 x i1> %m, i32 %n)
  %zext_oeq = zext <vscale x 1 x i1> %oeq to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_oeq, <vscale x 1 x i64>* %store_addr

  %ogt = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 2, <vscale x 1 x i1> %m, i32 %n)
  %zext_ogt = zext <vscale x 1 x i1> %ogt to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_ogt, <vscale x 1 x i64>* %store_addr

  %oge = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 3, <vscale x 1 x i1> %m, i32 %n)
  %zext_oge = zext <vscale x 1 x i1> %oge to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_oge, <vscale x 1 x i64>* %store_addr

  %olt = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 4, <vscale x 1 x i1> %m, i32 %n)
  %zext_olt = zext <vscale x 1 x i1> %olt to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_olt, <vscale x 1 x i64>* %store_addr

  %ole = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 5, <vscale x 1 x i1> %m, i32 %n)
  %zext_ole = zext <vscale x 1 x i1> %ole to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_ole, <vscale x 1 x i64>* %store_addr

  ;%one = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 6, <vscale x 1 x i1> %m, i32 %n)
  ;%zext_one = zext <vscale x 1 x i1> %one to <vscale x 1 x i64>
  ;store <vscale x 1 x i64> %zext_one, <vscale x 1 x i64>* %store_addr

  ;%ord = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 7, <vscale x 1 x i1> %m, i32 %n)
  ;%zext_ord = zext <vscale x 1 x i1> %ord to <vscale x 1 x i64>
  ;store <vscale x 1 x i64> %zext_ord, <vscale x 1 x i64>* %store_addr

  ;%uno = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 8, <vscale x 1 x i1> %m, i32 %n)
  ;%zext_uno = zext <vscale x 1 x i1> %uno to <vscale x 1 x i64>
  ;store <vscale x 1 x i64> %zext_uno, <vscale x 1 x i64>* %store_addr

  ;%ueq = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 9, <vscale x 1 x i1> %m, i32 %n)
  ;%zext_ueq = zext <vscale x 1 x i1> %ueq to <vscale x 1 x i64>
  ;store <vscale x 1 x i64> %zext_ueq, <vscale x 1 x i64>* %store_addr

  %ugt = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 10, <vscale x 1 x i1> %m, i32 %n)
  %zext_ugt = zext <vscale x 1 x i1> %ugt to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_ugt, <vscale x 1 x i64>* %store_addr

  %uge = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 11, <vscale x 1 x i1> %m, i32 %n)
  %zext_uge = zext <vscale x 1 x i1> %uge to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_uge, <vscale x 1 x i64>* %store_addr

  %ult = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 12, <vscale x 1 x i1> %m, i32 %n)
  %zext_ult = zext <vscale x 1 x i1> %ult to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_ult, <vscale x 1 x i64>* %store_addr

  %ule = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 13, <vscale x 1 x i1> %m, i32 %n)
  %zext_ule = zext <vscale x 1 x i1> %ule to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_ule, <vscale x 1 x i64>* %store_addr

  %une = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 14, <vscale x 1 x i1> %m, i32 %n)
  %zext_une = zext <vscale x 1 x i1> %une to <vscale x 1 x i64>
  store <vscale x 1 x i64> %zext_une, <vscale x 1 x i64>* %store_addr

  ;%true = call <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i8 15, <vscale x 1 x i1> %m, i32 %n)
  ;%zext_true = zext <vscale x 1 x i1> %true to <vscale x 1 x i64>
  ;store <vscale x 1 x i64> %zext_true, <vscale x 1 x i64>* %store_addr

  ret void
}

define void @test_vp_fcmp_2(<vscale x 2 x float> %a, <vscale x 2 x float> %b, <vscale x 2 x i1> %m, i32 %n) nounwind {
; CHECK-O0-LABEL: test_vp_fcmp_2:
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
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmfeq.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmv.v.i v2, 0
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmflt.vv v1, v17, v16, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmfle.vv v1, v17, v16, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmflt.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmfle.vv v1, v16, v17, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmfeq.vv v1, v17, v17
; CHECK-O0-NEXT:    # implicit-def: $v4
; CHECK-O0-NEXT:    vmfeq.vv v4, v16, v16
; CHECK-O0-NEXT:    vmand.mm v1, v4, v1
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    # implicit-def: $v4
; CHECK-O0-NEXT:    vmflt.vv v4, v17, v16, v0.t
; CHECK-O0-NEXT:    vmornot.mm v0, v4, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    # implicit-def: $v4
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmfle.vv v4, v17, v16, v0.t
; CHECK-O0-NEXT:    vmornot.mm v0, v4, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    # implicit-def: $v4
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmflt.vv v4, v16, v17, v0.t
; CHECK-O0-NEXT:    vmornot.mm v0, v4, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    # implicit-def: $v4
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmfle.vv v4, v16, v17, v0.t
; CHECK-O0-NEXT:    vmornot.mm v0, v4, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a0, a0, e32,m1
; CHECK-O0-NEXT:    vmfne.vv v1, v16, v17, v0.t
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
; CHECK-O2-LABEL: test_vp_fcmp_2:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    vmv1r.v v1, v0
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmfeq.vv v0, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmv.v.i v4, 0
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmflt.vv v0, v17, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmfle.vv v0, v17, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmflt.vv v0, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmfle.vv v0, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmfeq.vv v2, v17, v17
; CHECK-O2-NEXT:    vmfeq.vv v3, v16, v16
; CHECK-O2-NEXT:    vmand.mm v2, v3, v2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmflt.vv v3, v17, v16, v0.t
; CHECK-O2-NEXT:    vmornot.mm v0, v3, v2
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmerge.vim v6, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v6, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmfle.vv v3, v17, v16, v0.t
; CHECK-O2-NEXT:    vmornot.mm v0, v3, v2
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmerge.vim v6, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v6, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmflt.vv v3, v16, v17, v0.t
; CHECK-O2-NEXT:    vmornot.mm v0, v3, v2
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmerge.vim v6, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v6, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmfle.vv v3, v16, v17, v0.t
; CHECK-O2-NEXT:    vmornot.mm v0, v3, v2
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a0, a0, e32,m1
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmfne.vv v0, v16, v17, v0.t
; CHECK-O2-NEXT:    vsetvli a0, zero, e64,m2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 2 x i64>*

  ;%false = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 0, <vscale x 2 x i1> %m, i32 %n)
  ;%zext_false = zext <vscale x 2 x i1> %false to <vscale x 2 x i64>
  ;store <vscale x 2 x i64> %zext_false, <vscale x 2 x i64>* %store_addr

  %oeq = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 1, <vscale x 2 x i1> %m, i32 %n)
  %zext_oeq = zext <vscale x 2 x i1> %oeq to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_oeq, <vscale x 2 x i64>* %store_addr

  %ogt = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 2, <vscale x 2 x i1> %m, i32 %n)
  %zext_ogt = zext <vscale x 2 x i1> %ogt to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_ogt, <vscale x 2 x i64>* %store_addr

  %oge = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 3, <vscale x 2 x i1> %m, i32 %n)
  %zext_oge = zext <vscale x 2 x i1> %oge to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_oge, <vscale x 2 x i64>* %store_addr

  %olt = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 4, <vscale x 2 x i1> %m, i32 %n)
  %zext_olt = zext <vscale x 2 x i1> %olt to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_olt, <vscale x 2 x i64>* %store_addr

  %ole = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 5, <vscale x 2 x i1> %m, i32 %n)
  %zext_ole = zext <vscale x 2 x i1> %ole to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_ole, <vscale x 2 x i64>* %store_addr

  ;%one = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 6, <vscale x 2 x i1> %m, i32 %n)
  ;%zext_one = zext <vscale x 2 x i1> %one to <vscale x 2 x i64>
  ;store <vscale x 2 x i64> %zext_one, <vscale x 2 x i64>* %store_addr

  ;%ord = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 7, <vscale x 2 x i1> %m, i32 %n)
  ;%zext_ord = zext <vscale x 2 x i1> %ord to <vscale x 2 x i64>
  ;store <vscale x 2 x i64> %zext_ord, <vscale x 2 x i64>* %store_addr

  ;%uno = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 8, <vscale x 2 x i1> %m, i32 %n)
  ;%zext_uno = zext <vscale x 2 x i1> %uno to <vscale x 2 x i64>
  ;store <vscale x 2 x i64> %zext_uno, <vscale x 2 x i64>* %store_addr

  ;%ueq = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 9, <vscale x 2 x i1> %m, i32 %n)
  ;%zext_ueq = zext <vscale x 2 x i1> %ueq to <vscale x 2 x i64>
  ;store <vscale x 2 x i64> %zext_ueq, <vscale x 2 x i64>* %store_addr

  %ugt = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 10, <vscale x 2 x i1> %m, i32 %n)
  %zext_ugt = zext <vscale x 2 x i1> %ugt to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_ugt, <vscale x 2 x i64>* %store_addr

  %uge = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 11, <vscale x 2 x i1> %m, i32 %n)
  %zext_uge = zext <vscale x 2 x i1> %uge to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_uge, <vscale x 2 x i64>* %store_addr

  %ult = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 12, <vscale x 2 x i1> %m, i32 %n)
  %zext_ult = zext <vscale x 2 x i1> %ult to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_ult, <vscale x 2 x i64>* %store_addr

  %ule = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 13, <vscale x 2 x i1> %m, i32 %n)
  %zext_ule = zext <vscale x 2 x i1> %ule to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_ule, <vscale x 2 x i64>* %store_addr

  %une = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 14, <vscale x 2 x i1> %m, i32 %n)
  %zext_une = zext <vscale x 2 x i1> %une to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_une, <vscale x 2 x i64>* %store_addr

  ;%true = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float> %a, <vscale x 2 x float> %b, i8 15, <vscale x 2 x i1> %m, i32 %n)
  ;%zext_true = zext <vscale x 2 x i1> %true to <vscale x 2 x i64>
  ;store <vscale x 2 x i64> %zext_true, <vscale x 2 x i64>* %store_addr

  ret void
}

define void @test_vp_fcmp_3(<vscale x 2 x double> %a, <vscale x 2 x double> %b, <vscale x 2 x i1> %m, i32 %n) nounwind {
; CHECK-O0-LABEL: test_vp_fcmp_3:
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
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmfeq.vv v1, v16, v18, v0.t
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmv.v.i v2, 0
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmflt.vv v1, v18, v16, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmfle.vv v1, v18, v16, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmflt.vv v1, v16, v18, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmfle.vv v1, v16, v18, v0.t
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmfeq.vv v1, v18, v18
; CHECK-O0-NEXT:    # implicit-def: $v4
; CHECK-O0-NEXT:    vmfeq.vv v4, v16, v16
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmand.mm v1, v4, v1
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    # implicit-def: $v4
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmflt.vv v4, v18, v16, v0.t
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmornot.mm v0, v4, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    # implicit-def: $v4
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmfle.vv v4, v18, v16, v0.t
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmornot.mm v0, v4, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    # implicit-def: $v4
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmflt.vv v4, v16, v18, v0.t
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmornot.mm v0, v4, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    vmv1r.v v0, v1
; CHECK-O0-NEXT:    # implicit-def: $v4
; CHECK-O0-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O0-NEXT:    vmfle.vv v4, v16, v18, v0.t
; CHECK-O0-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O0-NEXT:    vmornot.mm v0, v4, v1
; CHECK-O0-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O0-NEXT:    vmerge.vim v4, v2, 1, v0
; CHECK-O0-NEXT:    vse.v v4, (a1)
; CHECK-O0-NEXT:    ld a2, -40(s0)
; CHECK-O0-NEXT:    vl1r.v v0, (a2)
; CHECK-O0-NEXT:    # implicit-def: $v1
; CHECK-O0-NEXT:    vsetvli a0, a0, e64,m2
; CHECK-O0-NEXT:    vmfne.vv v1, v16, v18, v0.t
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
; CHECK-O2-LABEL: test_vp_fcmp_3:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    vmv1r.v v1, v0
; CHECK-O2-NEXT:    lui a1, %hi(scratch)
; CHECK-O2-NEXT:    addi a1, a1, %lo(scratch)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmfeq.vv v2, v16, v18, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmv.v.i v4, 0
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmflt.vv v2, v18, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmfle.vv v2, v18, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmflt.vv v2, v16, v18, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmfle.vv v2, v16, v18, v0.t
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmfeq.vv v2, v18, v18
; CHECK-O2-NEXT:    vmfeq.vv v3, v16, v16
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmand.mm v2, v3, v2
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmflt.vv v3, v18, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmornot.mm v0, v3, v2
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmerge.vim v6, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v6, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmfle.vv v3, v18, v16, v0.t
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmornot.mm v0, v3, v2
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmerge.vim v6, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v6, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmflt.vv v3, v16, v18, v0.t
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmornot.mm v0, v3, v2
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmerge.vim v6, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v6, (a1)
; CHECK-O2-NEXT:    vsetvli a2, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v2
; CHECK-O2-NEXT:    vmfle.vv v3, v16, v18, v0.t
; CHECK-O2-NEXT:    vsetvli a2, a0, e32,m1
; CHECK-O2-NEXT:    vmornot.mm v0, v3, v2
; CHECK-O2-NEXT:    vsetvli a2, zero, e64,m2
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    vsetvli a0, a0, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmfne.vv v1, v16, v18, v0.t
; CHECK-O2-NEXT:    vsetvli a0, zero, e64,m2
; CHECK-O2-NEXT:    vmv1r.v v0, v1
; CHECK-O2-NEXT:    vmerge.vim v2, v4, 1, v0
; CHECK-O2-NEXT:    vse.v v2, (a1)
; CHECK-O2-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 2 x i64>*

  ;%false = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 0, <vscale x 2 x i1> %m, i32 %n)
  ;%zext_false = zext <vscale x 2 x i1> %false to <vscale x 2 x i64>
  ;store <vscale x 2 x i64> %zext_false, <vscale x 2 x i64>* %store_addr

  %oeq = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 1, <vscale x 2 x i1> %m, i32 %n)
  %zext_oeq = zext <vscale x 2 x i1> %oeq to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_oeq, <vscale x 2 x i64>* %store_addr

  %ogt = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 2, <vscale x 2 x i1> %m, i32 %n)
  %zext_ogt = zext <vscale x 2 x i1> %ogt to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_ogt, <vscale x 2 x i64>* %store_addr

  %oge = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 3, <vscale x 2 x i1> %m, i32 %n)
  %zext_oge = zext <vscale x 2 x i1> %oge to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_oge, <vscale x 2 x i64>* %store_addr

  %olt = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 4, <vscale x 2 x i1> %m, i32 %n)
  %zext_olt = zext <vscale x 2 x i1> %olt to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_olt, <vscale x 2 x i64>* %store_addr

  %ole = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 5, <vscale x 2 x i1> %m, i32 %n)
  %zext_ole = zext <vscale x 2 x i1> %ole to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_ole, <vscale x 2 x i64>* %store_addr

  ;%one = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 6, <vscale x 2 x i1> %m, i32 %n)
  ;%zext_one = zext <vscale x 2 x i1> %one to <vscale x 2 x i64>
  ;store <vscale x 2 x i64> %zext_one, <vscale x 2 x i64>* %store_addr

  ;%ord = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 7, <vscale x 2 x i1> %m, i32 %n)
  ;%zext_ord = zext <vscale x 2 x i1> %ord to <vscale x 2 x i64>
  ;store <vscale x 2 x i64> %zext_ord, <vscale x 2 x i64>* %store_addr

  ;%uno = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 8, <vscale x 2 x i1> %m, i32 %n)
  ;%zext_uno = zext <vscale x 2 x i1> %uno to <vscale x 2 x i64>
  ;store <vscale x 2 x i64> %zext_uno, <vscale x 2 x i64>* %store_addr

  ;%ueq = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 9, <vscale x 2 x i1> %m, i32 %n)
  ;%zext_ueq = zext <vscale x 2 x i1> %ueq to <vscale x 2 x i64>
  ;store <vscale x 2 x i64> %zext_ueq, <vscale x 2 x i64>* %store_addr

  %ugt = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 10, <vscale x 2 x i1> %m, i32 %n)
  %zext_ugt = zext <vscale x 2 x i1> %ugt to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_ugt, <vscale x 2 x i64>* %store_addr

  %uge = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 11, <vscale x 2 x i1> %m, i32 %n)
  %zext_uge = zext <vscale x 2 x i1> %uge to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_uge, <vscale x 2 x i64>* %store_addr

  %ult = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 12, <vscale x 2 x i1> %m, i32 %n)
  %zext_ult = zext <vscale x 2 x i1> %ult to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_ult, <vscale x 2 x i64>* %store_addr

  %ule = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 13, <vscale x 2 x i1> %m, i32 %n)
  %zext_ule = zext <vscale x 2 x i1> %ule to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_ule, <vscale x 2 x i64>* %store_addr

  %une = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 14, <vscale x 2 x i1> %m, i32 %n)
  %zext_une = zext <vscale x 2 x i1> %une to <vscale x 2 x i64>
  store <vscale x 2 x i64> %zext_une, <vscale x 2 x i64>* %store_addr

  ;%true = call <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i8 15, <vscale x 2 x i1> %m, i32 %n)
  ;%zext_true = zext <vscale x 2 x i1> %true to <vscale x 2 x i64>
  ;store <vscale x 2 x i64> %zext_true, <vscale x 2 x i64>* %store_addr

  ret void
}

; store
declare void @llvm.vp.store.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>*, i32, <vscale x 1 x i1>, i32)
declare void @llvm.vp.store.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>*, i32, <vscale x 2 x i1>, i32)
declare void @llvm.vp.store.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>*, i32, <vscale x 2 x i1>, i32)

; fcmp
declare <vscale x 1 x i1> @llvm.vp.fcmp.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, i8 immarg, <vscale x 1 x i1>, i32)
declare <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f32(<vscale x 2 x float>, <vscale x 2 x float>, i8 immarg, <vscale x 2 x i1>, i32)
declare <vscale x 2 x i1> @llvm.vp.fcmp.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, i8 immarg, <vscale x 2 x i1>, i32)
