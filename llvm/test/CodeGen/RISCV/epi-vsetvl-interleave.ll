; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+epi -verify-machineinstrs -O0 < %s \
; RUN:    | FileCheck --check-prefix=CHECK-O0 %s
; RUN: llc -mtriple=riscv64 -mattr=+epi -verify-machineinstrs -O2 < %s \
; RUN:    | FileCheck --check-prefix=CHECK-O2 %s

@scratch = global i8 0, align 16

declare i64 @llvm.epi.vsetvl(
  i64, i64, i64);

declare <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
  <vscale x 1 x double>,
  <vscale x 1 x double>,
  i64);

declare <vscale x 2 x double> @llvm.epi.vfadd.nxv2f64.nxv2f64(
  <vscale x 2 x double>,
  <vscale x 2 x double>,
  i64);

declare <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(
  <vscale x 2 x float>,
  <vscale x 2 x float>,
  i64);

declare <vscale x 1 x double> @llvm.epi.vload.nxv1f64(
  <vscale x 1 x double>*,
  i64);

declare <vscale x 2 x double> @llvm.epi.vload.nxv2f64(
  <vscale x 2 x double>*,
  i64);

declare <vscale x 2 x float> @llvm.epi.vload.nxv2f32(
  <vscale x 2 x float>*,
  i64);

declare void @llvm.epi.vstore.nxv1f64(
  <vscale x 1 x double>,
  <vscale x 1 x double>*,
  i64);

declare void @llvm.epi.vstore.nxv2f64(
  <vscale x 2 x double>,
  <vscale x 2 x double>*,
  i64);

declare void @llvm.epi.vstore.nxv2f32(
  <vscale x 2 x float>,
  <vscale x 2 x float>*,
  i64);

define void @test_vsetvl_interleave_sew(<vscale x 1 x double>* %vd, <vscale x 2 x float>* %vf, i64 signext %avl)
; CHECK-O0-LABEL: test_vsetvl_interleave_sew:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    addi sp, sp, -32
; CHECK-O0-NEXT:    .cfi_def_cfa_offset 32
; CHECK-O0-NEXT:    vsetvli a3, a2, e64, m1
; CHECK-O0-NEXT:    vsetvli a4, a2, e32, m1
; CHECK-O0-NEXT:    vsetvli a5, a3, e64, m1
; CHECK-O0-NEXT:    vle.v v0, (a0)
; CHECK-O0-NEXT:    vfadd.vv v0, v0, v0
; CHECK-O0-NEXT:    lui a5, %hi(scratch)
; CHECK-O0-NEXT:    addi a5, a5, %lo(scratch)
; CHECK-O0-NEXT:    vsetvli a6, a4, e32, m1
; CHECK-O0-NEXT:    vle.v v1, (a1)
; CHECK-O0-NEXT:    vsetvli a6, a3, e64, m1
; CHECK-O0-NEXT:    vse.v v0, (a5)
; CHECK-O0-NEXT:    vsetvli a3, a4, e32, m1
; CHECK-O0-NEXT:    vfadd.vv v0, v1, v1
; CHECK-O0-NEXT:    vse.v v0, (a5)
; CHECK-O0-NEXT:    sd a2, 24(sp)
; CHECK-O0-NEXT:    sd a1, 16(sp)
; CHECK-O0-NEXT:    sd a0, 8(sp)
; CHECK-O0-NEXT:    addi sp, sp, 32
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vsetvl_interleave_sew:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    vsetvli a3, a2, e64, m1
; CHECK-O2-NEXT:    vsetvli a2, a2, e32, m1
; CHECK-O2-NEXT:    vle.v v0, (a1)
; CHECK-O2-NEXT:    vsetvli a1, a3, e64, m1
; CHECK-O2-NEXT:    vle.v v1, (a0)
; CHECK-O2-NEXT:    vfadd.vv v1, v1, v1
; CHECK-O2-NEXT:    lui a0, %hi(scratch)
; CHECK-O2-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O2-NEXT:    vse.v v1, (a0)
; CHECK-O2-NEXT:    vsetvli a1, a2, e32, m1
; CHECK-O2-NEXT:    vfadd.vv v0, v0, v0
; CHECK-O2-NEXT:    vse.v v0, (a0)
; CHECK-O2-NEXT:    ret
{
  %gvl_d = call i64 @llvm.epi.vsetvl(
    i64 %avl, i64 3, i64 0)

  %gvl_f = call i64 @llvm.epi.vsetvl(
    i64 %avl, i64 2, i64 0)

  %vec_d = call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(
    <vscale x 1 x double>* %vd,
    i64 %gvl_d)

  %vec_f = call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(
    <vscale x 2 x float>* %vf,
    i64 %gvl_f)

  %add_d = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
    <vscale x 1 x double> %vec_d,
    <vscale x 1 x double> %vec_d,
    i64 %gvl_d)

  %add_f = call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(
    <vscale x 2 x float> %vec_f,
    <vscale x 2 x float> %vec_f,
    i64 %gvl_f)

  %store_addr_d = bitcast i8* @scratch to <vscale x 1 x double>*

  %store_addr_f = bitcast i8* @scratch to <vscale x 2 x float>*

  call void @llvm.epi.vstore.nxv1f64(
    <vscale x 1 x double> %add_d,
    <vscale x 1 x double>* %store_addr_d,
    i64 %gvl_d)

  call void @llvm.epi.vstore.nxv2f32(
    <vscale x 2 x float> %add_f,
    <vscale x 2 x float>* %store_addr_f,
    i64 %gvl_f)

    ret void
}

define void @test_vsetvl_interleave_vlmul(<vscale x 1 x double>* %vm1, <vscale x 2 x double>* %vm2, i64 signext %avl)
; CHECK-O0-LABEL: test_vsetvl_interleave_vlmul:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    addi sp, sp, -32
; CHECK-O0-NEXT:    .cfi_def_cfa_offset 32
; CHECK-O0-NEXT:    vsetvli a3, a2, e64, m1
; CHECK-O0-NEXT:    vsetvli a4, a2, e64, m2
; CHECK-O0-NEXT:    vsetvli a5, a3, e64, m1
; CHECK-O0-NEXT:    vle.v v0, (a0)
; CHECK-O0-NEXT:    vfadd.vv v0, v0, v0
; CHECK-O0-NEXT:    lui a5, %hi(scratch)
; CHECK-O0-NEXT:    addi a5, a5, %lo(scratch)
; CHECK-O0-NEXT:    vsetvli a6, a4, e64, m2
; CHECK-O0-NEXT:    vle.v v2, (a1)
; CHECK-O0-NEXT:    vsetvli a6, a3, e64, m1
; CHECK-O0-NEXT:    vse.v v0, (a5)
; CHECK-O0-NEXT:    vsetvli a3, a4, e64, m2
; CHECK-O0-NEXT:    vfadd.vv v2, v2, v2
; CHECK-O0-NEXT:    vse.v v2, (a5)
; CHECK-O0-NEXT:    sd a2, 24(sp)
; CHECK-O0-NEXT:    sd a1, 16(sp)
; CHECK-O0-NEXT:    sd a0, 8(sp)
; CHECK-O0-NEXT:    addi sp, sp, 32
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vsetvl_interleave_vlmul:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    vsetvli a3, a2, e64, m1
; CHECK-O2-NEXT:    vsetvli a2, a2, e64, m2
; CHECK-O2-NEXT:    vle.v v0, (a1)
; CHECK-O2-NEXT:    vsetvli a1, a3, e64, m1
; CHECK-O2-NEXT:    vle.v v2, (a0)
; CHECK-O2-NEXT:    vfadd.vv v2, v2, v2
; CHECK-O2-NEXT:    lui a0, %hi(scratch)
; CHECK-O2-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O2-NEXT:    vse.v v2, (a0)
; CHECK-O2-NEXT:    vsetvli a1, a2, e64, m2
; CHECK-O2-NEXT:    vfadd.vv v0, v0, v0
; CHECK-O2-NEXT:    vse.v v0, (a0)
; CHECK-O2-NEXT:    ret
{
  %gvl_m1 = call i64 @llvm.epi.vsetvl(
    i64 %avl, i64 3, i64 0)

  %gvl_m2 = call i64 @llvm.epi.vsetvl(
    i64 %avl, i64 3, i64 1)

  %vec_m1 = call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(
    <vscale x 1 x double>* %vm1,
    i64 %gvl_m1)

  %vec_m2 = call <vscale x 2 x double> @llvm.epi.vload.nxv2f64(
    <vscale x 2 x double>* %vm2,
    i64 %gvl_m2)

  %add_m1 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
    <vscale x 1 x double> %vec_m1,
    <vscale x 1 x double> %vec_m1,
    i64 %gvl_m1)

  %add_m2 = call <vscale x 2 x double> @llvm.epi.vfadd.nxv2f64.nxv2f64(
    <vscale x 2 x double> %vec_m2,
    <vscale x 2 x double> %vec_m2,
    i64 %gvl_m2)

  %store_addr_m1 = bitcast i8* @scratch to <vscale x 1 x double>*

  %store_addr_m2 = bitcast i8* @scratch to <vscale x 2 x double>*

  call void @llvm.epi.vstore.nxv1f64(
    <vscale x 1 x double> %add_m1,
    <vscale x 1 x double>* %store_addr_m1,
    i64 %gvl_m1)

  call void @llvm.epi.vstore.nxv2f64(
    <vscale x 2 x double> %add_m2,
    <vscale x 2 x double>* %store_addr_m2,
    i64 %gvl_m2)

    ret void
}
