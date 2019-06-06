; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+epi -verify-machineinstrs -O0 < %s \
; RUN:    | FileCheck --check-prefix=CHECK-O0 %s
; RUN: llc -mtriple=riscv64 -mattr=+epi -verify-machineinstrs -O2 < %s \
; RUN:    | FileCheck --check-prefix=CHECK-O2 %s

@scratch = global i8 0, align 16

declare i64 @llvm.epi.vsetvl(
  i64, i64, i64);

declare i64 @llvm.epi.vreadvl();

declare <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
  <vscale x 1 x double>,
  <vscale x 1 x double>,
  i64);

declare <vscale x 1 x double> @llvm.epi.vload.nxv1f64(
  <vscale x 1 x double>*,
  i64);

declare void @llvm.epi.vstore.nxv1f64(
  <vscale x 1 x double>,
  <vscale x 1 x double>*,
  i64);

define void @test_vsetvl_vtype(<vscale x 1 x double>* %v, i64 %avl)
; CHECK-O0-LABEL: test_vsetvl_vtype:
; CHECK-O0:       # %bb.0:
; CHECK-O0-NEXT:    vsetvli a2, a1, e64, m1
; CHECK-O0-NEXT:    srli a3, a2, 1
; CHECK-O0-NEXT:    #APP
; CHECK-O0-NEXT:    rdvtype t0; vsetvl x0, a3, t0
; CHECK-O0-NEXT:    #NO_APP
; CHECK-O0-NEXT:    vsetvli a3, a2, e64, m1
; CHECK-O0-NEXT:    vle.v v0, (a0)
; CHECK-O0-NEXT:    vfadd.vv v0, v0, v0
; CHECK-O0-NEXT:    lui a3, %hi(scratch)
; CHECK-O0-NEXT:    addi a3, a3, %lo(scratch)
; CHECK-O0-NEXT:    vse.v v0, (a3)
; CHECK-O0-NEXT:    ret
;
; CHECK-O2-LABEL: test_vsetvl_vtype:
; CHECK-O2:       # %bb.0:
; CHECK-O2-NEXT:    vsetvli a1, a1, e64, m1
; CHECK-O2-NEXT:    srli a2, a1, 1
; CHECK-O2-NEXT:    #APP
; CHECK-O2-NEXT:    rdvtype t0; vsetvl x0, a2, t0
; CHECK-O2-NEXT:    #NO_APP
; CHECK-O2-NEXT:    vsetvli a2, a1, e64, m1
; CHECK-O2-NEXT:    vle.v v0, (a0)
; CHECK-O2-NEXT:    vfadd.vv v0, v0, v0
; CHECK-O2-NEXT:    lui a0, %hi(scratch)
; CHECK-O2-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-O2-NEXT:    vse.v v0, (a0)
; CHECK-O2-NEXT:    ret
{
  %gvl = call i64 @llvm.epi.vsetvl(
    i64 %avl,
    i64 3,
    i64 0)

  %new_vl = lshr i64 %gvl, 1

  call void asm "rdvtype t0; vsetvl x0, $0, t0", "r,~{vl}"(i64 %new_vl)

  %vec = call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(
    <vscale x 1 x double>* %v,
    i64 %gvl)

  %add = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
    <vscale x 1 x double> %vec,
    <vscale x 1 x double> %vec,
    i64 %gvl)

  %store_addr = bitcast i8* @scratch to <vscale x 1 x double>*

  call void @llvm.epi.vstore.nxv1f64(
    <vscale x 1 x double> %add,
    <vscale x 1 x double>* %store_addr,
    i64 %gvl)

  ret void
}
