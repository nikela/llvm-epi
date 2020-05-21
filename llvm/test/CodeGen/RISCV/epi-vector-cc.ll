; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+m,+f,+d,+a,+c,+experimental-v \
; RUN:    -verify-machineinstrs < %s | FileCheck %s

declare <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
                                 <vscale x 1 x double> %v1,
                                 <vscale x 1 x double> %v2,
                                 i64 %gvl)

define <vscale x 1 x double> @basic_callee(
; CHECK-LABEL: basic_callee:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, a0, e64,m1
; CHECK-NEXT:    vfadd.vv v16, v16, v17
; CHECK-NEXT:    ret
                                 <vscale x 1 x double> %v1,
                                 <vscale x 1 x double> %v2,
                                 i64 %gvl) nounwind
{
  %v3 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
      <vscale x 1 x double> %v1,
      <vscale x 1 x double> %v2,
      i64 %gvl)
  ret <vscale x 1 x double> %v3
}


define <vscale x 1 x double> @max_registers(
; CHECK-LABEL: max_registers:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, a0, e64,m1
; CHECK-NEXT:    vfadd.vv v1, v16, v17
; CHECK-NEXT:    vfadd.vv v1, v1, v19
; CHECK-NEXT:    vfadd.vv v1, v1, v20
; CHECK-NEXT:    vfadd.vv v1, v1, v21
; CHECK-NEXT:    vfadd.vv v1, v1, v22
; CHECK-NEXT:    vfadd.vv v16, v1, v23
; CHECK-NEXT:    ret
                                 <vscale x 1 x double> %v0,
                                 <vscale x 1 x double> %v1,
                                 <vscale x 1 x double> %v2,
                                 <vscale x 1 x double> %v3,
                                 <vscale x 1 x double> %v4,
                                 <vscale x 1 x double> %v5,
                                 <vscale x 1 x double> %v6,
                                 <vscale x 1 x double> %v7,
                                 i64 %gvl) nounwind
{
  %vt1 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
      <vscale x 1 x double> %v0,
      <vscale x 1 x double> %v1,
      i64 %gvl)
  %vt2 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
      <vscale x 1 x double> %vt1,
      <vscale x 1 x double> %v3,
      i64 %gvl)
  %vt3 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
      <vscale x 1 x double> %vt2,
      <vscale x 1 x double> %v4,
      i64 %gvl)
  %vt4 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
      <vscale x 1 x double> %vt3,
      <vscale x 1 x double> %v5,
      i64 %gvl)
  %vt5 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
      <vscale x 1 x double> %vt4,
      <vscale x 1 x double> %v6,
      i64 %gvl)
  %vt6 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
      <vscale x 1 x double> %vt5,
      <vscale x 1 x double> %v7,
      i64 %gvl)
  ret <vscale x 1 x double> %vt6
}

define <vscale x 1 x double> @too_many_registers_1(
; CHECK-LABEL: too_many_registers_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a3, zero, e64,m1
; CHECK-NEXT:    vle.v v1, (a2)
; CHECK-NEXT:    vle.v v2, (a1)
; CHECK-NEXT:    vsetvli a0, a0, e64,m1
; CHECK-NEXT:    vfadd.vv v16, v2, v1
; CHECK-NEXT:    ret
                                 i64 %gvl,
                                 <vscale x 1 x double> %v0,
                                 <vscale x 1 x double> %v1,
                                 <vscale x 1 x double> %v2,
                                 <vscale x 1 x double> %v3,
                                 <vscale x 1 x double> %v4,
                                 <vscale x 1 x double> %v5,
                                 <vscale x 1 x double> %v6,
                                 <vscale x 1 x double> %v7,
                                 <vscale x 1 x double> %v8,
                                 <vscale x 1 x double> %v9) nounwind
{
  %vt = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
      <vscale x 1 x double> %v8,
      <vscale x 1 x double> %v9,
      i64 %gvl)
  ret <vscale x 1 x double> %vt
}

define <vscale x 1 x double> @too_many_registers_2(
; CHECK-LABEL: too_many_registers_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a3, zero, e64,m1
; CHECK-NEXT:    vle.v v1, (a1)
; CHECK-NEXT:    vle.v v2, (a0)
; CHECK-NEXT:    vsetvli a0, a2, e64,m1
; CHECK-NEXT:    vfadd.vv v16, v2, v1
; CHECK-NEXT:    ret
                                 <vscale x 1 x double> %v0,
                                 <vscale x 1 x double> %v1,
                                 <vscale x 1 x double> %v2,
                                 <vscale x 1 x double> %v3,
                                 <vscale x 1 x double> %v4,
                                 <vscale x 1 x double> %v5,
                                 <vscale x 1 x double> %v6,
                                 <vscale x 1 x double> %v7,
                                 <vscale x 1 x double> %v8,
                                 <vscale x 1 x double> %v9,
                                 i64 %gvl) nounwind
{
  %vt = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
      <vscale x 1 x double> %v8,
      <vscale x 1 x double> %v9,
      i64 %gvl)
  ret <vscale x 1 x double> %vt
}

define <vscale x 1 x double> @too_many_registers_3(
; CHECK-LABEL: too_many_registers_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld a0, 8(sp)
; CHECK-NEXT:    ld a1, 0(sp)
; CHECK-NEXT:    vsetvli a2, zero, e64,m1
; CHECK-NEXT:    vle.v v1, (a0)
; CHECK-NEXT:    vle.v v2, (a1)
; CHECK-NEXT:    vsetvli a0, a7, e64,m1
; CHECK-NEXT:    vfadd.vv v16, v2, v1
; CHECK-NEXT:    ret
                                 i64 %i0,
                                 i64 %i1,
                                 i64 %i2,
                                 i64 %i3,
                                 i64 %i4,
                                 i64 %i5,
                                 i64 %i6,
                                 i64 %gvl,
                                 <vscale x 1 x double> %v0,
                                 <vscale x 1 x double> %v1,
                                 <vscale x 1 x double> %v2,
                                 <vscale x 1 x double> %v3,
                                 <vscale x 1 x double> %v4,
                                 <vscale x 1 x double> %v5,
                                 <vscale x 1 x double> %v6,
                                 <vscale x 1 x double> %v7,
                                 <vscale x 1 x double> %v8,
                                 <vscale x 1 x double> %v9) nounwind
{
  %vt = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(
      <vscale x 1 x double> %v8,
      <vscale x 1 x double> %v9,
      i64 %gvl)
  ret <vscale x 1 x double> %vt
}

declare <vscale x 1 x double> @llvm.epi.vfadd.mask.nxv1f64.nxv1f64.nxv1i1(
      <vscale x 1 x double> %vmerge,
      <vscale x 1 x double> %v1,
      <vscale x 1 x double> %v2,
      <vscale x 1 x i1> %mask,
      i64 %gvl)

define <vscale x 1 x double> @first_mask_0(
; CHECK-LABEL: first_mask_0:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, a0, e64,m1
; CHECK-NEXT:    vfadd.vv v16, v16, v17, v0.t
; CHECK-NEXT:    ret
                                 <vscale x 1 x i1> %mask,
                                 <vscale x 1 x double> %v1,
                                 <vscale x 1 x double> %v2,
                                 i64 %gvl) nounwind
{
  %v3 = call <vscale x 1 x double> @llvm.epi.vfadd.mask.nxv1f64.nxv1f64.nxv1i1(
      <vscale x 1 x double> %v1,
      <vscale x 1 x double> %v1,
      <vscale x 1 x double> %v2,
      <vscale x 1 x i1> %mask,
      i64 %gvl)
  ret <vscale x 1 x double> %v3
}

define <vscale x 1 x double> @first_mask_2(
; CHECK-LABEL: first_mask_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, a0, e64,m1
; CHECK-NEXT:    vfadd.vv v16, v16, v17, v0.t
; CHECK-NEXT:    ret
                                 <vscale x 1 x double> %v1,
                                 <vscale x 1 x double> %v2,
                                 <vscale x 1 x i1> %mask,
                                 i64 %gvl) nounwind
{
  %v3 = call <vscale x 1 x double> @llvm.epi.vfadd.mask.nxv1f64.nxv1f64.nxv1i1(
      <vscale x 1 x double> %v1,
      <vscale x 1 x double> %v1,
      <vscale x 1 x double> %v2,
      <vscale x 1 x i1> %mask,
      i64 %gvl)
  ret <vscale x 1 x double> %v3
}

define <vscale x 1 x double> @second_mask_1(
; CHECK-LABEL: second_mask_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmv1r.v v0, v18
; CHECK-NEXT:    vsetvli a0, a0, e64,m1
; CHECK-NEXT:    vfadd.vv v16, v16, v17, v0.t
; CHECK-NEXT:    ret
                                 <vscale x 1 x double> %v1,
                                 <vscale x 1 x double> %v2,
                                 <vscale x 1 x i1> %mask1,
                                 <vscale x 1 x i1> %mask2,
                                 i64 %gvl) nounwind
{
  %v3 = call <vscale x 1 x double> @llvm.epi.vfadd.mask.nxv1f64.nxv1f64.nxv1i1(
      <vscale x 1 x double> %v1,
      <vscale x 1 x double> %v1,
      <vscale x 1 x double> %v2,
      <vscale x 1 x i1> %mask2,
      i64 %gvl)
  ret <vscale x 1 x double> %v3
}

define <vscale x 1 x i1> @return_mask_1(<vscale x 1 x i1> %mask1) nounwind
; CHECK-LABEL: return_mask_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ret
{
  ret <vscale x 1 x i1> %mask1
}

define <vscale x 1 x i1> @return_mask_2(<vscale x 1 x i1> %mask1,
; CHECK-LABEL: return_mask_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmv1r.v v0, v16
; CHECK-NEXT:    ret
                                      <vscale x 1 x i1> %mask2) nounwind
{
  ret <vscale x 1 x i1> %mask2
}
