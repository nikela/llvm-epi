; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+epi -verify-machineinstrs -O0 < %s \
; RUN:    | FileCheck --check-prefix=SPILL-O0 %s
; RUN: llc -mtriple=riscv64 -mattr=+epi -verify-machineinstrs -O2 < %s \
; RUN:    | FileCheck --check-prefix=SPILL-O2 %s

define void @builtins_f64(<vscale x 1 x double>* %vaddr) nounwind {
; SPILL-O0-LABEL: builtins_f64:
; SPILL-O0:       # %bb.0: # %entry
; SPILL-O0-NEXT:    addi sp, sp, -48
; SPILL-O0-NEXT:    sd ra, 40(sp)
; SPILL-O0-NEXT:    sd s0, 32(sp)
; SPILL-O0-NEXT:    addi s0, sp, 48
; SPILL-O0-NEXT:    rdvtype a2
; SPILL-O0-NEXT:    rdvl a1
; SPILL-O0-NEXT:    vsetvli a3, zero, e64, m1
; SPILL-O0-NEXT:    slli a3, a3, 3
; SPILL-O0-NEXT:    sub sp, sp, a3
; SPILL-O0-NEXT:    andi sp, sp, -16
; SPILL-O0-NEXT:    sd sp, -24(s0)
; SPILL-O0-NEXT:    sub sp, sp, a3
; SPILL-O0-NEXT:    andi sp, sp, -16
; SPILL-O0-NEXT:    sd sp, -32(s0)
; SPILL-O0-NEXT:    vsetvl zero, a1, a2
; SPILL-O0-NEXT:    vsetvli a1, zero, e64, m1
; SPILL-O0-NEXT:    vle.v v0, (a0)
; SPILL-O0-NEXT:    mv a1, zero
; SPILL-O0-NEXT:    vsetvli a2, a1, e64, m1
; SPILL-O0-NEXT:    vfadd.vv v1, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v2, v1, v1
; SPILL-O0-NEXT:    vfadd.vv v3, v2, v2
; SPILL-O0-NEXT:    vfadd.vv v4, v3, v3
; SPILL-O0-NEXT:    vfadd.vv v5, v4, v4
; SPILL-O0-NEXT:    vfadd.vv v6, v5, v5
; SPILL-O0-NEXT:    vfadd.vv v7, v6, v6
; SPILL-O0-NEXT:    vfadd.vv v16, v7, v7
; SPILL-O0-NEXT:    vfadd.vv v17, v16, v16
; SPILL-O0-NEXT:    vfadd.vv v18, v17, v17
; SPILL-O0-NEXT:    vfadd.vv v19, v18, v18
; SPILL-O0-NEXT:    vfadd.vv v20, v19, v19
; SPILL-O0-NEXT:    vfadd.vv v21, v20, v20
; SPILL-O0-NEXT:    vfadd.vv v22, v21, v21
; SPILL-O0-NEXT:    vfadd.vv v23, v22, v22
; SPILL-O0-NEXT:    vfadd.vv v8, v23, v23
; SPILL-O0-NEXT:    vfadd.vv v9, v8, v8
; SPILL-O0-NEXT:    vfadd.vv v10, v9, v9
; SPILL-O0-NEXT:    vfadd.vv v11, v10, v10
; SPILL-O0-NEXT:    vfadd.vv v12, v11, v11
; SPILL-O0-NEXT:    vfadd.vv v13, v12, v12
; SPILL-O0-NEXT:    vfadd.vv v14, v13, v13
; SPILL-O0-NEXT:    vfadd.vv v15, v14, v14
; SPILL-O0-NEXT:    vfadd.vv v24, v15, v15
; SPILL-O0-NEXT:    vfadd.vv v25, v24, v24
; SPILL-O0-NEXT:    vfadd.vv v26, v25, v25
; SPILL-O0-NEXT:    vfadd.vv v27, v26, v26
; SPILL-O0-NEXT:    vfadd.vv v28, v27, v27
; SPILL-O0-NEXT:    vfadd.vv v29, v28, v28
; SPILL-O0-NEXT:    vfadd.vv v30, v29, v29
; SPILL-O0-NEXT:    vfadd.vv v31, v30, v30
; SPILL-O0-NEXT:    rdvtype a3
; SPILL-O0-NEXT:    rdvl a2
; SPILL-O0-NEXT:    ld a4, -24(s0)
; SPILL-O0-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O0-NEXT:    vse.v v0, (a4)
; SPILL-O0-NEXT:    vsetvl zero, a2, a3
; SPILL-O0-NEXT:    vfadd.vv v0, v31, v31
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O0-NEXT:    rdvtype a3
; SPILL-O0-NEXT:    rdvl a2
; SPILL-O0-NEXT:    ld a4, -32(s0)
; SPILL-O0-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O0-NEXT:    vse.v v1, (a4)
; SPILL-O0-NEXT:    vsetvl zero, a2, a3
; SPILL-O0-NEXT:    rdvtype a3
; SPILL-O0-NEXT:    rdvl a2
; SPILL-O0-NEXT:    ld a4, -24(s0)
; SPILL-O0-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O0-NEXT:    vle.v v1, (a4)
; SPILL-O0-NEXT:    vsetvl zero, a2, a3
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v1
; SPILL-O0-NEXT:    rdvtype a3
; SPILL-O0-NEXT:    rdvl a2
; SPILL-O0-NEXT:    ld a4, -32(s0)
; SPILL-O0-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O0-NEXT:    vle.v v1, (a4)
; SPILL-O0-NEXT:    vsetvl zero, a2, a3
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v1
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v2
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v3
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v4
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v5
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v6
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v7
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v16
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v17
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v18
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v19
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v20
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v21
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v22
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v23
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v8
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v9
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v10
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v11
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v12
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v13
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v14
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v15
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v24
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v25
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v26
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v27
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v28
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v29
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v30
; SPILL-O0-NEXT:    vfadd.vv v0, v0, v31
; SPILL-O0-NEXT:    vsetvli a1, zero, e64, m1
; SPILL-O0-NEXT:    vse.v v0, (a0)
; SPILL-O0-NEXT:    sd a0, -40(s0)
; SPILL-O0-NEXT:    addi sp, s0, -48
; SPILL-O0-NEXT:    ld s0, 32(sp)
; SPILL-O0-NEXT:    ld ra, 40(sp)
; SPILL-O0-NEXT:    addi sp, sp, 48
; SPILL-O0-NEXT:    ret
;
; SPILL-O2-LABEL: builtins_f64:
; SPILL-O2:       # %bb.0: # %entry
; SPILL-O2-NEXT:    addi sp, sp, -32
; SPILL-O2-NEXT:    sd ra, 24(sp)
; SPILL-O2-NEXT:    sd s0, 16(sp)
; SPILL-O2-NEXT:    addi s0, sp, 32
; SPILL-O2-NEXT:    rdvtype a2
; SPILL-O2-NEXT:    rdvl a1
; SPILL-O2-NEXT:    vsetvli a3, zero, e64, m1
; SPILL-O2-NEXT:    slli a3, a3, 3
; SPILL-O2-NEXT:    sub sp, sp, a3
; SPILL-O2-NEXT:    andi sp, sp, -16
; SPILL-O2-NEXT:    sd sp, -24(s0)
; SPILL-O2-NEXT:    sub sp, sp, a3
; SPILL-O2-NEXT:    andi sp, sp, -16
; SPILL-O2-NEXT:    sd sp, -32(s0)
; SPILL-O2-NEXT:    vsetvl zero, a1, a2
; SPILL-O2-NEXT:    vsetvli a1, zero, e64, m1
; SPILL-O2-NEXT:    vle.v v0, (a0)
; SPILL-O2-NEXT:    rdvtype a2
; SPILL-O2-NEXT:    rdvl a1
; SPILL-O2-NEXT:    ld a3, -24(s0)
; SPILL-O2-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O2-NEXT:    vse.v v0, (a3)
; SPILL-O2-NEXT:    vsetvl zero, a1, a2
; SPILL-O2-NEXT:    mv a1, zero
; SPILL-O2-NEXT:    vsetvli a2, a1, e64, m1
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    rdvtype a3
; SPILL-O2-NEXT:    rdvl a2
; SPILL-O2-NEXT:    ld a4, -32(s0)
; SPILL-O2-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O2-NEXT:    vse.v v0, (a4)
; SPILL-O2-NEXT:    vsetvl zero, a2, a3
; SPILL-O2-NEXT:    vfadd.vv v2, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v3, v2, v2
; SPILL-O2-NEXT:    vfadd.vv v4, v3, v3
; SPILL-O2-NEXT:    vfadd.vv v5, v4, v4
; SPILL-O2-NEXT:    vfadd.vv v6, v5, v5
; SPILL-O2-NEXT:    vfadd.vv v7, v6, v6
; SPILL-O2-NEXT:    vfadd.vv v16, v7, v7
; SPILL-O2-NEXT:    vfadd.vv v17, v16, v16
; SPILL-O2-NEXT:    vfadd.vv v18, v17, v17
; SPILL-O2-NEXT:    vfadd.vv v19, v18, v18
; SPILL-O2-NEXT:    vfadd.vv v20, v19, v19
; SPILL-O2-NEXT:    vfadd.vv v21, v20, v20
; SPILL-O2-NEXT:    vfadd.vv v22, v21, v21
; SPILL-O2-NEXT:    vfadd.vv v23, v22, v22
; SPILL-O2-NEXT:    vfadd.vv v8, v23, v23
; SPILL-O2-NEXT:    vfadd.vv v9, v8, v8
; SPILL-O2-NEXT:    vfadd.vv v10, v9, v9
; SPILL-O2-NEXT:    vfadd.vv v11, v10, v10
; SPILL-O2-NEXT:    vfadd.vv v12, v11, v11
; SPILL-O2-NEXT:    vfadd.vv v13, v12, v12
; SPILL-O2-NEXT:    vfadd.vv v14, v13, v13
; SPILL-O2-NEXT:    vfadd.vv v15, v14, v14
; SPILL-O2-NEXT:    vfadd.vv v24, v15, v15
; SPILL-O2-NEXT:    vfadd.vv v25, v24, v24
; SPILL-O2-NEXT:    vfadd.vv v26, v25, v25
; SPILL-O2-NEXT:    vfadd.vv v27, v26, v26
; SPILL-O2-NEXT:    vfadd.vv v28, v27, v27
; SPILL-O2-NEXT:    vfadd.vv v29, v28, v28
; SPILL-O2-NEXT:    vfadd.vv v30, v29, v29
; SPILL-O2-NEXT:    vfadd.vv v31, v30, v30
; SPILL-O2-NEXT:    vfadd.vv v0, v31, v31
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v0
; SPILL-O2-NEXT:    rdvtype a3
; SPILL-O2-NEXT:    rdvl a2
; SPILL-O2-NEXT:    ld a4, -24(s0)
; SPILL-O2-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O2-NEXT:    vle.v v1, (a4)
; SPILL-O2-NEXT:    vsetvl zero, a2, a3
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v1
; SPILL-O2-NEXT:    rdvtype a3
; SPILL-O2-NEXT:    rdvl a2
; SPILL-O2-NEXT:    ld a4, -32(s0)
; SPILL-O2-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O2-NEXT:    vle.v v1, (a4)
; SPILL-O2-NEXT:    vsetvl zero, a2, a3
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v1
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v2
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v3
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v4
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v5
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v6
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v7
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v16
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v17
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v18
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v19
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v20
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v21
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v22
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v23
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v8
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v9
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v10
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v11
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v12
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v13
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v14
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v15
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v24
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v25
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v26
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v27
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v28
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v29
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v30
; SPILL-O2-NEXT:    vfadd.vv v0, v0, v31
; SPILL-O2-NEXT:    vsetvli a1, zero, e64, m1
; SPILL-O2-NEXT:    vse.v v0, (a0)
; SPILL-O2-NEXT:    addi sp, s0, -32
; SPILL-O2-NEXT:    ld s0, 16(sp)
; SPILL-O2-NEXT:    ld ra, 24(sp)
; SPILL-O2-NEXT:    addi sp, sp, 32
; SPILL-O2-NEXT:    ret


entry:
  %v10 = load <vscale x 1 x double>, <vscale x 1 x double>* %vaddr, align 8

  %v11 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v10, <vscale x 1 x double> %v10, i64 0)
  %v12 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v11, <vscale x 1 x double> %v11, i64 0)
  %v13 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v12, <vscale x 1 x double> %v12, i64 0)
  %v14 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v13, <vscale x 1 x double> %v13, i64 0)
  %v15 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v14, <vscale x 1 x double> %v14, i64 0)
  %v16 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v15, <vscale x 1 x double> %v15, i64 0)
  %v17 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v16, <vscale x 1 x double> %v16, i64 0)
  %v18 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v17, <vscale x 1 x double> %v17, i64 0)
  %v19 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v18, <vscale x 1 x double> %v18, i64 0)
  %v20 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v19, <vscale x 1 x double> %v19, i64 0)
  %v21 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v20, <vscale x 1 x double> %v20, i64 0)
  %v22 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v21, <vscale x 1 x double> %v21, i64 0)
  %v23 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v22, <vscale x 1 x double> %v22, i64 0)
  %v24 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v23, <vscale x 1 x double> %v23, i64 0)
  %v25 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v24, <vscale x 1 x double> %v24, i64 0)
  %v26 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v25, <vscale x 1 x double> %v25, i64 0)
  %v27 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v26, <vscale x 1 x double> %v26, i64 0)
  %v28 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v27, <vscale x 1 x double> %v27, i64 0)
  %v29 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v28, <vscale x 1 x double> %v28, i64 0)
  %v30 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v29, <vscale x 1 x double> %v29, i64 0)
  %v31 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v30, <vscale x 1 x double> %v30, i64 0)
  %v32 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v31, <vscale x 1 x double> %v31, i64 0)
  %v33 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v32, <vscale x 1 x double> %v32, i64 0)
  %v34 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v33, <vscale x 1 x double> %v33, i64 0)
  %v35 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v34, <vscale x 1 x double> %v34, i64 0)
  %v36 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v35, <vscale x 1 x double> %v35, i64 0)
  %v37 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v36, <vscale x 1 x double> %v36, i64 0)
  %v38 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v37, <vscale x 1 x double> %v37, i64 0)
  %v39 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v38, <vscale x 1 x double> %v38, i64 0)
  %v40 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v39, <vscale x 1 x double> %v39, i64 0)
  %v41 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v40, <vscale x 1 x double> %v40, i64 0)
  %v42 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v41, <vscale x 1 x double> %v41, i64 0)
  %v43 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v42, <vscale x 1 x double> %v42, i64 0)
  %v44 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v43, <vscale x 1 x double> %v43, i64 0)
  %v45 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v44, <vscale x 1 x double> %v44, i64 0)
  %v46 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v45, <vscale x 1 x double> %v45, i64 0)
  %v47 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v46, <vscale x 1 x double> %v46, i64 0)
  %v48 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v47, <vscale x 1 x double> %v47, i64 0)
  %v49 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v48, <vscale x 1 x double> %v48, i64 0)
  %v50 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v49, <vscale x 1 x double> %v49, i64 0)
  %v51 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v50, <vscale x 1 x double> %v50, i64 0)
  %v52 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v51, <vscale x 1 x double> %v51, i64 0)
  %v53 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v52, <vscale x 1 x double> %v52, i64 0)
  %v54 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v53, <vscale x 1 x double> %v53, i64 0)
  %v55 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v54, <vscale x 1 x double> %v54, i64 0)
  %v56 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v55, <vscale x 1 x double> %v55, i64 0)
  %v57 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v56, <vscale x 1 x double> %v56, i64 0)
  %v58 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v57, <vscale x 1 x double> %v57, i64 0)
  %v59 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v58, <vscale x 1 x double> %v58, i64 0)
  %v60 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v59, <vscale x 1 x double> %v59, i64 0)
  %v61 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v60, <vscale x 1 x double> %v60, i64 0)
  %v62 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v61, <vscale x 1 x double> %v61, i64 0)
  %v63 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v62, <vscale x 1 x double> %v62, i64 0)
  %v64 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v63, <vscale x 1 x double> %v63, i64 0)
  %v65 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v64, <vscale x 1 x double> %v64, i64 0)
  %v66 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v65, <vscale x 1 x double> %v65, i64 0)
  %v67 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v66, <vscale x 1 x double> %v66, i64 0)
  %v68 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v67, <vscale x 1 x double> %v67, i64 0)
  %v69 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v68, <vscale x 1 x double> %v68, i64 0)
  %v70 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v69, <vscale x 1 x double> %v69, i64 0)

  %v71 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v70, <vscale x 1 x double> %v10, i64 0)
  %v72 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v71, <vscale x 1 x double> %v11, i64 0)
  %v73 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v72, <vscale x 1 x double> %v12, i64 0)
  %v74 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v73, <vscale x 1 x double> %v13, i64 0)
  %v75 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v74, <vscale x 1 x double> %v14, i64 0)
  %v76 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v75, <vscale x 1 x double> %v15, i64 0)
  %v77 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v76, <vscale x 1 x double> %v16, i64 0)
  %v78 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v77, <vscale x 1 x double> %v17, i64 0)
  %v79 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v78, <vscale x 1 x double> %v18, i64 0)
  %v80 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v79, <vscale x 1 x double> %v19, i64 0)
  %v81 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v80, <vscale x 1 x double> %v20, i64 0)
  %v82 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v81, <vscale x 1 x double> %v21, i64 0)
  %v83 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v82, <vscale x 1 x double> %v22, i64 0)
  %v84 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v83, <vscale x 1 x double> %v23, i64 0)
  %v85 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v84, <vscale x 1 x double> %v24, i64 0)
  %v86 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v85, <vscale x 1 x double> %v25, i64 0)
  %v87 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v86, <vscale x 1 x double> %v26, i64 0)
  %v88 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v87, <vscale x 1 x double> %v27, i64 0)
  %v89 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v88, <vscale x 1 x double> %v28, i64 0)
  %v90 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v89, <vscale x 1 x double> %v29, i64 0)
  %v91 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v90, <vscale x 1 x double> %v30, i64 0)
  %v92 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v91, <vscale x 1 x double> %v31, i64 0)
  %v93 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v92, <vscale x 1 x double> %v32, i64 0)
  %v94 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v93, <vscale x 1 x double> %v33, i64 0)
  %v95 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v94, <vscale x 1 x double> %v34, i64 0)
  %v96 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v95, <vscale x 1 x double> %v35, i64 0)
  %v97 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v96, <vscale x 1 x double> %v36, i64 0)
  %v98 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v97, <vscale x 1 x double> %v37, i64 0)
  %v99 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v98, <vscale x 1 x double> %v38, i64 0)
  %v100 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v99, <vscale x 1 x double> %v39, i64 0)
  %v101 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v100, <vscale x 1 x double> %v40, i64 0)
  %v102 = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double> %v101, <vscale x 1 x double> %v41, i64 0)

  store <vscale x 1 x double> %v102, <vscale x 1 x double>* %vaddr, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, i64)
