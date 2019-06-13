; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+epi -verify-machineinstrs -O0 < %s \
; RUN:    | FileCheck --check-prefix=SPILL-O0 %s
; RUN: llc -mtriple=riscv64 -mattr=+epi -verify-machineinstrs -O2 < %s \
; RUN:    | FileCheck --check-prefix=SPILL-O2 %s

@.str = private unnamed_addr constant [6 x i8] c"hello\00", align 1

define <vscale x 1 x double> @foo(<vscale x 1 x double> %a, <vscale x 1 x double> %b, <vscale x 1 x double> %c, i64 %gvl) nounwind
; SPILL-O0-LABEL: foo:
; SPILL-O0:       # %bb.0:
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
; SPILL-O0-NEXT:    sd sp, -32(s0)
; SPILL-O0-NEXT:    sub sp, sp, a3
; SPILL-O0-NEXT:    andi sp, sp, -16
; SPILL-O0-NEXT:    sd sp, -40(s0)
; SPILL-O0-NEXT:    vsetvl zero, a1, a2
; SPILL-O0-NEXT:    lui a1, %hi(.L.str)
; SPILL-O0-NEXT:    addi a1, a1, %lo(.L.str)
; SPILL-O0-NEXT:    sd a0, -32(s0)
; SPILL-O0-NEXT:    mv a0, a1
; SPILL-O0-NEXT:    rdvtype a2
; SPILL-O0-NEXT:    rdvl a1
; SPILL-O0-NEXT:    ld a3, -32(s0)
; SPILL-O0-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O0-NEXT:    vse.v v17, (a3)
; SPILL-O0-NEXT:    vsetvl zero, a1, a2
; SPILL-O0-NEXT:    rdvtype a2
; SPILL-O0-NEXT:    rdvl a1
; SPILL-O0-NEXT:    ld a3, -40(s0)
; SPILL-O0-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O0-NEXT:    vse.v v16, (a3)
; SPILL-O0-NEXT:    vsetvl zero, a1, a2
; SPILL-O0-NEXT:    call puts
; SPILL-O0-NEXT:    ld a1, -32(s0)
; SPILL-O0-NEXT:    vsetvli a2, a1, e64, m1
; SPILL-O0-NEXT:    rdvtype a2
; SPILL-O0-NEXT:    rdvl a0
; SPILL-O0-NEXT:    ld a3, -40(s0)
; SPILL-O0-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O0-NEXT:    vle.v v0, (a3)
; SPILL-O0-NEXT:    vsetvl zero, a0, a2
; SPILL-O0-NEXT:    rdvtype a2
; SPILL-O0-NEXT:    rdvl a0
; SPILL-O0-NEXT:    ld a3, -32(s0)
; SPILL-O0-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O0-NEXT:    vle.v v1, (a3)
; SPILL-O0-NEXT:    vsetvl zero, a0, a2
; SPILL-O0-NEXT:    vfadd.vv v2, v0, v1
; SPILL-O0-NEXT:    vfadd.vv v16, v0, v2
; SPILL-O0-NEXT:    addi sp, s0, -48
; SPILL-O0-NEXT:    ld s0, 32(sp)
; SPILL-O0-NEXT:    ld ra, 40(sp)
; SPILL-O0-NEXT:    addi sp, sp, 48
; SPILL-O0-NEXT:    ret
;
; SPILL-O2-LABEL: foo:
; SPILL-O2:       # %bb.0:
; SPILL-O2-NEXT:    addi sp, sp, -48
; SPILL-O2-NEXT:    sd ra, 40(sp)
; SPILL-O2-NEXT:    sd s0, 32(sp)
; SPILL-O2-NEXT:    sd s1, 24(sp)
; SPILL-O2-NEXT:    addi s0, sp, 48
; SPILL-O2-NEXT:    rdvtype a2
; SPILL-O2-NEXT:    rdvl a1
; SPILL-O2-NEXT:    vsetvli a3, zero, e64, m1
; SPILL-O2-NEXT:    slli a3, a3, 3
; SPILL-O2-NEXT:    sub sp, sp, a3
; SPILL-O2-NEXT:    andi sp, sp, -16
; SPILL-O2-NEXT:    sd sp, -32(s0)
; SPILL-O2-NEXT:    sub sp, sp, a3
; SPILL-O2-NEXT:    andi sp, sp, -16
; SPILL-O2-NEXT:    sd sp, -40(s0)
; SPILL-O2-NEXT:    vsetvl zero, a1, a2
; SPILL-O2-NEXT:    mv s1, a0
; SPILL-O2-NEXT:    rdvtype a1
; SPILL-O2-NEXT:    rdvl a0
; SPILL-O2-NEXT:    ld a2, -32(s0)
; SPILL-O2-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O2-NEXT:    vse.v v17, (a2)
; SPILL-O2-NEXT:    vsetvl zero, a0, a1
; SPILL-O2-NEXT:    rdvtype a1
; SPILL-O2-NEXT:    rdvl a0
; SPILL-O2-NEXT:    ld a2, -40(s0)
; SPILL-O2-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O2-NEXT:    vse.v v16, (a2)
; SPILL-O2-NEXT:    vsetvl zero, a0, a1
; SPILL-O2-NEXT:    lui a0, %hi(.L.str)
; SPILL-O2-NEXT:    addi a0, a0, %lo(.L.str)
; SPILL-O2-NEXT:    call puts
; SPILL-O2-NEXT:    vsetvli a0, s1, e64, m1
; SPILL-O2-NEXT:    rdvtype a1
; SPILL-O2-NEXT:    rdvl a0
; SPILL-O2-NEXT:    ld a2, -40(s0)
; SPILL-O2-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O2-NEXT:    vle.v v1, (a2)
; SPILL-O2-NEXT:    vsetvl zero, a0, a1
; SPILL-O2-NEXT:    rdvtype a1
; SPILL-O2-NEXT:    rdvl a0
; SPILL-O2-NEXT:    ld a2, -32(s0)
; SPILL-O2-NEXT:    vsetvli zero, zero, e64, m1
; SPILL-O2-NEXT:    vle.v v0, (a2)
; SPILL-O2-NEXT:    vsetvl zero, a0, a1
; SPILL-O2-NEXT:    vfadd.vv v0, v1, v0
; SPILL-O2-NEXT:    vfadd.vv v16, v1, v0
; SPILL-O2-NEXT:    addi sp, s0, -48
; SPILL-O2-NEXT:    ld s1, 24(sp)
; SPILL-O2-NEXT:    ld s0, 32(sp)
; SPILL-O2-NEXT:    ld ra, 40(sp)
; SPILL-O2-NEXT:    addi sp, sp, 48
; SPILL-O2-NEXT:    ret
{
   %x = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i64 %gvl)
   %call = call signext i32 @puts(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i64 0, i64 0))
   %z = call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %x, i64 %gvl)
   ret <vscale x 1 x double> %z
}

declare <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(<vscale x 1 x double> %a, <vscale x 1 x double> %b, i64 %gvl)
declare i32 @puts(i8*);
