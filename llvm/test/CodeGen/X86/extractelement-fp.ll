; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-- -mattr=avx2,fma | FileCheck %s

define float @fneg_v4f32(<4 x float> %x) nounwind {
; CHECK-LABEL: fneg_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vbroadcastss {{.*#+}} xmm1 = [-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0]
; CHECK-NEXT:    vxorps %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %v = fneg <4 x float> %x
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @fneg_v4f64(<4 x double> %x) nounwind {
; CHECK-LABEL: fneg_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vbroadcastsd {{.*#+}} ymm1 = [-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0]
; CHECK-NEXT:    vxorps %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = fneg <4 x double> %x
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @fadd_v4f32(<4 x float> %x, <4 x float> %y) nounwind {
; CHECK-LABEL: fadd_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vaddps %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %v = fadd <4 x float> %x, %y
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @fadd_v4f64(<4 x double> %x, <4 x double> %y) nounwind {
; CHECK-LABEL: fadd_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vaddpd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = fadd <4 x double> %x, %y
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @fsub_v4f32(<4 x float> %x, <4 x float> %y) nounwind {
; CHECK-LABEL: fsub_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsubps %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %v = fsub <4 x float> %x, %y
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @fsub_v4f64(<4 x double> %x, <4 x double> %y) nounwind {
; CHECK-LABEL: fsub_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsubpd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = fsub <4 x double> %x, %y
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @fmul_v4f32(<4 x float> %x, <4 x float> %y) nounwind {
; CHECK-LABEL: fmul_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmulps %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %v = fmul <4 x float> %x, %y
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @fmul_v4f64(<4 x double> %x, <4 x double> %y) nounwind {
; CHECK-LABEL: fmul_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmulpd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = fmul <4 x double> %x, %y
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @fdiv_v4f32(<4 x float> %x, <4 x float> %y) nounwind {
; CHECK-LABEL: fdiv_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vdivps %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %v = fdiv <4 x float> %x, %y
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @fdiv_v4f64(<4 x double> %x, <4 x double> %y) nounwind {
; CHECK-LABEL: fdiv_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vdivpd %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = fdiv <4 x double> %x, %y
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @frem_v4f32(<4 x float> %x, <4 x float> %y) nounwind {
; CHECK-LABEL: frem_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    jmp fmodf # TAILCALL
  %v = frem <4 x float> %x, %y
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @frem_v4f64(<4 x double> %x, <4 x double> %y) nounwind {
; CHECK-LABEL: frem_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 killed $ymm0
; CHECK-NEXT:    # kill: def $xmm1 killed $xmm1 killed $ymm1
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    jmp fmod # TAILCALL
  %v = frem <4 x double> %x, %y
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @fsqrt_v4f32(<4 x float> %x) nounwind {
; CHECK-LABEL: fsqrt_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsqrtps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %v = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %x)
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @fsqrt_v4f64(<4 x double> %x) nounwind {
; CHECK-LABEL: fsqrt_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsqrtpd %ymm0, %ymm0
; CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 killed $ymm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = call <4 x double> @llvm.sqrt.v4f64(<4 x double> %x)
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @fsin_v4f32(<4 x float> %x) nounwind {
; CHECK-LABEL: fsin_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    jmp sinf # TAILCALL
  %v = call <4 x float> @llvm.sin.v4f32(<4 x float> %x)
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @fsin_v4f64(<4 x double> %x) nounwind {
; CHECK-LABEL: fsin_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 killed $ymm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    jmp sin # TAILCALL
  %v = call <4 x double> @llvm.sin.v4f64(<4 x double> %x)
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @fma_v4f32(<4 x float> %x, <4 x float> %y, <4 x float> %z) nounwind {
; CHECK-LABEL: fma_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vfmadd213ps {{.*#+}} xmm0 = (xmm1 * xmm0) + xmm2
; CHECK-NEXT:    retq
  %v = call <4 x float> @llvm.fma.v4f32(<4 x float> %x, <4 x float> %y, <4 x float> %z)
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @fma_v4f64(<4 x double> %x, <4 x double> %y, <4 x double> %z) nounwind {
; CHECK-LABEL: fma_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vfmadd213pd {{.*#+}} ymm0 = (ymm1 * ymm0) + ymm2
; CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 killed $ymm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = call <4 x double> @llvm.fma.v4f64(<4 x double> %x, <4 x double> %y, <4 x double> %z)
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @fabs_v4f32(<4 x float> %x) nounwind {
; CHECK-LABEL: fabs_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vbroadcastss {{.*#+}} xmm1 = [NaN,NaN,NaN,NaN]
; CHECK-NEXT:    vandps %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %v = call <4 x float> @llvm.fabs.v4f32(<4 x float> %x)
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @fabs_v4f64(<4 x double> %x) nounwind {
; CHECK-LABEL: fabs_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vbroadcastsd {{.*#+}} ymm1 = [NaN,NaN,NaN,NaN]
; CHECK-NEXT:    vandps %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = call <4 x double> @llvm.fabs.v4f64(<4 x double> %x)
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @fmaxnum_v4f32(<4 x float> %x, <4 x float> %y) nounwind {
; CHECK-LABEL: fmaxnum_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmaxps %xmm0, %xmm1, %xmm2
; CHECK-NEXT:    vcmpunordps %xmm0, %xmm0, %xmm0
; CHECK-NEXT:    vblendvps %xmm0, %xmm1, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %v = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %x, <4 x float> %y)
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @fmaxnum_v4f64(<4 x double> %x, <4 x double> %y) nounwind {
; CHECK-LABEL: fmaxnum_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmaxpd %ymm0, %ymm1, %ymm2
; CHECK-NEXT:    vcmpunordpd %ymm0, %ymm0, %ymm0
; CHECK-NEXT:    vblendvpd %ymm0, %ymm1, %ymm2, %ymm0
; CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 killed $ymm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %x, <4 x double> %y)
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @fminnum_v4f32(<4 x float> %x, <4 x float> %y) nounwind {
; CHECK-LABEL: fminnum_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vminps %xmm0, %xmm1, %xmm2
; CHECK-NEXT:    vcmpunordps %xmm0, %xmm0, %xmm0
; CHECK-NEXT:    vblendvps %xmm0, %xmm1, %xmm2, %xmm0
; CHECK-NEXT:    retq
  %v = call <4 x float> @llvm.minnum.v4f32(<4 x float> %x, <4 x float> %y)
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @fminnum_v4f64(<4 x double> %x, <4 x double> %y) nounwind {
; CHECK-LABEL: fminnum_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vminpd %ymm0, %ymm1, %ymm2
; CHECK-NEXT:    vcmpunordpd %ymm0, %ymm0, %ymm0
; CHECK-NEXT:    vblendvpd %ymm0, %ymm1, %ymm2, %ymm0
; CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 killed $ymm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = call <4 x double> @llvm.minnum.v4f64(<4 x double> %x, <4 x double> %y)
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

;define float @fmaximum_v4f32(<4 x float> %x, <4 x float> %y) nounwind {
;  %v = call <4 x float> @llvm.maximum.v4f32(<4 x float> %x, <4 x float> %y)
;  %r = extractelement <4 x float> %v, i32 0
;  ret float %r
;}

;define double @fmaximum_v4f64(<4 x double> %x, <4 x double> %y) nounwind {
;  %v = call <4 x double> @llvm.maximum.v4f64(<4 x double> %x, <4 x double> %y)
;  %r = extractelement <4 x double> %v, i32 0
;  ret double %r
;}

;define float @fminimum_v4f32(<4 x float> %x, <4 x float> %y) nounwind {
;  %v = call <4 x float> @llvm.minimum.v4f32(<4 x float> %x, <4 x float> %y)
;  %r = extractelement <4 x float> %v, i32 0
;  ret float %r
;}

;define double @fminimum_v4f64(<4 x double> %x, <4 x double> %y) nounwind {
;  %v = call <4 x double> @llvm.minimum.v4f64(<4 x double> %x, <4 x double> %y)
;  %r = extractelement <4 x double> %v, i32 0
;  ret double %r
;}

define float @copysign_v4f32(<4 x float> %x, <4 x float> %y) nounwind {
; CHECK-LABEL: copysign_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vbroadcastss {{.*#+}} xmm2 = [-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0]
; CHECK-NEXT:    vandps %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vbroadcastss {{.*#+}} xmm2 = [NaN,NaN,NaN,NaN]
; CHECK-NEXT:    vandps %xmm2, %xmm0, %xmm0
; CHECK-NEXT:    vorps %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %v = call <4 x float> @llvm.copysign.v4f32(<4 x float> %x, <4 x float> %y)
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @copysign_v4f64(<4 x double> %x, <4 x double> %y) nounwind {
; CHECK-LABEL: copysign_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vbroadcastsd {{.*#+}} ymm2 = [NaN,NaN,NaN,NaN]
; CHECK-NEXT:    vbroadcastsd {{.*#+}} ymm3 = [-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0]
; CHECK-NEXT:    vandps %xmm3, %xmm1, %xmm1
; CHECK-NEXT:    vandps %xmm2, %xmm0, %xmm0
; CHECK-NEXT:    vorps %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = call <4 x double> @llvm.copysign.v4f64(<4 x double> %x, <4 x double> %y)
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @floor_v4f32(<4 x float> %x) nounwind {
; CHECK-LABEL: floor_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundps $9, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %v = call <4 x float> @llvm.floor.v4f32(<4 x float> %x)
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @floor_v4f64(<4 x double> %x) nounwind {
; CHECK-LABEL: floor_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundpd $9, %ymm0, %ymm0
; CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 killed $ymm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = call <4 x double> @llvm.floor.v4f64(<4 x double> %x)
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @ceil_v4f32(<4 x float> %x) nounwind {
; CHECK-LABEL: ceil_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundps $10, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %v = call <4 x float> @llvm.ceil.v4f32(<4 x float> %x)
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @ceil_v4f64(<4 x double> %x) nounwind {
; CHECK-LABEL: ceil_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundpd $10, %ymm0, %ymm0
; CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 killed $ymm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = call <4 x double> @llvm.ceil.v4f64(<4 x double> %x)
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @trunc_v4f32(<4 x float> %x) nounwind {
; CHECK-LABEL: trunc_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundps $11, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %v = call <4 x float> @llvm.trunc.v4f32(<4 x float> %x)
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @trunc_v4f64(<4 x double> %x) nounwind {
; CHECK-LABEL: trunc_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundpd $11, %ymm0, %ymm0
; CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 killed $ymm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = call <4 x double> @llvm.trunc.v4f64(<4 x double> %x)
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @rint_v4f32(<4 x float> %x) nounwind {
; CHECK-LABEL: rint_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundps $4, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %v = call <4 x float> @llvm.rint.v4f32(<4 x float> %x)
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @rint_v4f64(<4 x double> %x) nounwind {
; CHECK-LABEL: rint_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundpd $4, %ymm0, %ymm0
; CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 killed $ymm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = call <4 x double> @llvm.rint.v4f64(<4 x double> %x)
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @nearbyint_v4f32(<4 x float> %x) nounwind {
; CHECK-LABEL: nearbyint_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundps $12, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %v = call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %x)
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @nearbyint_v4f64(<4 x double> %x) nounwind {
; CHECK-LABEL: nearbyint_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vroundpd $12, %ymm0, %ymm0
; CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 killed $ymm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %v = call <4 x double> @llvm.nearbyint.v4f64(<4 x double> %x)
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

define float @round_v4f32(<4 x float> %x) nounwind {
; CHECK-LABEL: round_v4f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    jmp roundf # TAILCALL
  %v = call <4 x float> @llvm.round.v4f32(<4 x float> %x)
  %r = extractelement <4 x float> %v, i32 0
  ret float %r
}

define double @round_v4f64(<4 x double> %x) nounwind {
; CHECK-LABEL: round_v4f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 killed $ymm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    jmp round # TAILCALL
  %v = call <4 x double> @llvm.round.v4f64(<4 x double> %x)
  %r = extractelement <4 x double> %v, i32 0
  ret double %r
}

declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)
declare <4 x double> @llvm.sqrt.v4f64(<4 x double>)
declare <4 x float> @llvm.sin.v4f32(<4 x float>)
declare <4 x double> @llvm.sin.v4f64(<4 x double>)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <4 x double> @llvm.fma.v4f64(<4 x double>, <4 x double>, <4 x double>)
declare <4 x float> @llvm.fabs.v4f32(<4 x float>)
declare <4 x double> @llvm.fabs.v4f64(<4 x double>)
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>)
declare <4 x double> @llvm.maxnum.v4f64(<4 x double>, <4 x double>)
declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>)
declare <4 x double> @llvm.minnum.v4f64(<4 x double>, <4 x double>)
declare <4 x float> @llvm.maximum.v4f32(<4 x float>, <4 x float>)
declare <4 x double> @llvm.maximum.v4f64(<4 x double>, <4 x double>)
declare <4 x float> @llvm.minimum.v4f32(<4 x float>, <4 x float>)
declare <4 x double> @llvm.minimum.v4f64(<4 x double>, <4 x double>)
declare <4 x float> @llvm.copysign.v4f32(<4 x float>, <4 x float>)
declare <4 x double> @llvm.copysign.v4f64(<4 x double>, <4 x double>)
declare <4 x float> @llvm.floor.v4f32(<4 x float>)
declare <4 x double> @llvm.floor.v4f64(<4 x double>)
declare <4 x float> @llvm.ceil.v4f32(<4 x float>)
declare <4 x double> @llvm.ceil.v4f64(<4 x double>)
declare <4 x float> @llvm.trunc.v4f32(<4 x float>)
declare <4 x double> @llvm.trunc.v4f64(<4 x double>)
declare <4 x float> @llvm.rint.v4f32(<4 x float>)
declare <4 x double> @llvm.rint.v4f64(<4 x double>)
declare <4 x float> @llvm.nearbyint.v4f32(<4 x float>)
declare <4 x double> @llvm.nearbyint.v4f64(<4 x double>)
declare <4 x float> @llvm.round.v4f32(<4 x float>)
declare <4 x double> @llvm.round.v4f64(<4 x double>)
