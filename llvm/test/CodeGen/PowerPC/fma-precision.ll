; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc64le-linux-gnu | FileCheck %s

; Verify that the fold of a*b-c*d respect the uses of a*b
define double @fsub1(double %a, double %b, double %c, double %d)  {
; CHECK-LABEL: fsub1:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    xsmuldp 3, 4, 3
; CHECK-NEXT:    xsmuldp 0, 2, 1
; CHECK-NEXT:    xsmsubadp 3, 2, 1
; CHECK-NEXT:    xsmuldp 1, 0, 3
; CHECK-NEXT:    blr
entry:
  %mul = fmul fast double %b, %a
  %mul1 = fmul fast double %d, %c
  %sub = fsub fast double %mul, %mul1
  %mul3 = fmul fast double %mul, %sub
  ret double %mul3
}

; Verify that the fold of a*b-c*d respect the uses of c*d
define double @fsub2(double %a, double %b, double %c, double %d)  {
; CHECK-LABEL: fsub2:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    xsmuldp 0, 4, 3
; CHECK-NEXT:    fmr 3, 0
; CHECK-NEXT:    xsmsubadp 3, 2, 1
; CHECK-NEXT:    xsmuldp 1, 0, 3
; CHECK-NEXT:    blr
entry:
  %mul = fmul fast double %b, %a
  %mul1 = fmul fast double %d, %c
  %sub = fsub fast double %mul, %mul1
  %mul3 = fmul fast double %mul1, %sub
  ret double %mul3
}

; Verify that the fold of a*b-c*d if there is no uses of a*b and c*d
define double @fsub3(double %a, double %b, double %c, double %d)  {
; CHECK-LABEL: fsub3:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    xsmuldp 0, 4, 3
; CHECK-NEXT:    xsmsubadp 0, 2, 1
; CHECK-NEXT:    fmr 1, 0
; CHECK-NEXT:    blr
entry:
  %mul = fmul fast double %b, %a
  %mul1 = fmul fast double %d, %c
  %sub = fsub fast double %mul, %mul1
  ret double %sub
}

; Verify that the fold of a*b+c*d respect the uses of a*b
define double @fadd1(double %a, double %b, double %c, double %d)  {
; CHECK-LABEL: fadd1:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    xsmuldp 0, 2, 1
; CHECK-NEXT:    fmr 1, 0
; CHECK-NEXT:    xsmaddadp 1, 4, 3
; CHECK-NEXT:    xsmuldp 1, 0, 1
; CHECK-NEXT:    blr
entry:
  %mul = fmul fast double %b, %a
  %mul1 = fmul fast double %d, %c
  %add = fadd fast double %mul1, %mul
  %mul3 = fmul fast double %mul, %add
  ret double %mul3
}

; Verify that the fold of a*b+c*d respect the uses of c*d
define double @fadd2(double %a, double %b, double %c, double %d)  {
; CHECK-LABEL: fadd2:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    xsmuldp 0, 4, 3
; CHECK-NEXT:    fmr 3, 0
; CHECK-NEXT:    xsmaddadp 3, 2, 1
; CHECK-NEXT:    xsmuldp 1, 0, 3
; CHECK-NEXT:    blr
entry:
  %mul = fmul fast double %b, %a
  %mul1 = fmul fast double %d, %c
  %add = fadd fast double %mul1, %mul
  %mul3 = fmul fast double %mul1, %add
  ret double %mul3
}

; Verify that the fold of a*b+c*d if there is no uses of a*b and c*d
define double @fadd3(double %a, double %b, double %c, double %d)  {
; CHECK-LABEL: fadd3:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    xsmuldp 1, 2, 1
; CHECK-NEXT:    xsmaddadp 1, 4, 3
; CHECK-NEXT:    blr
entry:
  %mul = fmul fast double %b, %a
  %mul1 = fmul fast double %d, %c
  %add = fadd fast double %mul1, %mul
  ret double %add
}

define double @fma_multi_uses1(double %a, double %b, double %c, double %d, double* %p1, double* %p2, double* %p3) {
; CHECK-LABEL: fma_multi_uses1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xsmuldp 5, 1, 2
; CHECK-NEXT:    xsmuldp 0, 3, 4
; CHECK-NEXT:    stfd 5, 0(7)
; CHECK-NEXT:    stfd 5, 0(8)
; CHECK-NEXT:    stfd 0, 0(9)
; CHECK-NEXT:    xsmsubadp 0, 1, 2
; CHECK-NEXT:    fmr 1, 0
; CHECK-NEXT:    blr
  %ab = fmul fast double %a, %b
  %cd = fmul fast double %c, %d
  store double %ab, double* %p1 ; extra use of %ab
  store double %ab, double* %p2 ; another extra use of %ab
  store double %cd, double* %p3 ; extra use of %cd
  %r = fsub fast double %ab, %cd
  ret double %r
}

define double @fma_multi_uses2(double %a, double %b, double %c, double %d, double* %p1, double* %p2, double* %p3) {
; CHECK-LABEL: fma_multi_uses2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xsmuldp 5, 1, 2
; CHECK-NEXT:    xsmuldp 0, 3, 4
; CHECK-NEXT:    stfd 5, 0(7)
; CHECK-NEXT:    stfd 0, 0(8)
; CHECK-NEXT:    stfd 0, 0(9)
; CHECK-NEXT:    xsmsubadp 0, 1, 2
; CHECK-NEXT:    fmr 1, 0
; CHECK-NEXT:    blr
  %ab = fmul fast double %a, %b
  %cd = fmul fast double %c, %d
  store double %ab, double* %p1 ; extra use of %ab
  store double %cd, double* %p2 ; extra use of %cd
  store double %cd, double* %p3 ; another extra use of %cd
  %r = fsub fast double %ab, %cd
  ret double %r
}

define double @fma_multi_uses3(double %a, double %b, double %c, double %d, double %f, double %g, double* %p1, double* %p2, double* %p3) {
; CHECK-LABEL: fma_multi_uses3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xsmuldp 0, 1, 2
; CHECK-NEXT:    xsmuldp 3, 3, 4
; CHECK-NEXT:    ld 3, 96(1)
; CHECK-NEXT:    stfd 0, 0(9)
; CHECK-NEXT:    stfd 0, 0(10)
; CHECK-NEXT:    fmr 0, 3
; CHECK-NEXT:    xsmsubadp 3, 1, 2
; CHECK-NEXT:    xsmsubadp 0, 5, 6
; CHECK-NEXT:    xsmuldp 4, 5, 6
; CHECK-NEXT:    stfd 4, 0(3)
; CHECK-NEXT:    xsadddp 1, 3, 0
; CHECK-NEXT:    blr
  %ab = fmul fast double %a, %b
  %cd = fmul fast double %c, %d
  %fg = fmul fast double %f, %g
  store double %ab, double* %p1 ; extra use of %ab
  store double %ab, double* %p2 ; another extra use of %ab
  store double %fg, double* %p3 ; extra use of %fg
  %q = fsub fast double %fg, %cd ; The uses of %cd reduce to 1 after %r is folded. 2 uses of %fg, fold %cd, remove def of %cd
  %r = fsub fast double %ab, %cd ; Fold %r before %q. 3 uses of %ab, 2 uses of %cd, fold %cd
  %add = fadd fast double %r, %q
  ret double %add
}
