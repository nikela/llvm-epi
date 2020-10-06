; RUN: opt < %s -mtriple riscv64 -mattr +m,+a,+f,+d,+experimental-v \
; RUN:   -prefer-predicate-over-epilogue=predicate-dont-vectorize -S \
; RUN:   -loop-vectorize -debug-only=loop-vectorize 2>&1 \
; RUN:   | FileCheck %s

;Check that register usage for VF=8 is >32, thus selecting MaxVF=4.
;void highRegUage(int N, double *restrict c, double *restrict a,
;                 double *restrict b, double *restrict e, double *restrict f,
;                 double *restrict g) {
;  int i;
;  for (i = 1; i < N + 1; i++) {
;    c[i] = a[i] + b[i];
;    e[i] = a[i] - b[i] + c[i];
;    f[i] = a[i] * b[i] + c[i] - e[i];
;    g[i] = a[i] * c[i] + b[i] + e[i] - f[i];
;  }
;}

; CHECK-LABEL: highRegUage
; CHECK: LV: The Smallest and Widest types: 64 / 64 bits.
; CHECK: LV: The Widest register safe to use is: 512 bits.
; CHECK: LV(REG): Calculating max register usage:
; CHECK: LV(REG): VF = vscale x 2
; CHECK-NEXT: LV(REG): Found max usage: 2 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 2 registers
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 10 registers
; CHECK-NEXT: LV(REG): Found invariant usage: 1 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 2 registers
; CHECK-NEXT: LV(REG): VF = vscale x 4
; CHECK-NEXT: LV(REG): Found max usage: 2 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 2 registers
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 20 registers
; CHECK-NEXT: LV(REG): Found invariant usage: 1 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 4 registers
; CHECK-NEXT: LV(REG): VF = vscale x 8
; CHECK-NEXT: LV(REG): Found max usage: 2 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::ScalarRC, 2 registers
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 40 registers
; CHECK-NEXT: LV(REG): Found invariant usage: 1 item
; CHECK-NEXT: LV(REG): RegisterClass: Generic::VectorRC, 8 registers
; CHECK: LV: Selecting VF: 4.
; 
; ModuleID = './vec-reg-usage.c'
source_filename = "./vec-reg-usage.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128-v128:128:128-v256:128:128-v512:128:128-v1024:128:128"
target triple = "riscv64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nounwind
define dso_local void @highRegUage(i32 signext %N, double* noalias nocapture %c, double* noalias nocapture readonly %a, double* noalias nocapture readonly %b, double* noalias nocapture %e, double* noalias nocapture %f, double* noalias nocapture %g) local_unnamed_addr {
entry:
  %cmp.not75 = icmp slt i32 %N, 1
  br i1 %cmp.not75, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %0 = add nuw i32 %N, 1
  %wide.trip.count = zext i32 %0 to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 1, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %a, i64 %indvars.iv
  %1 = load double, double* %arrayidx, align 8
  %arrayidx2 = getelementptr inbounds double, double* %b, i64 %indvars.iv
  %2 = load double, double* %arrayidx2, align 8
  %add3 = fadd double %1, %2
  %arrayidx5 = getelementptr inbounds double, double* %c, i64 %indvars.iv
  store double %add3, double* %arrayidx5, align 8
  %sub = fsub double %1, %2
  %add12 = fadd double %sub, %add3
  %arrayidx14 = getelementptr inbounds double, double* %e, i64 %indvars.iv
  store double %add12, double* %arrayidx14, align 8
  %mul = fmul double %1, %2
  %add21 = fadd double %mul, %add3
  %sub24 = fsub double %add21, %add12
  %arrayidx26 = getelementptr inbounds double, double* %f, i64 %indvars.iv
  store double %sub24, double* %arrayidx26, align 8
  %mul31 = fmul double %1, %add3
  %add34 = fadd double %2, %mul31
  %add37 = fadd double %add12, %add34
  %sub40 = fsub double %add37, %sub24
  %arrayidx42 = getelementptr inbounds double, double* %g, i64 %indvars.iv
  store double %sub40, double* %arrayidx42, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}
