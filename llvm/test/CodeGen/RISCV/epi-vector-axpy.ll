; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple riscv64 -mattr=+m,+a,+f,+d,+epi < %s | FileCheck %s

define dso_local void @saxpy(i32 signext %N, float* noalias nocapture %y, float* noalias nocapture readonly %x, float %alpha) nounwind {
; CHECK-LABEL: saxpy:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addi a4, zero, 1
; CHECK-NEXT:    blt a0, a4, .LBB0_8
; CHECK-NEXT:  # %bb.1: # %for.body.preheader
; CHECK-NEXT:    fmv.w.x ft0, a3
; CHECK-NEXT:    slli a0, a0, 32
; CHECK-NEXT:    srli a7, a0, 32
; CHECK-NEXT:    vsetvli a0, zero, e64, m1
; CHECK-NEXT:    slli a0, a0, 1
; CHECK-NEXT:    bgeu a7, a0, .LBB0_3
; CHECK-NEXT:  # %bb.2:
; CHECK-NEXT:    mv t1, zero
; CHECK-NEXT:    j .LBB0_6
; CHECK-NEXT:  .LBB0_3: # %vector.ph
; CHECK-NEXT:    vsetvli a0, zero, e64, m1
; CHECK-NEXT:    slli t0, a0, 3
; CHECK-NEXT:    vsetvli a3, zero, e32, m1
; CHECK-NEXT:    vfmv.v.f v0, ft0
; CHECK-NEXT:    slli a5, a0, 1
; CHECK-NEXT:    remu a6, a7, a5
; CHECK-NEXT:    sub t1, a7, a6
; CHECK-NEXT:    mv a0, zero
; CHECK-NEXT:    mv a4, zero
; CHECK-NEXT:  .LBB0_4: # %vector.body
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    add a3, a2, a0
; CHECK-NEXT:    vle.v v1, (a3)
; CHECK-NEXT:    add a3, a1, a0
; CHECK-NEXT:    vle.v v2, (a3)
; CHECK-NEXT:    vfmadd.vv v1, v0, v2
; CHECK-NEXT:    vse.v v1, (a3)
; CHECK-NEXT:    add a0, a0, t0
; CHECK-NEXT:    add a4, a4, a5
; CHECK-NEXT:    bne a4, t1, .LBB0_4
; CHECK-NEXT:  # %bb.5: # %middle.block
; CHECK-NEXT:    beqz a6, .LBB0_8
; CHECK-NEXT:  .LBB0_6: # %for.body.preheader17
; CHECK-NEXT:    sub a0, a7, t1
; CHECK-NEXT:    slli a3, t1, 2
; CHECK-NEXT:    add a2, a2, a3
; CHECK-NEXT:    add a1, a1, a3
; CHECK-NEXT:  .LBB0_7: # %for.body
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    flw ft1, 0(a1)
; CHECK-NEXT:    flw ft2, 0(a2)
; CHECK-NEXT:    fmadd.s ft1, ft2, ft0, ft1
; CHECK-NEXT:    fsw ft1, 0(a1)
; CHECK-NEXT:    addi a2, a2, 4
; CHECK-NEXT:    addi a1, a1, 4
; CHECK-NEXT:    addi a0, a0, -1
; CHECK-NEXT:    bnez a0, .LBB0_7
; CHECK-NEXT:  .LBB0_8: # %for.cond.cleanup
; CHECK-NEXT:    ret
entry:
  %cmp11 = icmp sgt i32 %N, 0
  br i1 %cmp11, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  %0 = call i64 @llvm.experimental.vector.vscale.i64()
  %step.vscale = shl i64 %0, 1
  %min.iters.check = icmp ugt i64 %step.vscale, %wide.trip.count
  br i1 %min.iters.check, label %for.body.preheader17, label %vector.ph

for.body.preheader17:                             ; preds = %middle.block, %for.body.preheader
  %indvars.iv.ph = phi i64 [ 0, %for.body.preheader ], [ %n.vec, %middle.block ]
  br label %for.body

vector.ph:                                        ; preds = %for.body.preheader
  %1 = call i64 @llvm.experimental.vector.vscale.i64()
  %step.vscale13 = shl i64 %1, 1
  %n.mod.vf = urem i64 %wide.trip.count, %step.vscale13
  %n.vec = sub nsw i64 %wide.trip.count, %n.mod.vf
  %broadcast.splatinsert14 = insertelement <vscale x 2 x float> undef, float %alpha, i32 0
  %broadcast.splat15 = shufflevector <vscale x 2 x float> %broadcast.splatinsert14, <vscale x 2 x float> undef, <vscale x 2 x i32> zeroinitializer
  %2 = call i64 @llvm.experimental.vector.vscale.i64()
  %index.vscale = shl i64 %2, 1
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %3 = getelementptr inbounds float, float* %x, i64 %index
  %4 = bitcast float* %3 to <vscale x 2 x float>*
  %wide.load = load <vscale x 2 x float>, <vscale x 2 x float>* %4, align 4
  %5 = fmul fast <vscale x 2 x float> %wide.load, %broadcast.splat15
  %6 = getelementptr inbounds float, float* %y, i64 %index
  %7 = bitcast float* %6 to <vscale x 2 x float>*
  %wide.load16 = load <vscale x 2 x float>, <vscale x 2 x float>* %7, align 4
  %8 = fadd fast <vscale x 2 x float> %5, %wide.load16
  %9 = bitcast float* %6 to <vscale x 2 x float>*
  store <vscale x 2 x float> %8, <vscale x 2 x float>* %9, align 4
  %index.next = add i64 %index, %index.vscale
  %10 = icmp eq i64 %index.next, %n.vec
  br i1 %10, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %n.mod.vf, 0
  br i1 %cmp.n, label %for.cond.cleanup, label %for.body.preheader17

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader17, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ %indvars.iv.ph, %for.body.preheader17 ]
  %arrayidx = getelementptr inbounds float, float* %x, i64 %indvars.iv
  %11 = load float, float* %arrayidx, align 4
  %mul = fmul fast float %11, %alpha
  %arrayidx2 = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %12 = load float, float* %arrayidx2, align 4
  %add = fadd fast float %mul, %12
  store float %add, float* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define dso_local void @daxpy(i32 signext %N, double* noalias nocapture %y, double* noalias nocapture readonly %x, double %alpha) nounwind {
; CHECK-LABEL: daxpy:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addi a4, zero, 1
; CHECK-NEXT:    blt a0, a4, .LBB1_8
; CHECK-NEXT:  # %bb.1: # %for.body.preheader
; CHECK-NEXT:    fmv.d.x ft0, a3
; CHECK-NEXT:    slli a0, a0, 32
; CHECK-NEXT:    srli a7, a0, 32
; CHECK-NEXT:    vsetvli a0, zero, e64, m1
; CHECK-NEXT:    bgeu a7, a0, .LBB1_3
; CHECK-NEXT:  # %bb.2:
; CHECK-NEXT:    mv t1, zero
; CHECK-NEXT:    j .LBB1_6
; CHECK-NEXT:  .LBB1_3: # %vector.ph
; CHECK-NEXT:    vsetvli a5, zero, e64, m1
; CHECK-NEXT:    slli t0, a5, 3
; CHECK-NEXT:    vfmv.v.f v0, ft0
; CHECK-NEXT:    remu a6, a7, a5
; CHECK-NEXT:    sub t1, a7, a6
; CHECK-NEXT:    mv a0, zero
; CHECK-NEXT:    mv a4, zero
; CHECK-NEXT:  .LBB1_4: # %vector.body
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    add a3, a2, a0
; CHECK-NEXT:    vle.v v1, (a3)
; CHECK-NEXT:    add a3, a1, a0
; CHECK-NEXT:    vle.v v2, (a3)
; CHECK-NEXT:    vfmadd.vv v1, v0, v2
; CHECK-NEXT:    vse.v v1, (a3)
; CHECK-NEXT:    add a0, a0, t0
; CHECK-NEXT:    add a4, a4, a5
; CHECK-NEXT:    bne a4, t1, .LBB1_4
; CHECK-NEXT:  # %bb.5: # %middle.block
; CHECK-NEXT:    beqz a6, .LBB1_8
; CHECK-NEXT:  .LBB1_6: # %for.body.preheader17
; CHECK-NEXT:    sub a0, a7, t1
; CHECK-NEXT:    slli a3, t1, 3
; CHECK-NEXT:    add a2, a2, a3
; CHECK-NEXT:    add a1, a1, a3
; CHECK-NEXT:  .LBB1_7: # %for.body
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    fld ft1, 0(a1)
; CHECK-NEXT:    fld ft2, 0(a2)
; CHECK-NEXT:    fmadd.d ft1, ft2, ft0, ft1
; CHECK-NEXT:    fsd ft1, 0(a1)
; CHECK-NEXT:    addi a2, a2, 8
; CHECK-NEXT:    addi a1, a1, 8
; CHECK-NEXT:    addi a0, a0, -1
; CHECK-NEXT:    bnez a0, .LBB1_7
; CHECK-NEXT:  .LBB1_8: # %for.cond.cleanup
; CHECK-NEXT:    ret
entry:
  %cmp11 = icmp sgt i32 %N, 0
  br i1 %cmp11, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  %0 = call i64 @llvm.experimental.vector.vscale.i64()
  %min.iters.check = icmp ugt i64 %0, %wide.trip.count
  br i1 %min.iters.check, label %for.body.preheader17, label %vector.ph

for.body.preheader17:                             ; preds = %middle.block, %for.body.preheader
  %indvars.iv.ph = phi i64 [ 0, %for.body.preheader ], [ %n.vec, %middle.block ]
  br label %for.body

vector.ph:                                        ; preds = %for.body.preheader
  %1 = call i64 @llvm.experimental.vector.vscale.i64()
  %n.mod.vf = urem i64 %wide.trip.count, %1
  %n.vec = sub nsw i64 %wide.trip.count, %n.mod.vf
  %broadcast.splatinsert14 = insertelement <vscale x 1 x double> undef, double %alpha, i32 0
  %broadcast.splat15 = shufflevector <vscale x 1 x double> %broadcast.splatinsert14, <vscale x 1 x double> undef, <vscale x 1 x i32> zeroinitializer
  %2 = call i64 @llvm.experimental.vector.vscale.i64()
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %3 = getelementptr inbounds double, double* %x, i64 %index
  %4 = bitcast double* %3 to <vscale x 1 x double>*
  %wide.load = load <vscale x 1 x double>, <vscale x 1 x double>* %4, align 8
  %5 = fmul fast <vscale x 1 x double> %wide.load, %broadcast.splat15
  %6 = getelementptr inbounds double, double* %y, i64 %index
  %7 = bitcast double* %6 to <vscale x 1 x double>*
  %wide.load16 = load <vscale x 1 x double>, <vscale x 1 x double>* %7, align 8
  %8 = fadd fast <vscale x 1 x double> %5, %wide.load16
  %9 = bitcast double* %6 to <vscale x 1 x double>*
  store <vscale x 1 x double> %8, <vscale x 1 x double>* %9, align 8
  %index.next = add i64 %index, %2
  %10 = icmp eq i64 %index.next, %n.vec
  br i1 %10, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %n.mod.vf, 0
  br i1 %cmp.n, label %for.cond.cleanup, label %for.body.preheader17

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader17, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ %indvars.iv.ph, %for.body.preheader17 ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %11 = load double, double* %arrayidx, align 8
  %mul = fmul fast double %11, %alpha
  %arrayidx2 = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %12 = load double, double* %arrayidx2, align 8
  %add = fadd fast double %mul, %12
  store double %add, double* %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nounwind readnone
declare i64 @llvm.experimental.vector.vscale.i64()

