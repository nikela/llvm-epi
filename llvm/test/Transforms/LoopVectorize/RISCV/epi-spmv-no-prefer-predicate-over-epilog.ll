; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -mtriple riscv64 -mattr +m,+a,+f,+d,+zepi -S -O3 -scalable-vectorization=only -riscv-v-vector-bits-min=64 -force-vector-interleave=1 < %s  -o - | FileCheck %s
; ModuleID = './spmv.c'
source_filename = "./spmv.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128-v128:128:128-v256:128:128-v512:128:128-v1024:128:128"
target triple = "riscv64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nounwind
define dso_local void @spmv(double* nocapture readonly %a, i64* nocapture readonly %ia, i64* nocapture readonly %ja, double* nocapture readonly %x, double* nocapture %y, i32 signext %nrows) local_unnamed_addr {
; CHECK-LABEL: @spmv(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP37:%.*]] = icmp sgt i32 [[NROWS:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP37]], label [[FOR_BODY_PREHEADER:%.*]], label [[FOR_END18:%.*]]
; CHECK:       for.body.preheader:
; CHECK-NEXT:    [[WIDE_TRIP_COUNT:%.*]] = zext i32 [[NROWS]] to i64
; CHECK-NEXT:    [[DOTPRE:%.*]] = load i64, ptr [[IA:%.*]], align 8
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[TMP0:%.*]] = phi i64 [ [[DOTPRE]], [[FOR_BODY_PREHEADER]] ], [ [[TMP1:%.*]], [[FOR_END:%.*]] ]
; CHECK-NEXT:    [[INDVARS_IV39:%.*]] = phi i64 [ 0, [[FOR_BODY_PREHEADER]] ], [ [[INDVARS_IV_NEXT40:%.*]], [[FOR_END]] ]
; CHECK-NEXT:    [[SEXT:%.*]] = shl i64 [[TMP0]], 32
; CHECK-NEXT:    [[CONV231:%.*]] = ashr exact i64 [[SEXT]], 32
; CHECK-NEXT:    [[INDVARS_IV_NEXT40]] = add nuw nsw i64 [[INDVARS_IV39]], 1
; CHECK-NEXT:    [[ARRAYIDX4:%.*]] = getelementptr inbounds i64, ptr [[IA]], i64 [[INDVARS_IV_NEXT40]]
; CHECK-NEXT:    [[TMP1]] = load i64, ptr [[ARRAYIDX4]], align 8
; CHECK-NEXT:    [[CMP532:%.*]] = icmp sgt i64 [[TMP1]], [[CONV231]]
; CHECK-NEXT:    br i1 [[CMP532]], label [[FOR_BODY7_PREHEADER:%.*]], label [[FOR_END]]
; CHECK:       for.body7.preheader:
; CHECK-NEXT:    [[TMP2:%.*]] = sub i64 [[TMP1]], [[CONV231]]
; CHECK-NEXT:    [[TMP3:%.*]] = tail call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[TMP2]], [[TMP3]]
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[FOR_BODY7_PREHEADER3:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP4:%.*]] = tail call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[TMP2]], [[TMP4]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[TMP2]], [[N_MOD_VF]]
; CHECK-NEXT:    [[IND_END:%.*]] = add i64 [[CONV231]], [[N_VEC]]
; CHECK-NEXT:    [[TMP5:%.*]] = tail call i64 @llvm.vscale.i64()
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 1 x double> [ zeroinitializer, [[VECTOR_PH]] ], [ [[TMP10:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[OFFSET_IDX:%.*]] = add i64 [[CONV231]], [[INDEX]]
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds double, ptr [[A:%.*]], i64 [[OFFSET_IDX]]
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <vscale x 1 x double>, ptr [[TMP6]], align 8
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i64, ptr [[JA:%.*]], i64 [[OFFSET_IDX]]
; CHECK-NEXT:    [[WIDE_LOAD2:%.*]] = load <vscale x 1 x i64>, ptr [[TMP7]], align 8
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds double, ptr [[X:%.*]], <vscale x 1 x i64> [[WIDE_LOAD2]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = tail call <vscale x 1 x double> @llvm.masked.gather.nxv1f64.nxv1p0(<vscale x 1 x ptr> [[TMP8]], i32 8, <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> poison, i1 true, i64 0), <vscale x 1 x i1> poison, <vscale x 1 x i32> zeroinitializer), <vscale x 1 x double> poison)
; CHECK-NEXT:    [[TMP9:%.*]] = fmul fast <vscale x 1 x double> [[WIDE_MASKED_GATHER]], [[WIDE_LOAD]]
; CHECK-NEXT:    [[TMP10]] = fadd fast <vscale x 1 x double> [[TMP9]], [[VEC_PHI]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP5]]
; CHECK-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[TMP12:%.*]] = tail call fast double @llvm.vector.reduce.fadd.nxv1f64(double -0.000000e+00, <vscale x 1 x double> [[TMP10]])
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[N_MOD_VF]], 0
; CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_END]], label [[FOR_BODY7_PREHEADER3]]
; CHECK:       for.body7.preheader3:
; CHECK-NEXT:    [[INDVARS_IV_PH:%.*]] = phi i64 [ [[CONV231]], [[FOR_BODY7_PREHEADER]] ], [ [[IND_END]], [[MIDDLE_BLOCK]] ]
; CHECK-NEXT:    [[SUM_034_PH:%.*]] = phi double [ 0.000000e+00, [[FOR_BODY7_PREHEADER]] ], [ [[TMP12]], [[MIDDLE_BLOCK]] ]
; CHECK-NEXT:    br label [[FOR_BODY7:%.*]]
; CHECK:       for.body7:
; CHECK-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY7]] ], [ [[INDVARS_IV_PH]], [[FOR_BODY7_PREHEADER3]] ]
; CHECK-NEXT:    [[SUM_034:%.*]] = phi double [ [[ADD13:%.*]], [[FOR_BODY7]] ], [ [[SUM_034_PH]], [[FOR_BODY7_PREHEADER3]] ]
; CHECK-NEXT:    [[ARRAYIDX9:%.*]] = getelementptr inbounds double, ptr [[A]], i64 [[INDVARS_IV]]
; CHECK-NEXT:    [[TMP13:%.*]] = load double, ptr [[ARRAYIDX9]], align 8
; CHECK-NEXT:    [[ARRAYIDX11:%.*]] = getelementptr inbounds i64, ptr [[JA]], i64 [[INDVARS_IV]]
; CHECK-NEXT:    [[TMP14:%.*]] = load i64, ptr [[ARRAYIDX11]], align 8
; CHECK-NEXT:    [[ARRAYIDX12:%.*]] = getelementptr inbounds double, ptr [[X]], i64 [[TMP14]]
; CHECK-NEXT:    [[TMP15:%.*]] = load double, ptr [[ARRAYIDX12]], align 8
; CHECK-NEXT:    [[MUL:%.*]] = fmul fast double [[TMP15]], [[TMP13]]
; CHECK-NEXT:    [[ADD13]] = fadd fast double [[MUL]], [[SUM_034]]
; CHECK-NEXT:    [[INDVARS_IV_NEXT]] = add nsw i64 [[INDVARS_IV]], 1
; CHECK-NEXT:    [[EXITCOND_NOT1:%.*]] = icmp eq i64 [[INDVARS_IV_NEXT]], [[TMP1]]
; CHECK-NEXT:    br i1 [[EXITCOND_NOT1]], label [[FOR_END]], label [[FOR_BODY7]], !llvm.loop [[LOOP3:![0-9]+]]
; CHECK:       for.end:
; CHECK-NEXT:    [[SUM_0_LCSSA:%.*]] = phi double [ 0.000000e+00, [[FOR_BODY]] ], [ [[TMP12]], [[MIDDLE_BLOCK]] ], [ [[ADD13]], [[FOR_BODY7]] ]
; CHECK-NEXT:    [[ARRAYIDX15:%.*]] = getelementptr inbounds double, ptr [[Y:%.*]], i64 [[INDVARS_IV39]]
; CHECK-NEXT:    store double [[SUM_0_LCSSA]], ptr [[ARRAYIDX15]], align 8
; CHECK-NEXT:    [[EXITCOND_NOT:%.*]] = icmp eq i64 [[INDVARS_IV_NEXT40]], [[WIDE_TRIP_COUNT]]
; CHECK-NEXT:    br i1 [[EXITCOND_NOT]], label [[FOR_END18]], label [[FOR_BODY]]
; CHECK:       for.end18:
; CHECK-NEXT:    ret void
;
entry:
  %cmp37 = icmp sgt i32 %nrows, 0
  br i1 %cmp37, label %for.body.preheader, label %for.end18

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %nrows to i64
  %.pre = load i64, i64* %ia, align 8
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.end
  %0 = phi i64 [ %.pre, %for.body.preheader ], [ %1, %for.end ]
  %indvars.iv39 = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next40, %for.end ]
  %sext = shl i64 %0, 32
  %conv231 = ashr exact i64 %sext, 32
  %indvars.iv.next40 = add nuw nsw i64 %indvars.iv39, 1
  %arrayidx4 = getelementptr inbounds i64, i64* %ia, i64 %indvars.iv.next40
  %1 = load i64, i64* %arrayidx4, align 8
  %cmp532 = icmp sgt i64 %1, %conv231
  br i1 %cmp532, label %for.body7.lr.ph, label %for.end

for.body7.lr.ph:                                  ; preds = %for.body
  %sext41 = shl i64 %0, 32
  %2 = ashr exact i64 %sext41, 32
  br label %for.body7

for.body7:                                        ; preds = %for.body7.lr.ph, %for.body7
  %indvars.iv = phi i64 [ %2, %for.body7.lr.ph ], [ %indvars.iv.next, %for.body7 ]
  %sum.034 = phi double [ 0.000000e+00, %for.body7.lr.ph ], [ %add13, %for.body7 ]
  %arrayidx9 = getelementptr inbounds double, double* %a, i64 %indvars.iv
  %3 = load double, double* %arrayidx9, align 8
  %arrayidx11 = getelementptr inbounds i64, i64* %ja, i64 %indvars.iv
  %4 = load i64, i64* %arrayidx11, align 8
  %arrayidx12 = getelementptr inbounds double, double* %x, i64 %4
  %5 = load double, double* %arrayidx12, align 8
  %mul = fmul fast double %5, %3
  %add13 = fadd fast double %mul, %sum.034
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %cmp5 = icmp sgt i64 %1, %indvars.iv.next
  br i1 %cmp5, label %for.body7, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body7
  %add13.lcssa = phi double [ %add13, %for.body7 ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %for.body
  %sum.0.lcssa = phi double [ 0.000000e+00, %for.body ], [ %add13.lcssa, %for.end.loopexit ]
  %arrayidx15 = getelementptr inbounds double, double* %y, i64 %indvars.iv39
  store double %sum.0.lcssa, double* %arrayidx15, align 8
  %exitcond.not = icmp eq i64 %indvars.iv.next40, %wide.trip.count
  br i1 %exitcond.not, label %for.end18.loopexit, label %for.body

for.end18.loopexit:                               ; preds = %for.end
  br label %for.end18

for.end18:                                        ; preds = %for.end18.loopexit, %entry
  ret void
}
