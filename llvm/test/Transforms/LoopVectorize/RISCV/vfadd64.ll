; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -mtriple riscv64 -mattr +m,+a,+f,+d,+epi -S -loop-vectorize < %s  -o - \
; RUN:     | FileCheck %s
; ModuleID = 'vfadd64.c'
source_filename = "vfadd64.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nounwind
define dso_local void @vfadd64(double* noalias nocapture %dz, double* noalias nocapture readonly %dx, double* noalias nocapture readonly %dy, i32 signext %n) local_unnamed_addr #0 {
; CHECK-LABEL: @vfadd64(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP10:%.*]] = icmp sgt i32 [[N:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP10]], label [[FOR_BODY_PREHEADER:%.*]], label [[FOR_END:%.*]]
; CHECK:       for.body.preheader:
; CHECK-NEXT:    [[WIDE_TRIP_COUNT:%.*]] = zext i32 [[N]] to i64
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.experimental.vector.vscale.i64()
; CHECK-NEXT:    [[STEP_VSCALE:%.*]] = mul i64 [[TMP0]], 1
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[WIDE_TRIP_COUNT]], [[STEP_VSCALE]]
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP1:%.*]] = call i64 @llvm.experimental.vector.vscale.i64()
; CHECK-NEXT:    [[STEP_VSCALE1:%.*]] = mul i64 1, [[TMP1]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[WIDE_TRIP_COUNT]], [[STEP_VSCALE1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[N_MOD_VF]]
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 1 x i64> undef, i64 [[INDEX]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 1 x i64> [[BROADCAST_SPLATINSERT]], <vscale x 1 x i64> undef, <vscale x 1 x i32> zeroinitializer
; CHECK-NEXT:    [[STEPVEC_BASE:%.*]] = call <vscale x 1 x i64> @llvm.experimental.vector.stepvector.nxv1i64()
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.experimental.vector.vscale.i64()
; CHECK-NEXT:    [[STARTIDX_VSCALE:%.*]] = mul i64 [[TMP2]], 0
; CHECK-NEXT:    [[STARTINDEX_SPLATINSERT:%.*]] = insertelement <vscale x 1 x i64> undef, i64 [[STARTIDX_VSCALE]], i32 0
; CHECK-NEXT:    [[STARTINDEX_SPLAT:%.*]] = shufflevector <vscale x 1 x i64> [[STARTINDEX_SPLATINSERT]], <vscale x 1 x i64> undef, <vscale x 1 x i32> zeroinitializer
; CHECK-NEXT:    [[STEPVEC:%.*]] = add <vscale x 1 x i64> [[STEPVEC_BASE]], [[STARTINDEX_SPLAT]]
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 1 x i64> [[BROADCAST_SPLAT]], [[STEPVEC]]
; CHECK-NEXT:    [[TMP3:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds double, double* [[DX:%.*]], i64 [[TMP3]]
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds double, double* [[TMP4]], i32 0
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast double* [[TMP5]] to <vscale x 1 x double>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <vscale x 1 x double>, <vscale x 1 x double>* [[TMP6]], align 8, !tbaa !2
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds double, double* [[DY:%.*]], i64 [[TMP3]]
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds double, double* [[TMP7]], i32 0
; CHECK-NEXT:    [[TMP9:%.*]] = bitcast double* [[TMP8]] to <vscale x 1 x double>*
; CHECK-NEXT:    [[WIDE_LOAD2:%.*]] = load <vscale x 1 x double>, <vscale x 1 x double>* [[TMP9]], align 8, !tbaa !2
; CHECK-NEXT:    [[TMP10:%.*]] = fadd <vscale x 1 x double> [[WIDE_LOAD]], [[WIDE_LOAD2]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr inbounds double, double* [[DZ:%.*]], i64 [[TMP3]]
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr inbounds double, double* [[TMP11]], i32 0
; CHECK-NEXT:    [[TMP13:%.*]] = bitcast double* [[TMP12]] to <vscale x 1 x double>*
; CHECK-NEXT:    store <vscale x 1 x double> [[TMP10]], <vscale x 1 x double>* [[TMP13]], align 8, !tbaa !2
; CHECK-NEXT:    [[TMP14:%.*]] = call i64 @llvm.experimental.vector.vscale.i64()
; CHECK-NEXT:    [[INDEX_VSCALE:%.*]] = mul i64 [[TMP14]], 1
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[INDEX_VSCALE]]
; CHECK-NEXT:    [[TMP15:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP15]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop !6
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[WIDE_TRIP_COUNT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_END_LOOPEXIT:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[FOR_BODY_PREHEADER]] ]
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ]
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds double, double* [[DX]], i64 [[INDVARS_IV]]
; CHECK-NEXT:    [[TMP16:%.*]] = load double, double* [[ARRAYIDX]], align 8, !tbaa !2
; CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds double, double* [[DY]], i64 [[INDVARS_IV]]
; CHECK-NEXT:    [[TMP17:%.*]] = load double, double* [[ARRAYIDX2]], align 8, !tbaa !2
; CHECK-NEXT:    [[ADD:%.*]] = fadd double [[TMP16]], [[TMP17]]
; CHECK-NEXT:    [[ARRAYIDX4:%.*]] = getelementptr inbounds double, double* [[DZ]], i64 [[INDVARS_IV]]
; CHECK-NEXT:    store double [[ADD]], double* [[ARRAYIDX4]], align 8, !tbaa !2
; CHECK-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; CHECK-NEXT:    [[EXITCOND:%.*]] = icmp eq i64 [[INDVARS_IV_NEXT]], [[WIDE_TRIP_COUNT]]
; CHECK-NEXT:    br i1 [[EXITCOND]], label [[FOR_END_LOOPEXIT]], label [[FOR_BODY]], !llvm.loop !8
; CHECK:       for.end.loopexit:
; CHECK-NEXT:    br label [[FOR_END]]
; CHECK:       for.end:
; CHECK-NEXT:    ret void
;
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %dx, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8, !tbaa !2
  %arrayidx2 = getelementptr inbounds double, double* %dy, i64 %indvars.iv
  %1 = load double, double* %arrayidx2, align 8, !tbaa !2
  %add = fadd double %0, %1
  %arrayidx4 = getelementptr inbounds double, double* %dz, i64 %indvars.iv
  store double %add, double* %arrayidx4, align 8, !tbaa !2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

attributes #0 = { nofree norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+d,+epi,+f,+m,-relax" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 (git@repo.hca.bsc.es:EPI/System-Software/llvm-mono.git 6c96ed45a7b0120200aea55d2099a8cf001c674b)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
