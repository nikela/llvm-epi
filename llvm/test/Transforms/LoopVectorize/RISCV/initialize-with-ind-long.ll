; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -mtriple riscv64 -mattr +m,+a,+f,+d,+epi -S -loop-vectorize < %s  -o - \
; RUN:     | FileCheck %s
; ModuleID = 't3.i'
source_filename = "t3.i"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define dso_local signext i32 @main(i32 signext %argc, i8** nocapture readnone %argv) local_unnamed_addr #0 {
; CHECK-LABEL: @main(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca [1024 x i64], align 8
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast [1024 x i64]* [[A]] to i8*
; CHECK-NEXT:    call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull [[TMP0]]) #3
; CHECK-NEXT:    [[TMP1:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[STEP_VSCALE:%.*]] = mul i64 [[TMP1]], 1
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 1024, [[STEP_VSCALE]]
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[STEP_VSCALE1:%.*]] = mul i64 1, [[TMP2]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 1024, [[STEP_VSCALE1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 1024, [[N_MOD_VF]]
; CHECK-NEXT:    [[STEPVEC_BASE:%.*]] = call <vscale x 1 x i64> @llvm.experimental.vector.stepvector.nxv1i64()
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 1 x i64> shufflevector (<vscale x 1 x i64> insertelement (<vscale x 1 x i64> undef, i64 0, i32 0), <vscale x 1 x i64> undef, <vscale x 1 x i32> zeroinitializer), [[STEPVEC_BASE]]
; CHECK-NEXT:    [[TMP3:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP4:%.*]] = mul i64 1, [[TMP3]]
; CHECK-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <vscale x 1 x i64> undef, i64 [[TMP4]], i32 0
; CHECK-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <vscale x 1 x i64> [[DOTSPLATINSERT]], <vscale x 1 x i64> undef, <vscale x 1 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <vscale x 1 x i64> [ [[INDUCTION]], [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP5:%.*]] = add nuw nsw <vscale x 1 x i64> [[VEC_IND]], shufflevector (<vscale x 1 x i64> insertelement (<vscale x 1 x i64> undef, i64 42, i32 0), <vscale x 1 x i64> undef, <vscale x 1 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP6:%.*]] = extractelement <vscale x 1 x i64> [[VEC_IND]], i32 0
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds [1024 x i64], [1024 x i64]* [[A]], i64 0, i64 [[TMP6]]
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i64, i64* [[TMP7]], i32 0
; CHECK-NEXT:    [[TMP9:%.*]] = bitcast i64* [[TMP8]] to <vscale x 1 x i64>*
; CHECK-NEXT:    store <vscale x 1 x i64> [[TMP5]], <vscale x 1 x i64>* [[TMP9]], align 8, !tbaa !2
; CHECK-NEXT:    [[TMP10:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[INDEX_VSCALE:%.*]] = mul i64 [[TMP10]], 1
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[INDEX_VSCALE]]
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <vscale x 1 x i64> [[VEC_IND]], [[DOTSPLAT]]
; CHECK-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop !6
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 1024, [[N_VEC]]
; CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_COND_CLEANUP:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[ENTRY:%.*]] ]
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    [[IDXPROM:%.*]] = sext i32 [[ARGC:%.*]] to i64
; CHECK-NEXT:    [[ARRAYIDX1:%.*]] = getelementptr inbounds [1024 x i64], [1024 x i64]* [[A]], i64 0, i64 [[IDXPROM]]
; CHECK-NEXT:    [[TMP12:%.*]] = load i64, i64* [[ARRAYIDX1]], align 8, !tbaa !2
; CHECK-NEXT:    [[CONV:%.*]] = trunc i64 [[TMP12]] to i32
; CHECK-NEXT:    call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull [[TMP0]]) #3
; CHECK-NEXT:    ret i32 [[CONV]]
; CHECK:       for.body:
; CHECK-NEXT:    [[I_07:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], [ [[INC:%.*]], [[FOR_BODY]] ]
; CHECK-NEXT:    [[ADD:%.*]] = add nuw nsw i64 [[I_07]], 42
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [1024 x i64], [1024 x i64]* [[A]], i64 0, i64 [[I_07]]
; CHECK-NEXT:    store i64 [[ADD]], i64* [[ARRAYIDX]], align 8, !tbaa !2
; CHECK-NEXT:    [[INC]] = add nuw nsw i64 [[I_07]], 1
; CHECK-NEXT:    [[EXITCOND:%.*]] = icmp eq i64 [[INC]], 1024
; CHECK-NEXT:    br i1 [[EXITCOND]], label [[FOR_COND_CLEANUP]], label [[FOR_BODY]], !llvm.loop !8
;
entry:
  %a = alloca [1024 x i64], align 8
  %0 = bitcast [1024 x i64]* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull %0) #2
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %idxprom = sext i32 %argc to i64
  %arrayidx1 = getelementptr inbounds [1024 x i64], [1024 x i64]* %a, i64 0, i64 %idxprom
  %1 = load i64, i64* %arrayidx1, align 8, !tbaa !2
  %conv = trunc i64 %1 to i32
  call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull %0) #2
  ret i32 %conv

for.body:                                         ; preds = %for.body, %entry
  %i.07 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %add = add nuw nsw i64 %i.07, 42
  %arrayidx = getelementptr inbounds [1024 x i64], [1024 x i64]* %a, i64 0, i64 %i.07
  store i64 %add, i64* %arrayidx, align 8, !tbaa !2
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond = icmp eq i64 %inc, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+d,+epi,+f,+m,-relax" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 (git@repo.hca.bsc.es:EPI/System-Software/llvm-mono.git 823a818e0f44dc9b594f14a328ec52b247f3611a)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"long", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
