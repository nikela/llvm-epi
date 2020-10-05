; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -mtriple riscv64 -mattr +m,+a,+f,+d,+experimental-v -S \
; RUN:    -loop-vectorize < %s -o - | FileCheck %s
; ModuleID = 't2.i'
source_filename = "t2.i"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define dso_local signext i32 @main(i32 signext %argc, i8** nocapture readnone %argv) local_unnamed_addr #0 {
; CHECK-LABEL: @main(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca [1024 x double], align 8
; CHECK-NEXT:    [[B:%.*]] = alloca [1024 x double], align 8
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast [1024 x double]* [[A]] to i8*
; CHECK-NEXT:    call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull [[TMP0]]) [[ATTR4:#.*]]
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast [1024 x double]* [[B]] to i8*
; CHECK-NEXT:    call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull [[TMP1]]) [[ATTR4]]
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[STEP_VSCALE:%.*]] = mul i64 [[TMP2]], 8
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 1024, [[STEP_VSCALE]]
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP3:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[STEP_VSCALE1:%.*]] = mul i64 8, [[TMP3]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 1024, [[STEP_VSCALE1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 1024, [[N_MOD_VF]]
; CHECK-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <vscale x 8 x i64> undef, i64 0, i32 0
; CHECK-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <vscale x 8 x i64> [[DOTSPLATINSERT]], <vscale x 8 x i64> undef, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT:    [[STEPVEC_BASE:%.*]] = call <vscale x 8 x i64> @llvm.experimental.vector.stepvector.nxv8i64()
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 8 x i64> [[DOTSPLAT]], [[STEPVEC_BASE]]
; CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP5:%.*]] = mul i64 8, [[TMP4]]
; CHECK-NEXT:    [[DOTSPLATINSERT2:%.*]] = insertelement <vscale x 8 x i64> undef, i64 [[TMP5]], i32 0
; CHECK-NEXT:    [[DOTSPLAT3:%.*]] = shufflevector <vscale x 8 x i64> [[DOTSPLATINSERT2]], <vscale x 8 x i64> undef, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT:    [[DOTSPLATINSERT4:%.*]] = insertelement <vscale x 8 x i32> undef, i32 0, i32 0
; CHECK-NEXT:    [[DOTSPLAT5:%.*]] = shufflevector <vscale x 8 x i32> [[DOTSPLATINSERT4]], <vscale x 8 x i32> undef, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT:    [[STEPVEC_BASE6:%.*]] = call <vscale x 8 x i32> @llvm.experimental.vector.stepvector.nxv8i32()
; CHECK-NEXT:    [[INDUCTION7:%.*]] = add <vscale x 8 x i32> [[DOTSPLAT5]], [[STEPVEC_BASE6]]
; CHECK-NEXT:    [[TMP6:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP7:%.*]] = mul i32 8, [[TMP6]]
; CHECK-NEXT:    [[DOTSPLATINSERT8:%.*]] = insertelement <vscale x 8 x i32> undef, i32 [[TMP7]], i32 0
; CHECK-NEXT:    [[DOTSPLAT9:%.*]] = shufflevector <vscale x 8 x i32> [[DOTSPLATINSERT8]], <vscale x 8 x i32> undef, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT:    [[DOTSPLATINSERT12:%.*]] = insertelement <vscale x 8 x i32> undef, i32 0, i32 0
; CHECK-NEXT:    [[DOTSPLAT13:%.*]] = shufflevector <vscale x 8 x i32> [[DOTSPLATINSERT12]], <vscale x 8 x i32> undef, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT:    [[STEPVEC_BASE14:%.*]] = call <vscale x 8 x i32> @llvm.experimental.vector.stepvector.nxv8i32()
; CHECK-NEXT:    [[INDUCTION15:%.*]] = add <vscale x 8 x i32> [[DOTSPLAT13]], [[STEPVEC_BASE14]]
; CHECK-NEXT:    [[TMP8:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP9:%.*]] = mul i32 8, [[TMP8]]
; CHECK-NEXT:    [[DOTSPLATINSERT16:%.*]] = insertelement <vscale x 8 x i32> undef, i32 [[TMP9]], i32 0
; CHECK-NEXT:    [[DOTSPLAT17:%.*]] = shufflevector <vscale x 8 x i32> [[DOTSPLATINSERT16]], <vscale x 8 x i32> undef, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i32> undef, i32 1024, i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 8 x i32> [[BROADCAST_SPLATINSERT]], <vscale x 8 x i32> undef, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <vscale x 8 x i64> [ [[INDUCTION]], [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND10:%.*]] = phi <vscale x 8 x i32> [ [[INDUCTION7]], [[VECTOR_PH]] ], [ [[VEC_IND_NEXT11:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND18:%.*]] = phi <vscale x 8 x i32> [ [[INDUCTION15]], [[VECTOR_PH]] ], [ [[VEC_IND_NEXT19:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP10:%.*]] = sitofp <vscale x 8 x i32> [[VEC_IND10]] to <vscale x 8 x double>
; CHECK-NEXT:    [[TMP11:%.*]] = extractelement <vscale x 8 x i64> [[VEC_IND]], i32 0
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr inbounds [1024 x double], [1024 x double]* [[A]], i64 0, i64 [[TMP11]]
; CHECK-NEXT:    [[TMP13:%.*]] = getelementptr inbounds double, double* [[TMP12]], i32 0
; CHECK-NEXT:    [[TMP14:%.*]] = bitcast double* [[TMP13]] to <vscale x 1 x double>*
; CHECK-NEXT:    store <vscale x 1 x double> [[TMP10]], <vscale x 1 x double>* [[TMP14]], align 8, [[TBAA2:!tbaa !.*]]
; CHECK-NEXT:    [[TMP15:%.*]] = sub <vscale x 1 x i32> [[BROADCAST_SPLAT]], [[VEC_IND18]]
; CHECK-NEXT:    [[TMP16:%.*]] = sitofp <vscale x 1 x i32> [[TMP15]] to <vscale x 1 x double>
; CHECK-NEXT:    [[TMP17:%.*]] = getelementptr inbounds [1024 x double], [1024 x double]* [[B]], i64 0, i64 [[TMP11]]
; CHECK-NEXT:    [[TMP18:%.*]] = getelementptr inbounds double, double* [[TMP17]], i32 0
; CHECK-NEXT:    [[TMP19:%.*]] = bitcast double* [[TMP18]] to <vscale x 1 x double>*
; CHECK-NEXT:    store <vscale x 1 x double> [[TMP16]], <vscale x 1 x double>* [[TMP19]], align 8, [[TBAA2]]
; CHECK-NEXT:    [[TMP20:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[INDEX_VSCALE:%.*]] = mul i64 [[TMP20]], 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[INDEX_VSCALE]]
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <vscale x 8 x i64> [[VEC_IND]], [[DOTSPLAT3]]
; CHECK-NEXT:    [[VEC_IND_NEXT11]] = add <vscale x 8 x i32> [[VEC_IND10]], [[DOTSPLAT9]]
; CHECK-NEXT:    [[VEC_IND_NEXT19]] = add <vscale x 8 x i32> [[VEC_IND18]], [[DOTSPLAT17]]
; CHECK-NEXT:    [[TMP21:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP21]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], [[LOOP6:!llvm.loop !.*]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 1024, [[N_VEC]]
; CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_COND_CLEANUP:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[ENTRY:%.*]] ]
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    [[IDXPROM4:%.*]] = sext i32 [[ARGC:%.*]] to i64
; CHECK-NEXT:    [[ARRAYIDX5:%.*]] = getelementptr inbounds [1024 x double], [1024 x double]* [[A]], i64 0, i64 [[IDXPROM4]]
; CHECK-NEXT:    [[TMP22:%.*]] = load double, double* [[ARRAYIDX5]], align 8, [[TBAA2]]
; CHECK-NEXT:    [[ARRAYIDX7:%.*]] = getelementptr inbounds [1024 x double], [1024 x double]* [[B]], i64 0, i64 [[IDXPROM4]]
; CHECK-NEXT:    [[TMP23:%.*]] = load double, double* [[ARRAYIDX7]], align 8, [[TBAA2]]
; CHECK-NEXT:    [[ADD:%.*]] = fadd double [[TMP22]], [[TMP23]]
; CHECK-NEXT:    [[CONV8:%.*]] = fptosi double [[ADD]] to i32
; CHECK-NEXT:    call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull [[TMP1]]) [[ATTR4]]
; CHECK-NEXT:    call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull [[TMP0]]) [[ATTR4]]
; CHECK-NEXT:    ret i32 [[CONV8]]
; CHECK:       for.body:
; CHECK-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ]
; CHECK-NEXT:    [[TMP24:%.*]] = trunc i64 [[INDVARS_IV]] to i32
; CHECK-NEXT:    [[CONV:%.*]] = sitofp i32 [[TMP24]] to double
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [1024 x double], [1024 x double]* [[A]], i64 0, i64 [[INDVARS_IV]]
; CHECK-NEXT:    store double [[CONV]], double* [[ARRAYIDX]], align 8, [[TBAA2]]
; CHECK-NEXT:    [[TMP25:%.*]] = trunc i64 [[INDVARS_IV]] to i32
; CHECK-NEXT:    [[TMP26:%.*]] = sub i32 1024, [[TMP25]]
; CHECK-NEXT:    [[CONV1:%.*]] = sitofp i32 [[TMP26]] to double
; CHECK-NEXT:    [[ARRAYIDX3:%.*]] = getelementptr inbounds [1024 x double], [1024 x double]* [[B]], i64 0, i64 [[INDVARS_IV]]
; CHECK-NEXT:    store double [[CONV1]], double* [[ARRAYIDX3]], align 8, [[TBAA2]]
; CHECK-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; CHECK-NEXT:    [[EXITCOND:%.*]] = icmp eq i64 [[INDVARS_IV_NEXT]], 1024
; CHECK-NEXT:    br i1 [[EXITCOND]], label [[FOR_COND_CLEANUP]], label [[FOR_BODY]], [[LOOP8:!llvm.loop !.*]]
;
entry:
  %a = alloca [1024 x double], align 8
  %b = alloca [1024 x double], align 8
  %0 = bitcast [1024 x double]* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull %0) #2
  %1 = bitcast [1024 x double]* %b to i8*
  call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull %1) #2
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %idxprom4 = sext i32 %argc to i64
  %arrayidx5 = getelementptr inbounds [1024 x double], [1024 x double]* %a, i64 0, i64 %idxprom4
  %2 = load double, double* %arrayidx5, align 8, !tbaa !2
  %arrayidx7 = getelementptr inbounds [1024 x double], [1024 x double]* %b, i64 0, i64 %idxprom4
  %3 = load double, double* %arrayidx7, align 8, !tbaa !2
  %add = fadd double %2, %3
  %conv8 = fptosi double %add to i32
  call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull %1) #2
  call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull %0) #2
  ret i32 %conv8

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %4 = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %4 to double
  %arrayidx = getelementptr inbounds [1024 x double], [1024 x double]* %a, i64 0, i64 %indvars.iv
  store double %conv, double* %arrayidx, align 8, !tbaa !2
  %5 = trunc i64 %indvars.iv to i32
  %6 = sub i32 1024, %5
  %conv1 = sitofp i32 %6 to double
  %arrayidx3 = getelementptr inbounds [1024 x double], [1024 x double]* %b, i64 0, i64 %indvars.iv
  store double %conv1, double* %arrayidx3, align 8, !tbaa !2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+d,+experimental-v,+f,+m,-relax" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 (git@repo.hca.bsc.es:EPI/System-Software/llvm-mono.git 823a818e0f44dc9b594f14a328ec52b247f3611a)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
