; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -mtriple=riscv64 -mattr=+v,+a,+f,+d,+c,+m \
; RUN:     -codegenprepare -S -o - %s | FileCheck %s

define dso_local void @saxpy(i64 %N, double %d, double* noalias nocapture %c, double* noalias nocapture readonly %a, double* noalias nocapture readonly %b) local_unnamed_addr #0 {
; CHECK-LABEL: @saxpy(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP8:%.*]] = icmp sgt i64 [[N:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP8]], label [[VECTOR_PH:%.*]], label [[FOR_END:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds double, ptr [[A:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP1:%.*]] = sub i64 [[N]], [[INDEX]]
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.epi.vsetvl(i64 [[TMP1]], i64 3, i64 0)
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast ptr [[TMP0]] to ptr
; CHECK-NEXT:    [[TMP4:%.*]] = trunc i64 [[TMP2]] to i32
; CHECK-NEXT:    [[VP_OP_LOAD:%.*]] = call <vscale x 1 x double> @llvm.vp.load.nxv1f64.p0(ptr [[TMP3]], <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> undef, i1 true, i32 0), <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer), i32 [[TMP4]]), !tbaa [[TBAA4:![0-9]+]]
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds double, ptr [[B:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast ptr [[TMP5]] to ptr
; CHECK-NEXT:    [[TMP7:%.*]] = trunc i64 [[TMP2]] to i32
; CHECK-NEXT:    [[VP_OP_LOAD14:%.*]] = call <vscale x 1 x double> @llvm.vp.load.nxv1f64.p0(ptr [[TMP6]], <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> undef, i1 true, i32 0), <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer), i32 [[TMP7]]), !tbaa [[TBAA4]]
; CHECK-NEXT:    [[TMP8:%.*]] = trunc i64 [[TMP2]] to i32
; CHECK-NEXT:    [[TMP9:%.*]] = insertelement <vscale x 1 x double> undef, double [[D:%.*]], i32 0
; CHECK-NEXT:    [[TMP10:%.*]] = shufflevector <vscale x 1 x double> [[TMP9]], <vscale x 1 x double> undef, <vscale x 1 x i32> zeroinitializer
; CHECK-NEXT:    [[VP_OP:%.*]] = call <vscale x 1 x double> @llvm.vp.fmul.nxv1f64(<vscale x 1 x double> [[VP_OP_LOAD14]], <vscale x 1 x double> [[TMP10]], <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> undef, i1 true, i32 0), <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer), i32 [[TMP8]])
; CHECK-NEXT:    [[TMP11:%.*]] = trunc i64 [[TMP2]] to i32
; CHECK-NEXT:    [[VP_OP21:%.*]] = call <vscale x 1 x double> @llvm.vp.fadd.nxv1f64(<vscale x 1 x double> [[VP_OP_LOAD]], <vscale x 1 x double> [[VP_OP]], <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> undef, i1 true, i32 0), <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer), i32 [[TMP11]])
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr inbounds double, ptr [[C:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP13:%.*]] = bitcast ptr [[TMP12]] to ptr
; CHECK-NEXT:    [[TMP14:%.*]] = trunc i64 [[TMP2]] to i32
; CHECK-NEXT:    call void @llvm.vp.store.nxv1f64.p0(<vscale x 1 x double> [[VP_OP21]], ptr [[TMP13]], <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> undef, i1 true, i32 0), <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer), i32 [[TMP14]]), !tbaa [[TBAA4]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP2]]
; CHECK-NEXT:    [[TMP15:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[TMP15]], label [[FOR_END]], label [[VECTOR_BODY]], !llvm.loop [[LOOP8:![0-9]+]]
; CHECK:       for.end:
; CHECK-NEXT:    ret void
;
entry:
  %cmp8 = icmp sgt i64 %N, 0
  br i1 %cmp8, label %vector.ph, label %for.end

vector.ph:                                        ; preds = %entry
  %broadcast.splatinsert15 = insertelement <vscale x 1 x double> undef, double %d, i32 0
  %broadcast.splat16 = shufflevector <vscale x 1 x double> %broadcast.splatinsert15, <vscale x 1 x double> undef, <vscale x 1 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds double, double* %a, i64 %index
  %1 = sub i64 %N, %index
  %2 = call i64 @llvm.epi.vsetvl(i64 %1, i64 3, i64 0)
  %3 = bitcast double* %0 to <vscale x 1 x double>*
  %4 = trunc i64 %2 to i32
  %vp.op.load = call <vscale x 1 x double> @llvm.vp.load.nxv1f64.p0nxv1f64(<vscale x 1 x double>* %3, <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> undef, i1 true, i32 0), <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer), i32 %4), !tbaa !4
  %5 = getelementptr inbounds double, double* %b, i64 %index
  %6 = bitcast double* %5 to <vscale x 1 x double>*
  %7 = trunc i64 %2 to i32
  %vp.op.load14 = call <vscale x 1 x double> @llvm.vp.load.nxv1f64.p0nxv1f64(<vscale x 1 x double>* %6, <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> undef, i1 true, i32 0), <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer), i32 %7), !tbaa !4
  %8 = trunc i64 %2 to i32
  %vp.op = call <vscale x 1 x double> @llvm.vp.fmul.nxv1f64(<vscale x 1 x double> %vp.op.load14, <vscale x 1 x double> %broadcast.splat16, <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> undef, i1 true, i32 0), <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer), i32 %8)
  %9 = trunc i64 %2 to i32
  %vp.op21 = call <vscale x 1 x double> @llvm.vp.fadd.nxv1f64(<vscale x 1 x double> %vp.op.load, <vscale x 1 x double> %vp.op, <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> undef, i1 true, i32 0), <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer), i32 %9)
  %10 = getelementptr inbounds double, double* %c, i64 %index
  %11 = bitcast double* %10 to <vscale x 1 x double>*
  %12 = trunc i64 %2 to i32
  call void @llvm.vp.store.nxv1f64.p0nxv1f64(<vscale x 1 x double> %vp.op21, <vscale x 1 x double>* %11, <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> undef, i1 true, i32 0), <vscale x 1 x i1> undef, <vscale x 1 x i32> zeroinitializer), i32 %12), !tbaa !4
  %index.next = add i64 %index, %2
  %13 = icmp eq i64 %index.next, %N
  br i1 %13, label %for.end, label %vector.body, !llvm.loop !8

for.end:                                          ; preds = %vector.body, %entry
  ret void
}

declare i64 @llvm.epi.vsetvl(i64, i64, i64) #1

declare <vscale x 1 x double> @llvm.vp.load.nxv1f64.p0nxv1f64(<vscale x 1 x double>* nocapture, <vscale x 1 x i1>, i32) #2

declare <vscale x 1 x double> @llvm.vp.fmul.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, <vscale x 1 x i1>, i32) #3

declare <vscale x 1 x double> @llvm.vp.fadd.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, <vscale x 1 x i1>, i32) #3

declare void @llvm.vp.store.nxv1f64.p0nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>* nocapture, <vscale x 1 x i1>, i32) #4

attributes #0 = { nofree norecurse nounwind "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+d,+v,+f,+m,-relax,-save-restore" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { argmemonly nosync nounwind readonly willreturn }
attributes #3 = { nounwind readnone willreturn }
attributes #4 = { argmemonly nosync nounwind willreturn writeonly }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64d"}
!2 = !{i32 1, !"SmallDataLimit", i32 8}
!3 = !{!"clang version 12.0.0 (git@repo.hca.bsc.es:EPI/System-Software/llvm-mono.git 3bad630a07b9e6351b7a96098b907fec4c90becb)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"double", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.isvectorized", i32 1}
