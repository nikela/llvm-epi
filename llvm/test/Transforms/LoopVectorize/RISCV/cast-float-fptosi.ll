; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -mtriple riscv64 -mattr +m,+a,+f,+d,+experimental-v \
; RUN:    -prefer-predicate-over-epilog -S -O2 < %s  -o - | FileCheck %s
; ModuleID = './cast-float-fptosi.c'
source_filename = "./cast-float-fptosi.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128-v128:128:128-v256:128:128-v512:128:128-v1024:128:128"
target triple = "riscv64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nounwind
define dso_local void @bar(i32 signext %N, i32* noalias nocapture %c, float* noalias nocapture readonly %a, float* noalias nocapture readonly %b) local_unnamed_addr {
; CHECK-LABEL: @bar(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP10:%.*]] = icmp sgt i32 [[N:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP10]], label [[FOR_BODY_PREHEADER:%.*]], label [[FOR_END:%.*]]
; CHECK:       for.body.preheader:
; CHECK-NEXT:    [[WIDE_TRIP_COUNT:%.*]] = zext i32 [[N]] to i64
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[FOR_BODY_PREHEADER]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds float, float* [[A:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP1:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[INDEX]]
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.epi.vsetvl(i64 [[TMP1]], i64 2, i64 0)
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast float* [[TMP0]] to <vscale x 2 x float>*
; CHECK-NEXT:    [[TMP4:%.*]] = trunc i64 [[TMP2]] to i32
; CHECK-NEXT:    [[VP_OP_LOAD:%.*]] = call <vscale x 2 x float> @llvm.vp.load.nxv2f32.p0nxv2f32(<vscale x 2 x float>* [[TMP3]], i32 4, <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> undef, i1 true, i32 0), <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer), i32 [[TMP4]])
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds float, float* [[B:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast float* [[TMP5]] to <vscale x 2 x float>*
; CHECK-NEXT:    [[TMP7:%.*]] = trunc i64 [[TMP2]] to i32
; CHECK-NEXT:    [[VP_OP_LOAD5:%.*]] = call <vscale x 2 x float> @llvm.vp.load.nxv2f32.p0nxv2f32(<vscale x 2 x float>* [[TMP6]], i32 4, <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> undef, i1 true, i32 0), <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer), i32 [[TMP7]])
; CHECK-NEXT:    [[TMP8:%.*]] = trunc i64 [[TMP2]] to i32
; CHECK-NEXT:    [[VP_OP:%.*]] = call <vscale x 2 x float> @llvm.vp.fadd.nxv2f32(<vscale x 2 x float> [[VP_OP_LOAD]], <vscale x 2 x float> [[VP_OP_LOAD5]], metadata !"round.tonearest", metadata !"fpexcept.ignore", <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> undef, i1 true, i32 0), <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer), i32 [[TMP8]])
; CHECK-NEXT:    [[TMP9:%.*]] = trunc i64 [[TMP2]] to i32
; CHECK-NEXT:    [[VP_CAST:%.*]] = call <vscale x 2 x i32> @llvm.vp.fptosi.nxv2i32.nxv2f32(<vscale x 2 x float> [[VP_OP]], metadata !"fpexcept.ignore", <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> undef, i1 true, i32 0), <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer), i32 [[TMP9]])
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds i32, i32* [[C:%.*]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP11:%.*]] = bitcast i32* [[TMP10]] to <vscale x 2 x i32>*
; CHECK-NEXT:    [[TMP12:%.*]] = trunc i64 [[TMP2]] to i32
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0nxv2i32(<vscale x 2 x i32> [[VP_CAST]], <vscale x 2 x i32>* [[TMP11]], i32 4, <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> undef, i1 true, i32 0), <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer), i32 [[TMP12]])
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP2]]
; CHECK-NEXT:    [[TMP13:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[WIDE_TRIP_COUNT]]
; CHECK-NEXT:    br i1 [[TMP13]], label [[FOR_END]], label [[VECTOR_BODY]], !llvm.loop !0
; CHECK:       for.end:
; CHECK-NEXT:    ret void
;
entry:
  %cmp10 = icmp sgt i32 %N, 0
  br i1 %cmp10, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %indvars.iv
  %0 = load float, float* %arrayidx
  %arrayidx2 = getelementptr inbounds float, float* %b, i64 %indvars.iv
  %1 = load float, float* %arrayidx2
  %add = fadd float %0, %1
  %conv = fptosi float %add to i32
  %arrayidx4 = getelementptr inbounds i32, i32* %c, i64 %indvars.iv
  store i32 %conv, i32* %arrayidx4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}
