; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -mtriple riscv64 -mattr +m,+a,+f,+d,+v -prefer-predicate-over-epilog -S -loop-vectorize < %s  -o - | FileCheck %s
; ModuleID = './simple-add.c'
source_filename = "./simple-add.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128-v128:128:128-v256:128:128-v512:128:128-v1024:128:128"
target triple = "riscv64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nounwind
define dso_local void @simple_add(i32 signext %N, i32* noalias nocapture %c, i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b) local_unnamed_addr {
; CHECK-LABEL: @simple_add(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP10:%.*]] = icmp sgt i32 [[N:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP10]], label [[FOR_BODY_PREHEADER:%.*]], label [[FOR_END:%.*]]
; CHECK:       for.body.preheader:
; CHECK-NEXT:    [[WIDE_TRIP_COUNT:%.*]] = zext i32 [[N]] to i64
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], 1
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> undef, i64 [[TRIP_COUNT_MINUS_1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 2 x i64> [[BROADCAST_SPLATINSERT]], <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <vscale x 2 x i64> undef, i64 [[INDEX]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <vscale x 2 x i64> [[BROADCAST_SPLATINSERT1]], <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT:    [[STEPVEC_BASE:%.*]] = call <vscale x 2 x i64> @llvm.experimental.vector.stepvector.nxv2i64()
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 2 x i64> [[BROADCAST_SPLAT2]], [[STEPVEC_BASE]]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, i32* [[A:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = icmp ule <vscale x 2 x i64> [[INDUCTION]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP3:%.*]] = sub i64 [[WIDE_TRIP_COUNT]], [[INDEX]]
; CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.epi.vsetvl(i64 [[TMP3]], i64 2, i64 0)
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i32, i32* [[TMP1]], i32 0
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast i32* [[TMP5]] to <vscale x 2 x i32>*
; CHECK-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <vscale x 2 x i1> undef, i1 true, i32 0
; CHECK-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <vscale x 2 x i1> [[DOTSPLATINSERT]], <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = trunc i64 [[TMP4]] to i32
; CHECK-NEXT:    [[VP_OP_LOAD:%.*]] = call <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0nxv2i32(<vscale x 2 x i32>* [[TMP6]], i32 4, <vscale x 2 x i1> [[DOTSPLAT]], i32 [[TMP7]])
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, i32* [[B:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr inbounds i32, i32* [[TMP8]], i32 0
; CHECK-NEXT:    [[TMP10:%.*]] = bitcast i32* [[TMP9]] to <vscale x 2 x i32>*
; CHECK-NEXT:    [[TMP11:%.*]] = trunc i64 [[TMP4]] to i32
; CHECK-NEXT:    [[VP_OP_LOAD5:%.*]] = call <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0nxv2i32(<vscale x 2 x i32>* [[TMP10]], i32 4, <vscale x 2 x i1> [[DOTSPLAT]], i32 [[TMP11]])
; CHECK-NEXT:    [[TMP12:%.*]] = trunc i64 [[TMP4]] to i32
; CHECK-NEXT:    [[VP_OP:%.*]] = call <vscale x 2 x i32> @llvm.vp.add.nxv2i32(<vscale x 2 x i32> [[VP_OP_LOAD5]], <vscale x 2 x i32> [[VP_OP_LOAD]], <vscale x 2 x i1> [[DOTSPLAT]], i32 [[TMP12]])
; CHECK-NEXT:    [[TMP13:%.*]] = getelementptr inbounds i32, i32* [[C:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP14:%.*]] = getelementptr inbounds i32, i32* [[TMP13]], i32 0
; CHECK-NEXT:    [[TMP15:%.*]] = bitcast i32* [[TMP14]] to <vscale x 2 x i32>*
; CHECK-NEXT:    [[TMP16:%.*]] = trunc i64 [[TMP4]] to i32
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0nxv2i32(<vscale x 2 x i32> [[VP_OP]], <vscale x 2 x i32>* [[TMP15]], i32 4, <vscale x 2 x i1> [[DOTSPLAT]], i32 [[TMP16]])
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP4]]
; CHECK-NEXT:    [[TMP17:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[WIDE_TRIP_COUNT]]
; CHECK-NEXT:    br i1 [[TMP17]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop !0
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label [[FOR_END_LOOPEXIT:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[WIDE_TRIP_COUNT]], [[MIDDLE_BLOCK]] ], [ 0, [[FOR_BODY_PREHEADER]] ]
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ]
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, i32* [[A]], i64 [[INDVARS_IV]]
; CHECK-NEXT:    [[TMP18:%.*]] = load i32, i32* [[ARRAYIDX]], align 4
; CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds i32, i32* [[B]], i64 [[INDVARS_IV]]
; CHECK-NEXT:    [[TMP19:%.*]] = load i32, i32* [[ARRAYIDX2]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP19]], [[TMP18]]
; CHECK-NEXT:    [[ARRAYIDX4:%.*]] = getelementptr inbounds i32, i32* [[C]], i64 [[INDVARS_IV]]
; CHECK-NEXT:    store i32 [[ADD]], i32* [[ARRAYIDX4]], align 4
; CHECK-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; CHECK-NEXT:    [[EXITCOND:%.*]] = icmp eq i64 [[INDVARS_IV_NEXT]], [[WIDE_TRIP_COUNT]]
; CHECK-NEXT:    br i1 [[EXITCOND]], label [[FOR_END_LOOPEXIT]], label [[FOR_BODY]], !llvm.loop !2
; CHECK:       for.end.loopexit:
; CHECK-NEXT:    br label [[FOR_END]]
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
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, i32* %c, i64 %indvars.iv
  store i32 %add, i32* %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}
