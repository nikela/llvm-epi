; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -O2 -expand-reductions -S < %s | FileCheck %s

; Test if SLP vector reduction patterns are recognized
; and optionally converted to reduction intrinsics and 
; back to raw IR.

target triple = "x86_64--"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @add_v4i32(i32* %p) #0 {
; CHECK-LABEL: @add_v4i32(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast i32* [[P:%.*]] to <4 x i32>*
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, <4 x i32>* [[TMP0]], align 4, !tbaa !0
; CHECK-NEXT:    [[RDX_SHUF:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX:%.*]] = add <4 x i32> [[TMP1]], [[RDX_SHUF]]
; CHECK-NEXT:    [[RDX_SHUF3:%.*]] = shufflevector <4 x i32> [[BIN_RDX]], <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX4:%.*]] = add <4 x i32> [[BIN_RDX]], [[RDX_SHUF3]]
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <4 x i32> [[BIN_RDX4]], i32 0
; CHECK-NEXT:    ret i32 [[TMP2]]
;
entry:
  br label %for.cond

for.cond:
  %r.0 = phi i32 [ 0, %entry ], [ %add, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 4
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  br label %for.end

for.body:
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, i32* %p, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4, !tbaa !3
  %add = add nsw i32 %r.0, %0
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret i32 %r.0
}

define signext i16 @mul_v8i16(i16* %p) #0 {
; CHECK-LABEL: @mul_v8i16(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast i16* [[P:%.*]] to <8 x i16>*
; CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i16>, <8 x i16>* [[TMP0]], align 2, !tbaa !4
; CHECK-NEXT:    [[RDX_SHUF:%.*]] = shufflevector <8 x i16> [[TMP1]], <8 x i16> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX:%.*]] = mul <8 x i16> [[TMP1]], [[RDX_SHUF]]
; CHECK-NEXT:    [[RDX_SHUF3:%.*]] = shufflevector <8 x i16> [[BIN_RDX]], <8 x i16> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX4:%.*]] = mul <8 x i16> [[BIN_RDX]], [[RDX_SHUF3]]
; CHECK-NEXT:    [[RDX_SHUF5:%.*]] = shufflevector <8 x i16> [[BIN_RDX4]], <8 x i16> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX6:%.*]] = mul <8 x i16> [[BIN_RDX4]], [[RDX_SHUF5]]
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <8 x i16> [[BIN_RDX6]], i32 0
; CHECK-NEXT:    ret i16 [[TMP2]]
;
entry:
  br label %for.cond

for.cond:
  %r.0 = phi i16 [ 1, %entry ], [ %conv2, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 8
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  br label %for.end

for.body:
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i16, i16* %p, i64 %idxprom
  %0 = load i16, i16* %arrayidx, align 2, !tbaa !7
  %conv = sext i16 %0 to i32
  %conv1 = sext i16 %r.0 to i32
  %mul = mul nsw i32 %conv1, %conv
  %conv2 = trunc i32 %mul to i16
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret i16 %r.0
}

define signext i8 @or_v16i8(i8* %p) #0 {
; CHECK-LABEL: @or_v16i8(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast i8* [[P:%.*]] to <16 x i8>*
; CHECK-NEXT:    [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[TMP0]], align 1, !tbaa !6
; CHECK-NEXT:    [[RDX_SHUF:%.*]] = shufflevector <16 x i8> [[TMP1]], <16 x i8> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX:%.*]] = or <16 x i8> [[TMP1]], [[RDX_SHUF]]
; CHECK-NEXT:    [[RDX_SHUF4:%.*]] = shufflevector <16 x i8> [[BIN_RDX]], <16 x i8> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX5:%.*]] = or <16 x i8> [[BIN_RDX]], [[RDX_SHUF4]]
; CHECK-NEXT:    [[RDX_SHUF6:%.*]] = shufflevector <16 x i8> [[BIN_RDX5]], <16 x i8> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX7:%.*]] = or <16 x i8> [[BIN_RDX5]], [[RDX_SHUF6]]
; CHECK-NEXT:    [[RDX_SHUF8:%.*]] = shufflevector <16 x i8> [[BIN_RDX7]], <16 x i8> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX9:%.*]] = or <16 x i8> [[BIN_RDX7]], [[RDX_SHUF8]]
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <16 x i8> [[BIN_RDX9]], i32 0
; CHECK-NEXT:    ret i8 [[TMP2]]
;
entry:
  br label %for.cond

for.cond:
  %r.0 = phi i8 [ 0, %entry ], [ %conv2, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 16
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  br label %for.end

for.body:
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i8, i8* %p, i64 %idxprom
  %0 = load i8, i8* %arrayidx, align 1, !tbaa !9
  %conv = sext i8 %0 to i32
  %conv1 = sext i8 %r.0 to i32
  %or = or i32 %conv1, %conv
  %conv2 = trunc i32 %or to i8
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret i8 %r.0
}

define i32 @smin_v4i32(i32* %p) #0 {
; CHECK-LABEL: @smin_v4i32(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast i32* [[P:%.*]] to <4 x i32>*
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, <4 x i32>* [[TMP0]], align 4, !tbaa !0
; CHECK-NEXT:    [[RDX_SHUF:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
; CHECK-NEXT:    [[RDX_MINMAX_CMP:%.*]] = icmp slt <4 x i32> [[TMP1]], [[RDX_SHUF]]
; CHECK-NEXT:    [[RDX_MINMAX_SELECT:%.*]] = select <4 x i1> [[RDX_MINMAX_CMP]], <4 x i32> [[TMP1]], <4 x i32> [[RDX_SHUF]]
; CHECK-NEXT:    [[RDX_SHUF3:%.*]] = shufflevector <4 x i32> [[RDX_MINMAX_SELECT]], <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[RDX_MINMAX_CMP4:%.*]] = icmp slt <4 x i32> [[RDX_MINMAX_SELECT]], [[RDX_SHUF3]]
; CHECK-NEXT:    [[RDX_MINMAX_SELECT5:%.*]] = select <4 x i1> [[RDX_MINMAX_CMP4]], <4 x i32> [[RDX_MINMAX_SELECT]], <4 x i32> [[RDX_SHUF3]]
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <4 x i32> [[RDX_MINMAX_SELECT5]], i32 0
; CHECK-NEXT:    ret i32 [[TMP2]]
;
entry:
  br label %for.cond

for.cond:
  %r.0 = phi i32 [ 2147483647, %entry ], [ %cond, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 4
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  br label %for.end

for.body:
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, i32* %p, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4, !tbaa !3
  %cmp1 = icmp slt i32 %0, %r.0
  br i1 %cmp1, label %cond.true, label %cond.false

cond.true:
  %idxprom2 = sext i32 %i.0 to i64
  %arrayidx3 = getelementptr inbounds i32, i32* %p, i64 %idxprom2
  %1 = load i32, i32* %arrayidx3, align 4, !tbaa !3
  br label %cond.end

cond.false:
  br label %cond.end

cond.end:
  %cond = phi i32 [ %1, %cond.true ], [ %r.0, %cond.false ]
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret i32 %r.0
}

define i32 @umax_v4i32(i32* %p) #0 {
; CHECK-LABEL: @umax_v4i32(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast i32* [[P:%.*]] to <4 x i32>*
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, <4 x i32>* [[TMP0]], align 4, !tbaa !0
; CHECK-NEXT:    [[RDX_SHUF:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
; CHECK-NEXT:    [[RDX_MINMAX_CMP:%.*]] = icmp ugt <4 x i32> [[TMP1]], [[RDX_SHUF]]
; CHECK-NEXT:    [[RDX_MINMAX_SELECT:%.*]] = select <4 x i1> [[RDX_MINMAX_CMP]], <4 x i32> [[TMP1]], <4 x i32> [[RDX_SHUF]]
; CHECK-NEXT:    [[RDX_SHUF3:%.*]] = shufflevector <4 x i32> [[RDX_MINMAX_SELECT]], <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[RDX_MINMAX_CMP4:%.*]] = icmp ugt <4 x i32> [[RDX_MINMAX_SELECT]], [[RDX_SHUF3]]
; CHECK-NEXT:    [[RDX_MINMAX_SELECT5:%.*]] = select <4 x i1> [[RDX_MINMAX_CMP4]], <4 x i32> [[RDX_MINMAX_SELECT]], <4 x i32> [[RDX_SHUF3]]
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <4 x i32> [[RDX_MINMAX_SELECT5]], i32 0
; CHECK-NEXT:    ret i32 [[TMP2]]
;
entry:
  br label %for.cond

for.cond:
  %r.0 = phi i32 [ 0, %entry ], [ %cond, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 4
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  br label %for.end

for.body:
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, i32* %p, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4, !tbaa !3
  %cmp1 = icmp ugt i32 %0, %r.0
  br i1 %cmp1, label %cond.true, label %cond.false

cond.true:
  %idxprom2 = sext i32 %i.0 to i64
  %arrayidx3 = getelementptr inbounds i32, i32* %p, i64 %idxprom2
  %1 = load i32, i32* %arrayidx3, align 4, !tbaa !3
  br label %cond.end

cond.false:
  br label %cond.end

cond.end:
  %cond = phi i32 [ %1, %cond.true ], [ %r.0, %cond.false ]
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret i32 %r.0
}

define float @fadd_v4i32(float* %p) #0 {
; CHECK-LABEL: @fadd_v4i32(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast float* [[P:%.*]] to <4 x float>*
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x float>, <4 x float>* [[TMP0]], align 4, !tbaa !7
; CHECK-NEXT:    [[RDX_SHUF:%.*]] = shufflevector <4 x float> [[TMP1]], <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX:%.*]] = fadd fast <4 x float> [[TMP1]], [[RDX_SHUF]]
; CHECK-NEXT:    [[RDX_SHUF3:%.*]] = shufflevector <4 x float> [[BIN_RDX]], <4 x float> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX4:%.*]] = fadd fast <4 x float> [[BIN_RDX]], [[RDX_SHUF3]]
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <4 x float> [[BIN_RDX4]], i32 0
; CHECK-NEXT:    [[OP_EXTRA:%.*]] = fadd fast float [[TMP2]], 4.200000e+01
; CHECK-NEXT:    ret float [[OP_EXTRA]]
;
entry:
  br label %for.cond

for.cond:
  %r.0 = phi float [ 4.200000e+01, %entry ], [ %add, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 4
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  br label %for.end

for.body:
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds float, float* %p, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4, !tbaa !10
  %add = fadd fast float %r.0, %0
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret float %r.0
}

define float @fmul_v4i32(float* %p) #0 {
; CHECK-LABEL: @fmul_v4i32(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast float* [[P:%.*]] to <4 x float>*
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x float>, <4 x float>* [[TMP0]], align 4, !tbaa !7
; CHECK-NEXT:    [[RDX_SHUF:%.*]] = shufflevector <4 x float> [[TMP1]], <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX:%.*]] = fmul fast <4 x float> [[TMP1]], [[RDX_SHUF]]
; CHECK-NEXT:    [[RDX_SHUF3:%.*]] = shufflevector <4 x float> [[BIN_RDX]], <4 x float> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX4:%.*]] = fmul fast <4 x float> [[BIN_RDX]], [[RDX_SHUF3]]
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <4 x float> [[BIN_RDX4]], i32 0
; CHECK-NEXT:    [[OP_EXTRA:%.*]] = fmul fast float [[TMP2]], 4.200000e+01
; CHECK-NEXT:    ret float [[OP_EXTRA]]
;
entry:
  br label %for.cond

for.cond:
  %r.0 = phi float [ 4.200000e+01, %entry ], [ %mul, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 4
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  br label %for.end

for.body:
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds float, float* %p, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4, !tbaa !10
  %mul = fmul fast float %r.0, %0
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret float %r.0
}

define float @fmin_v4i32(float* %p) #0 {
; CHECK-LABEL: @fmin_v4i32(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load float, float* [[P:%.*]], align 4, !tbaa !7
; CHECK-NEXT:    [[TMP1:%.*]] = tail call fast float @llvm.minnum.f32(float [[TMP0]], float 0x47EFFFFFE0000000)
; CHECK-NEXT:    [[ARRAYIDX_1:%.*]] = getelementptr inbounds float, float* [[P]], i64 1
; CHECK-NEXT:    [[TMP2:%.*]] = load float, float* [[ARRAYIDX_1]], align 4, !tbaa !7
; CHECK-NEXT:    [[TMP3:%.*]] = tail call fast float @llvm.minnum.f32(float [[TMP2]], float [[TMP1]])
; CHECK-NEXT:    [[ARRAYIDX_2:%.*]] = getelementptr inbounds float, float* [[P]], i64 2
; CHECK-NEXT:    [[TMP4:%.*]] = load float, float* [[ARRAYIDX_2]], align 4, !tbaa !7
; CHECK-NEXT:    [[TMP5:%.*]] = tail call fast float @llvm.minnum.f32(float [[TMP4]], float [[TMP3]])
; CHECK-NEXT:    [[ARRAYIDX_3:%.*]] = getelementptr inbounds float, float* [[P]], i64 3
; CHECK-NEXT:    [[TMP6:%.*]] = load float, float* [[ARRAYIDX_3]], align 4, !tbaa !7
; CHECK-NEXT:    [[TMP7:%.*]] = tail call fast float @llvm.minnum.f32(float [[TMP6]], float [[TMP5]])
; CHECK-NEXT:    ret float [[TMP7]]
;
entry:
  br label %for.cond

for.cond:
  %r.0 = phi float [  0x47EFFFFFE0000000, %entry ], [ %cond, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 4
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  br label %for.end

for.body:
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds float, float* %p, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4, !tbaa !10
  %cmp1 = fcmp fast olt float %0, %r.0
  br i1 %cmp1, label %cond.true, label %cond.false

cond.true:
  %idxprom2 = sext i32 %i.0 to i64
  %arrayidx3 = getelementptr inbounds float, float* %p, i64 %idxprom2
  %1 = load float, float* %arrayidx3, align 4, !tbaa !10
  br label %cond.end

cond.false:
  br label %cond.end

cond.end:
  %cond = phi fast float [ %1, %cond.true ], [ %r.0, %cond.false ]
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret float %r.0
}

attributes #0 = { nounwind ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+avx,+cx16,+cx8,+fxsr,+mmx,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "unsafe-fp-math"="true" "use-soft-float"="false" }
;attributes #1 = { argmemonly nounwind willreturn }

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git a9fe69c359de653015c39e413e48630d069abe27)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"short", !5, i64 0}
!9 = !{!5, !5, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !5, i64 0}
