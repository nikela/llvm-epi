; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -basicaa -slp-vectorizer -S -mtriple=x86_64-unknown-linux-gnu -mattr=+avx  | FileCheck %s --check-prefixes=CHECK,AVX1
; RUN: opt < %s -basicaa -slp-vectorizer -S -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 | FileCheck %s --check-prefixes=CHECK,AVX2

define void @powof2div_uniform(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32* noalias nocapture readonly %c){
; CHECK-LABEL: @powof2div_uniform(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARRAYIDX3:%.*]] = getelementptr inbounds i32, i32* [[B:%.*]], i64 1
; CHECK-NEXT:    [[ARRAYIDX4:%.*]] = getelementptr inbounds i32, i32* [[C:%.*]], i64 1
; CHECK-NEXT:    [[ARRAYIDX7:%.*]] = getelementptr inbounds i32, i32* [[A:%.*]], i64 1
; CHECK-NEXT:    [[ARRAYIDX8:%.*]] = getelementptr inbounds i32, i32* [[B]], i64 2
; CHECK-NEXT:    [[ARRAYIDX9:%.*]] = getelementptr inbounds i32, i32* [[C]], i64 2
; CHECK-NEXT:    [[ARRAYIDX12:%.*]] = getelementptr inbounds i32, i32* [[A]], i64 2
; CHECK-NEXT:    [[ARRAYIDX13:%.*]] = getelementptr inbounds i32, i32* [[B]], i64 3
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast i32* [[B]] to <4 x i32>*
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, <4 x i32>* [[TMP0]], align 4
; CHECK-NEXT:    [[ARRAYIDX14:%.*]] = getelementptr inbounds i32, i32* [[C]], i64 3
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast i32* [[C]] to <4 x i32>*
; CHECK-NEXT:    [[TMP3:%.*]] = load <4 x i32>, <4 x i32>* [[TMP2]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = add nsw <4 x i32> [[TMP3]], [[TMP1]]
; CHECK-NEXT:    [[TMP5:%.*]] = sdiv <4 x i32> [[TMP4]], <i32 2, i32 2, i32 2, i32 2>
; CHECK-NEXT:    [[ARRAYIDX17:%.*]] = getelementptr inbounds i32, i32* [[A]], i64 3
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast i32* [[A]] to <4 x i32>*
; CHECK-NEXT:    store <4 x i32> [[TMP5]], <4 x i32>* [[TMP6]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %0 = load i32, i32* %b, align 4
  %1 = load i32, i32* %c, align 4
  %add = add nsw i32 %1, %0
  %div = sdiv i32 %add, 2
  store i32 %div, i32* %a, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 1
  %2 = load i32, i32* %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds i32, i32* %c, i64 1
  %3 = load i32, i32* %arrayidx4, align 4
  %add5 = add nsw i32 %3, %2
  %div6 = sdiv i32 %add5, 2
  %arrayidx7 = getelementptr inbounds i32, i32* %a, i64 1
  store i32 %div6, i32* %arrayidx7, align 4
  %arrayidx8 = getelementptr inbounds i32, i32* %b, i64 2
  %4 = load i32, i32* %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds i32, i32* %c, i64 2
  %5 = load i32, i32* %arrayidx9, align 4
  %add10 = add nsw i32 %5, %4
  %div11 = sdiv i32 %add10, 2
  %arrayidx12 = getelementptr inbounds i32, i32* %a, i64 2
  store i32 %div11, i32* %arrayidx12, align 4
  %arrayidx13 = getelementptr inbounds i32, i32* %b, i64 3
  %6 = load i32, i32* %arrayidx13, align 4
  %arrayidx14 = getelementptr inbounds i32, i32* %c, i64 3
  %7 = load i32, i32* %arrayidx14, align 4
  %add15 = add nsw i32 %7, %6
  %div16 = sdiv i32 %add15, 2
  %arrayidx17 = getelementptr inbounds i32, i32* %a, i64 3
  store i32 %div16, i32* %arrayidx17, align 4
  ret void
}

define void @powof2div_nonuniform(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32* noalias nocapture readonly %c){
; CHECK-LABEL: @powof2div_nonuniform(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load i32, i32* [[B:%.*]], align 4
; CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* [[C:%.*]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP1]], [[TMP0]]
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[ADD]], 2
; CHECK-NEXT:    store i32 [[DIV]], i32* [[A:%.*]], align 4
; CHECK-NEXT:    [[ARRAYIDX3:%.*]] = getelementptr inbounds i32, i32* [[B]], i64 1
; CHECK-NEXT:    [[TMP2:%.*]] = load i32, i32* [[ARRAYIDX3]], align 4
; CHECK-NEXT:    [[ARRAYIDX4:%.*]] = getelementptr inbounds i32, i32* [[C]], i64 1
; CHECK-NEXT:    [[TMP3:%.*]] = load i32, i32* [[ARRAYIDX4]], align 4
; CHECK-NEXT:    [[ADD5:%.*]] = add nsw i32 [[TMP3]], [[TMP2]]
; CHECK-NEXT:    [[DIV6:%.*]] = sdiv i32 [[ADD5]], 4
; CHECK-NEXT:    [[ARRAYIDX7:%.*]] = getelementptr inbounds i32, i32* [[A]], i64 1
; CHECK-NEXT:    store i32 [[DIV6]], i32* [[ARRAYIDX7]], align 4
; CHECK-NEXT:    [[ARRAYIDX8:%.*]] = getelementptr inbounds i32, i32* [[B]], i64 2
; CHECK-NEXT:    [[TMP4:%.*]] = load i32, i32* [[ARRAYIDX8]], align 4
; CHECK-NEXT:    [[ARRAYIDX9:%.*]] = getelementptr inbounds i32, i32* [[C]], i64 2
; CHECK-NEXT:    [[TMP5:%.*]] = load i32, i32* [[ARRAYIDX9]], align 4
; CHECK-NEXT:    [[ADD10:%.*]] = add nsw i32 [[TMP5]], [[TMP4]]
; CHECK-NEXT:    [[DIV11:%.*]] = sdiv i32 [[ADD10]], 8
; CHECK-NEXT:    [[ARRAYIDX12:%.*]] = getelementptr inbounds i32, i32* [[A]], i64 2
; CHECK-NEXT:    store i32 [[DIV11]], i32* [[ARRAYIDX12]], align 4
; CHECK-NEXT:    [[ARRAYIDX13:%.*]] = getelementptr inbounds i32, i32* [[B]], i64 3
; CHECK-NEXT:    [[TMP6:%.*]] = load i32, i32* [[ARRAYIDX13]], align 4
; CHECK-NEXT:    [[ARRAYIDX14:%.*]] = getelementptr inbounds i32, i32* [[C]], i64 3
; CHECK-NEXT:    [[TMP7:%.*]] = load i32, i32* [[ARRAYIDX14]], align 4
; CHECK-NEXT:    [[ADD15:%.*]] = add nsw i32 [[TMP7]], [[TMP6]]
; CHECK-NEXT:    [[DIV16:%.*]] = sdiv i32 [[ADD15]], 16
; CHECK-NEXT:    [[ARRAYIDX17:%.*]] = getelementptr inbounds i32, i32* [[A]], i64 3
; CHECK-NEXT:    store i32 [[DIV16]], i32* [[ARRAYIDX17]], align 4
; CHECK-NEXT:    ret void
;
entry:
  %0 = load i32, i32* %b, align 4
  %1 = load i32, i32* %c, align 4
  %add = add nsw i32 %1, %0
  %div = sdiv i32 %add, 2
  store i32 %div, i32* %a, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 1
  %2 = load i32, i32* %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds i32, i32* %c, i64 1
  %3 = load i32, i32* %arrayidx4, align 4
  %add5 = add nsw i32 %3, %2
  %div6 = sdiv i32 %add5, 4
  %arrayidx7 = getelementptr inbounds i32, i32* %a, i64 1
  store i32 %div6, i32* %arrayidx7, align 4
  %arrayidx8 = getelementptr inbounds i32, i32* %b, i64 2
  %4 = load i32, i32* %arrayidx8, align 4
  %arrayidx9 = getelementptr inbounds i32, i32* %c, i64 2
  %5 = load i32, i32* %arrayidx9, align 4
  %add10 = add nsw i32 %5, %4
  %div11 = sdiv i32 %add10, 8
  %arrayidx12 = getelementptr inbounds i32, i32* %a, i64 2
  store i32 %div11, i32* %arrayidx12, align 4
  %arrayidx13 = getelementptr inbounds i32, i32* %b, i64 3
  %6 = load i32, i32* %arrayidx13, align 4
  %arrayidx14 = getelementptr inbounds i32, i32* %c, i64 3
  %7 = load i32, i32* %arrayidx14, align 4
  %add15 = add nsw i32 %7, %6
  %div16 = sdiv i32 %add15, 16
  %arrayidx17 = getelementptr inbounds i32, i32* %a, i64 3
  store i32 %div16, i32* %arrayidx17, align 4
  ret void
}

