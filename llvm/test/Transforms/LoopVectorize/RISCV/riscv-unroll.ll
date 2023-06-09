; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -passes=loop-vectorize -force-target-max-vector-interleave=1 -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-min=128 -scalable-vectorization=off -S | FileCheck %s --check-prefix=LMUL1
; RUN: opt < %s -passes=loop-vectorize -force-target-max-vector-interleave=1 -mtriple=riscv32 -mattr=+v -riscv-v-vector-bits-min=128 -scalable-vectorization=off -S | FileCheck %s --check-prefix=LMUL1
; RUN: opt < %s -passes=loop-vectorize -force-target-max-vector-interleave=1 -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-min=128 -scalable-vectorization=off -riscv-v-register-bit-width-lmul=2 -S | FileCheck %s --check-prefix=LMUL2
; RUN: opt < %s -passes=loop-vectorize -force-target-max-vector-interleave=1 -mtriple=riscv32 -mattr=+v -riscv-v-vector-bits-min=128 -scalable-vectorization=off -riscv-v-register-bit-width-lmul=2 -S | FileCheck %s --check-prefix=LMUL2

; Function Attrs: nounwind
define ptr @array_add(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr %c, i32 %size) {
; LMUL1-LABEL: @array_add(
; LMUL1-NEXT:  entry:
; LMUL1-NEXT:    [[CMP10:%.*]] = icmp sgt i32 [[SIZE:%.*]], 0
; LMUL1-NEXT:    br i1 [[CMP10]], label [[FOR_BODY_PREHEADER:%.*]], label [[FOR_END:%.*]]
; LMUL1:       for.body.preheader:
; LMUL1-NEXT:    [[TMP0:%.*]] = add i32 [[SIZE]], -1
; LMUL1-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64
; LMUL1-NEXT:    [[TMP2:%.*]] = add nuw nsw i64 [[TMP1]], 1
; LMUL1-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[TMP2]], 8
; LMUL1-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; LMUL1:       vector.ph:
; LMUL1-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[TMP2]], 8
; LMUL1-NEXT:    [[N_VEC:%.*]] = sub i64 [[TMP2]], [[N_MOD_VF]]
; LMUL1-NEXT:    br label [[VECTOR_BODY:%.*]]
; LMUL1:       vector.body:
; LMUL1-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; LMUL1-NEXT:    [[TMP3:%.*]] = add i64 [[INDEX]], 0
; LMUL1-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i32, ptr [[A:%.*]], i64 [[TMP3]]
; LMUL1-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i32, ptr [[TMP4]], i32 0
; LMUL1-NEXT:    [[WIDE_LOAD:%.*]] = load <8 x i32>, ptr [[TMP5]], align 4
; LMUL1-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, ptr [[B:%.*]], i64 [[TMP3]]
; LMUL1-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, ptr [[TMP6]], i32 0
; LMUL1-NEXT:    [[WIDE_LOAD1:%.*]] = load <8 x i32>, ptr [[TMP7]], align 4
; LMUL1-NEXT:    [[TMP8:%.*]] = add nsw <8 x i32> [[WIDE_LOAD1]], [[WIDE_LOAD]]
; LMUL1-NEXT:    [[TMP9:%.*]] = getelementptr inbounds i32, ptr [[C:%.*]], i64 [[TMP3]]
; LMUL1-NEXT:    [[TMP10:%.*]] = getelementptr inbounds i32, ptr [[TMP9]], i32 0
; LMUL1-NEXT:    store <8 x i32> [[TMP8]], ptr [[TMP10]], align 4
; LMUL1-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 8
; LMUL1-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; LMUL1-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; LMUL1:       middle.block:
; LMUL1-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[TMP2]], [[N_VEC]]
; LMUL1-NEXT:    br i1 [[CMP_N]], label [[FOR_END_LOOPEXIT:%.*]], label [[SCALAR_PH]]
; LMUL1:       scalar.ph:
; LMUL1-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[FOR_BODY_PREHEADER]] ]
; LMUL1-NEXT:    br label [[FOR_BODY:%.*]]
; LMUL1:       for.body:
; LMUL1-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ], [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ]
; LMUL1-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr [[A]], i64 [[INDVARS_IV]]
; LMUL1-NEXT:    [[TMP12:%.*]] = load i32, ptr [[ARRAYIDX]], align 4
; LMUL1-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds i32, ptr [[B]], i64 [[INDVARS_IV]]
; LMUL1-NEXT:    [[TMP13:%.*]] = load i32, ptr [[ARRAYIDX2]], align 4
; LMUL1-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP13]], [[TMP12]]
; LMUL1-NEXT:    [[ARRAYIDX4:%.*]] = getelementptr inbounds i32, ptr [[C]], i64 [[INDVARS_IV]]
; LMUL1-NEXT:    store i32 [[ADD]], ptr [[ARRAYIDX4]], align 4
; LMUL1-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; LMUL1-NEXT:    [[LFTR_WIDEIV:%.*]] = trunc i64 [[INDVARS_IV_NEXT]] to i32
; LMUL1-NEXT:    [[EXITCOND:%.*]] = icmp eq i32 [[LFTR_WIDEIV]], [[SIZE]]
; LMUL1-NEXT:    br i1 [[EXITCOND]], label [[FOR_END_LOOPEXIT]], label [[FOR_BODY]], !llvm.loop [[LOOP3:![0-9]+]]
; LMUL1:       for.end.loopexit:
; LMUL1-NEXT:    br label [[FOR_END]]
; LMUL1:       for.end:
; LMUL1-NEXT:    ret ptr [[C]]
;
; LMUL2-LABEL: @array_add(
; LMUL2-NEXT:  entry:
; LMUL2-NEXT:    [[CMP10:%.*]] = icmp sgt i32 [[SIZE:%.*]], 0
; LMUL2-NEXT:    br i1 [[CMP10]], label [[FOR_BODY_PREHEADER:%.*]], label [[FOR_END:%.*]]
; LMUL2:       for.body.preheader:
; LMUL2-NEXT:    [[TMP0:%.*]] = add i32 [[SIZE]], -1
; LMUL2-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64
; LMUL2-NEXT:    [[TMP2:%.*]] = add nuw nsw i64 [[TMP1]], 1
; LMUL2-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[TMP2]], 8
; LMUL2-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; LMUL2:       vector.ph:
; LMUL2-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[TMP2]], 8
; LMUL2-NEXT:    [[N_VEC:%.*]] = sub i64 [[TMP2]], [[N_MOD_VF]]
; LMUL2-NEXT:    br label [[VECTOR_BODY:%.*]]
; LMUL2:       vector.body:
; LMUL2-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; LMUL2-NEXT:    [[TMP3:%.*]] = add i64 [[INDEX]], 0
; LMUL2-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i32, ptr [[A:%.*]], i64 [[TMP3]]
; LMUL2-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i32, ptr [[TMP4]], i32 0
; LMUL2-NEXT:    [[WIDE_LOAD:%.*]] = load <8 x i32>, ptr [[TMP5]], align 4
; LMUL2-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, ptr [[B:%.*]], i64 [[TMP3]]
; LMUL2-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, ptr [[TMP6]], i32 0
; LMUL2-NEXT:    [[WIDE_LOAD1:%.*]] = load <8 x i32>, ptr [[TMP7]], align 4
; LMUL2-NEXT:    [[TMP8:%.*]] = add nsw <8 x i32> [[WIDE_LOAD1]], [[WIDE_LOAD]]
; LMUL2-NEXT:    [[TMP9:%.*]] = getelementptr inbounds i32, ptr [[C:%.*]], i64 [[TMP3]]
; LMUL2-NEXT:    [[TMP10:%.*]] = getelementptr inbounds i32, ptr [[TMP9]], i32 0
; LMUL2-NEXT:    store <8 x i32> [[TMP8]], ptr [[TMP10]], align 4
; LMUL2-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 8
; LMUL2-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; LMUL2-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; LMUL2:       middle.block:
; LMUL2-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[TMP2]], [[N_VEC]]
; LMUL2-NEXT:    br i1 [[CMP_N]], label [[FOR_END_LOOPEXIT:%.*]], label [[SCALAR_PH]]
; LMUL2:       scalar.ph:
; LMUL2-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[FOR_BODY_PREHEADER]] ]
; LMUL2-NEXT:    br label [[FOR_BODY:%.*]]
; LMUL2:       for.body:
; LMUL2-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ], [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ]
; LMUL2-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr [[A]], i64 [[INDVARS_IV]]
; LMUL2-NEXT:    [[TMP12:%.*]] = load i32, ptr [[ARRAYIDX]], align 4
; LMUL2-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds i32, ptr [[B]], i64 [[INDVARS_IV]]
; LMUL2-NEXT:    [[TMP13:%.*]] = load i32, ptr [[ARRAYIDX2]], align 4
; LMUL2-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP13]], [[TMP12]]
; LMUL2-NEXT:    [[ARRAYIDX4:%.*]] = getelementptr inbounds i32, ptr [[C]], i64 [[INDVARS_IV]]
; LMUL2-NEXT:    store i32 [[ADD]], ptr [[ARRAYIDX4]], align 4
; LMUL2-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; LMUL2-NEXT:    [[LFTR_WIDEIV:%.*]] = trunc i64 [[INDVARS_IV_NEXT]] to i32
; LMUL2-NEXT:    [[EXITCOND:%.*]] = icmp eq i32 [[LFTR_WIDEIV]], [[SIZE]]
; LMUL2-NEXT:    br i1 [[EXITCOND]], label [[FOR_END_LOOPEXIT]], label [[FOR_BODY]], !llvm.loop [[LOOP3:![0-9]+]]
; LMUL2:       for.end.loopexit:
; LMUL2-NEXT:    br label [[FOR_END]]
; LMUL2:       for.end:
; LMUL2-NEXT:    ret ptr [[C]]
;
entry:
  %cmp10 = icmp sgt i32 %size, 0
  br i1 %cmp10, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  store i32 %add, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %size
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret ptr %c
}
