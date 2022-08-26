; RUN: opt -S -loop-vectorize < %s -o - 2>&1 | FileCheck %s

; ModuleID = 'custom/LoopVectorize/declare_simd-wrong_simdlen.c'
source_filename = "custom/LoopVectorize/declare_simd-wrong_simdlen.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-linux-gnu"

; Function Attrs: nounwind
define dso_local void @inbranch(i64 noundef %N, ptr nocapture noundef %C, ptr nocapture noundef readonly %A, ptr nocapture noundef readonly %B) local_unnamed_addr #0 {
entry:
  %cmp = icmp sgt i64 %N, 0
  br i1 %cmp, label %omp.inner.for.body.preheader, label %simd.if.end

omp.inner.for.body.preheader:                     ; preds = %entry
  br label %omp.inner.for.body

omp.inner.for.body:                               ; preds = %omp.inner.for.body.preheader, %omp.inner.for.inc
  %.omp.iv.025 = phi i64 [ %add11, %omp.inner.for.inc ], [ 0, %omp.inner.for.body.preheader ]
  %arrayidx = getelementptr inbounds float, ptr %C, i64 %.omp.iv.025
  %0 = load float, ptr %arrayidx, align 4, !tbaa !5, !llvm.access.group !9
  %cmp7 = fcmp ogt float %0, 4.200000e+01
  br i1 %cmp7, label %if.then, label %omp.inner.for.inc

if.then:                                          ; preds = %omp.inner.for.body
  %arrayidx8 = getelementptr inbounds float, ptr %A, i64 %.omp.iv.025
  %1 = load float, ptr %arrayidx8, align 4, !tbaa !5, !llvm.access.group !9
  %arrayidx9 = getelementptr inbounds float, ptr %B, i64 %.omp.iv.025
  %2 = load float, ptr %arrayidx9, align 4, !tbaa !5, !llvm.access.group !9
  %call = tail call float @foo(float noundef %1, float noundef %2) #2, !llvm.access.group !9
  store float %call, ptr %arrayidx, align 4, !tbaa !5, !llvm.access.group !9
  br label %omp.inner.for.inc

omp.inner.for.inc:                                ; preds = %if.then, %omp.inner.for.body
  %add11 = add nuw nsw i64 %.omp.iv.025, 1
  %exitcond.not = icmp eq i64 %add11, %N
  br i1 %exitcond.not, label %simd.if.end.loopexit, label %omp.inner.for.body, !llvm.loop !10

simd.if.end.loopexit:                             ; preds = %omp.inner.for.inc
  br label %simd.if.end

simd.if.end:                                      ; preds = %simd.if.end.loopexit, %entry
  ret void
}

; CHECK:      remark: <unknown>:0:0: UserVF ignored because of invalid costs.
; CHECK-NEXT: remark: <unknown>:0:0: Instruction with invalid costs prevented vectorization at VF=(vscale x 1): call to foo

; Function Attrs: nounwind
define dso_local void @notinbranch(i64 noundef %N, ptr nocapture noundef writeonly %C, ptr nocapture noundef readonly %A, ptr nocapture noundef readonly %B) local_unnamed_addr #0 {
entry:
  %cmp = icmp sgt i64 %N, 0
  br i1 %cmp, label %omp.inner.for.body.preheader, label %simd.if.end

omp.inner.for.body.preheader:                     ; preds = %entry
  br label %omp.inner.for.body

omp.inner.for.body:                               ; preds = %omp.inner.for.body.preheader, %omp.inner.for.body
  %.omp.iv.021 = phi i64 [ %add9, %omp.inner.for.body ], [ 0, %omp.inner.for.body.preheader ]
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %.omp.iv.021
  %0 = load float, ptr %arrayidx, align 4, !tbaa !5, !llvm.access.group !9
  %arrayidx7 = getelementptr inbounds float, ptr %B, i64 %.omp.iv.021
  %1 = load float, ptr %arrayidx7, align 4, !tbaa !5, !llvm.access.group !9
  %call = tail call float @foo(float noundef %0, float noundef %1) #2, !llvm.access.group !9
  %arrayidx8 = getelementptr inbounds float, ptr %C, i64 %.omp.iv.021
  store float %call, ptr %arrayidx8, align 4, !tbaa !5, !llvm.access.group !9
  %add9 = add nuw nsw i64 %.omp.iv.021, 1
  %exitcond.not = icmp eq i64 %add9, %N
  br i1 %exitcond.not, label %simd.if.end.loopexit, label %omp.inner.for.body, !llvm.loop !10

simd.if.end.loopexit:                             ; preds = %omp.inner.for.body
  br label %simd.if.end

simd.if.end:                                      ; preds = %simd.if.end.loopexit, %entry
  ret void
}

; CHECK:      remark: <unknown>:0:0: UserVF ignored because of invalid costs.
; CHECK-NEXT: remark: <unknown>:0:0: Instruction with invalid costs prevented vectorization at VF=(vscale x 1): call to foo

declare dso_local float @foo(float noundef, float noundef) local_unnamed_addr #1

declare dso_local <vscale x 4 x float> @_ZGVEMk4vv_foo(<vscale x 4 x float> noundef, <vscale x 4 x float> noundef, <vscale x 4 x i1>, i32 zeroext) local_unnamed_addr #1

declare dso_local <vscale x 4 x float> @_ZGVENk4vv_foo(<vscale x 4 x float> noundef, <vscale x 4 x float> noundef, i32 zeroext) local_unnamed_addr #1

attributes #0 = { nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+a,+c,+d,+f,+m,+zepi,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl32b,+zvl64b,-relax,-save-restore" }
attributes #1 = { "_ZGVEMk4vv_foo" "_ZGVENk4vv_foo" "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+a,+c,+d,+f,+m,+zepi,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl32b,+zvl64b,-relax,-save-restore" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}
!nvvm.annotations = !{}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64d"}
!2 = !{i32 7, !"openmp", i32 50}
!3 = !{i32 1, !"SmallDataLimit", i32 8}
!4 = !{!"clang version 15.0.0"}
!5 = !{!6, !6, i64 0}
!6 = !{!"float", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{}
!10 = distinct !{!10, !11, !12, !13, !14}
!11 = !{!"llvm.loop.parallel_accesses", !9}
!12 = !{!"llvm.loop.vectorize.width", i32 1}
!13 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!14 = !{!"llvm.loop.vectorize.enable", i1 true}
