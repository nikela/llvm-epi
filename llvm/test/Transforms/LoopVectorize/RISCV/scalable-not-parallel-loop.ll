; RUN: opt -mtriple riscv64 -mattr +m,+a,+f,+d,+experimental-v -S \
; RUN:    -loop-vectorize < %s -o - | FileCheck %s --check-prefix=CHECK-IR
; RUN: opt -mtriple riscv64 -mattr +m,+a,+f,+d,+experimental-v -S \
; RUN:    -loop-vectorize \
; RUN:    -prefer-predicate-over-epilogue=predicate-dont-vectorize < %s -o - \
; RUN:    | FileCheck %s --check-prefix=CHECK-IR
; RUN: opt -mtriple riscv64 -mattr +m,+a,+f,+d,+experimental-v -S \
; RUN:    -loop-vectorize -debug-only=loop-vectorize \
; RUN:    < %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-DBG
; RUN: opt -mtriple riscv64 -mattr +m,+a,+f,+d,+experimental-v -S \
; RUN:    -loop-vectorize -debug-only=loop-vectorize \
; RUN:    -prefer-predicate-over-epilogue=predicate-dont-vectorize < %s -o - \
; RUN:    2>&1 | FileCheck %s --check-prefix=CHECK-DBG

; We do not support this vectorization yet (though we plan to).
; CHECK-IR-NOT: vector.body
; CHECK-DBG: LV: Not vectorizing: LV: Scalable vectorization does not support non-infinite distance yet.

; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128-v128:128:128-v256:128:128-v512:128:128-v1024:128:128"
target triple = "riscv64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nounwind
define dso_local void @not_parallel(i32 signext %N, float* noalias nocapture readonly %a, float* noalias nocapture %b) local_unnamed_addr #0 !dbg !8 {
entry:
  %cmp11 = icmp sgt i32 %N, 4, !dbg !10
  br i1 %cmp11, label %for.body.preheader, label %for.cond.cleanup, !dbg !11

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64, !dbg !10
  br label %for.body, !dbg !11

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup, !dbg !12

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void, !dbg !12

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 4, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %0 = add nsw i64 %indvars.iv, -4, !dbg !13
  %arrayidx = getelementptr inbounds float, float* %b, i64 %0, !dbg !14
  %1 = load float, float* %arrayidx, align 4, !dbg !14, !tbaa !15
  %arrayidx2 = getelementptr inbounds float, float* %a, i64 %indvars.iv, !dbg !19
  %2 = load float, float* %arrayidx2, align 4, !dbg !19, !tbaa !15
  %add = fadd float %1, %2, !dbg !20
  %arrayidx4 = getelementptr inbounds float, float* %b, i64 %indvars.iv, !dbg !21
  store float %add, float* %arrayidx4, align 4, !dbg !22, !tbaa !15
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !23
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count, !dbg !10
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body, !dbg !11, !llvm.loop !24
}

attributes #0 = { nofree norecurse nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+d,+experimental-v,+experimental-zvlsseg,+f,+m,-relax,-save-restore" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 13.0.0 (git@repo.hca.bsc.es:EPI/System-Software/llvm-mono.git 707fcbe1bf274b847e455eefc600b61000a4e64f)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/home/rferrer/work/llvm-build")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, !"target-abi", !"lp64d"}
!6 = !{i32 1, !"SmallDataLimit", i32 8}
!7 = !{!"clang version 13.0.0 (git@repo.hca.bsc.es:EPI/System-Software/llvm-mono.git 707fcbe1bf274b847e455eefc600b61000a4e64f)"}
!8 = distinct !DISubprogram(name: "not_parallel", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !2)
!10 = !DILocation(line: 2, column: 21, scope: !8)
!11 = !DILocation(line: 2, column: 3, scope: !8)
!12 = !DILocation(line: 5, column: 1, scope: !8)
!13 = !DILocation(line: 3, column: 16, scope: !8)
!14 = !DILocation(line: 3, column: 12, scope: !8)
!15 = !{!16, !16, i64 0}
!16 = !{!"float", !17, i64 0}
!17 = !{!"omnipotent char", !18, i64 0}
!18 = !{!"Simple C/C++ TBAA"}
!19 = !DILocation(line: 3, column: 23, scope: !8)
!20 = !DILocation(line: 3, column: 21, scope: !8)
!21 = !DILocation(line: 3, column: 5, scope: !8)
!22 = !DILocation(line: 3, column: 10, scope: !8)
!23 = !DILocation(line: 2, column: 27, scope: !8)
!24 = distinct !{!24, !11, !25, !26}
!25 = !DILocation(line: 4, column: 3, scope: !8)
!26 = !{!"llvm.loop.mustprogress"}
