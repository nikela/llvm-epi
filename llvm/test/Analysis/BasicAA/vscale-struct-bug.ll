; RUN: opt %s -O3 -S | FileCheck %s
;;; failed AA will cause MemoryDependenceAnalysis
;;; give wrong info to DSE and then removes dead store
;;; store i32* %i, i32** %p, align 8, !tbaa !4
;;; store i32* %j, i32** %q, align 8, !tbaa !7
;;; finally INSTCOMBINE optimizatopn will generate "ret 1"

%struct.Foo = type { i32*, i32* }

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #0

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #0

declare dso_local void @bar(i32** nocapture readonly) local_unnamed_addr

define dso_local signext i32 @foo() local_unnamed_addr {
entry:
  %f = alloca %struct.Foo, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %0 = bitcast %struct.Foo* %f to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0)
  %1 = bitcast i32* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %1)
  store i32 1, i32* %i, align 4, !tbaa !0
  %2 = bitcast i32* %j to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %2)
  store i32 2, i32* %j, align 4, !tbaa !0
  %p = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 0
  store i32* %i, i32** %p, align 8, !tbaa !4
  %q = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i64 0, i32 1
  store i32* %j, i32** %q, align 8, !tbaa !7
  call void @bar(i32** nonnull %q)
  %3 = load i32, i32* %i, align 4, !tbaa !0
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %2)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %1)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0)
; CHECK: foo
; CHECK: ret i32 %3
  ret i32 %3
}

attributes #0 = { argmemonly nounwind willreturn }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !6, i64 0}
!5 = !{!"Foo", !6, i64 0, !6, i64 8}
!6 = !{!"any pointer", !2, i64 0}
!7 = !{!5, !6, i64 8}
