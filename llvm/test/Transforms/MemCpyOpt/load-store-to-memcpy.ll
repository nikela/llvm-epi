; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -basic-aa -scoped-noalias-aa -memcpyopt -S %s -enable-memcpyopt-memoryssa=0 | FileCheck %s --check-prefixes=CHECK,NO_MSSA
; RUN: opt -basic-aa -scoped-noalias-aa -memcpyopt -S %s -enable-memcpyopt-memoryssa=1 -verify-memoryssa | FileCheck %s --check-prefixes=CHECK,MSSA

%T = type { i8, i32 }

; Ensure load-store forwarding of an aggregate is interpreted as
; a memmove when the source and dest may alias
define void @test_memmove(%T* align 8 %a, %T* align 16 %b) {
; CHECK-LABEL: @test_memmove(
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast %T* [[B:%.*]] to i8*
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast %T* [[A:%.*]] to i8*
; CHECK-NEXT:    call void @llvm.memmove.p0i8.p0i8.i64(i8* align 16 [[TMP1]], i8* align 8 [[TMP2]], i64 8, i1 false)
; CHECK-NEXT:    ret void
;
  %val = load %T, %T* %a, align 8
  store %T %val, %T* %b, align 16
  ret void
}

; Ensure load-store forwarding of an aggregate is interpreted as
; a memcpy when the source and dest do not alias
define void @test_memcpy(%T* noalias align 8 %a, %T* noalias align 16 %b) {
; CHECK-LABEL: @test_memcpy(
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast %T* [[B:%.*]] to i8*
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast %T* [[A:%.*]] to i8*
; CHECK-NEXT:    call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP1]], i8* align 8 [[TMP2]], i64 8, i1 false)
; CHECK-NEXT:    ret void
;
  %val = load %T, %T* %a, align 8
  store %T %val, %T* %b, align 16
  ret void
}

; memcpy(%d, %a) should not be generated since store2 may-aliases load %a.
define void @f(%T* %a, %T* %b, %T* %c, %T* %d) {
; CHECK-LABEL: @f(
; CHECK-NEXT:    [[VAL:%.*]] = load [[T:%.*]], %T* [[A:%.*]], align 4, !alias.scope !0
; CHECK-NEXT:    store [[T]] { i8 23, i32 23 }, %T* [[B:%.*]], align 4, !alias.scope !3
; CHECK-NEXT:    store [[T]] { i8 44, i32 44 }, %T* [[C:%.*]], align 4, !alias.scope !6, !noalias !3
; CHECK-NEXT:    store [[T]] [[VAL]], %T* [[D:%.*]], align 4, !alias.scope !9, !noalias !12
; CHECK-NEXT:    ret void
;
  %val = load %T, %T* %a, !alias.scope !{!10}

  ; store1 may-aliases the load
  store %T { i8 23, i32 23 }, %T* %b, !alias.scope !{!11}

  ; store2 may-aliases the load and store3
  store %T { i8 44, i32 44 }, %T* %c, !alias.scope !{!12}, !noalias !{!11}

  ; store3
  store %T %val, %T* %d, !alias.scope !{!13}, !noalias !{!10, !11}
  ret void
}

!0 = !{!0}
!1 = !{!1}
!2 = !{!2}
!3 = !{!3}

!10 = !{ !10, !0 }
!11 = !{ !11, !1 }
!12 = !{ !12, !2 }
!13 = !{ !13, !3 }
