; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -indvars -S | FileCheck %s

define void @test1() nounwind {
; CHECK-LABEL: @test1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[BB:%.*]]
; CHECK:       bb:
; CHECK-NEXT:    [[IV_INT:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[DOTINT:%.*]], [[BB]] ]
; CHECK-NEXT:    [[INDVAR_CONV:%.*]] = sitofp i32 [[IV_INT]] to double
; CHECK-NEXT:    [[TMP0:%.*]] = tail call i32 @foo(double [[INDVAR_CONV]]) [[ATTR0:#.*]]
; CHECK-NEXT:    [[DOTINT]] = add nuw nsw i32 [[IV_INT]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult i32 [[DOTINT]], 10000
; CHECK-NEXT:    br i1 [[TMP1]], label [[BB]], label [[RETURN:%.*]]
; CHECK:       return:
; CHECK-NEXT:    ret void
;
entry:
  br label %bb

bb:		; preds = %bb, %entry
  %iv = phi double [ 0.000000e+00, %entry ], [ %1, %bb ]
  %0 = tail call i32 @foo(double %iv) nounwind
  %1 = fadd double %iv, 1.000000e+00
  %2 = fcmp olt double %1, 1.000000e+04
  br i1 %2, label %bb, label %return

return:		; preds = %bb
  ret void
}

declare i32 @foo(double)

define void @test2() nounwind {
; CHECK-LABEL: @test2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[BB:%.*]]
; CHECK:       bb:
; CHECK-NEXT:    [[IV_INT:%.*]] = phi i32 [ -10, [[ENTRY:%.*]] ], [ [[DOTINT:%.*]], [[BB]] ]
; CHECK-NEXT:    [[INDVAR_CONV:%.*]] = sitofp i32 [[IV_INT]] to double
; CHECK-NEXT:    [[TMP0:%.*]] = tail call i32 @foo(double [[INDVAR_CONV]]) [[ATTR0]]
; CHECK-NEXT:    [[DOTINT]] = add nsw i32 [[IV_INT]], 2
; CHECK-NEXT:    [[TMP1:%.*]] = icmp slt i32 [[DOTINT]], -1
; CHECK-NEXT:    br i1 [[TMP1]], label [[BB]], label [[RETURN:%.*]]
; CHECK:       return:
; CHECK-NEXT:    ret void
;
entry:
  br label %bb

bb:		; preds = %bb, %entry
  %iv = phi double [ -10.000000e+00, %entry ], [ %1, %bb ]
  %0 = tail call i32 @foo(double %iv) nounwind
  %1 = fadd double %iv, 2.000000e+00
  %2 = fcmp olt double %1, -1.000000e+00
  br i1 %2, label %bb, label %return

return:		; preds = %bb
  ret void
}


define void @test3() nounwind {
; CHECK-LABEL: @test3(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[BB:%.*]]
; CHECK:       bb:
; CHECK-NEXT:    [[IV:%.*]] = phi double [ 0.000000e+00, [[ENTRY:%.*]] ], [ [[TMP1:%.*]], [[BB]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = tail call i32 @foo(double [[IV]]) [[ATTR0]]
; CHECK-NEXT:    [[TMP1]] = fadd double [[IV]], 1.000000e+00
; CHECK-NEXT:    br i1 false, label [[BB]], label [[RETURN:%.*]]
; CHECK:       return:
; CHECK-NEXT:    ret void
;
entry:
  br label %bb

bb:		; preds = %bb, %entry
  %iv = phi double [ 0.000000e+00, %entry ], [ %1, %bb ]
  %0 = tail call i32 @foo(double %iv) nounwind
  %1 = fadd double %iv, 1.000000e+00
  %2 = fcmp olt double %1, -1.000000e+00
  br i1 %2, label %bb, label %return

return:
  ret void
}

define void @test4() nounwind {
; CHECK-LABEL: @test4(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[BB:%.*]]
; CHECK:       bb:
; CHECK-NEXT:    [[IV_INT:%.*]] = phi i32 [ 40, [[ENTRY:%.*]] ], [ [[DOTINT:%.*]], [[BB]] ]
; CHECK-NEXT:    [[INDVAR_CONV:%.*]] = sitofp i32 [[IV_INT]] to double
; CHECK-NEXT:    [[TMP0:%.*]] = tail call i32 @foo(double [[INDVAR_CONV]]) [[ATTR0]]
; CHECK-NEXT:    [[DOTINT]] = add nsw i32 [[IV_INT]], -1
; CHECK-NEXT:    br i1 false, label [[BB]], label [[RETURN:%.*]]
; CHECK:       return:
; CHECK-NEXT:    ret void
;
entry:
  br label %bb

bb:		; preds = %bb, %entry
  %iv = phi double [ 40.000000e+00, %entry ], [ %1, %bb ]
  %0 = tail call i32 @foo(double %iv) nounwind
  %1 = fadd double %iv, -1.000000e+00
  %2 = fcmp olt double %1, 1.000000e+00
  br i1 %2, label %bb, label %return

return:
  ret void
}

; PR6761
define void @test5() nounwind {
; <label>:0
; CHECK-LABEL: @test5(
; CHECK-NEXT:    br label [[TMP1:%.*]]
; CHECK:       1:
; CHECK-NEXT:    [[DOTINT:%.*]] = phi i32 [ 9, [[TMP0:%.*]] ], [ [[DOTINT1:%.*]], [[TMP1]] ]
; CHECK-NEXT:    [[TMP2:%.*]] = tail call i32 @foo(double 0.000000e+00)
; CHECK-NEXT:    [[DOTINT1]] = add nsw i32 [[DOTINT]], -1
; CHECK-NEXT:    [[TMP3:%.*]] = icmp slt i32 [[DOTINT1]], 0
; CHECK-NEXT:    br i1 [[TMP3]], label [[EXIT:%.*]], label [[TMP1]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
  br label %1

; <label>:1
  %2 = phi double [ 9.000000e+00, %0 ], [ %4, %1 ]
  %3 = tail call i32 @foo(double 0.0)
  %4 = fadd double %2, -1.000000e+00
  %5 = fcmp ult double %4, 0.000000e+00
  br i1 %5, label %exit, label %1

exit:
  ret void
}

define double @test_max_be() {
; CHECK-LABEL: @test_max_be(
; CHECK-NEXT:  bb4:
; CHECK-NEXT:    br label [[BB8:%.*]]
; CHECK:       bb8:
; CHECK-NEXT:    [[TMP10:%.*]] = phi double [ 0.000000e+00, [[BB4:%.*]] ], [ [[TMP12:%.*]], [[BB22:%.*]] ]
; CHECK-NEXT:    [[TMP11_INT:%.*]] = phi i32 [ 0, [[BB4]] ], [ [[TMP13_INT:%.*]], [[BB22]] ]
; CHECK-NEXT:    [[INDVAR_CONV:%.*]] = sitofp i32 [[TMP11_INT]] to double
; CHECK-NEXT:    [[TMP12]] = fadd double [[TMP10]], [[INDVAR_CONV]]
; CHECK-NEXT:    [[TMP13_INT]] = add nuw nsw i32 [[TMP11_INT]], 1
; CHECK-NEXT:    [[TMP14:%.*]] = icmp slt i32 [[TMP13_INT]], 99999
; CHECK-NEXT:    br i1 [[TMP14]], label [[BB22]], label [[BB6:%.*]]
; CHECK:       bb22:
; CHECK-NEXT:    br i1 true, label [[BB8]], label [[BB6]]
; CHECK:       bb6:
; CHECK-NEXT:    [[TMP12_LCSSA:%.*]] = phi double [ [[TMP12]], [[BB22]] ], [ [[TMP12]], [[BB8]] ]
; CHECK-NEXT:    ret double [[TMP12_LCSSA]]
;
bb4:
  br label %bb8

bb8:
  %tmp9 = phi i64 [ 1, %bb4 ], [ %tmp23, %bb22 ]
  %tmp10 = phi double [ 0.000000e+00, %bb4 ], [ %tmp12, %bb22 ]
  %tmp11 = phi double [ 0.000000e+00, %bb4 ], [ %tmp13, %bb22 ]
  %tmp12 = fadd double %tmp10, %tmp11
  %tmp13 = fadd double %tmp11, 1.000000e+00
  %tmp14 = fcmp olt double %tmp13, 9.999900e+04
  br i1 %tmp14, label %bb22, label %bb6

bb22:
  %tmp23 = add nuw nsw i64 %tmp9, 1
  %tmp24 = icmp ult i64 %tmp9, 1048576
  br i1 %tmp24, label %bb8, label %bb6

bb6:
  ret double %tmp12
}

define float @test_max_be2() {
; CHECK-LABEL: @test_max_be2(
; CHECK-NEXT:  bb4:
; CHECK-NEXT:    br label [[BB8:%.*]]
; CHECK:       bb8:
; CHECK-NEXT:    [[TMP10:%.*]] = phi float [ 0.000000e+00, [[BB4:%.*]] ], [ [[TMP12:%.*]], [[BB22:%.*]] ]
; CHECK-NEXT:    [[TMP11_INT:%.*]] = phi i32 [ 0, [[BB4]] ], [ [[TMP13_INT:%.*]], [[BB22]] ]
; CHECK-NEXT:    [[INDVAR_CONV:%.*]] = sitofp i32 [[TMP11_INT]] to float
; CHECK-NEXT:    [[TMP12]] = fadd float [[TMP10]], [[INDVAR_CONV]]
; CHECK-NEXT:    [[TMP13_INT]] = add nuw nsw i32 [[TMP11_INT]], 1
; CHECK-NEXT:    [[TMP14:%.*]] = icmp slt i32 [[TMP13_INT]], 99999
; CHECK-NEXT:    br i1 [[TMP14]], label [[BB22]], label [[BB6:%.*]]
; CHECK:       bb22:
; CHECK-NEXT:    br i1 true, label [[BB8]], label [[BB6]]
; CHECK:       bb6:
; CHECK-NEXT:    [[TMP12_LCSSA:%.*]] = phi float [ [[TMP12]], [[BB22]] ], [ [[TMP12]], [[BB8]] ]
; CHECK-NEXT:    ret float [[TMP12_LCSSA]]
;
bb4:
  br label %bb8

bb8:
  %tmp9 = phi i64 [ 1, %bb4 ], [ %tmp23, %bb22 ]
  %tmp10 = phi float [ 0.000000e+00, %bb4 ], [ %tmp12, %bb22 ]
  %tmp11 = phi float [ 0.000000e+00, %bb4 ], [ %tmp13, %bb22 ]
  %tmp12 = fadd float %tmp10, %tmp11
  %tmp13 = fadd float %tmp11, 1.000000e+00
  %tmp14 = fcmp olt float %tmp13, 9.999900e+04
  br i1 %tmp14, label %bb22, label %bb6

bb22:
  %tmp23 = add nuw nsw i64 %tmp9, 1
  %tmp24 = icmp ult i64 %tmp9, 1048576
  br i1 %tmp24, label %bb8, label %bb6

bb6:
  ret float %tmp12
}

; Bounds check
define float @test_max_be3() {
; CHECK-LABEL: @test_max_be3(
; CHECK-NEXT:  bb4:
; CHECK-NEXT:    br label [[BB8:%.*]]
; CHECK:       bb8:
; CHECK-NEXT:    [[TMP10:%.*]] = phi float [ 0.000000e+00, [[BB4:%.*]] ], [ [[TMP12:%.*]], [[BB22:%.*]] ]
; CHECK-NEXT:    [[TMP11_INT:%.*]] = phi i32 [ 0, [[BB4]] ], [ [[TMP13_INT:%.*]], [[BB22]] ]
; CHECK-NEXT:    [[INDVAR_CONV:%.*]] = sitofp i32 [[TMP11_INT]] to float
; CHECK-NEXT:    [[TMP12]] = fadd float [[TMP10]], [[INDVAR_CONV]]
; CHECK-NEXT:    [[TMP13_INT]] = add nuw nsw i32 [[TMP11_INT]], 1
; CHECK-NEXT:    [[TMP14:%.*]] = icmp slt i32 [[TMP13_INT]], 99999
; CHECK-NEXT:    br i1 [[TMP14]], label [[BB22]], label [[BB6:%.*]]
; CHECK:       bb22:
; CHECK-NEXT:    br i1 true, label [[BB8]], label [[BB6]]
; CHECK:       bb6:
; CHECK-NEXT:    [[TMP12_LCSSA:%.*]] = phi float [ [[TMP12]], [[BB22]] ], [ [[TMP12]], [[BB8]] ]
; CHECK-NEXT:    ret float [[TMP12_LCSSA]]
;
bb4:
  br label %bb8

bb8:
  %tmp9 = phi i64 [ 1, %bb4 ], [ %tmp23, %bb22 ]
  %tmp10 = phi float [ 0.000000e+00, %bb4 ], [ %tmp12, %bb22 ]
  %tmp11 = phi float [ 0.000000e+00, %bb4 ], [ %tmp13, %bb22 ]
  %tmp12 = fadd float %tmp10, %tmp11
  %tmp13 = fadd float %tmp11, 1.000000e+00
  %tmp14 = fcmp olt float %tmp13, 9.999900e+04
  br i1 %tmp14, label %bb22, label %bb6

bb22:
  %tmp23 = add nuw nsw i64 %tmp9, 1
  ;; 2^23 = 16777215
  %tmp24 = icmp ult i64 %tmp9, 16777215
  br i1 %tmp24, label %bb8, label %bb6

bb6:
  ret float %tmp12
}


; Show that given a computeable exit count, we can remove an
; fcmp of a casted integer IV. (TODO)
define void @fcmp1() nounwind {
; CHECK-LABEL: @fcmp1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[BB:%.*]]
; CHECK:       bb:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[IV_NEXT:%.*]], [[BACKEDGE:%.*]] ]
; CHECK-NEXT:    [[CMP1:%.*]] = icmp ult i64 [[IV]], 20000
; CHECK-NEXT:    br i1 [[CMP1]], label [[BACKEDGE]], label [[RETURN:%.*]]
; CHECK:       backedge:
; CHECK-NEXT:    [[IV_FP:%.*]] = sitofp i64 [[IV]] to double
; CHECK-NEXT:    [[TMP0:%.*]] = tail call i32 @foo(double [[IV_FP]]) [[ATTR0]]
; CHECK-NEXT:    [[IV_NEXT]] = add nuw nsw i64 [[IV]], 1
; CHECK-NEXT:    [[CMP2:%.*]] = fcmp olt double [[IV_FP]], 1.000000e+04
; CHECK-NEXT:    br i1 [[CMP2]], label [[BB]], label [[RETURN]]
; CHECK:       return:
; CHECK-NEXT:    ret void
;
entry:
  br label %bb

bb:		; preds = %bb, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %cmp1 = icmp slt i64 %iv, 20000
  br i1 %cmp1, label %backedge, label %return

backedge:
  %iv.fp = sitofp i64 %iv to double
  tail call i32 @foo(double %iv.fp) nounwind
  %iv.next = add nsw nuw i64 %iv, 1
  %cmp2 = fcmp olt double %iv.fp, 1.000000e+04
  br i1 %cmp2, label %bb, label %return

return:		; preds = %bb
  ret void
}

define void @fcmp2() nounwind {
; CHECK-LABEL: @fcmp2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[BB:%.*]]
; CHECK:       bb:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[IV_NEXT:%.*]], [[BACKEDGE:%.*]] ]
; CHECK-NEXT:    [[CMP1:%.*]] = icmp ult i64 [[IV]], 2000
; CHECK-NEXT:    br i1 [[CMP1]], label [[BACKEDGE]], label [[RETURN:%.*]]
; CHECK:       backedge:
; CHECK-NEXT:    [[IV_FP:%.*]] = sitofp i64 [[IV]] to double
; CHECK-NEXT:    [[TMP0:%.*]] = tail call i32 @foo(double [[IV_FP]]) [[ATTR0]]
; CHECK-NEXT:    [[IV_NEXT]] = add nuw nsw i64 [[IV]], 1
; CHECK-NEXT:    [[CMP2:%.*]] = fcmp olt double [[IV_FP]], 1.000000e+04
; CHECK-NEXT:    br i1 [[CMP2]], label [[BB]], label [[RETURN]]
; CHECK:       return:
; CHECK-NEXT:    ret void
;
entry:
  br label %bb

bb:		; preds = %bb, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  %cmp1 = icmp slt i64 %iv, 2000
  br i1 %cmp1, label %backedge, label %return

backedge:
  %iv.fp = sitofp i64 %iv to double
  tail call i32 @foo(double %iv.fp) nounwind
  %iv.next = add nsw nuw i64 %iv, 1
  %cmp2 = fcmp olt double %iv.fp, 1.000000e+04
  br i1 %cmp2, label %bb, label %return

return:		; preds = %bb
  ret void
}

define void @fcmp_neg1() nounwind {
; CHECK-LABEL: @fcmp_neg1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[BB:%.*]]
; CHECK:       bb:
; CHECK-NEXT:    [[IV:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[IV_NEXT:%.*]], [[BACKEDGE:%.*]] ]
; CHECK-NEXT:    [[CMP1:%.*]] = icmp ult i64 [[IV]], -20
; CHECK-NEXT:    br i1 [[CMP1]], label [[BACKEDGE]], label [[RETURN:%.*]]
; CHECK:       backedge:
; CHECK-NEXT:    [[IV_FP:%.*]] = sitofp i64 [[IV]] to double
; CHECK-NEXT:    [[TMP0:%.*]] = tail call i32 @foo(double [[IV_FP]]) [[ATTR0]]
; CHECK-NEXT:    [[IV_NEXT]] = add nuw i64 [[IV]], 1
; CHECK-NEXT:    [[CMP2:%.*]] = fcmp olt double [[IV_FP]], 1.000000e+04
; CHECK-NEXT:    br i1 [[CMP2]], label [[BB]], label [[RETURN]]
; CHECK:       return:
; CHECK-NEXT:    ret void
;
entry:
  br label %bb

bb:		; preds = %bb, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %backedge ]
  ;; Range fact outside precise integer region
  %cmp1 = icmp ult i64 %iv, -20
  br i1 %cmp1, label %backedge, label %return

backedge:
  %iv.fp = sitofp i64 %iv to double
  tail call i32 @foo(double %iv.fp) nounwind
  %iv.next = add nuw i64 %iv, 1
  %cmp2 = fcmp olt double %iv.fp, 1.000000e+04
  br i1 %cmp2, label %bb, label %return

return:		; preds = %bb
  ret void
}
