; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -correlated-propagation -S | FileCheck %s

define i8 @simple(i1) {
; CHECK-LABEL: @simple(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[S:%.*]] = select i1 [[TMP0:%.*]], i8 0, i8 1
; CHECK-NEXT:    br i1 [[TMP0]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    ret i8 0
; CHECK:       else:
; CHECK-NEXT:    ret i8 1
;
entry:
  %s = select i1 %0, i8 0, i8 1
  br i1 %0, label %then, label %else

then:
  %a = phi i8 [ %s, %entry ]
  ret i8 %a

else:
  %b = phi i8 [ %s, %entry ]
  ret i8 %b
}

define void @loop(i32) {
; CHECK-LABEL: @loop(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[IDX:%.*]] = phi i32 [ [[TMP0:%.*]], [[ENTRY:%.*]] ], [ [[TMP2:%.*]], [[LOOP]] ]
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i32 [[IDX]], 0
; CHECK-NEXT:    [[TMP2]] = add i32 [[IDX]], -1
; CHECK-NEXT:    [[SEL:%.*]] = select i1 [[TMP1]], i32 0, i32 [[TMP2]]
; CHECK-NEXT:    br i1 [[TMP1]], label [[OUT:%.*]], label [[LOOP]]
; CHECK:       out:
; CHECK-NEXT:    ret void
;
entry:
  br label %loop

loop:
  %idx = phi i32 [ %0, %entry ], [ %sel, %loop ]
  %1 = icmp eq i32 %idx, 0
  %2 = add i32 %idx, -1
  %sel = select i1 %1, i32 0, i32 %2
  br i1 %1, label %out, label %loop

out:
  ret void
}

define i8 @not_correlated(i1, i1) {
; CHECK-LABEL: @not_correlated(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[S:%.*]] = select i1 [[TMP0:%.*]], i8 0, i8 1
; CHECK-NEXT:    br i1 [[TMP1:%.*]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    ret i8 [[S]]
; CHECK:       else:
; CHECK-NEXT:    ret i8 [[S]]
;
entry:
  %s = select i1 %0, i8 0, i8 1
  br i1 %1, label %then, label %else

then:
  %a = phi i8 [ %s, %entry ]
  ret i8 %a

else:
  %b = phi i8 [ %s, %entry ]
  ret i8 %b
}

@c = global i32 0, align 4
@b = global i32 0, align 4

define i32 @PR23752() {
; CHECK-LABEL: @PR23752(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[SEL:%.*]] = select i1 icmp sgt (i32* @b, i32* @c), i32 0, i32 1
; CHECK-NEXT:    [[CMP:%.*]] = icmp ne i32 [[SEL]], 1
; CHECK-NEXT:    br i1 [[CMP]], label [[FOR_BODY]], label [[IF_END:%.*]]
; CHECK:       if.end:
; CHECK-NEXT:    ret i32 1
;
entry:
  br label %for.body

for.body:
  %phi = phi i32 [ 0, %entry ], [ %sel, %for.body ]
  %sel = select i1 icmp sgt (i32* @b, i32* @c), i32 %phi, i32 1
  %cmp = icmp ne i32 %sel, 1
  br i1 %cmp, label %for.body, label %if.end


if.end:
  ret i32 %sel
}

define i1 @test1(i32* %p, i1 %unknown) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    [[PVAL:%.*]] = load i32, i32* [[P:%.*]], align 4
; CHECK-NEXT:    [[CMP1:%.*]] = icmp slt i32 [[PVAL]], 255
; CHECK-NEXT:    br i1 [[CMP1]], label [[NEXT:%.*]], label [[EXIT:%.*]]
; CHECK:       next:
; CHECK-NEXT:    [[MIN:%.*]] = select i1 [[UNKNOWN:%.*]], i32 [[PVAL]], i32 5
; CHECK-NEXT:    ret i1 false
; CHECK:       exit:
; CHECK-NEXT:    ret i1 true
;
  %pval = load i32, i32* %p
  %cmp1 = icmp slt i32 %pval, 255
  br i1 %cmp1, label %next, label %exit

next:
  %min = select i1 %unknown, i32 %pval, i32 5
  %res = icmp eq i32 %min, 255
  ret i1 %res

exit:
  ret i1 true
}

; Check that we take a conservative meet
define i1 @test2(i32* %p, i32 %qval, i1 %unknown) {
; CHECK-LABEL: @test2(
; CHECK-NEXT:    [[PVAL:%.*]] = load i32, i32* [[P:%.*]], align 4
; CHECK-NEXT:    [[CMP1:%.*]] = icmp slt i32 [[PVAL]], 255
; CHECK-NEXT:    br i1 [[CMP1]], label [[NEXT:%.*]], label [[EXIT:%.*]]
; CHECK:       next:
; CHECK-NEXT:    [[MIN:%.*]] = select i1 [[UNKNOWN:%.*]], i32 [[PVAL]], i32 [[QVAL:%.*]]
; CHECK-NEXT:    [[RES:%.*]] = icmp eq i32 [[MIN]], 255
; CHECK-NEXT:    ret i1 [[RES]]
; CHECK:       exit:
; CHECK-NEXT:    ret i1 true
;
  %pval = load i32, i32* %p
  %cmp1 = icmp slt i32 %pval, 255
  br i1 %cmp1, label %next, label %exit

next:
  %min = select i1 %unknown, i32 %pval, i32 %qval
  %res = icmp eq i32 %min, 255
  ret i1 %res

exit:
  ret i1 true
}

; Same as @test2, but for the opposite select input
define i1 @test3(i32* %p, i32 %qval, i1 %unknown) {
; CHECK-LABEL: @test3(
; CHECK-NEXT:    [[PVAL:%.*]] = load i32, i32* [[P:%.*]], align 4
; CHECK-NEXT:    [[CMP1:%.*]] = icmp slt i32 [[PVAL]], 255
; CHECK-NEXT:    br i1 [[CMP1]], label [[NEXT:%.*]], label [[EXIT:%.*]]
; CHECK:       next:
; CHECK-NEXT:    [[MIN:%.*]] = select i1 [[UNKNOWN:%.*]], i32 [[QVAL:%.*]], i32 [[PVAL]]
; CHECK-NEXT:    [[RES:%.*]] = icmp eq i32 [[MIN]], 255
; CHECK-NEXT:    ret i1 [[RES]]
; CHECK:       exit:
; CHECK-NEXT:    ret i1 true
;
  %pval = load i32, i32* %p
  %cmp1 = icmp slt i32 %pval, 255
  br i1 %cmp1, label %next, label %exit

next:
  %min = select i1 %unknown, i32 %qval, i32 %pval
  %res = icmp eq i32 %min, 255
  ret i1 %res

exit:
  ret i1 true
}

; Conflicting constants (i.e. isOverdefined result)
; NOTE: Using doubles in this version is a bit of a hack.  This
; is to get around the fact that all integers (including constants
; and non-constants) are actually represented as constant-ranges.
define i1 @test4(i32* %p, i32 %qval, i1 %unknown) {
; CHECK-LABEL: @test4(
; CHECK-NEXT:    [[PVAL:%.*]] = load i32, i32* [[P:%.*]], align 4
; CHECK-NEXT:    [[CMP1:%.*]] = icmp slt i32 [[PVAL]], 255
; CHECK-NEXT:    br i1 [[CMP1]], label [[NEXT:%.*]], label [[EXIT:%.*]]
; CHECK:       next:
; CHECK-NEXT:    [[MIN:%.*]] = select i1 [[UNKNOWN:%.*]], double 1.000000e+00, double 0.000000e+00
; CHECK-NEXT:    [[RES:%.*]] = fcmp oeq double [[MIN]], 3.000000e+02
; CHECK-NEXT:    ret i1 [[RES]]
; CHECK:       exit:
; CHECK-NEXT:    ret i1 true
;
  %pval = load i32, i32* %p
  %cmp1 = icmp slt i32 %pval, 255
  br i1 %cmp1, label %next, label %exit

next:
  %min = select i1 %unknown, double 1.0, double 0.0
  %res = fcmp oeq double %min, 300.0
  ret i1 %res

exit:
  ret i1 true
}

;; Using the condition to clamp the result
;;

define i1 @test5(i32* %p, i1 %unknown) {
; CHECK-LABEL: @test5(
; CHECK-NEXT:    [[PVAL:%.*]] = load i32, i32* [[P:%.*]], align 4
; CHECK-NEXT:    [[CMP1:%.*]] = icmp slt i32 [[PVAL]], 255
; CHECK-NEXT:    br i1 [[CMP1]], label [[NEXT:%.*]], label [[EXIT:%.*]]
; CHECK:       next:
; CHECK-NEXT:    [[COND:%.*]] = icmp sgt i32 [[PVAL]], 0
; CHECK-NEXT:    [[MIN:%.*]] = select i1 [[COND]], i32 [[PVAL]], i32 5
; CHECK-NEXT:    ret i1 false
; CHECK:       exit:
; CHECK-NEXT:    ret i1 true
;
  %pval = load i32, i32* %p
  %cmp1 = icmp slt i32 %pval, 255
  br i1 %cmp1, label %next, label %exit

next:
  %cond = icmp sgt i32 %pval, 0
  %min = select i1 %cond, i32 %pval, i32 5
  %res = icmp eq i32 %min, -1
  ret i1 %res

exit:
  ret i1 true
}

define i1 @test6(i32* %p, i1 %unknown) {
; CHECK-LABEL: @test6(
; CHECK-NEXT:    [[PVAL:%.*]] = load i32, i32* [[P:%.*]], align 4
; CHECK-NEXT:    [[CMP1:%.*]] = icmp ult i32 [[PVAL]], 255
; CHECK-NEXT:    br i1 [[CMP1]], label [[NEXT:%.*]], label [[EXIT:%.*]]
; CHECK:       next:
; CHECK-NEXT:    [[COND:%.*]] = icmp ne i32 [[PVAL]], 254
; CHECK-NEXT:    [[SEL:%.*]] = select i1 [[COND]], i32 [[PVAL]], i32 1
; CHECK-NEXT:    ret i1 true
; CHECK:       exit:
; CHECK-NEXT:    ret i1 true
;
  %pval = load i32, i32* %p
  %cmp1 = icmp ult i32 %pval, 255
  br i1 %cmp1, label %next, label %exit

next:
  %cond = icmp ne i32 %pval, 254
  %sel = select i1 %cond, i32 %pval, i32 1
  %res = icmp slt i32 %sel, 254
  ret i1 %res

exit:
  ret i1 true
}
