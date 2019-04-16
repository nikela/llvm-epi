; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S -correlated-propagation < %s | FileCheck %s

; Check that debug locations are preserved. For more info see:
;   https://llvm.org/docs/SourceLevelDebugging.html#fixing-errors
; RUN: opt < %s -enable-debugify -correlated-propagation -S 2>&1 | \
; RUN:   FileCheck %s -check-prefix=DEBUG
; DEBUG: CheckModuleDebugify: PASS

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)

declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32)

declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32)

declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32)

declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32)

declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32)

declare void @llvm.trap()


define i32 @signed_add(i32 %x, i32 %y) {
; CHECK-LABEL: @signed_add(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp sgt i32 [[Y:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[LAND_LHS_TRUE:%.*]], label [[LOR_LHS_FALSE:%.*]]
; CHECK:       land.lhs.true:
; CHECK-NEXT:    [[TMP0:%.*]] = sub nsw i32 2147483647, [[Y]]
; CHECK-NEXT:    [[TMP1:%.*]] = insertvalue { i32, i1 } undef, i32 [[TMP0]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = insertvalue { i32, i1 } [[TMP1]], i1 false, 1
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue { i32, i1 } [[TMP2]], 1
; CHECK-NEXT:    br i1 [[TMP3]], label [[TRAP:%.*]], label [[CONT:%.*]]
; CHECK:       trap:
; CHECK-NEXT:    tail call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK:       cont:
; CHECK-NEXT:    [[TMP4:%.*]] = extractvalue { i32, i1 } [[TMP2]], 0
; CHECK-NEXT:    [[CMP1:%.*]] = icmp slt i32 [[TMP4]], [[X:%.*]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[COND_END:%.*]], label [[COND_FALSE:%.*]]
; CHECK:       lor.lhs.false:
; CHECK-NEXT:    [[CMP2:%.*]] = icmp slt i32 [[Y]], 0
; CHECK-NEXT:    br i1 [[CMP2]], label [[LAND_LHS_TRUE3:%.*]], label [[COND_FALSE]]
; CHECK:       land.lhs.true3:
; CHECK-NEXT:    [[TMP5:%.*]] = sub nsw i32 -2147483648, [[Y]]
; CHECK-NEXT:    [[TMP6:%.*]] = insertvalue { i32, i1 } undef, i32 [[TMP5]], 0
; CHECK-NEXT:    [[TMP7:%.*]] = insertvalue { i32, i1 } [[TMP6]], i1 false, 1
; CHECK-NEXT:    [[TMP8:%.*]] = extractvalue { i32, i1 } [[TMP7]], 1
; CHECK-NEXT:    br i1 [[TMP8]], label [[TRAP]], label [[CONT4:%.*]]
; CHECK:       cont4:
; CHECK-NEXT:    [[TMP9:%.*]] = extractvalue { i32, i1 } [[TMP7]], 0
; CHECK-NEXT:    [[CMP5:%.*]] = icmp sgt i32 [[TMP9]], [[X]]
; CHECK-NEXT:    br i1 [[CMP5]], label [[COND_END]], label [[COND_FALSE]]
; CHECK:       cond.false:
; CHECK-NEXT:    [[TMP10:%.*]] = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 [[X]], i32 [[Y]])
; CHECK-NEXT:    [[TMP11:%.*]] = extractvalue { i32, i1 } [[TMP10]], 0
; CHECK-NEXT:    [[TMP12:%.*]] = extractvalue { i32, i1 } [[TMP10]], 1
; CHECK-NEXT:    br i1 [[TMP12]], label [[TRAP]], label [[COND_END]]
; CHECK:       cond.end:
; CHECK-NEXT:    [[COND:%.*]] = phi i32 [ 0, [[CONT4]] ], [ 0, [[CONT]] ], [ [[TMP11]], [[COND_FALSE]] ]
; CHECK-NEXT:    ret i32 [[COND]]
;
entry:
  %cmp = icmp sgt i32 %y, 0
  br i1 %cmp, label %land.lhs.true, label %lor.lhs.false

land.lhs.true:                                    ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 2147483647, i32 %y)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:                                             ; preds = %land.lhs.true, %land.lhs.true3, %cond.false
  tail call void @llvm.trap()
  unreachable

cont:                                             ; preds = %land.lhs.true
  %2 = extractvalue { i32, i1 } %0, 0
  %cmp1 = icmp slt i32 %2, %x
  br i1 %cmp1, label %cond.end, label %cond.false

lor.lhs.false:                                    ; preds = %entry
  %cmp2 = icmp slt i32 %y, 0
  br i1 %cmp2, label %land.lhs.true3, label %cond.false

land.lhs.true3:                                   ; preds = %lor.lhs.false
  %3 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 -2147483648, i32 %y)
  %4 = extractvalue { i32, i1 } %3, 1
  br i1 %4, label %trap, label %cont4

cont4:                                            ; preds = %land.lhs.true3
  %5 = extractvalue { i32, i1 } %3, 0
  %cmp5 = icmp sgt i32 %5, %x
  br i1 %cmp5, label %cond.end, label %cond.false

cond.false:                                       ; preds = %cont, %cont4, %lor.lhs.false
  %6 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %x, i32 %y)
  %7 = extractvalue { i32, i1 } %6, 0
  %8 = extractvalue { i32, i1 } %6, 1
  br i1 %8, label %trap, label %cond.end

cond.end:                                         ; preds = %cond.false, %cont, %cont4
  %cond = phi i32 [ 0, %cont4 ], [ 0, %cont ], [ %7, %cond.false ]
  ret i32 %cond
}

define i32 @unsigned_add(i32 %x, i32 %y) {
; CHECK-LABEL: @unsigned_add(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = sub nuw i32 -1, [[Y:%.*]]
; CHECK-NEXT:    [[TMP1:%.*]] = insertvalue { i32, i1 } undef, i32 [[TMP0]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = insertvalue { i32, i1 } [[TMP1]], i1 false, 1
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue { i32, i1 } [[TMP2]], 1
; CHECK-NEXT:    br i1 [[TMP3]], label [[TRAP:%.*]], label [[CONT:%.*]]
; CHECK:       trap:
; CHECK-NEXT:    tail call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK:       cont:
; CHECK-NEXT:    [[TMP4:%.*]] = extractvalue { i32, i1 } [[TMP2]], 0
; CHECK-NEXT:    [[CMP1:%.*]] = icmp ult i32 [[TMP4]], [[X:%.*]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[COND_END:%.*]], label [[COND_FALSE:%.*]]
; CHECK:       cond.false:
; CHECK-NEXT:    [[TMP5:%.*]] = tail call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 [[X]], i32 [[Y]])
; CHECK-NEXT:    [[TMP6:%.*]] = extractvalue { i32, i1 } [[TMP5]], 0
; CHECK-NEXT:    [[TMP7:%.*]] = extractvalue { i32, i1 } [[TMP5]], 1
; CHECK-NEXT:    br i1 [[TMP7]], label [[TRAP]], label [[COND_END]]
; CHECK:       cond.end:
; CHECK-NEXT:    [[COND:%.*]] = phi i32 [ 0, [[CONT]] ], [ [[TMP6]], [[COND_FALSE]] ]
; CHECK-NEXT:    ret i32 [[COND]]
;
entry:
  %0 = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 -1, i32 %y)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:                                             ; preds = %cond.false, %entry
  tail call void @llvm.trap()
  unreachable

cont:                                             ; preds = %entry
  %2 = extractvalue { i32, i1 } %0, 0
  %cmp1 = icmp ult i32 %2, %x
  br i1 %cmp1, label %cond.end, label %cond.false

cond.false:                                       ; preds = %cont
  %3 = tail call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %x, i32 %y)
  %4 = extractvalue { i32, i1 } %3, 0
  %5 = extractvalue { i32, i1 } %3, 1
  br i1 %5, label %trap, label %cond.end

cond.end:                                         ; preds = %cond.false, %cont
  %cond = phi i32 [ 0, %cont ], [ %4, %cond.false ]
  ret i32 %cond
}

define i32 @signed_sub(i32 %x, i32 %y) {
; CHECK-LABEL: @signed_sub(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp slt i32 [[Y:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[LAND_LHS_TRUE:%.*]], label [[LOR_LHS_FALSE:%.*]]
; CHECK:       land.lhs.true:
; CHECK-NEXT:    [[TMP0:%.*]] = add nsw i32 [[Y]], 2147483647
; CHECK-NEXT:    [[TMP1:%.*]] = insertvalue { i32, i1 } undef, i32 [[TMP0]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = insertvalue { i32, i1 } [[TMP1]], i1 false, 1
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue { i32, i1 } [[TMP2]], 1
; CHECK-NEXT:    br i1 [[TMP3]], label [[TRAP:%.*]], label [[CONT:%.*]]
; CHECK:       trap:
; CHECK-NEXT:    tail call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK:       cont:
; CHECK-NEXT:    [[TMP4:%.*]] = extractvalue { i32, i1 } [[TMP2]], 0
; CHECK-NEXT:    [[CMP1:%.*]] = icmp slt i32 [[TMP4]], [[X:%.*]]
; CHECK-NEXT:    br i1 [[CMP1]], label [[COND_END:%.*]], label [[COND_FALSE:%.*]]
; CHECK:       lor.lhs.false:
; CHECK-NEXT:    [[CMP2:%.*]] = icmp eq i32 [[Y]], 0
; CHECK-NEXT:    br i1 [[CMP2]], label [[COND_FALSE]], label [[LAND_LHS_TRUE3:%.*]]
; CHECK:       land.lhs.true3:
; CHECK-NEXT:    [[TMP5:%.*]] = add nsw i32 [[Y]], -2147483648
; CHECK-NEXT:    [[TMP6:%.*]] = insertvalue { i32, i1 } undef, i32 [[TMP5]], 0
; CHECK-NEXT:    [[TMP7:%.*]] = insertvalue { i32, i1 } [[TMP6]], i1 false, 1
; CHECK-NEXT:    [[TMP8:%.*]] = extractvalue { i32, i1 } [[TMP7]], 1
; CHECK-NEXT:    br i1 [[TMP8]], label [[TRAP]], label [[CONT4:%.*]]
; CHECK:       cont4:
; CHECK-NEXT:    [[TMP9:%.*]] = extractvalue { i32, i1 } [[TMP7]], 0
; CHECK-NEXT:    [[CMP5:%.*]] = icmp sgt i32 [[TMP9]], [[X]]
; CHECK-NEXT:    br i1 [[CMP5]], label [[COND_END]], label [[COND_FALSE]]
; CHECK:       cond.false:
; CHECK-NEXT:    [[TMP10:%.*]] = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 [[X]], i32 [[Y]])
; CHECK-NEXT:    [[TMP11:%.*]] = extractvalue { i32, i1 } [[TMP10]], 0
; CHECK-NEXT:    [[TMP12:%.*]] = extractvalue { i32, i1 } [[TMP10]], 1
; CHECK-NEXT:    br i1 [[TMP12]], label [[TRAP]], label [[COND_END]]
; CHECK:       cond.end:
; CHECK-NEXT:    [[COND:%.*]] = phi i32 [ 0, [[CONT4]] ], [ 0, [[CONT]] ], [ [[TMP11]], [[COND_FALSE]] ]
; CHECK-NEXT:    ret i32 [[COND]]
;
entry:
  %cmp = icmp slt i32 %y, 0
  br i1 %cmp, label %land.lhs.true, label %lor.lhs.false

land.lhs.true:                                    ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %y, i32 2147483647)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:                                             ; preds = %land.lhs.true, %land.lhs.true3, %cond.false
  tail call void @llvm.trap()
  unreachable

cont:                                             ; preds = %land.lhs.true
  %2 = extractvalue { i32, i1 } %0, 0
  %cmp1 = icmp slt i32 %2, %x
  br i1 %cmp1, label %cond.end, label %cond.false

lor.lhs.false:                                    ; preds = %entry
  %cmp2 = icmp eq i32 %y, 0
  br i1 %cmp2, label %cond.false, label %land.lhs.true3

land.lhs.true3:                                   ; preds = %lor.lhs.false
  %3 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %y, i32 -2147483648)
  %4 = extractvalue { i32, i1 } %3, 1
  br i1 %4, label %trap, label %cont4

cont4:                                            ; preds = %land.lhs.true3
  %5 = extractvalue { i32, i1 } %3, 0
  %cmp5 = icmp sgt i32 %5, %x
  br i1 %cmp5, label %cond.end, label %cond.false

cond.false:                                       ; preds = %lor.lhs.false, %cont, %cont4
  %6 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %x, i32 %y)
  %7 = extractvalue { i32, i1 } %6, 0
  %8 = extractvalue { i32, i1 } %6, 1
  br i1 %8, label %trap, label %cond.end

cond.end:                                         ; preds = %cond.false, %cont, %cont4
  %cond = phi i32 [ 0, %cont4 ], [ 0, %cont ], [ %7, %cond.false ]
  ret i32 %cond
}

define i32 @unsigned_sub(i32 %x, i32 %y) {
; CHECK-LABEL: @unsigned_sub(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp ult i32 [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[COND_END:%.*]], label [[COND_FALSE:%.*]]
; CHECK:       cond.false:
; CHECK-NEXT:    [[TMP0:%.*]] = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 [[X]], i32 [[Y]])
; CHECK-NEXT:    [[TMP1:%.*]] = extractvalue { i32, i1 } [[TMP0]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = extractvalue { i32, i1 } [[TMP0]], 1
; CHECK-NEXT:    br i1 [[TMP2]], label [[TRAP:%.*]], label [[COND_END]]
; CHECK:       trap:
; CHECK-NEXT:    tail call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK:       cond.end:
; CHECK-NEXT:    [[COND:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[TMP1]], [[COND_FALSE]] ]
; CHECK-NEXT:    ret i32 [[COND]]
;
entry:
  %cmp = icmp ult i32 %x, %y
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %x, i32 %y)
  %1 = extractvalue { i32, i1 } %0, 0
  %2 = extractvalue { i32, i1 } %0, 1
  br i1 %2, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %1, %cond.false ]
  ret i32 %cond
}

define i32 @signed_add_r1(i32 %x) {
; CHECK-LABEL: @signed_add_r1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X:%.*]], 2147483647
; CHECK-NEXT:    br i1 [[CMP]], label [[COND_END:%.*]], label [[COND_FALSE:%.*]]
; CHECK:       cond.false:
; CHECK-NEXT:    [[TMP0:%.*]] = add nsw i32 [[X]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = insertvalue { i32, i1 } undef, i32 [[TMP0]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = insertvalue { i32, i1 } [[TMP1]], i1 false, 1
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue { i32, i1 } [[TMP2]], 0
; CHECK-NEXT:    [[TMP4:%.*]] = extractvalue { i32, i1 } [[TMP2]], 1
; CHECK-NEXT:    br i1 [[TMP4]], label [[TRAP:%.*]], label [[COND_END]]
; CHECK:       trap:
; CHECK-NEXT:    tail call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK:       cond.end:
; CHECK-NEXT:    [[COND:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[TMP3]], [[COND_FALSE]] ]
; CHECK-NEXT:    ret i32 [[COND]]
;
entry:
  %cmp = icmp eq i32 %x, 2147483647
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %x, i32 1)
  %1 = extractvalue { i32, i1 } %0, 0
  %2 = extractvalue { i32, i1 } %0, 1
  br i1 %2, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %1, %cond.false ]
  ret i32 %cond
}

define i32 @unsigned_add_r1(i32 %x) {
; CHECK-LABEL: @unsigned_add_r1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X:%.*]], -1
; CHECK-NEXT:    br i1 [[CMP]], label [[COND_END:%.*]], label [[COND_FALSE:%.*]]
; CHECK:       cond.false:
; CHECK-NEXT:    [[TMP0:%.*]] = add nuw i32 [[X]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = insertvalue { i32, i1 } undef, i32 [[TMP0]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = insertvalue { i32, i1 } [[TMP1]], i1 false, 1
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue { i32, i1 } [[TMP2]], 0
; CHECK-NEXT:    [[TMP4:%.*]] = extractvalue { i32, i1 } [[TMP2]], 1
; CHECK-NEXT:    br i1 [[TMP4]], label [[TRAP:%.*]], label [[COND_END]]
; CHECK:       trap:
; CHECK-NEXT:    tail call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK:       cond.end:
; CHECK-NEXT:    [[COND:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[TMP3]], [[COND_FALSE]] ]
; CHECK-NEXT:    ret i32 [[COND]]
;
entry:
  %cmp = icmp eq i32 %x, -1
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %x, i32 1)
  %1 = extractvalue { i32, i1 } %0, 0
  %2 = extractvalue { i32, i1 } %0, 1
  br i1 %2, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %1, %cond.false ]
  ret i32 %cond
}

define i32 @signed_sub_r1(i32 %x) {
; CHECK-LABEL: @signed_sub_r1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X:%.*]], -2147483648
; CHECK-NEXT:    br i1 [[CMP]], label [[COND_END:%.*]], label [[COND_FALSE:%.*]]
; CHECK:       cond.false:
; CHECK-NEXT:    [[TMP0:%.*]] = sub nsw i32 [[X]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = insertvalue { i32, i1 } undef, i32 [[TMP0]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = insertvalue { i32, i1 } [[TMP1]], i1 false, 1
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue { i32, i1 } [[TMP2]], 0
; CHECK-NEXT:    [[TMP4:%.*]] = extractvalue { i32, i1 } [[TMP2]], 1
; CHECK-NEXT:    br i1 [[TMP4]], label [[TRAP:%.*]], label [[COND_END]]
; CHECK:       trap:
; CHECK-NEXT:    tail call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK:       cond.end:
; CHECK-NEXT:    [[COND:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[TMP3]], [[COND_FALSE]] ]
; CHECK-NEXT:    ret i32 [[COND]]
;
entry:
  %cmp = icmp eq i32 %x, -2147483648
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %x, i32 1)
  %1 = extractvalue { i32, i1 } %0, 0
  %2 = extractvalue { i32, i1 } %0, 1
  br i1 %2, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %1, %cond.false ]
  ret i32 %cond
}

define i32 @unsigned_sub_r1(i32 %x) {
; CHECK-LABEL: @unsigned_sub_r1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[COND_END:%.*]], label [[COND_FALSE:%.*]]
; CHECK:       cond.false:
; CHECK-NEXT:    [[TMP0:%.*]] = sub nuw i32 [[X]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = insertvalue { i32, i1 } undef, i32 [[TMP0]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = insertvalue { i32, i1 } [[TMP1]], i1 false, 1
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue { i32, i1 } [[TMP2]], 0
; CHECK-NEXT:    [[TMP4:%.*]] = extractvalue { i32, i1 } [[TMP2]], 1
; CHECK-NEXT:    br i1 [[TMP4]], label [[TRAP:%.*]], label [[COND_END]]
; CHECK:       trap:
; CHECK-NEXT:    tail call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK:       cond.end:
; CHECK-NEXT:    [[COND:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[TMP3]], [[COND_FALSE]] ]
; CHECK-NEXT:    ret i32 [[COND]]
;
entry:
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %x, i32 1)
  %1 = extractvalue { i32, i1 } %0, 0
  %2 = extractvalue { i32, i1 } %0, 1
  br i1 %2, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %1, %cond.false ]
  ret i32 %cond
}

define i32 @signed_add_rn1(i32 %x) {
; CHECK-LABEL: @signed_add_rn1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X:%.*]], -2147483648
; CHECK-NEXT:    br i1 [[CMP]], label [[COND_END:%.*]], label [[COND_FALSE:%.*]]
; CHECK:       cond.false:
; CHECK-NEXT:    [[TMP0:%.*]] = add nsw i32 [[X]], -1
; CHECK-NEXT:    [[TMP1:%.*]] = insertvalue { i32, i1 } undef, i32 [[TMP0]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = insertvalue { i32, i1 } [[TMP1]], i1 false, 1
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue { i32, i1 } [[TMP2]], 0
; CHECK-NEXT:    [[TMP4:%.*]] = extractvalue { i32, i1 } [[TMP2]], 1
; CHECK-NEXT:    br i1 [[TMP4]], label [[TRAP:%.*]], label [[COND_END]]
; CHECK:       trap:
; CHECK-NEXT:    tail call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK:       cond.end:
; CHECK-NEXT:    [[COND:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[TMP3]], [[COND_FALSE]] ]
; CHECK-NEXT:    ret i32 [[COND]]
;
entry:
  %cmp = icmp eq i32 %x, -2147483648
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %x, i32 -1)
  %1 = extractvalue { i32, i1 } %0, 0
  %2 = extractvalue { i32, i1 } %0, 1
  br i1 %2, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %1, %cond.false ]
  ret i32 %cond
}

define i32 @signed_sub_rn1(i32 %x) {
; CHECK-LABEL: @signed_sub_rn1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X:%.*]], 2147483647
; CHECK-NEXT:    br i1 [[CMP]], label [[COND_END:%.*]], label [[COND_FALSE:%.*]]
; CHECK:       cond.false:
; CHECK-NEXT:    [[TMP0:%.*]] = sub nsw i32 [[X]], -1
; CHECK-NEXT:    [[TMP1:%.*]] = insertvalue { i32, i1 } undef, i32 [[TMP0]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = insertvalue { i32, i1 } [[TMP1]], i1 false, 1
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue { i32, i1 } [[TMP2]], 0
; CHECK-NEXT:    [[TMP4:%.*]] = extractvalue { i32, i1 } [[TMP2]], 1
; CHECK-NEXT:    br i1 [[TMP4]], label [[TRAP:%.*]], label [[COND_END]]
; CHECK:       trap:
; CHECK-NEXT:    tail call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK:       cond.end:
; CHECK-NEXT:    [[COND:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[TMP3]], [[COND_FALSE]] ]
; CHECK-NEXT:    ret i32 [[COND]]
;
entry:
  %cmp = icmp eq i32 %x, 2147483647
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %x, i32 -1)
  %1 = extractvalue { i32, i1 } %0, 0
  %2 = extractvalue { i32, i1 } %0, 1
  br i1 %2, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %1, %cond.false ]
  ret i32 %cond
}

define i32 @unsigned_mul(i32 %x) {
; CHECK-LABEL: @unsigned_mul(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp ugt i32 [[X:%.*]], 10000
; CHECK-NEXT:    br i1 [[CMP]], label [[COND_END:%.*]], label [[COND_FALSE:%.*]]
; CHECK:       cond.false:
; CHECK-NEXT:    [[MULO1:%.*]] = mul nuw i32 [[X]], 100
; CHECK-NEXT:    [[TMP0:%.*]] = insertvalue { i32, i1 } undef, i32 [[MULO1]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = insertvalue { i32, i1 } [[TMP0]], i1 false, 1
; CHECK-NEXT:    [[RES:%.*]] = extractvalue { i32, i1 } [[TMP1]], 0
; CHECK-NEXT:    [[OV:%.*]] = extractvalue { i32, i1 } [[TMP1]], 1
; CHECK-NEXT:    br i1 [[OV]], label [[TRAP:%.*]], label [[COND_END]]
; CHECK:       trap:
; CHECK-NEXT:    tail call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK:       cond.end:
; CHECK-NEXT:    [[COND:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[RES]], [[COND_FALSE]] ]
; CHECK-NEXT:    ret i32 [[COND]]
;
entry:
  %cmp = icmp ugt i32 %x, 10000
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %mulo = tail call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %x, i32 100)
  %res = extractvalue { i32, i1 } %mulo, 0
  %ov = extractvalue { i32, i1 } %mulo, 1
  br i1 %ov, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %res, %cond.false ]
  ret i32 %cond
}

define i32 @signed_mul(i32 %x) {
; CHECK-LABEL: @signed_mul(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP1:%.*]] = icmp sgt i32 [[X:%.*]], 10000
; CHECK-NEXT:    [[CMP2:%.*]] = icmp slt i32 [[X]], -10000
; CHECK-NEXT:    [[CMP3:%.*]] = or i1 [[CMP1]], [[CMP2]]
; CHECK-NEXT:    br i1 [[CMP3]], label [[COND_END:%.*]], label [[COND_FALSE:%.*]]
; CHECK:       cond.false:
; CHECK-NEXT:    [[MULO1:%.*]] = mul nsw i32 [[X]], 100
; CHECK-NEXT:    [[TMP0:%.*]] = insertvalue { i32, i1 } undef, i32 [[MULO1]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = insertvalue { i32, i1 } [[TMP0]], i1 false, 1
; CHECK-NEXT:    [[RES:%.*]] = extractvalue { i32, i1 } [[TMP1]], 0
; CHECK-NEXT:    [[OV:%.*]] = extractvalue { i32, i1 } [[TMP1]], 1
; CHECK-NEXT:    br i1 [[OV]], label [[TRAP:%.*]], label [[COND_END]]
; CHECK:       trap:
; CHECK-NEXT:    tail call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK:       cond.end:
; CHECK-NEXT:    [[COND:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[RES]], [[COND_FALSE]] ]
; CHECK-NEXT:    ret i32 [[COND]]
;
entry:
  %cmp1 = icmp sgt i32 %x, 10000
  %cmp2 = icmp slt i32 %x, -10000
  %cmp3 = or i1 %cmp1, %cmp2
  br i1 %cmp3, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %mulo = tail call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %x, i32 100)
  %res = extractvalue { i32, i1 } %mulo, 0
  %ov = extractvalue { i32, i1 } %mulo, 1
  br i1 %ov, label %trap, label %cond.end

trap:                                             ; preds = %cond.false
  tail call void @llvm.trap()
  unreachable

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i32 [ 0, %entry ], [ %res, %cond.false ]
  ret i32 %cond
}

declare i32 @bar(i32)

define void @unsigned_loop(i32 %i) {
; CHECK-LABEL: @unsigned_loop(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP3:%.*]] = icmp eq i32 [[I:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP3]], label [[WHILE_END:%.*]], label [[WHILE_BODY_PREHEADER:%.*]]
; CHECK:       while.body.preheader:
; CHECK-NEXT:    br label [[WHILE_BODY:%.*]]
; CHECK:       while.body:
; CHECK-NEXT:    [[I_ADDR_04:%.*]] = phi i32 [ [[TMP4:%.*]], [[CONT:%.*]] ], [ [[I]], [[WHILE_BODY_PREHEADER]] ]
; CHECK-NEXT:    [[CALL:%.*]] = tail call i32 @bar(i32 [[I_ADDR_04]])
; CHECK-NEXT:    [[TMP0:%.*]] = sub nuw i32 [[I_ADDR_04]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = insertvalue { i32, i1 } undef, i32 [[TMP0]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = insertvalue { i32, i1 } [[TMP1]], i1 false, 1
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue { i32, i1 } [[TMP2]], 1
; CHECK-NEXT:    br i1 [[TMP3]], label [[TRAP:%.*]], label [[CONT]]
; CHECK:       trap:
; CHECK-NEXT:    tail call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK:       cont:
; CHECK-NEXT:    [[TMP4]] = extractvalue { i32, i1 } [[TMP2]], 0
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP4]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[WHILE_END]], label [[WHILE_BODY]]
; CHECK:       while.end:
; CHECK-NEXT:    ret void
;
entry:
  %cmp3 = icmp eq i32 %i, 0
  br i1 %cmp3, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %cont
  %i.addr.04 = phi i32 [ %2, %cont ], [ %i, %while.body.preheader ]
  %call = tail call i32 @bar(i32 %i.addr.04)
  %0 = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %i.addr.04, i32 1)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:                                             ; preds = %while.body
  tail call void @llvm.trap()
  unreachable

cont:                                             ; preds = %while.body
  %2 = extractvalue { i32, i1 } %0, 0
  %cmp = icmp eq i32 %2, 0
  br i1 %cmp, label %while.end, label %while.body

while.end:                                        ; preds = %cont, %entry
  ret void
}

define void @intrinsic_into_phi(i32 %n) {
; CHECK-LABEL: @intrinsic_into_phi(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[CONT:%.*]]
; CHECK:       for.cond:
; CHECK-NEXT:    [[TMP0:%.*]] = add nsw i32 [[DOTLCSSA:%.*]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = insertvalue { i32, i1 } undef, i32 [[TMP0]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = insertvalue { i32, i1 } [[TMP1]], i1 false, 1
; CHECK-NEXT:    [[TMP3:%.*]] = extractvalue { i32, i1 } [[TMP2]], 1
; CHECK-NEXT:    br i1 [[TMP3]], label [[TRAP:%.*]], label [[CONT]]
; CHECK:       trap:
; CHECK-NEXT:    tail call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK:       cont:
; CHECK-NEXT:    [[TMP4:%.*]] = phi { i32, i1 } [ zeroinitializer, [[ENTRY:%.*]] ], [ [[TMP2]], [[FOR_COND:%.*]] ]
; CHECK-NEXT:    [[TMP5:%.*]] = extractvalue { i32, i1 } [[TMP4]], 0
; CHECK-NEXT:    [[CALL9:%.*]] = tail call i32 @bar(i32 [[TMP5]])
; CHECK-NEXT:    [[TOBOOL10:%.*]] = icmp eq i32 [[CALL9]], 0
; CHECK-NEXT:    br i1 [[TOBOOL10]], label [[WHILE_END:%.*]], label [[WHILE_BODY_PREHEADER:%.*]]
; CHECK:       while.body.preheader:
; CHECK-NEXT:    br label [[WHILE_BODY:%.*]]
; CHECK:       while.cond:
; CHECK-NEXT:    [[TMP6:%.*]] = extractvalue { i32, i1 } [[TMP8:%.*]], 0
; CHECK-NEXT:    [[CALL:%.*]] = tail call i32 @bar(i32 [[TMP6]])
; CHECK-NEXT:    [[TOBOOL:%.*]] = icmp eq i32 [[CALL]], 0
; CHECK-NEXT:    br i1 [[TOBOOL]], label [[WHILE_END]], label [[WHILE_BODY]]
; CHECK:       while.body:
; CHECK-NEXT:    [[TMP7:%.*]] = phi i32 [ [[TMP6]], [[WHILE_COND:%.*]] ], [ [[TMP5]], [[WHILE_BODY_PREHEADER]] ]
; CHECK-NEXT:    [[TMP8]] = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 [[TMP7]], i32 1)
; CHECK-NEXT:    [[TMP9:%.*]] = extractvalue { i32, i1 } [[TMP8]], 1
; CHECK-NEXT:    br i1 [[TMP9]], label [[TRAP]], label [[WHILE_COND]]
; CHECK:       while.end:
; CHECK-NEXT:    [[DOTLCSSA]] = phi i32 [ [[TMP5]], [[CONT]] ], [ [[TMP6]], [[WHILE_COND]] ]
; CHECK-NEXT:    [[CMP:%.*]] = icmp slt i32 [[DOTLCSSA]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[CMP]], label [[FOR_COND]], label [[CLEANUP2:%.*]]
; CHECK:       cleanup2:
; CHECK-NEXT:    ret void
;
entry:
  br label %cont

for.cond:                                         ; preds = %while.end
  %0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %.lcssa, i32 1)
  %1 = extractvalue { i32, i1 } %0, 1
  br i1 %1, label %trap, label %cont

trap:                                             ; preds = %for.cond, %while.body
  tail call void @llvm.trap()
  unreachable

cont:                                             ; preds = %entry, %for.cond
  %2 = phi { i32, i1 } [ zeroinitializer, %entry ], [ %0, %for.cond ]
  %3 = extractvalue { i32, i1 } %2, 0
  %call9 = tail call i32 @bar(i32 %3)
  %tobool10 = icmp eq i32 %call9, 0
  br i1 %tobool10, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %cont
  br label %while.body

while.cond:                                       ; preds = %while.body
  %4 = extractvalue { i32, i1 } %6, 0
  %call = tail call i32 @bar(i32 %4)
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %while.end, label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.cond
  %5 = phi i32 [ %4, %while.cond ], [ %3, %while.body.preheader ]
  %6 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %5, i32 1)
  %7 = extractvalue { i32, i1 } %6, 1
  br i1 %7, label %trap, label %while.cond

while.end:                                        ; preds = %while.cond, %cont
  %.lcssa = phi i32 [ %3, %cont ], [ %4, %while.cond ]
  %cmp = icmp slt i32 %.lcssa, %n
  br i1 %cmp, label %for.cond, label %cleanup2

cleanup2:                                         ; preds = %while.end
  ret void
}
