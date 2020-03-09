; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --function-signature --scrub-attributes
; RUN: opt -attributor -attributor-manifest-internal --attributor-disable=false -attributor-max-iterations-verify -attributor-annotate-decl-cs -attributor-max-iterations=16 -S < %s | FileCheck %s --check-prefix=ATTRIBUTOR
; FIXME: Figure out why we need 16 iterations here.

; UTC_ARGS: --disable

declare void @deref_phi_user(i32* %a);

; TEST 1
; take mininimum of return values
;
define i32* @test1(i32* dereferenceable(4) %0, double* dereferenceable(8) %1, i1 zeroext %2) local_unnamed_addr {
; ATTRIBUTOR: define nonnull dereferenceable(4) i32* @test1(i32* nofree nonnull readnone dereferenceable(4) "no-capture-maybe-returned" %0, double* nofree nonnull readnone dereferenceable(8) "no-capture-maybe-returned" %1, i1 zeroext %2)
  %4 = bitcast double* %1 to i32*
  %5 = select i1 %2, i32* %0, i32* %4
  ret i32* %5
}

; TEST 2
define i32* @test2(i32* dereferenceable_or_null(4) %0, double* dereferenceable(8) %1, i1 zeroext %2) local_unnamed_addr {
; ATTRIBUTOR: define dereferenceable_or_null(4) i32* @test2(i32* nofree readnone dereferenceable_or_null(4) "no-capture-maybe-returned" %0, double* nofree nonnull readnone dereferenceable(8) "no-capture-maybe-returned" %1, i1 zeroext %2)
  %4 = bitcast double* %1 to i32*
  %5 = select i1 %2, i32* %0, i32* %4
  ret i32* %5
}

; TEST 3
; GEP inbounds
define i32* @test3_1(i32* dereferenceable(8) %0) local_unnamed_addr {
; ATTRIBUTOR: define nonnull dereferenceable(4) i32* @test3_1(i32* nofree nonnull readnone dereferenceable(8) "no-capture-maybe-returned" %0)
  %ret = getelementptr inbounds i32, i32* %0, i64 1
  ret i32* %ret
}

define i32* @test3_2(i32* dereferenceable_or_null(32) %0) local_unnamed_addr {
; ATTRIBUTOR: define nonnull dereferenceable(16) i32* @test3_2(i32* nofree readnone dereferenceable_or_null(32) "no-capture-maybe-returned" %0)
  %ret = getelementptr inbounds i32, i32* %0, i64 4
  ret i32* %ret
}

define i32* @test3_3(i32* dereferenceable(8) %0, i32* dereferenceable(16) %1, i1 %2) local_unnamed_addr {
; ATTRIBUTOR: define nonnull dereferenceable(4) i32* @test3_3(i32* nofree nonnull readnone dereferenceable(8) "no-capture-maybe-returned" %0, i32* nofree nonnull readnone dereferenceable(16) "no-capture-maybe-returned" %1, i1 %2) local_unnamed_addr
  %ret1 = getelementptr inbounds i32, i32* %0, i64 1
  %ret2 = getelementptr inbounds i32, i32* %1, i64 2
  %ret = select i1 %2, i32* %ret1, i32* %ret2
  ret i32* %ret
}

; TEST 4
; Better than known in IR.

define dereferenceable(4) i32* @test4(i32* dereferenceable(8) %0) local_unnamed_addr {
; ATTRIBUTOR: define nonnull dereferenceable(8) i32* @test4(i32* nofree nonnull readnone returned dereferenceable(8) "no-capture-maybe-returned" %0)
  ret i32* %0
}

; TEST 5
; loop in which dereferenceabily "grows"
define void @deref_phi_growing(i32* dereferenceable(4000) %a) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %a.addr.0 = phi i32* [ %a, %entry ], [ %incdec.ptr, %for.inc ]
; ATTRIBUTOR: call void @deref_phi_user(i32* nonnull dereferenceable(4000) %a.addr.0)
  call void @deref_phi_user(i32* %a.addr.0)
  %tmp = load i32, i32* %a.addr.0, align 4
  %cmp = icmp slt i32 %i.0, %tmp
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  br label %for.end

for.body:                                         ; preds = %for.cond
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %incdec.ptr = getelementptr inbounds i32, i32* %a.addr.0, i64 -1
  %inc = add nuw nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond.cleanup
  ret void
}

; TEST 6
; loop in which dereferenceabily "shrinks"
define void @deref_phi_shrinking(i32* dereferenceable(4000) %a) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %a.addr.0 = phi i32* [ %a, %entry ], [ %incdec.ptr, %for.inc ]
; ATTRIBUTOR: call void @deref_phi_user(i32* nonnull %a.addr.0)
  call void @deref_phi_user(i32* %a.addr.0)
  %tmp = load i32, i32* %a.addr.0, align 4
  %cmp = icmp slt i32 %i.0, %tmp
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  br label %for.end

for.body:                                         ; preds = %for.cond
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %incdec.ptr = getelementptr inbounds i32, i32* %a.addr.0, i64 1
  %inc = add nuw nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond.cleanup
  ret void
}

; TEST 7
; share known infomation in must-be-executed-context
declare i32* @unkown_ptr() willreturn nounwind
declare i32 @unkown_f(i32*) willreturn nounwind
define i32* @f7_0(i32* %ptr) {
; ATTRIBUTOR: define nonnull dereferenceable(8) i32* @f7_0(i32* nonnull returned dereferenceable(8) %ptr)
  %T = tail call i32 @unkown_f(i32* dereferenceable(8) %ptr)
  ret i32* %ptr
}

; ATTRIBUTOR: define void @f7_1(i32* nonnull dereferenceable(4) %ptr, i1 %c)
define void @f7_1(i32* %ptr, i1 %c) {

; ATTRIBUTOR:   %A = tail call i32 @unkown_f(i32* nonnull dereferenceable(4) %ptr)
  %A = tail call i32 @unkown_f(i32* %ptr)

  %ptr.0 = load i32, i32* %ptr
  ; deref 4 hold

; FIXME: this should be %B = tail call i32 @unkown_f(i32* nonnull dereferenceable(4) %ptr)
; ATTRIBUTOR:   %B = tail call i32 @unkown_f(i32* nonnull dereferenceable(4) %ptr)
  %B = tail call i32 @unkown_f(i32* dereferenceable(1) %ptr)

  br i1%c, label %if.true, label %if.false
if.true:
; ATTRIBUTOR:   %C = tail call i32 @unkown_f(i32* nonnull dereferenceable(8) %ptr)
  %C = tail call i32 @unkown_f(i32* %ptr)

; ATTRIBUTOR:   %D = tail call i32 @unkown_f(i32* nonnull dereferenceable(8) %ptr)
  %D = tail call i32 @unkown_f(i32* dereferenceable(8) %ptr)

; ATTRIBUTOR:   %E = tail call i32 @unkown_f(i32* nonnull dereferenceable(8) %ptr)
  %E = tail call i32 @unkown_f(i32* %ptr)

  ret void

if.false:
  ret void
}

; ATTRIBUTOR: define void @f7_2(i1 %c)
define void @f7_2(i1 %c) {

  %ptr =  tail call i32* @unkown_ptr()

; ATTRIBUTOR:   %A = tail call i32 @unkown_f(i32* nonnull dereferenceable(4) %ptr)
  %A = tail call i32 @unkown_f(i32* %ptr)

  %arg_a.0 = load i32, i32* %ptr
  ; deref 4 hold

; ATTRIBUTOR:   %B = tail call i32 @unkown_f(i32* nonnull dereferenceable(4) %ptr)
  %B = tail call i32 @unkown_f(i32* dereferenceable(1) %ptr)

  br i1%c, label %if.true, label %if.false
if.true:

; ATTRIBUTOR:   %C = tail call i32 @unkown_f(i32* nonnull dereferenceable(8) %ptr)
  %C = tail call i32 @unkown_f(i32* %ptr)

; ATTRIBUTOR:   %D = tail call i32 @unkown_f(i32* nonnull dereferenceable(8) %ptr)
  %D = tail call i32 @unkown_f(i32* dereferenceable(8) %ptr)

  %E = tail call i32 @unkown_f(i32* %ptr)
; ATTRIBUTOR:   %E = tail call i32 @unkown_f(i32* nonnull dereferenceable(8) %ptr)

  ret void

if.false:
  ret void
}

define i32* @f7_3() {
; ATTRIBUTOR: define nonnull align 16 dereferenceable(4) i32* @f7_3()
  %ptr = tail call i32* @unkown_ptr()
  store i32 10, i32* %ptr, align 16
  ret i32* %ptr
}

define i32* @test_for_minus_index(i32* %p) {
; FIXME: This should have a return dereferenceable(8) but we need to make sure it will work in loops as well.
; ATTRIBUTOR: define nonnull i32* @test_for_minus_index(i32* nofree nonnull writeonly "no-capture-maybe-returned" %p)
  %q = getelementptr inbounds i32, i32* %p, i32 -2
  store i32 1, i32* %q
  ret i32* %q
}

define void @deref_or_null_and_nonnull(i32* dereferenceable_or_null(100) %0) {
; ATTRIBUTOR: define void @deref_or_null_and_nonnull(i32* nocapture nofree nonnull writeonly dereferenceable(100) %0)
  store i32 1, i32* %0
  ret void
}

; UTC_ARGS: --enable

; TEST 8
; Use Constant range in deereferenceable
; void g(int *p, long long int *range){
;   int r = *range ; // [10, 99]
;   fill_range(p, *range);
; }

; void fill_range(int* p, long long int start){
;   for(long long int i = start;i<start+10;i++){
;     // If p[i] is inbounds, p is dereferenceable(40) at least.
;     p[i] = i;
;   }
; }

define internal void @fill_range_not_inbounds(i32* %p, i64 %start){
; ATTRIBUTOR-LABEL: define {{[^@]+}}@fill_range_not_inbounds
; ATTRIBUTOR-SAME: (i32* nocapture nofree writeonly [[P:%.*]], i64 [[START:%.*]])
; ATTRIBUTOR-NEXT:  entry:
; ATTRIBUTOR-NEXT:    [[TMP0:%.*]] = add nsw i64 [[START]], 9
; ATTRIBUTOR-NEXT:    br label [[FOR_BODY:%.*]]
; ATTRIBUTOR:       for.cond.cleanup:
; ATTRIBUTOR-NEXT:    ret void
; ATTRIBUTOR:       for.body:
; ATTRIBUTOR-NEXT:    [[I_06:%.*]] = phi i64 [ [[START]], [[ENTRY:%.*]] ], [ [[INC:%.*]], [[FOR_BODY]] ]
; ATTRIBUTOR-NEXT:    [[CONV:%.*]] = trunc i64 [[I_06]] to i32
; ATTRIBUTOR-NEXT:    [[ARRAYIDX:%.*]] = getelementptr i32, i32* [[P]], i64 [[I_06]]
; ATTRIBUTOR-NEXT:    store i32 [[CONV]], i32* [[ARRAYIDX]], align 4
; ATTRIBUTOR-NEXT:    [[INC]] = add nsw i64 [[I_06]], 1
; ATTRIBUTOR-NEXT:    [[CMP:%.*]] = icmp slt i64 [[I_06]], [[TMP0]]
; ATTRIBUTOR-NEXT:    br i1 [[CMP]], label [[FOR_BODY]], label [[FOR_COND_CLEANUP:%.*]]
;
; NOTE: %p should not be dereferenceable
entry:
  %0 = add nsw i64 %start, 9
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %i.06 = phi i64 [ %start, %entry ], [ %inc, %for.body ]
  %conv = trunc i64 %i.06 to i32
  %arrayidx = getelementptr i32, i32* %p, i64 %i.06
  store i32 %conv, i32* %arrayidx, align 4
  %inc = add nsw i64 %i.06, 1
  %cmp = icmp slt i64 %i.06, %0
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}
define internal void @fill_range_inbounds(i32* %p, i64 %start){
; ATTRIBUTOR-LABEL: define {{[^@]+}}@fill_range_inbounds
; ATTRIBUTOR-SAME: (i32* nocapture nofree writeonly [[P:%.*]], i64 [[START:%.*]])
; ATTRIBUTOR-NEXT:  entry:
; ATTRIBUTOR-NEXT:    [[TMP0:%.*]] = add nsw i64 [[START]], 9
; ATTRIBUTOR-NEXT:    br label [[FOR_BODY:%.*]]
; ATTRIBUTOR:       for.cond.cleanup:
; ATTRIBUTOR-NEXT:    ret void
; ATTRIBUTOR:       for.body:
; ATTRIBUTOR-NEXT:    [[I_06:%.*]] = phi i64 [ [[START]], [[ENTRY:%.*]] ], [ [[INC:%.*]], [[FOR_BODY]] ]
; ATTRIBUTOR-NEXT:    [[CONV:%.*]] = trunc i64 [[I_06]] to i32
; ATTRIBUTOR-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, i32* [[P]], i64 [[I_06]]
; ATTRIBUTOR-NEXT:    store i32 [[CONV]], i32* [[ARRAYIDX]], align 4
; ATTRIBUTOR-NEXT:    [[INC]] = add nsw i64 [[I_06]], 1
; ATTRIBUTOR-NEXT:    [[CMP:%.*]] = icmp slt i64 [[I_06]], [[TMP0]]
; ATTRIBUTOR-NEXT:    br i1 [[CMP]], label [[FOR_BODY]], label [[FOR_COND_CLEANUP:%.*]]
;
; FIXME: %p should be dereferenceable(40)
entry:
  %0 = add nsw i64 %start, 9
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %i.06 = phi i64 [ %start, %entry ], [ %inc, %for.body ]
  %conv = trunc i64 %i.06 to i32
  %arrayidx = getelementptr inbounds i32, i32* %p, i64 %i.06
  store i32 %conv, i32* %arrayidx, align 4
  %inc = add nsw i64 %i.06, 1
  %cmp = icmp slt i64 %i.06, %0
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

define void @call_fill_range(i32* nocapture %p, i64* nocapture readonly %range) {
; ATTRIBUTOR-LABEL: define {{[^@]+}}@call_fill_range
; ATTRIBUTOR-SAME: (i32* nocapture nofree writeonly [[P:%.*]], i64* nocapture nofree nonnull readonly align 8 dereferenceable(8) [[RANGE:%.*]])
; ATTRIBUTOR-NEXT:  entry:
; ATTRIBUTOR-NEXT:    [[TMP0:%.*]] = load i64, i64* [[RANGE]], align 8, !range !0
; ATTRIBUTOR-NEXT:    tail call void @fill_range_inbounds(i32* nocapture nofree writeonly [[P]], i64 [[TMP0]])
; ATTRIBUTOR-NEXT:    tail call void @fill_range_not_inbounds(i32* nocapture nofree writeonly [[P]], i64 [[TMP0]])
; ATTRIBUTOR-NEXT:    ret void
;
entry:
  %0 = load i64, i64* %range, align 8, !range !0
  tail call void @fill_range_inbounds(i32* %p, i64 %0)
  tail call void @fill_range_not_inbounds(i32* %p, i64 %0)
  ret void
}

declare void @use0() willreturn nounwind
declare void @use1(i8*) willreturn nounwind
declare void @use2(i8*, i8*) willreturn nounwind
declare void @use3(i8*, i8*, i8*) willreturn nounwind
; simple path test
; if(..)
;   fun2(dereferenceable(8) %a, dereferenceable(8) %b)
; else
;   fun2(dereferenceable(4) %a, %b)
; We can say that %a is dereferenceable(4) but %b is not.
define void @simple-path(i8* %a, i8 * %b, i8 %c) {
; ATTRIBUTOR: define void @simple-path(i8* nonnull dereferenceable(4) %a, i8* %b, i8 %c)
  %cmp = icmp eq i8 %c, 0
  br i1 %cmp, label %if.then, label %if.else
if.then:
  tail call void @use2(i8* dereferenceable(8) %a, i8* dereferenceable(8) %b)
  ret void
if.else:
  tail call void @use2(i8* dereferenceable(4) %a, i8* %b)
  ret void
}
; More complex test
; {
; fun1(dereferenceable(4) %a)
; if(..)
;    ... (willreturn & nounwind)
;    fun1(dereferenceable(12) %a)
; else
;    ... (willreturn & nounwind)
;    fun1(dereferenceable(16) %a)
; fun1(dereferenceable(8) %a)
; }
; %a is dereferenceable(12)

define void @complex-path(i8* %a, i8* %b, i8 %c) {
; ATTRIBUTOR: define void @complex-path(i8* nonnull dereferenceable(12) %a, i8* nocapture nofree readnone %b, i8 %c)
  %cmp = icmp eq i8 %c, 0
  tail call void @use1(i8* dereferenceable(4) %a)
  br i1 %cmp, label %cont.then, label %cont.else
cont.then:
  tail call void @use1(i8* dereferenceable(12) %a)
  br label %cont2
cont.else:
  tail call void @use1(i8* dereferenceable(16) %a)
  br label %cont2
cont2:
  tail call void @use1(i8* dereferenceable(8) %a)
  ret void
}

;  void rec-branch-1(int a, int b, int c, int *ptr) {
;    if (a) {
;      if (b)
;        *ptr = 1;
;      else
;        *ptr = 2;
;    } else {
;      if (c)
;        *ptr = 3;
;      else
;        *ptr = 4;
;    }
;  }
;
; FIXME: %ptr should be dereferenceable(4)
; ATTRIBUTOR: define dso_local void @rec-branch-1(i32 %a, i32 %b, i32 %c, i32* nocapture nofree writeonly %ptr)
define dso_local void @rec-branch-1(i32 %a, i32 %b, i32 %c, i32* %ptr) {
entry:
  %tobool = icmp eq i32 %a, 0
  br i1 %tobool, label %if.else3, label %if.then

if.then:                                          ; preds = %entry
  %tobool1 = icmp eq i32 %b, 0
  br i1 %tobool1, label %if.else, label %if.then2

if.then2:                                         ; preds = %if.then
  store i32 1, i32* %ptr, align 4
  br label %if.end8

if.else:                                          ; preds = %if.then
  store i32 2, i32* %ptr, align 4
  br label %if.end8

if.else3:                                         ; preds = %entry
  %tobool4 = icmp eq i32 %c, 0
  br i1 %tobool4, label %if.else6, label %if.then5

if.then5:                                         ; preds = %if.else3
  store i32 3, i32* %ptr, align 4
  br label %if.end8

if.else6:                                         ; preds = %if.else3
  store i32 4, i32* %ptr, align 4
  br label %if.end8

if.end8:                                          ; preds = %if.then5, %if.else6, %if.then2, %if.else
  ret void
}

;  void rec-branch-2(int a, int b, int c, int *ptr) {
;    if (a) {
;      if (b)
;        *ptr = 1;
;      else
;        *ptr = 2;
;    } else {
;      if (c)
;        *ptr = 3;
;      else
;        rec-branch-2(1, 1, 1, ptr);
;    }
;  }
; FIXME: %ptr should be dereferenceable(4)
; ATTRIBUTOR: define dso_local void @rec-branch-2(i32 %a, i32 %b, i32 %c, i32* nocapture nofree writeonly %ptr)
define dso_local void @rec-branch-2(i32 %a, i32 %b, i32 %c, i32* %ptr) {
entry:
  %tobool = icmp eq i32 %a, 0
  br i1 %tobool, label %if.else3, label %if.then

if.then:                                          ; preds = %entry
  %tobool1 = icmp eq i32 %b, 0
  br i1 %tobool1, label %if.else, label %if.then2

if.then2:                                         ; preds = %if.then
  store i32 1, i32* %ptr, align 4
  br label %if.end8

if.else:                                          ; preds = %if.then
  store i32 2, i32* %ptr, align 4
  br label %if.end8

if.else3:                                         ; preds = %entry
  %tobool4 = icmp eq i32 %c, 0
  br i1 %tobool4, label %if.else6, label %if.then5

if.then5:                                         ; preds = %if.else3
  store i32 3, i32* %ptr, align 4
  br label %if.end8

if.else6:                                         ; preds = %if.else3
  tail call void @rec-branch-2(i32 1, i32 1, i32 1, i32* %ptr)
  br label %if.end8

if.end8:                                          ; preds = %if.then5, %if.else6, %if.then2, %if.else
  ret void
}

!0 = !{i64 10, i64 100}

