; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S -instsimplify < %s | FileCheck %s

define <4 x i32> @test1(<4 x i32> %A) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    ret <4 x i32> poison
;
  %I = insertelement <4 x i32> %A, i32 5, i64 4294967296
  ret <4 x i32> %I
}

define <4 x i32> @test2(<4 x i32> %A) {
; CHECK-LABEL: @test2(
; CHECK-NEXT:    ret <4 x i32> poison
;
  %I = insertelement <4 x i32> %A, i32 5, i64 4
  ret <4 x i32> %I
}

define <4 x i32> @test3(<4 x i32> %A) {
; CHECK-LABEL: @test3(
; CHECK-NEXT:    [[I:%.*]] = insertelement <4 x i32> [[A:%.*]], i32 5, i64 1
; CHECK-NEXT:    ret <4 x i32> [[I]]
;
  %I = insertelement <4 x i32> %A, i32 5, i64 1
  ret <4 x i32> %I
}

define <4 x i32> @test4(<4 x i32> %A) {
; CHECK-LABEL: @test4(
; CHECK-NEXT:    ret <4 x i32> poison
;
  %I = insertelement <4 x i32> %A, i32 5, i128 100
  ret <4 x i32> %I
}

define <4 x i32> @test5(<4 x i32> %A) {
; CHECK-LABEL: @test5(
; CHECK-NEXT:    ret <4 x i32> poison
;
  %I = insertelement <4 x i32> %A, i32 5, i64 undef
  ret <4 x i32> %I
}

define <4 x i32> @test5_poison(<4 x i32> %A) {
; CHECK-LABEL: @test5_poison(
; CHECK-NEXT:    ret <4 x i32> poison
;
  %I = insertelement <4 x i32> %A, i32 5, i64 poison
  ret <4 x i32> %I
}

define <4 x i32> @elem_poison(<4 x i32> %A) {
; CHECK-LABEL: @elem_poison(
; CHECK-NEXT:    ret <4 x i32> [[A:%.*]]
;
  %B = insertelement <4 x i32> %A, i32 poison, i32 1
  ret <4 x i32> %B
}

; The undef may be replacing a poison value, so it is not safe to just return 'A'.

define <4 x i32> @PR1286(<4 x i32> %A) {
; CHECK-LABEL: @PR1286(
; CHECK-NEXT:    [[B:%.*]] = insertelement <4 x i32> [[A:%.*]], i32 undef, i32 1
; CHECK-NEXT:    ret <4 x i32> [[B]]
;
  %B = insertelement <4 x i32> %A, i32 undef, i32 1
  ret <4 x i32> %B
}

; Constant is not poison, so this can simplify.

define <2 x i32> @undef_into_constant_vector_with_variable_index(<2 x i32> %A, i32 %Index) {
; CHECK-LABEL: @undef_into_constant_vector_with_variable_index(
; CHECK-NEXT:    ret <2 x i32> <i32 42, i32 -42>
;
  %B = insertelement <2 x i32> <i32 42, i32 -42>, i32 undef, i32 %Index
  ret <2 x i32> %B
}

define <8 x i8> @extract_insert_same_vec_and_index(<8 x i8> %in) {
; CHECK-LABEL: @extract_insert_same_vec_and_index(
; CHECK-NEXT:    ret <8 x i8> [[IN:%.*]]
;
  %val = extractelement <8 x i8> %in, i32 5
  %vec = insertelement <8 x i8> %in, i8 %val, i32 5
  ret <8 x i8> %vec
}

define <8 x i8> @extract_insert_same_vec_and_index2(<8 x i8> %in, i32 %index) {
; CHECK-LABEL: @extract_insert_same_vec_and_index2(
; CHECK-NEXT:    ret <8 x i8> [[IN:%.*]]
;
  %val = extractelement <8 x i8> %in, i32 %index
  %vec = insertelement <8 x i8> %in, i8 %val, i32 %index
  ret <8 x i8> %vec
}

; The insert is in an unreachable block, so it is allowed to point to itself.
; This would crash via stack overflow.

define void @PR43218() {
; CHECK-LABEL: @PR43218(
; CHECK-NEXT:  end:
; CHECK-NEXT:    ret void
; CHECK:       unreachable_infloop:
; CHECK-NEXT:    [[EXTRACT:%.*]] = extractelement <2 x i64> [[BOGUS:%.*]], i32 0
; CHECK-NEXT:    [[T0:%.*]] = inttoptr i64 [[EXTRACT]] to i16****
; CHECK-NEXT:    [[BOGUS]] = insertelement <2 x i64> [[BOGUS]], i64 undef, i32 1
; CHECK-NEXT:    br label [[UNREACHABLE_INFLOOP:%.*]]
;
end:
  ret void

unreachable_infloop:
  %extract = extractelement <2 x i64> %bogus, i32 0
  %t0 = inttoptr i64 %extract to i16****
  %bogus = insertelement <2 x i64> %bogus, i64 undef, i32 1
  br label %unreachable_infloop
}
