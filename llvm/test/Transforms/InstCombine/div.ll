; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; This test makes sure that div instructions are properly eliminated.

; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @test1(i32 %A) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    ret i32 [[A:%.*]]
;
  %B = sdiv i32 %A, 1
  ret i32 %B
}

define i32 @test2(i32 %A) {
; CHECK-LABEL: @test2(
; CHECK-NEXT:    [[B:%.*]] = lshr i32 [[A:%.*]], 3
; CHECK-NEXT:    ret i32 [[B]]
;
  %B = udiv i32 %A, 8
  ret i32 %B
}

define i32 @sdiv_by_minus1(i32 %A) {
; CHECK-LABEL: @sdiv_by_minus1(
; CHECK-NEXT:    [[B:%.*]] = sub i32 0, [[A:%.*]]
; CHECK-NEXT:    ret i32 [[B]]
;
  %B = sdiv i32 %A, -1
  ret i32 %B
}

define <2 x i64> @sdiv_by_minus1_vec(<2 x i64> %x) {
; CHECK-LABEL: @sdiv_by_minus1_vec(
; CHECK-NEXT:    [[DIV:%.*]] = sub <2 x i64> zeroinitializer, [[X:%.*]]
; CHECK-NEXT:    ret <2 x i64> [[DIV]]
;
  %div = sdiv <2 x i64> %x, <i64 -1, i64 -1>
  ret <2 x i64> %div
}

define <2 x i64> @sdiv_by_minus1_vec_undef_elt(<2 x i64> %x) {
; CHECK-LABEL: @sdiv_by_minus1_vec_undef_elt(
; CHECK-NEXT:    ret <2 x i64> undef
;
  %div = sdiv <2 x i64> %x, <i64 -1, i64 undef>
  ret <2 x i64> %div
}

define i32 @sdiv_by_sext_minus1(i1 %x, i32 %y) {
; CHECK-LABEL: @sdiv_by_sext_minus1(
; CHECK-NEXT:    [[DIV:%.*]] = sub i32 0, [[Y:%.*]]
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %sext = sext i1 %x to i32
  %div = sdiv i32 %y, %sext
  ret i32 %div
}

define <2 x i32> @sdiv_by_sext_minus1_vec(<2 x i1> %x, <2 x i32> %y) {
; CHECK-LABEL: @sdiv_by_sext_minus1_vec(
; CHECK-NEXT:    [[DIV:%.*]] = sub <2 x i32> zeroinitializer, [[Y:%.*]]
; CHECK-NEXT:    ret <2 x i32> [[DIV]]
;
  %sext = sext <2 x i1> %x to <2 x i32>
  %div = sdiv <2 x i32> %y, %sext
  ret <2 x i32> %div
}

define i8 @udiv_by_negative(i8 %x) {
; CHECK-LABEL: @udiv_by_negative(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ugt i8 [[X:%.*]], -7
; CHECK-NEXT:    [[A:%.*]] = zext i1 [[TMP1]] to i8
; CHECK-NEXT:    ret i8 [[A]]
;
  %A = udiv i8 %x, 250
  ret i8 %A
}

define i32 @udiv_by_minus1(i32 %A) {
; CHECK-LABEL: @udiv_by_minus1(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i32 [[A:%.*]], -1
; CHECK-NEXT:    [[B:%.*]] = zext i1 [[TMP1]] to i32
; CHECK-NEXT:    ret i32 [[B]]
;
  %B = udiv i32 %A, -1
  ret i32 %B
}

define <2 x i64> @udiv_by_minus1_vec(<2 x i64> %x) {
; CHECK-LABEL: @udiv_by_minus1_vec(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq <2 x i64> [[X:%.*]], <i64 -1, i64 -1>
; CHECK-NEXT:    [[DIV:%.*]] = zext <2 x i1> [[TMP1]] to <2 x i64>
; CHECK-NEXT:    ret <2 x i64> [[DIV]]
;
  %div = udiv <2 x i64> %x, <i64 -1, i64 -1>
  ret <2 x i64> %div
}

define i32 @udiv_by_sext_all_ones(i1 %x, i32 %y) {
; CHECK-LABEL: @udiv_by_sext_all_ones(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i32 [[Y:%.*]], -1
; CHECK-NEXT:    [[DIV:%.*]] = zext i1 [[TMP1]] to i32
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %sext = sext i1 %x to i32
  %div = udiv i32 %y, %sext
  ret i32 %div
}

define <2 x i32> @udiv_by_sext_all_ones_vec(<2 x i1> %x, <2 x i32> %y) {
; CHECK-LABEL: @udiv_by_sext_all_ones_vec(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq <2 x i32> [[Y:%.*]], <i32 -1, i32 -1>
; CHECK-NEXT:    [[DIV:%.*]] = zext <2 x i1> [[TMP1]] to <2 x i32>
; CHECK-NEXT:    ret <2 x i32> [[DIV]]
;
  %sext = sext <2 x i1> %x to <2 x i32>
  %div = udiv <2 x i32> %y, %sext
  ret <2 x i32> %div
}

define i32 @test5(i32 %A) {
; CHECK-LABEL: @test5(
; CHECK-NEXT:    ret i32 0
;
  %B = udiv i32 %A, -16
  %C = udiv i32 %B, -4
  ret i32 %C
}

define i1 @test6(i32 %A) {
; CHECK-LABEL: @test6(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult i32 [[A:%.*]], 123
; CHECK-NEXT:    ret i1 [[TMP1]]
;
  %B = udiv i32 %A, 123
  ; A < 123
  %C = icmp eq i32 %B, 0
  ret i1 %C
}

define i1 @test7(i32 %A) {
; CHECK-LABEL: @test7(
; CHECK-NEXT:    [[A_OFF:%.*]] = add i32 [[A:%.*]], -20
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult i32 [[A_OFF]], 10
; CHECK-NEXT:    ret i1 [[TMP1]]
;
  %B = udiv i32 %A, 10
  ; A >= 20 && A < 30
  %C = icmp eq i32 %B, 2
  ret i1 %C
}

define <2 x i1> @test7vec(<2 x i32> %A) {
; CHECK-LABEL: @test7vec(
; CHECK-NEXT:    [[A_OFF:%.*]] = add <2 x i32> [[A:%.*]], <i32 -20, i32 -20>
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult <2 x i32> [[A_OFF]], <i32 10, i32 10>
; CHECK-NEXT:    ret <2 x i1> [[TMP1]]
;
  %B = udiv <2 x i32> %A, <i32 10, i32 10>
  %C = icmp eq <2 x i32> %B, <i32 2, i32 2>
  ret <2 x i1> %C
}

define i1 @test8(i8 %A) {
; CHECK-LABEL: @test8(
; CHECK-NEXT:    [[C:%.*]] = icmp ugt i8 [[A:%.*]], -11
; CHECK-NEXT:    ret i1 [[C]]
;
  %B = udiv i8 %A, 123
  ; A >= 246
  %C = icmp eq i8 %B, 2
  ret i1 %C
}

define <2 x i1> @test8vec(<2 x i8> %A) {
; CHECK-LABEL: @test8vec(
; CHECK-NEXT:    [[C:%.*]] = icmp ugt <2 x i8> [[A:%.*]], <i8 -11, i8 -11>
; CHECK-NEXT:    ret <2 x i1> [[C]]
;
  %B = udiv <2 x i8> %A, <i8 123, i8 123>
  %C = icmp eq <2 x i8> %B, <i8 2, i8 2>
  ret <2 x i1> %C
}

define i1 @test9(i8 %A) {
; CHECK-LABEL: @test9(
; CHECK-NEXT:    [[C:%.*]] = icmp ult i8 [[A:%.*]], -10
; CHECK-NEXT:    ret i1 [[C]]
;
  %B = udiv i8 %A, 123
  ; A < 246
  %C = icmp ne i8 %B, 2
  ret i1 %C
}

define <2 x i1> @test9vec(<2 x i8> %A) {
; CHECK-LABEL: @test9vec(
; CHECK-NEXT:    [[C:%.*]] = icmp ult <2 x i8> [[A:%.*]], <i8 -10, i8 -10>
; CHECK-NEXT:    ret <2 x i1> [[C]]
;
  %B = udiv <2 x i8> %A, <i8 123, i8 123>
  %C = icmp ne <2 x i8> %B, <i8 2, i8 2>
  ret <2 x i1> %C
}

define i32 @test10(i32 %X, i1 %C) {
; CHECK-LABEL: @test10(
; CHECK-NEXT:    [[R_V:%.*]] = select i1 [[C:%.*]], i32 6, i32 3
; CHECK-NEXT:    [[R:%.*]] = lshr i32 [[X:%.*]], [[R_V]]
; CHECK-NEXT:    ret i32 [[R]]
;
  %V = select i1 %C, i32 64, i32 8
  %R = udiv i32 %X, %V
  ret i32 %R
}

define i32 @test11(i32 %X, i1 %C) {
; CHECK-LABEL: @test11(
; CHECK-NEXT:    [[B_V:%.*]] = select i1 [[C:%.*]], i32 10, i32 5
; CHECK-NEXT:    [[B:%.*]] = lshr i32 [[X:%.*]], [[B_V]]
; CHECK-NEXT:    ret i32 [[B]]
;
  %A = select i1 %C, i32 1024, i32 32
  %B = udiv i32 %X, %A
  ret i32 %B
}

; PR2328
define i32 @test12(i32 %x) {
; CHECK-LABEL: @test12(
; CHECK-NEXT:    ret i32 1
;
  %tmp3 = udiv i32 %x, %x		; 1
  ret i32 %tmp3
}

define i32 @test13(i32 %x) {
; CHECK-LABEL: @test13(
; CHECK-NEXT:    ret i32 1
;
  %tmp3 = sdiv i32 %x, %x		; 1
  ret i32 %tmp3
}

define i32 @test14(i8 %x) {
; CHECK-LABEL: @test14(
; CHECK-NEXT:    ret i32 0
;
  %zext = zext i8 %x to i32
  %div = udiv i32 %zext, 257	; 0
  ret i32 %div
}

; PR9814
define i32 @test15(i32 %a, i32 %b) {
; CHECK-LABEL: @test15(
; CHECK-NEXT:    [[TMP1:%.*]] = add i32 [[B:%.*]], -2
; CHECK-NEXT:    [[DIV2:%.*]] = lshr i32 [[A:%.*]], [[TMP1]]
; CHECK-NEXT:    ret i32 [[DIV2]]
;
  %shl = shl i32 1, %b
  %div = lshr i32 %shl, 2
  %div2 = udiv i32 %a, %div
  ret i32 %div2
}

define <2 x i64> @test16(<2 x i64> %x) {
; CHECK-LABEL: @test16(
; CHECK-NEXT:    [[DIV:%.*]] = udiv <2 x i64> [[X:%.*]], <i64 192, i64 192>
; CHECK-NEXT:    ret <2 x i64> [[DIV]]
;
  %shr = lshr <2 x i64> %x, <i64 5, i64 5>
  %div = udiv <2 x i64> %shr, <i64 6, i64 6>
  ret <2 x i64> %div
}

define i32 @test19(i32 %x) {
; CHECK-LABEL: @test19(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i32 [[X:%.*]], 1
; CHECK-NEXT:    [[A:%.*]] = zext i1 [[TMP1]] to i32
; CHECK-NEXT:    ret i32 [[A]]
;
  %A = udiv i32 1, %x
  ret i32 %A
}

define <2 x i32> @test19vec(<2 x i32> %x) {
; CHECK-LABEL: @test19vec(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq <2 x i32> [[X:%.*]], <i32 1, i32 1>
; CHECK-NEXT:    [[A:%.*]] = zext <2 x i1> [[TMP1]] to <2 x i32>
; CHECK-NEXT:    ret <2 x i32> [[A]]
;
  %A = udiv <2 x i32> <i32 1, i32 1>, %x
  ret <2 x i32> %A
}

define i32 @test20(i32 %x) {
; CHECK-LABEL: @test20(
; CHECK-NEXT:    [[TMP1:%.*]] = add i32 [[X:%.*]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = icmp ult i32 [[TMP1]], 3
; CHECK-NEXT:    [[A:%.*]] = select i1 [[TMP2]], i32 [[X]], i32 0
; CHECK-NEXT:    ret i32 [[A]]
;
  %A = sdiv i32 1, %x
  ret i32 %A
}

define <2 x i32> @test20vec(<2 x i32> %x) {
; CHECK-LABEL: @test20vec(
; CHECK-NEXT:    [[TMP1:%.*]] = add <2 x i32> [[X:%.*]], <i32 1, i32 1>
; CHECK-NEXT:    [[TMP2:%.*]] = icmp ult <2 x i32> [[TMP1]], <i32 3, i32 3>
; CHECK-NEXT:    [[A:%.*]] = select <2 x i1> [[TMP2]], <2 x i32> [[X]], <2 x i32> zeroinitializer
; CHECK-NEXT:    ret <2 x i32> [[A]]
;
  %A = sdiv <2 x i32> <i32 1, i32 1>, %x
  ret <2 x i32> %A
}

define i32 @test21(i32 %a) {
; CHECK-LABEL: @test21(
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[A:%.*]], 3
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %shl = shl nsw i32 %a, 2
  %div = sdiv i32 %shl, 12
  ret i32 %div
}

define i32 @test22(i32 %a) {
; CHECK-LABEL: @test22(
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[A:%.*]], 4
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %mul = mul nsw i32 %a, 3
  %div = sdiv i32 %mul, 12
  ret i32 %div
}

define i32 @test23(i32 %a) {
; CHECK-LABEL: @test23(
; CHECK-NEXT:    [[DIV:%.*]] = udiv i32 [[A:%.*]], 3
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %shl = shl nuw i32 %a, 2
  %div = udiv i32 %shl, 12
  ret i32 %div
}

define i32 @test24(i32 %a) {
; CHECK-LABEL: @test24(
; CHECK-NEXT:    [[DIV:%.*]] = lshr i32 [[A:%.*]], 2
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %mul = mul nuw i32 %a, 3
  %div = udiv i32 %mul, 12
  ret i32 %div
}

define i32 @test25(i32 %a) {
; CHECK-LABEL: @test25(
; CHECK-NEXT:    [[DIV:%.*]] = shl nsw i32 [[A:%.*]], 1
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %shl = shl nsw i32 %a, 2
  %div = sdiv i32 %shl, 2
  ret i32 %div
}

define i32 @test26(i32 %a) {
; CHECK-LABEL: @test26(
; CHECK-NEXT:    [[DIV:%.*]] = shl nsw i32 [[A:%.*]], 2
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %mul = mul nsw i32 %a, 12
  %div = sdiv i32 %mul, 3
  ret i32 %div
}

define i32 @test27(i32 %a) {
; CHECK-LABEL: @test27(
; CHECK-NEXT:    [[DIV:%.*]] = shl nuw i32 [[A:%.*]], 1
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %shl = shl nuw i32 %a, 2
  %div = udiv i32 %shl, 2
  ret i32 %div
}

define i32 @test28(i32 %a) {
; CHECK-LABEL: @test28(
; CHECK-NEXT:    [[DIV:%.*]] = mul nuw i32 [[A:%.*]], 12
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %mul = mul nuw i32 %a, 36
  %div = udiv i32 %mul, 3
  ret i32 %div
}

define i32 @test29(i32 %a) {
; CHECK-LABEL: @test29(
; CHECK-NEXT:    [[MUL_LOBIT:%.*]] = and i32 [[A:%.*]], 1
; CHECK-NEXT:    ret i32 [[MUL_LOBIT]]
;
  %mul = shl nsw i32 %a, 31
  %div = sdiv i32 %mul, -2147483648
  ret i32 %div
}

define i32 @test30(i32 %a) {
; CHECK-LABEL: @test30(
; CHECK-NEXT:    ret i32 [[A:%.*]]
;
  %mul = shl nuw i32 %a, 31
  %div = udiv i32 %mul, -2147483648
  ret i32 %div
}

define <2 x i32> @test31(<2 x i32> %x) {
; CHECK-LABEL: @test31(
; CHECK-NEXT:    ret <2 x i32> zeroinitializer
;
  %shr = lshr <2 x i32> %x, <i32 31, i32 31>
  %div = udiv <2 x i32> %shr, <i32 2147483647, i32 2147483647>
  ret <2 x i32> %div
}

define i32 @test32(i32 %a, i32 %b) {
; CHECK-LABEL: @test32(
; CHECK-NEXT:    [[SHL:%.*]] = shl i32 2, [[B:%.*]]
; CHECK-NEXT:    [[DIV:%.*]] = lshr i32 [[SHL]], 2
; CHECK-NEXT:    [[DIV2:%.*]] = udiv i32 [[A:%.*]], [[DIV]]
; CHECK-NEXT:    ret i32 [[DIV2]]
;
  %shl = shl i32 2, %b
  %div = lshr i32 %shl, 2
  %div2 = udiv i32 %a, %div
  ret i32 %div2
}

define <2 x i64> @test33(<2 x i64> %x) {
; CHECK-LABEL: @test33(
; CHECK-NEXT:    [[DIV:%.*]] = udiv exact <2 x i64> [[X:%.*]], <i64 192, i64 192>
; CHECK-NEXT:    ret <2 x i64> [[DIV]]
;
  %shr = lshr exact <2 x i64> %x, <i64 5, i64 5>
  %div = udiv exact <2 x i64> %shr, <i64 6, i64 6>
  ret <2 x i64> %div
}

; -X / C --> X / -C (if negation does not overflow)

define i8 @sdiv_negated_dividend_constant_divisor(i8 %x) {
; CHECK-LABEL: @sdiv_negated_dividend_constant_divisor(
; CHECK-NEXT:    [[D:%.*]] = sdiv i8 [[X:%.*]], 42
; CHECK-NEXT:    ret i8 [[D]]
;
  %neg = sub nsw i8 0, %x
  %d = sdiv i8 %neg, -42
  ret i8 %d
}

define <2 x i8> @sdiv_negated_dividend_constant_divisor_vec_splat(<2 x i8> %x) {
; CHECK-LABEL: @sdiv_negated_dividend_constant_divisor_vec_splat(
; CHECK-NEXT:    [[D:%.*]] = sdiv <2 x i8> [[X:%.*]], <i8 42, i8 42>
; CHECK-NEXT:    ret <2 x i8> [[D]]
;
  %neg = sub nsw <2 x i8> zeroinitializer, %x
  %d = sdiv <2 x i8> %neg, <i8 -42, i8 -42>
  ret <2 x i8> %d
}

define i8 @sdiv_exact_negated_dividend_constant_divisor(i8 %x) {
; CHECK-LABEL: @sdiv_exact_negated_dividend_constant_divisor(
; CHECK-NEXT:    [[D:%.*]] = sdiv exact i8 [[X:%.*]], 42
; CHECK-NEXT:    ret i8 [[D]]
;
  %neg = sub nsw i8 0, %x
  %d = sdiv exact i8 %neg, -42
  ret i8 %d
}

define <2 x i8> @sdiv_exact_negated_dividend_constant_divisor_vec_splat(<2 x i8> %x) {
; CHECK-LABEL: @sdiv_exact_negated_dividend_constant_divisor_vec_splat(
; CHECK-NEXT:    [[D:%.*]] = sdiv exact <2 x i8> [[X:%.*]], <i8 42, i8 42>
; CHECK-NEXT:    ret <2 x i8> [[D]]
;
  %neg = sub nsw <2 x i8> zeroinitializer, %x
  %d = sdiv exact <2 x i8> %neg, <i8 -42, i8 -42>
  ret <2 x i8> %d
}

define i8 @sdiv_negated_dividend_constant_divisor_smin(i8 %x) {
; CHECK-LABEL: @sdiv_negated_dividend_constant_divisor_smin(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i8 [[X:%.*]], -128
; CHECK-NEXT:    [[D:%.*]] = zext i1 [[TMP1]] to i8
; CHECK-NEXT:    ret i8 [[D]]
;
  %neg = sub nsw i8 0, %x
  %d = sdiv i8 %neg, -128
  ret i8 %d
}

define <2 x i8> @sdiv_negated_dividend_constant_divisor_vec_splat_smin(<2 x i8> %x) {
; CHECK-LABEL: @sdiv_negated_dividend_constant_divisor_vec_splat_smin(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq <2 x i8> [[X:%.*]], <i8 -128, i8 -128>
; CHECK-NEXT:    [[D:%.*]] = zext <2 x i1> [[TMP1]] to <2 x i8>
; CHECK-NEXT:    ret <2 x i8> [[D]]
;
  %neg = sub nsw <2 x i8> zeroinitializer, %x
  %d = sdiv <2 x i8> %neg, <i8 -128, i8 -128>
  ret <2 x i8> %d
}

define <2 x i8> @sdiv_negated_dividend_constant_divisor_vec_undef(<2 x i8> %x) {
; CHECK-LABEL: @sdiv_negated_dividend_constant_divisor_vec_undef(
; CHECK-NEXT:    ret <2 x i8> undef
;
  %neg = sub nsw <2 x i8> zeroinitializer, %x
  %d = sdiv <2 x i8> %neg, <i8 -128, i8 undef>
  ret <2 x i8> %d
}

define <2 x i64> @sdiv_negated_dividend_constant_divisor_vec(<2 x i64> %x) {
; CHECK-LABEL: @sdiv_negated_dividend_constant_divisor_vec(
; CHECK-NEXT:    [[NEG:%.*]] = sub nsw <2 x i64> zeroinitializer, [[X:%.*]]
; CHECK-NEXT:    [[DIV:%.*]] = sdiv <2 x i64> [[NEG]], <i64 3, i64 4>
; CHECK-NEXT:    ret <2 x i64> [[DIV]]
;
  %neg = sub nsw <2 x i64> zeroinitializer, %x
  %div = sdiv <2 x i64> %neg, <i64 3, i64 4>
  ret <2 x i64> %div
}

define <2 x i64> @sdiv_exact_negated_dividend_constant_divisor_vec(<2 x i64> %x) {
; CHECK-LABEL: @sdiv_exact_negated_dividend_constant_divisor_vec(
; CHECK-NEXT:    [[NEG:%.*]] = sub nsw <2 x i64> zeroinitializer, [[X:%.*]]
; CHECK-NEXT:    [[DIV:%.*]] = sdiv exact <2 x i64> [[NEG]], <i64 3, i64 4>
; CHECK-NEXT:    ret <2 x i64> [[DIV]]
;
  %neg = sub nsw <2 x i64> zeroinitializer, %x
  %div = sdiv exact <2 x i64> %neg, <i64 3, i64 4>
  ret <2 x i64> %div
}

; Can't negate signed min vector element.

define <2 x i8> @sdiv_exact_negated_dividend_constant_divisor_vec_overflow(<2 x i8> %x) {
; CHECK-LABEL: @sdiv_exact_negated_dividend_constant_divisor_vec_overflow(
; CHECK-NEXT:    [[NEG:%.*]] = sub nsw <2 x i8> zeroinitializer, [[X:%.*]]
; CHECK-NEXT:    [[DIV:%.*]] = sdiv exact <2 x i8> [[NEG]], <i8 -128, i8 42>
; CHECK-NEXT:    ret <2 x i8> [[DIV]]
;
  %neg = sub nsw <2 x i8> zeroinitializer, %x
  %div = sdiv exact <2 x i8> %neg, <i8 -128, i8 42>
  ret <2 x i8> %div
}

define i32 @test35(i32 %A) {
; CHECK-LABEL: @test35(
; CHECK-NEXT:    [[AND:%.*]] = and i32 [[A:%.*]], 2147483647
; CHECK-NEXT:    [[MUL:%.*]] = udiv exact i32 [[AND]], 2147483647
; CHECK-NEXT:    ret i32 [[MUL]]
;
  %and = and i32 %A, 2147483647
  %mul = sdiv exact i32 %and, 2147483647
  ret i32 %mul
}

define <2 x i32> @test35vec(<2 x i32> %A) {
; CHECK-LABEL: @test35vec(
; CHECK-NEXT:    [[AND:%.*]] = and <2 x i32> [[A:%.*]], <i32 2147483647, i32 2147483647>
; CHECK-NEXT:    [[MUL:%.*]] = udiv exact <2 x i32> [[AND]], <i32 2147483647, i32 2147483647>
; CHECK-NEXT:    ret <2 x i32> [[MUL]]
;
  %and = and <2 x i32> %A, <i32 2147483647, i32 2147483647>
  %mul = sdiv exact <2 x i32> %and, <i32 2147483647, i32 2147483647>
  ret <2 x i32> %mul
}

define i32 @test36(i32 %A) {
; CHECK-LABEL: @test36(
; CHECK-NEXT:    [[AND:%.*]] = and i32 [[A:%.*]], 2147483647
; CHECK-NEXT:    [[MUL:%.*]] = lshr exact i32 [[AND]], [[A]]
; CHECK-NEXT:    ret i32 [[MUL]]
;
  %and = and i32 %A, 2147483647
  %shl = shl nsw i32 1, %A
  %mul = sdiv exact i32 %and, %shl
  ret i32 %mul
}

define <2 x i32> @test36vec(<2 x i32> %A) {
; CHECK-LABEL: @test36vec(
; CHECK-NEXT:    [[AND:%.*]] = and <2 x i32> [[A:%.*]], <i32 2147483647, i32 2147483647>
; CHECK-NEXT:    [[MUL:%.*]] = lshr exact <2 x i32> [[AND]], [[A]]
; CHECK-NEXT:    ret <2 x i32> [[MUL]]
;
  %and = and <2 x i32> %A, <i32 2147483647, i32 2147483647>
  %shl = shl nsw <2 x i32> <i32 1, i32 1>, %A
  %mul = sdiv exact <2 x i32> %and, %shl
  ret <2 x i32> %mul
}

define i32 @test37(i32* %b) {
; CHECK-LABEL: @test37(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    store i32 0, i32* [[B:%.*]], align 4
; CHECK-NEXT:    br i1 undef, label [[LOR_RHS:%.*]], label [[LOR_END:%.*]]
; CHECK:       lor.rhs:
; CHECK-NEXT:    br label [[LOR_END]]
; CHECK:       lor.end:
; CHECK-NEXT:    ret i32 0
;
entry:
  store i32 0, i32* %b, align 4
  %0 = load i32, i32* %b, align 4
  br i1 undef, label %lor.rhs, label %lor.end

lor.rhs:                                          ; preds = %entry
  %mul = mul nsw i32 undef, %0
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %t.0 = phi i32 [ %0, %entry ], [ %mul, %lor.rhs ]
  %div = sdiv i32 %t.0, 2
  ret i32 %div
}

; We can perform the division in the smaller type.

define i32 @shrink(i8 %x) {
; CHECK-LABEL: @shrink(
; CHECK-NEXT:    [[TMP1:%.*]] = sdiv i8 [[X:%.*]], 127
; CHECK-NEXT:    [[DIV:%.*]] = sext i8 [[TMP1]] to i32
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %conv = sext i8 %x to i32
  %div = sdiv i32 %conv, 127
  ret i32 %div
}

; Division in the smaller type can lead to more optimizations.

define i32 @zap(i8 %x) {
; CHECK-LABEL: @zap(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i8 [[X:%.*]], -128
; CHECK-NEXT:    [[DIV:%.*]] = zext i1 [[TMP1]] to i32
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %conv = sext i8 %x to i32
  %div = sdiv i32 %conv, -128
  ret i32 %div
}

; Splat constant divisors should get the same folds.

define <3 x i32> @shrink_vec(<3 x i8> %x) {
; CHECK-LABEL: @shrink_vec(
; CHECK-NEXT:    [[TMP1:%.*]] = sdiv <3 x i8> [[X:%.*]], <i8 127, i8 127, i8 127>
; CHECK-NEXT:    [[DIV:%.*]] = sext <3 x i8> [[TMP1]] to <3 x i32>
; CHECK-NEXT:    ret <3 x i32> [[DIV]]
;
  %conv = sext <3 x i8> %x to <3 x i32>
  %div = sdiv <3 x i32> %conv, <i32 127, i32 127, i32 127>
  ret <3 x i32> %div
}

define <2 x i32> @zap_vec(<2 x i8> %x) {
; CHECK-LABEL: @zap_vec(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq <2 x i8> [[X:%.*]], <i8 -128, i8 -128>
; CHECK-NEXT:    [[DIV:%.*]] = zext <2 x i1> [[TMP1]] to <2 x i32>
; CHECK-NEXT:    ret <2 x i32> [[DIV]]
;
  %conv = sext <2 x i8> %x to <2 x i32>
  %div = sdiv <2 x i32> %conv, <i32 -128, i32 -128>
  ret <2 x i32> %div
}

; But we can't do this if the signed constant won't fit in the original type.

define i32 @shrink_no(i8 %x) {
; CHECK-LABEL: @shrink_no(
; CHECK-NEXT:    [[CONV:%.*]] = sext i8 [[X:%.*]] to i32
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 [[CONV]], 128
; CHECK-NEXT:    ret i32 [[DIV]]
;
  %conv = sext i8 %x to i32
  %div = sdiv i32 %conv, 128
  ret i32 %div
}

; When the divisor is known larger than the quotient,
; InstSimplify should kill it before InstCombine sees it.

define i32 @shrink_no2(i8 %x) {
; CHECK-LABEL: @shrink_no2(
; CHECK-NEXT:    ret i32 0
;
  %conv = sext i8 %x to i32
  %div = sdiv i32 %conv, -129
  ret i32 %div
}

define i32 @shrink_no3(i16 %x) {
; CHECK-LABEL: @shrink_no3(
; CHECK-NEXT:    ret i32 0
;
  %conv = sext i16 %x to i32
  %div = sdiv i32 %conv, 65535
  ret i32 %div
}

; This previously crashed when trying to simplify the zext/icmp this becomes.
define <2 x i8> @PR34841(<2 x i8> %x) {
; CHECK-LABEL: @PR34841(
; CHECK-NEXT:    ret <2 x i8> zeroinitializer
;
  %neg = and <2 x i8> %x, <i8 2, i8 2>
  %div = udiv <2 x i8> <i8 1, i8 1>, %neg
  ret <2 x i8> %div
}

; X / (X * Y) -> 1 / Y if the multiplication does not overflow

define i8 @div_factor_signed(i8 %x, i8 %y) {
; CHECK-LABEL: @div_factor_signed(
; CHECK-NEXT:    [[TMP1:%.*]] = add i8 [[Y:%.*]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = icmp ult i8 [[TMP1]], 3
; CHECK-NEXT:    [[R:%.*]] = select i1 [[TMP2]], i8 [[Y]], i8 0
; CHECK-NEXT:    ret i8 [[R]]
;
  %a = mul nsw i8 %x, %y
  %r = sdiv i8 %x, %a
  ret i8 %r
}

; X / (Y * X) -> 1 / Y if the multiplication does not overflow

define <2 x i8> @div_factor_signed_vec(<2 x i8> %x, <2 x i8> %y) {
; CHECK-LABEL: @div_factor_signed_vec(
; CHECK-NEXT:    [[TMP1:%.*]] = add <2 x i8> [[Y:%.*]], <i8 1, i8 1>
; CHECK-NEXT:    [[TMP2:%.*]] = icmp ult <2 x i8> [[TMP1]], <i8 3, i8 3>
; CHECK-NEXT:    [[R:%.*]] = select <2 x i1> [[TMP2]], <2 x i8> [[Y]], <2 x i8> zeroinitializer
; CHECK-NEXT:    ret <2 x i8> [[R]]
;
  %a = mul nsw <2 x i8> %y, %x
  %r = sdiv <2 x i8> %x, %a
  ret <2 x i8> %r
}

; X / (Y * X) -> 1 / Y if the multiplication does not overflow

define i8 @div_factor_unsigned(i8 %x, i8 %y) {
; CHECK-LABEL: @div_factor_unsigned(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i8 [[Y:%.*]], 1
; CHECK-NEXT:    [[R:%.*]] = zext i1 [[TMP1]] to i8
; CHECK-NEXT:    ret i8 [[R]]
;
  %a = mul nuw i8 %y, %x
  %r = udiv i8 %x, %a
  ret i8 %r
}

; X / (X * Y) -> 1 / Y if the multiplication does not overflow

define <2 x i8> @div_factor_unsigned_vec(<2 x i8> %x, <2 x i8> %y) {
; CHECK-LABEL: @div_factor_unsigned_vec(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq <2 x i8> [[Y:%.*]], <i8 1, i8 1>
; CHECK-NEXT:    [[R:%.*]] = zext <2 x i1> [[TMP1]] to <2 x i8>
; CHECK-NEXT:    ret <2 x i8> [[R]]
;
  %a = mul nuw <2 x i8> %x, %y
  %r = udiv <2 x i8> %x, %a
  ret <2 x i8> %r
}

define i8 @udiv_common_factor(i8 %x, i8 %y, i8 %z) {
; CHECK-LABEL: @udiv_common_factor(
; CHECK-NEXT:    [[C:%.*]] = udiv i8 [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    ret i8 [[C]]
;
  %a = mul nuw i8 %z, %x
  %b = mul nuw i8 %z, %y
  %c = udiv i8 %a, %b
  ret i8 %c
}

define <2 x i8> @udiv_common_factor_commute1_vec(<2 x i8> %x, <2 x i8> %y, <2 x i8> %z) {
; CHECK-LABEL: @udiv_common_factor_commute1_vec(
; CHECK-NEXT:    [[C:%.*]] = udiv <2 x i8> [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    ret <2 x i8> [[C]]
;
  %a = mul nuw <2 x i8> %x, %z
  %b = mul nuw <2 x i8> %z, %y
  %c = udiv <2 x i8> %a, %b
  ret <2 x i8> %c
}

define i8 @udiv_common_factor_commute2(i8 %x, i8 %y, i8 %z) {
; CHECK-LABEL: @udiv_common_factor_commute2(
; CHECK-NEXT:    [[C:%.*]] = udiv i8 [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    ret i8 [[C]]
;
  %a = mul nuw i8 %x, %z
  %b = mul nuw i8 %y, %z
  %c = udiv i8 %a, %b
  ret i8 %c
}

define i8 @udiv_common_factor_commute3(i8 %x, i8 %y, i8 %z) {
; CHECK-LABEL: @udiv_common_factor_commute3(
; CHECK-NEXT:    [[C:%.*]] = udiv i8 [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    ret i8 [[C]]
;
  %a = mul nuw i8 %z, %x
  %b = mul nuw i8 %y, %z
  %c = udiv i8 %a, %b
  ret i8 %c
}

; Negative test: both mul must be 'nuw'.

define i8 @udiv_common_factor_not_nuw(i8 %x, i8 %y, i8 %z) {
; CHECK-LABEL: @udiv_common_factor_not_nuw(
; CHECK-NEXT:    [[A:%.*]] = mul i8 [[Z:%.*]], [[X:%.*]]
; CHECK-NEXT:    [[B:%.*]] = mul nuw i8 [[Z]], [[Y:%.*]]
; CHECK-NEXT:    [[C:%.*]] = udiv i8 [[A]], [[B]]
; CHECK-NEXT:    ret i8 [[C]]
;
  %a = mul i8 %z, %x
  %b = mul nuw i8 %z, %y
  %c = udiv i8 %a, %b
  ret i8 %c
}

; Negative test: both mul must be 'nuw'.

define <2 x i8> @udiv_common_factor_not_nuw_vec(<2 x i8> %x, <2 x i8> %y, <2 x i8> %z) {
; CHECK-LABEL: @udiv_common_factor_not_nuw_vec(
; CHECK-NEXT:    [[A:%.*]] = mul nuw <2 x i8> [[Z:%.*]], [[X:%.*]]
; CHECK-NEXT:    [[B:%.*]] = mul <2 x i8> [[Z]], [[Y:%.*]]
; CHECK-NEXT:    [[C:%.*]] = udiv <2 x i8> [[A]], [[B]]
; CHECK-NEXT:    ret <2 x i8> [[C]]
;
  %a = mul nuw <2 x i8> %z, %x
  %b = mul <2 x i8> %z, %y
  %c = udiv <2 x i8> %a, %b
  ret <2 x i8> %c
}

define i32 @test_exact_nsw_exact(i32 %x) {
; CHECK-LABEL: @test_exact_nsw_exact(
; CHECK-NEXT:    [[NEG:%.*]] = sdiv exact i32 [[X:%.*]], -3
; CHECK-NEXT:    ret i32 [[NEG]]
;
  %div = sdiv exact i32 %x, 3
  %neg = sub nsw i32 0, %div
  ret i32 %neg
}

define <2 x i64> @test_exact_vec(<2 x i64> %x) {
; CHECK-LABEL: @test_exact_vec(
; CHECK-NEXT:    [[DIV:%.*]] = sdiv exact <2 x i64> [[X:%.*]], <i64 3, i64 4>
; CHECK-NEXT:    [[NEG:%.*]] = sub nsw <2 x i64> zeroinitializer, [[DIV]]
; CHECK-NEXT:    ret <2 x i64> [[NEG]]
;
  %div = sdiv exact <2 x i64> %x, <i64 3, i64 4>
  %neg = sub nsw <2 x i64> zeroinitializer, %div
  ret <2 x i64> %neg
}

; Constant is safe to negate.

define <2 x i8> @negate_sdiv_vec_splat(<2 x i8> %x) {
; CHECK-LABEL: @negate_sdiv_vec_splat(
; CHECK-NEXT:    [[NEG:%.*]] = sdiv <2 x i8> [[X:%.*]], <i8 -42, i8 -42>
; CHECK-NEXT:    ret <2 x i8> [[NEG]]
;
  %div = sdiv <2 x i8> %x, <i8 42, i8 42>
  %neg = sub <2 x i8> zeroinitializer, %div
  ret <2 x i8> %neg
}

; Dividing by undef is UB.

define <2 x i8> @negate_sdiv_vec_undef_elt(<2 x i8> %x) {
; CHECK-LABEL: @negate_sdiv_vec_undef_elt(
; CHECK-NEXT:    ret <2 x i8> undef
;
  %div = sdiv <2 x i8> %x, <i8 undef, i8 42>
  %neg = sub <2 x i8> zeroinitializer, %div
  ret <2 x i8> %neg
}

; Division by -1 may be UB (if numerator is the signed min val), but div-by-1 can be simplified.

define <2 x i8> @negate_sdiv_vec_splat_one(<2 x i8> %x) {
; CHECK-LABEL: @negate_sdiv_vec_splat_one(
; CHECK-NEXT:    [[NEG:%.*]] = sub <2 x i8> zeroinitializer, [[X:%.*]]
; CHECK-NEXT:    ret <2 x i8> [[NEG]]
;
  %div = sdiv <2 x i8> %x, <i8 1, i8 1>
  %neg = sub <2 x i8> zeroinitializer, %div
  ret <2 x i8> %neg
}

; Can't negate signed-min constant, but can convert to a compare..

define <2 x i8> @negate_sdiv_vec_splat_signed_min(<2 x i8> %x) {
; CHECK-LABEL: @negate_sdiv_vec_splat_signed_min(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq <2 x i8> [[X:%.*]], <i8 -128, i8 -128>
; CHECK-NEXT:    [[NEG:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i8>
; CHECK-NEXT:    ret <2 x i8> [[NEG]]
;
  %div = sdiv <2 x i8> %x, <i8 -128, i8 -128>
  %neg = sub <2 x i8> zeroinitializer, %div
  ret <2 x i8> %neg
}

; Division by -1 may be UB for any element of a vector.

define <2 x i8> @negate_sdiv_vec_one_element(<2 x i8> %x) {
; CHECK-LABEL: @negate_sdiv_vec_one_element(
; CHECK-NEXT:    [[DIV:%.*]] = sdiv <2 x i8> [[X:%.*]], <i8 -1, i8 1>
; CHECK-NEXT:    [[NEG:%.*]] = sub <2 x i8> zeroinitializer, [[DIV]]
; CHECK-NEXT:    ret <2 x i8> [[NEG]]
;
  %div = sdiv <2 x i8> %x, <i8 -1, i8 1>
  %neg = sub <2 x i8> zeroinitializer, %div
  ret <2 x i8> %neg
}

; Can't negate signed-min constant for any element of a vector.

define <2 x i8> @negate_sdiv_vec_signed_min_elt(<2 x i8> %x) {
; CHECK-LABEL: @negate_sdiv_vec_signed_min_elt(
; CHECK-NEXT:    [[DIV:%.*]] = sdiv <2 x i8> [[X:%.*]], <i8 -1, i8 -128>
; CHECK-NEXT:    [[NEG:%.*]] = sub <2 x i8> zeroinitializer, [[DIV]]
; CHECK-NEXT:    ret <2 x i8> [[NEG]]
;
  %div = sdiv <2 x i8> %x, <i8 -1, i8 -128>
  %neg = sub <2 x i8> zeroinitializer, %div
  ret <2 x i8> %neg
}

; Division by -1 may be UB and can't negate signed-min.

define <2 x i8> @negate_sdiv_vec_signed_min_and_one_elt(<2 x i8> %x) {
; CHECK-LABEL: @negate_sdiv_vec_signed_min_and_one_elt(
; CHECK-NEXT:    [[DIV:%.*]] = sdiv <2 x i8> [[X:%.*]], <i8 1, i8 -128>
; CHECK-NEXT:    [[NEG:%.*]] = sub <2 x i8> zeroinitializer, [[DIV]]
; CHECK-NEXT:    ret <2 x i8> [[NEG]]
;
  %div = sdiv <2 x i8> %x, <i8 1, i8 -128>
  %neg = sub <2 x i8> zeroinitializer, %div
  ret <2 x i8> %neg
}

define i32 @test_exact_nonsw_exact(i32 %x) {
; CHECK-LABEL: @test_exact_nonsw_exact(
; CHECK-NEXT:    [[NEG:%.*]] = sdiv exact i32 [[X:%.*]], -3
; CHECK-NEXT:    ret i32 [[NEG]]
;
  %div = sdiv exact i32 %x, 3
  %neg = sub i32 0, %div
  ret i32 %neg
}

define i32 @test_exact_nsw_noexact(i32 %x) {
; CHECK-LABEL: @test_exact_nsw_noexact(
; CHECK-NEXT:    [[NEG:%.*]] = sdiv i32 [[X:%.*]], -3
; CHECK-NEXT:    ret i32 [[NEG]]
;
  %div = sdiv i32 %x, 3
  %neg = sub nsw i32 0, %div
  ret i32 %neg
}

define i32 @test_exact_nonsw_noexact(i32 %x) {
; CHECK-LABEL: @test_exact_nonsw_noexact(
; CHECK-NEXT:    [[NEG:%.*]] = sdiv i32 [[X:%.*]], -3
; CHECK-NEXT:    ret i32 [[NEG]]
;
  %div = sdiv i32 %x, 3
  %neg = sub i32 0, %div
  ret i32 %neg
}

define i32 @test_exact_div_nonconst(i32 %x, i32 %y) {
; CHECK-LABEL: @test_exact_div_nonconst(
; CHECK-NEXT:    [[DIV:%.*]] = sdiv exact i32 [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    [[NEG:%.*]] = sub nsw i32 0, [[DIV]]
; CHECK-NEXT:    ret i32 [[NEG]]
;
  %div = sdiv exact i32 %x, %y
  %neg = sub nsw i32 0, %div
  ret i32 %neg
}

define i32 @test_exact_div_one(i32 %x) {
; CHECK-LABEL: @test_exact_div_one(
; CHECK-NEXT:    [[NEG:%.*]] = sub nsw i32 0, [[X:%.*]]
; CHECK-NEXT:    ret i32 [[NEG]]
;
  %div = sdiv exact i32 %x, 1
  %neg = sub nsw i32 0, %div
  ret i32 %neg
}

define i8 @test_exact_div_minSigned(i8 %x) {
; CHECK-LABEL: @test_exact_div_minSigned(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i8 [[X:%.*]], -128
; CHECK-NEXT:    [[NEG:%.*]] = sext i1 [[TMP1]] to i8
; CHECK-NEXT:    ret i8 [[NEG]]
;
  %div = sdiv exact i8 %x, -128
  %neg = sub nsw i8 0, %div
  ret i8 %neg
}

; X / INT_MIN --> X == INT_MIN

define i8 @sdiv_by_int_min(i8 %x) {
; CHECK-LABEL: @sdiv_by_int_min(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i8 [[X:%.*]], -128
; CHECK-NEXT:    [[D:%.*]] = zext i1 [[TMP1]] to i8
; CHECK-NEXT:    ret i8 [[D]]
;
  %d = sdiv i8 %x, -128
  ret i8 %d
}

define <2 x i8> @sdiv_by_int_min_vec_splat(<2 x i8> %x) {
; CHECK-LABEL: @sdiv_by_int_min_vec_splat(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq <2 x i8> [[X:%.*]], <i8 -128, i8 -128>
; CHECK-NEXT:    [[D:%.*]] = zext <2 x i1> [[TMP1]] to <2 x i8>
; CHECK-NEXT:    ret <2 x i8> [[D]]
;
  %d = sdiv <2 x i8> %x, <i8 -128, i8 -128>
  ret <2 x i8> %d
}

define <2 x i8> @sdiv_by_int_min_vec_splat_undef(<2 x i8> %x) {
; CHECK-LABEL: @sdiv_by_int_min_vec_splat_undef(
; CHECK-NEXT:    ret <2 x i8> undef
;
  %d = sdiv <2 x i8> %x, <i8 -128, i8 undef>
  ret <2 x i8> %d
}
