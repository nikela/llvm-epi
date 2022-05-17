! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes=CHECK %s
! RUN: flang-new -fc1 -fdefault-integer-8 -emit-fir %s -o - | FileCheck --check-prefix=CHECK-DEFAULT-KIND-8 %s
function nonconstant(prec) result(k)
! CHECK-LABEL: func @_QPnonconstant(
! CHECK: %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFnonconstantEk"}
! CHECK-NEXT: %[[VAL_1:.*]] = fir.load %[[VAL_2:.*]] : !fir.ref<i32>
! CHECK-NEXT: %[[VAL_3:.*]] = fir.call @_FortranASelectedIntKind(%[[VAL_1]]) : (i32) -> i32
! CHECK-NEXT: fir.store %[[VAL_3]] to %[[VAL_0]] : !fir.ref<i32>
! CHECK-NEXT: %[[VAL_4:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK-NEXT: return %[[VAL_4]] : i32

! CHECK-DEFAULT-KIND-8-LABEL: func @_QPnonconstant(
! CHECK-DEFAULT-KIND-8: %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "prec"}) -> i64 {
! CHECK-DEFAULT-KIND-8-NEXT: %[[VAL_1:.*]] = fir.alloca i64 {bindc_name = "k", uniq_name = "_QFnonconstantEk"}
! CHECK-DEFAULT-KIND-8-NEXT: %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<i64>
! CHECK-DEFAULT-KIND-8-NEXT: %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i64) -> i32
! CHECK-DEFAULT-KIND-8-NEXT: %[[VAL_4:.*]] = fir.call @_FortranASelectedIntKind(%[[VAL_3]]) : (i32) -> i32
! CHECK-DEFAULT-KIND-8-NEXT: %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> i64
! CHECK-DEFAULT-KIND-8-NEXT: fir.store %[[VAL_5]] to %[[VAL_1]] : !fir.ref<i64>
! CHECK-DEFAULT-KIND-8-NEXT: %[[VAL_6:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
! CHECK-DEFAULT-KIND-8-NEXT: return %[[VAL_6]] : i64

  implicit none
  integer :: prec, k
  k = selected_int_kind(prec)
end function nonconstant

