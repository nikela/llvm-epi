! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

subroutine case1(l)
  implicit none
  integer :: x
  integer(kind=8) :: l

  l = loc(x)
end subroutine

subroutine case2(l, n)
  implicit none
  integer :: n
  integer :: a(n)
  integer(kind=8) :: l

  l = loc(a)
end subroutine

subroutine case3(l, a)
  implicit none
  integer :: a(:)
  integer(kind=8) :: l

  l = loc(a)
end subroutine

subroutine case4(l, pa)
  implicit none
  integer, pointer :: pa(:)
  integer(kind=8) :: l

  l = loc(pa)
end subroutine
! CHECK-LABEL: func.func @_QPcase1(
! CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "l"}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFcase1Ex"}
! CHECK:         %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<i32>) -> i64
! CHECK:         fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<i64>
! CHECK:         return
! CHECK:       }
!
! CHECK-LABEL: func.func @_QPcase2(
! CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "l"},
! CHECK-SAME:                      %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:         %[[VAL_2:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i32) -> i64
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:         %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_6:.*]] = arith.cmpi sgt, %[[VAL_4]], %[[VAL_5]] : index
! CHECK:         %[[VAL_7:.*]] = arith.select %[[VAL_6]], %[[VAL_4]], %[[VAL_5]] : index
! CHECK:         %[[VAL_8:.*]] = fir.alloca !fir.array<?xi32>, %[[VAL_7]] {bindc_name = "a", uniq_name = "_QFcase2Ea"}
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.array<?xi32>>) -> i64
! CHECK:         fir.store %[[VAL_9]] to %[[VAL_0]] : !fir.ref<i64>
! CHECK:         return
! CHECK:       }
!
! CHECK-LABEL: func.func @_QPcase3(
! CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "l"},
! CHECK-SAME:                      %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<?xi32>>) -> i64
! CHECK:         fir.store %[[VAL_3]] to %[[VAL_0]] : !fir.ref<i64>
! CHECK:         return
! CHECK:       }
!
! CHECK-LABEL: func.func @_QPcase4(
! CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "l"},
! CHECK-SAME:                      %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "pa"}) {
! CHECK:         %[[VAL_2:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ref<!fir.ptr<!fir.array<?xi32>>>
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.ptr<!fir.array<?xi32>>>) -> i64
! CHECK:         fir.store %[[VAL_4]] to %[[VAL_0]] : !fir.ref<i64>
! CHECK:         return
! CHECK:       }

