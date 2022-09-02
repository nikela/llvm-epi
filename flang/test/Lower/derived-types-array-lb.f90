! Test basic parts of derived type entities lowering
! RUN: bbc -emit-fir %s -o - | FileCheck %s

module moo
  type t
    integer :: a(2:9)
  end type t

  type t2
    integer :: a(11:99)
  end type t2

  type t3
    type(t2) :: y(22:55)
  end type t3

contains

  subroutine sub1(x)
    implicit none
    type(t) :: x
    x % a (3) = 42
  end subroutine sub1


  subroutine sub2(x)
    implicit none
    type(t3) :: x
    x % y(33) % a (44) = 1111
  end subroutine sub2

end module moo

! CHECK-LABEL: func.func @_QMmooPsub1(
! CHECK-SAME:                         %[[VAL_0:.*]]: !fir.ref<!fir.type<_QMmooTt{a:!fir.array<8xi32>}>> {fir.bindc_name = "x"}) {
! CHECK:         %[[VAL_1:.*]] = arith.constant 42 : i32
! CHECK:         %[[VAL_2:.*]] = fir.field_index a, !fir.type<_QMmooTt{a:!fir.array<8xi32>}>
! CHECK:         %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<!fir.type<_QMmooTt{a:!fir.array<8xi32>}>>, !fir.field) -> !fir.ref<!fir.array<8xi32>>
! CHECK:         %[[VAL_4:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 3 : i64
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
! CHECK:         %[[VAL_7:.*]] = arith.subi %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:         %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_7]] : (!fir.ref<!fir.array<8xi32>>, i64) -> !fir.ref<i32>
! CHECK:         fir.store %[[VAL_1]] to %[[VAL_8]] : !fir.ref<i32>
! CHECK:         return
! CHECK:       }
!
! CHECK-LABEL: func.func @_QMmooPsub2(
! CHECK-SAME:                         %[[VAL_0:.*]]: !fir.ref<!fir.type<_QMmooTt3{y:!fir.array<34x!fir.type<_QMmooTt2{a:!fir.array<89xi32>}>>}>> {fir.bindc_name = "x"}) {
! CHECK:         %[[VAL_1:.*]] = arith.constant 1111 : i32
! CHECK:         %[[VAL_2:.*]] = fir.field_index y, !fir.type<_QMmooTt3{y:!fir.array<34x!fir.type<_QMmooTt2{a:!fir.array<89xi32>}>>}>
! CHECK:         %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<!fir.type<_QMmooTt3{y:!fir.array<34x!fir.type<_QMmooTt2{a:!fir.array<89xi32>}>>}>>, !fir.field) -> !fir.ref<!fir.array<34x!fir.type<_QMmooTt2{a:!fir.array<89xi32>}>>>
! CHECK:         %[[VAL_4:.*]] = arith.constant 22 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 33 : i64
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_4]] : (index) -> i64
! CHECK:         %[[VAL_7:.*]] = arith.subi %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:         %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_7]] : (!fir.ref<!fir.array<34x!fir.type<_QMmooTt2{a:!fir.array<89xi32>}>>>, i64) -> !fir.ref<!fir.type<_QMmooTt2{a:!fir.array<89xi32>}>>
! CHECK:         %[[VAL_9:.*]] = fir.field_index a, !fir.type<_QMmooTt2{a:!fir.array<89xi32>}>
! CHECK:         %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_9]] : (!fir.ref<!fir.type<_QMmooTt2{a:!fir.array<89xi32>}>>, !fir.field) -> !fir.ref<!fir.array<89xi32>>
! CHECK:         %[[VAL_11:.*]] = arith.constant 11 : index
! CHECK:         %[[VAL_12:.*]] = arith.constant 44 : i64
! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_11]] : (index) -> i64
! CHECK:         %[[VAL_14:.*]] = arith.subi %[[VAL_12]], %[[VAL_13]] : i64
! CHECK:         %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_10]], %[[VAL_14]] : (!fir.ref<!fir.array<89xi32>>, i64) -> !fir.ref<i32>
! CHECK:         fir.store %[[VAL_1]] to %[[VAL_15]] : !fir.ref<i32>
! CHECK:         return
! CHECK:       }

