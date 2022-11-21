! Testing that derived types can be used by value.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

module moo
  implicit none

  type mytype
    integer :: x
  end type mytype

  contains
    subroutine myfoo(a)
      implicit none
      type(mytype), value :: a

      if (a % x /= 42) stop "invalid input"
      a % x = a % x + 1
    end subroutine myfoo

    subroutine mysub(x)
      implicit none
      integer :: x
      type(mytype) :: k

      k % x = x
      call myfoo(k)
      if (k % x /= 42) stop "invalid output"
    end subroutine mysub

    subroutine myblah
      call mysub(42)
    end subroutine myblah
end module moo

! CHECK-LABEL: func.func @_QMmooPmyfoo(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.ref<!fir.type<_QMmooTmytype{x:i32}>> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_1:.*]] = fir.field_index x, !fir.type<_QMmooTmytype{x:i32}>
! CHECK:         %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<!fir.type<_QMmooTmytype{x:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK:         %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_4:.*]] = arith.constant 42 : i32
! CHECK:         %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i32
! CHECK:         cf.cond_br %[[VAL_5]], ^bb1, ^bb2
! CHECK:       ^bb1:
! CHECK:         %[[VAL_6:.*]] = fir.address_of(@_QQcl.696E76616C696420696E707574) : !fir.ref<!fir.char<1,13>>
! CHECK:         %[[VAL_7:.*]] = arith.constant 13 : index
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_7]] : (index) -> i64
! CHECK:         %[[VAL_10:.*]] = arith.constant false
! CHECK:         %[[VAL_11:.*]] = arith.constant false
! CHECK:         %[[VAL_12:.*]] = fir.call @_FortranAStopStatementText(%[[VAL_8]], %[[VAL_9]], %[[VAL_10]], %[[VAL_11]]) fastmath<contract> : (!fir.ref<i8>, i64, i1, i1) -> none
! CHECK:         fir.unreachable
! CHECK:       ^bb2:
! CHECK:         %[[VAL_13:.*]] = fir.field_index x, !fir.type<_QMmooTmytype{x:i32}>
! CHECK:         %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_13]] : (!fir.ref<!fir.type<_QMmooTmytype{x:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK:         %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_17:.*]] = arith.addi %[[VAL_15]], %[[VAL_16]] : i32
! CHECK:         %[[VAL_18:.*]] = fir.field_index x, !fir.type<_QMmooTmytype{x:i32}>
! CHECK:         %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_18]] : (!fir.ref<!fir.type<_QMmooTmytype{x:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK:         fir.store %[[VAL_17]] to %[[VAL_19]] : !fir.ref<i32>
! CHECK:         return
! CHECK:       }
!
! CHECK-LABEL: func.func @_QMmooPmysub(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "x"}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.type<_QMmooTmytype{x:i32}>
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.type<_QMmooTmytype{x:i32}> {bindc_name = "k", uniq_name = "_QMmooFmysubEk"}
! CHECK:         %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_4:.*]] = fir.field_index x, !fir.type<_QMmooTmytype{x:i32}>
! CHECK:         %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_4]] : (!fir.ref<!fir.type<_QMmooTmytype{x:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK:         fir.store %[[VAL_3]] to %[[VAL_5]] : !fir.ref<i32>
! CHECK:         %[[VAL_6:.*]] = fir.field_index x, !fir.type<_QMmooTmytype{x:i32}>
! CHECK:         %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_6]] : (!fir.ref<!fir.type<_QMmooTmytype{x:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK:         %[[VAL_8:.*]] = fir.field_index x, !fir.type<_QMmooTmytype{x:i32}>
! CHECK:         %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_8]] : (!fir.ref<!fir.type<_QMmooTmytype{x:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
! CHECK:         fir.store %[[VAL_10]] to %[[VAL_9]] : !fir.ref<i32>
! CHECK:         fir.call @_QMmooPmyfoo(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.type<_QMmooTmytype{x:i32}>>) -> ()
! CHECK:         %[[VAL_11:.*]] = fir.field_index x, !fir.type<_QMmooTmytype{x:i32}>
! CHECK:         %[[VAL_12:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_11]] : (!fir.ref<!fir.type<_QMmooTmytype{x:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK:         %[[VAL_13:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
! CHECK:         %[[VAL_14:.*]] = arith.constant 42 : i32
! CHECK:         %[[VAL_15:.*]] = arith.cmpi ne, %[[VAL_13]], %[[VAL_14]] : i32
! CHECK:         cf.cond_br %[[VAL_15]], ^bb1, ^bb2
! CHECK:       ^bb1:
! CHECK:         %[[VAL_16:.*]] = fir.address_of(@_QQcl.696E76616C6964206F7574707574) : !fir.ref<!fir.char<1,14>>
! CHECK:         %[[VAL_17:.*]] = arith.constant 14 : index
! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,14>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_17]] : (index) -> i64
! CHECK:         %[[VAL_20:.*]] = arith.constant false
! CHECK:         %[[VAL_21:.*]] = arith.constant false
! CHECK:         %[[VAL_22:.*]] = fir.call @_FortranAStopStatementText(%[[VAL_18]], %[[VAL_19]], %[[VAL_20]], %[[VAL_21]]) fastmath<contract> : (!fir.ref<i8>, i64, i1, i1) -> none
! CHECK:         fir.unreachable
! CHECK:       ^bb2:
! CHECK:         return
! CHECK:       }
!
! CHECK-LABEL: func.func @_QMmooPmyblah() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref}
! CHECK:         %[[VAL_1:.*]] = arith.constant 42 : i32
! CHECK:         fir.store %[[VAL_1]] to %[[VAL_0]] : !fir.ref<i32>
! CHECK:         fir.call @_QMmooPmysub(%[[VAL_0]]) fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:         return
! CHECK:       }
