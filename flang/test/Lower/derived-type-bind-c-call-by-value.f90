! RUN: bbc -emit-fir -o - %s | FileCheck %s
module moo
  implicit none

  type, bind(c) :: mytype
    integer(kind=8) :: myaddr
  end type mytype

contains

  subroutine foo(buf_ptr, y)
    implicit none
    type(mytype)  , intent(inout)           :: buf_ptr
    integer :: y

    integer :: w
    interface
      integer function bar(x, buf_ptr) bind(c, name='bar_c')
        import :: mytype
        implicit none
        integer, value :: x
        type(mytype)    , value      :: buf_ptr
      end function bar
    end interface

    w = bar(y, buf_ptr)
  end subroutine foo
end module moo

! CHECK-LABEL: func.func @_QMmooPfoo(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.type<_QMmooTmytype{myaddr:i64}>> {fir.bindc_name = "buf_ptr"},
! CHECK-SAME:                        %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "y"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "w", uniq_name = "_QMmooFfooEw"}
! CHECK:         %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.type<_QMmooTmytype{myaddr:i64}>>
! CHECK:         %[[VAL_5:.*]] = fir.call @bar_c(%[[VAL_3]], %[[VAL_4]]) : (i32, !fir.type<_QMmooTmytype{myaddr:i64}>) -> i32
! CHECK:         fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:         return
! CHECK:       }
! CHECK:       func.func private @bar_c(i32, !fir.type<_QMmooTmytype{myaddr:i64}>) -> i32 attributes {fir.bindc_name = "bar_c"}
