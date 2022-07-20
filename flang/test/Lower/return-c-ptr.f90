! RUN: bbc -emit-fir %s -o - | FileCheck %s

subroutine sub(p)
  use iso_c_binding
  implicit none
  type(c_ptr) :: p

  interface
    function my_str(x) result(v) bind(c)
      use iso_c_binding
      implicit none
      integer(c_int), value :: x
      type(c_ptr) :: v
    end function my_str
  end interface

  p = my_str(3)
end subroutine sub

! CHECK-LABEL: func.func @_QPsub(
! CHECK-SAME:                    %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "p"}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {adapt.valuebyref}
! CHECK:         %[[VAL_2:.*]] = arith.constant 3 : i32
! CHECK:         %[[VAL_3:.*]] = fir.call @my_str(%[[VAL_2]]) : (i32) -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:         %[[VAL_4:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_4]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_6:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_6]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_5]] : !fir.ref<i64>
! CHECK:         fir.store %[[VAL_8]] to %[[VAL_7]] : !fir.ref<i64>
! CHECK:         return
! CHECK:       }
