! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

module my_mod
  use iso_c_binding
  implicit none

  contains

   integer function my_error_handler(estack_id, data_inout) bind(c)
     integer :: estack_id, data_inout
     my_error_handler = 42
   end function

   subroutine test(x)
     type(c_funptr) :: x

     x = c_funloc(my_error_handler)
   end subroutine test

end module my_mod

! CHECK-LABEL: func.func @_QMmy_modPtest(
! CHECK-SAME:                            %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>> {fir.bindc_name = "x"}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}> {adapt.valuebyref}
! CHECK:         %[[VAL_2:.*]] = fir.address_of(@my_error_handler) : (!fir.ref<i32>, !fir.ref<i32>) -> i32
! CHECK:         %[[VAL_3:.*]] = fir.undefined !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_2]] : ((!fir.ref<i32>, !fir.ref<i32>) -> i32) -> i64
! CHECK:         %[[VAL_5:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_4]], ["__address", !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>] : (!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>, i64) -> !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         fir.store %[[VAL_5]] to %[[VAL_1]] : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>
! CHECK:         %[[VAL_6:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_6]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_8:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_8]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_7]] : !fir.ref<i64>
! CHECK:         fir.store %[[VAL_10]] to %[[VAL_9]] : !fir.ref<i64>
! CHECK:         return
! CHECK:       }
