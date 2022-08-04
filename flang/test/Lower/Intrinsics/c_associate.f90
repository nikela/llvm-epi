! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

subroutine sub1(p, l)
  use iso_c_binding
  implicit none
  type(c_ptr) :: p
  logical :: l

  l = c_associated(p)
end subroutine sub1

subroutine sub2(p1, p2, l)
  use iso_c_binding
  implicit none
  type(c_ptr) :: p1
  type(c_ptr), optional :: p2
  logical :: l

  l = c_associated(p1, p2)
end subroutine sub2

subroutine sub3(p, l)
  use iso_c_binding
  implicit none
  type(c_funptr) :: p
  logical :: l

  l = c_associated(p)
end subroutine sub3

subroutine sub4(p1, p2, l)
  use iso_c_binding
  implicit none
  type(c_funptr) :: p1
  type(c_funptr), optional :: p2
  logical :: l

  l = c_associated(p1, p2)
end subroutine sub4

! CHECK-LABEL: func.func @_QPsub1(
! CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "p"},
! CHECK-SAME:                     %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l"}) {
! CHECK:         %[[VAL_2:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<i64>
! CHECK:         %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i1) -> !fir.logical<4>
! CHECK:         fir.store %[[VAL_7]] to %[[VAL_1]] : !fir.ref<!fir.logical<4>>
! CHECK:         return
! CHECK:       }
!
! CHECK-LABEL: func.func @_QPsub2(
! CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "p1"},
! CHECK-SAME:                     %[[VAL_1:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "p2", fir.optional},
! CHECK-SAME:                     %[[VAL_2:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l"}) {
! CHECK:         %[[VAL_3:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_3]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i64>
! CHECK:         %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_3]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.ref<i64>
! CHECK:         %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_7]] : i64
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i1) -> !fir.logical<4>
! CHECK:         fir.store %[[VAL_9]] to %[[VAL_2]] : !fir.ref<!fir.logical<4>>
! CHECK:         return
! CHECK:       }
!
! CHECK-LABEL: func.func @_QPsub3(
! CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>> {fir.bindc_name = "p"},
! CHECK-SAME:                     %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l"}) {
! CHECK:         %[[VAL_2:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<i64>
! CHECK:         %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i1) -> !fir.logical<4>
! CHECK:         fir.store %[[VAL_7]] to %[[VAL_1]] : !fir.ref<!fir.logical<4>>
! CHECK:         return
! CHECK:       }
!
! CHECK-LABEL: func.func @_QPsub4(
! CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>> {fir.bindc_name = "p1"},
! CHECK-SAME:                     %[[VAL_1:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>> {fir.bindc_name = "p2", fir.optional},
! CHECK-SAME:                     %[[VAL_2:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l"}) {
! CHECK:         %[[VAL_3:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>
! CHECK:         %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_3]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i64>
! CHECK:         %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_3]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.ref<i64>
! CHECK:         %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_7]] : i64
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i1) -> !fir.logical<4>
! CHECK:         fir.store %[[VAL_9]] to %[[VAL_2]] : !fir.ref<!fir.logical<4>>
! CHECK:         return
! CHECK:       }
