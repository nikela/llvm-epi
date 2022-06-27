! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

subroutine case1(cp, p)
  use iso_c_binding, only : c_loc, c_ptr
  implicit none
  integer, pointer :: p
  type(c_ptr) :: cp

  cp = c_loc(p)
end subroutine

subroutine case2(cp, pa)
  use iso_c_binding, only : c_loc, c_ptr
  implicit none
  integer, pointer :: pa(:)
  type(c_ptr) :: cp

  cp = c_loc(pa)
end subroutine

subroutine case3(cp, x)
  use iso_c_binding, only : c_loc, c_ptr
  implicit none
  integer, target :: x
  type(c_ptr) :: cp

  cp = c_loc(x)
end subroutine

subroutine case4(cp, a)
  use iso_c_binding, only : c_loc, c_ptr
  implicit none
  integer, target :: a(:)
  type(c_ptr) :: cp

  cp = c_loc(a)
end subroutine

subroutine case5(cp, a, n)
  use iso_c_binding, only : c_loc, c_ptr
  implicit none
  integer :: n
  integer, target :: a(n)
  type(c_ptr) :: cp

  cp = c_loc(a)
end subroutine
! CHECK-LABEL: func.func @_QPcase1(
! CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cp"},
! CHECK-SAME:                      %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "p"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {adapt.valuebyref}
! CHECK:         %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:         %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:         %[[VAL_5:.*]] = fir.embox %[[VAL_4]] : (!fir.ptr<i32>) -> !fir.box<i32>
! CHECK:         %[[VAL_6:.*]] = fir.box_addr %[[VAL_5]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK:         %[[VAL_7:.*]] = fir.undefined !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<i32>) -> i64
! CHECK:         %[[VAL_9:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_8]], ["__address", !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>] : (!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>, i64) -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         fir.store %[[VAL_9]] to %[[VAL_2]] : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:         %[[VAL_10:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_10]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_12:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_12]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_14:.*]] = fir.load %[[VAL_11]] : !fir.ref<i64>
! CHECK:         fir.store %[[VAL_14]] to %[[VAL_13]] : !fir.ref<i64>
! CHECK:         return
! CHECK:       }
!
! CHECK-LABEL: func.func @_QPcase2(
! CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cp"},
! CHECK-SAME:                      %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "pa"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {adapt.valuebyref}
! CHECK:         %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ref<!fir.ptr<!fir.array<?xi32>>>
! CHECK:         %[[VAL_5:.*]] = fir.undefined !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.ptr<!fir.array<?xi32>>>) -> i64
! CHECK:         %[[VAL_7:.*]] = fir.insert_value %[[VAL_5]], %[[VAL_6]], ["__address", !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>] : (!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>, i64) -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         fir.store %[[VAL_7]] to %[[VAL_2]] : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:         %[[VAL_8:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_8]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_10:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_10]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_9]] : !fir.ref<i64>
! CHECK:         fir.store %[[VAL_12]] to %[[VAL_11]] : !fir.ref<i64>
! CHECK:         return
! CHECK:       }
!
! CHECK-LABEL: func.func @_QPcase3(
! CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cp"},
! CHECK-SAME:                      %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "x", fir.target}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {adapt.valuebyref}
! CHECK:         %[[VAL_3:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:         %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK:         %[[VAL_5:.*]] = fir.undefined !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<i32>) -> i64
! CHECK:         %[[VAL_7:.*]] = fir.insert_value %[[VAL_5]], %[[VAL_6]], ["__address", !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>] : (!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>, i64) -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         fir.store %[[VAL_7]] to %[[VAL_2]] : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:         %[[VAL_8:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_8]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_10:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_10]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_9]] : !fir.ref<i64>
! CHECK:         fir.store %[[VAL_12]] to %[[VAL_11]] : !fir.ref<i64>
! CHECK:         return
! CHECK:       }
!
! CHECK-LABEL: func.func @_QPcase4(
! CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cp"},
! CHECK-SAME:                      %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a", fir.target}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {adapt.valuebyref}
! CHECK:         %[[VAL_3:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:         %[[VAL_4:.*]] = fir.undefined !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.array<?xi32>>) -> i64
! CHECK:         %[[VAL_6:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_5]], ["__address", !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>] : (!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>, i64) -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         fir.store %[[VAL_6]] to %[[VAL_2]] : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:         %[[VAL_7:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_7]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_9:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_9]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_11:.*]] = fir.load %[[VAL_8]] : !fir.ref<i64>
! CHECK:         fir.store %[[VAL_11]] to %[[VAL_10]] : !fir.ref<i64>
! CHECK:         return
! CHECK:       }
!
! CHECK-LABEL: func.func @_QPcase5(
! CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cp"},
! CHECK-SAME:                      %[[VAL_1:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "a", fir.target},
! CHECK-SAME:                      %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {adapt.valuebyref}
! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> i64
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_6]], %[[VAL_7]] : index
! CHECK:         %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_6]], %[[VAL_7]] : index
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.embox %[[VAL_1]](%[[VAL_10]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:         %[[VAL_12:.*]] = fir.box_addr %[[VAL_11]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:         %[[VAL_13:.*]] = fir.undefined !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.array<?xi32>>) -> i64
! CHECK:         %[[VAL_15:.*]] = fir.insert_value %[[VAL_13]], %[[VAL_14]], ["__address", !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>] : (!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>, i64) -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         fir.store %[[VAL_15]] to %[[VAL_3]] : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
! CHECK:         %[[VAL_16:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_17:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_16]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_18:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_18]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_20:.*]] = fir.load %[[VAL_17]] : !fir.ref<i64>
! CHECK:         fir.store %[[VAL_20]] to %[[VAL_19]] : !fir.ref<i64>
! CHECK:         return
! CHECK:       }
