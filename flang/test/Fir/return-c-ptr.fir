// RUN: fir-opt %s --abstract-result-on-func-opt | FileCheck %s

func.func @_QPsub(%arg0: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "p"}) {
  %c3_i32 = arith.constant 3 : i32
  %0 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {adapt.valuebyref}
  %1 = fir.call @my_str(%c3_i32) : (i32) -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  fir.store %1 to %0 : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
  %2 = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
  %3 = fir.coordinate_of %0, %2 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
  %4 = fir.coordinate_of %arg0, %2 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
  %5 = fir.load %3 : !fir.ref<i64>
  fir.store %5 to %4 : !fir.ref<i64>
  return
}

// CHECK-LABEL:   func.func @_QPsub(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "p"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {adapt.valuebyref}
// CHECK:           %[[VAL_3:.*]] = fir.call @my_str(%[[VAL_1]]) : (i32) -> !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK:           fir.store %[[VAL_3]] to %[[VAL_2]] : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
// CHECK:           %[[VAL_4:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK:           %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_4]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
// CHECK:           %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_4]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
// CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_5]] : !fir.ref<i64>
// CHECK:           fir.store %[[VAL_7]] to %[[VAL_6]] : !fir.ref<i64>
// CHECK:           return
// CHECK:         }
