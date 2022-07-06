! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! SET_EXPONENT
! CHECK-LABEL: set_exponent_test
subroutine set_exponent_test

  real(kind = 4) :: x1 = 178.1378e-4
  real(kind = 8) :: x2 = 178.1378e-4
  integer :: i = 17
! CHECK: %[[addri:.*]] = fir.address_of(@_QFset_exponent_testEi) : !fir.ref<i32>
! CHECK: %[[addrx1:.*]] = fir.address_of(@_QFset_exponent_testEx1) : !fir.ref<f32>
! CHECK: %[[addrx2:.*]] = fir.address_of(@_QFset_exponent_testEx2) : !fir.ref<f64>

  x1 = set_exponent(x1, i)
! CHECK: %[[x1:.*]] = fir.load %[[addrx1:.*]] : !fir.ref<f32>
! CHECK: %[[i1:.*]] = fir.load %[[addri:.*]] : !fir.ref<i32>
! CHECK: %[[i64v1:.*]] = fir.convert %[[i1:.*]] : (i32) -> i64
! CHECK: %[[result1:.*]] = fir.call @_FortranASetExponent4(%[[x1:.*]], %[[i64v1:.*]]) : (f32, i64) -> f32
! CHECK: fir.store %[[result1:.*]] to %[[addrx1:.*]] : !fir.ref<f32>

  x2 = set_exponent(x2, i)
! CHECK: %[[x2:.*]] = fir.load %[[addrx2:.*]] : !fir.ref<f64>
! CHECK: %[[i2:.*]] = fir.load %[[addri:.*]] : !fir.ref<i32>
! CHECK: %[[i64v2:.*]] = fir.convert %[[i2:.*]] : (i32) -> i64
! CHECK: %[[result2:.*]] = fir.call @_FortranASetExponent8(%[[x2:.*]], %[[i64v2:.*]]) : (f64, i64) -> f64
! CHECK: fir.store %[[result2:.*]] to %[[addrx2:.*]] : !fir.ref<f64>

end subroutine set_exponent_test

