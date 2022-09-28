! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! Test abs intrinsic for various types (int, float, complex)

! CHECK-LABEL: func @_QPabs_testi
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>
subroutine abs_testi(a, b)
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:  %[[VAL_3:.*]] = arith.constant 31 : i32
! CHECK:  %[[VAL_4:.*]] = arith.shrsi %[[VAL_2]], %[[VAL_3]] : i32
! CHECK:  %[[VAL_5:.*]] = arith.xori %[[VAL_2]], %[[VAL_4]] : i32
! CHECK:  %[[VAL_6:.*]] = arith.subi %[[VAL_5]], %[[VAL_4]] : i32
! CHECK:  fir.store %[[VAL_6]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:  return
  integer :: a, b
  b = abs(a)
end subroutine

! CHECK-LABEL: func @_QPabs_testr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<f32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<f32>{{.*}}) {
subroutine abs_testr(a, b)
! CHECK: %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
! CHECK: %[[VAL_3:.*]] = math.absf %[[VAL_2]] : f32
! CHECK: fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<f32>
! CHECK: return
  real :: a, b
  b = abs(a)
end subroutine

! CHECK-LABEL: func @_QPabs_testd(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<f64>{{.*}}, %[[VAL_1:.*]]: !fir.ref<f64>{{.*}}) {
subroutine abs_testd(a, b)
! CHECK: %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<f64>
! CHECK: %[[VAL_3:.*]] = math.absf %[[VAL_2]] : f64
! CHECK: fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<f64>
! CHECK: return
  real(kind=8) :: a, b
  b = abs(a)
end subroutine

! CHECK-LABEL: func @_QPabs_testzr(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.complex<4>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<f32>{{.*}}) {
subroutine abs_testzr(a, b)
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_3:.*]] = fir.call @cabsf(%[[VAL_2]]) : (!fir.complex<4>) -> f32
! CHECK:  fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<f32>
! CHECK:  return
  complex :: a
  real :: b
  b = abs(a)
end subroutine abs_testzr

! CHECK-LABEL: func @_QPabs_testzd(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.complex<8>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<f64>{{.*}}) {
subroutine abs_testzd(a, b)
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.complex<8>>
! CHECK:  %[[VAL_3:.*]] = fir.call @cabs(%[[VAL_2]]) : (!fir.complex<8>) -> f64
! CHECK:  fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<f64>
! CHECK:  return
  complex(kind=8) :: a
  real(kind=8) :: b
  b = abs(a)
end subroutine abs_testzd
