! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPverify_r4
function verify_r4(a) result(l)
! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_2:.*]] : !fir.ref<f32>
! CHECK:         %[[VAL_3:.*]] = arith.cmpf une, %[[VAL_1]], %[[VAL_1]] : f32
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i1) -> !fir.logical<4>

  use, intrinsic :: ieee_arithmetic, only : ieee_is_nan
  implicit none

  real(kind=4) :: a
  logical :: l

  l = ieee_is_nan(a)
end function verify_r4

! CHECK-LABEL: func @_QPverify_r8
function verify_r8(a) result(l)
! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_8:.*]] : !fir.ref<f64>
! CHECK:         %[[VAL_9:.*]] = arith.cmpf une, %[[VAL_7]], %[[VAL_7]] : f64
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i1) -> !fir.logical<4>

  use, intrinsic :: ieee_arithmetic, only : ieee_is_nan
  implicit none

  real(kind=8) :: a
  logical :: l

  l = ieee_is_nan(a)
end function verify_r8
