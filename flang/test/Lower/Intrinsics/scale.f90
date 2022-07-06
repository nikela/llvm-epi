! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: scale_test1
subroutine scale_test1(x, i)
    real :: x, res
  ! CHECK: %[[res:.*]] = fir.alloca f32 {bindc_name = "res", uniq_name = "_QFscale_test1Eres"}
  ! CHECK: %[[x:.*]] = fir.load %arg0 : !fir.ref<f32>
    integer :: i
  ! CHECK: %[[i0:.*]] = fir.load %arg1 : !fir.ref<i32>
    res = scale(x, i)
  ! CHECK: %[[i1:.*]] = fir.convert %[[i0]] : (i32) -> i64
  ! CHECK: %[[tmp:.*]] = fir.call @_FortranAScale4(%[[x]], %[[i1]]) : (f32, i64) -> f32
  ! CHECK: fir.store %[[tmp]] to %[[res]] : !fir.ref<f32>
  end subroutine scale_test1
  
  ! CHECK-LABEL: scale_test2
  subroutine scale_test2(x, i)
    real(kind=8) :: x, res
  ! CHECK: %[[res:.*]] = fir.alloca f64 {bindc_name = "res", uniq_name = "_QFscale_test2Eres"}
  ! CHECK: %[[x:.*]] = fir.load %arg0 : !fir.ref<f64>
    integer :: i
  ! CHECK: %[[i0:.*]] = fir.load %arg1 : !fir.ref<i32>
    res = scale(x, i)
  ! CHECK: %[[i1:.*]] = fir.convert %[[i0]] : (i32) -> i64
  ! CHECK: %[[tmp:.*]] = fir.call @_FortranAScale8(%[[x]], %[[i1]]) : (f64, i64) -> f64
  ! CHECK: fir.store %[[tmp]] to %[[res]] : !fir.ref<f64>
  end subroutine scale_test2
