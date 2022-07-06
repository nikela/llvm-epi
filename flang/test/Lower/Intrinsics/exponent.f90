! RUN: bbc -emit-fir %s -o - | FileCheck %s

! EXPONENT
! CHECK-LABEL: exponent_test
subroutine exponent_test

    integer :: i1, i2, i4
  ! CHECK: %[[i0:.*]] = fir.alloca i32 {bindc_name = "i1", uniq_name = "_QFexponent_testEi1"}
  ! CHECK: %[[i1:.*]] = fir.alloca i32 {bindc_name = "i2", uniq_name = "_QFexponent_testEi2"}
  ! CHECK: %[[i3:.*]] = fir.alloca i32 {bindc_name = "i4", uniq_name = "_QFexponent_testEi4"}
  
    real(kind = 4) :: x1
    real(kind = 8) :: x2
  ! CHECK: %[[x0:.*]] = fir.alloca f32 {bindc_name = "x1", uniq_name = "_QFexponent_testEx1"}
  ! CHECK: %[[x1:.*]] = fir.alloca f64 {bindc_name = "x2", uniq_name = "_QFexponent_testEx2"}
  
    i1 = exponent(x1)
  ! CHECK: %[[temp0:.*]] = fir.load %[[x0:.*]] : !fir.ref<f32>
  ! CHECK: %[[result0:.*]] = fir.call @_FortranAExponent4_4(%[[temp0:.*]]) : (f32) -> i32
  ! CHECK: fir.store %[[result0:.*]] to %[[i0:.*]] : !fir.ref<i32>
  
    i2 = exponent(x2)
  ! CHECK: %[[temp1:.*]] = fir.load %[[x1:.*]] : !fir.ref<f64>
  ! CHECK: %[[result1:.*]] = fir.call @_FortranAExponent8_4(%[[temp1:.*]]) : (f64) -> i32
  ! CHECK: fir.store %[[result1:.*]] to %[[i1:.*]] : !fir.ref<i32>
  
  end subroutine exponent_test
