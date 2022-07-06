! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: nearest_test1
subroutine nearest_test1(x, s)
    real :: x, s, res
  ! CHECK: %[[res:.*]] = fir.alloca f32 {bindc_name = "res", uniq_name = "_QFnearest_test1Eres"}
  ! CHECK: %[[x:.*]] = fir.load %arg0 : !fir.ref<f32>
  ! CHECK: %[[s:.*]] = fir.load %arg1 : !fir.ref<f32>
  ! CHECK: %[[zero:.*]] = arith.constant 0.000000e+00 : f32
  ! CHECK: %[[cmp:.*]] = arith.cmpf ogt, %[[s]], %[[zero]] : f32
  ! CHECK: %[[pos:.*]] = arith.select %[[cmp]], %true, %false : i1
    res = nearest(x, s)
  ! CHECK: %[[tmp:.*]] = fir.call @_FortranANearest4(%[[x]], %[[pos]]) : (f32, i1) -> f32
  ! CHECK: fir.store %[[tmp]] to %[[res]] : !fir.ref<f32>
  end subroutine nearest_test1
  
  ! CHECK-LABEL: nearest_test2
  subroutine nearest_test2(x, s)
    real(kind=8) :: x, s, res
  ! CHECK: %[[res:.*]] = fir.alloca f64 {bindc_name = "res", uniq_name = "_QFnearest_test2Eres"}
  ! CHECK: %[[x:.*]] = fir.load %arg0 : !fir.ref<f64>
  ! CHECK: %[[s:.*]] = fir.load %arg1 : !fir.ref<f64>
  ! CHECK: %[[zero:.*]] = arith.constant 0.000000e+00 : f64
  ! CHECK: %[[cmp:.*]] = arith.cmpf ogt, %[[s]], %[[zero]] : f64
  ! CHECK: %[[pos:.*]] = arith.select %[[cmp]], %true, %false : i1
    res = nearest(x, s)
  ! CHECK: %[[tmp:.*]] = fir.call @_FortranANearest8(%[[x]], %[[pos]]) : (f64, i1) -> f64
  ! CHECK: fir.store %[[tmp]] to %[[res]] : !fir.ref<f64>
  end subroutine nearest_test2
