! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPrrspacing_test2(
! CHECK-SAME: %[[x:[^:]+]]: !fir.ref<f64>{{.*}}) -> f64
real*8 function rrspacing_test2(x)
  real*8 :: x
  rrspacing_test2 = rrspacing(x)
! CHECK: %[[a1:.*]] = fir.load %[[x]] : !fir.ref<f64>
! CHECK: %{{.*}} = fir.call @_FortranARRSpacing8(%[[a1]]) : (f64) -> f64
end function
