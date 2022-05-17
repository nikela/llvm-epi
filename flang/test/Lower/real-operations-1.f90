! RUN: bbc %s -o - | FileCheck %s

! Test real add on real kinds.

! CHECK-LABEL: real4
REAL(4) FUNCTION real4(x0, x1)
  REAL(4) :: x0
  REAL(4) :: x1
  ! CHECK-DAG: %[[v1:.+]] = fir.load %arg0 : !fir.ref<f32>
  ! CHECK-DAG: %[[v2:.+]] = fir.load %arg1 : !fir.ref<f32>
  ! CHECK: %[[v3:.+]] = arith.addf %[[v1]], %[[v2]] : f32
  real4 = x0 + x1
  ! CHECK: return %{{.*}} : f32
END FUNCTION real4

! CHECK-LABEL: defreal
REAL FUNCTION defreal(x0, x1)
  REAL :: x0
  REAL :: x1
  ! CHECK-DAG: %[[v1:.+]] = fir.load %arg0 : !fir.ref<f32>
  ! CHECK-DAG: %[[v2:.+]] = fir.load %arg1 : !fir.ref<f32>
  ! CHECK: %[[v3:.+]] = arith.addf %[[v1]], %[[v2]] : f32
  defreal = x0 + x1
  ! CHECK: return %{{.*}} : f32
END FUNCTION defreal

! CHECK-LABEL: real8
REAL(8) FUNCTION real8(x0, x1)
  REAL(8) :: x0
  REAL(8) :: x1
  ! CHECK-DAG: %[[v1:.+]] = fir.load %arg0 : !fir.ref<f64>
  ! CHECK-DAG: %[[v2:.+]] = fir.load %arg1 : !fir.ref<f64>
  ! CHECK: %[[v3:.+]] = arith.addf %[[v1]], %[[v2]] : f64
  real8 = x0 + x1
  ! CHECK: return %{{.*}} : f64
END FUNCTION real8

! CHECK-LABEL: doubleprec
DOUBLE PRECISION FUNCTION doubleprec(x0, x1)
  DOUBLE PRECISION :: x0
  DOUBLE PRECISION :: x1
  ! CHECK-DAG: %[[v1:.+]] = fir.load %arg0 : !fir.ref<f64>
  ! CHECK-DAG: %[[v2:.+]] = fir.load %arg1 : !fir.ref<f64>
  ! CHECK: %[[v3:.+]] = arith.addf %[[v1]], %[[v2]] : f64
  doubleprec = x0 + x1
  ! CHECK: return %{{.*}} : f64
END FUNCTION doubleprec

! CHECK-LABEL: real16
REAL(16) FUNCTION real16(x0, x1)
  REAL(16) :: x0
  REAL(16) :: x1
  ! CHECK-DAG: %[[v1:.+]] = fir.load %arg0 : !fir.ref<f128>
  ! CHECK-DAG: %[[v2:.+]] = fir.load %arg1 : !fir.ref<f128>
  ! CHECK: %[[v3:.+]] = arith.addf %[[v1]], %[[v2]] : f128
  real16 = x0 + x1
  ! CHECK: return %{{.*}} : f128
END FUNCTION real16

! CHECK-LABEL: real16b
REAL(16) FUNCTION real16b(x0, x1)
  REAL(16) :: x0
  REAL(16) :: x1
  ! CHECK-DAG: %[[v0:.+]] = arith.constant 4.0{{.*}} : f128
  ! CHECK-DAG: %[[v1:.+]] = fir.load %arg0 : !fir.ref<f128>
  ! CHECK-DAG: %[[v2:.+]] = fir.load %arg1 : !fir.ref<f128>
  ! CHECK: %[[v3:.+]] = arith.addf %[[v1]], %[[v2]] : f128
  ! CHECK: %[[v4:.+]] = arith.subf %[[v3]], %[[v0]] : f128
  real16b = x0 + x1 - 4.0_16
  ! CHECK: return %{{.*}} : f128
END FUNCTION real16b
