// RUN: %clang_cc1 -triple riscv64-linux-gnu -target-feature +zepi -fopenmp-simd -emit-llvm -O2 -o - %s | FileCheck %s --check-prefix=RISCV64
// RUN: %clang_cc1 -triple riscv64-linux-gnu -target-feature +zepi -fopenmp-simd -emit-llvm -O2 -o - %s -verify

#pragma omp declare simd
#pragma omp declare simd simdlen(1) // expected-warning {{The value specified in simdlen multiplied by the size of the data type must be included in the [64,512] range.}}
#pragma omp declare simd simdlen(5) // expected-warning {{The value specified in simdlen must be a power of 2.}}
float fsquare(float a) {
  return a * a;
}

// RISCV64: "_ZGVEMk16v_fsquare" "_ZGVEMk2v_fsquare" "_ZGVEMk4v_fsquare" "_ZGVEMk8v_fsquare" "_ZGVENk16v_fsquare" "_ZGVENk2v_fsquare" "_ZGVENk4v_fsquare" "_ZGVENk8v_fsquare"

#pragma omp declare simd simdlen(1)
double dsquare(double a) {
  return a * a;
}

// RISCV64: "_ZGVEMk1v_dsquare" "_ZGVENk1v_dsquare"

#pragma omp declare simd notinbranch
double nbsquare(double a) {
  return a * a;
}

// RISCV64: "_ZGVENk1v_nbsquare" "_ZGVENk2v_nbsquare" "_ZGVENk4v_nbsquare" "_ZGVENk8v_nbsquare"

#pragma omp declare simd inbranch
double bsquare(double a) {
  return a * a;
}

// RISCV64: "_ZGVEMk1v_bsquare" "_ZGVEMk2v_bsquare" "_ZGVEMk4v_bsquare" "_ZGVEMk8v_bsquare"

#pragma omp declare simd simdlen(1) linear(a : 1)
double lsquare(double *a) {
  return *a * *a;
}

// RISCV64: "_ZGVEMk1l8_lsquare" "_ZGVENk1l8_lsquare"

#pragma omp declare simd simdlen(1) uniform(b)
double usquare(double a, double b) {
  return a * b;
}

// RISCV64: "_ZGVEMk1vu_usquare" "_ZGVENk1vu_usquare"
