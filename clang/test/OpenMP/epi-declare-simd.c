// RUN: %clang_cc1 -triple riscv64-linux-gnu -target-feature +zepi -fopenmp-simd -emit-llvm -O2 -o - %s | FileCheck %s --check-prefix=RISCV64

#pragma omp declare simd
#pragma omp declare simd simdlen(1)
#pragma omp declare simd simdlen(5)
float fsquare(float a) {
  return a * a;
}

// RISCV64: "_ZGVEM16v_fsquare" "_ZGVEM2v_fsquare" "_ZGVEM4v_fsquare" "_ZGVEM8v_fsquare" "_ZGVEN16v_fsquare" "_ZGVEN2v_fsquare" "_ZGVEN4v_fsquare" "_ZGVEN8v_fsquare"
// RISCV64-NOT: _ZGVEM1v_fsquare
// RISCV64-NOT: _ZGVEM5v_fsquare

#pragma omp declare simd simdlen(1)
double dsquare(double a) {
  return a * a;
}

// RISCV64: "_ZGVEM1v_dsquare" "_ZGVEN1v_dsquare"
// RISCV64-NOT: "_ZGVEM2v_dsquare" "_ZGVEM4v_dsquare" "_ZGVEM8v_dsquare" "_ZGVEN2v_dsquare" "_ZGVEN4v_dsquare" "_ZGVEN8v_dsquare"

#pragma omp declare simd notinbranch
double nbsquare(double a) {
  return a * a;
}

// RISCV64: "_ZGVEN1v_nbsquare" "_ZGVEN2v_nbsquare" "_ZGVEN4v_nbsquare" "_ZGVEN8v_nbsquare"
// RISCV64-NOT: "_ZGVEM1v_nbsquare" "_ZGVEM2v_nbsquare" "_ZGVEM4v_nbsquare" "_ZGVEM8v_nbsquare"

#pragma omp declare simd inbranch
double bsquare(double a) {
  return a * a;
}

// RISCV64: "_ZGVEM1v_bsquare" "_ZGVEM2v_bsquare" "_ZGVEM4v_bsquare" "_ZGVEM8v_bsquare"
// RISCV64-NOT: "_ZGVEN1v_bsquare" "_ZGVEN2v_bsquare" "_ZGVEN4v_bsquare" "_ZGVEN8v_bsquare"

#pragma omp declare simd simdlen(1) linear(a : 1)
double lsquare(double *a) {
  return *a * *a;
}

// RISCV64: "_ZGVEM1l8_lsquare" "_ZGVEN1l8_lsquare"

#pragma omp declare simd simdlen(1) uniform(b)
double usquare(double a, double b) {
  return a * b;
}

// RISCV64: "_ZGVEM1vu_usquare" "_ZGVEN1vu_usquare"
