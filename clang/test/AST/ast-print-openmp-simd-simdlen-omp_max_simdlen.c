// RUN: %clang_cc1 -ast-print -fopenmp %s
// RUN: %clang_cc1 -ast-print -fopenmp-simd %s

void foo(int Y, int *A) {
  // CHECK: #pragma omp simd simdlen(omp_max_simdlen)
#pragma omp simd simdlen(omp_max_simdlen)
  for (int I = 0; I < Y; I++) {
    A[I] = I + 1;
  }
  // CHECK: #pragma omp simd simdlen(omp_max_simdlen: 2)
#pragma omp simd simdlen(omp_max_simdlen : 2)
  for (int I = 0; I < Y; I++) {
    A[I] = I + 1;
  }
}
