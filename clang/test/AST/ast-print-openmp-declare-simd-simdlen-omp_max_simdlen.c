// RUN: %clang_cc1 -ast-print -fopenmp -vectorize-wfv %s
// RUN: %clang_cc1 -ast-print -fopenmp-simd -vectorize-wfv %s

// CHECK: #pragma omp declare simd simdlen(omp_max_simdlen)
#pragma omp declare simd simdlen(omp_max_simdlen)
double add1(double A, double B, double Fact) {
  double C;
  C = A + B + Fact;
  return C;
}

// CHECK: #pragma omp declare simd simdlen(omp_max_simdlen: 1)
#pragma omp declare simd simdlen(omp_max_simdlen : 1)
double add2(double A, double B, double Fact) {
  double C;
  C = A + B + Fact;
  return C;
}
