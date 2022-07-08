// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

void foo(int Y, int *A) {
#pragma omp simd simdlen(omp_max_simdlen)
  for (int I = 0; I < Y; I++) {
    A[I] = I + 1;
  }
#pragma omp simd simdlen(omp_max_simdlen : 1)
  for (int I = 0; I < Y; I++) {
    A[I] = I + 1;
  }
#pragma omp simd simdlen(omp_max_simdlen : 0) // expected-error {{argument to 'simdlen' clause must be a strictly positive integer value}}
  for (int I = 0; I < Y; I++) {
    A[I] = I + 1;
  }
#pragma omp simd simdlen(omp_max_simdlen:) // expected-error {{expected expression}}
  for (int I = 0; I < Y; I++) {
    A[I] = I + 1;
  }
#pragma omp simd simdlen(omp_max_simdlen 1) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int I = 0; I < Y; I++) {
    A[I] = I + 1;
  }
#pragma omp simd simdlen(max_simdlen : 1) // expected-error {{use of undeclared identifier 'max_simdlen'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int I = 0; I < Y; I++) {
    A[I] = I + 1;
  }
}
