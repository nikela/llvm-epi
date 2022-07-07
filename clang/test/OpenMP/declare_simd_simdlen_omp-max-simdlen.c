// RUN: %clang_cc1 -verify -fopenmp -vectorize-wfv %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -vectorize-wfv %s -Wuninitialized

#pragma omp declare simd simdlen(omp_max_simdlen)
double add1(double A, double B, double Fact) {
  double C;
  C = A + B + Fact;
  return C;
}

#pragma omp declare simd simdlen(omp_max_simdlen : 1)
double add2(double A, double B, double Fact) {
  double C;
  C = A + B + Fact;
  return C;
}

#pragma omp declare simd simdlen(omp_max_simdlen : 0) // expected-error {{argument to 'simdlen' clause must be a strictly positive integer value}}
double add3(double A, double B, double Fact) {
  double C;
  C = A + B + Fact;
  return C;
}

#pragma omp declare simd simdlen(omp_max_simdlen:) // expected-error {{expected expression}}
double add4(double A, double B, double Fact) {
  double C;
  C = A + B + Fact;
  return C;
}

#pragma omp declare simd simdlen(omp_max_simdlen 1) // expected-error {{expected ')'}} expected-note {{to match this '('}}
double add5(double A, double B, double Fact) {
  double C;
  C = A + B + Fact;
  return C;
}

#pragma omp declare simd simdlen(max_simdlen : 1) // expected-error {{use of undeclared identifier 'max_simdlen'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
double add6(double A, double B, double Fact) {
  double C;
  C = A + B + Fact;
  return C;
}
