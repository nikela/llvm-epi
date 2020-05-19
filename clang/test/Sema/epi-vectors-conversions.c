// RUN: %clang_cc1 -triple riscv64 -mepi -verify -fsyntax-only %s

void foo(void) {
  __epi_2xi32 vc1;
  __epi_1xi64 vc2;
  __epi_1xi64x2 vc3;

  vc1 = vc2; // expected-error-re {{assigning to {{.*}} from incompatible type {{.*}}}}
  vc3 = vc1; // expected-error-re {{assigning to {{.*}} from incompatible type {{.*}}}}
}

