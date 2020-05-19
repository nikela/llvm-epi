// RUN: %clang_cc1 -triple riscv64 -mepi -verify -fsyntax-only %s

void foo(void)
{
  __epi_2xf32 va;
  &va; // expected-error {{address of EPI vector requested}}

  __epi_1xi64x2 va_2;
  &va_2; // expected-error {{address of EPI vector requested}}
}

void bar(void *p)
{
  *(__epi_2xf32 *)p; // expected-error {{pointer to EPI vector type '__epi_2xf32' (vector of 'float') is invalid}}
  *(__epi_1xi64x2 *)p; // expected-error {{pointer to EPI vector type '__epi_1xi64x2' (aka 'struct __epi_1xi64x2') is invalid}}
}
