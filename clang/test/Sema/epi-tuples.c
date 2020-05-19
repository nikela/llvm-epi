// RUN: %clang_cc1 -triple riscv64 -verify -mepi -fsyntax-only %s
// expected-no-diagnostics
void tuple_2(void)
{
  __epi_1xi64x2 va;
  (void)va.v0;
  (void)va.v1;
}

void tuple_4(void)
{
  __epi_1xi64x4 va;
  (void)va.v0;
  (void)va.v1;
  (void)va.v2;
  (void)va.v3;
}

void tuple_8(void)
{
  __epi_1xi64x8 va;
  (void)va.v0;
  (void)va.v1;
  (void)va.v2;
  (void)va.v3;
  (void)va.v4;
  (void)va.v5;
  (void)va.v6;
  (void)va.v7;
}

void tuple_write(__epi_1xi64 a)
{
  __epi_1xi64x2 ta;
  ta.v0 = a;
  ta.v1 = a;
  (void)ta;
}

void tuple_assig(void)
{
  __epi_1xi64x2 ta, tb;
  ta = tb;
  (void)ta;
  (void)tb;
}
