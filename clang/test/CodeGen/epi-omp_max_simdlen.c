// RUN: %clang -mepi -O2 -S -emit-llvm -fopenmp -mllvm -opt-bisect-limit=76 \
// RUN:   -o - %s 2>&1 | FileCheck %s

void star(double *A, double *B, double *C, int N, int *Ioff) {
  int I;
  #pragma omp simd simdlen(omp_max_simdlen)
  for (I = 0; I < N; I++)
    A[I] *= B[I] * C[I + *Ioff];
}

// CHECK: "llvm.loop.vectorize.scalable.enable", i1 true
