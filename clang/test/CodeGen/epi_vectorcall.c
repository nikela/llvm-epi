// RUN: %clang_cc1 -triple riscv64 -mepi -emit-llvm -disable-llvm-passes -o - %s \
// RUN:       | FileCheck %s
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -verify %s

void __attribute__((epi_vectorcall)) f(int *); // expected-warning {{'epi_vectorcall' calling convention is not supported for this target}}

// CHECK-LABEL: @p(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[N_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[A_ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:    store i32 [[N:%.*]], i32* [[N_ADDR]], align 4
// CHECK-NEXT:    store i32* [[A:%.*]], i32** [[A_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32*, i32** [[A_ADDR]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* [[N_ADDR]], align 4
// CHECK-NEXT:    [[IDXPROM:%.*]] = sext i32 [[TMP1]] to i64
// CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, i32* [[TMP0]], i64 [[IDXPROM]]
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, i32* [[ARRAYIDX]], align 4
// CHECK-NEXT:    ret i32 [[TMP2]]
int __attribute__((epi_vectorcall)) p(int N, int *a) { // expected-warning {{'epi_vectorcall' calling convention is not supported for this target}}
  return a[N];
}

// CHECK-LABEL: @g(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[N_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[A_ADDR:%.*]] = alloca i32*, align 8
// CHECK-NEXT:    store i32 [[N:%.*]], i32* [[N_ADDR]], align 4
// CHECK-NEXT:    store i32* [[A:%.*]], i32** [[A_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32*, i32** [[A_ADDR]], align 8
// CHECK-NEXT:    call epi_vectorcall void @f(i32* noundef [[TMP0]])
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* [[N_ADDR]], align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load i32*, i32** [[A_ADDR]], align 8
// CHECK-NEXT:    [[CALL:%.*]] = call epi_vectorcall signext i32 @p(i32 noundef signext [[TMP1]], i32* noundef [[TMP2]])
// CHECK-NEXT:    ret i32 [[CALL]]
int g(int N, int *a) {
  f(a);
  return p(N, a);
}

// CHECK: declare epi_vectorcall {{[^@]+}} @f
