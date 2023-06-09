// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py
// RUN: %clang_cc1 -triple riscv64 -mepi -emit-llvm -o- %s \
// RUN:       | FileCheck %s

// CHECK-LABEL: @foo(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[A:%.*]] = alloca <vscale x 16 x i1>, align 1
// CHECK-NEXT:    [[B:%.*]] = alloca <vscale x 32 x i1>, align 1
// CHECK-NEXT:    [[TMP0:%.*]] = load <vscale x 16 x i1>, ptr [[A]], align 1
// CHECK-NEXT:    [[TMP1:%.*]] = load <vscale x 32 x i1>, ptr [[B]], align 1
// CHECK-NEXT:    ret void
//
void foo(void)
{
  __epi_16xi1 a;
  (void)a;
  __epi_32xi1 b;
  (void)b;
}
