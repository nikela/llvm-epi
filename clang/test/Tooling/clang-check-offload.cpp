// This test not seem to work in a cross-compiler setting.
// XFAIL: riscv
// RUN: not clang-check "%s" -- -c -x hip -nogpulib 2>&1 | FileCheck %s

// CHECK: C++ requires
invalid;
