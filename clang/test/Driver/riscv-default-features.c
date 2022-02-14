// RUN: %clang -target riscv32-unknown-elf -S -emit-llvm %s -o - | FileCheck %s -check-prefix=RV32
// RUN: %clang -target riscv64-unknown-elf -S -emit-llvm %s -o - | FileCheck %s -check-prefix=RV64

// RV32: "target-features"="+a,+c,+m"
// RV64: "target-features"="+64bit,+a,+c,+m"

// Dummy function
int foo(){
  return  3;
}
