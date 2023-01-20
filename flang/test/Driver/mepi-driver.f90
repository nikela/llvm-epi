! This expects RISC-V be enabled as it should be.
! RUN: %flang --target=riscv64-unknown-linux-gnu -mepi %s -O2 -o - -### 2>&1 | FileCheck %s

! CHECK: "-fc1"
! CHECK-SAME: "-triple" "riscv64-unknown-linux-gnu"
! CHECK-SAME: "-emit-obj"
! CHECK-SAME: "-O2"
! CHECK-SAME: "-vectorize-loops"
! CHECK-SAME: "-vectorize-slp"
! CHECK-SAME: "-mframe-pointer=none"
! CHECK-SAME: "-mrelocation-model" "pic" "-pic-level" "2" "-pic-is-pie"
! CHECK-SAME: "-target-cpu" "generic-rv64"
! CHECK-SAME: "-target-feature" "+m"
! CHECK-SAME: "-target-feature" "+a"
! CHECK-SAME: "-target-feature" "+f"
! CHECK-SAME: "-target-feature" "+d"
! CHECK-SAME: "-target-feature" "+c"
! CHECK-SAME: "-target-feature" "+zepi"
! CHECK-SAME: "-target-feature" "+zve32f"
! CHECK-SAME: "-target-feature" "+zve32x"
! CHECK-SAME: "-target-feature" "+zve64d"
! CHECK-SAME: "-target-feature" "+zve64f"
! CHECK-SAME: "-target-feature" "+zve64x"
! CHECK-SAME: "-target-feature" "+zvl32b"
! CHECK-SAME: "-target-feature" "+zvl64b"
! CHECK-SAME: "-mepi"
! CHECK-SAME: "-mllvm" "--scalable-vectorization=only"
! CHECK-SAME: "-mllvm" "--prefer-predicate-over-epilogue=predicate-dont-vectorize"
! CHECK-SAME: "-mllvm" "-epi-pipeline"
! CHECK-SAME: "-mllvm" "-riscv-v-vector-bits-min=64"
! CHECK-SAME: "-mllvm" "-scev-cheap-expansion-budget=8"
! CHECK-SAME: "-O2"

PROGRAM MAIN
  CONTINUE
END PROGRAM MAIN
