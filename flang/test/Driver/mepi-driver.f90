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
! CHECK-SAME: "-target-feature" "+64bit" "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d" "-target-feature" "+c" "-target-feature" "-relax" "-target-feature" "-save-restore" "-target-feature" "+zepi" "-mepi" "-mllvm" "--scalable-vectorization=only" "-mllvm" "--prefer-predicate-over-epilogue=predicate-dont-vectorize" "-mllvm" "-epi-pipeline" "-mllvm" "-riscv-v-vector-bits-min=64" "-mllvm" "-scev-cheap-expansion-budget=8"

PROGRAM MAIN
  CONTINUE
END PROGRAM MAIN
