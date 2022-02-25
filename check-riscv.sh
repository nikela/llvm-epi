#!/bin/bash -e

ninja check-llvm-codegen-riscv
ninja check-llvm-debuginfo-riscv
ninja check-llvm-mc-riscv
ninja check-llvm-object-riscv
ninja check-llvm-transforms-simplifycfg-riscv
ninja check-llvm-analysis-costmodel-riscv
ninja check-llvm-transforms-loopvectorize

# Clang tests relevant for RISC-V are a bit scattered.
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
CLANG_TESTS=$(find ${SCRIPT_DIR}/clang/test -type f "-(" -name "*.cpp" -o -name "*.c" "-)" \
     -a "-(" -name "*epi*"  -o -name "*riscv*" -o -name "*rvv*" -o "-path" "*RISCV*" "-)" \
     -a -not "-(" -path "*Input*" "-)")
./bin/llvm-lit -sv ${CLANG_TESTS}
