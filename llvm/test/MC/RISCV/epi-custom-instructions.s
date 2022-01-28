// RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+v < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+v < %s \
// RUN:        | llvm-objdump -d --mattr=+v - \
// RUN:        | FileCheck %s --check-prefix=CHECK-INST

# CHECK-INST: vzip2.vv v1, v2, v3
# CHECK-ENCODING:  [0x8b,0x80,0x21,0xd6]
vzip2.vv v1, v2, v3

# CHECK-INST: vunzip2.vv v1, v2, v3
# CHECK-ENCODING: [0x8b,0x80,0x21,0xda]
vunzip2.vv v1, v2, v3

# CHECK-INST: vtrn.vv v1, v2, v3
# CHECK-ENCODING: [0x8b,0x80,0x21,0xde]
vtrn.vv v1, v2, v3


