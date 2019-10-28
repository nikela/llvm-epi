# RUN: llvm-mc --assemble --triple riscv64 -mattr m,+a,+f,+d,+c,+epi \
# RUN:         --show-encoding < %s | FileCheck %s

vmerge.vvm  v1, v2, v3, v0
# CHECK: [0xd7,0x80,0x21,0x5c]

vmerge.vxm  v1, v2, gp, v0
# CHECK: [0xd7,0xc0,0x21,0x5c]

vmerge.vim  v1, v2, 1, v0
# CHECK: [0xd7,0xb0,0x20,0x5c]

vfmerge.vfm v1, v2, ft3, v0
# CHECK: [0xd7,0xd0,0x21,0x5c]

vadc.vvm  v1, v2, v3, v0
# CHECK: [0xd7,0x80,0x21,0x42]

vadc.vxm  v1, v2, gp, v0
# CHECK: [0xd7,0xc0,0x21,0x42]

vadc.vim  v1, v2, 1, v0
# CHECK: [0xd7,0xb0,0x20,0x42]
