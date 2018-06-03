# NOTE: Assertions have been autogenerated by utils/update_mca_test_checks.py
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=btver2 -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=BTVER2 %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=znver1 -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=ZNVER1 %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=sandybridge -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=SANDYBRIDGE %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=ivybridge -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=IVYBRIDGE %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=haswell -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=HASWELL %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=broadwell -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=BROADWELL %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=knl -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=KNL %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=skylake -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=SKX %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=skylake-avx512 -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=SKX-AVX512 %s
# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=slm -resource-pressure=false -instruction-info=false < %s | FileCheck --check-prefix=ALL --check-prefix=SLM %s

add %edi, %eax

# ALL:              Iterations:        100
# ALL-NEXT:         Instructions:      100
# ALL-NEXT:         Total Cycles:      103

# BROADWELL-NEXT:   Dispatch Width:    4
# BTVER2-NEXT:      Dispatch Width:    2
# HASWELL-NEXT:     Dispatch Width:    4
# IVYBRIDGE-NEXT:   Dispatch Width:    4
# KNL-NEXT:         Dispatch Width:    4
# SANDYBRIDGE-NEXT: Dispatch Width:    4
# SKX-NEXT:         Dispatch Width:    6
# SKX-AVX512-NEXT:  Dispatch Width:    6
# SLM-NEXT:         Dispatch Width:    2
# ZNVER1-NEXT:      Dispatch Width:    4

# ALL-NEXT:         IPC:               0.97

# BROADWELL-NEXT:   Block RThroughput: 0.3
# BTVER2-NEXT:      Block RThroughput: 0.5
# HASWELL-NEXT:     Block RThroughput: 0.3
# IVYBRIDGE-NEXT:   Block RThroughput: 0.3
# KNL-NEXT:         Block RThroughput: 0.3
# SANDYBRIDGE-NEXT: Block RThroughput: 0.3
# SKX-NEXT:         Block RThroughput: 0.3
# SKX-AVX512-NEXT:  Block RThroughput: 0.3
# SLM-NEXT:         Block RThroughput: 0.5
# ZNVER1-NEXT:      Block RThroughput: 0.3

