# NOTE: Assertions have been autogenerated by utils/update_mca_test_checks.py
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=bdver2 -timeline -timeline-max-iterations=3 < %s | FileCheck %s

leaq 8(%rsp, %rdi, 2), %rax
vbroadcastss (%rax), %ymm0

# CHECK:      Iterations:        100
# CHECK-NEXT: Instructions:      200
# CHECK-NEXT: Total Cycles:      206
# CHECK-NEXT: Total uOps:        400

# CHECK:      Dispatch Width:    4
# CHECK-NEXT: uOps Per Cycle:    1.94
# CHECK-NEXT: IPC:               0.97
# CHECK-NEXT: Block RThroughput: 2.0

# CHECK:      Instruction Info:
# CHECK-NEXT: [1]: #uOps
# CHECK-NEXT: [2]: Latency
# CHECK-NEXT: [3]: RThroughput
# CHECK-NEXT: [4]: MayLoad
# CHECK-NEXT: [5]: MayStore
# CHECK-NEXT: [6]: HasSideEffects (U)

# CHECK:      [1]    [2]    [3]    [4]    [5]    [6]    Instructions:
# CHECK-NEXT:  2      1     0.50                        leaq	8(%rsp,%rdi,2), %rax
# CHECK-NEXT:  2      6     2.00    *                   vbroadcastss	(%rax), %ymm0

# CHECK:      Resources:
# CHECK-NEXT: [0.0] - PdAGLU01
# CHECK-NEXT: [0.1] - PdAGLU01
# CHECK-NEXT: [1]   - PdBranch
# CHECK-NEXT: [2]   - PdCount
# CHECK-NEXT: [3]   - PdDiv
# CHECK-NEXT: [4]   - PdEX0
# CHECK-NEXT: [5]   - PdEX1
# CHECK-NEXT: [6]   - PdFPCVT
# CHECK-NEXT: [7.0] - PdFPFMA
# CHECK-NEXT: [7.1] - PdFPFMA
# CHECK-NEXT: [8.0] - PdFPMAL
# CHECK-NEXT: [8.1] - PdFPMAL
# CHECK-NEXT: [9]   - PdFPMMA
# CHECK-NEXT: [10]  - PdFPSTO
# CHECK-NEXT: [11]  - PdFPU0
# CHECK-NEXT: [12]  - PdFPU1
# CHECK-NEXT: [13]  - PdFPU2
# CHECK-NEXT: [14]  - PdFPU3
# CHECK-NEXT: [15]  - PdFPXBR
# CHECK-NEXT: [16.0] - PdLoad
# CHECK-NEXT: [16.1] - PdLoad
# CHECK-NEXT: [17]  - PdMul
# CHECK-NEXT: [18]  - PdStore

# CHECK:      Resource pressure per iteration:
# CHECK-NEXT: [0.0]  [0.1]  [1]    [2]    [3]    [4]    [5]    [6]    [7.0]  [7.1]  [8.0]  [8.1]  [9]    [10]   [11]   [12]   [13]   [14]   [15]   [16.0] [16.1] [17]   [18]
# CHECK-NEXT:  -     1.00    -      -      -     0.50   0.50    -     2.00   2.00    -      -      -      -     1.00   1.00    -      -      -      -     1.00    -      -

# CHECK:      Resource pressure by instruction:
# CHECK-NEXT: [0.0]  [0.1]  [1]    [2]    [3]    [4]    [5]    [6]    [7.0]  [7.1]  [8.0]  [8.1]  [9]    [10]   [11]   [12]   [13]   [14]   [15]   [16.0] [16.1] [17]   [18]   Instructions:
# CHECK-NEXT:  -      -      -      -      -     0.50   0.50    -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -     leaq	8(%rsp,%rdi,2), %rax
# CHECK-NEXT:  -     1.00    -      -      -      -      -      -     2.00   2.00    -      -      -      -     1.00   1.00    -      -      -      -     1.00    -      -     vbroadcastss	(%rax), %ymm0

# CHECK:      Timeline view:
# CHECK-NEXT:                     012
# CHECK-NEXT: Index     0123456789

# CHECK:      [0,0]     DeER .    . .   leaq	8(%rsp,%rdi,2), %rax
# CHECK-NEXT: [0,1]     DeeeeeeER . .   vbroadcastss	(%rax), %ymm0
# CHECK-NEXT: [1,0]     .DeE----R . .   leaq	8(%rsp,%rdi,2), %rax
# CHECK-NEXT: [1,1]     .DeeeeeeER. .   vbroadcastss	(%rax), %ymm0
# CHECK-NEXT: [2,0]     . DeE----R. .   leaq	8(%rsp,%rdi,2), %rax
# CHECK-NEXT: [2,1]     . D==eeeeeeER   vbroadcastss	(%rax), %ymm0

# CHECK:      Average Wait times (based on the timeline view):
# CHECK-NEXT: [0]: Executions
# CHECK-NEXT: [1]: Average time spent waiting in a scheduler's queue
# CHECK-NEXT: [2]: Average time spent waiting in a scheduler's queue while ready
# CHECK-NEXT: [3]: Average time elapsed from WB until retire stage

# CHECK:            [0]    [1]    [2]    [3]
# CHECK-NEXT: 0.     3     1.0    1.0    2.7       leaq	8(%rsp,%rdi,2), %rax
# CHECK-NEXT: 1.     3     1.7    0.7    0.0       vbroadcastss	(%rax), %ymm0
