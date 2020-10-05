; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s | FileCheck %s
target triple = "thumbv6m-unknown-unknown-eabi"

define void @vla_emergency_spill(i32 %n) {
; CHECK-LABEL: vla_emergency_spill:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, r7, lr}
; CHECK-NEXT:    push {r4, r5, r6, r7, lr}
; CHECK-NEXT:    .setfp r7, sp, #12
; CHECK-NEXT:    add r7, sp, #12
; CHECK-NEXT:    ldr r6, .LCPI0_0
; CHECK-NEXT:    .pad #4100
; CHECK-NEXT:    add sp, r6
; CHECK-NEXT:    mov r6, sp
; CHECK-NEXT:    adds r0, r0, #7
; CHECK-NEXT:    movs r1, #7
; CHECK-NEXT:    bics r0, r1
; CHECK-NEXT:    mov r1, sp
; CHECK-NEXT:    subs r0, r1, r0
; CHECK-NEXT:    mov sp, r0
; CHECK-NEXT:    adds r1, r6, #4
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    str r0, [r6]
; CHECK-NEXT:    ldr r0, .LCPI0_1
; CHECK-NEXT:    str r5, [r0, r6]
; CHECK-NEXT:    ldr r0, [r6]
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    subs r4, r7, #7
; CHECK-NEXT:    subs r4, #5
; CHECK-NEXT:    mov sp, r4
; CHECK-NEXT:    pop {r4, r5, r6, r7, pc}
; CHECK-NEXT:    .p2align 2
; CHECK-NEXT:  @ %bb.1:
; CHECK-NEXT:  .LCPI0_0:
; CHECK-NEXT:    .long 4294963196 @ 0xffffeffc
; CHECK-NEXT:  .LCPI0_1:
; CHECK-NEXT:    .long 1024 @ 0x400
entry:
  %x = alloca [1024 x i32], align 4
  %vla = alloca i8, i32 %n, align 1
  %asm1 = call { i32, i32, i32, i32, i32, i32 } asm "", "={r0},={r1},={r2},={r3},={r4},={r5},0,1,2,3,4,5"(i8* %vla, [1024 x i32]* %x, i32 undef, i32 undef, i32 undef, i32 undef)
  %asmresult = extractvalue { i32, i32, i32, i32, i32, i32 } %asm1, 0
  %asmresult1 = extractvalue { i32, i32, i32, i32, i32, i32 } %asm1, 1
  %asmresult2 = extractvalue { i32, i32, i32, i32, i32, i32 } %asm1, 2
  %asmresult3 = extractvalue { i32, i32, i32, i32, i32, i32 } %asm1, 3
  %asmresult4 = extractvalue { i32, i32, i32, i32, i32, i32 } %asm1, 4
  %asmresult5 = extractvalue { i32, i32, i32, i32, i32, i32 } %asm1, 5
  %arrayidx = getelementptr inbounds [1024 x i32], [1024 x i32]* %x, i32 0, i32 255
  store i32 %asmresult5, i32* %arrayidx, align 4
  call void asm sideeffect "", "{r0},{r1},{r2},{r3},{r4},{r5}"(i32 %asmresult, i32 %asmresult1, i32 %asmresult2, i32 %asmresult3, i32 %asmresult4, i32 %asmresult5) #2
  ret void
}

define void @simple_emergency_spill(i32 %n) {
; CHECK-LABEL: simple_emergency_spill:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, r7, lr}
; CHECK-NEXT:    push {r4, r5, r6, r7, lr}
; CHECK-NEXT:    ldr r7, .LCPI1_0
; CHECK-NEXT:    .pad #8196
; CHECK-NEXT:    add sp, r7
; CHECK-NEXT:    add r0, sp, #4
; CHECK-NEXT:    ldr r1, .LCPI1_2
; CHECK-NEXT:    add r1, sp
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    str r0, [sp]
; CHECK-NEXT:    ldr r0, .LCPI1_3
; CHECK-NEXT:    add r0, sp
; CHECK-NEXT:    str r5, [r0]
; CHECK-NEXT:    ldr r0, [sp]
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    ldr r7, .LCPI1_1
; CHECK-NEXT:    add sp, r7
; CHECK-NEXT:    pop {r4, r5, r6, r7, pc}
; CHECK-NEXT:    .p2align 2
; CHECK-NEXT:  @ %bb.1:
; CHECK-NEXT:  .LCPI1_0:
; CHECK-NEXT:    .long 4294959100 @ 0xffffdffc
; CHECK-NEXT:  .LCPI1_1:
; CHECK-NEXT:    .long 8196 @ 0x2004
; CHECK-NEXT:  .LCPI1_2:
; CHECK-NEXT:    .long 4100 @ 0x1004
; CHECK-NEXT:  .LCPI1_3:
; CHECK-NEXT:    .long 5120 @ 0x1400
entry:
  %x = alloca [1024 x i32], align 4
  %y = alloca [1024 x i32], align 4
  %asm1 = call { i32, i32, i32, i32, i32, i32, i32, i32 } asm "", "={r0},={r1},={r2},={r3},={r4},={r5},={r6},={r7},0,1,2,3,4,5,6,7"([1024 x i32]* %y, [1024 x i32]* %x, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef)
  %asmresult = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 0
  %asmresult2 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 1
  %asmresult3 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 2
  %asmresult4 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 3
  %asmresult5 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 4
  %asmresult6 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 5
  %asmresult7 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 6
  %asmresult8 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 7
  %arrayidx = getelementptr inbounds [1024 x i32], [1024 x i32]* %x, i32 0, i32 255
  store i32 %asmresult6, i32* %arrayidx, align 4
  call void asm sideeffect "", "{r0},{r1},{r2},{r3},{r4},{r5},{r6},{r7}"(i32 %asmresult, i32 %asmresult2, i32 %asmresult3, i32 %asmresult4, i32 %asmresult5, i32 %asmresult6, i32 %asmresult7, i32 %asmresult8)
  ret void
}

; We have some logic to try to spill registers instead of allocating an
; emergency spill slot, but for targets where the stack alignment is 8,
; it only triggers when there are two available registers.  (This is
; maybe worth looking into, to improve the generated code quality.)
;
; The scavenger itself only cares whether a register is allocatable, not
; whether it was actually spilled in the prologue, and r7 is first on
; the priority list, so we use it anyway.  This is likely to confuse
; debuggers, so maybe worth changing at some point.
define void @simple_emergency_spill_nor7(i32 %n) {
; CHECK-LABEL: simple_emergency_spill_nor7:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, lr}
; CHECK-NEXT:    push {r4, r5, r6, lr}
; CHECK-NEXT:    ldr r6, .LCPI2_0
; CHECK-NEXT:    .pad #8196
; CHECK-NEXT:    add sp, r6
; CHECK-NEXT:    add r0, sp, #4
; CHECK-NEXT:    ldr r1, .LCPI2_2
; CHECK-NEXT:    add r1, sp
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    str r7, [sp]
; CHECK-NEXT:    ldr r7, .LCPI2_3
; CHECK-NEXT:    add r7, sp
; CHECK-NEXT:    str r5, [r7]
; CHECK-NEXT:    ldr r7, [sp]
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    ldr r6, .LCPI2_1
; CHECK-NEXT:    add sp, r6
; CHECK-NEXT:    pop {r4, r5, r6, pc}
; CHECK-NEXT:    .p2align 2
; CHECK-NEXT:  @ %bb.1:
; CHECK-NEXT:  .LCPI2_0:
; CHECK-NEXT:    .long 4294959100 @ 0xffffdffc
; CHECK-NEXT:  .LCPI2_1:
; CHECK-NEXT:    .long 8196 @ 0x2004
; CHECK-NEXT:  .LCPI2_2:
; CHECK-NEXT:    .long 4100 @ 0x1004
; CHECK-NEXT:  .LCPI2_3:
; CHECK-NEXT:    .long 5120 @ 0x1400
entry:
  %x = alloca [1024 x i32], align 4
  %y = alloca [1024 x i32], align 4
  %asm1 = call { i32, i32, i32, i32, i32, i32, i32 } asm "", "={r0},={r1},={r2},={r3},={r4},={r5},={r6},0,1,2,3,4,5,6"([1024 x i32]* %y, [1024 x i32]* %x, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef)
  %asmresult = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 0
  %asmresult2 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 1
  %asmresult3 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 2
  %asmresult4 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 3
  %asmresult5 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 4
  %asmresult6 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 5
  %asmresult7 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 6
  %arrayidx = getelementptr inbounds [1024 x i32], [1024 x i32]* %x, i32 0, i32 255
  store i32 %asmresult6, i32* %arrayidx, align 4
  call void asm sideeffect "", "{r0},{r1},{r2},{r3},{r4},{r5},{r6}"(i32 %asmresult, i32 %asmresult2, i32 %asmresult3, i32 %asmresult4, i32 %asmresult5, i32 %asmresult6, i32 %asmresult7)
  ret void
}

define void @arg_emergency_spill(i32 %n, i32 %n2, i32 %n3, i32 %n4, [252 x i32]* byval %p) {
; CHECK-LABEL: arg_emergency_spill:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, r7, lr}
; CHECK-NEXT:    push {r4, r5, r6, r7, lr}
; CHECK-NEXT:    .pad #4
; CHECK-NEXT:    sub sp, #4
; CHECK-NEXT:    add r0, sp, #24
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    str r0, [sp]
; CHECK-NEXT:    ldr r0, .LCPI3_0
; CHECK-NEXT:    add r0, sp
; CHECK-NEXT:    str r5, [r0]
; CHECK-NEXT:    ldr r0, [sp]
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    add sp, #4
; CHECK-NEXT:    pop {r4, r5, r6, r7, pc}
; CHECK-NEXT:    .p2align 2
; CHECK-NEXT:  @ %bb.1:
; CHECK-NEXT:  .LCPI3_0:
; CHECK-NEXT:    .long 1028 @ 0x404
entry:
  %pp = getelementptr inbounds [252 x i32], [252 x i32]* %p, i32 0, i32 0
  %asm1 = call { i32, i32, i32, i32, i32, i32, i32, i32 } asm "", "={r0},={r1},={r2},={r3},={r4},={r5},={r6},={r7},0,1,2,3,4,5,6,7"(i32* %pp, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef)
  %asmresult = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 0
  %asmresult2 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 1
  %asmresult3 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 2
  %asmresult4 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 3
  %asmresult5 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 4
  %asmresult6 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 5
  %asmresult7 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 6
  %asmresult8 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 7
  %arrayidx = getelementptr inbounds i32, i32* %pp, i32 251
  store i32 %asmresult6, i32* %arrayidx, align 4
  call void asm sideeffect "", "{r0},{r1},{r2},{r3},{r4},{r5},{r6},{r7}"(i32 %asmresult, i32 %asmresult2, i32 %asmresult3, i32 %asmresult4, i32 %asmresult5, i32 %asmresult6, i32 %asmresult7, i32 %asmresult8)
  ret void
}

; We currently overestimate the amount of required stack space by 16 bytes,
; so this is the largest stack that doesn't require an emergency spill slot.
define void @arg_no_emergency_spill(i32 %n, i32 %n2, i32 %n3, i32 %n4, [248 x i32]* byval %p) {
; CHECK-LABEL: arg_no_emergency_spill:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, r7, lr}
; CHECK-NEXT:    push {r4, r5, r6, r7, lr}
; CHECK-NEXT:    add r0, sp, #20
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    str r5, [sp, #1008]
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    pop {r4, r5, r6, r7, pc}
entry:
  %pp = getelementptr inbounds [248 x i32], [248 x i32]* %p, i32 0, i32 0
  %asm1 = call { i32, i32, i32, i32, i32, i32, i32, i32 } asm "", "={r0},={r1},={r2},={r3},={r4},={r5},={r6},={r7},0,1,2,3,4,5,6,7"(i32* %pp, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef)
  %asmresult = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 0
  %asmresult2 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 1
  %asmresult3 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 2
  %asmresult4 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 3
  %asmresult5 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 4
  %asmresult6 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 5
  %asmresult7 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 6
  %asmresult8 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32 } %asm1, 7
  %arrayidx = getelementptr inbounds i32, i32* %pp, i32 247
  store i32 %asmresult6, i32* %arrayidx, align 4
  call void asm sideeffect "", "{r0},{r1},{r2},{r3},{r4},{r5},{r6},{r7}"(i32 %asmresult, i32 %asmresult2, i32 %asmresult3, i32 %asmresult4, i32 %asmresult5, i32 %asmresult6, i32 %asmresult7, i32 %asmresult8)
  ret void
}

define void @aligned_emergency_spill(i32 %n, i32 %n2, i32 %n3, i32 %n4, [31 x i32]* byval %p) {
; CHECK-LABEL: aligned_emergency_spill:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, r7, lr}
; CHECK-NEXT:    push {r4, r5, r6, r7, lr}
; CHECK-NEXT:    .setfp r7, sp, #12
; CHECK-NEXT:    add r7, sp, #12
; CHECK-NEXT:    .pad #44
; CHECK-NEXT:    sub sp, #44
; CHECK-NEXT:    mov r4, sp
; CHECK-NEXT:    lsrs r4, r4, #4
; CHECK-NEXT:    lsls r4, r4, #4
; CHECK-NEXT:    mov sp, r4
; CHECK-NEXT:    add r0, sp, #16
; CHECK-NEXT:    adds r1, r7, #7
; CHECK-NEXT:    adds r1, #1
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    str r0, [sp]
; CHECK-NEXT:    ldr r0, .LCPI5_0
; CHECK-NEXT:    str r5, [r0, r7]
; CHECK-NEXT:    ldr r0, [sp]
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    subs r4, r7, #7
; CHECK-NEXT:    subs r4, #5
; CHECK-NEXT:    mov sp, r4
; CHECK-NEXT:    pop {r4, r5, r6, r7, pc}
; CHECK-NEXT:    .p2align 2
; CHECK-NEXT:  @ %bb.1:
; CHECK-NEXT:  .LCPI5_0:
; CHECK-NEXT:    .long 128 @ 0x80
entry:
  %y = alloca [4 x i32], align 16
  %pp = getelementptr inbounds [31 x i32], [31 x i32]* %p, i32 0, i32 0
  %asm1 = call { i32, i32, i32, i32, i32, i32, i32 } asm "", "={r0},={r1},={r2},={r3},={r4},={r5},={r6},0,1,2,3,4,5,6"([4 x i32]* %y, i32* %pp, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef) #3
  %asmresult = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 0
  %asmresult2 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 1
  %asmresult3 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 2
  %asmresult4 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 3
  %asmresult5 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 4
  %asmresult6 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 5
  %asmresult7 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 6
  %arrayidx = getelementptr inbounds i32, i32* %pp, i32 30
  store i32 %asmresult6, i32* %arrayidx, align 4
  call void asm sideeffect "", "{r0},{r1},{r2},{r3},{r4},{r5},{r6}"(i32 %asmresult, i32 %asmresult2, i32 %asmresult3, i32 %asmresult4, i32 %asmresult5, i32 %asmresult6, i32 %asmresult7)
  ret void
}

; This function should have no emergency spill slot, so its stack should be
; smaller than @aligned_emergency_spill.
define void @aligned_no_emergency_spill(i32 %n, i32 %n2, i32 %n3, i32 %n4, [30 x i32]* byval %p) {
; CHECK-LABEL: aligned_no_emergency_spill:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, r7, lr}
; CHECK-NEXT:    push {r4, r5, r6, r7, lr}
; CHECK-NEXT:    .setfp r7, sp, #12
; CHECK-NEXT:    add r7, sp, #12
; CHECK-NEXT:    .pad #28
; CHECK-NEXT:    sub sp, #28
; CHECK-NEXT:    mov r4, sp
; CHECK-NEXT:    lsrs r4, r4, #4
; CHECK-NEXT:    lsls r4, r4, #4
; CHECK-NEXT:    mov sp, r4
; CHECK-NEXT:    mov r0, sp
; CHECK-NEXT:    adds r1, r7, #7
; CHECK-NEXT:    adds r1, #1
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    str r5, [r7, #124]
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    subs r4, r7, #7
; CHECK-NEXT:    subs r4, #5
; CHECK-NEXT:    mov sp, r4
; CHECK-NEXT:    pop {r4, r5, r6, r7, pc}
entry:
  %y = alloca [4 x i32], align 16
  %pp = getelementptr inbounds [30 x i32], [30 x i32]* %p, i32 0, i32 0
  %asm1 = call { i32, i32, i32, i32, i32, i32, i32 } asm "", "={r0},={r1},={r2},={r3},={r4},={r5},={r6},0,1,2,3,4,5,6"([4 x i32]* %y, i32* %pp, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef) #3
  %asmresult = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 0
  %asmresult2 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 1
  %asmresult3 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 2
  %asmresult4 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 3
  %asmresult5 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 4
  %asmresult6 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 5
  %asmresult7 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 6
  %arrayidx = getelementptr inbounds i32, i32* %pp, i32 29
  store i32 %asmresult6, i32* %arrayidx, align 4
  call void asm sideeffect "", "{r0},{r1},{r2},{r3},{r4},{r5},{r6}"(i32 %asmresult, i32 %asmresult2, i32 %asmresult3, i32 %asmresult4, i32 %asmresult5, i32 %asmresult6, i32 %asmresult7)
  ret void
}

; This function shouldn't fail to compile.  (It's UB, so it doesn't really
; matter what it compiles to, exactly, but we need to check at some point
; so we don't generate code that requires an emergency spill slot we never
; allocated.  If the store gets eliminated, this testcase probably needs
; to be rewritten.)
define void @aligned_out_of_range_access(i32 %n, i32 %n2, i32 %n3, i32 %n4, [30 x i32]* byval %p) {
; CHECK-LABEL: aligned_out_of_range_access:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, r7, lr}
; CHECK-NEXT:    push {r4, r5, r6, r7, lr}
; CHECK-NEXT:    .setfp r7, sp, #12
; CHECK-NEXT:    add r7, sp, #12
; CHECK-NEXT:    .pad #44
; CHECK-NEXT:    sub sp, #44
; CHECK-NEXT:    mov r4, sp
; CHECK-NEXT:    lsrs r4, r4, #4
; CHECK-NEXT:    lsls r4, r4, #4
; CHECK-NEXT:    mov sp, r4
; CHECK-NEXT:    add r0, sp, #16
; CHECK-NEXT:    adds r1, r7, #7
; CHECK-NEXT:    adds r1, #1
; CHECK-NEXT:    str r1, [sp, #12] @ 4-byte Spill
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    str r0, [sp, #8] @ 4-byte Spill
; CHECK-NEXT:    ldr r0, [sp, #12] @ 4-byte Reload
; CHECK-NEXT:    str r5, [r0, #120]
; CHECK-NEXT:    ldr r0, [sp, #8] @ 4-byte Reload
; CHECK-NEXT:    @APP
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    subs r4, r7, #7
; CHECK-NEXT:    subs r4, #5
; CHECK-NEXT:    mov sp, r4
; CHECK-NEXT:    pop {r4, r5, r6, r7, pc}
entry:
  %y = alloca [4 x i32], align 16
  %pp = getelementptr inbounds [30 x i32], [30 x i32]* %p, i32 0, i32 0
  %asm1 = call { i32, i32, i32, i32, i32, i32, i32 } asm "", "={r0},={r1},={r2},={r3},={r4},={r5},={r6},0,1,2,3,4,5,6"([4 x i32]* %y, i32* %pp, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef) #3
  %asmresult = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 0
  %asmresult2 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 1
  %asmresult3 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 2
  %asmresult4 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 3
  %asmresult5 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 4
  %asmresult6 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 5
  %asmresult7 = extractvalue { i32, i32, i32, i32, i32, i32, i32 } %asm1, 6
  %arrayidx = getelementptr inbounds i32, i32* %pp, i32 30
  store i32 %asmresult6, i32* %arrayidx, align 4
  call void asm sideeffect "", "{r0},{r1},{r2},{r3},{r4},{r5},{r6}"(i32 %asmresult, i32 %asmresult2, i32 %asmresult3, i32 %asmresult4, i32 %asmresult5, i32 %asmresult6, i32 %asmresult7)
  ret void
}
