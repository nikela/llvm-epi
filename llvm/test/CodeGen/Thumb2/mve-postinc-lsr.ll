; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=thumbv8.1m.main-arm-none-eabi -mattr=+mve.fp %s -o - | FileCheck %s

; Check some LSR loop postinc

; fma loop with a destination that is the same as one of the sources
define void @fma(float* noalias nocapture readonly %A, float* noalias nocapture readonly %B, float* noalias nocapture %C, i32 %n) {
; CHECK-LABEL: fma:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, lr}
; CHECK-NEXT:    push {r4, r5, r6, lr}
; CHECK-NEXT:    cmp r3, #1
; CHECK-NEXT:    blt .LBB0_8
; CHECK-NEXT:  @ %bb.1: @ %for.body.preheader
; CHECK-NEXT:    cmp r3, #3
; CHECK-NEXT:    bhi .LBB0_3
; CHECK-NEXT:  @ %bb.2:
; CHECK-NEXT:    mov.w r12, #0
; CHECK-NEXT:    b .LBB0_6
; CHECK-NEXT:  .LBB0_3: @ %vector.ph
; CHECK-NEXT:    bic r12, r3, #3
; CHECK-NEXT:    movs r5, #1
; CHECK-NEXT:    sub.w r6, r12, #4
; CHECK-NEXT:    mov r4, r0
; CHECK-NEXT:    add.w lr, r5, r6, lsr #2
; CHECK-NEXT:    mov r5, r1
; CHECK-NEXT:    mov r6, r2
; CHECK-NEXT:    dls lr, lr
; CHECK-NEXT:  .LBB0_4: @ %vector.body
; CHECK-NEXT:    @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    vldrw.u32 q0, [r4], #16
; CHECK-NEXT:    vldrw.u32 q1, [r5], #16
; CHECK-NEXT:    vldrw.u32 q2, [r6]
; CHECK-NEXT:    vfma.f32 q2, q1, q0
; CHECK-NEXT:    vstrb.8 q2, [r6], #16
; CHECK-NEXT:    le lr, .LBB0_4
; CHECK-NEXT:  @ %bb.5: @ %middle.block
; CHECK-NEXT:    cmp r12, r3
; CHECK-NEXT:    it eq
; CHECK-NEXT:    popeq {r4, r5, r6, pc}
; CHECK-NEXT:  .LBB0_6: @ %for.body.preheader12
; CHECK-NEXT:    sub.w lr, r3, r12
; CHECK-NEXT:    add.w r0, r0, r12, lsl #2
; CHECK-NEXT:    add.w r1, r1, r12, lsl #2
; CHECK-NEXT:    add.w r2, r2, r12, lsl #2
; CHECK-NEXT:    dls lr, lr
; CHECK-NEXT:  .LBB0_7: @ %for.body
; CHECK-NEXT:    @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    vldr s0, [r0]
; CHECK-NEXT:    adds r0, #4
; CHECK-NEXT:    vldr s2, [r1]
; CHECK-NEXT:    adds r1, #4
; CHECK-NEXT:    vldr s4, [r2]
; CHECK-NEXT:    vfma.f32 s4, s2, s0
; CHECK-NEXT:    vstr s4, [r2]
; CHECK-NEXT:    adds r2, #4
; CHECK-NEXT:    le lr, .LBB0_7
; CHECK-NEXT:  .LBB0_8: @ %for.cond.cleanup
; CHECK-NEXT:    pop {r4, r5, r6, pc}
entry:
  %cmp8 = icmp sgt i32 %n, 0
  br i1 %cmp8, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %min.iters.check = icmp ult i32 %n, 4
  br i1 %min.iters.check, label %for.body.preheader12, label %vector.ph

for.body.preheader12:                             ; preds = %middle.block, %for.body.preheader
  %i.09.ph = phi i32 [ 0, %for.body.preheader ], [ %n.vec, %middle.block ]
  br label %for.body

vector.ph:                                        ; preds = %for.body.preheader
  %n.vec = and i32 %n, -4
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds float, float* %A, i32 %index
  %1 = bitcast float* %0 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %1, align 4
  %2 = getelementptr inbounds float, float* %B, i32 %index
  %3 = bitcast float* %2 to <4 x float>*
  %wide.load10 = load <4 x float>, <4 x float>* %3, align 4
  %4 = fmul fast <4 x float> %wide.load10, %wide.load
  %5 = getelementptr inbounds float, float* %C, i32 %index
  %6 = bitcast float* %5 to <4 x float>*
  %wide.load11 = load <4 x float>, <4 x float>* %6, align 4
  %7 = fadd fast <4 x float> %wide.load11, %4
  %8 = bitcast float* %5 to <4 x float>*
  store <4 x float> %7, <4 x float>* %8, align 4
  %index.next = add i32 %index, 4
  %9 = icmp eq i32 %index.next, %n.vec
  br i1 %9, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i32 %n.vec, %n
  br i1 %cmp.n, label %for.cond.cleanup, label %for.body.preheader12

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader12, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ %i.09.ph, %for.body.preheader12 ]
  %arrayidx = getelementptr inbounds float, float* %A, i32 %i.09
  %10 = load float, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, float* %B, i32 %i.09
  %11 = load float, float* %arrayidx1, align 4
  %mul = fmul fast float %11, %10
  %arrayidx2 = getelementptr inbounds float, float* %C, i32 %i.09
  %12 = load float, float* %arrayidx2, align 4
  %add = fadd fast float %12, %mul
  store float %add, float* %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}


; Same as above but tail predicated
; FIXME: The postinc here is put on the load, not the store. An extra mov is needed in the loop because of it.
define void @fma_tailpred(float* noalias nocapture readonly %A, float* noalias nocapture readonly %B, float* noalias nocapture %C, i32 %n) {
; CHECK-LABEL: fma_tailpred:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, lr}
; CHECK-NEXT:    push {r4, lr}
; CHECK-NEXT:    .vsave {d8, d9}
; CHECK-NEXT:    vpush {d8, d9}
; CHECK-NEXT:    cmp r3, #1
; CHECK-NEXT:    blt .LBB1_3
; CHECK-NEXT:  @ %bb.1: @ %vector.ph
; CHECK-NEXT:    add.w r12, r3, #3
; CHECK-NEXT:    adr r4, .LCPI1_0
; CHECK-NEXT:    bic r12, r12, #3
; CHECK-NEXT:    mov.w lr, #1
; CHECK-NEXT:    sub.w r12, r12, #4
; CHECK-NEXT:    subs r3, #1
; CHECK-NEXT:    vldrw.u32 q0, [r4]
; CHECK-NEXT:    vdup.32 q1, r3
; CHECK-NEXT:    add.w lr, lr, r12, lsr #2
; CHECK-NEXT:    mov.w r12, #0
; CHECK-NEXT:    mov r3, r2
; CHECK-NEXT:    dls lr, lr
; CHECK-NEXT:  .LBB1_2: @ %vector.body
; CHECK-NEXT:    @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    vdup.32 q2, r12
; CHECK-NEXT:    add.w r12, r12, #4
; CHECK-NEXT:    vorr q2, q2, q0
; CHECK-NEXT:    vpttt.u32 cs, q1, q2
; CHECK-NEXT:    vldrwt.u32 q2, [r0], #16
; CHECK-NEXT:    vldrwt.u32 q3, [r1], #16
; CHECK-NEXT:    vldrwt.u32 q4, [r3], #16
; CHECK-NEXT:    vfma.f32 q4, q3, q2
; CHECK-NEXT:    vpst
; CHECK-NEXT:    vstrwt.32 q4, [r2]
; CHECK-NEXT:    mov r2, r3
; CHECK-NEXT:    le lr, .LBB1_2
; CHECK-NEXT:  .LBB1_3: @ %for.cond.cleanup
; CHECK-NEXT:    vpop {d8, d9}
; CHECK-NEXT:    pop {r4, pc}
; CHECK-NEXT:    .p2align 4
; CHECK-NEXT:  @ %bb.4:
; CHECK-NEXT:  .LCPI1_0:
; CHECK-NEXT:    .long 0 @ 0x0
; CHECK-NEXT:    .long 1 @ 0x1
; CHECK-NEXT:    .long 2 @ 0x2
; CHECK-NEXT:    .long 3 @ 0x3
entry:
  %cmp8 = icmp sgt i32 %n, 0
  br i1 %cmp8, label %vector.ph, label %for.cond.cleanup

vector.ph:                                        ; preds = %entry
  %n.rnd.up = add i32 %n, 3
  %n.vec = and i32 %n.rnd.up, -4
  %trip.count.minus.1 = add i32 %n, -1
  %broadcast.splatinsert10 = insertelement <4 x i32> undef, i32 %trip.count.minus.1, i32 0
  %broadcast.splat11 = shufflevector <4 x i32> %broadcast.splatinsert10, <4 x i32> undef, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = or <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %0 = getelementptr inbounds float, float* %A, i32 %index
  %1 = icmp ule <4 x i32> %induction, %broadcast.splat11
  %2 = bitcast float* %0 to <4 x float>*
  %wide.masked.load = call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %2, i32 4, <4 x i1> %1, <4 x float> undef)
  %3 = getelementptr inbounds float, float* %B, i32 %index
  %4 = bitcast float* %3 to <4 x float>*
  %wide.masked.load12 = call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %4, i32 4, <4 x i1> %1, <4 x float> undef)
  %5 = fmul fast <4 x float> %wide.masked.load12, %wide.masked.load
  %6 = getelementptr inbounds float, float* %C, i32 %index
  %7 = bitcast float* %6 to <4 x float>*
  %wide.masked.load13 = call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %7, i32 4, <4 x i1> %1, <4 x float> undef)
  %8 = fadd fast <4 x float> %wide.masked.load13, %5
  %9 = bitcast float* %6 to <4 x float>*
  call void @llvm.masked.store.v4f32.p0v4f32(<4 x float> %8, <4 x float>* %9, i32 4, <4 x i1> %1)
  %index.next = add i32 %index, 4
  %10 = icmp eq i32 %index.next, %n.vec
  br i1 %10, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}


; Multiple loads of the loop with a common base
define i8* @test(i8* nocapture readonly %input_row, i8* nocapture readonly %input_col, i16 zeroext %output_ch, i16 zeroext %num_cols, i32 %col_offset, i16 signext %activation_min, i16 zeroext %row_len, i32* nocapture readonly %bias, i8* returned %out) {
; CHECK-LABEL: test:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-NEXT:    .pad #20
; CHECK-NEXT:    sub sp, #20
; CHECK-NEXT:    cmp r3, #4
; CHECK-NEXT:    strd r0, r1, [sp, #12] @ 8-byte Folded Spill
; CHECK-NEXT:    bne .LBB2_8
; CHECK-NEXT:  @ %bb.1: @ %for.cond.preheader
; CHECK-NEXT:    cmp r2, #0
; CHECK-NEXT:    beq .LBB2_8
; CHECK-NEXT:  @ %bb.2: @ %for.body.lr.ph
; CHECK-NEXT:    ldr r3, [sp, #64]
; CHECK-NEXT:    mov.w r11, #0
; CHECK-NEXT:    ldr r1, [sp, #16] @ 4-byte Reload
; CHECK-NEXT:    ldr.w r9, [sp, #56]
; CHECK-NEXT:    add.w r0, r1, r3, lsl #1
; CHECK-NEXT:    str r0, [sp, #8] @ 4-byte Spill
; CHECK-NEXT:    adds r0, r1, r3
; CHECK-NEXT:    str r0, [sp, #4] @ 4-byte Spill
; CHECK-NEXT:    add.w r0, r3, r3, lsl #1
; CHECK-NEXT:    add r0, r1
; CHECK-NEXT:    str r0, [sp] @ 4-byte Spill
; CHECK-NEXT:    adds r0, r3, #7
; CHECK-NEXT:    lsrs r0, r0, #3
; CHECK-NEXT:    b .LBB2_5
; CHECK-NEXT:  .LBB2_3: @ in Loop: Header=BB2_5 Depth=1
; CHECK-NEXT:    mov r8, r12
; CHECK-NEXT:    mov r10, r12
; CHECK-NEXT:    mov r6, r12
; CHECK-NEXT:  .LBB2_4: @ %for.cond.cleanup23
; CHECK-NEXT:    @ in Loop: Header=BB2_5 Depth=1
; CHECK-NEXT:    ldr r3, [sp, #72]
; CHECK-NEXT:    add.w r1, r10, r8
; CHECK-NEXT:    add r1, r6
; CHECK-NEXT:    add r1, r12
; CHECK-NEXT:    strb.w r1, [r3, r11]
; CHECK-NEXT:    add.w r11, r11, #1
; CHECK-NEXT:    cmp r11, r2
; CHECK-NEXT:    beq .LBB2_8
; CHECK-NEXT:  .LBB2_5: @ %for.body
; CHECK-NEXT:    @ =>This Loop Header: Depth=1
; CHECK-NEXT:    @ Child Loop BB2_7 Depth 2
; CHECK-NEXT:    ldr r1, [sp, #68]
; CHECK-NEXT:    subs.w lr, r0, r0
; CHECK-NEXT:    ldr.w r12, [r1, r11, lsl #2]
; CHECK-NEXT:    ble .LBB2_3
; CHECK-NEXT:  @ %bb.6: @ %for.body24.preheader
; CHECK-NEXT:    @ in Loop: Header=BB2_5 Depth=1
; CHECK-NEXT:    ldr r3, [sp, #64]
; CHECK-NEXT:    mov r6, r12
; CHECK-NEXT:    ldr r1, [sp, #12] @ 4-byte Reload
; CHECK-NEXT:    dls lr, lr
; CHECK-NEXT:    ldr r5, [sp, #8] @ 4-byte Reload
; CHECK-NEXT:    mov r10, r12
; CHECK-NEXT:    mla r7, r11, r3, r1
; CHECK-NEXT:    ldr r1, [sp, #16] @ 4-byte Reload
; CHECK-NEXT:    ldrd r4, r3, [sp] @ 8-byte Folded Reload
; CHECK-NEXT:    mov r8, r12
; CHECK-NEXT:  .LBB2_7: @ %for.body24
; CHECK-NEXT:    @ Parent Loop BB2_5 Depth=1
; CHECK-NEXT:    @ => This Inner Loop Header: Depth=2
; CHECK-NEXT:    vldrb.s16 q0, [r4], #8
; CHECK-NEXT:    vadd.i16 q1, q0, r9
; CHECK-NEXT:    vldrb.s16 q0, [r7], #8
; CHECK-NEXT:    vmlava.s16 r12, q0, q1
; CHECK-NEXT:    vldrb.s16 q1, [r5], #8
; CHECK-NEXT:    vadd.i16 q1, q1, r9
; CHECK-NEXT:    vmlava.s16 r6, q0, q1
; CHECK-NEXT:    vldrb.s16 q1, [r3], #8
; CHECK-NEXT:    vadd.i16 q1, q1, r9
; CHECK-NEXT:    vmlava.s16 r10, q0, q1
; CHECK-NEXT:    vldrb.s16 q1, [r1], #8
; CHECK-NEXT:    vadd.i16 q1, q1, r9
; CHECK-NEXT:    vmlava.s16 r8, q0, q1
; CHECK-NEXT:    le lr, .LBB2_7
; CHECK-NEXT:    b .LBB2_4
; CHECK-NEXT:  .LBB2_8: @ %if.end
; CHECK-NEXT:    ldr r0, [sp, #72]
; CHECK-NEXT:    add sp, #20
; CHECK-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11, pc}
entry:
  %cmp = icmp eq i16 %num_cols, 4
  br i1 %cmp, label %for.cond.preheader, label %if.end

for.cond.preheader:                               ; preds = %entry
  %conv2 = zext i16 %output_ch to i32
  %cmp3114 = icmp eq i16 %output_ch, 0
  br i1 %cmp3114, label %if.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %for.cond.preheader
  %conv5 = zext i16 %row_len to i32
  %add.ptr9 = getelementptr inbounds i8, i8* %input_col, i32 %conv5
  %mul11 = shl nuw nsw i32 %conv5, 1
  %add.ptr12 = getelementptr inbounds i8, i8* %input_col, i32 %mul11
  %mul14 = mul nuw nsw i32 %conv5, 3
  %add.ptr15 = getelementptr inbounds i8, i8* %input_col, i32 %mul14
  %add = add nuw nsw i32 %conv5, 7
  %div = lshr i32 %add, 3
  %conv25 = trunc i32 %col_offset to i16
  %.splatinsert = insertelement <8 x i16> undef, i16 %conv25, i32 0
  %.splat = shufflevector <8 x i16> %.splatinsert, <8 x i16> undef, <8 x i32> zeroinitializer
  br label %for.body

for.body:                                         ; preds = %for.cond.cleanup23, %for.body.lr.ph
  %i_out_ch.0116 = phi i32 [ 0, %for.body.lr.ph ], [ %inc37, %for.cond.cleanup23 ]
  %i_row_loop.0115 = phi i32 [ undef, %for.body.lr.ph ], [ %i_row_loop.1.lcssa, %for.cond.cleanup23 ]
  %arrayidx = getelementptr inbounds i32, i32* %bias, i32 %i_out_ch.0116
  %0 = load i32, i32* %arrayidx, align 4
  %cmp2199 = icmp slt i32 %i_row_loop.0115, %div
  br i1 %cmp2199, label %for.body24.preheader, label %for.cond.cleanup23

for.body24.preheader:                             ; preds = %for.body
  %mul = mul nuw nsw i32 %i_out_ch.0116, %conv5
  %add.ptr = getelementptr inbounds i8, i8* %input_row, i32 %mul
  br label %for.body24

for.cond.cleanup23:                               ; preds = %for.body24, %for.body
  %acc_0.0.lcssa = phi i32 [ %0, %for.body ], [ %20, %for.body24 ]
  %acc_1.0.lcssa = phi i32 [ %0, %for.body ], [ %21, %for.body24 ]
  %acc_2.0.lcssa = phi i32 [ %0, %for.body ], [ %22, %for.body24 ]
  %acc_3.0.lcssa = phi i32 [ %0, %for.body ], [ %23, %for.body24 ]
  %i_row_loop.1.lcssa = phi i32 [ %i_row_loop.0115, %for.body ], [ %div, %for.body24 ]
  %add31 = add nsw i32 %acc_1.0.lcssa, %acc_0.0.lcssa
  %add32 = add nsw i32 %add31, %acc_2.0.lcssa
  %add33 = add nsw i32 %add32, %acc_3.0.lcssa
  %conv34 = trunc i32 %add33 to i8
  %arrayidx35 = getelementptr inbounds i8, i8* %out, i32 %i_out_ch.0116
  store i8 %conv34, i8* %arrayidx35, align 1
  %inc37 = add nuw nsw i32 %i_out_ch.0116, 1
  %exitcond120 = icmp eq i32 %inc37, %conv2
  br i1 %exitcond120, label %if.end, label %for.body

for.body24:                                       ; preds = %for.body24, %for.body24.preheader
  %ip_r0.0109 = phi i8* [ %add.ptr26, %for.body24 ], [ %add.ptr, %for.body24.preheader ]
  %ip_c0.0108 = phi i8* [ %add.ptr27, %for.body24 ], [ %input_col, %for.body24.preheader ]
  %ip_c1.0107 = phi i8* [ %add.ptr28, %for.body24 ], [ %add.ptr9, %for.body24.preheader ]
  %ip_c2.0106 = phi i8* [ %add.ptr29, %for.body24 ], [ %add.ptr12, %for.body24.preheader ]
  %i_row_loop.1105 = phi i32 [ %inc, %for.body24 ], [ %i_row_loop.0115, %for.body24.preheader ]
  %ip_c3.0104 = phi i8* [ %add.ptr30, %for.body24 ], [ %add.ptr15, %for.body24.preheader ]
  %acc_3.0103 = phi i32 [ %23, %for.body24 ], [ %0, %for.body24.preheader ]
  %acc_2.0102 = phi i32 [ %22, %for.body24 ], [ %0, %for.body24.preheader ]
  %acc_1.0101 = phi i32 [ %21, %for.body24 ], [ %0, %for.body24.preheader ]
  %acc_0.0100 = phi i32 [ %20, %for.body24 ], [ %0, %for.body24.preheader ]
  %1 = bitcast i8* %ip_r0.0109 to <8 x i8>*
  %2 = load <8 x i8>, <8 x i8>* %1, align 1
  %3 = sext <8 x i8> %2 to <8 x i16>
  %add.ptr26 = getelementptr inbounds i8, i8* %ip_r0.0109, i32 8
  %4 = bitcast i8* %ip_c0.0108 to <8 x i8>*
  %5 = load <8 x i8>, <8 x i8>* %4, align 1
  %6 = sext <8 x i8> %5 to <8 x i16>
  %add.ptr27 = getelementptr inbounds i8, i8* %ip_c0.0108, i32 8
  %7 = add <8 x i16> %.splat, %6
  %8 = bitcast i8* %ip_c1.0107 to <8 x i8>*
  %9 = load <8 x i8>, <8 x i8>* %8, align 1
  %10 = sext <8 x i8> %9 to <8 x i16>
  %add.ptr28 = getelementptr inbounds i8, i8* %ip_c1.0107, i32 8
  %11 = add <8 x i16> %.splat, %10
  %12 = bitcast i8* %ip_c2.0106 to <8 x i8>*
  %13 = load <8 x i8>, <8 x i8>* %12, align 1
  %14 = sext <8 x i8> %13 to <8 x i16>
  %add.ptr29 = getelementptr inbounds i8, i8* %ip_c2.0106, i32 8
  %15 = add <8 x i16> %.splat, %14
  %16 = bitcast i8* %ip_c3.0104 to <8 x i8>*
  %17 = load <8 x i8>, <8 x i8>* %16, align 1
  %18 = sext <8 x i8> %17 to <8 x i16>
  %add.ptr30 = getelementptr inbounds i8, i8* %ip_c3.0104, i32 8
  %19 = add <8 x i16> %.splat, %18
  %20 = tail call i32 @llvm.arm.mve.vmldava.v8i16(i32 0, i32 0, i32 0, i32 %acc_0.0100, <8 x i16> %3, <8 x i16> %7)
  %21 = tail call i32 @llvm.arm.mve.vmldava.v8i16(i32 0, i32 0, i32 0, i32 %acc_1.0101, <8 x i16> %3, <8 x i16> %11)
  %22 = tail call i32 @llvm.arm.mve.vmldava.v8i16(i32 0, i32 0, i32 0, i32 %acc_2.0102, <8 x i16> %3, <8 x i16> %15)
  %23 = tail call i32 @llvm.arm.mve.vmldava.v8i16(i32 0, i32 0, i32 0, i32 %acc_3.0103, <8 x i16> %3, <8 x i16> %19)
  %inc = add nsw i32 %i_row_loop.1105, 1
  %exitcond = icmp eq i32 %inc, %div
  br i1 %exitcond, label %for.cond.cleanup23, label %for.body24

if.end:                                           ; preds = %for.cond.cleanup23, %for.cond.preheader, %entry
  ret i8* %out
}

; Same as above with optsize
define i8* @test_optsize(i8* nocapture readonly %input_row, i8* nocapture readonly %input_col, i16 zeroext %output_ch, i16 zeroext %num_cols, i32 %col_offset, i16 signext %activation_min, i16 zeroext %row_len, i32* nocapture readonly %bias, i8* returned %out) optsize {
; CHECK-LABEL: test_optsize:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-NEXT:    .pad #20
; CHECK-NEXT:    sub sp, #20
; CHECK-NEXT:    cmp r3, #4
; CHECK-NEXT:    strd r0, r1, [sp, #12] @ 8-byte Folded Spill
; CHECK-NEXT:    bne .LBB3_8
; CHECK-NEXT:  @ %bb.1: @ %for.cond.preheader
; CHECK-NEXT:    cmp r2, #0
; CHECK-NEXT:    beq .LBB3_8
; CHECK-NEXT:  @ %bb.2: @ %for.body.lr.ph
; CHECK-NEXT:    ldr r3, [sp, #64]
; CHECK-NEXT:    mov.w r11, #0
; CHECK-NEXT:    ldr r1, [sp, #16] @ 4-byte Reload
; CHECK-NEXT:    ldr.w r9, [sp, #56]
; CHECK-NEXT:    add.w r0, r1, r3, lsl #1
; CHECK-NEXT:    str r0, [sp, #8] @ 4-byte Spill
; CHECK-NEXT:    adds r0, r1, r3
; CHECK-NEXT:    str r0, [sp, #4] @ 4-byte Spill
; CHECK-NEXT:    add.w r0, r3, r3, lsl #1
; CHECK-NEXT:    add r0, r1
; CHECK-NEXT:    str r0, [sp] @ 4-byte Spill
; CHECK-NEXT:    adds r0, r3, #7
; CHECK-NEXT:    lsrs r0, r0, #3
; CHECK-NEXT:  .LBB3_3: @ %for.body
; CHECK-NEXT:    @ =>This Loop Header: Depth=1
; CHECK-NEXT:    @ Child Loop BB3_5 Depth 2
; CHECK-NEXT:    ldr r1, [sp, #68]
; CHECK-NEXT:    subs.w lr, r0, r0
; CHECK-NEXT:    ldr.w r12, [r1, r11, lsl #2]
; CHECK-NEXT:    ble .LBB3_6
; CHECK-NEXT:  @ %bb.4: @ %for.body24.preheader
; CHECK-NEXT:    @ in Loop: Header=BB3_3 Depth=1
; CHECK-NEXT:    ldr r3, [sp, #64]
; CHECK-NEXT:    mov r6, r12
; CHECK-NEXT:    ldr r1, [sp, #12] @ 4-byte Reload
; CHECK-NEXT:    dls lr, lr
; CHECK-NEXT:    ldr r5, [sp, #8] @ 4-byte Reload
; CHECK-NEXT:    mov r10, r12
; CHECK-NEXT:    mla r7, r11, r3, r1
; CHECK-NEXT:    ldr r1, [sp, #16] @ 4-byte Reload
; CHECK-NEXT:    ldrd r4, r3, [sp] @ 8-byte Folded Reload
; CHECK-NEXT:    mov r8, r12
; CHECK-NEXT:  .LBB3_5: @ %for.body24
; CHECK-NEXT:    @ Parent Loop BB3_3 Depth=1
; CHECK-NEXT:    @ => This Inner Loop Header: Depth=2
; CHECK-NEXT:    vldrb.s16 q0, [r4], #8
; CHECK-NEXT:    vadd.i16 q1, q0, r9
; CHECK-NEXT:    vldrb.s16 q0, [r7], #8
; CHECK-NEXT:    vmlava.s16 r12, q0, q1
; CHECK-NEXT:    vldrb.s16 q1, [r5], #8
; CHECK-NEXT:    vadd.i16 q1, q1, r9
; CHECK-NEXT:    vmlava.s16 r6, q0, q1
; CHECK-NEXT:    vldrb.s16 q1, [r3], #8
; CHECK-NEXT:    vadd.i16 q1, q1, r9
; CHECK-NEXT:    vmlava.s16 r10, q0, q1
; CHECK-NEXT:    vldrb.s16 q1, [r1], #8
; CHECK-NEXT:    vadd.i16 q1, q1, r9
; CHECK-NEXT:    vmlava.s16 r8, q0, q1
; CHECK-NEXT:    le lr, .LBB3_5
; CHECK-NEXT:    b .LBB3_7
; CHECK-NEXT:  .LBB3_6: @ in Loop: Header=BB3_3 Depth=1
; CHECK-NEXT:    mov r8, r12
; CHECK-NEXT:    mov r10, r12
; CHECK-NEXT:    mov r6, r12
; CHECK-NEXT:  .LBB3_7: @ %for.cond.cleanup23
; CHECK-NEXT:    @ in Loop: Header=BB3_3 Depth=1
; CHECK-NEXT:    ldr r3, [sp, #72]
; CHECK-NEXT:    add.w r1, r10, r8
; CHECK-NEXT:    add r1, r6
; CHECK-NEXT:    add r1, r12
; CHECK-NEXT:    strb.w r1, [r3, r11]
; CHECK-NEXT:    add.w r11, r11, #1
; CHECK-NEXT:    cmp r11, r2
; CHECK-NEXT:    bne .LBB3_3
; CHECK-NEXT:  .LBB3_8: @ %if.end
; CHECK-NEXT:    ldr r0, [sp, #72]
; CHECK-NEXT:    add sp, #20
; CHECK-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11, pc}
entry:
  %cmp = icmp eq i16 %num_cols, 4
  br i1 %cmp, label %for.cond.preheader, label %if.end

for.cond.preheader:                               ; preds = %entry
  %conv2 = zext i16 %output_ch to i32
  %cmp3114 = icmp eq i16 %output_ch, 0
  br i1 %cmp3114, label %if.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %for.cond.preheader
  %conv5 = zext i16 %row_len to i32
  %add.ptr9 = getelementptr inbounds i8, i8* %input_col, i32 %conv5
  %mul11 = shl nuw nsw i32 %conv5, 1
  %add.ptr12 = getelementptr inbounds i8, i8* %input_col, i32 %mul11
  %mul14 = mul nuw nsw i32 %conv5, 3
  %add.ptr15 = getelementptr inbounds i8, i8* %input_col, i32 %mul14
  %add = add nuw nsw i32 %conv5, 7
  %div = lshr i32 %add, 3
  %conv25 = trunc i32 %col_offset to i16
  %.splatinsert = insertelement <8 x i16> undef, i16 %conv25, i32 0
  %.splat = shufflevector <8 x i16> %.splatinsert, <8 x i16> undef, <8 x i32> zeroinitializer
  br label %for.body

for.body:                                         ; preds = %for.cond.cleanup23, %for.body.lr.ph
  %i_out_ch.0116 = phi i32 [ 0, %for.body.lr.ph ], [ %inc37, %for.cond.cleanup23 ]
  %i_row_loop.0115 = phi i32 [ undef, %for.body.lr.ph ], [ %i_row_loop.1.lcssa, %for.cond.cleanup23 ]
  %arrayidx = getelementptr inbounds i32, i32* %bias, i32 %i_out_ch.0116
  %0 = load i32, i32* %arrayidx, align 4
  %cmp2199 = icmp slt i32 %i_row_loop.0115, %div
  br i1 %cmp2199, label %for.body24.preheader, label %for.cond.cleanup23

for.body24.preheader:                             ; preds = %for.body
  %mul = mul nuw nsw i32 %i_out_ch.0116, %conv5
  %add.ptr = getelementptr inbounds i8, i8* %input_row, i32 %mul
  br label %for.body24

for.cond.cleanup23:                               ; preds = %for.body24, %for.body
  %acc_0.0.lcssa = phi i32 [ %0, %for.body ], [ %20, %for.body24 ]
  %acc_1.0.lcssa = phi i32 [ %0, %for.body ], [ %21, %for.body24 ]
  %acc_2.0.lcssa = phi i32 [ %0, %for.body ], [ %22, %for.body24 ]
  %acc_3.0.lcssa = phi i32 [ %0, %for.body ], [ %23, %for.body24 ]
  %i_row_loop.1.lcssa = phi i32 [ %i_row_loop.0115, %for.body ], [ %div, %for.body24 ]
  %add31 = add nsw i32 %acc_1.0.lcssa, %acc_0.0.lcssa
  %add32 = add nsw i32 %add31, %acc_2.0.lcssa
  %add33 = add nsw i32 %add32, %acc_3.0.lcssa
  %conv34 = trunc i32 %add33 to i8
  %arrayidx35 = getelementptr inbounds i8, i8* %out, i32 %i_out_ch.0116
  store i8 %conv34, i8* %arrayidx35, align 1
  %inc37 = add nuw nsw i32 %i_out_ch.0116, 1
  %exitcond120 = icmp eq i32 %inc37, %conv2
  br i1 %exitcond120, label %if.end, label %for.body

for.body24:                                       ; preds = %for.body24, %for.body24.preheader
  %ip_r0.0109 = phi i8* [ %add.ptr26, %for.body24 ], [ %add.ptr, %for.body24.preheader ]
  %ip_c0.0108 = phi i8* [ %add.ptr27, %for.body24 ], [ %input_col, %for.body24.preheader ]
  %ip_c1.0107 = phi i8* [ %add.ptr28, %for.body24 ], [ %add.ptr9, %for.body24.preheader ]
  %ip_c2.0106 = phi i8* [ %add.ptr29, %for.body24 ], [ %add.ptr12, %for.body24.preheader ]
  %i_row_loop.1105 = phi i32 [ %inc, %for.body24 ], [ %i_row_loop.0115, %for.body24.preheader ]
  %ip_c3.0104 = phi i8* [ %add.ptr30, %for.body24 ], [ %add.ptr15, %for.body24.preheader ]
  %acc_3.0103 = phi i32 [ %23, %for.body24 ], [ %0, %for.body24.preheader ]
  %acc_2.0102 = phi i32 [ %22, %for.body24 ], [ %0, %for.body24.preheader ]
  %acc_1.0101 = phi i32 [ %21, %for.body24 ], [ %0, %for.body24.preheader ]
  %acc_0.0100 = phi i32 [ %20, %for.body24 ], [ %0, %for.body24.preheader ]
  %1 = bitcast i8* %ip_r0.0109 to <8 x i8>*
  %2 = load <8 x i8>, <8 x i8>* %1, align 1
  %3 = sext <8 x i8> %2 to <8 x i16>
  %add.ptr26 = getelementptr inbounds i8, i8* %ip_r0.0109, i32 8
  %4 = bitcast i8* %ip_c0.0108 to <8 x i8>*
  %5 = load <8 x i8>, <8 x i8>* %4, align 1
  %6 = sext <8 x i8> %5 to <8 x i16>
  %add.ptr27 = getelementptr inbounds i8, i8* %ip_c0.0108, i32 8
  %7 = add <8 x i16> %.splat, %6
  %8 = bitcast i8* %ip_c1.0107 to <8 x i8>*
  %9 = load <8 x i8>, <8 x i8>* %8, align 1
  %10 = sext <8 x i8> %9 to <8 x i16>
  %add.ptr28 = getelementptr inbounds i8, i8* %ip_c1.0107, i32 8
  %11 = add <8 x i16> %.splat, %10
  %12 = bitcast i8* %ip_c2.0106 to <8 x i8>*
  %13 = load <8 x i8>, <8 x i8>* %12, align 1
  %14 = sext <8 x i8> %13 to <8 x i16>
  %add.ptr29 = getelementptr inbounds i8, i8* %ip_c2.0106, i32 8
  %15 = add <8 x i16> %.splat, %14
  %16 = bitcast i8* %ip_c3.0104 to <8 x i8>*
  %17 = load <8 x i8>, <8 x i8>* %16, align 1
  %18 = sext <8 x i8> %17 to <8 x i16>
  %add.ptr30 = getelementptr inbounds i8, i8* %ip_c3.0104, i32 8
  %19 = add <8 x i16> %.splat, %18
  %20 = tail call i32 @llvm.arm.mve.vmldava.v8i16(i32 0, i32 0, i32 0, i32 %acc_0.0100, <8 x i16> %3, <8 x i16> %7)
  %21 = tail call i32 @llvm.arm.mve.vmldava.v8i16(i32 0, i32 0, i32 0, i32 %acc_1.0101, <8 x i16> %3, <8 x i16> %11)
  %22 = tail call i32 @llvm.arm.mve.vmldava.v8i16(i32 0, i32 0, i32 0, i32 %acc_2.0102, <8 x i16> %3, <8 x i16> %15)
  %23 = tail call i32 @llvm.arm.mve.vmldava.v8i16(i32 0, i32 0, i32 0, i32 %acc_3.0103, <8 x i16> %3, <8 x i16> %19)
  %inc = add nsw i32 %i_row_loop.1105, 1
  %exitcond = icmp eq i32 %inc, %div
  br i1 %exitcond, label %for.cond.cleanup23, label %for.body24

if.end:                                           ; preds = %for.cond.cleanup23, %for.cond.preheader, %entry
  ret i8* %out
}


; Similar but predicated
define i32 @arm_nn_mat_mul_core_4x_s8(i32 %row_elements, i32 %offset, i8* %row_base, i8* %col_base, i32* nocapture readnone %sum_col, i32* nocapture %output) {
; CHECK-LABEL: arm_nn_mat_mul_core_4x_s8:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r4, r5, r6, r7, r8, r10, lr}
; CHECK-NEXT:    push.w {r4, r5, r6, r7, r8, r10, lr}
; CHECK-NEXT:    add.w r7, r0, #15
; CHECK-NEXT:    ldr.w r12, [sp, #32]
; CHECK-NEXT:    mov.w lr, #1
; CHECK-NEXT:    asrs r6, r7, #31
; CHECK-NEXT:    add.w r4, r7, r6, lsr #28
; CHECK-NEXT:    asrs r5, r4, #4
; CHECK-NEXT:    cmp r5, #1
; CHECK-NEXT:    it gt
; CHECK-NEXT:    asrgt.w lr, r4, #4
; CHECK-NEXT:    cmp r0, #1
; CHECK-NEXT:    blt .LBB4_3
; CHECK-NEXT:  @ %bb.1: @ %for.body.preheader
; CHECK-NEXT:    adds r5, r2, r1
; CHECK-NEXT:    add.w r7, r2, r1, lsl #1
; CHECK-NEXT:    add.w r1, r1, r1, lsl #1
; CHECK-NEXT:    mov.w r8, #0
; CHECK-NEXT:    add r1, r2
; CHECK-NEXT:    movs r6, #0
; CHECK-NEXT:    movs r4, #0
; CHECK-NEXT:    mov.w r10, #0
; CHECK-NEXT:    dlstp.8 lr, r0
; CHECK-NEXT:  .LBB4_2: @ %for.body
; CHECK-NEXT:    @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    vldrb.u8 q0, [r3], #16
; CHECK-NEXT:    vldrb.u8 q1, [r1], #16
; CHECK-NEXT:    vmlava.s8 r10, q1, q0
; CHECK-NEXT:    vldrb.u8 q1, [r7], #16
; CHECK-NEXT:    vmlava.s8 r4, q1, q0
; CHECK-NEXT:    vldrb.u8 q1, [r5], #16
; CHECK-NEXT:    vmlava.s8 r6, q1, q0
; CHECK-NEXT:    vldrb.u8 q1, [r2], #16
; CHECK-NEXT:    vmlava.s8 r8, q1, q0
; CHECK-NEXT:    letp lr, .LBB4_2
; CHECK-NEXT:    b .LBB4_4
; CHECK-NEXT:  .LBB4_3:
; CHECK-NEXT:    mov.w r10, #0
; CHECK-NEXT:    movs r4, #0
; CHECK-NEXT:    movs r6, #0
; CHECK-NEXT:    mov.w r8, #0
; CHECK-NEXT:  .LBB4_4: @ %for.cond.cleanup
; CHECK-NEXT:    movs r0, #0
; CHECK-NEXT:    strd r8, r6, [r12]
; CHECK-NEXT:    strd r4, r10, [r12, #8]
; CHECK-NEXT:    pop.w {r4, r5, r6, r7, r8, r10, pc}
entry:
  %add = add nsw i32 %row_elements, 15
  %div = sdiv i32 %add, 16
  %cmp84 = icmp sgt i32 %row_elements, 0
  br i1 %cmp84, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %mul2 = mul nsw i32 %offset, 3
  %add.ptr3 = getelementptr inbounds i8, i8* %row_base, i32 %mul2
  %mul = shl nsw i32 %offset, 1
  %add.ptr1 = getelementptr inbounds i8, i8* %row_base, i32 %mul
  %add.ptr = getelementptr inbounds i8, i8* %row_base, i32 %offset
  %0 = icmp sgt i32 %div, 1
  %smax = select i1 %0, i32 %div, i32 1
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %acc_n.sroa.12.0.lcssa = phi i32 [ 0, %entry ], [ %15, %for.body ]
  %acc_n.sroa.9.0.lcssa = phi i32 [ 0, %entry ], [ %12, %for.body ]
  %acc_n.sroa.6.0.lcssa = phi i32 [ 0, %entry ], [ %9, %for.body ]
  %acc_n.sroa.0.0.lcssa = phi i32 [ 0, %entry ], [ %6, %for.body ]
  store i32 %acc_n.sroa.0.0.lcssa, i32* %output, align 4
  %arrayidx19 = getelementptr inbounds i32, i32* %output, i32 1
  store i32 %acc_n.sroa.6.0.lcssa, i32* %arrayidx19, align 4
  %arrayidx21 = getelementptr inbounds i32, i32* %output, i32 2
  store i32 %acc_n.sroa.9.0.lcssa, i32* %arrayidx21, align 4
  %arrayidx23 = getelementptr inbounds i32, i32* %output, i32 3
  store i32 %acc_n.sroa.12.0.lcssa, i32* %arrayidx23, align 4
  ret i32 0

for.body:                                         ; preds = %for.body, %for.body.preheader
  %col_base.addr.095 = phi i8* [ %add.ptr4, %for.body ], [ %col_base, %for.body.preheader ]
  %acc_n.sroa.0.094 = phi i32 [ %6, %for.body ], [ 0, %for.body.preheader ]
  %acc_n.sroa.6.093 = phi i32 [ %9, %for.body ], [ 0, %for.body.preheader ]
  %acc_n.sroa.9.092 = phi i32 [ %12, %for.body ], [ 0, %for.body.preheader ]
  %i.091 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %row_elem.090 = phi i32 [ %sub, %for.body ], [ %row_elements, %for.body.preheader ]
  %acc_n.sroa.12.089 = phi i32 [ %15, %for.body ], [ 0, %for.body.preheader ]
  %ip_row_3.088 = phi i8* [ %add.ptr15, %for.body ], [ %add.ptr3, %for.body.preheader ]
  %ip_row_2.087 = phi i8* [ %add.ptr14, %for.body ], [ %add.ptr1, %for.body.preheader ]
  %ip_row_1.086 = phi i8* [ %add.ptr13, %for.body ], [ %add.ptr, %for.body.preheader ]
  %ip_row_0.085 = phi i8* [ %add.ptr12, %for.body ], [ %row_base, %for.body.preheader ]
  %1 = tail call <16 x i1> @llvm.arm.mve.vctp8(i32 %row_elem.090)
  %sub = add nsw i32 %row_elem.090, -16
  %2 = bitcast i8* %col_base.addr.095 to <16 x i8>*
  %3 = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %2, i32 1, <16 x i1> %1, <16 x i8> zeroinitializer)
  %add.ptr4 = getelementptr inbounds i8, i8* %col_base.addr.095, i32 16
  %4 = bitcast i8* %ip_row_0.085 to <16 x i8>*
  %5 = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %4, i32 1, <16 x i1> %1, <16 x i8> zeroinitializer)
  %6 = tail call i32 @llvm.arm.mve.vmldava.predicated.v16i8.v16i1(i32 0, i32 0, i32 0, i32 %acc_n.sroa.0.094, <16 x i8> %5, <16 x i8> %3, <16 x i1> %1)
  %7 = bitcast i8* %ip_row_1.086 to <16 x i8>*
  %8 = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %7, i32 1, <16 x i1> %1, <16 x i8> zeroinitializer)
  %9 = tail call i32 @llvm.arm.mve.vmldava.predicated.v16i8.v16i1(i32 0, i32 0, i32 0, i32 %acc_n.sroa.6.093, <16 x i8> %8, <16 x i8> %3, <16 x i1> %1)
  %10 = bitcast i8* %ip_row_2.087 to <16 x i8>*
  %11 = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %10, i32 1, <16 x i1> %1, <16 x i8> zeroinitializer)
  %12 = tail call i32 @llvm.arm.mve.vmldava.predicated.v16i8.v16i1(i32 0, i32 0, i32 0, i32 %acc_n.sroa.9.092, <16 x i8> %11, <16 x i8> %3, <16 x i1> %1)
  %13 = bitcast i8* %ip_row_3.088 to <16 x i8>*
  %14 = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %13, i32 1, <16 x i1> %1, <16 x i8> zeroinitializer)
  %15 = tail call i32 @llvm.arm.mve.vmldava.predicated.v16i8.v16i1(i32 0, i32 0, i32 0, i32 %acc_n.sroa.12.089, <16 x i8> %14, <16 x i8> %3, <16 x i1> %1)
  %add.ptr12 = getelementptr inbounds i8, i8* %ip_row_0.085, i32 16
  %add.ptr13 = getelementptr inbounds i8, i8* %ip_row_1.086, i32 16
  %add.ptr14 = getelementptr inbounds i8, i8* %ip_row_2.087, i32 16
  %add.ptr15 = getelementptr inbounds i8, i8* %ip_row_3.088, i32 16
  %inc = add nuw nsw i32 %i.091, 1
  %exitcond = icmp eq i32 %inc, %smax
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

declare <16 x i1> @llvm.arm.mve.vctp8(i32)
declare i32 @llvm.arm.mve.pred.v2i.v16i1(<16 x i1>)
declare <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>*, i32 immarg, <4 x i1>, <4 x float>) #1
declare void @llvm.masked.store.v4f32.p0v4f32(<4 x float>, <4 x float>*, i32 immarg, <4 x i1>) #2
declare <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>*, i32 immarg, <16 x i1>, <16 x i8>)
declare i32 @llvm.experimental.vector.reduce.add.v16i8(<16 x i32> %ext4)
declare i32 @llvm.arm.mve.vmldava.v8i16(i32, i32, i32, i32, <8 x i16>, <8 x i16>)
declare i32 @llvm.arm.mve.vmldava.predicated.v16i8.v16i1(i32, i32, i32, i32, <16 x i8>, <16 x i8>, <16 x i1>)
