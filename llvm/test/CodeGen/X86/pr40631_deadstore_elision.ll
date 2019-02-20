; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -o - %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

%struct.sk_buff = type { i8* }
%struct.xt_action_param = type { i8*, i8*, i8*, i32, i32, i8 }

define i32 @ipt_do_table(%struct.sk_buff* noalias nocapture readonly) {
; CHECK-LABEL: ipt_do_table:
; CHECK:       # %bb.0:
; CHECK-NEXT:    subq $40, %rsp
; CHECK-NEXT:    .cfi_def_cfa_offset 48
; CHECK-NEXT:    movq (%rdi), %rax
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    movaps %xmm0, (%rsp)
; CHECK-NEXT:    movaps %xmm0, {{[0-9]+}}(%rsp)
; CHECK-NEXT:    movq $0, {{[0-9]+}}(%rsp)
; CHECK-NEXT:    movaps {{.*#+}} xmm0 = [12297829382473034410,12297829382473034410]
; CHECK-NEXT:    movaps %xmm0, (%rsp)
; CHECK-NEXT:    movabsq $-6148914691236517206, %rcx # imm = 0xAAAAAAAAAAAAAAAA
; CHECK-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; CHECK-NEXT:    movq %rcx, {{[0-9]+}}(%rsp)
; CHECK-NEXT:    movb $-86, {{[0-9]+}}(%rsp)
; CHECK-NEXT:    movzwl 2(%rax), %ecx
; CHECK-NEXT:    andl $8191, %ecx # imm = 0x1FFF
; CHECK-NEXT:    movl %ecx, {{[0-9]+}}(%rsp)
; CHECK-NEXT:    movzbl (%rax), %eax
; CHECK-NEXT:    andl $15, %eax
; CHECK-NEXT:    movl %eax, {{[0-9]+}}(%rsp)
; CHECK-NEXT:    movb $0, {{[0-9]+}}(%rsp)
; CHECK-NEXT:    movq %rsp, %rdi
; CHECK-NEXT:    callq use_it
; CHECK-NEXT:    addq $40, %rsp
; CHECK-NEXT:    .cfi_def_cfa_offset 8
; CHECK-NEXT:    retq
  %2 = alloca %struct.xt_action_param, align 16
  %3 = getelementptr inbounds %struct.sk_buff, %struct.sk_buff* %0, i64 0, i32 0
  %4 = load i8*, i8** %3, align 8
  %5 = bitcast %struct.xt_action_param* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %5) #3
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 %5, i8 0, i64 40, i1 false)
  %6 = bitcast %struct.xt_action_param* %2 to <2 x i8*>*
  store <2 x i8*> <i8* inttoptr (i64 -6148914691236517206 to i8*), i8* inttoptr (i64 -6148914691236517206 to i8*)>, <2 x i8*>* %6, align 16
  %7 = getelementptr inbounds %struct.xt_action_param, %struct.xt_action_param* %2, i64 0, i32 2
  store i8* inttoptr (i64 -6148914691236517206 to i8*), i8** %7, align 16
  %8 = getelementptr inbounds %struct.xt_action_param, %struct.xt_action_param* %2, i64 0, i32 3
  %9 = getelementptr inbounds %struct.xt_action_param, %struct.xt_action_param* %2, i64 0, i32 4
  %10 = getelementptr inbounds %struct.xt_action_param, %struct.xt_action_param* %2, i64 0, i32 5
  %11 = getelementptr inbounds i8, i8* %4, i64 2
  %12 = bitcast i8* %11 to i16*
  %13 = bitcast i32* %8 to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull align 8 %13, i8 -86, i64 9, i1 false)
  %14 = load i16, i16* %12, align 2
  %15 = and i16 %14, 8191
  %16 = zext i16 %15 to i32
  store i32 %16, i32* %8, align 8
  %17 = load i8, i8* %4, align 2
  %18 = and i8 %17, 15
  %19 = zext i8 %18 to i32
  store i32 %19, i32* %9, align 4
  store i8 0, i8* %10, align 16
  %20 = call i32 @use_it(%struct.xt_action_param* nonnull %2)
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %5)
  ret i32 %20
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

declare dso_local i32 @use_it(%struct.xt_action_param*) local_unnamed_addr

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
