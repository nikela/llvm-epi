; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse4.2 | FileCheck %s

define i1 @cmp_allbits_concat_i8(i8 %x, i8 %y) {
; CHECK-LABEL: cmp_allbits_concat_i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movzbl %sil, %eax
; CHECK-NEXT:    shll $8, %edi
; CHECK-NEXT:    orl %eax, %edi
; CHECK-NEXT:    cmpw $-1, %di
; CHECK-NEXT:    sete %al
; CHECK-NEXT:    retq
  %zx = zext i8 %x to i16
  %zy = zext i8 %y to i16
  %sh = shl i16 %zx, 8
  %or = or i16 %zy, %sh
  %r = icmp eq i16 %or, -1
  ret i1 %r
}

define i1 @cmp_anybits_concat_i32(i32 %x, i32 %y) {
; CHECK-LABEL: cmp_anybits_concat_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $edi killed $edi def $rdi
; CHECK-NEXT:    movl %esi, %eax
; CHECK-NEXT:    shlq $32, %rdi
; CHECK-NEXT:    orq %rax, %rdi
; CHECK-NEXT:    setne %al
; CHECK-NEXT:    retq
  %zx = zext i32 %x to i64
  %zy = zext i32 %y to i64
  %sh = shl i64 %zx, 32
  %or = or i64 %zy, %sh
  %r = icmp ne i64 %or, 0
  ret i1 %r
}

define <16 x i8> @cmp_allbits_concat_v16i8(<16 x i8> %x, <16 x i8> %y) {
; CHECK-LABEL: cmp_allbits_concat_v16i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movdqa %xmm1, %xmm2
; CHECK-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; CHECK-NEXT:    punpckhbw {{.*#+}} xmm1 = xmm1[8],xmm0[8],xmm1[9],xmm0[9],xmm1[10],xmm0[10],xmm1[11],xmm0[11],xmm1[12],xmm0[12],xmm1[13],xmm0[13],xmm1[14],xmm0[14],xmm1[15],xmm0[15]
; CHECK-NEXT:    pcmpeqd %xmm0, %xmm0
; CHECK-NEXT:    pcmpeqw %xmm0, %xmm1
; CHECK-NEXT:    pcmpeqw %xmm2, %xmm0
; CHECK-NEXT:    packsswb %xmm1, %xmm0
; CHECK-NEXT:    retq
  %zx = zext <16 x i8> %x to <16 x i16>
  %zy = zext <16 x i8> %y to <16 x i16>
  %sh = shl <16 x i16> %zx, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %or = or <16 x i16> %zy, %sh
  %r = icmp eq <16 x i16> %or, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  %s = sext <16 x i1> %r to <16 x i8>
  ret <16 x i8> %s
}

define <2 x i64> @cmp_nobits_concat_v2i64(<2 x i64> %x, <2 x i64> %y) {
; CHECK-LABEL: cmp_nobits_concat_v2i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %xmm0, %rax
; CHECK-NEXT:    pextrq $1, %xmm0, %rcx
; CHECK-NEXT:    movq %xmm1, %rdx
; CHECK-NEXT:    pextrq $1, %xmm1, %rsi
; CHECK-NEXT:    xorl %edi, %edi
; CHECK-NEXT:    orq %rcx, %rsi
; CHECK-NEXT:    sete %dil
; CHECK-NEXT:    negq %rdi
; CHECK-NEXT:    movq %rdi, %xmm1
; CHECK-NEXT:    xorl %ecx, %ecx
; CHECK-NEXT:    orq %rax, %rdx
; CHECK-NEXT:    sete %cl
; CHECK-NEXT:    negq %rcx
; CHECK-NEXT:    movq %rcx, %xmm0
; CHECK-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; CHECK-NEXT:    retq
  %zx = zext <2 x i64> %x to <2 x i128>
  %zy = zext <2 x i64> %y to <2 x i128>
  %sh = shl <2 x i128> %zx, <i128 64, i128 64>
  %or = or <2 x i128> %zy, %sh
  %r = icmp eq <2 x i128> %or, zeroinitializer
  %s = sext <2 x i1> %r to <2 x i64>
  ret <2 x i64> %s
}
