; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=i386-apple-darwin -mattr=-avx,+sse2 -show-mc-encoding | FileCheck %s --check-prefix=CHECK --check-prefix=SSE
; RUN: llc < %s -mtriple=i386-apple-darwin -mattr=+avx2 -show-mc-encoding | FileCheck %s --check-prefix=CHECK --check-prefix=VCHECK --check-prefix=AVX2
; RUN: llc < %s -mtriple=i386-apple-darwin -mcpu=skx -show-mc-encoding | FileCheck %s --check-prefix=CHECK --check-prefix=VCHECK --check-prefix=SKX

; NOTE: This should use IR equivalent to what is generated by clang/test/CodeGen/sse2-builtins.c


define <16 x i8> @test_x86_sse2_paddus_b(<16 x i8> %a0, <16 x i8> %a1) {
; SSE-LABEL: test_x86_sse2_paddus_b:
; SSE:       ## %bb.0:
; SSE-NEXT:    paddusb %xmm1, %xmm0 ## encoding: [0x66,0x0f,0xdc,0xc1]
; SSE-NEXT:    retl ## encoding: [0xc3]
;
; AVX2-LABEL: test_x86_sse2_paddus_b:
; AVX2:       ## %bb.0:
; AVX2-NEXT:    vpaddusb %xmm1, %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0xdc,0xc1]
; AVX2-NEXT:    retl ## encoding: [0xc3]
;
; SKX-LABEL: test_x86_sse2_paddus_b:
; SKX:       ## %bb.0:
; SKX-NEXT:    vpaddusb %xmm1, %xmm0, %xmm0 ## EVEX TO VEX Compression encoding: [0xc5,0xf9,0xdc,0xc1]
; SKX-NEXT:    retl ## encoding: [0xc3]
  %1 = add <16 x i8> %a0, %a1
  %2 = icmp ugt <16 x i8> %a0, %1
  %3 = select <16 x i1> %2, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, <16 x i8> %1
  ret <16 x i8> %3
}

define <8 x i16> @test_x86_sse2_paddus_w(<8 x i16> %a0, <8 x i16> %a1) {
; SSE-LABEL: test_x86_sse2_paddus_w:
; SSE:       ## %bb.0:
; SSE-NEXT:    paddusw %xmm1, %xmm0 ## encoding: [0x66,0x0f,0xdd,0xc1]
; SSE-NEXT:    retl ## encoding: [0xc3]
;
; AVX2-LABEL: test_x86_sse2_paddus_w:
; AVX2:       ## %bb.0:
; AVX2-NEXT:    vpaddusw %xmm1, %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0xdd,0xc1]
; AVX2-NEXT:    retl ## encoding: [0xc3]
;
; SKX-LABEL: test_x86_sse2_paddus_w:
; SKX:       ## %bb.0:
; SKX-NEXT:    vpaddusw %xmm1, %xmm0, %xmm0 ## EVEX TO VEX Compression encoding: [0xc5,0xf9,0xdd,0xc1]
; SKX-NEXT:    retl ## encoding: [0xc3]
  %1 = add <8 x i16> %a0, %a1
  %2 = icmp ugt <8 x i16> %a0, %1
  %3 = select <8 x i1> %2, <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>, <8 x i16> %1
  ret <8 x i16> %3
}

define <16 x i8> @test_x86_sse2_psubus_b(<16 x i8> %a0, <16 x i8> %a1) {
; SSE-LABEL: test_x86_sse2_psubus_b:
; SSE:       ## %bb.0:
; SSE-NEXT:    psubusb %xmm1, %xmm0 ## encoding: [0x66,0x0f,0xd8,0xc1]
; SSE-NEXT:    retl ## encoding: [0xc3]
;
; AVX2-LABEL: test_x86_sse2_psubus_b:
; AVX2:       ## %bb.0:
; AVX2-NEXT:    vpsubusb %xmm1, %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0xd8,0xc1]
; AVX2-NEXT:    retl ## encoding: [0xc3]
;
; SKX-LABEL: test_x86_sse2_psubus_b:
; SKX:       ## %bb.0:
; SKX-NEXT:    vpsubusb %xmm1, %xmm0, %xmm0 ## EVEX TO VEX Compression encoding: [0xc5,0xf9,0xd8,0xc1]
; SKX-NEXT:    retl ## encoding: [0xc3]
  %cmp = icmp ugt <16 x i8> %a0, %a1
  %sel = select <16 x i1> %cmp, <16 x i8> %a0, <16 x i8> %a1
  %sub = sub <16 x i8> %sel, %a1
  ret <16 x i8> %sub
}

define <8 x i16> @test_x86_sse2_psubus_w(<8 x i16> %a0, <8 x i16> %a1) {
; SSE-LABEL: test_x86_sse2_psubus_w:
; SSE:       ## %bb.0:
; SSE-NEXT:    psubusw %xmm1, %xmm0 ## encoding: [0x66,0x0f,0xd9,0xc1]
; SSE-NEXT:    retl ## encoding: [0xc3]
;
; AVX2-LABEL: test_x86_sse2_psubus_w:
; AVX2:       ## %bb.0:
; AVX2-NEXT:    vpsubusw %xmm1, %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0xd9,0xc1]
; AVX2-NEXT:    retl ## encoding: [0xc3]
;
; SKX-LABEL: test_x86_sse2_psubus_w:
; SKX:       ## %bb.0:
; SKX-NEXT:    vpsubusw %xmm1, %xmm0, %xmm0 ## EVEX TO VEX Compression encoding: [0xc5,0xf9,0xd9,0xc1]
; SKX-NEXT:    retl ## encoding: [0xc3]
  %cmp = icmp ugt <8 x i16> %a0, %a1
  %sel = select <8 x i1> %cmp, <8 x i16> %a0, <8 x i16> %a1
  %sub = sub <8 x i16> %sel, %a1
  ret <8 x i16> %sub
}

define <8 x i8> @test_x86_sse2_paddus_b_64(<8 x i8> %a0, <8 x i8> %a1) {
; SSE-LABEL: test_x86_sse2_paddus_b_64:
; SSE:       ## %bb.0:
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0]
; SSE-NEXT:    ## encoding: [0x66,0x0f,0x6f,0x15,A,A,A,A]
; SSE-NEXT:    ## fixup A - offset: 4, value: LCPI4_0, kind: FK_Data_4
; SSE-NEXT:    paddw %xmm0, %xmm1 ## encoding: [0x66,0x0f,0xfd,0xc8]
; SSE-NEXT:    pand %xmm2, %xmm0 ## encoding: [0x66,0x0f,0xdb,0xc2]
; SSE-NEXT:    pand %xmm1, %xmm2 ## encoding: [0x66,0x0f,0xdb,0xd1]
; SSE-NEXT:    pcmpgtw %xmm2, %xmm0 ## encoding: [0x66,0x0f,0x65,0xc2]
; SSE-NEXT:    movdqa %xmm0, %xmm2 ## encoding: [0x66,0x0f,0x6f,0xd0]
; SSE-NEXT:    pandn %xmm1, %xmm2 ## encoding: [0x66,0x0f,0xdf,0xd1]
; SSE-NEXT:    pand LCPI4_0, %xmm0 ## encoding: [0x66,0x0f,0xdb,0x05,A,A,A,A]
; SSE-NEXT:    ## fixup A - offset: 4, value: LCPI4_0, kind: FK_Data_4
; SSE-NEXT:    por %xmm2, %xmm0 ## encoding: [0x66,0x0f,0xeb,0xc2]
; SSE-NEXT:    retl ## encoding: [0xc3]
;
; AVX2-LABEL: test_x86_sse2_paddus_b_64:
; AVX2:       ## %bb.0:
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm2 = [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0]
; AVX2-NEXT:    ## encoding: [0xc5,0xf9,0x6f,0x15,A,A,A,A]
; AVX2-NEXT:    ## fixup A - offset: 4, value: LCPI4_0, kind: FK_Data_4
; AVX2-NEXT:    vpand %xmm2, %xmm0, %xmm3 ## encoding: [0xc5,0xf9,0xdb,0xda]
; AVX2-NEXT:    vpaddw %xmm1, %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0xfd,0xc1]
; AVX2-NEXT:    vpand %xmm2, %xmm0, %xmm1 ## encoding: [0xc5,0xf9,0xdb,0xca]
; AVX2-NEXT:    vpcmpgtw %xmm1, %xmm3, %xmm1 ## encoding: [0xc5,0xe1,0x65,0xc9]
; AVX2-NEXT:    vpblendvb %xmm1, %xmm2, %xmm0, %xmm0 ## encoding: [0xc4,0xe3,0x79,0x4c,0xc2,0x10]
; AVX2-NEXT:    retl ## encoding: [0xc3]
;
; SKX-LABEL: test_x86_sse2_paddus_b_64:
; SKX:       ## %bb.0:
; SKX-NEXT:    vmovdqa LCPI4_0, %xmm2 ## EVEX TO VEX Compression xmm2 = [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0]
; SKX-NEXT:    ## encoding: [0xc5,0xf9,0x6f,0x15,A,A,A,A]
; SKX-NEXT:    ## fixup A - offset: 4, value: LCPI4_0, kind: FK_Data_4
; SKX-NEXT:    vpand %xmm2, %xmm0, %xmm3 ## EVEX TO VEX Compression encoding: [0xc5,0xf9,0xdb,0xda]
; SKX-NEXT:    vpaddw %xmm1, %xmm0, %xmm0 ## EVEX TO VEX Compression encoding: [0xc5,0xf9,0xfd,0xc1]
; SKX-NEXT:    vpand %xmm2, %xmm0, %xmm1 ## EVEX TO VEX Compression encoding: [0xc5,0xf9,0xdb,0xca]
; SKX-NEXT:    vpcmpnleuw %xmm1, %xmm3, %k1 ## encoding: [0x62,0xf3,0xe5,0x08,0x3e,0xc9,0x06]
; SKX-NEXT:    vmovdqu16 LCPI4_0, %xmm0 {%k1} ## encoding: [0x62,0xf1,0xff,0x09,0x6f,0x05,A,A,A,A]
; SKX-NEXT:    ## fixup A - offset: 6, value: LCPI4_0, kind: FK_Data_4
; SKX-NEXT:    retl ## encoding: [0xc3]
  %1 = add <8 x i8> %a0, %a1
  %2 = icmp ugt <8 x i8> %a0, %1
  %3 = select <8 x i1> %2, <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, <8 x i8> %1
  ret <8 x i8> %3
}

define <4 x i16> @test_x86_sse2_paddus_w_64(<4 x i16> %a0, <4 x i16> %a1) {
; SSE-LABEL: test_x86_sse2_paddus_w_64:
; SSE:       ## %bb.0:
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [65535,0,65535,0,65535,0,65535,0]
; SSE-NEXT:    ## encoding: [0x66,0x0f,0x6f,0x15,A,A,A,A]
; SSE-NEXT:    ## fixup A - offset: 4, value: LCPI5_0, kind: FK_Data_4
; SSE-NEXT:    paddd %xmm0, %xmm1 ## encoding: [0x66,0x0f,0xfe,0xc8]
; SSE-NEXT:    pand %xmm2, %xmm0 ## encoding: [0x66,0x0f,0xdb,0xc2]
; SSE-NEXT:    pand %xmm1, %xmm2 ## encoding: [0x66,0x0f,0xdb,0xd1]
; SSE-NEXT:    pcmpgtd %xmm2, %xmm0 ## encoding: [0x66,0x0f,0x66,0xc2]
; SSE-NEXT:    movdqa %xmm0, %xmm2 ## encoding: [0x66,0x0f,0x6f,0xd0]
; SSE-NEXT:    pandn %xmm1, %xmm2 ## encoding: [0x66,0x0f,0xdf,0xd1]
; SSE-NEXT:    pand LCPI5_0, %xmm0 ## encoding: [0x66,0x0f,0xdb,0x05,A,A,A,A]
; SSE-NEXT:    ## fixup A - offset: 4, value: LCPI5_0, kind: FK_Data_4
; SSE-NEXT:    por %xmm2, %xmm0 ## encoding: [0x66,0x0f,0xeb,0xc2]
; SSE-NEXT:    retl ## encoding: [0xc3]
;
; AVX2-LABEL: test_x86_sse2_paddus_w_64:
; AVX2:       ## %bb.0:
; AVX2-NEXT:    vpxor %xmm2, %xmm2, %xmm2 ## encoding: [0xc5,0xe9,0xef,0xd2]
; AVX2-NEXT:    vpblendw $170, %xmm2, %xmm0, %xmm3 ## encoding: [0xc4,0xe3,0x79,0x0e,0xda,0xaa]
; AVX2-NEXT:    ## xmm3 = xmm0[0],xmm2[1],xmm0[2],xmm2[3],xmm0[4],xmm2[5],xmm0[6],xmm2[7]
; AVX2-NEXT:    vpaddd %xmm1, %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0xfe,0xc1]
; AVX2-NEXT:    vpblendw $170, %xmm2, %xmm0, %xmm1 ## encoding: [0xc4,0xe3,0x79,0x0e,0xca,0xaa]
; AVX2-NEXT:    ## xmm1 = xmm0[0],xmm2[1],xmm0[2],xmm2[3],xmm0[4],xmm2[5],xmm0[6],xmm2[7]
; AVX2-NEXT:    vpcmpgtd %xmm1, %xmm3, %xmm1 ## encoding: [0xc5,0xe1,0x66,0xc9]
; AVX2-NEXT:    vbroadcastss {{.*#+}} xmm2 = [65535,65535,65535,65535]
; AVX2-NEXT:    ## encoding: [0xc4,0xe2,0x79,0x18,0x15,A,A,A,A]
; AVX2-NEXT:    ## fixup A - offset: 5, value: LCPI5_0, kind: FK_Data_4
; AVX2-NEXT:    vblendvps %xmm1, %xmm2, %xmm0, %xmm0 ## encoding: [0xc4,0xe3,0x79,0x4a,0xc2,0x10]
; AVX2-NEXT:    retl ## encoding: [0xc3]
;
; SKX-LABEL: test_x86_sse2_paddus_w_64:
; SKX:       ## %bb.0:
; SKX-NEXT:    vpxor %xmm2, %xmm2, %xmm2 ## EVEX TO VEX Compression encoding: [0xc5,0xe9,0xef,0xd2]
; SKX-NEXT:    vpblendw $170, %xmm2, %xmm0, %xmm3 ## encoding: [0xc4,0xe3,0x79,0x0e,0xda,0xaa]
; SKX-NEXT:    ## xmm3 = xmm0[0],xmm2[1],xmm0[2],xmm2[3],xmm0[4],xmm2[5],xmm0[6],xmm2[7]
; SKX-NEXT:    vpaddd %xmm1, %xmm0, %xmm0 ## EVEX TO VEX Compression encoding: [0xc5,0xf9,0xfe,0xc1]
; SKX-NEXT:    vpblendw $170, %xmm2, %xmm0, %xmm1 ## encoding: [0xc4,0xe3,0x79,0x0e,0xca,0xaa]
; SKX-NEXT:    ## xmm1 = xmm0[0],xmm2[1],xmm0[2],xmm2[3],xmm0[4],xmm2[5],xmm0[6],xmm2[7]
; SKX-NEXT:    vpcmpnleud %xmm1, %xmm3, %k1 ## encoding: [0x62,0xf3,0x65,0x08,0x1e,0xc9,0x06]
; SKX-NEXT:    vpbroadcastd LCPI5_0, %xmm0 {%k1} ## encoding: [0x62,0xf2,0x7d,0x09,0x58,0x05,A,A,A,A]
; SKX-NEXT:    ## fixup A - offset: 6, value: LCPI5_0, kind: FK_Data_4
; SKX-NEXT:    retl ## encoding: [0xc3]
  %1 = add <4 x i16> %a0, %a1
  %2 = icmp ugt <4 x i16> %a0, %1
  %3 = select <4 x i1> %2, <4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>, <4 x i16> %1
  ret <4 x i16> %3
}

define <8 x i8> @test_x86_sse2_psubus_b_64(<8 x i8> %a0, <8 x i8> %a1) {
; SSE-LABEL: test_x86_sse2_psubus_b_64:
; SSE:       ## %bb.0:
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0]
; SSE-NEXT:    ## encoding: [0x66,0x0f,0x6f,0x15,A,A,A,A]
; SSE-NEXT:    ## fixup A - offset: 4, value: LCPI6_0, kind: FK_Data_4
; SSE-NEXT:    movdqa %xmm1, %xmm3 ## encoding: [0x66,0x0f,0x6f,0xd9]
; SSE-NEXT:    pand %xmm2, %xmm3 ## encoding: [0x66,0x0f,0xdb,0xda]
; SSE-NEXT:    pand %xmm2, %xmm0 ## encoding: [0x66,0x0f,0xdb,0xc2]
; SSE-NEXT:    pmaxsw %xmm3, %xmm0 ## encoding: [0x66,0x0f,0xee,0xc3]
; SSE-NEXT:    psubw %xmm1, %xmm0 ## encoding: [0x66,0x0f,0xf9,0xc1]
; SSE-NEXT:    retl ## encoding: [0xc3]
;
; AVX2-LABEL: test_x86_sse2_psubus_b_64:
; AVX2:       ## %bb.0:
; AVX2-NEXT:    vmovdqa {{.*#+}} xmm2 = [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0]
; AVX2-NEXT:    ## encoding: [0xc5,0xf9,0x6f,0x15,A,A,A,A]
; AVX2-NEXT:    ## fixup A - offset: 4, value: LCPI6_0, kind: FK_Data_4
; AVX2-NEXT:    vpand %xmm2, %xmm1, %xmm3 ## encoding: [0xc5,0xf1,0xdb,0xda]
; AVX2-NEXT:    vpand %xmm2, %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0xdb,0xc2]
; AVX2-NEXT:    vpmaxuw %xmm3, %xmm0, %xmm0 ## encoding: [0xc4,0xe2,0x79,0x3e,0xc3]
; AVX2-NEXT:    vpsubw %xmm1, %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0xf9,0xc1]
; AVX2-NEXT:    retl ## encoding: [0xc3]
;
; SKX-LABEL: test_x86_sse2_psubus_b_64:
; SKX:       ## %bb.0:
; SKX-NEXT:    vmovdqa LCPI6_0, %xmm2 ## EVEX TO VEX Compression xmm2 = [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0]
; SKX-NEXT:    ## encoding: [0xc5,0xf9,0x6f,0x15,A,A,A,A]
; SKX-NEXT:    ## fixup A - offset: 4, value: LCPI6_0, kind: FK_Data_4
; SKX-NEXT:    vpand %xmm2, %xmm1, %xmm3 ## EVEX TO VEX Compression encoding: [0xc5,0xf1,0xdb,0xda]
; SKX-NEXT:    vpand %xmm2, %xmm0, %xmm0 ## EVEX TO VEX Compression encoding: [0xc5,0xf9,0xdb,0xc2]
; SKX-NEXT:    vpmaxuw %xmm3, %xmm0, %xmm0 ## EVEX TO VEX Compression encoding: [0xc4,0xe2,0x79,0x3e,0xc3]
; SKX-NEXT:    vpsubw %xmm1, %xmm0, %xmm0 ## EVEX TO VEX Compression encoding: [0xc5,0xf9,0xf9,0xc1]
; SKX-NEXT:    retl ## encoding: [0xc3]
  %cmp = icmp ugt <8 x i8> %a0, %a1
  %sel = select <8 x i1> %cmp, <8 x i8> %a0, <8 x i8> %a1
  %sub = sub <8 x i8> %sel, %a1
  ret <8 x i8> %sub
}

define <4 x i16> @test_x86_sse2_psubus_w_64(<4 x i16> %a0, <4 x i16> %a1) {
; SSE-LABEL: test_x86_sse2_psubus_w_64:
; SSE:       ## %bb.0:
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [65535,0,65535,0,65535,0,65535,0]
; SSE-NEXT:    ## encoding: [0x66,0x0f,0x6f,0x15,A,A,A,A]
; SSE-NEXT:    ## fixup A - offset: 4, value: LCPI7_0, kind: FK_Data_4
; SSE-NEXT:    movdqa %xmm1, %xmm3 ## encoding: [0x66,0x0f,0x6f,0xd9]
; SSE-NEXT:    pand %xmm2, %xmm3 ## encoding: [0x66,0x0f,0xdb,0xda]
; SSE-NEXT:    pand %xmm2, %xmm0 ## encoding: [0x66,0x0f,0xdb,0xc2]
; SSE-NEXT:    movdqa %xmm0, %xmm2 ## encoding: [0x66,0x0f,0x6f,0xd0]
; SSE-NEXT:    pcmpgtd %xmm3, %xmm2 ## encoding: [0x66,0x0f,0x66,0xd3]
; SSE-NEXT:    pand %xmm2, %xmm0 ## encoding: [0x66,0x0f,0xdb,0xc2]
; SSE-NEXT:    pandn %xmm3, %xmm2 ## encoding: [0x66,0x0f,0xdf,0xd3]
; SSE-NEXT:    por %xmm0, %xmm2 ## encoding: [0x66,0x0f,0xeb,0xd0]
; SSE-NEXT:    psubd %xmm1, %xmm2 ## encoding: [0x66,0x0f,0xfa,0xd1]
; SSE-NEXT:    movdqa %xmm2, %xmm0 ## encoding: [0x66,0x0f,0x6f,0xc2]
; SSE-NEXT:    retl ## encoding: [0xc3]
;
; AVX2-LABEL: test_x86_sse2_psubus_w_64:
; AVX2:       ## %bb.0:
; AVX2-NEXT:    vpxor %xmm2, %xmm2, %xmm2 ## encoding: [0xc5,0xe9,0xef,0xd2]
; AVX2-NEXT:    vpblendw $170, %xmm2, %xmm1, %xmm3 ## encoding: [0xc4,0xe3,0x71,0x0e,0xda,0xaa]
; AVX2-NEXT:    ## xmm3 = xmm1[0],xmm2[1],xmm1[2],xmm2[3],xmm1[4],xmm2[5],xmm1[6],xmm2[7]
; AVX2-NEXT:    vpblendw $170, %xmm2, %xmm0, %xmm0 ## encoding: [0xc4,0xe3,0x79,0x0e,0xc2,0xaa]
; AVX2-NEXT:    ## xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3],xmm0[4],xmm2[5],xmm0[6],xmm2[7]
; AVX2-NEXT:    vpmaxud %xmm3, %xmm0, %xmm0 ## encoding: [0xc4,0xe2,0x79,0x3f,0xc3]
; AVX2-NEXT:    vpsubd %xmm1, %xmm0, %xmm0 ## encoding: [0xc5,0xf9,0xfa,0xc1]
; AVX2-NEXT:    retl ## encoding: [0xc3]
;
; SKX-LABEL: test_x86_sse2_psubus_w_64:
; SKX:       ## %bb.0:
; SKX-NEXT:    vpxor %xmm2, %xmm2, %xmm2 ## EVEX TO VEX Compression encoding: [0xc5,0xe9,0xef,0xd2]
; SKX-NEXT:    vpblendw $170, %xmm2, %xmm1, %xmm3 ## encoding: [0xc4,0xe3,0x71,0x0e,0xda,0xaa]
; SKX-NEXT:    ## xmm3 = xmm1[0],xmm2[1],xmm1[2],xmm2[3],xmm1[4],xmm2[5],xmm1[6],xmm2[7]
; SKX-NEXT:    vpblendw $170, %xmm2, %xmm0, %xmm0 ## encoding: [0xc4,0xe3,0x79,0x0e,0xc2,0xaa]
; SKX-NEXT:    ## xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3],xmm0[4],xmm2[5],xmm0[6],xmm2[7]
; SKX-NEXT:    vpmaxud %xmm3, %xmm0, %xmm0 ## EVEX TO VEX Compression encoding: [0xc4,0xe2,0x79,0x3f,0xc3]
; SKX-NEXT:    vpsubd %xmm1, %xmm0, %xmm0 ## EVEX TO VEX Compression encoding: [0xc5,0xf9,0xfa,0xc1]
; SKX-NEXT:    retl ## encoding: [0xc3]
  %cmp = icmp ugt <4 x i16> %a0, %a1
  %sel = select <4 x i1> %cmp, <4 x i16> %a0, <4 x i16> %a1
  %sub = sub <4 x i16> %sel, %a1
  ret <4 x i16> %sub
}

