; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse2 | FileCheck %s --check-prefixes=SSE,SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+xop  | FileCheck %s --check-prefixes=XOP
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx  | FileCheck %s --check-prefixes=AVX,AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 | FileCheck %s --check-prefixes=AVX,AVX2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx512f | FileCheck %s --check-prefixes=AVX,AVX512F

;
; 128-bit vectors
;

define <2 x i64> @bitselect_v2i64_rr(<2 x i64>, <2 x i64>) {
; SSE-LABEL: bitselect_v2i64_rr:
; SSE:       # %bb.0:
; SSE-NEXT:    andps {{.*}}(%rip), %xmm0
; SSE-NEXT:    andps {{.*}}(%rip), %xmm1
; SSE-NEXT:    orps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; XOP-LABEL: bitselect_v2i64_rr:
; XOP:       # %bb.0:
; XOP-NEXT:    vandps {{.*}}(%rip), %xmm0, %xmm0
; XOP-NEXT:    vandps {{.*}}(%rip), %xmm1, %xmm1
; XOP-NEXT:    vorps %xmm0, %xmm1, %xmm0
; XOP-NEXT:    retq
;
; AVX-LABEL: bitselect_v2i64_rr:
; AVX:       # %bb.0:
; AVX-NEXT:    vandps {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vandps {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:    vorps %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
  %3 = and <2 x i64> %0, <i64 4294967296, i64 12884901890>
  %4 = and <2 x i64> %1, <i64 -4294967297, i64 -12884901891>
  %5 = or <2 x i64> %4, %3
  ret <2 x i64> %5
}

define <2 x i64> @bitselect_v2i64_rm(<2 x i64>, <2 x i64>* nocapture readonly) {
; SSE-LABEL: bitselect_v2i64_rm:
; SSE:       # %bb.0:
; SSE-NEXT:    movaps (%rdi), %xmm1
; SSE-NEXT:    andps {{.*}}(%rip), %xmm0
; SSE-NEXT:    andps {{.*}}(%rip), %xmm1
; SSE-NEXT:    orps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; XOP-LABEL: bitselect_v2i64_rm:
; XOP:       # %bb.0:
; XOP-NEXT:    vmovaps (%rdi), %xmm1
; XOP-NEXT:    vandps {{.*}}(%rip), %xmm0, %xmm0
; XOP-NEXT:    vandps {{.*}}(%rip), %xmm1, %xmm1
; XOP-NEXT:    vorps %xmm0, %xmm1, %xmm0
; XOP-NEXT:    retq
;
; AVX-LABEL: bitselect_v2i64_rm:
; AVX:       # %bb.0:
; AVX-NEXT:    vmovaps (%rdi), %xmm1
; AVX-NEXT:    vandps {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vandps {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:    vorps %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
  %3 = load <2 x i64>, <2 x i64>* %1
  %4 = and <2 x i64> %0, <i64 8589934593, i64 3>
  %5 = and <2 x i64> %3, <i64 -8589934594, i64 -4>
  %6 = or <2 x i64> %5, %4
  ret <2 x i64> %6
}

define <2 x i64> @bitselect_v2i64_mr(<2 x i64>* nocapture readonly, <2 x i64>) {
; SSE-LABEL: bitselect_v2i64_mr:
; SSE:       # %bb.0:
; SSE-NEXT:    movaps (%rdi), %xmm1
; SSE-NEXT:    andps {{.*}}(%rip), %xmm1
; SSE-NEXT:    andps {{.*}}(%rip), %xmm0
; SSE-NEXT:    orps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; XOP-LABEL: bitselect_v2i64_mr:
; XOP:       # %bb.0:
; XOP-NEXT:    vmovaps (%rdi), %xmm1
; XOP-NEXT:    vandps {{.*}}(%rip), %xmm1, %xmm1
; XOP-NEXT:    vandps {{.*}}(%rip), %xmm0, %xmm0
; XOP-NEXT:    vorps %xmm0, %xmm1, %xmm0
; XOP-NEXT:    retq
;
; AVX-LABEL: bitselect_v2i64_mr:
; AVX:       # %bb.0:
; AVX-NEXT:    vmovaps (%rdi), %xmm1
; AVX-NEXT:    vandps {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:    vandps {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vorps %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
  %3 = load <2 x i64>, <2 x i64>* %0
  %4 = and <2 x i64> %3, <i64 12884901890, i64 4294967296>
  %5 = and <2 x i64> %1, <i64 -12884901891, i64 -4294967297>
  %6 = or <2 x i64> %4, %5
  ret <2 x i64> %6
}

define <2 x i64> @bitselect_v2i64_mm(<2 x i64>* nocapture readonly, <2 x i64>* nocapture readonly) {
; SSE-LABEL: bitselect_v2i64_mm:
; SSE:       # %bb.0:
; SSE-NEXT:    movaps (%rdi), %xmm1
; SSE-NEXT:    movaps (%rsi), %xmm0
; SSE-NEXT:    andps {{.*}}(%rip), %xmm1
; SSE-NEXT:    andps {{.*}}(%rip), %xmm0
; SSE-NEXT:    orps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; XOP-LABEL: bitselect_v2i64_mm:
; XOP:       # %bb.0:
; XOP-NEXT:    vmovaps (%rdi), %xmm0
; XOP-NEXT:    vmovaps (%rsi), %xmm1
; XOP-NEXT:    vandps {{.*}}(%rip), %xmm0, %xmm0
; XOP-NEXT:    vandps {{.*}}(%rip), %xmm1, %xmm1
; XOP-NEXT:    vorps %xmm0, %xmm1, %xmm0
; XOP-NEXT:    retq
;
; AVX-LABEL: bitselect_v2i64_mm:
; AVX:       # %bb.0:
; AVX-NEXT:    vmovaps (%rdi), %xmm0
; AVX-NEXT:    vmovaps (%rsi), %xmm1
; AVX-NEXT:    vandps {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vandps {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:    vorps %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
  %3 = load <2 x i64>, <2 x i64>* %0
  %4 = load <2 x i64>, <2 x i64>* %1
  %5 = and <2 x i64> %3, <i64 3, i64 8589934593>
  %6 = and <2 x i64> %4, <i64 -4, i64 -8589934594>
  %7 = or <2 x i64> %6, %5
  ret <2 x i64> %7
}

;
; 256-bit vectors
;

define <4 x i64> @bitselect_v4i64_rr(<4 x i64>, <4 x i64>) {
; SSE-LABEL: bitselect_v4i64_rr:
; SSE:       # %bb.0:
; SSE-NEXT:    andps {{.*}}(%rip), %xmm1
; SSE-NEXT:    andps {{.*}}(%rip), %xmm0
; SSE-NEXT:    andps {{.*}}(%rip), %xmm3
; SSE-NEXT:    orps %xmm3, %xmm1
; SSE-NEXT:    andps {{.*}}(%rip), %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; XOP-LABEL: bitselect_v4i64_rr:
; XOP:       # %bb.0:
; XOP-NEXT:    vandps {{.*}}(%rip), %ymm0, %ymm0
; XOP-NEXT:    vandps {{.*}}(%rip), %ymm1, %ymm1
; XOP-NEXT:    vorps %ymm0, %ymm1, %ymm0
; XOP-NEXT:    retq
;
; AVX-LABEL: bitselect_v4i64_rr:
; AVX:       # %bb.0:
; AVX-NEXT:    vandps {{.*}}(%rip), %ymm0, %ymm0
; AVX-NEXT:    vandps {{.*}}(%rip), %ymm1, %ymm1
; AVX-NEXT:    vorps %ymm0, %ymm1, %ymm0
; AVX-NEXT:    retq
  %3 = and <4 x i64> %0, <i64 4294967296, i64 12884901890, i64 12884901890, i64 12884901890>
  %4 = and <4 x i64> %1, <i64 -4294967297, i64 -12884901891, i64 -12884901891, i64 -12884901891>
  %5 = or <4 x i64> %4, %3
  ret <4 x i64> %5
}

define <4 x i64> @bitselect_v4i64_rm(<4 x i64>, <4 x i64>* nocapture readonly) {
; SSE-LABEL: bitselect_v4i64_rm:
; SSE:       # %bb.0:
; SSE-NEXT:    movaps {{.*#+}} xmm2 = [8589934593,3]
; SSE-NEXT:    andps %xmm2, %xmm1
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    movaps {{.*#+}} xmm2 = [18446744065119617022,18446744073709551612]
; SSE-NEXT:    movaps 16(%rdi), %xmm3
; SSE-NEXT:    andps %xmm2, %xmm3
; SSE-NEXT:    orps %xmm3, %xmm1
; SSE-NEXT:    andps (%rdi), %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; XOP-LABEL: bitselect_v4i64_rm:
; XOP:       # %bb.0:
; XOP-NEXT:    vmovaps (%rdi), %ymm1
; XOP-NEXT:    vandps {{.*}}(%rip), %ymm0, %ymm0
; XOP-NEXT:    vandps {{.*}}(%rip), %ymm1, %ymm1
; XOP-NEXT:    vorps %ymm0, %ymm1, %ymm0
; XOP-NEXT:    retq
;
; AVX-LABEL: bitselect_v4i64_rm:
; AVX:       # %bb.0:
; AVX-NEXT:    vmovaps (%rdi), %ymm1
; AVX-NEXT:    vandps {{.*}}(%rip), %ymm0, %ymm0
; AVX-NEXT:    vandps {{.*}}(%rip), %ymm1, %ymm1
; AVX-NEXT:    vorps %ymm0, %ymm1, %ymm0
; AVX-NEXT:    retq
  %3 = load <4 x i64>, <4 x i64>* %1
  %4 = and <4 x i64> %0, <i64 8589934593, i64 3, i64 8589934593, i64 3>
  %5 = and <4 x i64> %3, <i64 -8589934594, i64 -4, i64 -8589934594, i64 -4>
  %6 = or <4 x i64> %5, %4
  ret <4 x i64> %6
}

define <4 x i64> @bitselect_v4i64_mr(<4 x i64>* nocapture readonly, <4 x i64>) {
; SSE-LABEL: bitselect_v4i64_mr:
; SSE:       # %bb.0:
; SSE-NEXT:    movaps {{.*#+}} xmm2 = [12884901890,4294967296]
; SSE-NEXT:    movaps 16(%rdi), %xmm3
; SSE-NEXT:    andps %xmm2, %xmm3
; SSE-NEXT:    andps (%rdi), %xmm2
; SSE-NEXT:    movaps {{.*#+}} xmm4 = [18446744060824649725,18446744069414584319]
; SSE-NEXT:    andps %xmm4, %xmm1
; SSE-NEXT:    orps %xmm3, %xmm1
; SSE-NEXT:    andps %xmm4, %xmm0
; SSE-NEXT:    orps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; XOP-LABEL: bitselect_v4i64_mr:
; XOP:       # %bb.0:
; XOP-NEXT:    vmovaps (%rdi), %ymm1
; XOP-NEXT:    vandps {{.*}}(%rip), %ymm1, %ymm1
; XOP-NEXT:    vandps {{.*}}(%rip), %ymm0, %ymm0
; XOP-NEXT:    vorps %ymm0, %ymm1, %ymm0
; XOP-NEXT:    retq
;
; AVX-LABEL: bitselect_v4i64_mr:
; AVX:       # %bb.0:
; AVX-NEXT:    vmovaps (%rdi), %ymm1
; AVX-NEXT:    vandps {{.*}}(%rip), %ymm1, %ymm1
; AVX-NEXT:    vandps {{.*}}(%rip), %ymm0, %ymm0
; AVX-NEXT:    vorps %ymm0, %ymm1, %ymm0
; AVX-NEXT:    retq
  %3 = load <4 x i64>, <4 x i64>* %0
  %4 = and <4 x i64> %3, <i64 12884901890, i64 4294967296, i64 12884901890, i64 4294967296>
  %5 = and <4 x i64> %1, <i64 -12884901891, i64 -4294967297, i64 -12884901891, i64 -4294967297>
  %6 = or <4 x i64> %4, %5
  ret <4 x i64> %6
}

define <4 x i64> @bitselect_v4i64_mm(<4 x i64>* nocapture readonly, <4 x i64>* nocapture readonly) {
; SSE-LABEL: bitselect_v4i64_mm:
; SSE:       # %bb.0:
; SSE-NEXT:    movaps {{.*#+}} xmm2 = [3,8589934593]
; SSE-NEXT:    movaps 16(%rdi), %xmm3
; SSE-NEXT:    andps %xmm2, %xmm3
; SSE-NEXT:    andps (%rdi), %xmm2
; SSE-NEXT:    movaps {{.*#+}} xmm0 = [18446744073709551612,18446744065119617022]
; SSE-NEXT:    movaps 16(%rsi), %xmm1
; SSE-NEXT:    andps %xmm0, %xmm1
; SSE-NEXT:    orps %xmm3, %xmm1
; SSE-NEXT:    andps (%rsi), %xmm0
; SSE-NEXT:    orps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; XOP-LABEL: bitselect_v4i64_mm:
; XOP:       # %bb.0:
; XOP-NEXT:    vmovaps (%rdi), %ymm0
; XOP-NEXT:    vmovaps (%rsi), %ymm1
; XOP-NEXT:    vandps {{.*}}(%rip), %ymm0, %ymm0
; XOP-NEXT:    vandps {{.*}}(%rip), %ymm1, %ymm1
; XOP-NEXT:    vorps %ymm0, %ymm1, %ymm0
; XOP-NEXT:    retq
;
; AVX-LABEL: bitselect_v4i64_mm:
; AVX:       # %bb.0:
; AVX-NEXT:    vmovaps (%rdi), %ymm0
; AVX-NEXT:    vmovaps (%rsi), %ymm1
; AVX-NEXT:    vandps {{.*}}(%rip), %ymm0, %ymm0
; AVX-NEXT:    vandps {{.*}}(%rip), %ymm1, %ymm1
; AVX-NEXT:    vorps %ymm0, %ymm1, %ymm0
; AVX-NEXT:    retq
  %3 = load <4 x i64>, <4 x i64>* %0
  %4 = load <4 x i64>, <4 x i64>* %1
  %5 = and <4 x i64> %3, <i64 3, i64 8589934593, i64 3, i64 8589934593>
  %6 = and <4 x i64> %4, <i64 -4, i64 -8589934594, i64 -4, i64 -8589934594>
  %7 = or <4 x i64> %6, %5
  ret <4 x i64> %7
}

;
; 512-bit vectors
;

define <8 x i64> @bitselect_v8i64_rr(<8 x i64>, <8 x i64>) {
; SSE-LABEL: bitselect_v8i64_rr:
; SSE:       # %bb.0:
; SSE-NEXT:    movaps {{.*#+}} xmm8 = [12884901890,12884901890]
; SSE-NEXT:    andps %xmm8, %xmm3
; SSE-NEXT:    movaps {{.*#+}} xmm9 = [4294967296,12884901890]
; SSE-NEXT:    andps %xmm9, %xmm2
; SSE-NEXT:    andps %xmm8, %xmm1
; SSE-NEXT:    andps %xmm9, %xmm0
; SSE-NEXT:    movaps {{.*#+}} xmm8 = [18446744060824649725,18446744060824649725]
; SSE-NEXT:    andps %xmm8, %xmm7
; SSE-NEXT:    orps %xmm7, %xmm3
; SSE-NEXT:    movaps {{.*#+}} xmm7 = [18446744069414584319,18446744060824649725]
; SSE-NEXT:    andps %xmm7, %xmm6
; SSE-NEXT:    orps %xmm6, %xmm2
; SSE-NEXT:    andps %xmm5, %xmm8
; SSE-NEXT:    orps %xmm8, %xmm1
; SSE-NEXT:    andps %xmm4, %xmm7
; SSE-NEXT:    orps %xmm7, %xmm0
; SSE-NEXT:    retq
;
; XOP-LABEL: bitselect_v8i64_rr:
; XOP:       # %bb.0:
; XOP-NEXT:    vmovaps {{.*#+}} ymm4 = [4294967296,12884901890,12884901890,12884901890]
; XOP-NEXT:    vandps %ymm4, %ymm1, %ymm1
; XOP-NEXT:    vandps %ymm4, %ymm0, %ymm0
; XOP-NEXT:    vmovaps {{.*#+}} ymm4 = [18446744069414584319,18446744060824649725,18446744060824649725,18446744060824649725]
; XOP-NEXT:    vandps %ymm4, %ymm3, %ymm3
; XOP-NEXT:    vorps %ymm1, %ymm3, %ymm1
; XOP-NEXT:    vandps %ymm4, %ymm2, %ymm2
; XOP-NEXT:    vorps %ymm0, %ymm2, %ymm0
; XOP-NEXT:    retq
;
; AVX1-LABEL: bitselect_v8i64_rr:
; AVX1:       # %bb.0:
; AVX1-NEXT:    vmovaps {{.*#+}} ymm4 = [4294967296,12884901890,12884901890,12884901890]
; AVX1-NEXT:    vandps %ymm4, %ymm1, %ymm1
; AVX1-NEXT:    vandps %ymm4, %ymm0, %ymm0
; AVX1-NEXT:    vmovaps {{.*#+}} ymm4 = [18446744069414584319,18446744060824649725,18446744060824649725,18446744060824649725]
; AVX1-NEXT:    vandps %ymm4, %ymm3, %ymm3
; AVX1-NEXT:    vorps %ymm1, %ymm3, %ymm1
; AVX1-NEXT:    vandps %ymm4, %ymm2, %ymm2
; AVX1-NEXT:    vorps %ymm0, %ymm2, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: bitselect_v8i64_rr:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vmovaps {{.*#+}} ymm4 = [4294967296,12884901890,12884901890,12884901890]
; AVX2-NEXT:    vandps %ymm4, %ymm1, %ymm1
; AVX2-NEXT:    vandps %ymm4, %ymm0, %ymm0
; AVX2-NEXT:    vmovaps {{.*#+}} ymm4 = [18446744069414584319,18446744060824649725,18446744060824649725,18446744060824649725]
; AVX2-NEXT:    vandps %ymm4, %ymm3, %ymm3
; AVX2-NEXT:    vorps %ymm1, %ymm3, %ymm1
; AVX2-NEXT:    vandps %ymm4, %ymm2, %ymm2
; AVX2-NEXT:    vorps %ymm0, %ymm2, %ymm0
; AVX2-NEXT:    retq
;
; AVX512F-LABEL: bitselect_v8i64_rr:
; AVX512F:       # %bb.0:
; AVX512F-NEXT:    vpandq {{.*}}(%rip), %zmm0, %zmm0
; AVX512F-NEXT:    vpandq {{.*}}(%rip), %zmm1, %zmm1
; AVX512F-NEXT:    vporq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
  %3 = and <8 x i64> %0, <i64 4294967296, i64 12884901890, i64 12884901890, i64 12884901890, i64 4294967296, i64 12884901890, i64 12884901890, i64 12884901890>
  %4 = and <8 x i64> %1, <i64 -4294967297, i64 -12884901891, i64 -12884901891, i64 -12884901891, i64 -4294967297, i64 -12884901891, i64 -12884901891, i64 -12884901891>
  %5 = or <8 x i64> %4, %3
  ret <8 x i64> %5
}

define <8 x i64> @bitselect_v8i64_rm(<8 x i64>, <8 x i64>* nocapture readonly) {
; SSE-LABEL: bitselect_v8i64_rm:
; SSE:       # %bb.0:
; SSE-NEXT:    movaps {{.*#+}} xmm4 = [8589934593,3]
; SSE-NEXT:    andps %xmm4, %xmm3
; SSE-NEXT:    andps %xmm4, %xmm2
; SSE-NEXT:    andps %xmm4, %xmm1
; SSE-NEXT:    andps %xmm4, %xmm0
; SSE-NEXT:    movaps {{.*#+}} xmm4 = [18446744065119617022,18446744073709551612]
; SSE-NEXT:    movaps 48(%rdi), %xmm5
; SSE-NEXT:    andps %xmm4, %xmm5
; SSE-NEXT:    orps %xmm5, %xmm3
; SSE-NEXT:    movaps 32(%rdi), %xmm5
; SSE-NEXT:    andps %xmm4, %xmm5
; SSE-NEXT:    orps %xmm5, %xmm2
; SSE-NEXT:    movaps 16(%rdi), %xmm5
; SSE-NEXT:    andps %xmm4, %xmm5
; SSE-NEXT:    orps %xmm5, %xmm1
; SSE-NEXT:    andps (%rdi), %xmm4
; SSE-NEXT:    orps %xmm4, %xmm0
; SSE-NEXT:    retq
;
; XOP-LABEL: bitselect_v8i64_rm:
; XOP:       # %bb.0:
; XOP-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = [8589934593,3,8589934593,3]
; XOP-NEXT:    # ymm2 = mem[0,1,0,1]
; XOP-NEXT:    vandps %ymm2, %ymm1, %ymm1
; XOP-NEXT:    vandps %ymm2, %ymm0, %ymm0
; XOP-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = [18446744065119617022,18446744073709551612,18446744065119617022,18446744073709551612]
; XOP-NEXT:    # ymm2 = mem[0,1,0,1]
; XOP-NEXT:    vandps 32(%rdi), %ymm2, %ymm3
; XOP-NEXT:    vorps %ymm1, %ymm3, %ymm1
; XOP-NEXT:    vandps (%rdi), %ymm2, %ymm2
; XOP-NEXT:    vorps %ymm0, %ymm2, %ymm0
; XOP-NEXT:    retq
;
; AVX1-LABEL: bitselect_v8i64_rm:
; AVX1:       # %bb.0:
; AVX1-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = [8589934593,3,8589934593,3]
; AVX1-NEXT:    # ymm2 = mem[0,1,0,1]
; AVX1-NEXT:    vandps %ymm2, %ymm1, %ymm1
; AVX1-NEXT:    vandps %ymm2, %ymm0, %ymm0
; AVX1-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = [18446744065119617022,18446744073709551612,18446744065119617022,18446744073709551612]
; AVX1-NEXT:    # ymm2 = mem[0,1,0,1]
; AVX1-NEXT:    vandps 32(%rdi), %ymm2, %ymm3
; AVX1-NEXT:    vorps %ymm1, %ymm3, %ymm1
; AVX1-NEXT:    vandps (%rdi), %ymm2, %ymm2
; AVX1-NEXT:    vorps %ymm0, %ymm2, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: bitselect_v8i64_rm:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = [8589934593,3,8589934593,3]
; AVX2-NEXT:    # ymm2 = mem[0,1,0,1]
; AVX2-NEXT:    vandps %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vandps %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = [18446744065119617022,18446744073709551612,18446744065119617022,18446744073709551612]
; AVX2-NEXT:    # ymm2 = mem[0,1,0,1]
; AVX2-NEXT:    vandps 32(%rdi), %ymm2, %ymm3
; AVX2-NEXT:    vorps %ymm1, %ymm3, %ymm1
; AVX2-NEXT:    vandps (%rdi), %ymm2, %ymm2
; AVX2-NEXT:    vorps %ymm0, %ymm2, %ymm0
; AVX2-NEXT:    retq
;
; AVX512F-LABEL: bitselect_v8i64_rm:
; AVX512F:       # %bb.0:
; AVX512F-NEXT:    vmovdqa64 (%rdi), %zmm1
; AVX512F-NEXT:    vpandq {{.*}}(%rip), %zmm0, %zmm0
; AVX512F-NEXT:    vpandq {{.*}}(%rip), %zmm1, %zmm1
; AVX512F-NEXT:    vporq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
  %3 = load <8 x i64>, <8 x i64>* %1
  %4 = and <8 x i64> %0, <i64 8589934593, i64 3, i64 8589934593, i64 3, i64 8589934593, i64 3, i64 8589934593, i64 3>
  %5 = and <8 x i64> %3, <i64 -8589934594, i64 -4, i64 -8589934594, i64 -4, i64 -8589934594, i64 -4, i64 -8589934594, i64 -4>
  %6 = or <8 x i64> %5, %4
  ret <8 x i64> %6
}

define <8 x i64> @bitselect_v8i64_mr(<8 x i64>* nocapture readonly, <8 x i64>) {
; SSE-LABEL: bitselect_v8i64_mr:
; SSE:       # %bb.0:
; SSE-NEXT:    movaps {{.*#+}} xmm4 = [12884901890,4294967296]
; SSE-NEXT:    movaps 48(%rdi), %xmm5
; SSE-NEXT:    andps %xmm4, %xmm5
; SSE-NEXT:    movaps 32(%rdi), %xmm6
; SSE-NEXT:    andps %xmm4, %xmm6
; SSE-NEXT:    movaps 16(%rdi), %xmm7
; SSE-NEXT:    andps %xmm4, %xmm7
; SSE-NEXT:    andps (%rdi), %xmm4
; SSE-NEXT:    movaps {{.*#+}} xmm8 = [18446744060824649725,18446744069414584319]
; SSE-NEXT:    andps %xmm8, %xmm3
; SSE-NEXT:    orps %xmm5, %xmm3
; SSE-NEXT:    andps %xmm8, %xmm2
; SSE-NEXT:    orps %xmm6, %xmm2
; SSE-NEXT:    andps %xmm8, %xmm1
; SSE-NEXT:    orps %xmm7, %xmm1
; SSE-NEXT:    andps %xmm8, %xmm0
; SSE-NEXT:    orps %xmm4, %xmm0
; SSE-NEXT:    retq
;
; XOP-LABEL: bitselect_v8i64_mr:
; XOP:       # %bb.0:
; XOP-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = [12884901890,4294967296,12884901890,4294967296]
; XOP-NEXT:    # ymm2 = mem[0,1,0,1]
; XOP-NEXT:    vandps 32(%rdi), %ymm2, %ymm3
; XOP-NEXT:    vandps (%rdi), %ymm2, %ymm2
; XOP-NEXT:    vbroadcastf128 {{.*#+}} ymm4 = [18446744060824649725,18446744069414584319,18446744060824649725,18446744069414584319]
; XOP-NEXT:    # ymm4 = mem[0,1,0,1]
; XOP-NEXT:    vandps %ymm4, %ymm1, %ymm1
; XOP-NEXT:    vorps %ymm1, %ymm3, %ymm1
; XOP-NEXT:    vandps %ymm4, %ymm0, %ymm0
; XOP-NEXT:    vorps %ymm0, %ymm2, %ymm0
; XOP-NEXT:    retq
;
; AVX1-LABEL: bitselect_v8i64_mr:
; AVX1:       # %bb.0:
; AVX1-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = [12884901890,4294967296,12884901890,4294967296]
; AVX1-NEXT:    # ymm2 = mem[0,1,0,1]
; AVX1-NEXT:    vandps 32(%rdi), %ymm2, %ymm3
; AVX1-NEXT:    vandps (%rdi), %ymm2, %ymm2
; AVX1-NEXT:    vbroadcastf128 {{.*#+}} ymm4 = [18446744060824649725,18446744069414584319,18446744060824649725,18446744069414584319]
; AVX1-NEXT:    # ymm4 = mem[0,1,0,1]
; AVX1-NEXT:    vandps %ymm4, %ymm1, %ymm1
; AVX1-NEXT:    vorps %ymm1, %ymm3, %ymm1
; AVX1-NEXT:    vandps %ymm4, %ymm0, %ymm0
; AVX1-NEXT:    vorps %ymm0, %ymm2, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: bitselect_v8i64_mr:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = [12884901890,4294967296,12884901890,4294967296]
; AVX2-NEXT:    # ymm2 = mem[0,1,0,1]
; AVX2-NEXT:    vandps 32(%rdi), %ymm2, %ymm3
; AVX2-NEXT:    vandps (%rdi), %ymm2, %ymm2
; AVX2-NEXT:    vbroadcastf128 {{.*#+}} ymm4 = [18446744060824649725,18446744069414584319,18446744060824649725,18446744069414584319]
; AVX2-NEXT:    # ymm4 = mem[0,1,0,1]
; AVX2-NEXT:    vandps %ymm4, %ymm1, %ymm1
; AVX2-NEXT:    vorps %ymm1, %ymm3, %ymm1
; AVX2-NEXT:    vandps %ymm4, %ymm0, %ymm0
; AVX2-NEXT:    vorps %ymm0, %ymm2, %ymm0
; AVX2-NEXT:    retq
;
; AVX512F-LABEL: bitselect_v8i64_mr:
; AVX512F:       # %bb.0:
; AVX512F-NEXT:    vmovdqa64 (%rdi), %zmm1
; AVX512F-NEXT:    vpandq {{.*}}(%rip), %zmm1, %zmm1
; AVX512F-NEXT:    vpandq {{.*}}(%rip), %zmm0, %zmm0
; AVX512F-NEXT:    vporq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
  %3 = load <8 x i64>, <8 x i64>* %0
  %4 = and <8 x i64> %3, <i64 12884901890, i64 4294967296, i64 12884901890, i64 4294967296, i64 12884901890, i64 4294967296, i64 12884901890, i64 4294967296>
  %5 = and <8 x i64> %1, <i64 -12884901891, i64 -4294967297, i64 -12884901891, i64 -4294967297, i64 -12884901891, i64 -4294967297, i64 -12884901891, i64 -4294967297>
  %6 = or <8 x i64> %4, %5
  ret <8 x i64> %6
}

define <8 x i64> @bitselect_v8i64_mm(<8 x i64>* nocapture readonly, <8 x i64>* nocapture readonly) {
; SSE-LABEL: bitselect_v8i64_mm:
; SSE:       # %bb.0:
; SSE-NEXT:    movaps {{.*#+}} xmm4 = [3,8589934593]
; SSE-NEXT:    movaps 48(%rdi), %xmm1
; SSE-NEXT:    andps %xmm4, %xmm1
; SSE-NEXT:    movaps 32(%rdi), %xmm5
; SSE-NEXT:    andps %xmm4, %xmm5
; SSE-NEXT:    movaps 16(%rdi), %xmm6
; SSE-NEXT:    andps %xmm4, %xmm6
; SSE-NEXT:    andps (%rdi), %xmm4
; SSE-NEXT:    movaps {{.*#+}} xmm0 = [18446744073709551612,18446744065119617022]
; SSE-NEXT:    movaps 48(%rsi), %xmm3
; SSE-NEXT:    andps %xmm0, %xmm3
; SSE-NEXT:    orps %xmm1, %xmm3
; SSE-NEXT:    movaps 32(%rsi), %xmm2
; SSE-NEXT:    andps %xmm0, %xmm2
; SSE-NEXT:    orps %xmm5, %xmm2
; SSE-NEXT:    movaps 16(%rsi), %xmm1
; SSE-NEXT:    andps %xmm0, %xmm1
; SSE-NEXT:    orps %xmm6, %xmm1
; SSE-NEXT:    andps (%rsi), %xmm0
; SSE-NEXT:    orps %xmm4, %xmm0
; SSE-NEXT:    retq
;
; XOP-LABEL: bitselect_v8i64_mm:
; XOP:       # %bb.0:
; XOP-NEXT:    vbroadcastf128 {{.*#+}} ymm0 = [3,8589934593,3,8589934593]
; XOP-NEXT:    # ymm0 = mem[0,1,0,1]
; XOP-NEXT:    vandps 32(%rdi), %ymm0, %ymm1
; XOP-NEXT:    vandps (%rdi), %ymm0, %ymm0
; XOP-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = [18446744073709551612,18446744065119617022,18446744073709551612,18446744065119617022]
; XOP-NEXT:    # ymm2 = mem[0,1,0,1]
; XOP-NEXT:    vandps 32(%rsi), %ymm2, %ymm3
; XOP-NEXT:    vorps %ymm1, %ymm3, %ymm1
; XOP-NEXT:    vandps (%rsi), %ymm2, %ymm2
; XOP-NEXT:    vorps %ymm0, %ymm2, %ymm0
; XOP-NEXT:    retq
;
; AVX1-LABEL: bitselect_v8i64_mm:
; AVX1:       # %bb.0:
; AVX1-NEXT:    vbroadcastf128 {{.*#+}} ymm0 = [3,8589934593,3,8589934593]
; AVX1-NEXT:    # ymm0 = mem[0,1,0,1]
; AVX1-NEXT:    vandps 32(%rdi), %ymm0, %ymm1
; AVX1-NEXT:    vandps (%rdi), %ymm0, %ymm0
; AVX1-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = [18446744073709551612,18446744065119617022,18446744073709551612,18446744065119617022]
; AVX1-NEXT:    # ymm2 = mem[0,1,0,1]
; AVX1-NEXT:    vandps 32(%rsi), %ymm2, %ymm3
; AVX1-NEXT:    vorps %ymm1, %ymm3, %ymm1
; AVX1-NEXT:    vandps (%rsi), %ymm2, %ymm2
; AVX1-NEXT:    vorps %ymm0, %ymm2, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: bitselect_v8i64_mm:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vbroadcastf128 {{.*#+}} ymm0 = [3,8589934593,3,8589934593]
; AVX2-NEXT:    # ymm0 = mem[0,1,0,1]
; AVX2-NEXT:    vandps 32(%rdi), %ymm0, %ymm1
; AVX2-NEXT:    vandps (%rdi), %ymm0, %ymm0
; AVX2-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = [18446744073709551612,18446744065119617022,18446744073709551612,18446744065119617022]
; AVX2-NEXT:    # ymm2 = mem[0,1,0,1]
; AVX2-NEXT:    vandps 32(%rsi), %ymm2, %ymm3
; AVX2-NEXT:    vorps %ymm1, %ymm3, %ymm1
; AVX2-NEXT:    vandps (%rsi), %ymm2, %ymm2
; AVX2-NEXT:    vorps %ymm0, %ymm2, %ymm0
; AVX2-NEXT:    retq
;
; AVX512F-LABEL: bitselect_v8i64_mm:
; AVX512F:       # %bb.0:
; AVX512F-NEXT:    vmovdqa64 (%rdi), %zmm0
; AVX512F-NEXT:    vmovdqa64 (%rsi), %zmm1
; AVX512F-NEXT:    vpandq {{.*}}(%rip), %zmm0, %zmm0
; AVX512F-NEXT:    vpandq {{.*}}(%rip), %zmm1, %zmm1
; AVX512F-NEXT:    vporq %zmm0, %zmm1, %zmm0
; AVX512F-NEXT:    retq
  %3 = load <8 x i64>, <8 x i64>* %0
  %4 = load <8 x i64>, <8 x i64>* %1
  %5 = and <8 x i64> %3, <i64 3, i64 8589934593, i64 3, i64 8589934593, i64 3, i64 8589934593, i64 3, i64 8589934593>
  %6 = and <8 x i64> %4, <i64 -4, i64 -8589934594, i64 -4, i64 -8589934594, i64 -4, i64 -8589934594, i64 -4, i64 -8589934594>
  %7 = or <8 x i64> %6, %5
  ret <8 x i64> %7
}
