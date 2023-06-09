; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -mattr=+use-experimental-zeroing-pseudos < %s | FileCheck %s

;; ASR
define <vscale x 16 x i8> @asr_i8(<vscale x 16 x i8> %a, <vscale x 16 x i1> %pg) {
; CHECK-LABEL: asr_i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movprfx z0.b, p0/z, z0.b
; CHECK-NEXT:    asr z0.b, p0/m, z0.b, #8
; CHECK-NEXT:    ret
  %vsel = select <vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> zeroinitializer
  %ele = insertelement <vscale x 16 x i8> poison, i8 8, i32 0
  %shuffle = shufflevector <vscale x 16 x i8> %ele, <vscale x 16 x i8> poison, <vscale x 16 x i32> zeroinitializer
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %vsel, <vscale x 16 x i8> %shuffle)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @asr_i16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg) {
; CHECK-LABEL: asr_i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movprfx z0.h, p0/z, z0.h
; CHECK-NEXT:    asr z0.h, p0/m, z0.h, #16
; CHECK-NEXT:    ret
  %vsel = select <vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> zeroinitializer
  %ele = insertelement <vscale x 8 x i16> poison, i16 16, i32 0
  %shuffle = shufflevector <vscale x 8 x i16> %ele, <vscale x 8 x i16> poison, <vscale x 8 x i32> zeroinitializer
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.nxv8i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %vsel, <vscale x 8 x i16> %shuffle)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @asr_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg) local_unnamed_addr #0 {
; CHECK-LABEL: asr_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movprfx z0.s, p0/z, z0.s
; CHECK-NEXT:    asr z0.s, p0/m, z0.s, #32
; CHECK-NEXT:    ret
  %vsel = select <vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> zeroinitializer
  %ele = insertelement <vscale x 4 x i32> poison, i32 32, i32 0
  %shuffle = shufflevector <vscale x 4 x i32> %ele, <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.nxv4i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %vsel, <vscale x 4 x i32> %shuffle)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @asr_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg) {
; CHECK-LABEL: asr_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movprfx z0.d, p0/z, z0.d
; CHECK-NEXT:    asr z0.d, p0/m, z0.d, #64
; CHECK-NEXT:    ret
  %vsel = select <vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> zeroinitializer
  %ele = insertelement <vscale x 2 x i64> poison, i64 64, i32 0
  %shuffle = shufflevector <vscale x 2 x i64> %ele, <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.asr.nxv2i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %vsel, <vscale x 2 x i64> %shuffle)
  ret <vscale x 2 x i64> %res
}

;; LSL
define <vscale x 16 x i8> @lsl_i8(<vscale x 16 x i8> %a, <vscale x 16 x i1> %pg) {
; CHECK-LABEL: lsl_i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movprfx z0.b, p0/z, z0.b
; CHECK-NEXT:    lsl z0.b, p0/m, z0.b, #7
; CHECK-NEXT:    ret
  %vsel = select <vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> zeroinitializer
  %ele = insertelement <vscale x 16 x i8> poison, i8 7, i32 0
  %shuffle = shufflevector <vscale x 16 x i8> %ele, <vscale x 16 x i8> poison, <vscale x 16 x i32> zeroinitializer
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.lsl.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %vsel, <vscale x 16 x i8> %shuffle)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @lsl_i16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg) {
; CHECK-LABEL: lsl_i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movprfx z0.h, p0/z, z0.h
; CHECK-NEXT:    lsl z0.h, p0/m, z0.h, #15
; CHECK-NEXT:    ret
  %vsel = select <vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> zeroinitializer
  %ele = insertelement <vscale x 8 x i16> poison, i16 15, i32 0
  %shuffle = shufflevector <vscale x 8 x i16> %ele, <vscale x 8 x i16> poison, <vscale x 8 x i32> zeroinitializer
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.lsl.nxv8i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %vsel, <vscale x 8 x i16> %shuffle)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @lsl_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg) local_unnamed_addr #0 {
; CHECK-LABEL: lsl_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movprfx z0.s, p0/z, z0.s
; CHECK-NEXT:    lsl z0.s, p0/m, z0.s, #31
; CHECK-NEXT:    ret
  %vsel = select <vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> zeroinitializer
  %ele = insertelement <vscale x 4 x i32> poison, i32 31, i32 0
  %shuffle = shufflevector <vscale x 4 x i32> %ele, <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.lsl.nxv4i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %vsel, <vscale x 4 x i32> %shuffle)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @lsl_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg) {
; CHECK-LABEL: lsl_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movprfx z0.d, p0/z, z0.d
; CHECK-NEXT:    lsl z0.d, p0/m, z0.d, #63
; CHECK-NEXT:    ret
  %vsel = select <vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> zeroinitializer
  %ele = insertelement <vscale x 2 x i64> poison, i64 63, i32 0
  %shuffle = shufflevector <vscale x 2 x i64> %ele, <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.lsl.nxv2i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %vsel, <vscale x 2 x i64> %shuffle)
  ret <vscale x 2 x i64> %res
}

;; LSR
define <vscale x 16 x i8> @lsr_i8(<vscale x 16 x i8> %a, <vscale x 16 x i1> %pg) {
; CHECK-LABEL: lsr_i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movprfx z0.b, p0/z, z0.b
; CHECK-NEXT:    lsr z0.b, p0/m, z0.b, #8
; CHECK-NEXT:    ret
  %vsel = select <vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> zeroinitializer
  %ele = insertelement <vscale x 16 x i8> poison, i8 8, i32 0
  %shuffle = shufflevector <vscale x 16 x i8> %ele, <vscale x 16 x i8> poison, <vscale x 16 x i32> zeroinitializer
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.lsr.nxv16i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %vsel, <vscale x 16 x i8> %shuffle)
  ret <vscale x 16 x i8> %res
}

define <vscale x 8 x i16> @lsr_i16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %pg) {
; CHECK-LABEL: lsr_i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movprfx z0.h, p0/z, z0.h
; CHECK-NEXT:    lsr z0.h, p0/m, z0.h, #16
; CHECK-NEXT:    ret
  %vsel = select <vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> zeroinitializer
  %ele = insertelement <vscale x 8 x i16> poison, i16 16, i32 0
  %shuffle = shufflevector <vscale x 8 x i16> %ele, <vscale x 8 x i16> poison, <vscale x 8 x i32> zeroinitializer
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.lsr.nxv8i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %vsel, <vscale x 8 x i16> %shuffle)
  ret <vscale x 8 x i16> %res
}

define <vscale x 4 x i32> @lsr_i32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %pg) local_unnamed_addr #0 {
; CHECK-LABEL: lsr_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movprfx z0.s, p0/z, z0.s
; CHECK-NEXT:    lsr z0.s, p0/m, z0.s, #32
; CHECK-NEXT:    ret
  %vsel = select <vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> zeroinitializer
  %ele = insertelement <vscale x 4 x i32> poison, i32 32, i32 0
  %shuffle = shufflevector <vscale x 4 x i32> %ele, <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.lsr.nxv4i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %vsel, <vscale x 4 x i32> %shuffle)
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @lsr_i64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %pg) {
; CHECK-LABEL: lsr_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movprfx z0.d, p0/z, z0.d
; CHECK-NEXT:    lsr z0.d, p0/m, z0.d, #64
; CHECK-NEXT:    ret
  %vsel = select <vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> zeroinitializer
  %ele = insertelement <vscale x 2 x i64> poison, i64 64, i32 0
  %shuffle = shufflevector <vscale x 2 x i64> %ele, <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.lsr.nxv2i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %vsel, <vscale x 2 x i64> %shuffle)
  ret <vscale x 2 x i64> %res
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.asr.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.asr.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.asr.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.asr.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.lsl.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.lsl.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.lsl.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.lsl.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.lsr.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.lsr.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.lsr.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.lsr.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
