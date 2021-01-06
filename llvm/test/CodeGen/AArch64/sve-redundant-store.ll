; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -O2 -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning:

; #include <arm_sve.h>
; #include <stdint.h>
;
; void redundant_store(uint32_t *p, svint32_t v) {
;     *p = 1;
;     *(svint32_t *)p = v;
; }

; Update me: Until dead store elimination is improved in DAGCombine, this will contain a redundant store.
;
define void @redundant_store(i32* nocapture %p, <vscale x 4 x i32> %v) {
; CHECK-LABEL: redundant_store:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #1
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    str w8, [x0]
; CHECK-NEXT:    st1w { z0.s }, p0, [x0]
; CHECK-NEXT:    ret
  store i32 1, i32* %p, align 4
  %1 = bitcast i32* %p to <vscale x 4 x i32>*
  store <vscale x 4 x i32> %v, <vscale x 4 x i32>* %1, align 16
  ret void
}
