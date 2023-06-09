; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 < %s | FileCheck %s --check-prefix=SMALL
; RUN: llc -mtriple=riscv64 -code-model=medium < %s | \
; RUN:     FileCheck %s --check-prefix=MEDIUM

@x = common dso_local global i32 0, align 4

define void @foo() nounwind {
; SMALL-LABEL: foo:
; SMALL:       # %bb.0: # %entry
; SMALL-NEXT:    lui a0, %hi(x)
; SMALL-NEXT:    lw a1, %lo(x)(a0)
; SMALL-NEXT:    addiw a1, a1, 1
; SMALL-NEXT:    sw a1, %lo(x)(a0)
; SMALL-NEXT:    ret
;
; MEDIUM-LABEL: foo:
; MEDIUM:       # %bb.0: # %entry
; MEDIUM-NEXT:  .Lpcrel_hi0:
; MEDIUM-NEXT:    auipc a0, %pcrel_hi(x)
; MEDIUM-NEXT:    lw a1, %pcrel_lo(.Lpcrel_hi0)(a0)
; MEDIUM-NEXT:    addiw a1, a1, 1
; MEDIUM-NEXT:    sw a1, %pcrel_lo(.Lpcrel_hi0)(a0)
; MEDIUM-NEXT:    ret
entry:
  %0 = load i32, i32* @x, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @x, align 4
  ret void
}

