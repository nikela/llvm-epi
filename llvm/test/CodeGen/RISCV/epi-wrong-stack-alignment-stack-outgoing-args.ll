; RUN: llc -mtriple riscv64 -mattr +m,+v -stop-after=finalize-isel < %s \
; RUN:   | FileCheck %s

; CHECK: maxAlignment:    16

define void @foo(i64 %len, i64 %xa, i64 %xb, i64 %xc) nounwind {
  %vla = alloca i64, i64 %len, align 8

  %x1 = sdiv i64 %xa, %xb
  %x2 = sdiv i64 %xa, %xc
  %x3 = sdiv i64 %x1, %xb
  %x4 = sdiv i64 %x2, %xc
  %x5 = sdiv i64 %x3, %xb
  %x6 = sdiv i64 %x4, %xc
  %x7 = sdiv i64 %x5, %xb
  %x8 = sdiv i64 %x6, %xc
  %x9 = sdiv i64 %x7, %xb
  %x10 = sdiv i64 %x8, %xc
  %x11 = sdiv i64 %x9, %xb
  %x12 = sdiv i64 %x10, %xc

  call void @bar(<vscale x 8 x i64> undef, <vscale x 8 x i64> undef,
                 <vscale x 8 x i64> undef, <vscale x 8 x i64> undef)

  call void @bar2(i64 %x1, i64 %x2, i64 %x3, i64 %x4, i64 %x5, i64 %x6, i64 %x7,
                  i64 %x8, i64 %x9, i64 %x10, i64 %x11, i64 %x12)
  ret void
}

declare void @bar(<vscale x 8 x i64> %a, <vscale x 8 x i64> %b,
                  <vscale x 8 x i64> %c, <vscale x 8 x i64> %d)

declare void @bar2(i64 %x1, i64 %x2, i64 %x3, i64 %x4, i64 %x5, i64 %x6,
                   i64 %x7, i64 %x8, i64 %x9, i64 %x10, i64 %x11, i64 %x12)

