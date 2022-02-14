; RUN: llc -mtriple=riscv64 -mattr=+f,+d,+zepi -epi-pipeline < %s
; RUN: not --crash llc -mtriple=riscv64 -mattr=+f,+d,+v -epi-pipeline < %s \
; RUN:  2>&1 | FileCheck %s

; CHECK: LLVM ERROR: EPI_VectorCall calling convention requires the F,D and EPI feature sets enabled

define epi_vectorcall i32 @p(i32 signext %N, i32* %a) {
entry:
  %idxprom = sext i32 %N to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  ret i32 %0
}

define signext i32 @g(i32 signext %N, i32* %a) {
entry:
  tail call epi_vectorcall void @f(i32* %a)
  %idxprom.i = sext i32 %N to i64
  %arrayidx.i = getelementptr inbounds i32, i32* %a, i64 %idxprom.i
  %0 = load i32, i32* %arrayidx.i, align 4
  ret i32 %0
}

declare epi_vectorcall void @f(i32*)
