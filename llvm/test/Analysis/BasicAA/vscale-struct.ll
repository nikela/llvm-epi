; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK:   NoAlias: <vscale x 1 x i64>* %v01, <vscale x 1 x i64>* %v02
%struct.__epi_1xi64x2 = type { <vscale x 1 x i64>, <vscale x 1 x i64> }

define void @foo()
{
    %va = alloca %struct.__epi_1xi64x2, align 8
    %v01 = getelementptr inbounds %struct.__epi_1xi64x2, %struct.__epi_1xi64x2* %va, i32 0, i32 0
    %v02 = getelementptr inbounds %struct.__epi_1xi64x2, %struct.__epi_1xi64x2* %va, i32 0, i32 1
    load <vscale x 1 x i64>, <vscale x 1 x i64>* %v01
    load <vscale x 1 x i64>, <vscale x 1 x i64>* %v02
    ret void
}
