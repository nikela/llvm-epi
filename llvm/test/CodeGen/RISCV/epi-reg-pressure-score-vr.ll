; RUN: llc -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c,+v -o - < %s \
; RUN:     -epi-pipeline --verify-machineinstrs | FileCheck %s

source_filename = "t29.cc"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128-v128:128:128-v256:128:128-v512:128:128-v1024:128:128"
target triple = "riscv64-unknown-linux-gnu"

; CHECK-NOT: vl1r.v
; CHECK-NOT: vs1r.v

; Function Attrs: nounwind
define dso_local signext i32 @main() local_unnamed_addr #0 {
entry:
  %call = tail call noalias align 16 dereferenceable_or_null(28672) i8* @malloc(i64 28672)
  %0 = bitcast i8* %call to float*
  %call1 = tail call noalias align 16 dereferenceable_or_null(28672) i8* @malloc(i64 28672)
  %1 = bitcast i8* %call1 to float*
  %2 = tail call i64 @llvm.epi.vsetvl(i64 256, i64 0, i64 2)
  %3 = bitcast i8* %call to <vscale x 2 x float>*
  %4 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* %3, i64 %2)
  %arrayidx = getelementptr inbounds float, float* %0, i64 256
  %5 = bitcast float* %arrayidx to <vscale x 2 x float>*
  %6 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %5, i64 %2)
  %arrayidx2 = getelementptr inbounds float, float* %0, i64 512
  %7 = bitcast float* %arrayidx2 to <vscale x 2 x float>*
  %8 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %7, i64 %2)
  %arrayidx3 = getelementptr inbounds float, float* %0, i64 768
  %9 = bitcast float* %arrayidx3 to <vscale x 2 x float>*
  %10 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %9, i64 %2)
  %arrayidx4 = getelementptr inbounds float, float* %0, i64 1024
  %11 = bitcast float* %arrayidx4 to <vscale x 2 x float>*
  %12 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %11, i64 %2)
  %arrayidx5 = getelementptr inbounds float, float* %0, i64 1280
  %13 = bitcast float* %arrayidx5 to <vscale x 2 x float>*
  %14 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %13, i64 %2)
  %arrayidx6 = getelementptr inbounds float, float* %0, i64 1536
  %15 = bitcast float* %arrayidx6 to <vscale x 2 x float>*
  %16 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %15, i64 %2)
  %arrayidx7 = getelementptr inbounds float, float* %0, i64 1792
  %17 = bitcast float* %arrayidx7 to <vscale x 2 x float>*
  %18 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %17, i64 %2)
  %arrayidx8 = getelementptr inbounds float, float* %0, i64 2048
  %19 = bitcast float* %arrayidx8 to <vscale x 2 x float>*
  %20 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %19, i64 %2)
  %arrayidx9 = getelementptr inbounds float, float* %0, i64 2304
  %21 = bitcast float* %arrayidx9 to <vscale x 2 x float>*
  %22 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %21, i64 %2)
  %arrayidx10 = getelementptr inbounds float, float* %0, i64 2560
  %23 = bitcast float* %arrayidx10 to <vscale x 2 x float>*
  %24 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %23, i64 %2)
  %arrayidx11 = getelementptr inbounds float, float* %0, i64 2816
  %25 = bitcast float* %arrayidx11 to <vscale x 2 x float>*
  %26 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %25, i64 %2)
  %arrayidx12 = getelementptr inbounds float, float* %0, i64 3072
  %27 = bitcast float* %arrayidx12 to <vscale x 2 x float>*
  %28 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %27, i64 %2)
  %arrayidx13 = getelementptr inbounds float, float* %0, i64 3328
  %29 = bitcast float* %arrayidx13 to <vscale x 2 x float>*
  %30 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %29, i64 %2)
  %arrayidx14 = getelementptr inbounds float, float* %0, i64 3584
  %31 = bitcast float* %arrayidx14 to <vscale x 2 x float>*
  %32 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %31, i64 %2)
  %arrayidx15 = getelementptr inbounds float, float* %0, i64 3840
  %33 = bitcast float* %arrayidx15 to <vscale x 2 x float>*
  %34 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %33, i64 %2)
  %arrayidx16 = getelementptr inbounds float, float* %0, i64 4096
  %35 = bitcast float* %arrayidx16 to <vscale x 2 x float>*
  %36 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %35, i64 %2)
  %arrayidx17 = getelementptr inbounds float, float* %0, i64 4352
  %37 = bitcast float* %arrayidx17 to <vscale x 2 x float>*
  %38 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %37, i64 %2)
  %arrayidx18 = getelementptr inbounds float, float* %0, i64 4608
  %39 = bitcast float* %arrayidx18 to <vscale x 2 x float>*
  %40 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %39, i64 %2)
  %arrayidx19 = getelementptr inbounds float, float* %0, i64 4864
  %41 = bitcast float* %arrayidx19 to <vscale x 2 x float>*
  %42 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %41, i64 %2)
  %arrayidx20 = getelementptr inbounds float, float* %0, i64 5120
  %43 = bitcast float* %arrayidx20 to <vscale x 2 x float>*
  %44 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %43, i64 %2)
  %arrayidx21 = getelementptr inbounds float, float* %0, i64 5376
  %45 = bitcast float* %arrayidx21 to <vscale x 2 x float>*
  %46 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %45, i64 %2)
  %arrayidx22 = getelementptr inbounds float, float* %0, i64 5632
  %47 = bitcast float* %arrayidx22 to <vscale x 2 x float>*
  %48 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %47, i64 %2)
  %arrayidx23 = getelementptr inbounds float, float* %0, i64 5888
  %49 = bitcast float* %arrayidx23 to <vscale x 2 x float>*
  %50 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %49, i64 %2)
  %arrayidx24 = getelementptr inbounds float, float* %0, i64 6144
  %51 = bitcast float* %arrayidx24 to <vscale x 2 x float>*
  %52 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %51, i64 %2)
  %arrayidx25 = getelementptr inbounds float, float* %0, i64 6400
  %53 = bitcast float* %arrayidx25 to <vscale x 2 x float>*
  %54 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %53, i64 %2)
  %arrayidx26 = getelementptr inbounds float, float* %0, i64 6656
  %55 = bitcast float* %arrayidx26 to <vscale x 2 x float>*
  %56 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %55, i64 %2)
  %arrayidx27 = getelementptr inbounds float, float* %0, i64 6912
  %57 = bitcast float* %arrayidx27 to <vscale x 2 x float>*
  %58 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %57, i64 %2)
  %59 = bitcast i8* %call1 to <vscale x 2 x float>*
  %60 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* %59, i64 %2)
  %61 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %4, <vscale x 2 x float> %60, i64 %2)
  %arrayidx28 = getelementptr inbounds float, float* %1, i64 256
  %62 = bitcast float* %arrayidx28 to <vscale x 2 x float>*
  %63 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %62, i64 %2)
  %64 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %6, <vscale x 2 x float> %63, i64 %2)
  %arrayidx29 = getelementptr inbounds float, float* %1, i64 512
  %65 = bitcast float* %arrayidx29 to <vscale x 2 x float>*
  %66 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %65, i64 %2)
  %67 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %8, <vscale x 2 x float> %66, i64 %2)
  %arrayidx30 = getelementptr inbounds float, float* %1, i64 768
  %68 = bitcast float* %arrayidx30 to <vscale x 2 x float>*
  %69 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %68, i64 %2)
  %70 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %10, <vscale x 2 x float> %69, i64 %2)
  %arrayidx31 = getelementptr inbounds float, float* %1, i64 1024
  %71 = bitcast float* %arrayidx31 to <vscale x 2 x float>*
  %72 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %71, i64 %2)
  %73 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %12, <vscale x 2 x float> %72, i64 %2)
  %arrayidx32 = getelementptr inbounds float, float* %1, i64 1280
  %74 = bitcast float* %arrayidx32 to <vscale x 2 x float>*
  %75 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %74, i64 %2)
  %76 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %14, <vscale x 2 x float> %75, i64 %2)
  %arrayidx33 = getelementptr inbounds float, float* %1, i64 1536
  %77 = bitcast float* %arrayidx33 to <vscale x 2 x float>*
  %78 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %77, i64 %2)
  %79 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %16, <vscale x 2 x float> %78, i64 %2)
  %arrayidx34 = getelementptr inbounds float, float* %1, i64 1792
  %80 = bitcast float* %arrayidx34 to <vscale x 2 x float>*
  %81 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %80, i64 %2)
  %82 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %18, <vscale x 2 x float> %81, i64 %2)
  %arrayidx35 = getelementptr inbounds float, float* %1, i64 2048
  %83 = bitcast float* %arrayidx35 to <vscale x 2 x float>*
  %84 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %83, i64 %2)
  %85 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %20, <vscale x 2 x float> %84, i64 %2)
  %arrayidx36 = getelementptr inbounds float, float* %1, i64 2304
  %86 = bitcast float* %arrayidx36 to <vscale x 2 x float>*
  %87 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %86, i64 %2)
  %88 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %22, <vscale x 2 x float> %87, i64 %2)
  %arrayidx37 = getelementptr inbounds float, float* %1, i64 2560
  %89 = bitcast float* %arrayidx37 to <vscale x 2 x float>*
  %90 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %89, i64 %2)
  %91 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %24, <vscale x 2 x float> %90, i64 %2)
  %arrayidx38 = getelementptr inbounds float, float* %1, i64 2816
  %92 = bitcast float* %arrayidx38 to <vscale x 2 x float>*
  %93 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %92, i64 %2)
  %94 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %26, <vscale x 2 x float> %93, i64 %2)
  %arrayidx39 = getelementptr inbounds float, float* %1, i64 3072
  %95 = bitcast float* %arrayidx39 to <vscale x 2 x float>*
  %96 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %95, i64 %2)
  %97 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %28, <vscale x 2 x float> %96, i64 %2)
  %arrayidx40 = getelementptr inbounds float, float* %1, i64 3328
  %98 = bitcast float* %arrayidx40 to <vscale x 2 x float>*
  %99 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %98, i64 %2)
  %100 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %30, <vscale x 2 x float> %99, i64 %2)
  %arrayidx41 = getelementptr inbounds float, float* %1, i64 3584
  %101 = bitcast float* %arrayidx41 to <vscale x 2 x float>*
  %102 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %101, i64 %2)
  %103 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %32, <vscale x 2 x float> %102, i64 %2)
  %arrayidx42 = getelementptr inbounds float, float* %1, i64 3840
  %104 = bitcast float* %arrayidx42 to <vscale x 2 x float>*
  %105 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %104, i64 %2)
  %106 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %34, <vscale x 2 x float> %105, i64 %2)
  %arrayidx43 = getelementptr inbounds float, float* %1, i64 4096
  %107 = bitcast float* %arrayidx43 to <vscale x 2 x float>*
  %108 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %107, i64 %2)
  %109 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %36, <vscale x 2 x float> %108, i64 %2)
  %arrayidx44 = getelementptr inbounds float, float* %1, i64 4352
  %110 = bitcast float* %arrayidx44 to <vscale x 2 x float>*
  %111 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %110, i64 %2)
  %112 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %38, <vscale x 2 x float> %111, i64 %2)
  %arrayidx45 = getelementptr inbounds float, float* %1, i64 4608
  %113 = bitcast float* %arrayidx45 to <vscale x 2 x float>*
  %114 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %113, i64 %2)
  %115 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %40, <vscale x 2 x float> %114, i64 %2)
  %arrayidx46 = getelementptr inbounds float, float* %1, i64 4864
  %116 = bitcast float* %arrayidx46 to <vscale x 2 x float>*
  %117 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %116, i64 %2)
  %118 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %42, <vscale x 2 x float> %117, i64 %2)
  %arrayidx47 = getelementptr inbounds float, float* %1, i64 5120
  %119 = bitcast float* %arrayidx47 to <vscale x 2 x float>*
  %120 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %119, i64 %2)
  %121 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %44, <vscale x 2 x float> %120, i64 %2)
  %arrayidx48 = getelementptr inbounds float, float* %1, i64 5376
  %122 = bitcast float* %arrayidx48 to <vscale x 2 x float>*
  %123 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %122, i64 %2)
  %124 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %46, <vscale x 2 x float> %123, i64 %2)
  %arrayidx49 = getelementptr inbounds float, float* %1, i64 5632
  %125 = bitcast float* %arrayidx49 to <vscale x 2 x float>*
  %126 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %125, i64 %2)
  %127 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %48, <vscale x 2 x float> %126, i64 %2)
  %arrayidx50 = getelementptr inbounds float, float* %1, i64 5888
  %128 = bitcast float* %arrayidx50 to <vscale x 2 x float>*
  %129 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %128, i64 %2)
  %130 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %50, <vscale x 2 x float> %129, i64 %2)
  %arrayidx51 = getelementptr inbounds float, float* %1, i64 6144
  %131 = bitcast float* %arrayidx51 to <vscale x 2 x float>*
  %132 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %131, i64 %2)
  %133 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %52, <vscale x 2 x float> %132, i64 %2)
  %arrayidx52 = getelementptr inbounds float, float* %1, i64 6400
  %134 = bitcast float* %arrayidx52 to <vscale x 2 x float>*
  %135 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %134, i64 %2)
  %136 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %54, <vscale x 2 x float> %135, i64 %2)
  %arrayidx53 = getelementptr inbounds float, float* %1, i64 6656
  %137 = bitcast float* %arrayidx53 to <vscale x 2 x float>*
  %138 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %137, i64 %2)
  %139 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %56, <vscale x 2 x float> %138, i64 %2)
  %arrayidx54 = getelementptr inbounds float, float* %1, i64 6912
  %140 = bitcast float* %arrayidx54 to <vscale x 2 x float>*
  %141 = tail call <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nonnull %140, i64 %2)
  %142 = tail call <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float> %58, <vscale x 2 x float> %141, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %61, <vscale x 2 x float>* %3, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %64, <vscale x 2 x float>* nonnull %5, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %67, <vscale x 2 x float>* nonnull %7, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %70, <vscale x 2 x float>* nonnull %9, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %73, <vscale x 2 x float>* nonnull %11, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %76, <vscale x 2 x float>* nonnull %13, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %79, <vscale x 2 x float>* nonnull %15, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %82, <vscale x 2 x float>* nonnull %17, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %85, <vscale x 2 x float>* nonnull %19, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %88, <vscale x 2 x float>* nonnull %21, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %91, <vscale x 2 x float>* nonnull %23, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %94, <vscale x 2 x float>* nonnull %25, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %97, <vscale x 2 x float>* nonnull %27, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %100, <vscale x 2 x float>* nonnull %29, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %103, <vscale x 2 x float>* nonnull %31, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %106, <vscale x 2 x float>* nonnull %33, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %109, <vscale x 2 x float>* nonnull %35, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %112, <vscale x 2 x float>* nonnull %37, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %115, <vscale x 2 x float>* nonnull %39, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %118, <vscale x 2 x float>* nonnull %41, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %121, <vscale x 2 x float>* nonnull %43, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %124, <vscale x 2 x float>* nonnull %45, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %127, <vscale x 2 x float>* nonnull %47, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %130, <vscale x 2 x float>* nonnull %49, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %133, <vscale x 2 x float>* nonnull %51, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %136, <vscale x 2 x float>* nonnull %53, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %139, <vscale x 2 x float>* nonnull %55, i64 %2)
  tail call void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float> %142, <vscale x 2 x float>* nonnull %57, i64 %2)
  tail call void @free(i8* %call)
  tail call void @free(i8* %call1)
  ret i32 0
}

; Function Attrs: inaccessiblememonly nofree nounwind willreturn mustprogress
declare dso_local noalias noundef align 16 i8* @malloc(i64 noundef) local_unnamed_addr #1

; Function Attrs: nofree nosync nounwind readnone
declare i64 @llvm.epi.vsetvl(i64, i64, i64) #2

; Function Attrs: nofree nounwind readonly
declare <vscale x 2 x float> @llvm.epi.vload.nxv2f32(<vscale x 2 x float>* nocapture, i64) #3

; Function Attrs: nofree nosync nounwind readnone
declare <vscale x 2 x float> @llvm.epi.vfadd.nxv2f32.nxv2f32(<vscale x 2 x float>, <vscale x 2 x float>, i64) #2

; Function Attrs: nounwind writeonly
declare void @llvm.epi.vstore.nxv2f32(<vscale x 2 x float>, <vscale x 2 x float>* nocapture, i64) #4

; Function Attrs: inaccessiblemem_or_argmemonly nounwind willreturn mustprogress
declare dso_local void @free(i8* nocapture noundef) local_unnamed_addr #5

attributes #0 = { nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+a,+c,+d,+v,+experimental-zvlsseg,+f,+m,-relax,-save-restore" }
attributes #1 = { inaccessiblememonly nofree nounwind willreturn mustprogress "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+a,+c,+d,+v,+experimental-zvlsseg,+f,+m,-relax,-save-restore" }
attributes #2 = { nofree nosync nounwind readnone }
attributes #3 = { nofree nounwind readonly }
attributes #4 = { nounwind writeonly }
attributes #5 = { inaccessiblemem_or_argmemonly nounwind willreturn mustprogress "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+a,+c,+d,+v,+experimental-zvlsseg,+f,+m,-relax,-save-restore" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64d"}
!2 = !{i32 1, !"SmallDataLimit", i32 8}
!3 = !{!"clang version 13.0.0"}
