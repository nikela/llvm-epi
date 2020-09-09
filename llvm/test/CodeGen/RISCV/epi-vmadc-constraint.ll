; RUN: llc -mtriple riscv64 -mattr +m,+f,+d,+c,+a,+experimental-v -o - < %s | \
; RUN:     FileCheck %s

; ModuleID = 'chacha.c'
source_filename = "chacha.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128-v128:128:128-v256:128:128-v512:128:128-v1024:128:128"
target triple = "riscv64-unknown-linux-gnu"

; CHECK-NOT: vmadc.vvm [[REG:v[0-9]+]]{{.*}}[[REG]]

%struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx = type { [16 x i32] }

; Function Attrs: nounwind
define dso_local void @foo(%struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* nocapture %x_, i8* nocapture readonly %m, i8* nocapture %c_, i32 signext %bytes) local_unnamed_addr
{
entry:
  %x.i = alloca [16 x i32], align 4
  %output = alloca [64 x i8], align 1
  %0 = getelementptr inbounds [64 x i8], [64 x i8]* %output, i64 0, i64 0
  %1 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 0
  %tobool.not = icmp eq i32 %bytes, 0
  br i1 %tobool.not, label %cleanup, label %if.end

if.end:                                           ; preds = %entry
  %cmp = icmp ugt i32 %bytes, 255
  br i1 %cmp, label %if.then1, label %for.cond211.preheader

if.then1:                                         ; preds = %if.end
  %2 = tail call i64 @llvm.epi.vsetvlmax(i64 2, i64 0)
  %3 = load i32, i32* %1, align 4
  %4 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %3, i64 %2)
  %arrayidx2 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 1
  %5 = load i32, i32* %arrayidx2, align 4
  %6 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %5, i64 %2)
  %arrayidx3 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 2
  %7 = load i32, i32* %arrayidx3, align 4
  %8 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %7, i64 %2)
  %arrayidx4 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 3
  %9 = load i32, i32* %arrayidx4, align 4
  %10 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %9, i64 %2)
  %arrayidx5 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 4
  %11 = load i32, i32* %arrayidx5, align 4
  %12 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %11, i64 %2)
  %arrayidx6 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 5
  %13 = load i32, i32* %arrayidx6, align 4
  %14 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %13, i64 %2)
  %arrayidx7 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 6
  %15 = load i32, i32* %arrayidx7, align 4
  %16 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %15, i64 %2)
  %arrayidx8 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 7
  %17 = load i32, i32* %arrayidx8, align 4
  %18 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %17, i64 %2)
  %arrayidx9 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 8
  %19 = load i32, i32* %arrayidx9, align 4
  %20 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %19, i64 %2)
  %arrayidx10 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 9
  %21 = load i32, i32* %arrayidx10, align 4
  %22 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %21, i64 %2)
  %arrayidx11 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 10
  %23 = load i32, i32* %arrayidx11, align 4
  %24 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %23, i64 %2)
  %arrayidx12 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 11
  %25 = load i32, i32* %arrayidx12, align 4
  %26 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %25, i64 %2)
  %arrayidx13 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 14
  %27 = load i32, i32* %arrayidx13, align 4
  %28 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %27, i64 %2)
  %arrayidx14 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 15
  %29 = load i32, i32* %arrayidx14, align 4
  %30 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %29, i64 %2)
  %arrayidx17 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 12
  %arrayidx18 = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 13
  br label %while.body

while.body:                                       ; preds = %if.then1, %for.end
  %m.addr.0752 = phi i8* [ %m, %if.then1 ], [ %add.ptr206, %for.end ]
  %bytes.addr.0751 = phi i32 [ %bytes, %if.then1 ], [ %conv202, %for.end ]
  %out.0750 = phi i8* [ %c_, %if.then1 ], [ %add.ptr204, %for.end ]
  %carry.0749 = phi <vscale x 2 x i1> [ undef, %if.then1 ], [ %40, %for.end ]
  %div = lshr i32 %bytes.addr.0751, 7
  %31 = and i32 %div, 33554430
  %div16 = zext i32 %31 to i64
  %32 = tail call i64 @llvm.epi.vsetvl(i64 %div16, i64 3, i64 0)
  %mul = shl i64 %32, 1
  %33 = load i32, i32* %arrayidx17, align 4
  %34 = load i32, i32* %arrayidx18, align 4
  %conv19 = zext i32 %33 to i64
  %conv20 = zext i32 %34 to i64
  %shl = shl nuw i64 %conv20, 32
  %or = or i64 %shl, %conv19
  %35 = tail call <vscale x 2 x i32> @llvm.epi.vid.nxv2i32(i64 %mul)
  %36 = tail call <vscale x 2 x i1> @llvm.epi.vmxor.nxv2i1.nxv2i1(<vscale x 2 x i1> %carry.0749, <vscale x 2 x i1> %carry.0749, i64 %mul)
  %37 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %33, i64 %mul)
  %38 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 %34, i64 %mul)
  %39 = tail call <vscale x 2 x i32> @llvm.epi.vadc.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %37, <vscale x 2 x i32> %35, <vscale x 2 x i1> %36, i64 %mul)
  %40 = tail call <vscale x 2 x i1> @llvm.epi.vmadc.carry.in.nxv2i1.nxv2i32.nxv2i32(<vscale x 2 x i32> %37, <vscale x 2 x i32> %35, <vscale x 2 x i1> %36, i64 %mul)
  %41 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 0, i64 %mul)
  %42 = tail call <vscale x 2 x i32> @llvm.epi.vadc.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %38, <vscale x 2 x i32> %41, <vscale x 2 x i1> %40, i64 %mul)
  %add = add i64 %or, %mul
  %conv22 = trunc i64 %add to i32
  store i32 %conv22, i32* %arrayidx17, align 4
  %shr = lshr i64 %add, 32
  %conv25 = trunc i64 %shr to i32
  store i32 %conv25, i32* %arrayidx18, align 4
  %43 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 16, i64 %mul)
  %44 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 12, i64 %mul)
  %45 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 20, i64 %mul)
  %46 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 8, i64 %mul)
  %47 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 24, i64 %mul)
  %48 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 7, i64 %mul)
  %49 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 25, i64 %mul)
  br label %for.body

for.body:                                         ; preds = %while.body, %for.body
  %x_0.0747 = phi <vscale x 2 x i32> [ %4, %while.body ], [ %140, %for.body ]
  %x_1.0746 = phi <vscale x 2 x i32> [ %6, %while.body ], [ %160, %for.body ]
  %x_2.0745 = phi <vscale x 2 x i32> [ %8, %while.body ], [ %180, %for.body ]
  %x_3.0744 = phi <vscale x 2 x i32> [ %10, %while.body ], [ %200, %for.body ]
  %x_4.0743 = phi <vscale x 2 x i32> [ %12, %while.body ], [ %209, %for.body ]
  %x_5.0742 = phi <vscale x 2 x i32> [ %14, %while.body ], [ %149, %for.body ]
  %x_6.0741 = phi <vscale x 2 x i32> [ %16, %while.body ], [ %169, %for.body ]
  %x_7.0740 = phi <vscale x 2 x i32> [ %18, %while.body ], [ %189, %for.body ]
  %x_8.0739 = phi <vscale x 2 x i32> [ %20, %while.body ], [ %185, %for.body ]
  %x_9.0738 = phi <vscale x 2 x i32> [ %22, %while.body ], [ %205, %for.body ]
  %x_10.0737 = phi <vscale x 2 x i32> [ %24, %while.body ], [ %145, %for.body ]
  %x_11.0736 = phi <vscale x 2 x i32> [ %26, %while.body ], [ %165, %for.body ]
  %x_12.0735 = phi <vscale x 2 x i32> [ %39, %while.body ], [ %164, %for.body ]
  %x_13.0734 = phi <vscale x 2 x i32> [ %42, %while.body ], [ %184, %for.body ]
  %x_14.0733 = phi <vscale x 2 x i32> [ %28, %while.body ], [ %204, %for.body ]
  %x_15.0732 = phi <vscale x 2 x i32> [ %30, %while.body ], [ %144, %for.body ]
  %i.0731 = phi i32 [ 0, %while.body ], [ %add29, %for.body ]
  %50 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_0.0747, <vscale x 2 x i32> %x_4.0743, i64 %mul)
  %51 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_12.0735, <vscale x 2 x i32> %50, i64 %mul)
  %52 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %51, <vscale x 2 x i32> %43, i64 %mul)
  %53 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %51, <vscale x 2 x i32> %43, i64 %mul)
  %54 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %52, <vscale x 2 x i32> %53, i64 %mul)
  %55 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_8.0739, <vscale x 2 x i32> %54, i64 %mul)
  %56 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_4.0743, <vscale x 2 x i32> %55, i64 %mul)
  %57 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %56, <vscale x 2 x i32> %44, i64 %mul)
  %58 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %56, <vscale x 2 x i32> %45, i64 %mul)
  %59 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %57, <vscale x 2 x i32> %58, i64 %mul)
  %60 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %50, <vscale x 2 x i32> %59, i64 %mul)
  %61 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %54, <vscale x 2 x i32> %60, i64 %mul)
  %62 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %61, <vscale x 2 x i32> %46, i64 %mul)
  %63 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %61, <vscale x 2 x i32> %47, i64 %mul)
  %64 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %62, <vscale x 2 x i32> %63, i64 %mul)
  %65 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %55, <vscale x 2 x i32> %64, i64 %mul)
  %66 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %59, <vscale x 2 x i32> %65, i64 %mul)
  %67 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %66, <vscale x 2 x i32> %48, i64 %mul)
  %68 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %66, <vscale x 2 x i32> %49, i64 %mul)
  %69 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %67, <vscale x 2 x i32> %68, i64 %mul)
  %70 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_1.0746, <vscale x 2 x i32> %x_5.0742, i64 %mul)
  %71 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_13.0734, <vscale x 2 x i32> %70, i64 %mul)
  %72 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %71, <vscale x 2 x i32> %43, i64 %mul)
  %73 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %71, <vscale x 2 x i32> %43, i64 %mul)
  %74 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %72, <vscale x 2 x i32> %73, i64 %mul)
  %75 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_9.0738, <vscale x 2 x i32> %74, i64 %mul)
  %76 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_5.0742, <vscale x 2 x i32> %75, i64 %mul)
  %77 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %76, <vscale x 2 x i32> %44, i64 %mul)
  %78 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %76, <vscale x 2 x i32> %45, i64 %mul)
  %79 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %77, <vscale x 2 x i32> %78, i64 %mul)
  %80 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %70, <vscale x 2 x i32> %79, i64 %mul)
  %81 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %74, <vscale x 2 x i32> %80, i64 %mul)
  %82 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %81, <vscale x 2 x i32> %46, i64 %mul)
  %83 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %81, <vscale x 2 x i32> %47, i64 %mul)
  %84 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %82, <vscale x 2 x i32> %83, i64 %mul)
  %85 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %75, <vscale x 2 x i32> %84, i64 %mul)
  %86 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %79, <vscale x 2 x i32> %85, i64 %mul)
  %87 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %86, <vscale x 2 x i32> %48, i64 %mul)
  %88 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %86, <vscale x 2 x i32> %49, i64 %mul)
  %89 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %87, <vscale x 2 x i32> %88, i64 %mul)
  %90 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_2.0745, <vscale x 2 x i32> %x_6.0741, i64 %mul)
  %91 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_14.0733, <vscale x 2 x i32> %90, i64 %mul)
  %92 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %91, <vscale x 2 x i32> %43, i64 %mul)
  %93 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %91, <vscale x 2 x i32> %43, i64 %mul)
  %94 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %92, <vscale x 2 x i32> %93, i64 %mul)
  %95 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_10.0737, <vscale x 2 x i32> %94, i64 %mul)
  %96 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_6.0741, <vscale x 2 x i32> %95, i64 %mul)
  %97 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %96, <vscale x 2 x i32> %44, i64 %mul)
  %98 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %96, <vscale x 2 x i32> %45, i64 %mul)
  %99 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %97, <vscale x 2 x i32> %98, i64 %mul)
  %100 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %90, <vscale x 2 x i32> %99, i64 %mul)
  %101 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %94, <vscale x 2 x i32> %100, i64 %mul)
  %102 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %101, <vscale x 2 x i32> %46, i64 %mul)
  %103 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %101, <vscale x 2 x i32> %47, i64 %mul)
  %104 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %102, <vscale x 2 x i32> %103, i64 %mul)
  %105 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %95, <vscale x 2 x i32> %104, i64 %mul)
  %106 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %99, <vscale x 2 x i32> %105, i64 %mul)
  %107 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %106, <vscale x 2 x i32> %48, i64 %mul)
  %108 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %106, <vscale x 2 x i32> %49, i64 %mul)
  %109 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %107, <vscale x 2 x i32> %108, i64 %mul)
  %110 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_3.0744, <vscale x 2 x i32> %x_7.0740, i64 %mul)
  %111 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_15.0732, <vscale x 2 x i32> %110, i64 %mul)
  %112 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %111, <vscale x 2 x i32> %43, i64 %mul)
  %113 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %111, <vscale x 2 x i32> %43, i64 %mul)
  %114 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %112, <vscale x 2 x i32> %113, i64 %mul)
  %115 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_11.0736, <vscale x 2 x i32> %114, i64 %mul)
  %116 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %x_7.0740, <vscale x 2 x i32> %115, i64 %mul)
  %117 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %116, <vscale x 2 x i32> %44, i64 %mul)
  %118 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %116, <vscale x 2 x i32> %45, i64 %mul)
  %119 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %117, <vscale x 2 x i32> %118, i64 %mul)
  %120 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %110, <vscale x 2 x i32> %119, i64 %mul)
  %121 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %114, <vscale x 2 x i32> %120, i64 %mul)
  %122 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %121, <vscale x 2 x i32> %46, i64 %mul)
  %123 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %121, <vscale x 2 x i32> %47, i64 %mul)
  %124 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %122, <vscale x 2 x i32> %123, i64 %mul)
  %125 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %115, <vscale x 2 x i32> %124, i64 %mul)
  %126 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %119, <vscale x 2 x i32> %125, i64 %mul)
  %127 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %126, <vscale x 2 x i32> %48, i64 %mul)
  %128 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %126, <vscale x 2 x i32> %49, i64 %mul)
  %129 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %127, <vscale x 2 x i32> %128, i64 %mul)
  %130 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %60, <vscale x 2 x i32> %89, i64 %mul)
  %131 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %124, <vscale x 2 x i32> %130, i64 %mul)
  %132 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %131, <vscale x 2 x i32> %43, i64 %mul)
  %133 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %131, <vscale x 2 x i32> %43, i64 %mul)
  %134 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %132, <vscale x 2 x i32> %133, i64 %mul)
  %135 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %105, <vscale x 2 x i32> %134, i64 %mul)
  %136 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %89, <vscale x 2 x i32> %135, i64 %mul)
  %137 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %136, <vscale x 2 x i32> %44, i64 %mul)
  %138 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %136, <vscale x 2 x i32> %45, i64 %mul)
  %139 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %137, <vscale x 2 x i32> %138, i64 %mul)
  %140 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %130, <vscale x 2 x i32> %139, i64 %mul)
  %141 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %134, <vscale x 2 x i32> %140, i64 %mul)
  %142 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %141, <vscale x 2 x i32> %46, i64 %mul)
  %143 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %141, <vscale x 2 x i32> %47, i64 %mul)
  %144 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %142, <vscale x 2 x i32> %143, i64 %mul)
  %145 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %135, <vscale x 2 x i32> %144, i64 %mul)
  %146 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %139, <vscale x 2 x i32> %145, i64 %mul)
  %147 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %146, <vscale x 2 x i32> %48, i64 %mul)
  %148 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %146, <vscale x 2 x i32> %49, i64 %mul)
  %149 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %147, <vscale x 2 x i32> %148, i64 %mul)
  %150 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %80, <vscale x 2 x i32> %109, i64 %mul)
  %151 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %64, <vscale x 2 x i32> %150, i64 %mul)
  %152 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %151, <vscale x 2 x i32> %43, i64 %mul)
  %153 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %151, <vscale x 2 x i32> %43, i64 %mul)
  %154 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %152, <vscale x 2 x i32> %153, i64 %mul)
  %155 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %125, <vscale x 2 x i32> %154, i64 %mul)
  %156 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %109, <vscale x 2 x i32> %155, i64 %mul)
  %157 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %156, <vscale x 2 x i32> %44, i64 %mul)
  %158 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %156, <vscale x 2 x i32> %45, i64 %mul)
  %159 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %157, <vscale x 2 x i32> %158, i64 %mul)
  %160 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %150, <vscale x 2 x i32> %159, i64 %mul)
  %161 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %154, <vscale x 2 x i32> %160, i64 %mul)
  %162 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %161, <vscale x 2 x i32> %46, i64 %mul)
  %163 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %161, <vscale x 2 x i32> %47, i64 %mul)
  %164 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %162, <vscale x 2 x i32> %163, i64 %mul)
  %165 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %155, <vscale x 2 x i32> %164, i64 %mul)
  %166 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %159, <vscale x 2 x i32> %165, i64 %mul)
  %167 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %166, <vscale x 2 x i32> %48, i64 %mul)
  %168 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %166, <vscale x 2 x i32> %49, i64 %mul)
  %169 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %167, <vscale x 2 x i32> %168, i64 %mul)
  %170 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %100, <vscale x 2 x i32> %129, i64 %mul)
  %171 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %84, <vscale x 2 x i32> %170, i64 %mul)
  %172 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %171, <vscale x 2 x i32> %43, i64 %mul)
  %173 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %171, <vscale x 2 x i32> %43, i64 %mul)
  %174 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %172, <vscale x 2 x i32> %173, i64 %mul)
  %175 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %65, <vscale x 2 x i32> %174, i64 %mul)
  %176 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %129, <vscale x 2 x i32> %175, i64 %mul)
  %177 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %176, <vscale x 2 x i32> %44, i64 %mul)
  %178 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %176, <vscale x 2 x i32> %45, i64 %mul)
  %179 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %177, <vscale x 2 x i32> %178, i64 %mul)
  %180 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %170, <vscale x 2 x i32> %179, i64 %mul)
  %181 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %174, <vscale x 2 x i32> %180, i64 %mul)
  %182 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %181, <vscale x 2 x i32> %46, i64 %mul)
  %183 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %181, <vscale x 2 x i32> %47, i64 %mul)
  %184 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %182, <vscale x 2 x i32> %183, i64 %mul)
  %185 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %175, <vscale x 2 x i32> %184, i64 %mul)
  %186 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %179, <vscale x 2 x i32> %185, i64 %mul)
  %187 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %186, <vscale x 2 x i32> %48, i64 %mul)
  %188 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %186, <vscale x 2 x i32> %49, i64 %mul)
  %189 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %187, <vscale x 2 x i32> %188, i64 %mul)
  %190 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %120, <vscale x 2 x i32> %69, i64 %mul)
  %191 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %104, <vscale x 2 x i32> %190, i64 %mul)
  %192 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %191, <vscale x 2 x i32> %43, i64 %mul)
  %193 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %191, <vscale x 2 x i32> %43, i64 %mul)
  %194 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %192, <vscale x 2 x i32> %193, i64 %mul)
  %195 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %85, <vscale x 2 x i32> %194, i64 %mul)
  %196 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %69, <vscale x 2 x i32> %195, i64 %mul)
  %197 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %196, <vscale x 2 x i32> %44, i64 %mul)
  %198 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %196, <vscale x 2 x i32> %45, i64 %mul)
  %199 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %197, <vscale x 2 x i32> %198, i64 %mul)
  %200 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %190, <vscale x 2 x i32> %199, i64 %mul)
  %201 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %194, <vscale x 2 x i32> %200, i64 %mul)
  %202 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %201, <vscale x 2 x i32> %46, i64 %mul)
  %203 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %201, <vscale x 2 x i32> %47, i64 %mul)
  %204 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %202, <vscale x 2 x i32> %203, i64 %mul)
  %205 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %195, <vscale x 2 x i32> %204, i64 %mul)
  %206 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %199, <vscale x 2 x i32> %205, i64 %mul)
  %207 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %206, <vscale x 2 x i32> %48, i64 %mul)
  %208 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %206, <vscale x 2 x i32> %49, i64 %mul)
  %209 = tail call <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32> %207, <vscale x 2 x i32> %208, i64 %mul)
  %add29 = add nuw nsw i32 %i.0731, 2
  %cmp27 = icmp ult i32 %i.0731, 18
  br i1 %cmp27, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %div30 = and i64 %32, 9223372036854775807
  %210 = tail call <vscale x 1 x i64> @llvm.epi.vid.nxv1i64(i64 %div30)
  %211 = tail call <vscale x 1 x i64> @llvm.epi.vmv.v.x.nxv1i64.i64(i64 8, i64 %div30)
  %212 = tail call <vscale x 1 x i64> @llvm.epi.vsll.nxv1i64.nxv1i64(<vscale x 1 x i64> %210, <vscale x 1 x i64> %211, i64 %div30)
  %213 = tail call <vscale x 1 x i64> @llvm.epi.vmv.v.x.nxv1i64.i64(i64 1, i64 %div30)
  %214 = tail call <vscale x 1 x i64> @llvm.epi.vsrl.nxv1i64.nxv1i64(<vscale x 1 x i64> %210, <vscale x 1 x i64> %213, i64 %div30)
  %215 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %212, <vscale x 1 x i64> %214, i64 %div30)
  %216 = tail call <vscale x 1 x i64> @llvm.epi.vadd.nxv1i64.nxv1i64(<vscale x 1 x i64> %215, <vscale x 1 x i64> %211, i64 %div30)
  %217 = tail call <vscale x 1 x i64> @llvm.epi.vand.nxv1i64.nxv1i64(<vscale x 1 x i64> %210, <vscale x 1 x i64> %213, i64 %div30)
  %218 = trunc <vscale x 1 x i64> %217 to <vscale x 1 x i1>
  %219 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %215, <vscale x 1 x i64> %216, <vscale x 1 x i1> %218, i64 %div30)
  %220 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %140, <vscale x 2 x i32> %4, i64 %mul)
  %221 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %160, <vscale x 2 x i32> %6, i64 %mul)
  %222 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %180, <vscale x 2 x i32> %8, i64 %mul)
  %223 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %200, <vscale x 2 x i32> %10, i64 %mul)
  %224 = tail call <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32 1, i64 %mul)
  %225 = tail call <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32> %35, <vscale x 2 x i32> %224, i64 %mul)
  %226 = tail call <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32> %225, <vscale x 2 x i32> %224, i64 %mul)
  %227 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %220, <vscale x 2 x i32> %226, i64 %mul)
  %228 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %221, <vscale x 2 x i32> %226, i64 %mul)
  %229 = tail call <vscale x 2 x i32> @llvm.epi.vand.nxv2i32.nxv2i32(<vscale x 2 x i32> %35, <vscale x 2 x i32> %224, i64 %mul)
  %230 = trunc <vscale x 2 x i32> %229 to <vscale x 2 x i1>
  %231 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %227, <vscale x 2 x i32> %228, <vscale x 2 x i1> %230, i64 %mul)
  %232 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %226, <vscale x 2 x i32> %224, i64 %mul)
  %233 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %220, <vscale x 2 x i32> %232, i64 %mul)
  %234 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %221, <vscale x 2 x i32> %232, i64 %mul)
  %235 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %233, <vscale x 2 x i32> %234, <vscale x 2 x i1> %230, i64 %mul)
  %236 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %222, <vscale x 2 x i32> %226, i64 %mul)
  %237 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %223, <vscale x 2 x i32> %226, i64 %mul)
  %238 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %236, <vscale x 2 x i32> %237, <vscale x 2 x i1> %230, i64 %mul)
  %239 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %222, <vscale x 2 x i32> %232, i64 %mul)
  %240 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %223, <vscale x 2 x i32> %232, i64 %mul)
  %241 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %239, <vscale x 2 x i32> %240, <vscale x 2 x i1> %230, i64 %mul)
  %242 = bitcast <vscale x 2 x i32> %231 to <vscale x 1 x i64>
  %243 = bitcast <vscale x 2 x i32> %238 to <vscale x 1 x i64>
  %244 = tail call <vscale x 1 x i64> @llvm.epi.vsll.nxv1i64.nxv1i64(<vscale x 1 x i64> %214, <vscale x 1 x i64> %213, i64 %div30)
  %245 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %242, <vscale x 1 x i64> %244, i64 %div30)
  %246 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %243, <vscale x 1 x i64> %244, i64 %div30)
  %247 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %245, <vscale x 1 x i64> %246, <vscale x 1 x i1> %218, i64 %div30)
  %248 = bitcast <vscale x 2 x i32> %235 to <vscale x 1 x i64>
  %249 = bitcast <vscale x 2 x i32> %241 to <vscale x 1 x i64>
  %250 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %248, <vscale x 1 x i64> %244, i64 %div30)
  %251 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %249, <vscale x 1 x i64> %244, i64 %div30)
  %252 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %250, <vscale x 1 x i64> %251, <vscale x 1 x i1> %218, i64 %div30)
  %253 = tail call <vscale x 1 x i64> @llvm.epi.vadd.nxv1i64.nxv1i64(<vscale x 1 x i64> %244, <vscale x 1 x i64> %213, i64 %div30)
  %254 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %242, <vscale x 1 x i64> %253, i64 %div30)
  %255 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %243, <vscale x 1 x i64> %253, i64 %div30)
  %256 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %254, <vscale x 1 x i64> %255, <vscale x 1 x i1> %218, i64 %div30)
  %257 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %248, <vscale x 1 x i64> %253, i64 %div30)
  %258 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %249, <vscale x 1 x i64> %253, i64 %div30)
  %259 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %257, <vscale x 1 x i64> %258, <vscale x 1 x i1> %218, i64 %div30)
  %260 = bitcast i8* %m.addr.0752 to <vscale x 1 x i64>*
  %261 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* %260, <vscale x 1 x i64> %219, i64 %div30)
  %262 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %247, <vscale x 1 x i64> %261, i64 %div30)
  %263 = bitcast i8* %out.0750 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %262, <vscale x 1 x i64>* %263, <vscale x 1 x i64> %219, i64 %div30)
  %add.ptr57 = getelementptr inbounds i8, i8* %m.addr.0752, i64 64
  %264 = bitcast i8* %add.ptr57 to <vscale x 1 x i64>*
  %265 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nonnull %264, <vscale x 1 x i64> %219, i64 %div30)
  %266 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %252, <vscale x 1 x i64> %265, i64 %div30)
  %add.ptr60 = getelementptr inbounds i8, i8* %out.0750, i64 64
  %267 = bitcast i8* %add.ptr60 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %266, <vscale x 1 x i64>* nonnull %267, <vscale x 1 x i64> %219, i64 %div30)
  %add.ptr62 = getelementptr inbounds i8, i8* %m.addr.0752, i64 128
  %268 = bitcast i8* %add.ptr62 to <vscale x 1 x i64>*
  %269 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nonnull %268, <vscale x 1 x i64> %219, i64 %div30)
  %270 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %256, <vscale x 1 x i64> %269, i64 %div30)
  %add.ptr65 = getelementptr inbounds i8, i8* %out.0750, i64 128
  %271 = bitcast i8* %add.ptr65 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %270, <vscale x 1 x i64>* nonnull %271, <vscale x 1 x i64> %219, i64 %div30)
  %add.ptr67 = getelementptr inbounds i8, i8* %m.addr.0752, i64 192
  %272 = bitcast i8* %add.ptr67 to <vscale x 1 x i64>*
  %273 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nonnull %272, <vscale x 1 x i64> %219, i64 %div30)
  %274 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %259, <vscale x 1 x i64> %273, i64 %div30)
  %add.ptr70 = getelementptr inbounds i8, i8* %out.0750, i64 192
  %275 = bitcast i8* %add.ptr70 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %274, <vscale x 1 x i64>* nonnull %275, <vscale x 1 x i64> %219, i64 %div30)
  %add.ptr72 = getelementptr inbounds i8, i8* %m.addr.0752, i64 16
  %add.ptr73 = getelementptr inbounds i8, i8* %out.0750, i64 16
  %276 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %209, <vscale x 2 x i32> %12, i64 %mul)
  %277 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %149, <vscale x 2 x i32> %14, i64 %mul)
  %278 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %169, <vscale x 2 x i32> %16, i64 %mul)
  %279 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %189, <vscale x 2 x i32> %18, i64 %mul)
  %280 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %276, <vscale x 2 x i32> %226, i64 %mul)
  %281 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %277, <vscale x 2 x i32> %226, i64 %mul)
  %282 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %280, <vscale x 2 x i32> %281, <vscale x 2 x i1> %230, i64 %mul)
  %283 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %276, <vscale x 2 x i32> %232, i64 %mul)
  %284 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %277, <vscale x 2 x i32> %232, i64 %mul)
  %285 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %283, <vscale x 2 x i32> %284, <vscale x 2 x i1> %230, i64 %mul)
  %286 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %278, <vscale x 2 x i32> %226, i64 %mul)
  %287 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %279, <vscale x 2 x i32> %226, i64 %mul)
  %288 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %286, <vscale x 2 x i32> %287, <vscale x 2 x i1> %230, i64 %mul)
  %289 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %278, <vscale x 2 x i32> %232, i64 %mul)
  %290 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %279, <vscale x 2 x i32> %232, i64 %mul)
  %291 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %289, <vscale x 2 x i32> %290, <vscale x 2 x i1> %230, i64 %mul)
  %292 = bitcast <vscale x 2 x i32> %282 to <vscale x 1 x i64>
  %293 = bitcast <vscale x 2 x i32> %288 to <vscale x 1 x i64>
  %294 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %292, <vscale x 1 x i64> %244, i64 %div30)
  %295 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %293, <vscale x 1 x i64> %244, i64 %div30)
  %296 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %294, <vscale x 1 x i64> %295, <vscale x 1 x i1> %218, i64 %div30)
  %297 = bitcast <vscale x 2 x i32> %285 to <vscale x 1 x i64>
  %298 = bitcast <vscale x 2 x i32> %291 to <vscale x 1 x i64>
  %299 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %297, <vscale x 1 x i64> %244, i64 %div30)
  %300 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %298, <vscale x 1 x i64> %244, i64 %div30)
  %301 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %299, <vscale x 1 x i64> %300, <vscale x 1 x i1> %218, i64 %div30)
  %302 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %292, <vscale x 1 x i64> %253, i64 %div30)
  %303 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %293, <vscale x 1 x i64> %253, i64 %div30)
  %304 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %302, <vscale x 1 x i64> %303, <vscale x 1 x i1> %218, i64 %div30)
  %305 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %297, <vscale x 1 x i64> %253, i64 %div30)
  %306 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %298, <vscale x 1 x i64> %253, i64 %div30)
  %307 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %305, <vscale x 1 x i64> %306, <vscale x 1 x i1> %218, i64 %div30)
  %308 = bitcast i8* %add.ptr72 to <vscale x 1 x i64>*
  %309 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nonnull %308, <vscale x 1 x i64> %219, i64 %div30)
  %310 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %296, <vscale x 1 x i64> %309, i64 %div30)
  %311 = bitcast i8* %add.ptr73 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %310, <vscale x 1 x i64>* nonnull %311, <vscale x 1 x i64> %219, i64 %div30)
  %add.ptr99 = getelementptr inbounds i8, i8* %m.addr.0752, i64 80
  %312 = bitcast i8* %add.ptr99 to <vscale x 1 x i64>*
  %313 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nonnull %312, <vscale x 1 x i64> %219, i64 %div30)
  %314 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %301, <vscale x 1 x i64> %313, i64 %div30)
  %add.ptr102 = getelementptr inbounds i8, i8* %out.0750, i64 80
  %315 = bitcast i8* %add.ptr102 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %314, <vscale x 1 x i64>* nonnull %315, <vscale x 1 x i64> %219, i64 %div30)
  %add.ptr104 = getelementptr inbounds i8, i8* %m.addr.0752, i64 144
  %316 = bitcast i8* %add.ptr104 to <vscale x 1 x i64>*
  %317 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nonnull %316, <vscale x 1 x i64> %219, i64 %div30)
  %318 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %304, <vscale x 1 x i64> %317, i64 %div30)
  %add.ptr107 = getelementptr inbounds i8, i8* %out.0750, i64 144
  %319 = bitcast i8* %add.ptr107 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %318, <vscale x 1 x i64>* nonnull %319, <vscale x 1 x i64> %219, i64 %div30)
  %add.ptr109 = getelementptr inbounds i8, i8* %m.addr.0752, i64 208
  %320 = bitcast i8* %add.ptr109 to <vscale x 1 x i64>*
  %321 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nonnull %320, <vscale x 1 x i64> %219, i64 %div30)
  %322 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %307, <vscale x 1 x i64> %321, i64 %div30)
  %add.ptr112 = getelementptr inbounds i8, i8* %out.0750, i64 208
  %323 = bitcast i8* %add.ptr112 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %322, <vscale x 1 x i64>* nonnull %323, <vscale x 1 x i64> %219, i64 %div30)
  %add.ptr114 = getelementptr inbounds i8, i8* %m.addr.0752, i64 32
  %add.ptr115 = getelementptr inbounds i8, i8* %out.0750, i64 32
  %324 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %185, <vscale x 2 x i32> %20, i64 %mul)
  %325 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %205, <vscale x 2 x i32> %22, i64 %mul)
  %326 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %145, <vscale x 2 x i32> %24, i64 %mul)
  %327 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %165, <vscale x 2 x i32> %26, i64 %mul)
  %328 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %324, <vscale x 2 x i32> %226, i64 %mul)
  %329 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %325, <vscale x 2 x i32> %226, i64 %mul)
  %330 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %328, <vscale x 2 x i32> %329, <vscale x 2 x i1> %230, i64 %mul)
  %331 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %324, <vscale x 2 x i32> %232, i64 %mul)
  %332 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %325, <vscale x 2 x i32> %232, i64 %mul)
  %333 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %331, <vscale x 2 x i32> %332, <vscale x 2 x i1> %230, i64 %mul)
  %334 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %326, <vscale x 2 x i32> %226, i64 %mul)
  %335 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %327, <vscale x 2 x i32> %226, i64 %mul)
  %336 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %334, <vscale x 2 x i32> %335, <vscale x 2 x i1> %230, i64 %mul)
  %337 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %326, <vscale x 2 x i32> %232, i64 %mul)
  %338 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %327, <vscale x 2 x i32> %232, i64 %mul)
  %339 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %337, <vscale x 2 x i32> %338, <vscale x 2 x i1> %230, i64 %mul)
  %340 = bitcast <vscale x 2 x i32> %330 to <vscale x 1 x i64>
  %341 = bitcast <vscale x 2 x i32> %336 to <vscale x 1 x i64>
  %342 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %340, <vscale x 1 x i64> %244, i64 %div30)
  %343 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %341, <vscale x 1 x i64> %244, i64 %div30)
  %344 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %342, <vscale x 1 x i64> %343, <vscale x 1 x i1> %218, i64 %div30)
  %345 = bitcast <vscale x 2 x i32> %333 to <vscale x 1 x i64>
  %346 = bitcast <vscale x 2 x i32> %339 to <vscale x 1 x i64>
  %347 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %345, <vscale x 1 x i64> %244, i64 %div30)
  %348 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %346, <vscale x 1 x i64> %244, i64 %div30)
  %349 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %347, <vscale x 1 x i64> %348, <vscale x 1 x i1> %218, i64 %div30)
  %350 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %340, <vscale x 1 x i64> %253, i64 %div30)
  %351 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %341, <vscale x 1 x i64> %253, i64 %div30)
  %352 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %350, <vscale x 1 x i64> %351, <vscale x 1 x i1> %218, i64 %div30)
  %353 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %345, <vscale x 1 x i64> %253, i64 %div30)
  %354 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %346, <vscale x 1 x i64> %253, i64 %div30)
  %355 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %353, <vscale x 1 x i64> %354, <vscale x 1 x i1> %218, i64 %div30)
  %356 = bitcast i8* %add.ptr114 to <vscale x 1 x i64>*
  %357 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nonnull %356, <vscale x 1 x i64> %219, i64 %div30)
  %358 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %344, <vscale x 1 x i64> %357, i64 %div30)
  %359 = bitcast i8* %add.ptr115 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %358, <vscale x 1 x i64>* nonnull %359, <vscale x 1 x i64> %219, i64 %div30)
  %add.ptr141 = getelementptr inbounds i8, i8* %m.addr.0752, i64 96
  %360 = bitcast i8* %add.ptr141 to <vscale x 1 x i64>*
  %361 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nonnull %360, <vscale x 1 x i64> %219, i64 %div30)
  %362 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %349, <vscale x 1 x i64> %361, i64 %div30)
  %add.ptr144 = getelementptr inbounds i8, i8* %out.0750, i64 96
  %363 = bitcast i8* %add.ptr144 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %362, <vscale x 1 x i64>* nonnull %363, <vscale x 1 x i64> %219, i64 %div30)
  %add.ptr146 = getelementptr inbounds i8, i8* %m.addr.0752, i64 160
  %364 = bitcast i8* %add.ptr146 to <vscale x 1 x i64>*
  %365 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nonnull %364, <vscale x 1 x i64> %219, i64 %div30)
  %366 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %352, <vscale x 1 x i64> %365, i64 %div30)
  %add.ptr149 = getelementptr inbounds i8, i8* %out.0750, i64 160
  %367 = bitcast i8* %add.ptr149 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %366, <vscale x 1 x i64>* nonnull %367, <vscale x 1 x i64> %219, i64 %div30)
  %add.ptr151 = getelementptr inbounds i8, i8* %m.addr.0752, i64 224
  %368 = bitcast i8* %add.ptr151 to <vscale x 1 x i64>*
  %369 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nonnull %368, <vscale x 1 x i64> %219, i64 %div30)
  %370 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %355, <vscale x 1 x i64> %369, i64 %div30)
  %add.ptr154 = getelementptr inbounds i8, i8* %out.0750, i64 224
  %371 = bitcast i8* %add.ptr154 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %370, <vscale x 1 x i64>* nonnull %371, <vscale x 1 x i64> %219, i64 %div30)
  %add.ptr156 = getelementptr inbounds i8, i8* %m.addr.0752, i64 48
  %add.ptr157 = getelementptr inbounds i8, i8* %out.0750, i64 48
  %372 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %164, <vscale x 2 x i32> %39, i64 %mul)
  %373 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %184, <vscale x 2 x i32> %42, i64 %mul)
  %374 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %204, <vscale x 2 x i32> %28, i64 %mul)
  %375 = tail call <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32> %144, <vscale x 2 x i32> %30, i64 %mul)
  %376 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %372, <vscale x 2 x i32> %226, i64 %mul)
  %377 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %373, <vscale x 2 x i32> %226, i64 %mul)
  %378 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %376, <vscale x 2 x i32> %377, <vscale x 2 x i1> %230, i64 %mul)
  %379 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %372, <vscale x 2 x i32> %232, i64 %mul)
  %380 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %373, <vscale x 2 x i32> %232, i64 %mul)
  %381 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %379, <vscale x 2 x i32> %380, <vscale x 2 x i1> %230, i64 %mul)
  %382 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %374, <vscale x 2 x i32> %226, i64 %mul)
  %383 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %375, <vscale x 2 x i32> %226, i64 %mul)
  %384 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %382, <vscale x 2 x i32> %383, <vscale x 2 x i1> %230, i64 %mul)
  %385 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %374, <vscale x 2 x i32> %232, i64 %mul)
  %386 = tail call <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32> %375, <vscale x 2 x i32> %232, i64 %mul)
  %387 = tail call <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32> %385, <vscale x 2 x i32> %386, <vscale x 2 x i1> %230, i64 %mul)
  %388 = bitcast <vscale x 2 x i32> %378 to <vscale x 1 x i64>
  %389 = bitcast <vscale x 2 x i32> %384 to <vscale x 1 x i64>
  %390 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %388, <vscale x 1 x i64> %244, i64 %div30)
  %391 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %389, <vscale x 1 x i64> %244, i64 %div30)
  %392 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %390, <vscale x 1 x i64> %391, <vscale x 1 x i1> %218, i64 %div30)
  %393 = bitcast <vscale x 2 x i32> %381 to <vscale x 1 x i64>
  %394 = bitcast <vscale x 2 x i32> %387 to <vscale x 1 x i64>
  %395 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %393, <vscale x 1 x i64> %244, i64 %div30)
  %396 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %394, <vscale x 1 x i64> %244, i64 %div30)
  %397 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %395, <vscale x 1 x i64> %396, <vscale x 1 x i1> %218, i64 %div30)
  %398 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %388, <vscale x 1 x i64> %253, i64 %div30)
  %399 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %389, <vscale x 1 x i64> %253, i64 %div30)
  %400 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %398, <vscale x 1 x i64> %399, <vscale x 1 x i1> %218, i64 %div30)
  %401 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %393, <vscale x 1 x i64> %253, i64 %div30)
  %402 = tail call <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64> %394, <vscale x 1 x i64> %253, i64 %div30)
  %403 = tail call <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64> %401, <vscale x 1 x i64> %402, <vscale x 1 x i1> %218, i64 %div30)
  %404 = bitcast i8* %add.ptr156 to <vscale x 1 x i64>*
  %405 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nonnull %404, <vscale x 1 x i64> %219, i64 %div30)
  %406 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %392, <vscale x 1 x i64> %405, i64 %div30)
  %407 = bitcast i8* %add.ptr157 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %406, <vscale x 1 x i64>* nonnull %407, <vscale x 1 x i64> %219, i64 %div30)
  %add.ptr183 = getelementptr inbounds i8, i8* %m.addr.0752, i64 112
  %408 = bitcast i8* %add.ptr183 to <vscale x 1 x i64>*
  %409 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nonnull %408, <vscale x 1 x i64> %219, i64 %div30)
  %410 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %397, <vscale x 1 x i64> %409, i64 %div30)
  %add.ptr186 = getelementptr inbounds i8, i8* %out.0750, i64 112
  %411 = bitcast i8* %add.ptr186 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %410, <vscale x 1 x i64>* nonnull %411, <vscale x 1 x i64> %219, i64 %div30)
  %add.ptr188 = getelementptr inbounds i8, i8* %m.addr.0752, i64 176
  %412 = bitcast i8* %add.ptr188 to <vscale x 1 x i64>*
  %413 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nonnull %412, <vscale x 1 x i64> %219, i64 %div30)
  %414 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %400, <vscale x 1 x i64> %413, i64 %div30)
  %add.ptr191 = getelementptr inbounds i8, i8* %out.0750, i64 176
  %415 = bitcast i8* %add.ptr191 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %414, <vscale x 1 x i64>* nonnull %415, <vscale x 1 x i64> %219, i64 %div30)
  %add.ptr193 = getelementptr inbounds i8, i8* %m.addr.0752, i64 240
  %416 = bitcast i8* %add.ptr193 to <vscale x 1 x i64>*
  %417 = tail call <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nonnull %416, <vscale x 1 x i64> %219, i64 %div30)
  %418 = tail call <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64> %403, <vscale x 1 x i64> %417, i64 %div30)
  %add.ptr196 = getelementptr inbounds i8, i8* %out.0750, i64 240
  %419 = bitcast i8* %add.ptr196 to <vscale x 1 x i64>*
  tail call void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64> %418, <vscale x 1 x i64>* nonnull %419, <vscale x 1 x i64> %219, i64 %div30)
  %mul200 = shl i64 %32, 7
  %420 = trunc i64 %mul200 to i32
  %conv202 = sub i32 %bytes.addr.0751, %420
  %add.ptr204 = getelementptr inbounds i8, i8* %out.0750, i64 %mul200
  %add.ptr206 = getelementptr inbounds i8, i8* %m.addr.0752, i64 %mul200
  %cmp15 = icmp ugt i32 %conv202, 255
  br i1 %cmp15, label %while.body, label %if.end207

if.end207:                                        ; preds = %for.end
  %tobool208.not = icmp eq i32 %conv202, 0
  br i1 %tobool208.not, label %cleanup, label %for.cond211.preheader

for.cond211.preheader:                            ; preds = %if.end, %if.end207
  %m.addr.1792 = phi i8* [ %add.ptr206, %if.end207 ], [ %m, %if.end ]
  %bytes.addr.1791 = phi i32 [ %conv202, %if.end207 ], [ %bytes, %if.end ]
  %out.1790 = phi i8* [ %add.ptr204, %if.end207 ], [ %c_, %if.end ]
  %input560.i = bitcast %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_ to i8*
  %421 = bitcast [16 x i32]* %x.i to i8*
  %arrayidx6.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 0
  %arrayidx7.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 4
  %arrayidx9.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 12
  %arrayidx15.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 8
  %arrayidx57.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 1
  %arrayidx58.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 5
  %arrayidx61.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 13
  %arrayidx71.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 9
  %arrayidx113.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 2
  %arrayidx114.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 6
  %arrayidx117.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 14
  %arrayidx127.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 10
  %arrayidx169.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 3
  %arrayidx170.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 7
  %arrayidx173.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 15
  %arrayidx183.i = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 11
  %arrayidx457.1.i = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 1
  %arrayidx457.2.i = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 2
  %arrayidx457.3.i = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 3
  %arrayidx457.4.i = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 4
  %arrayidx457.5.i = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 5
  %arrayidx457.6.i = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 6
  %arrayidx457.7.i = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 7
  %arrayidx457.8.i = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 8
  %arrayidx457.9.i = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 9
  %arrayidx457.10.i = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 10
  %arrayidx457.11.i = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 11
  %arrayidx457.12.i = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 12
  %arrayidx457.13.i = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 13
  %arrayidx457.14.i = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 14
  %arrayidx457.15.i = getelementptr inbounds %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx, %struct.crypto_stream_chacha20_dolbeau_riscv_v_ECRYPT_ctx* %x_, i64 0, i32 0, i64 15
  %arrayidx482.i726 = getelementptr inbounds [64 x i8], [64 x i8]* %output, i64 0, i64 1
  %arrayidx493.i727 = getelementptr inbounds [64 x i8], [64 x i8]* %output, i64 0, i64 2
  %arrayidx504.i728 = getelementptr inbounds [64 x i8], [64 x i8]* %output, i64 0, i64 3
  %add.ptr.i823 = getelementptr inbounds [64 x i8], [64 x i8]* %output, i64 0, i64 4
  %arrayidx482.i824 = getelementptr inbounds [64 x i8], [64 x i8]* %output, i64 0, i64 5
  %arrayidx493.i825 = getelementptr inbounds [64 x i8], [64 x i8]* %output, i64 0, i64 6
  %arrayidx504.i826 = getelementptr inbounds [64 x i8], [64 x i8]* %output, i64 0, i64 7
  br label %for.cond211

for.cond211:                                      ; preds = %for.cond211.preheader, %for.end256
  %out.2 = phi i8* [ %add.ptr258, %for.end256 ], [ %out.1790, %for.cond211.preheader ]
  %bytes.addr.2 = phi i32 [ %sub257, %for.end256 ], [ %bytes.addr.1791, %for.cond211.preheader ]
  %m.addr.2 = phi i8* [ %add.ptr259, %for.end256 ], [ %m.addr.1792, %for.cond211.preheader ]
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 4 dereferenceable(64) %421, i8* nonnull align 4 dereferenceable(64) %input560.i, i64 64, i1 false)
  %arrayidx6.promoted.i = load i32, i32* %arrayidx6.i, align 4
  %arrayidx7.promoted.i = load i32, i32* %arrayidx7.i, align 4
  %arrayidx9.promoted.i = load i32, i32* %arrayidx9.i, align 4
  %arrayidx15.promoted.i = load i32, i32* %arrayidx15.i, align 4
  %arrayidx57.promoted.i = load i32, i32* %arrayidx57.i, align 4
  %arrayidx58.promoted.i = load i32, i32* %arrayidx58.i, align 4
  %arrayidx61.promoted.i = load i32, i32* %arrayidx61.i, align 4
  %arrayidx71.promoted.i = load i32, i32* %arrayidx71.i, align 4
  %arrayidx113.promoted.i = load i32, i32* %arrayidx113.i, align 4
  %arrayidx114.promoted.i = load i32, i32* %arrayidx114.i, align 4
  %arrayidx117.promoted.i = load i32, i32* %arrayidx117.i, align 4
  %arrayidx127.promoted.i = load i32, i32* %arrayidx127.i, align 4
  %arrayidx169.promoted.i = load i32, i32* %arrayidx169.i, align 4
  %arrayidx170.promoted.i = load i32, i32* %arrayidx170.i, align 4
  %arrayidx173.promoted.i = load i32, i32* %arrayidx173.i, align 4
  %arrayidx183.promoted.i = load i32, i32* %arrayidx183.i, align 4
  br label %for.body5.i

for.cond451.preheader.i:                          ; preds = %for.body5.i
  %422 = load i32, i32* %1, align 4
  %add458.i = add i32 %422, %add255.i
  store i32 %add458.i, i32* %arrayidx6.i, align 4
  %423 = load i32, i32* %arrayidx457.1.i, align 4
  %add458.1.i = add i32 %423, %add311.i
  store i32 %add458.1.i, i32* %arrayidx57.i, align 4
  %424 = load i32, i32* %arrayidx457.2.i, align 4
  %add458.2.i = add i32 %424, %add367.i
  store i32 %add458.2.i, i32* %arrayidx113.i, align 4
  %425 = load i32, i32* %arrayidx457.3.i, align 4
  %add458.3.i = add i32 %425, %add423.i
  store i32 %add458.3.i, i32* %arrayidx169.i, align 4
  %426 = load i32, i32* %arrayidx457.4.i, align 4
  %add458.4.i = add i32 %426, %or447.i
  store i32 %add458.4.i, i32* %arrayidx7.i, align 4
  %427 = load i32, i32* %arrayidx457.5.i, align 4
  %add458.5.i = add i32 %427, %or279.i
  store i32 %add458.5.i, i32* %arrayidx58.i, align 4
  %428 = load i32, i32* %arrayidx457.6.i, align 4
  %add458.6.i = add i32 %428, %or335.i
  store i32 %add458.6.i, i32* %arrayidx114.i, align 4
  %429 = load i32, i32* %arrayidx457.7.i, align 4
  %add458.7.i = add i32 %429, %or391.i
  store i32 %add458.7.i, i32* %arrayidx170.i, align 4
  %430 = load i32, i32* %arrayidx457.8.i, align 4
  %add458.8.i = add i32 %430, %add381.i
  store i32 %add458.8.i, i32* %arrayidx15.i, align 4
  %431 = load i32, i32* %arrayidx457.9.i, align 4
  %add458.9.i = add i32 %431, %add437.i
  store i32 %add458.9.i, i32* %arrayidx71.i, align 4
  %432 = load i32, i32* %arrayidx457.10.i, align 4
  %add458.10.i = add i32 %432, %add269.i
  store i32 %add458.10.i, i32* %arrayidx127.i, align 4
  %433 = load i32, i32* %arrayidx457.11.i, align 4
  %add458.11.i = add i32 %433, %add325.i
  store i32 %add458.11.i, i32* %arrayidx183.i, align 4
  %434 = load i32, i32* %arrayidx457.12.i, align 4
  %add458.12.i = add i32 %434, %or321.i
  store i32 %add458.12.i, i32* %arrayidx9.i, align 4
  %435 = load i32, i32* %arrayidx457.13.i, align 4
  %add458.13.i = add i32 %435, %or377.i
  store i32 %add458.13.i, i32* %arrayidx61.i, align 4
  %436 = load i32, i32* %arrayidx457.14.i, align 4
  %add458.14.i = add i32 %436, %or433.i
  store i32 %add458.14.i, i32* %arrayidx117.i, align 4
  %437 = load i32, i32* %arrayidx457.15.i, align 4
  %add458.15.i = add i32 %437, %or265.i
  store i32 %add458.15.i, i32* %arrayidx173.i, align 4
  %extract.t.i = trunc i32 %add458.i to i8
  %extract.i = lshr i32 %add458.i, 8
  %extract.t562.i = trunc i32 %extract.i to i8
  %extract565.i = lshr i32 %add458.i, 16
  %extract.t566.i = trunc i32 %extract565.i to i8
  %extract569.i = lshr i32 %add458.i, 24
  %extract.t570.i = trunc i32 %extract569.i to i8
  store i8 %extract.t.i, i8* %0, align 1
  store i8 %extract.t562.i, i8* %arrayidx482.i726, align 1
  store i8 %extract.t566.i, i8* %arrayidx493.i727, align 1
  store i8 %extract.t570.i, i8* %arrayidx504.i728, align 1
  %extract.t = trunc i32 %add458.1.i to i8
  %extract = lshr i32 %add458.1.i, 8
  %extract.t775 = trunc i32 %extract to i8
  %extract778 = lshr i32 %add458.1.i, 16
  %extract.t779 = trunc i32 %extract778 to i8
  %extract782 = lshr i32 %add458.1.i, 24
  %extract.t783 = trunc i32 %extract782 to i8
  store i8 %extract.t, i8* %add.ptr.i823, align 1
  store i8 %extract.t775, i8* %arrayidx482.i824, align 1
  store i8 %extract.t779, i8* %arrayidx493.i825, align 1
  store i8 %extract.t783, i8* %arrayidx504.i826, align 1
  br label %do.body.do.body_crit_edge.i.do.body.do.body_crit_edge.i_crit_edge

for.body5.i:                                      ; preds = %for.body5.i, %for.cond211
  %add325552.i = phi i32 [ %arrayidx183.promoted.i, %for.cond211 ], [ %add325.i, %for.body5.i ]
  %or265551.i = phi i32 [ %arrayidx173.promoted.i, %for.cond211 ], [ %or265.i, %for.body5.i ]
  %or391550.i = phi i32 [ %arrayidx170.promoted.i, %for.cond211 ], [ %or391.i, %for.body5.i ]
  %add423549.i = phi i32 [ %arrayidx169.promoted.i, %for.cond211 ], [ %add423.i, %for.body5.i ]
  %add269548.i = phi i32 [ %arrayidx127.promoted.i, %for.cond211 ], [ %add269.i, %for.body5.i ]
  %or433547.i = phi i32 [ %arrayidx117.promoted.i, %for.cond211 ], [ %or433.i, %for.body5.i ]
  %or335546.i = phi i32 [ %arrayidx114.promoted.i, %for.cond211 ], [ %or335.i, %for.body5.i ]
  %add367545.i = phi i32 [ %arrayidx113.promoted.i, %for.cond211 ], [ %add367.i, %for.body5.i ]
  %add437544.i = phi i32 [ %arrayidx71.promoted.i, %for.cond211 ], [ %add437.i, %for.body5.i ]
  %or377543.i = phi i32 [ %arrayidx61.promoted.i, %for.cond211 ], [ %or377.i, %for.body5.i ]
  %or279542.i = phi i32 [ %arrayidx58.promoted.i, %for.cond211 ], [ %or279.i, %for.body5.i ]
  %add311541.i = phi i32 [ %arrayidx57.promoted.i, %for.cond211 ], [ %add311.i, %for.body5.i ]
  %add381540.i = phi i32 [ %arrayidx15.promoted.i, %for.cond211 ], [ %add381.i, %for.body5.i ]
  %or321539.i = phi i32 [ %arrayidx9.promoted.i, %for.cond211 ], [ %or321.i, %for.body5.i ]
  %or447538.i = phi i32 [ %arrayidx7.promoted.i, %for.cond211 ], [ %or447.i, %for.body5.i ]
  %add255537.i = phi i32 [ %arrayidx6.promoted.i, %for.cond211 ], [ %add255.i, %for.body5.i ]
  %i.1535.i = phi i32 [ 20, %for.cond211 ], [ %sub.i, %for.body5.i ]
  %add.i = add i32 %add255537.i, %or447538.i
  %xor.i = xor i32 %add.i, %or321539.i
  %shl.i = shl i32 %xor.i, 16
  %shr.i = lshr i32 %xor.i, 16
  %or.i = or i32 %shl.i, %shr.i
  %add17.i = add i32 %or.i, %add381540.i
  %xor21.i = xor i32 %add17.i, %or447538.i
  %shl22.i = shl i32 %xor21.i, 12
  %shr26.i = lshr i32 %xor21.i, 20
  %or27.i = or i32 %shl22.i, %shr26.i
  %add31.i = add i32 %or27.i, %add.i
  %xor35.i = xor i32 %add31.i, %or.i
  %shl36.i = shl i32 %xor35.i, 8
  %shr40.i = lshr i32 %xor35.i, 24
  %or41.i = or i32 %shl36.i, %shr40.i
  %add45.i = add i32 %or41.i, %add17.i
  %xor49.i = xor i32 %add45.i, %or27.i
  %shl50.i = shl i32 %xor49.i, 7
  %shr54.i = lshr i32 %xor49.i, 25
  %or55.i = or i32 %shl50.i, %shr54.i
  %add59.i = add i32 %add311541.i, %or279542.i
  %xor63.i = xor i32 %add59.i, %or377543.i
  %shl64.i = shl i32 %xor63.i, 16
  %shr68.i = lshr i32 %xor63.i, 16
  %or69.i = or i32 %shl64.i, %shr68.i
  %add73.i = add i32 %or69.i, %add437544.i
  %xor77.i = xor i32 %add73.i, %or279542.i
  %shl78.i = shl i32 %xor77.i, 12
  %shr82.i = lshr i32 %xor77.i, 20
  %or83.i = or i32 %shl78.i, %shr82.i
  %add87.i = add i32 %or83.i, %add59.i
  %xor91.i = xor i32 %add87.i, %or69.i
  %shl92.i = shl i32 %xor91.i, 8
  %shr96.i = lshr i32 %xor91.i, 24
  %or97.i = or i32 %shl92.i, %shr96.i
  %add101.i = add i32 %or97.i, %add73.i
  %xor105.i = xor i32 %add101.i, %or83.i
  %shl106.i = shl i32 %xor105.i, 7
  %shr110.i = lshr i32 %xor105.i, 25
  %or111.i = or i32 %shl106.i, %shr110.i
  %add115.i = add i32 %add367545.i, %or335546.i
  %xor119.i = xor i32 %add115.i, %or433547.i
  %shl120.i = shl i32 %xor119.i, 16
  %shr124.i = lshr i32 %xor119.i, 16
  %or125.i = or i32 %shl120.i, %shr124.i
  %add129.i = add i32 %or125.i, %add269548.i
  %xor133.i = xor i32 %add129.i, %or335546.i
  %shl134.i = shl i32 %xor133.i, 12
  %shr138.i = lshr i32 %xor133.i, 20
  %or139.i = or i32 %shl134.i, %shr138.i
  %add143.i = add i32 %or139.i, %add115.i
  %xor147.i = xor i32 %add143.i, %or125.i
  %shl148.i = shl i32 %xor147.i, 8
  %shr152.i = lshr i32 %xor147.i, 24
  %or153.i = or i32 %shl148.i, %shr152.i
  %add157.i = add i32 %or153.i, %add129.i
  %xor161.i = xor i32 %add157.i, %or139.i
  %shl162.i = shl i32 %xor161.i, 7
  %shr166.i = lshr i32 %xor161.i, 25
  %or167.i = or i32 %shl162.i, %shr166.i
  %add171.i = add i32 %add423549.i, %or391550.i
  %xor175.i = xor i32 %add171.i, %or265551.i
  %shl176.i = shl i32 %xor175.i, 16
  %shr180.i = lshr i32 %xor175.i, 16
  %or181.i = or i32 %shl176.i, %shr180.i
  %add185.i = add i32 %or181.i, %add325552.i
  %xor189.i = xor i32 %add185.i, %or391550.i
  %shl190.i = shl i32 %xor189.i, 12
  %shr194.i = lshr i32 %xor189.i, 20
  %or195.i = or i32 %shl190.i, %shr194.i
  %add199.i = add i32 %or195.i, %add171.i
  %xor203.i = xor i32 %add199.i, %or181.i
  %shl204.i = shl i32 %xor203.i, 8
  %shr208.i = lshr i32 %xor203.i, 24
  %or209.i = or i32 %shl204.i, %shr208.i
  %add213.i = add i32 %or209.i, %add185.i
  %xor217.i = xor i32 %add213.i, %or195.i
  %shl218.i = shl i32 %xor217.i, 7
  %shr222.i = lshr i32 %xor217.i, 25
  %or223.i = or i32 %shl218.i, %shr222.i
  %add227.i = add i32 %or111.i, %add31.i
  %xor231.i = xor i32 %add227.i, %or209.i
  %shl232.i = shl i32 %xor231.i, 16
  %shr236.i = lshr i32 %xor231.i, 16
  %or237.i = or i32 %shl232.i, %shr236.i
  %add241.i = add i32 %or237.i, %add157.i
  %xor245.i = xor i32 %add241.i, %or111.i
  %shl246.i = shl i32 %xor245.i, 12
  %shr250.i = lshr i32 %xor245.i, 20
  %or251.i = or i32 %shl246.i, %shr250.i
  %add255.i = add i32 %or251.i, %add227.i
  %xor259.i = xor i32 %add255.i, %or237.i
  %shl260.i = shl i32 %xor259.i, 8
  %shr264.i = lshr i32 %xor259.i, 24
  %or265.i = or i32 %shl260.i, %shr264.i
  %add269.i = add i32 %or265.i, %add241.i
  %xor273.i = xor i32 %add269.i, %or251.i
  %shl274.i = shl i32 %xor273.i, 7
  %shr278.i = lshr i32 %xor273.i, 25
  %or279.i = or i32 %shl274.i, %shr278.i
  %add283.i = add i32 %or167.i, %add87.i
  %xor287.i = xor i32 %or41.i, %add283.i
  %shl288.i = shl i32 %xor287.i, 16
  %shr292.i = lshr i32 %xor287.i, 16
  %or293.i = or i32 %shl288.i, %shr292.i
  %add297.i = add i32 %or293.i, %add213.i
  %xor301.i = xor i32 %add297.i, %or167.i
  %shl302.i = shl i32 %xor301.i, 12
  %shr306.i = lshr i32 %xor301.i, 20
  %or307.i = or i32 %shl302.i, %shr306.i
  %add311.i = add i32 %or307.i, %add283.i
  %xor315.i = xor i32 %add311.i, %or293.i
  %shl316.i = shl i32 %xor315.i, 8
  %shr320.i = lshr i32 %xor315.i, 24
  %or321.i = or i32 %shl316.i, %shr320.i
  %add325.i = add i32 %or321.i, %add297.i
  %xor329.i = xor i32 %add325.i, %or307.i
  %shl330.i = shl i32 %xor329.i, 7
  %shr334.i = lshr i32 %xor329.i, 25
  %or335.i = or i32 %shl330.i, %shr334.i
  %add339.i = add i32 %or223.i, %add143.i
  %xor343.i = xor i32 %or97.i, %add339.i
  %shl344.i = shl i32 %xor343.i, 16
  %shr348.i = lshr i32 %xor343.i, 16
  %or349.i = or i32 %shl344.i, %shr348.i
  %add353.i = add i32 %add45.i, %or349.i
  %xor357.i = xor i32 %add353.i, %or223.i
  %shl358.i = shl i32 %xor357.i, 12
  %shr362.i = lshr i32 %xor357.i, 20
  %or363.i = or i32 %shl358.i, %shr362.i
  %add367.i = add i32 %or363.i, %add339.i
  %xor371.i = xor i32 %add367.i, %or349.i
  %shl372.i = shl i32 %xor371.i, 8
  %shr376.i = lshr i32 %xor371.i, 24
  %or377.i = or i32 %shl372.i, %shr376.i
  %add381.i = add i32 %or377.i, %add353.i
  %xor385.i = xor i32 %add381.i, %or363.i
  %shl386.i = shl i32 %xor385.i, 7
  %shr390.i = lshr i32 %xor385.i, 25
  %or391.i = or i32 %shl386.i, %shr390.i
  %add395.i = add i32 %or55.i, %add199.i
  %xor399.i = xor i32 %add395.i, %or153.i
  %shl400.i = shl i32 %xor399.i, 16
  %shr404.i = lshr i32 %xor399.i, 16
  %or405.i = or i32 %shl400.i, %shr404.i
  %add409.i = add i32 %or405.i, %add101.i
  %xor413.i = xor i32 %add409.i, %or55.i
  %shl414.i = shl i32 %xor413.i, 12
  %shr418.i = lshr i32 %xor413.i, 20
  %or419.i = or i32 %shl414.i, %shr418.i
  %add423.i = add i32 %or419.i, %add395.i
  %xor427.i = xor i32 %add423.i, %or405.i
  %shl428.i = shl i32 %xor427.i, 8
  %shr432.i = lshr i32 %xor427.i, 24
  %or433.i = or i32 %shl428.i, %shr432.i
  %add437.i = add i32 %or433.i, %add409.i
  %xor441.i = xor i32 %add437.i, %or419.i
  %shl442.i = shl i32 %xor441.i, 7
  %shr446.i = lshr i32 %xor441.i, 25
  %or447.i = or i32 %shl442.i, %shr446.i
  %sub.i = add nsw i32 %i.1535.i, -2
  %cmp4.i = icmp ugt i32 %i.1535.i, 2
  br i1 %cmp4.i, label %for.body5.i, label %for.cond451.preheader.i

do.body.do.body_crit_edge.i.do.body.do.body_crit_edge.i_crit_edge: ; preds = %for.cond451.preheader.i, %do.body.do.body_crit_edge.i.do.body.do.body_crit_edge.i_crit_edge
  %indvars.iv.next.i827 = phi i64 [ 2, %for.cond451.preheader.i ], [ %indvars.iv.next.i, %do.body.do.body_crit_edge.i.do.body.do.body_crit_edge.i_crit_edge ]
  %arrayidx468.phi.trans.insert.i.phi.trans.insert = getelementptr inbounds [16 x i32], [16 x i32]* %x.i, i64 0, i64 %indvars.iv.next.i827
  %.pre.i.pre = load i32, i32* %arrayidx468.phi.trans.insert.i.phi.trans.insert, align 4
  %extract.t774 = trunc i32 %.pre.i.pre to i8
  %extract776 = lshr i32 %.pre.i.pre, 8
  %extract.t777 = trunc i32 %extract776 to i8
  %extract780 = lshr i32 %.pre.i.pre, 16
  %extract.t781 = trunc i32 %extract780 to i8
  %extract784 = lshr i32 %.pre.i.pre, 24
  %extract.t785 = trunc i32 %extract784 to i8
  %438 = shl nuw nsw i64 %indvars.iv.next.i827, 2
  %add.ptr.i = getelementptr inbounds [64 x i8], [64 x i8]* %output, i64 0, i64 %438
  store i8 %extract.t774, i8* %add.ptr.i, align 1
  %arrayidx482.i = getelementptr inbounds i8, i8* %add.ptr.i, i64 1
  store i8 %extract.t777, i8* %arrayidx482.i, align 1
  %arrayidx493.i = getelementptr inbounds i8, i8* %add.ptr.i, i64 2
  store i8 %extract.t781, i8* %arrayidx493.i, align 1
  %arrayidx504.i = getelementptr inbounds i8, i8* %add.ptr.i, i64 3
  store i8 %extract.t785, i8* %arrayidx504.i, align 1
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.next.i827, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, 16
  br i1 %exitcond.not.i, label %salsa20_wordtobyte.exit, label %do.body.do.body_crit_edge.i.do.body.do.body_crit_edge.i_crit_edge

salsa20_wordtobyte.exit:                          ; preds = %do.body.do.body_crit_edge.i.do.body.do.body_crit_edge.i_crit_edge
  %add213 = add i32 %434, 1
  store i32 %add213, i32* %arrayidx457.12.i, align 4
  %tobool216.not = icmp eq i32 %add213, 0
  br i1 %tobool216.not, label %if.then217, label %if.end221

if.then217:                                       ; preds = %salsa20_wordtobyte.exit
  %add219 = add i32 %435, 1
  store i32 %add219, i32* %arrayidx457.13.i, align 4
  br label %if.end221

if.end221:                                        ; preds = %if.then217, %salsa20_wordtobyte.exit
  %cmp222 = icmp ult i32 %bytes.addr.2, 65
  br i1 %cmp222, label %for.cond225.preheader, label %for.body243

for.cond225.preheader:                            ; preds = %if.end221
  %cmp226723.not = icmp eq i32 %bytes.addr.2, 0
  br i1 %cmp226723.not, label %cleanup, label %for.body228.preheader

for.body228.preheader:                            ; preds = %for.cond225.preheader
  %wide.trip.count = zext i32 %bytes.addr.2 to i64
  br label %for.body228

for.body228:                                      ; preds = %for.body228.preheader, %for.body228
  %indvars.iv = phi i64 [ 0, %for.body228.preheader ], [ %indvars.iv.next, %for.body228 ]
  %arrayidx229 = getelementptr inbounds i8, i8* %m.addr.2, i64 %indvars.iv
  %439 = load i8, i8* %arrayidx229, align 1
  %arrayidx232 = getelementptr inbounds [64 x i8], [64 x i8]* %output, i64 0, i64 %indvars.iv
  %440 = load i8, i8* %arrayidx232, align 1
  %xor722 = xor i8 %440, %439
  %arrayidx236 = getelementptr inbounds i8, i8* %out.2, i64 %indvars.iv
  store i8 %xor722, i8* %arrayidx236, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %cleanup, label %for.body228

for.body243:                                      ; preds = %if.end221, %for.body243
  %indvars.iv770 = phi i64 [ %indvars.iv.next771, %for.body243 ], [ 0, %if.end221 ]
  %arrayidx245 = getelementptr inbounds i8, i8* %m.addr.2, i64 %indvars.iv770
  %441 = load i8, i8* %arrayidx245, align 1
  %arrayidx248 = getelementptr inbounds [64 x i8], [64 x i8]* %output, i64 0, i64 %indvars.iv770
  %442 = load i8, i8* %arrayidx248, align 1
  %xor250721 = xor i8 %442, %441
  %arrayidx253 = getelementptr inbounds i8, i8* %out.2, i64 %indvars.iv770
  store i8 %xor250721, i8* %arrayidx253, align 1
  %indvars.iv.next771 = add nuw nsw i64 %indvars.iv770, 1
  %exitcond772.not = icmp eq i64 %indvars.iv.next771, 64
  br i1 %exitcond772.not, label %for.end256, label %for.body243

for.end256:                                       ; preds = %for.body243
  %sub257 = add i32 %bytes.addr.2, -64
  %add.ptr258 = getelementptr inbounds i8, i8* %out.2, i64 64
  %add.ptr259 = getelementptr inbounds i8, i8* %m.addr.2, i64 64
  br label %for.cond211

cleanup:                                          ; preds = %for.body228, %for.cond225.preheader, %if.end207, %entry
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @llvm.epi.vsetvlmax(i64, i64)

; Function Attrs: nounwind readnone
declare <vscale x 2 x i32> @llvm.epi.vmv.v.x.nxv2i32.i32(i32, i64)

; Function Attrs: nounwind readnone
declare i64 @llvm.epi.vsetvl(i64, i64, i64)

; Function Attrs: nounwind readnone
declare <vscale x 2 x i32> @llvm.epi.vid.nxv2i32(i64)

; Function Attrs: nounwind readnone
declare <vscale x 2 x i1> @llvm.epi.vmxor.nxv2i1.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 2 x i32> @llvm.epi.vadc.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i1>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 2 x i1> @llvm.epi.vmadc.carry.in.nxv2i1.nxv2i32.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i1>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 2 x i32> @llvm.epi.vadd.nxv2i32.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 2 x i32> @llvm.epi.vxor.nxv2i32.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 2 x i32> @llvm.epi.vsll.nxv2i32.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 2 x i32> @llvm.epi.vsrl.nxv2i32.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 1 x i64> @llvm.epi.vid.nxv1i64(i64)

; Function Attrs: nounwind readnone
declare <vscale x 1 x i64> @llvm.epi.vmv.v.x.nxv1i64.i64(i64, i64)

; Function Attrs: nounwind readnone
declare <vscale x 1 x i64> @llvm.epi.vsll.nxv1i64.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 1 x i64> @llvm.epi.vsrl.nxv1i64.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 1 x i64> @llvm.epi.vrgather.nxv1i64.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 1 x i64> @llvm.epi.vadd.nxv1i64.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 1 x i64> @llvm.epi.vand.nxv1i64.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 1 x i64> @llvm.epi.vmerge.nxv1i64.nxv1i64.nxv1i1(<vscale x 1 x i64>, <vscale x 1 x i64>, <vscale x 1 x i1>, i64)

; Function Attrs: nounwind readonly
declare <vscale x 1 x i64> @llvm.epi.vload.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>* nocapture, <vscale x 1 x i64>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 1 x i64> @llvm.epi.vxor.nxv1i64.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>, i64)

; Function Attrs: nounwind writeonly
declare void @llvm.epi.vstore.indexed.nxv1i64.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>* nocapture, <vscale x 1 x i64>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 2 x i32> @llvm.epi.vrgather.nxv2i32.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 2 x i32> @llvm.epi.vand.nxv2i32.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)

; Function Attrs: nounwind readnone
declare <vscale x 2 x i32> @llvm.epi.vmerge.nxv2i32.nxv2i32.nxv2i1(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i1>, i64)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)

; Function Attrs: argmemonly nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg)
