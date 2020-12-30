; RUN: llc -mtriple=riscv32 -mattr=+experimental-v -verify-machineinstrs \
; RUN:   --riscv-no-aliases < %s | FileCheck %s
declare <vscale x 1 x i8> @llvm.riscv.vmv.v.x.nxv1i8(
  i8,
  i32);

define <vscale x 1 x i8> @intrinsic_vmv.v.x_x_nxv1i8(i8 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv1i8
; CHECK:       vsetvli {{.*}}, a1, e8,mf8
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 1 x i8> @llvm.riscv.vmv.v.x.nxv1i8(
    i8 %0,
    i32 %1)

  ret <vscale x 1 x i8> %a
}

declare <vscale x 2 x i8> @llvm.riscv.vmv.v.x.nxv2i8(
  i8,
  i32);

define <vscale x 2 x i8> @intrinsic_vmv.v.x_x_nxv2i8(i8 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv2i8
; CHECK:       vsetvli {{.*}}, a1, e8,mf4
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 2 x i8> @llvm.riscv.vmv.v.x.nxv2i8(
    i8 %0,
    i32 %1)

  ret <vscale x 2 x i8> %a
}

declare <vscale x 4 x i8> @llvm.riscv.vmv.v.x.nxv4i8(
  i8,
  i32);

define <vscale x 4 x i8> @intrinsic_vmv.v.x_x_nxv4i8(i8 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv4i8
; CHECK:       vsetvli {{.*}}, a1, e8,mf2
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 4 x i8> @llvm.riscv.vmv.v.x.nxv4i8(
    i8 %0,
    i32 %1)

  ret <vscale x 4 x i8> %a
}

declare <vscale x 8 x i8> @llvm.riscv.vmv.v.x.nxv8i8(
  i8,
  i32);

define <vscale x 8 x i8> @intrinsic_vmv.v.x_x_nxv8i8(i8 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv8i8
; CHECK:       vsetvli {{.*}}, a1, e8,m1
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 8 x i8> @llvm.riscv.vmv.v.x.nxv8i8(
    i8 %0,
    i32 %1)

  ret <vscale x 8 x i8> %a
}

declare <vscale x 16 x i8> @llvm.riscv.vmv.v.x.nxv16i8(
  i8,
  i32);

define <vscale x 16 x i8> @intrinsic_vmv.v.x_x_nxv16i8(i8 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv16i8
; CHECK:       vsetvli {{.*}}, a1, e8,m2
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 16 x i8> @llvm.riscv.vmv.v.x.nxv16i8(
    i8 %0,
    i32 %1)

  ret <vscale x 16 x i8> %a
}

declare <vscale x 32 x i8> @llvm.riscv.vmv.v.x.nxv32i8(
  i8,
  i32);

define <vscale x 32 x i8> @intrinsic_vmv.v.x_x_nxv32i8(i8 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv32i8
; CHECK:       vsetvli {{.*}}, a1, e8,m4
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 32 x i8> @llvm.riscv.vmv.v.x.nxv32i8(
    i8 %0,
    i32 %1)

  ret <vscale x 32 x i8> %a
}

declare <vscale x 64 x i8> @llvm.riscv.vmv.v.x.nxv64i8(
  i8,
  i32);

define <vscale x 64 x i8> @intrinsic_vmv.v.x_x_nxv64i8(i8 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv64i8
; CHECK:       vsetvli {{.*}}, a1, e8,m8
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 64 x i8> @llvm.riscv.vmv.v.x.nxv64i8(
    i8 %0,
    i32 %1)

  ret <vscale x 64 x i8> %a
}

declare <vscale x 1 x i16> @llvm.riscv.vmv.v.x.nxv1i16(
  i16,
  i32);

define <vscale x 1 x i16> @intrinsic_vmv.v.x_x_nxv1i16(i16 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv1i16
; CHECK:       vsetvli {{.*}}, a1, e16,mf4
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 1 x i16> @llvm.riscv.vmv.v.x.nxv1i16(
    i16 %0,
    i32 %1)

  ret <vscale x 1 x i16> %a
}

declare <vscale x 2 x i16> @llvm.riscv.vmv.v.x.nxv2i16(
  i16,
  i32);

define <vscale x 2 x i16> @intrinsic_vmv.v.x_x_nxv2i16(i16 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv2i16
; CHECK:       vsetvli {{.*}}, a1, e16,mf2
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 2 x i16> @llvm.riscv.vmv.v.x.nxv2i16(
    i16 %0,
    i32 %1)

  ret <vscale x 2 x i16> %a
}

declare <vscale x 4 x i16> @llvm.riscv.vmv.v.x.nxv4i16(
  i16,
  i32);

define <vscale x 4 x i16> @intrinsic_vmv.v.x_x_nxv4i16(i16 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv4i16
; CHECK:       vsetvli {{.*}}, a1, e16,m1
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 4 x i16> @llvm.riscv.vmv.v.x.nxv4i16(
    i16 %0,
    i32 %1)

  ret <vscale x 4 x i16> %a
}

declare <vscale x 8 x i16> @llvm.riscv.vmv.v.x.nxv8i16(
  i16,
  i32);

define <vscale x 8 x i16> @intrinsic_vmv.v.x_x_nxv8i16(i16 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv8i16
; CHECK:       vsetvli {{.*}}, a1, e16,m2
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 8 x i16> @llvm.riscv.vmv.v.x.nxv8i16(
    i16 %0,
    i32 %1)

  ret <vscale x 8 x i16> %a
}

declare <vscale x 16 x i16> @llvm.riscv.vmv.v.x.nxv16i16(
  i16,
  i32);

define <vscale x 16 x i16> @intrinsic_vmv.v.x_x_nxv16i16(i16 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv16i16
; CHECK:       vsetvli {{.*}}, a1, e16,m4
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 16 x i16> @llvm.riscv.vmv.v.x.nxv16i16(
    i16 %0,
    i32 %1)

  ret <vscale x 16 x i16> %a
}

declare <vscale x 32 x i16> @llvm.riscv.vmv.v.x.nxv32i16(
  i16,
  i32);

define <vscale x 32 x i16> @intrinsic_vmv.v.x_x_nxv32i16(i16 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv32i16
; CHECK:       vsetvli {{.*}}, a1, e16,m8
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 32 x i16> @llvm.riscv.vmv.v.x.nxv32i16(
    i16 %0,
    i32 %1)

  ret <vscale x 32 x i16> %a
}

declare <vscale x 1 x i32> @llvm.riscv.vmv.v.x.nxv1i32(
  i32,
  i32);

define <vscale x 1 x i32> @intrinsic_vmv.v.x_x_nxv1i32(i32 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv1i32
; CHECK:       vsetvli {{.*}}, a1, e32,mf2
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 1 x i32> @llvm.riscv.vmv.v.x.nxv1i32(
    i32 %0,
    i32 %1)

  ret <vscale x 1 x i32> %a
}

declare <vscale x 2 x i32> @llvm.riscv.vmv.v.x.nxv2i32(
  i32,
  i32);

define <vscale x 2 x i32> @intrinsic_vmv.v.x_x_nxv2i32(i32 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv2i32
; CHECK:       vsetvli {{.*}}, a1, e32,m1
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 2 x i32> @llvm.riscv.vmv.v.x.nxv2i32(
    i32 %0,
    i32 %1)

  ret <vscale x 2 x i32> %a
}

declare <vscale x 4 x i32> @llvm.riscv.vmv.v.x.nxv4i32(
  i32,
  i32);

define <vscale x 4 x i32> @intrinsic_vmv.v.x_x_nxv4i32(i32 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv4i32
; CHECK:       vsetvli {{.*}}, a1, e32,m2
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 4 x i32> @llvm.riscv.vmv.v.x.nxv4i32(
    i32 %0,
    i32 %1)

  ret <vscale x 4 x i32> %a
}

declare <vscale x 8 x i32> @llvm.riscv.vmv.v.x.nxv8i32(
  i32,
  i32);

define <vscale x 8 x i32> @intrinsic_vmv.v.x_x_nxv8i32(i32 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv8i32
; CHECK:       vsetvli {{.*}}, a1, e32,m4
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 8 x i32> @llvm.riscv.vmv.v.x.nxv8i32(
    i32 %0,
    i32 %1)

  ret <vscale x 8 x i32> %a
}

declare <vscale x 16 x i32> @llvm.riscv.vmv.v.x.nxv16i32(
  i32,
  i32);

define <vscale x 16 x i32> @intrinsic_vmv.v.x_x_nxv16i32(i32 %0, i32 %1) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_x_nxv16i32
; CHECK:       vsetvli {{.*}}, a1, e32,m8
; CHECK:       vmv.v.x {{v[0-9]+}}, a0
  %a = call <vscale x 16 x i32> @llvm.riscv.vmv.v.x.nxv16i32(
    i32 %0,
    i32 %1)

  ret <vscale x 16 x i32> %a
}

define <vscale x 1 x i8> @intrinsic_vmv.v.x_i_nxv1i8(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv1i8
; CHECK:       vsetvli {{.*}}, a0, e8,mf8
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 1 x i8> @llvm.riscv.vmv.v.x.nxv1i8(
    i8 9,
    i32 %0)

  ret <vscale x 1 x i8> %a
}

define <vscale x 2 x i8> @intrinsic_vmv.v.x_i_nxv2i8(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv2i8
; CHECK:       vsetvli {{.*}}, a0, e8,mf4
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 2 x i8> @llvm.riscv.vmv.v.x.nxv2i8(
    i8 9,
    i32 %0)

  ret <vscale x 2 x i8> %a
}

define <vscale x 4 x i8> @intrinsic_vmv.v.x_i_nxv4i8(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv4i8
; CHECK:       vsetvli {{.*}}, a0, e8,mf2
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 4 x i8> @llvm.riscv.vmv.v.x.nxv4i8(
    i8 9,
    i32 %0)

  ret <vscale x 4 x i8> %a
}

define <vscale x 8 x i8> @intrinsic_vmv.v.x_i_nxv8i8(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv8i8
; CHECK:       vsetvli {{.*}}, a0, e8,m1
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 8 x i8> @llvm.riscv.vmv.v.x.nxv8i8(
    i8 9,
    i32 %0)

  ret <vscale x 8 x i8> %a
}

define <vscale x 16 x i8> @intrinsic_vmv.v.x_i_nxv16i8(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv16i8
; CHECK:       vsetvli {{.*}}, a0, e8,m2
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 16 x i8> @llvm.riscv.vmv.v.x.nxv16i8(
    i8 9,
    i32 %0)

  ret <vscale x 16 x i8> %a
}

define <vscale x 32 x i8> @intrinsic_vmv.v.x_i_nxv32i8(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv32i8
; CHECK:       vsetvli {{.*}}, a0, e8,m4
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 32 x i8> @llvm.riscv.vmv.v.x.nxv32i8(
    i8 9,
    i32 %0)

  ret <vscale x 32 x i8> %a
}

define <vscale x 64 x i8> @intrinsic_vmv.v.x_i_nxv64i8(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv64i8
; CHECK:       vsetvli {{.*}}, a0, e8,m8
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 64 x i8> @llvm.riscv.vmv.v.x.nxv64i8(
    i8 9,
    i32 %0)

  ret <vscale x 64 x i8> %a
}

define <vscale x 1 x i16> @intrinsic_vmv.v.x_i_nxv1i16(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv1i16
; CHECK:       vsetvli {{.*}}, a0, e16,mf4
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 1 x i16> @llvm.riscv.vmv.v.x.nxv1i16(
    i16 9,
    i32 %0)

  ret <vscale x 1 x i16> %a
}

define <vscale x 2 x i16> @intrinsic_vmv.v.x_i_nxv2i16(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv2i16
; CHECK:       vsetvli {{.*}}, a0, e16,mf2
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 2 x i16> @llvm.riscv.vmv.v.x.nxv2i16(
    i16 9,
    i32 %0)

  ret <vscale x 2 x i16> %a
}

define <vscale x 4 x i16> @intrinsic_vmv.v.x_i_nxv4i16(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv4i16
; CHECK:       vsetvli {{.*}}, a0, e16,m1
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 4 x i16> @llvm.riscv.vmv.v.x.nxv4i16(
    i16 9,
    i32 %0)

  ret <vscale x 4 x i16> %a
}

define <vscale x 8 x i16> @intrinsic_vmv.v.x_i_nxv8i16(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv8i16
; CHECK:       vsetvli {{.*}}, a0, e16,m2
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 8 x i16> @llvm.riscv.vmv.v.x.nxv8i16(
    i16 9,
    i32 %0)

  ret <vscale x 8 x i16> %a
}

define <vscale x 16 x i16> @intrinsic_vmv.v.x_i_nxv16i16(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv16i16
; CHECK:       vsetvli {{.*}}, a0, e16,m4
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 16 x i16> @llvm.riscv.vmv.v.x.nxv16i16(
    i16 9,
    i32 %0)

  ret <vscale x 16 x i16> %a
}

define <vscale x 32 x i16> @intrinsic_vmv.v.x_i_nxv32i16(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv32i16
; CHECK:       vsetvli {{.*}}, a0, e16,m8
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 32 x i16> @llvm.riscv.vmv.v.x.nxv32i16(
    i16 9,
    i32 %0)

  ret <vscale x 32 x i16> %a
}

define <vscale x 1 x i32> @intrinsic_vmv.v.x_i_nxv1i32(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv1i32
; CHECK:       vsetvli {{.*}}, a0, e32,mf2
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 1 x i32> @llvm.riscv.vmv.v.x.nxv1i32(
    i32 9,
    i32 %0)

  ret <vscale x 1 x i32> %a
}

define <vscale x 2 x i32> @intrinsic_vmv.v.x_i_nxv2i32(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv2i32
; CHECK:       vsetvli {{.*}}, a0, e32,m1
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 2 x i32> @llvm.riscv.vmv.v.x.nxv2i32(
    i32 9,
    i32 %0)

  ret <vscale x 2 x i32> %a
}

define <vscale x 4 x i32> @intrinsic_vmv.v.x_i_nxv4i32(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv4i32
; CHECK:       vsetvli {{.*}}, a0, e32,m2
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 4 x i32> @llvm.riscv.vmv.v.x.nxv4i32(
    i32 9,
    i32 %0)

  ret <vscale x 4 x i32> %a
}

define <vscale x 8 x i32> @intrinsic_vmv.v.x_i_nxv8i32(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv8i32
; CHECK:       vsetvli {{.*}}, a0, e32,m4
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 8 x i32> @llvm.riscv.vmv.v.x.nxv8i32(
    i32 9,
    i32 %0)

  ret <vscale x 8 x i32> %a
}

define <vscale x 16 x i32> @intrinsic_vmv.v.x_i_nxv16i32(i32 %0) nounwind {
entry:
; CHECK-LABEL: intrinsic_vmv.v.x_i_nxv16i32
; CHECK:       vsetvli {{.*}}, a0, e32,m8
; CHECK:       vmv.v.i {{v[0-9]+}}, 9
  %a = call <vscale x 16 x i32> @llvm.riscv.vmv.v.x.nxv16i32(
    i32 9,
    i32 %0)

  ret <vscale x 16 x i32> %a
}
