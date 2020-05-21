; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple riscv64 -mattr=+experimental-v < %s | FileCheck %s

@scratch = global i8 0, align 16

define void @lmul_1(<vscale x 1 x i64> %a, <vscale x 1 x i64> %b) nounwind {
; CHECK-LABEL: lmul_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lui a0, %hi(scratch)
; CHECK-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-NEXT:    vsetvli a1, zero, e64,m1
; CHECK-NEXT:    vmseq.vv v1, v16, v17
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m1
; CHECK-NEXT:    vmsne.vv v1, v16, v17
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m1
; CHECK-NEXT:    vmsleu.vv v1, v17, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m1
; CHECK-NEXT:    vmsltu.vv v1, v17, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m1
; CHECK-NEXT:    vmsltu.vv v1, v16, v17
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m1
; CHECK-NEXT:    vmsleu.vv v1, v16, v17
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m1
; CHECK-NEXT:    vmsle.vv v1, v17, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m1
; CHECK-NEXT:    vmslt.vv v1, v17, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m1
; CHECK-NEXT:    vmslt.vv v1, v16, v17
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m1
; CHECK-NEXT:    vmsle.vv v1, v16, v17
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 1 x i64>*

  %cmp_1 = icmp eq <vscale x 1 x i64> %a, %b
  %val_1 = zext <vscale x 1 x i1> %cmp_1 to <vscale x 1 x i64>
  store <vscale x 1 x i64> %val_1, <vscale x 1 x i64>* %store_addr

  %cmp_2 = icmp ne <vscale x 1 x i64> %a, %b
  %val_2 = zext <vscale x 1 x i1> %cmp_2 to <vscale x 1 x i64>
  store <vscale x 1 x i64> %val_2, <vscale x 1 x i64>* %store_addr

  %cmp_3 = icmp ugt <vscale x 1 x i64> %a, %b
  %val_3 = zext <vscale x 1 x i1> %cmp_3 to <vscale x 1 x i64>
  store <vscale x 1 x i64> %val_3, <vscale x 1 x i64>* %store_addr

  %cmp_4 = icmp uge <vscale x 1 x i64> %a, %b
  %val_4 = zext <vscale x 1 x i1> %cmp_4 to <vscale x 1 x i64>
  store <vscale x 1 x i64> %val_4, <vscale x 1 x i64>* %store_addr

  %cmp_5 = icmp ult <vscale x 1 x i64> %a, %b
  %val_5 = zext <vscale x 1 x i1> %cmp_5 to <vscale x 1 x i64>
  store <vscale x 1 x i64> %val_5, <vscale x 1 x i64>* %store_addr

  %cmp_6 = icmp ule <vscale x 1 x i64> %a, %b
  %val_6 = zext <vscale x 1 x i1> %cmp_6 to <vscale x 1 x i64>
  store <vscale x 1 x i64> %val_6, <vscale x 1 x i64>* %store_addr

  %cmp_7 = icmp sgt <vscale x 1 x i64> %a, %b
  %val_7 = zext <vscale x 1 x i1> %cmp_7 to <vscale x 1 x i64>
  store <vscale x 1 x i64> %val_7, <vscale x 1 x i64>* %store_addr

  %cmp_8 = icmp sge <vscale x 1 x i64> %a, %b
  %val_8 = zext <vscale x 1 x i1> %cmp_8 to <vscale x 1 x i64>
  store <vscale x 1 x i64> %val_8, <vscale x 1 x i64>* %store_addr

  %cmp_9 = icmp slt <vscale x 1 x i64> %a, %b
  %val_9 = zext <vscale x 1 x i1> %cmp_9 to <vscale x 1 x i64>
  store <vscale x 1 x i64> %val_9, <vscale x 1 x i64>* %store_addr

  %cmp_10 = icmp sle <vscale x 1 x i64> %a, %b
  %val_10 = zext <vscale x 1 x i1> %cmp_10 to <vscale x 1 x i64>
  store <vscale x 1 x i64> %val_10, <vscale x 1 x i64>* %store_addr

  ret void
}

define void @lmul_2(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) nounwind {
; CHECK-LABEL: lmul_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lui a0, %hi(scratch)
; CHECK-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-NEXT:    vsetvli a1, zero, e64,m2
; CHECK-NEXT:    vmseq.vv v1, v16, v18
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m2
; CHECK-NEXT:    vmsne.vv v1, v16, v18
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m2
; CHECK-NEXT:    vmsleu.vv v1, v18, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m2
; CHECK-NEXT:    vmsltu.vv v1, v18, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m2
; CHECK-NEXT:    vmsltu.vv v1, v16, v18
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m2
; CHECK-NEXT:    vmsleu.vv v1, v16, v18
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m2
; CHECK-NEXT:    vmsle.vv v1, v18, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m2
; CHECK-NEXT:    vmslt.vv v1, v18, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m2
; CHECK-NEXT:    vmslt.vv v1, v16, v18
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m2
; CHECK-NEXT:    vmsle.vv v1, v16, v18
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 2 x i32>*

  %cmp_1 = icmp eq <vscale x 2 x i64> %a, %b
  %val_1 = zext <vscale x 2 x i1> %cmp_1 to <vscale x 2 x i32>
  store <vscale x 2 x i32> %val_1, <vscale x 2 x i32>* %store_addr

  %cmp_2 = icmp ne <vscale x 2 x i64> %a, %b
  %val_2 = zext <vscale x 2 x i1> %cmp_2 to <vscale x 2 x i32>
  store <vscale x 2 x i32> %val_2, <vscale x 2 x i32>* %store_addr

  %cmp_3 = icmp ugt <vscale x 2 x i64> %a, %b
  %val_3 = zext <vscale x 2 x i1> %cmp_3 to <vscale x 2 x i32>
  store <vscale x 2 x i32> %val_3, <vscale x 2 x i32>* %store_addr

  %cmp_4 = icmp uge <vscale x 2 x i64> %a, %b
  %val_4 = zext <vscale x 2 x i1> %cmp_4 to <vscale x 2 x i32>
  store <vscale x 2 x i32> %val_4, <vscale x 2 x i32>* %store_addr

  %cmp_5 = icmp ult <vscale x 2 x i64> %a, %b
  %val_5 = zext <vscale x 2 x i1> %cmp_5 to <vscale x 2 x i32>
  store <vscale x 2 x i32> %val_5, <vscale x 2 x i32>* %store_addr

  %cmp_6 = icmp ule <vscale x 2 x i64> %a, %b
  %val_6 = zext <vscale x 2 x i1> %cmp_6 to <vscale x 2 x i32>
  store <vscale x 2 x i32> %val_6, <vscale x 2 x i32>* %store_addr

  %cmp_7 = icmp sgt <vscale x 2 x i64> %a, %b
  %val_7 = zext <vscale x 2 x i1> %cmp_7 to <vscale x 2 x i32>
  store <vscale x 2 x i32> %val_7, <vscale x 2 x i32>* %store_addr

  %cmp_8 = icmp sge <vscale x 2 x i64> %a, %b
  %val_8 = zext <vscale x 2 x i1> %cmp_8 to <vscale x 2 x i32>
  store <vscale x 2 x i32> %val_8, <vscale x 2 x i32>* %store_addr

  %cmp_9 = icmp slt <vscale x 2 x i64> %a, %b
  %val_9 = zext <vscale x 2 x i1> %cmp_9 to <vscale x 2 x i32>
  store <vscale x 2 x i32> %val_9, <vscale x 2 x i32>* %store_addr

  %cmp_10 = icmp sle <vscale x 2 x i64> %a, %b
  %val_10 = zext <vscale x 2 x i1> %cmp_10 to <vscale x 2 x i32>
  store <vscale x 2 x i32> %val_10, <vscale x 2 x i32>* %store_addr

  ret void
}

define void @lmul_4(<vscale x 4 x i64> %a, <vscale x 4 x i64> %b) nounwind {
; CHECK-LABEL: lmul_4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lui a0, %hi(scratch)
; CHECK-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-NEXT:    vsetvli a1, zero, e64,m4
; CHECK-NEXT:    vmseq.vv v1, v16, v20
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m4
; CHECK-NEXT:    vmsne.vv v1, v16, v20
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m4
; CHECK-NEXT:    vmsleu.vv v1, v20, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m4
; CHECK-NEXT:    vmsltu.vv v1, v20, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m4
; CHECK-NEXT:    vmsltu.vv v1, v16, v20
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m4
; CHECK-NEXT:    vmsleu.vv v1, v16, v20
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m4
; CHECK-NEXT:    vmsle.vv v1, v20, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m4
; CHECK-NEXT:    vmslt.vv v1, v20, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m4
; CHECK-NEXT:    vmslt.vv v1, v16, v20
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m4
; CHECK-NEXT:    vmsle.vv v1, v16, v20
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 4 x i16>*

  %cmp_1 = icmp eq <vscale x 4 x i64> %a, %b
  %val_1 = zext <vscale x 4 x i1> %cmp_1 to <vscale x 4 x i16>
  store <vscale x 4 x i16> %val_1, <vscale x 4 x i16>* %store_addr

  %cmp_2 = icmp ne <vscale x 4 x i64> %a, %b
  %val_2 = zext <vscale x 4 x i1> %cmp_2 to <vscale x 4 x i16>
  store <vscale x 4 x i16> %val_2, <vscale x 4 x i16>* %store_addr

  %cmp_3 = icmp ugt <vscale x 4 x i64> %a, %b
  %val_3 = zext <vscale x 4 x i1> %cmp_3 to <vscale x 4 x i16>
  store <vscale x 4 x i16> %val_3, <vscale x 4 x i16>* %store_addr

  %cmp_4 = icmp uge <vscale x 4 x i64> %a, %b
  %val_4 = zext <vscale x 4 x i1> %cmp_4 to <vscale x 4 x i16>
  store <vscale x 4 x i16> %val_4, <vscale x 4 x i16>* %store_addr

  %cmp_5 = icmp ult <vscale x 4 x i64> %a, %b
  %val_5 = zext <vscale x 4 x i1> %cmp_5 to <vscale x 4 x i16>
  store <vscale x 4 x i16> %val_5, <vscale x 4 x i16>* %store_addr

  %cmp_6 = icmp ule <vscale x 4 x i64> %a, %b
  %val_6 = zext <vscale x 4 x i1> %cmp_6 to <vscale x 4 x i16>
  store <vscale x 4 x i16> %val_6, <vscale x 4 x i16>* %store_addr

  %cmp_7 = icmp sgt <vscale x 4 x i64> %a, %b
  %val_7 = zext <vscale x 4 x i1> %cmp_7 to <vscale x 4 x i16>
  store <vscale x 4 x i16> %val_7, <vscale x 4 x i16>* %store_addr

  %cmp_8 = icmp sge <vscale x 4 x i64> %a, %b
  %val_8 = zext <vscale x 4 x i1> %cmp_8 to <vscale x 4 x i16>
  store <vscale x 4 x i16> %val_8, <vscale x 4 x i16>* %store_addr

  %cmp_9 = icmp slt <vscale x 4 x i64> %a, %b
  %val_9 = zext <vscale x 4 x i1> %cmp_9 to <vscale x 4 x i16>
  store <vscale x 4 x i16> %val_9, <vscale x 4 x i16>* %store_addr

  %cmp_10 = icmp sle <vscale x 4 x i64> %a, %b
  %val_10 = zext <vscale x 4 x i1> %cmp_10 to <vscale x 4 x i16>
  store <vscale x 4 x i16> %val_10, <vscale x 4 x i16>* %store_addr

  ret void
}

define void @lmul_8(<vscale x 8 x i64> %a, <vscale x 8 x i64> %b) nounwind {
; CHECK-LABEL: lmul_8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e64,m8
; CHECK-NEXT:    vle.v v8, (a0)
; CHECK-NEXT:    lui a0, %hi(scratch)
; CHECK-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-NEXT:    vmseq.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m8
; CHECK-NEXT:    vmsne.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m8
; CHECK-NEXT:    vmsleu.vv v1, v8, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m8
; CHECK-NEXT:    vmsltu.vv v1, v8, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m8
; CHECK-NEXT:    vmsltu.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m8
; CHECK-NEXT:    vmsleu.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m8
; CHECK-NEXT:    vmsle.vv v1, v8, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m8
; CHECK-NEXT:    vmslt.vv v1, v8, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m8
; CHECK-NEXT:    vmslt.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e64,m8
; CHECK-NEXT:    vmsle.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 8 x i8>*

  %cmp_1 = icmp eq <vscale x 8 x i64> %a, %b
  %val_1 = zext <vscale x 8 x i1> %cmp_1 to <vscale x 8 x i8>
  store <vscale x 8 x i8> %val_1, <vscale x 8 x i8>* %store_addr

  %cmp_2 = icmp ne <vscale x 8 x i64> %a, %b
  %val_2 = zext <vscale x 8 x i1> %cmp_2 to <vscale x 8 x i8>
  store <vscale x 8 x i8> %val_2, <vscale x 8 x i8>* %store_addr

  %cmp_3 = icmp ugt <vscale x 8 x i64> %a, %b
  %val_3 = zext <vscale x 8 x i1> %cmp_3 to <vscale x 8 x i8>
  store <vscale x 8 x i8> %val_3, <vscale x 8 x i8>* %store_addr

  %cmp_4 = icmp uge <vscale x 8 x i64> %a, %b
  %val_4 = zext <vscale x 8 x i1> %cmp_4 to <vscale x 8 x i8>
  store <vscale x 8 x i8> %val_4, <vscale x 8 x i8>* %store_addr

  %cmp_5 = icmp ult <vscale x 8 x i64> %a, %b
  %val_5 = zext <vscale x 8 x i1> %cmp_5 to <vscale x 8 x i8>
  store <vscale x 8 x i8> %val_5, <vscale x 8 x i8>* %store_addr

  %cmp_6 = icmp ule <vscale x 8 x i64> %a, %b
  %val_6 = zext <vscale x 8 x i1> %cmp_6 to <vscale x 8 x i8>
  store <vscale x 8 x i8> %val_6, <vscale x 8 x i8>* %store_addr

  %cmp_7 = icmp sgt <vscale x 8 x i64> %a, %b
  %val_7 = zext <vscale x 8 x i1> %cmp_7 to <vscale x 8 x i8>
  store <vscale x 8 x i8> %val_7, <vscale x 8 x i8>* %store_addr

  %cmp_8 = icmp sge <vscale x 8 x i64> %a, %b
  %val_8 = zext <vscale x 8 x i1> %cmp_8 to <vscale x 8 x i8>
  store <vscale x 8 x i8> %val_8, <vscale x 8 x i8>* %store_addr

  %cmp_9 = icmp slt <vscale x 8 x i64> %a, %b
  %val_9 = zext <vscale x 8 x i1> %cmp_9 to <vscale x 8 x i8>
  store <vscale x 8 x i8> %val_9, <vscale x 8 x i8>* %store_addr

  %cmp_10 = icmp sle <vscale x 8 x i64> %a, %b
  %val_10 = zext <vscale x 8 x i1> %cmp_10 to <vscale x 8 x i8>
  store <vscale x 8 x i8> %val_10, <vscale x 8 x i8>* %store_addr

  ret void
}

define void @lmul_8_i32(<vscale x 16 x i32> %a, <vscale x 16 x i32> %b) nounwind {
; CHECK-LABEL: lmul_8_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e32,m8
; CHECK-NEXT:    vle.v v8, (a0)
; CHECK-NEXT:    lui a0, %hi(scratch)
; CHECK-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-NEXT:    vmseq.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e32,m8
; CHECK-NEXT:    vmsne.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e32,m8
; CHECK-NEXT:    vmsleu.vv v1, v8, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e32,m8
; CHECK-NEXT:    vmsltu.vv v1, v8, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e32,m8
; CHECK-NEXT:    vmsltu.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e32,m8
; CHECK-NEXT:    vmsleu.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e32,m8
; CHECK-NEXT:    vmsle.vv v1, v8, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e32,m8
; CHECK-NEXT:    vmslt.vv v1, v8, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e32,m8
; CHECK-NEXT:    vmslt.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e32,m8
; CHECK-NEXT:    vmsle.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 16 x i4>*

  %cmp_1 = icmp eq <vscale x 16 x i32> %a, %b
  %val_1 = zext <vscale x 16 x i1> %cmp_1 to <vscale x 16 x i4>
  store <vscale x 16 x i4> %val_1, <vscale x 16 x i4>* %store_addr

  %cmp_2 = icmp ne <vscale x 16 x i32> %a, %b
  %val_2 = zext <vscale x 16 x i1> %cmp_2 to <vscale x 16 x i4>
  store <vscale x 16 x i4> %val_2, <vscale x 16 x i4>* %store_addr

  %cmp_3 = icmp ugt <vscale x 16 x i32> %a, %b
  %val_3 = zext <vscale x 16 x i1> %cmp_3 to <vscale x 16 x i4>
  store <vscale x 16 x i4> %val_3, <vscale x 16 x i4>* %store_addr

  %cmp_4 = icmp uge <vscale x 16 x i32> %a, %b
  %val_4 = zext <vscale x 16 x i1> %cmp_4 to <vscale x 16 x i4>
  store <vscale x 16 x i4> %val_4, <vscale x 16 x i4>* %store_addr

  %cmp_5 = icmp ult <vscale x 16 x i32> %a, %b
  %val_5 = zext <vscale x 16 x i1> %cmp_5 to <vscale x 16 x i4>
  store <vscale x 16 x i4> %val_5, <vscale x 16 x i4>* %store_addr

  %cmp_6 = icmp ule <vscale x 16 x i32> %a, %b
  %val_6 = zext <vscale x 16 x i1> %cmp_6 to <vscale x 16 x i4>
  store <vscale x 16 x i4> %val_6, <vscale x 16 x i4>* %store_addr

  %cmp_7 = icmp sgt <vscale x 16 x i32> %a, %b
  %val_7 = zext <vscale x 16 x i1> %cmp_7 to <vscale x 16 x i4>
  store <vscale x 16 x i4> %val_7, <vscale x 16 x i4>* %store_addr

  %cmp_8 = icmp sge <vscale x 16 x i32> %a, %b
  %val_8 = zext <vscale x 16 x i1> %cmp_8 to <vscale x 16 x i4>
  store <vscale x 16 x i4> %val_8, <vscale x 16 x i4>* %store_addr

  %cmp_9 = icmp slt <vscale x 16 x i32> %a, %b
  %val_9 = zext <vscale x 16 x i1> %cmp_9 to <vscale x 16 x i4>
  store <vscale x 16 x i4> %val_9, <vscale x 16 x i4>* %store_addr

  %cmp_10 = icmp sle <vscale x 16 x i32> %a, %b
  %val_10 = zext <vscale x 16 x i1> %cmp_10 to <vscale x 16 x i4>
  store <vscale x 16 x i4> %val_10, <vscale x 16 x i4>* %store_addr

  ret void
}

define void @lmul_8_i16(<vscale x 32 x i16> %a, <vscale x 32 x i16> %b) nounwind {
; CHECK-LABEL: lmul_8_i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a1, zero, e16,m8
; CHECK-NEXT:    vle.v v8, (a0)
; CHECK-NEXT:    lui a0, %hi(scratch)
; CHECK-NEXT:    addi a0, a0, %lo(scratch)
; CHECK-NEXT:    vmseq.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e16,m8
; CHECK-NEXT:    vmsne.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e16,m8
; CHECK-NEXT:    vmsleu.vv v1, v8, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e16,m8
; CHECK-NEXT:    vmsltu.vv v1, v8, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e16,m8
; CHECK-NEXT:    vmsltu.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e16,m8
; CHECK-NEXT:    vmsleu.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e16,m8
; CHECK-NEXT:    vmsle.vv v1, v8, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e16,m8
; CHECK-NEXT:    vmslt.vv v1, v8, v16
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e16,m8
; CHECK-NEXT:    vmslt.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    vsetvli a1, zero, e16,m8
; CHECK-NEXT:    vmsle.vv v1, v16, v8
; CHECK-NEXT:    vsetvli a1, zero, e8,m1
; CHECK-NEXT:    vse.v v1, (a0)
; CHECK-NEXT:    ret
  %store_addr = bitcast i8* @scratch to <vscale x 32 x i2>*
  %store_addr_2 = bitcast i8* @scratch to <vscale x 32 x i1>*

  %cmp_1 = icmp eq <vscale x 32 x i16> %a, %b
  %val_1 = zext <vscale x 32 x i1> %cmp_1 to <vscale x 32 x i2>
  store <vscale x 32 x i2> %val_1, <vscale x 32 x i2>* %store_addr

  %cmp_2 = icmp ne <vscale x 32 x i16> %a, %b
  %val_2 = zext <vscale x 32 x i1> %cmp_2 to <vscale x 32 x i2>
  store <vscale x 32 x i2> %val_2, <vscale x 32 x i2>* %store_addr

  %cmp_3 = icmp ugt <vscale x 32 x i16> %a, %b
  %val_3 = zext <vscale x 32 x i1> %cmp_3 to <vscale x 32 x i2>
  store <vscale x 32 x i2> %val_3, <vscale x 32 x i2>* %store_addr

  %cmp_4 = icmp uge <vscale x 32 x i16> %a, %b
  %val_4 = zext <vscale x 32 x i1> %cmp_4 to <vscale x 32 x i2>
  store <vscale x 32 x i2> %val_4, <vscale x 32 x i2>* %store_addr

  %cmp_5 = icmp ult <vscale x 32 x i16> %a, %b
  %val_5 = zext <vscale x 32 x i1> %cmp_5 to <vscale x 32 x i2>
  store <vscale x 32 x i2> %val_5, <vscale x 32 x i2>* %store_addr

  %cmp_6 = icmp ule <vscale x 32 x i16> %a, %b
  %val_6 = zext <vscale x 32 x i1> %cmp_6 to <vscale x 32 x i2>
  store <vscale x 32 x i2> %val_6, <vscale x 32 x i2>* %store_addr

  %cmp_7 = icmp sgt <vscale x 32 x i16> %a, %b
  %val_7 = zext <vscale x 32 x i1> %cmp_7 to <vscale x 32 x i2>
  store <vscale x 32 x i2> %val_7, <vscale x 32 x i2>* %store_addr

  %cmp_8 = icmp sge <vscale x 32 x i16> %a, %b
  %val_8 = zext <vscale x 32 x i1> %cmp_8 to <vscale x 32 x i2>
  store <vscale x 32 x i2> %val_8, <vscale x 32 x i2>* %store_addr

  %cmp_9 = icmp slt <vscale x 32 x i16> %a, %b
  %val_9 = zext <vscale x 32 x i1> %cmp_9 to <vscale x 32 x i2>
  store <vscale x 32 x i2> %val_9, <vscale x 32 x i2>* %store_addr

  %cmp_10 = icmp sle <vscale x 32 x i16> %a, %b
  %val_10 = zext <vscale x 32 x i1> %cmp_10 to <vscale x 32 x i2>
  store <vscale x 32 x i2> %val_10, <vscale x 32 x i2>* %store_addr

  ret void
}

; FIXME enable when nxv64i8 is supported
;define void @lmul_8_i8(<vscale x 64 x i8> %a, <vscale x 64 x i8> %b) nounwind {
;  %store_addr = bitcast i8* @scratch to <vscale x 64 x i1>*
;
;  %cmp_1 = icmp eq <vscale x 64 x i8> %a, %b
;  store <vscale x 64 x i1> %cmp_1, <vscale x 64 x i1>* %store_addr
;
;  %cmp_2 = icmp ne <vscale x 64 x i8> %a, %b
;  store <vscale x 64 x i1> %cmp_2, <vscale x 64 x i1>* %store_addr
;
;  %cmp_3 = icmp ugt <vscale x 64 x i8> %a, %b
;  store <vscale x 64 x i1> %cmp_3, <vscale x 64 x i1>* %store_addr
;
;  %cmp_4 = icmp uge <vscale x 64 x i8> %a, %b
;  store <vscale x 64 x i1> %cmp_4, <vscale x 64 x i1>* %store_addr
;
;  %cmp_5 = icmp ult <vscale x 64 x i8> %a, %b
;  store <vscale x 64 x i1> %cmp_5, <vscale x 64 x i1>* %store_addr
;
;  %cmp_6 = icmp ule <vscale x 64 x i8> %a, %b
;  store <vscale x 64 x i1> %cmp_6, <vscale x 64 x i1>* %store_addr
;
;  %cmp_7 = icmp sgt <vscale x 64 x i8> %a, %b
;  store <vscale x 64 x i1> %cmp_7, <vscale x 64 x i1>* %store_addr
;
;  %cmp_8 = icmp sge <vscale x 64 x i8> %a, %b
;  store <vscale x 64 x i1> %cmp_8, <vscale x 64 x i1>* %store_addr
;
;  %cmp_9 = icmp slt <vscale x 64 x i8> %a, %b
;  store <vscale x 64 x i1> %cmp_9, <vscale x 64 x i1>* %store_addr
;
;  %cmp_10 = icmp sle <vscale x 64 x i8> %a, %b
;  store <vscale x 64 x i1> %cmp_10, <vscale x 64 x i1>* %store_addr
;
;  ret void
;}
