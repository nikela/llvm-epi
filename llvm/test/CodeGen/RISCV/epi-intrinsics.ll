; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv64 -mattr=+m,+f,+d,+a,+c,+epi -verify-machineinstrs \
; RUN:     -no-epi-remove-redundant-vsetvl < %s | FileCheck %s

define void @test_vsetvl() nounwind
; CHECK-LABEL: test_vsetvl:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli a0, a0, e8, m1
; CHECK-NEXT:    vsetvli a0, a0, e16, m1
; CHECK-NEXT:    vsetvli a0, a0, e32, m1
; CHECK-NEXT:    vsetvli a0, a0, e64, m1
; CHECK-NEXT:    vsetvli a0, a0, e128, m1
; CHECK-NEXT:    vsetvli a0, a0, e8, m2
; CHECK-NEXT:    vsetvli a0, a0, e8, m4
; CHECK-NEXT:    vsetvli a0, a0, e8, m8
; CHECK-NEXT:    ret
{
entry:
  %a1 = call i64 @llvm.epi.vsetvl(i64 undef, i64 0, i64 0)
  %a2 = call i64 @llvm.epi.vsetvl(i64 undef, i64 1, i64 0)
  %a3 = call i64 @llvm.epi.vsetvl(i64 undef, i64 2, i64 0)
  %a4 = call i64 @llvm.epi.vsetvl(i64 undef, i64 3, i64 0)
  %a5 = call i64 @llvm.epi.vsetvl(i64 undef, i64 4, i64 0)
  %a6 = call i64 @llvm.epi.vsetvl(i64 undef, i64 0, i64 1)
  %a7 = call i64 @llvm.epi.vsetvl(i64 undef, i64 0, i64 2)
  %a8 = call i64 @llvm.epi.vsetvl(i64 undef, i64 0, i64 3)
  ret void
}

declare i64 @llvm.experimental.vector.vscale.i64()

define i64 @test_vscale() nounwind
; CHECK-LABEL: test_vscale:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vsetvli a0, zero, e64, m1
; CHECK-NEXT:    ret
{
entry:
  %a1 = call i64 @llvm.experimental.vector.vscale.i64()
  ret i64 %a1
}

declare i64 @llvm.epi.vsetvl(i64, i64, i64)

define void @test_load_stores() nounwind
; CHECK-LABEL: test_load_stores:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a0, a0, e8, m1
; CHECK-NEXT:    vle.v v0, (a0)
; CHECK-NEXT:    vsetvli a0, a0, e8, m1
; CHECK-NEXT:    vse.v v0, (a0)
; CHECK-NEXT:    vsetvli a0, a0, e16, m1
; CHECK-NEXT:    vle.v v0, (a0)
; CHECK-NEXT:    vsetvli a0, a0, e16, m1
; CHECK-NEXT:    vse.v v0, (a0)
; CHECK-NEXT:    vsetvli a0, a0, e32, m1
; CHECK-NEXT:    vle.v v0, (a0)
; CHECK-NEXT:    vsetvli a0, a0, e32, m1
; CHECK-NEXT:    vse.v v0, (a0)
; CHECK-NEXT:    vsetvli a0, a0, e64, m1
; CHECK-NEXT:    vle.v v0, (a0)
; CHECK-NEXT:    vsetvli a0, a0, e64, m1
; CHECK-NEXT:    vse.v v0, (a0)
; CHECK-NEXT:    vsetvli a0, a0, e32, m1
; CHECK-NEXT:    vle.v v0, (a0)
; CHECK-NEXT:    vsetvli a0, a0, e32, m1
; CHECK-NEXT:    vse.v v0, (a0)
; CHECK-NEXT:    vsetvli a0, a0, e64, m1
; CHECK-NEXT:    vle.v v0, (a0)
; CHECK-NEXT:    vsetvli a0, a0, e64, m1
; CHECK-NEXT:    vse.v v0, (a0)
; CHECK-NEXT:    ret
{
  %a1 = call <vscale x 8 x i8> @llvm.epi.vload.nxv1i8(<vscale x 8 x i8>* undef, i64 undef)
  call void @llvm.epi.vstore.nxv1i8(<vscale x 8 x i8> %a1, <vscale x 8 x i8>* undef, i64 undef)

  %a2 = call <vscale x 4 x i16> @llvm.epi.vload.nxv1i16(<vscale x 4 x i16>* undef, i64 undef)
  call void @llvm.epi.vstore.nxv1i16(<vscale x 4 x i16> %a2, <vscale x 4 x i16>* undef, i64 undef)

  %a3 = call <vscale x 2 x i32> @llvm.epi.vload.nxv1i32(<vscale x 2 x i32>* undef, i64 undef)
  call void @llvm.epi.vstore.nxv1i32(<vscale x 2 x i32> %a3, <vscale x 2 x i32>* undef, i64 undef)

  %a4 = call <vscale x 1 x i64> @llvm.epi.vload.nxv1i64(<vscale x 1 x i64>* undef, i64 undef)
  call void @llvm.epi.vstore.nxv1i64(<vscale x 1 x i64> %a4, <vscale x 1 x i64>* undef, i64 undef)

  %a5 = call <vscale x 2 x float> @llvm.epi.vload.nxv1f32(<vscale x 2 x float>* undef, i64 undef)
  call void @llvm.epi.vstore.nxv1f32(<vscale x 2 x float> %a5, <vscale x 2 x float>* undef, i64 undef)

  %a6 = call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* undef, i64 undef)
  call void @llvm.epi.vstore.nxv1f64(<vscale x 1 x double> %a6, <vscale x 1 x double>* undef, i64 undef)

  ret void
}

declare <vscale x 8 x i8> @llvm.epi.vload.nxv1i8(<vscale x 8 x i8>*, i64)
declare void @llvm.epi.vstore.nxv1i8(<vscale x 8 x i8>, <vscale x 8 x i8>*, i64)

declare <vscale x 4 x i16> @llvm.epi.vload.nxv1i16(<vscale x 4 x i16>*, i64)
declare void @llvm.epi.vstore.nxv1i16(<vscale x 4 x i16>, <vscale x 4 x i16>*, i64)

declare <vscale x 2 x i32> @llvm.epi.vload.nxv1i32(<vscale x 2 x i32>*, i64)
declare void @llvm.epi.vstore.nxv1i32(<vscale x 2 x i32>, <vscale x 2 x i32>*, i64)

declare <vscale x 1 x i64> @llvm.epi.vload.nxv1i64(<vscale x 1 x i64>*, i64)
declare void @llvm.epi.vstore.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>*, i64)

declare <vscale x 2 x float> @llvm.epi.vload.nxv1f32(<vscale x 2 x float>*, i64)
declare void @llvm.epi.vstore.nxv1f32(<vscale x 2 x float>, <vscale x 2 x float>*, i64)

declare <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>*, i64)
declare void @llvm.epi.vstore.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>*, i64)

; Check that the operands are swapped in the instruction
define void @test_greater_comparisons(<vscale x 1 x i64>* %pia,
; CHECK-LABEL: test_greater_comparisons:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetvli a6, a5, e64, m1
; CHECK-NEXT:    vle.v v0, (a0)
; CHECK-NEXT:    vsetvli a0, a5, e64, m1
; CHECK-NEXT:    vle.v v1, (a1)
; CHECK-NEXT:    vsetvli a0, a5, e64, m1
; CHECK-NEXT:    vmslt.vv v2, v1, v0
; CHECK-NEXT:    vsetvli a0, zero, e8, m1
; CHECK-NEXT:    vse.v v2, (a2)
; CHECK-NEXT:    vsetvli a0, a5, e64, m1
; CHECK-NEXT:    vmsltu.vv v0, v1, v0
; CHECK-NEXT:    vsetvli a0, zero, e8, m1
; CHECK-NEXT:    vse.v v0, (a2)
; CHECK-NEXT:    vsetvli a0, a5, e64, m1
; CHECK-NEXT:    vle.v v0, (a3)
; CHECK-NEXT:    vsetvli a0, a5, e64, m1
; CHECK-NEXT:    vle.v v1, (a4)
; CHECK-NEXT:    vsetvli a0, a5, e64, m1
; CHECK-NEXT:    vmfle.vv v2, v1, v0
; CHECK-NEXT:    vsetvli a0, a5, e64, m1
; CHECK-NEXT:    vmflt.vv v0, v1, v0
; CHECK-NEXT:    vsetvli a0, zero, e8, m1
; CHECK-NEXT:    vse.v v0, (a2)
; CHECK-NEXT:    vsetvli a0, zero, e8, m1
; CHECK-NEXT:    vse.v v2, (a2)
; CHECK-NEXT:    ret
                                      <vscale x 1 x i64>* %pib,
                                      <vscale x 1 x i64>* %pm,
                                      <vscale x 1 x double>* %pfa,
                                      <vscale x 1 x double>* %pfb,
                                      i64 %gvl) nounwind {
   %ia = call <vscale x 1 x i64> @llvm.epi.vload.nxv1i64(<vscale x 1 x i64>* %pia, i64 %gvl)
   %ib = call <vscale x 1 x i64> @llvm.epi.vload.nxv1i64(<vscale x 1 x i64>* %pib, i64 %gvl)

   %ma.1 = call <vscale x 1 x i1> @llvm.epi.vmsgt.nxv1i1.nxv1i64.nxv1i64(<vscale x 1 x i64> %ia, <vscale x 1 x i64> %ib, i64 %gvl)
   %zma.1 = zext <vscale x 1 x i1> %ma.1 to <vscale x 1 x i64>
   store volatile <vscale x 1 x i64> %zma.1, <vscale x 1 x i64> *%pm
   %ma.2 = call <vscale x 1 x i1> @llvm.epi.vmsgtu.nxv1i1.nxv1i64.nxv1i64(<vscale x 1 x i64> %ia, <vscale x 1 x i64> %ib, i64 %gvl)
   %zma.2 = zext <vscale x 1 x i1> %ma.2 to <vscale x 1 x i64>
   store volatile <vscale x 1 x i64> %zma.2, <vscale x 1 x i64> *%pm

   %fa = call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* %pfa, i64 %gvl)
   %fb = call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* %pfb, i64 %gvl)

   %mb.1 = call <vscale x 1 x i1> @llvm.epi.vmfgt.nxv1i1.nxv1f64.nxv1f64(<vscale x 1 x double> %fa, <vscale x 1 x double> %fb, i64 %gvl)
   %zmb.1 = zext <vscale x 1 x i1> %mb.1 to <vscale x 1 x i64>
   store volatile <vscale x 1 x i64> %zmb.1, <vscale x 1 x i64> *%pm
   %mb.2 = call <vscale x 1 x i1> @llvm.epi.vmfge.nxv1i1.nxv1f64.nxv1f64(<vscale x 1 x double> %fa, <vscale x 1 x double> %fb, i64 %gvl)
   %zmb.2 = zext <vscale x 1 x i1> %mb.2 to <vscale x 1 x i64>
   store volatile <vscale x 1 x i64> %zmb.2, <vscale x 1 x i64> *%pm

   ret void
}

declare <vscale x 1 x i1> @llvm.epi.vmsgt.nxv1i1.nxv1i64.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>, i64);
declare <vscale x 1 x i1> @llvm.epi.vmsgtu.nxv1i1.nxv1i64.nxv1i64(<vscale x 1 x i64>, <vscale x 1 x i64>, i64);
declare <vscale x 1 x i1> @llvm.epi.vmfgt.nxv1i1.nxv1f64.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, i64);
declare <vscale x 1 x i1> @llvm.epi.vmfge.nxv1i1.nxv1f64.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, i64);

