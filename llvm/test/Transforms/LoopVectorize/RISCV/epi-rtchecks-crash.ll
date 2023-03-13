; Test that we do not crash
; RUN: opt -mtriple riscv64 -mattr +m,+a,+f,+d,+zepi \
; RUN:   -scalable-vectorization=only \
; RUN:   -prefer-predicate-over-epilogue=predicate-dont-vectorize -S \
; RUN:   -passes=loop-vectorize -riscv-v-vector-bits-min=64 -disable-output < %s

; ModuleID = 'bugpoint-reduced-simplified.bc'
source_filename = "FIRModule"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

@_QMyommp0Elscmec = external local_unnamed_addr global i32
@_QMyomjfhEn_vmass = external local_unnamed_addr global i32
@_QMyoethfEr2es = external local_unnamed_addr global double
@_QMyoethfEr3ies = external local_unnamed_addr global double
@_QMyoethfEr3les = external local_unnamed_addr global double
@_QMyoethfEr4ies = external local_unnamed_addr global double
@_QMyoethfEr4les = external local_unnamed_addr global double
@_QMyoethfEr5alscp = external local_unnamed_addr global double
@_QMyoethfEr5alvcp = external local_unnamed_addr global double
@_QMyoethfEr5ies = external local_unnamed_addr global double
@_QMyoethfEr5les = external local_unnamed_addr global double
@_QMyoethfEralfdcp = external local_unnamed_addr global double
@_QMyoethfEralsdcp = external local_unnamed_addr global double
@_QMyoethfEralvdcp = external local_unnamed_addr global double
@_QMyomcstErcpd = external local_unnamed_addr global double
@_QMyomcstErd = external local_unnamed_addr global double
@_QMyomcstEretv = external local_unnamed_addr global double
@_QMyomcstErg = external local_unnamed_addr global double
@_QMyoethfErkoop1 = external local_unnamed_addr global double
@_QMyoethfErkoop2 = external local_unnamed_addr global double
@_QMyomcstErlstt = external local_unnamed_addr global double
@_QMyomcstErlvtt = external local_unnamed_addr global double
@_QMyoethfErtice = external local_unnamed_addr global double
@_QMyomcstErtt = external local_unnamed_addr global double
@_QMyoethfErtwat = external local_unnamed_addr global double
@_QMyoethfErtwat_rtice_r = external local_unnamed_addr global double
@_QMyomcstErv = external local_unnamed_addr global double
@_QFcloudscEtime1 = external hidden unnamed_addr global double
@_QFcloudscEtime2 = external hidden unnamed_addr global double
@_QFcloudscEtime3 = external hidden unnamed_addr global double
@_QMyoecldpEyrecldp = external local_unnamed_addr global { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }
@_QQcl.06a5b026b360d1afa7f78d14a9e3c506 = external constant [57 x i8]
@_QQcl.54696D65206F662073656374696F6E20312077617320 = external constant [22 x i8]
@_QQcl.207365636F6E6473 = external constant [8 x i8]
@_QQcl.54696D65206F662073656374696F6E20322077617320 = external constant [22 x i8]
@_QQcl.54696D65206F662073656374696F6E20332077617320 = external constant [22 x i8]

declare void @malloc() local_unnamed_addr #0

declare void @free() local_unnamed_addr #0

define void @cloudsc_(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8) local_unnamed_addr #1 {
.preheader10577:
  br label %._crit_edge10585

._crit_edge10585:                                 ; preds = %._crit_edge10585, %.preheader10577
  br i1 poison, label %.preheader10576.1, label %._crit_edge10585

.preheader10576.1:                                ; preds = %._crit_edge10585
  br label %._crit_edge10600

._crit_edge10600:                                 ; preds = %._crit_edge10600, %.preheader10576.1
  br i1 poison, label %.preheader10572, label %._crit_edge10600

.preheader10572:                                  ; preds = %._crit_edge10600
  br i1 poison, label %._crit_edge10603.4, label %.preheader10571

.preheader10571:                                  ; preds = %.preheader10571, %.preheader10572
  br label %.preheader10571

._crit_edge10603.4:                               ; preds = %.preheader10572
  br label %.lr.ph10604

.lr.ph10604:                                      ; preds = %.lr.ph10604, %._crit_edge10603.4
  br i1 poison, label %._crit_edge10606, label %.lr.ph10604

._crit_edge10606:                                 ; preds = %.lr.ph10604
  br i1 poison, label %._crit_edge10609.4, label %.preheader10569.1

.preheader10569.1:                                ; preds = %.preheader10569.1, %._crit_edge10606
  br label %.preheader10569.1

._crit_edge10609.4:                               ; preds = %._crit_edge10606
  br i1 poison, label %._crit_edge10612, label %.lr.ph10611.preheader

.lr.ph10611.preheader:                            ; preds = %._crit_edge10609.4
  br label %.lr.ph10611

.lr.ph10611:                                      ; preds = %.lr.ph10611, %.lr.ph10611.preheader
  br label %.lr.ph10611

._crit_edge10612:                                 ; preds = %._crit_edge10609.4
  br label %._crit_edge10636

._crit_edge10636:                                 ; preds = %._crit_edge10636, %._crit_edge10612
  br i1 poison, label %.loopexit10562.loopexit, label %._crit_edge10636

.loopexit10562.loopexit:                          ; preds = %._crit_edge10636
  %9 = load i32, ptr %0, align 4
  br i1 poison, label %.lr.ph10876.preheader, label %._crit_edge10877

.lr.ph10876.preheader:                            ; preds = %.loopexit10562.loopexit
  br label %.lr.ph10876

.lr.ph10876:                                      ; preds = %.lr.ph10876, %.lr.ph10876.preheader
  %10 = phi i64 [ %23, %.lr.ph10876 ], [ poison, %.lr.ph10876.preheader ]
  %11 = phi i32 [ %22, %.lr.ph10876 ], [ %9, %.lr.ph10876.preheader ]
  %12 = sext i32 %11 to i64
  %13 = add nsw i64 %12, -1
  %14 = getelementptr double, ptr %1, i64 %13
  store double 0.000000e+00, ptr %14, align 8
  %15 = getelementptr double, ptr %3, i64 %13
  store double 0.000000e+00, ptr %15, align 8
  %16 = getelementptr double, ptr %4, i64 %13
  store double 0.000000e+00, ptr %16, align 8
  %17 = getelementptr double, ptr %2, i64 %13
  store double 0.000000e+00, ptr %17, align 8
  %18 = getelementptr double, ptr %5, i64 %13
  store double 0.000000e+00, ptr %18, align 8
  %19 = getelementptr double, ptr %6, i64 %13
  store double 0.000000e+00, ptr %19, align 8
  %20 = getelementptr double, ptr %7, i64 %13
  store double 0.000000e+00, ptr %20, align 8
  %21 = getelementptr double, ptr %8, i64 %13
  store double 0.000000e+00, ptr %21, align 8
  %22 = add i32 %11, 1
  %23 = add nsw i64 %10, -1
  %24 = icmp ugt i64 %10, 1
  br i1 %24, label %.lr.ph10876, label %._crit_edge10877

._crit_edge10877:                                 ; preds = %.lr.ph10876, %.loopexit10562.loopexit
  ret void
}

declare void @_FortranACpuTime() local_unnamed_addr

declare void @vdiv_() local_unnamed_addr

declare void @vexp_() local_unnamed_addr

declare void @vrec_() local_unnamed_addr

declare void @cuadjtq_() local_unnamed_addr

declare void @vpow_() local_unnamed_addr

declare void @_FortranAioBeginExternalListOutput() local_unnamed_addr

declare void @_FortranAioOutputAscii() local_unnamed_addr

declare void @_FortranAioOutputReal64() local_unnamed_addr

declare void @_FortranAioEndIoStatement() local_unnamed_addr

declare void @_FortranASumReal8x1_contract_simplified() local_unnamed_addr

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.exp.f64(double) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.pow.f64(double, double) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.copysign.f64(double, double) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #2

attributes #0 = { "alloc-family"="malloc" }
attributes #1 = { "target-features"="+m,+a,+f,+d,+c,+zepi,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl32b,+zvl64b,-e,-h,-zihintpause,-zfhmin,-zfh,-zfinx,-zdinx,-zhinxmin,-zhinx,-zba,-zbb,-zbc,-zbs,-zbkb,-zbkc,-zbkx,-zknd,-zkne,-zknh,-zksed,-zksh,-zkr,-zkn,-zks,-zkt,-zk,-zmmul,-v,-zvl128b,-zvl256b,-zvl512b,-zvl1024b,-zvl2048b,-zvl4096b,-zvl8192b,-zvl16384b,-zvl32768b,-zvl65536b,-zicbom,-zicboz,-zicbop,-zicsr,-zifencei,-zawrs,-svnapot,-svpbmt,-svinval,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-experimental-zihintntl,-experimental-zca,-experimental-zcb,-experimental-zcd,-experimental-zcf,-experimental-zfa,-experimental-zvfh,-experimental-ztso,-relax,-save-restore" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"target-abi", !"lp64d"}
