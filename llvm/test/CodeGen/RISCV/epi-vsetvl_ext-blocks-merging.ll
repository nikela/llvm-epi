define void @test_preserve_extra(i64 %rvl, i64 %extra, i64 %x, <vscale x 1 x double>* %a, <vscale x 1 x double>* %b, <vscale x 1 x double>* %c) {
entry:
  %0 = tail call i64 @llvm.epi.vsetvl.ext(i64 %rvl, i64 2, i64 0, i64 %extra)
  %1 = tail call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* %a, i64 %0)
  %2 = tail call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* %b, i64 %0)
  %3 = tail call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(<vscale x 1 x double> %1, <vscale x 1 x double> %2, i64 %0)
  tail call void @llvm.epi.vstore.nxv1f64(<vscale x 1 x double> %3, <vscale x 1 x double>* %c, i64 %0)
  ret void
}

define void @test_in_if_branching(i64 %rvl, i64 %extra, i64 %x, <vscale x 1 x double>* %a, <vscale x 1 x double>* %b, <vscale x 1 x double>* %c) {
entry:
  %0 = tail call i64 @llvm.epi.vsetvl(i64 %rvl, i64 2, i64 0)
  %cmp = icmp sgt i64 %x, 3
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = tail call i64 @llvm.epi.vsetvl.ext(i64 %rvl, i64 3, i64 0, i64 %extra)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %gvl.0 = phi i64 [ %1, %if.then ], [ %0, %entry ]
  %2 = tail call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* %a, i64 %gvl.0)
  %3 = tail call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* %b, i64 %gvl.0)
  %4 = tail call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(<vscale x 1 x double> %2, <vscale x 1 x double> %3, i64 %gvl.0)
  tail call void @llvm.epi.vstore.nxv1f64(<vscale x 1 x double> %4, <vscale x 1 x double>* %c, i64 %gvl.0)
  ret void
}

define void @test_before_if_branching(i64 %rvl, i64 %extra, i64 %x, <vscale x 1 x double>* %a, <vscale x 1 x double>* %b, <vscale x 1 x double>* %c) {
entry:
  %0 = tail call i64 @llvm.epi.vsetvl.ext(i64 %rvl, i64 2, i64 0, i64 %extra)
  %cmp = icmp sgt i64 %x, 3
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = tail call i64 @llvm.epi.vsetvl(i64 %rvl, i64 3, i64 0)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %gvl.0 = phi i64 [ %1, %if.then ], [ %0, %entry ]
  %2 = tail call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* %a, i64 %gvl.0)
  %3 = tail call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* %b, i64 %gvl.0)
  %4 = tail call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(<vscale x 1 x double> %2, <vscale x 1 x double> %3, i64 %gvl.0)
  tail call void @llvm.epi.vstore.nxv1f64(<vscale x 1 x double> %4, <vscale x 1 x double>* %c, i64 %gvl.0)
  ret void
}

define void @test_if_else_branching(i64 %rvl, i64 %extra, i64 %x, <vscale x 1 x double>* %a, <vscale x 1 x double>* %b, <vscale x 1 x double>* %c) {
entry:
  %cmp = icmp sgt i64 %x, 3
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %0 = tail call i64 @llvm.epi.vsetvl(i64 %rvl, i64 3, i64 0)
  br label %if.end

if.else:                                          ; preds = %entry
  %1 = tail call i64 @llvm.epi.vsetvl.ext(i64 %rvl, i64 2, i64 0, i64 %extra)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %gvl.0 = phi i64 [ %0, %if.then ], [ %1, %if.else ]
  %2 = tail call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* %a, i64 %gvl.0)
  %3 = tail call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* %b, i64 %gvl.0)
  %4 = tail call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(<vscale x 1 x double> %2, <vscale x 1 x double> %3, i64 %gvl.0)
  tail call void @llvm.epi.vstore.nxv1f64(<vscale x 1 x double> %4, <vscale x 1 x double>* %c, i64 %gvl.0)
  ret void
}

define void @test_if_else_if_branching(i64 %rvl, i64 %extra, i64 %extra2, i64 %x, <vscale x 1 x double>* %a, <vscale x 1 x double>* %b, <vscale x 1 x double>* %c) {
entry:
  %cmp = icmp sgt i64 %x, 3
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %0 = tail call i64 @llvm.epi.vsetvl(i64 %rvl, i64 3, i64 0)
  br label %if.end3

if.else:                                          ; preds = %entry
  %1 = tail call i64 @llvm.epi.vsetvl.ext(i64 %rvl, i64 2, i64 0, i64 %extra)
  %cmp1 = icmp slt i64 %x, 1
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:                                         ; preds = %if.else
  %2 = tail call i64 @llvm.epi.vsetvl.ext(i64 %rvl, i64 3, i64 0, i64 %extra2)
  br label %if.end3

if.end3:                                          ; preds = %if.else, %if.then2, %if.then
  %gvl.0 = phi i64 [ %0, %if.then ], [ %2, %if.then2 ], [ %1, %if.else ]
  %3 = tail call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* %a, i64 %gvl.0)
  %4 = tail call <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* %b, i64 %gvl.0)
  %5 = tail call <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(<vscale x 1 x double> %3, <vscale x 1 x double> %4, i64 %gvl.0)
  tail call void @llvm.epi.vstore.nxv1f64(<vscale x 1 x double> %5, <vscale x 1 x double>* %c, i64 %gvl.0)
  ret void
}

declare i64 @llvm.epi.vsetvl.ext(i64, i64, i64, i64)
declare i64 @llvm.epi.vsetvl(i64, i64, i64)

declare <vscale x 1 x double> @llvm.epi.vload.nxv1f64(<vscale x 1 x double>* nocapture, i64)
declare <vscale x 1 x double> @llvm.epi.vfadd.nxv1f64.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>, i64)
declare void @llvm.epi.vstore.nxv1f64(<vscale x 1 x double>, <vscale x 1 x double>* nocapture, i64)
