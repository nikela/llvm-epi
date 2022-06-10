; RUN: opt -S --vec-clone-vp %s | FileCheck %s

define i32 @simple_control_flow(i32 %X, i32 %Y, i32 %W) #0 {
entry:
  %cmp = icmp sgt i32 %X, 4
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %add = add nuw nsw i32 %X, 4
  %add1 = add nsw i32 %Y, 5
  br label %if.end6

if.else:                                          ; preds = %entry
  %cmp2 = icmp sgt i32 %Y, 7
  br i1 %cmp2, label %if.then3, label %if.end6

if.then3:                                         ; preds = %if.else
  %add4 = add nsw i32 %X, 5
  %add5 = add nsw i32 %W, 5
  br label %if.end6

if.end6:                                          ; preds = %if.else, %if.then3, %if.then
  %X.addr.0 = phi i32 [ %add, %if.then ], [ %add4, %if.then3 ], [ %X, %if.else ]
  %Y.addr.0 = phi i32 [ %add1, %if.then ], [ %Y, %if.then3 ], [ %Y, %if.else ]
  %W.addr.0 = phi i32 [ %W, %if.then ], [ %add5, %if.then3 ], [ %W, %if.else ]
  %add7 = add nsw i32 %Y.addr.0, %X.addr.0
  %add8 = add nsw i32 %add7, %W.addr.0
  ret i32 %add8
}

define i32 @simple_control_flow_with_allocas(i32 %X, i32 %Y, i32 %W) #1 {
entry:
  %X.addr = alloca i32, align 4
  %Y.addr = alloca i32, align 4
  %W.addr = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  store i32 %Y, ptr %Y.addr, align 4
  store i32 %W, ptr %W.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  %cmp = icmp sgt i32 %0, 4
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %add = add nsw i32 %1, 4
  store i32 %add, ptr %X.addr, align 4
  %2 = load i32, ptr %Y.addr, align 4
  %add1 = add nsw i32 %2, 5
  store i32 %add1, ptr %Y.addr, align 4
  br label %if.end6

if.else:                                          ; preds = %entry
  %3 = load i32, ptr %Y.addr, align 4
  %cmp2 = icmp sgt i32 %3, 7
  br i1 %cmp2, label %if.then3, label %if.end

if.then3:                                         ; preds = %if.else
  %4 = load i32, ptr %X.addr, align 4
  %add4 = add nsw i32 %4, 5
  store i32 %add4, ptr %X.addr, align 4
  %5 = load i32, ptr %W.addr, align 4
  %add5 = add nsw i32 %5, 5
  store i32 %add5, ptr %W.addr, align 4
  br label %if.end

if.end:                                           ; preds = %if.then3, %if.else
  br label %if.end6

if.end6:                                          ; preds = %if.end, %if.then
  %6 = load i32, ptr %X.addr, align 4
  %7 = load i32, ptr %Y.addr, align 4
  %add7 = add nsw i32 %6, %7
  %8 = load i32, ptr %W.addr, align 4
  %add8 = add nsw i32 %add7, %8
  ret i32 %add8
}

; CHECK-LABEL: @_ZGVEMk2vvv_simple_control_flow(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VSCALE:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i32 [[VSCALE]], 2
; CHECK-NEXT:    [[ASSUME_COND:%.*]] = icmp ule i32 [[VL:%.*]], [[TMP1]]
; CHECK-NEXT:    call void @llvm.assume(i1 [[ASSUME_COND]])
; CHECK-NEXT:    [[ZEXT_MASK:%.*]] = zext <vscale x 2 x i1> [[MASK:%.*]] to <vscale x 2 x i32>
; CHECK-NEXT:    [[VEC_MASK:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0(<vscale x 2 x i32> [[ZEXT_MASK]], ptr [[VEC_MASK]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    [[VEC_X:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0(<vscale x 2 x i32> [[X:%.*]], ptr [[VEC_X]], <vscale x 2 x i1> [[MASK]], i32 [[VL]])
; CHECK-NEXT:    [[VEC_Y:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0(<vscale x 2 x i32> [[Y:%.*]], ptr [[VEC_Y]], <vscale x 2 x i1> [[MASK]], i32 [[VL]])
; CHECK-NEXT:    [[VEC_W:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0(<vscale x 2 x i32> [[W:%.*]], ptr [[VEC_W]], <vscale x 2 x i1> [[MASK]], i32 [[VL]])
; CHECK-NEXT:    [[VEC_RET:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    [[VL_CHECK:%.*]] = icmp uge i32 [[VL]], 0
; CHECK-NEXT:    br i1 [[VL_CHECK]], label %simd.loop, label %return
; CHECK:       simd.loop:                                        ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, %entry ], [ [[INDVAR:%.*]], %simd.loop.exit ]
; CHECK-NEXT:    [[VEC_MASK_GEP:%.*]] = getelementptr i32, ptr [[VEC_MASK]], i32 [[INDEX]]
; CHECK-NEXT:    [[MASK_PARAM:%.*]] = load i32, ptr [[VEC_MASK_GEP]], align 4
; CHECK-NEXT:    [[MASK_VALUE:%.*]] = icmp ne i32 [[MASK_PARAM]], 0
; CHECK-NEXT:    br i1 [[MASK_VALUE]], label %simd.loop.then, label %simd.loop.exit
; CHECK:       simd.loop.then:                                   ; preds = %simd.loop
; CHECK-NEXT:    [[VEC_X_GEP5:%.*]] = getelementptr i32, ptr [[VEC_X]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM6:%.*]] = load i32, ptr [[VEC_X_GEP5]], align 4
; CHECK-NEXT:    [[CMP:%.*]] = icmp sgt i32 [[VEC_X_ELEM6]], 4
; CHECK-NEXT:    br i1 [[CMP]], label %if.then, label %if.else
; CHECK:       if.then:                                          ; preds = %simd.loop.then
; CHECK-NEXT:    [[VEC_X_GEP3:%.*]] = getelementptr i32, ptr [[VEC_X]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM4:%.*]] = load i32, ptr [[VEC_X_GEP3]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add nuw nsw i32 [[VEC_X_ELEM4]], 4
; CHECK-NEXT:    [[VEC_Y_GEP11:%.*]] = getelementptr i32, ptr [[VEC_Y]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM12:%.*]] = load i32, ptr [[VEC_Y_GEP11]], align 4
; CHECK-NEXT:    [[ADD1:%.*]] = add nsw i32 [[VEC_Y_ELEM12]], 5
; CHECK-NEXT:    [[VEC_W_GEP13:%.*]] = getelementptr i32, ptr [[VEC_W]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_W_ELEM14:%.*]] = load i32, ptr [[VEC_W_GEP13]], align 4
; CHECK-NEXT:    br label %if.end6
; CHECK:       if.else:                                          ; preds = %simd.loop.then
; CHECK-NEXT:    [[VEC_Y_GEP9:%.*]] = getelementptr i32, ptr [[VEC_Y]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM10:%.*]] = load i32, ptr [[VEC_Y_GEP9]], align 4
; CHECK-NEXT:    [[CMP2:%.*]] = icmp sgt i32 [[VEC_Y_ELEM10]], 7
; CHECK-NEXT:    [[VEC_X_GEP:%.*]] = getelementptr i32, ptr [[VEC_X]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM:%.*]] = load i32, ptr [[VEC_X_GEP]], align 4
; CHECK-NEXT:    [[VEC_Y_GEP:%.*]] = getelementptr i32, ptr [[VEC_Y]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM:%.*]] = load i32, ptr [[VEC_Y_GEP]], align 4
; CHECK-NEXT:    [[VEC_W_GEP:%.*]] = getelementptr i32, ptr [[VEC_W]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_W_ELEM:%.*]] = load i32, ptr [[VEC_W_GEP]], align 4
; CHECK-NEXT:    br i1 [[CMP2]], label %if.then3, label %if.end6
; CHECK:       if.then3:                                         ; preds = %if.else
; CHECK-NEXT:    [[VEC_X_GEP1:%.*]] = getelementptr i32, ptr [[VEC_X]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM2:%.*]] = load i32, ptr [[VEC_X_GEP1]], align 4
; CHECK-NEXT:    [[ADD4:%.*]] = add nsw i32 [[VEC_X_ELEM2]], 5
; CHECK-NEXT:    [[VEC_W_GEP15:%.*]] = getelementptr i32, ptr [[VEC_W]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_W_ELEM16:%.*]] = load i32, ptr [[VEC_W_GEP15]], align 4
; CHECK-NEXT:    [[ADD5:%.*]] = add nsw i32 [[VEC_W_ELEM16]], 5
; CHECK-NEXT:    [[VEC_Y_GEP7:%.*]] = getelementptr i32, ptr [[VEC_Y]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM8:%.*]] = load i32, ptr [[VEC_Y_GEP7]], align 4
; CHECK-NEXT:    br label %if.end6
; CHECK:       if.end6:                                          ; preds = %if.then3, %if.else, %if.then
; CHECK-NEXT:    [[X_ADDR_0:%.*]] = phi i32 [ [[ADD]], %if.then ], [ [[ADD4]], %if.then3 ], [ [[VEC_X_ELEM]], %if.else ]
; CHECK-NEXT:    [[Y_ADDR_0:%.*]] = phi i32 [ [[ADD1]], %if.then ], [ [[VEC_Y_ELEM8]], %if.then3 ], [ [[VEC_Y_ELEM]], %if.else ]
; CHECK-NEXT:    [[W_ADDR_0:%.*]] = phi i32 [ [[VEC_W_ELEM14]], %if.then ], [ [[ADD5]], %if.then3 ], [ [[VEC_W_ELEM]], %if.else ]
; CHECK-NEXT:    [[ADD7:%.*]] = add nsw i32 [[Y_ADDR_0]], [[X_ADDR_0]]
; CHECK-NEXT:    [[ADD8:%.*]] = add nsw i32 [[ADD7]], [[W_ADDR_0]]
; CHECK-NEXT:    [[VEC_RET_GEP:%.*]] = getelementptr i32, ptr [[VEC_RET]], i32 [[INDEX]]
; CHECK-NEXT:    store i32 [[ADD8]], ptr [[VEC_RET_GEP]], align 4
; CHECK-NEXT:    br label %simd.loop.exit
; CHECK:       simd.loop.exit:                                   ; preds = %simd.loop, %if.end6
; CHECK-NEXT:    [[INDVAR:%.*]] = add nsw i32 [[INDEX]], 1
; CHECK-NEXT:    [[EXIT_COND:%.*]] = icmp eq i32 [[INDVAR]], [[VL]]
; CHECK-NEXT:    br i1 [[EXIT_COND]], label %return, label %simd.loop, !llvm.loop !0
; CHECK:       return:                                           ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[VEC_RET17:%.*]] = call <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0(ptr [[VEC_RET]], <vscale x 2 x i1> [[MASK]], i32 [[VL]])
; CHECK-NEXT:    ret <vscale x 2 x i32> [[VEC_RET17]]

; CHECK-LABEL: @_ZGVENk2vvv_simple_control_flow(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VSCALE:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i32 [[VSCALE]], 2
; CHECK-NEXT:    [[ASSUME_COND:%.*]] = icmp ule i32 [[VL:%.*]], [[TMP1]]
; CHECK-NEXT:    call void @llvm.assume(i1 [[ASSUME_COND]])
; CHECK-NEXT:    [[VEC_X:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0(<vscale x 2 x i32> [[X:%.*]], ptr [[VEC_X]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    [[VEC_Y:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0(<vscale x 2 x i32> [[Y:%.*]], ptr [[VEC_Y]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    [[VEC_W:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0(<vscale x 2 x i32> [[W:%.*]], ptr [[VEC_W]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    [[VEC_RET:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    [[VL_CHECK:%.*]] = icmp uge i32 [[VL]], 0
; CHECK-NEXT:    br i1 [[VL_CHECK]], label %simd.loop, label %return
; CHECK:       simd.loop:                                        ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, %entry ], [ [[INDVAR:%.*]], %simd.loop.exit ]
; CHECK-NEXT:    [[VEC_X_GEP5:%.*]] = getelementptr i32, ptr [[VEC_X]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM6:%.*]] = load i32, ptr [[VEC_X_GEP5]], align 4
; CHECK-NEXT:    [[CMP:%.*]] = icmp sgt i32 [[VEC_X_ELEM6]], 4
; CHECK-NEXT:    br i1 [[CMP]], label %if.then, label %if.else
; CHECK:       if.then:                                          ; preds = %simd.loop
; CHECK-NEXT:    [[VEC_X_GEP3:%.*]] = getelementptr i32, ptr [[VEC_X]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM4:%.*]] = load i32, ptr [[VEC_X_GEP3]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add nuw nsw i32 [[VEC_X_ELEM4]], 4
; CHECK-NEXT:    [[VEC_Y_GEP11:%.*]] = getelementptr i32, ptr [[VEC_Y]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM12:%.*]] = load i32, ptr [[VEC_Y_GEP11]], align 4
; CHECK-NEXT:    [[ADD1:%.*]] = add nsw i32 [[VEC_Y_ELEM12]], 5
; CHECK-NEXT:    [[VEC_W_GEP13:%.*]] = getelementptr i32, ptr [[VEC_W]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_W_ELEM14:%.*]] = load i32, ptr [[VEC_W_GEP13]], align 4
; CHECK-NEXT:    br label %if.end6
; CHECK:       if.else:                                          ; preds = %simd.loop
; CHECK-NEXT:    [[VEC_Y_GEP9:%.*]] = getelementptr i32, ptr [[VEC_Y]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM10:%.*]] = load i32, ptr [[VEC_Y_GEP9]], align 4
; CHECK-NEXT:    [[CMP2:%.*]] = icmp sgt i32 [[VEC_Y_ELEM10]], 7
; CHECK-NEXT:    [[VEC_X_GEP:%.*]] = getelementptr i32, ptr [[VEC_X]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM:%.*]] = load i32, ptr [[VEC_X_GEP]], align 4
; CHECK-NEXT:    [[VEC_Y_GEP:%.*]] = getelementptr i32, ptr [[VEC_Y]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM:%.*]] = load i32, ptr [[VEC_Y_GEP]], align 4
; CHECK-NEXT:    [[VEC_W_GEP:%.*]] = getelementptr i32, ptr [[VEC_W]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_W_ELEM:%.*]] = load i32, ptr [[VEC_W_GEP]], align 4
; CHECK-NEXT:    br i1 [[CMP2]], label %if.then3, label %if.end6
; CHECK:       if.then3:                                         ; preds = %if.else
; CHECK-NEXT:    [[VEC_X_GEP1:%.*]] = getelementptr i32, ptr [[VEC_X]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM2:%.*]] = load i32, ptr [[VEC_X_GEP1]], align 4
; CHECK-NEXT:    [[ADD4:%.*]] = add nsw i32 [[VEC_X_ELEM2]], 5
; CHECK-NEXT:    [[VEC_W_GEP15:%.*]] = getelementptr i32, ptr [[VEC_W]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_W_ELEM16:%.*]] = load i32, ptr [[VEC_W_GEP15]], align 4
; CHECK-NEXT:    [[ADD5:%.*]] = add nsw i32 [[VEC_W_ELEM16]], 5
; CHECK-NEXT:    [[VEC_Y_GEP7:%.*]] = getelementptr i32, ptr [[VEC_Y]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM8:%.*]] = load i32, ptr [[VEC_Y_GEP7]], align 4
; CHECK-NEXT:    br label %if.end6
; CHECK:       if.end6:                                          ; preds = %if.then3, %if.else, %if.then
; CHECK-NEXT:    [[X_ADDR_0:%.*]] = phi i32 [ [[ADD]], %if.then ], [ [[ADD4]], %if.then3 ], [ [[VEC_X_ELEM]], %if.else ]
; CHECK-NEXT:    [[Y_ADDR_0:%.*]] = phi i32 [ [[ADD1]], %if.then ], [ [[VEC_Y_ELEM8]], %if.then3 ], [ [[VEC_Y_ELEM]], %if.else ]
; CHECK-NEXT:    [[W_ADDR_0:%.*]] = phi i32 [ [[VEC_W_ELEM14]], %if.then ], [ [[ADD5]], %if.then3 ], [ [[VEC_W_ELEM]], %if.else ]
; CHECK-NEXT:    [[ADD7:%.*]] = add nsw i32 [[Y_ADDR_0]], [[X_ADDR_0]]
; CHECK-NEXT:    [[ADD8:%.*]] = add nsw i32 [[ADD7]], [[W_ADDR_0]]
; CHECK-NEXT:    [[VEC_RET_GEP:%.*]] = getelementptr i32, ptr [[VEC_RET]], i32 [[INDEX]]
; CHECK-NEXT:    store i32 [[ADD8]], ptr [[VEC_RET_GEP]], align 4
; CHECK-NEXT:    br label %simd.loop.exit
; CHECK:       simd.loop.exit:                                   ; preds = %if.end6
; CHECK-NEXT:    [[INDVAR:%.*]] = add nsw i32 [[INDEX]], 1
; CHECK-NEXT:    [[EXIT_COND:%.*]] = icmp eq i32 [[INDVAR]], [[VL]]
; CHECK-NEXT:    br i1 [[EXIT_COND]], label %return, label %simd.loop, !llvm.loop !8
; CHECK:       return:                                           ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[VEC_RET17:%.*]] = call <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0(ptr [[VEC_RET]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    ret <vscale x 2 x i32> [[VEC_RET17]]

; CHECK-LABEL: @_ZGVEMk2vvv_simple_control_flow_with_allocas(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VSCALE:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i32 [[VSCALE]], 2
; CHECK-NEXT:    [[ASSUME_COND:%.*]] = icmp ule i32 [[VL:%.*]], [[TMP1]]
; CHECK-NEXT:    call void @llvm.assume(i1 [[ASSUME_COND]])
; CHECK-NEXT:    [[ZEXT_MASK:%.*]] = zext <vscale x 2 x i1> [[MASK:%.*]] to <vscale x 2 x i32>
; CHECK-NEXT:    [[VEC_MASK:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0(<vscale x 2 x i32> [[ZEXT_MASK]], ptr [[VEC_MASK]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    [[VEC_X:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0(<vscale x 2 x i32> [[X:%.*]], ptr [[VEC_X]], <vscale x 2 x i1> [[MASK]], i32 [[VL]])
; CHECK-NEXT:    [[VEC_Y:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0(<vscale x 2 x i32> [[Y:%.*]], ptr [[VEC_Y]], <vscale x 2 x i1> [[MASK]], i32 [[VL]])
; CHECK-NEXT:    [[VEC_W:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0(<vscale x 2 x i32> [[W:%.*]], ptr [[VEC_W]], <vscale x 2 x i1> [[MASK]], i32 [[VL]])
; CHECK-NEXT:    [[VEC_RET:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    [[VL_CHECK:%.*]] = icmp uge i32 [[VL]], 0
; CHECK-NEXT:    br i1 [[VL_CHECK]], label %simd.loop, label %return
; CHECK:       simd.loop:                                        ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, %entry ], [ [[INDVAR:%.*]], %simd.loop.exit ]
; CHECK-NEXT:    [[VEC_MASK_GEP:%.*]] = getelementptr i32, ptr [[VEC_MASK]], i32 [[INDEX]]
; CHECK-NEXT:    [[MASK_PARAM:%.*]] = load i32, ptr [[VEC_MASK_GEP]], align 4
; CHECK-NEXT:    [[MASK_VALUE:%.*]] = icmp ne i32 [[MASK_PARAM]], 0
; CHECK-NEXT:    br i1 [[MASK_VALUE]], label %simd.loop.then, label %simd.loop.exit
; CHECK:       simd.loop.then:                                   ; preds = %simd.loop
; CHECK-NEXT:    [[VEC_X_GEP5:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_X]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM6:%.*]] = load i32, ptr [[VEC_X_GEP5]], align 4
; CHECK-NEXT:    [[CMP:%.*]] = icmp sgt i32 [[VEC_X_ELEM6]], 4
; CHECK-NEXT:    br i1 [[CMP]], label %if.then, label %if.else
; CHECK:       if.then:                                          ; preds = %simd.loop.then
; CHECK-NEXT:    [[VEC_X_GEP4:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_X]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM4:%.*]] = load i32, ptr [[VEC_X_GEP4]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[VEC_X_ELEM4]], 4
; CHECK-NEXT:    [[VEC_X_GEP3:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_X]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    store i32 [[ADD]], ptr [[VEC_X_GEP3]], align 4
; CHECK-NEXT:    [[VEC_Y_GEP8:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_Y]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM12:%.*]] = load i32, ptr [[VEC_Y_GEP8]], align 4
; CHECK-NEXT:    [[ADD1:%.*]] = add nsw i32 [[VEC_Y_ELEM12]], 5
; CHECK-NEXT:    [[VEC_Y_GEP7:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_Y]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    store i32 [[ADD1]], ptr [[VEC_Y_GEP7]], align 4
; CHECK-NEXT:    br label %if.end6
; CHECK:       if.else:                                          ; preds = %simd.loop.then
; CHECK-NEXT:    [[VEC_Y_GEP6:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_Y]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM10:%.*]] = load i32, ptr [[VEC_Y_GEP6]], align 4
; CHECK-NEXT:    [[CMP2:%.*]] = icmp sgt i32 [[VEC_Y_ELEM10]], 7
; CHECK-NEXT:    br i1 [[CMP2]], label %if.then3, label %if.end
; CHECK:       if.then3:                                         ; preds = %if.else
; CHECK-NEXT:    [[VEC_X_GEP2:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_X]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM2:%.*]] = load i32, ptr [[VEC_X_GEP2]], align 4
; CHECK-NEXT:    [[ADD4:%.*]] = add nsw i32 [[VEC_X_ELEM2]], 5
; CHECK-NEXT:    [[VEC_X_GEP1:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_X]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    store i32 [[ADD4]], ptr [[VEC_X_GEP1]], align 4
; CHECK-NEXT:    [[VEC_W_GEP10:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_W]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_W_ELEM16:%.*]] = load i32, ptr [[VEC_W_GEP10]], align 4
; CHECK-NEXT:    [[ADD5:%.*]] = add nsw i32 [[VEC_W_ELEM16]], 5
; CHECK-NEXT:    [[VEC_W_GEP9:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_W]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    store i32 [[ADD5]], ptr [[VEC_W_GEP9]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:                                           ; preds = %if.then3, %if.else
; CHECK-NEXT:    br label %if.end6
; CHECK:       if.end6:                                          ; preds = %if.end, %if.then
; CHECK-NEXT:    [[VEC_X_GEP:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_X]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM:%.*]] = load i32, ptr [[VEC_X_GEP]], align 4
; CHECK-NEXT:    [[VEC_Y_GEP:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_Y]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM:%.*]] = load i32, ptr [[VEC_Y_GEP]], align 4
; CHECK-NEXT:    [[ADD7:%.*]] = add nsw i32 [[VEC_X_ELEM]], [[VEC_Y_ELEM]]
; CHECK-NEXT:    [[VEC_W_GEP:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_W]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_W_ELEM:%.*]] = load i32, ptr [[VEC_W_GEP]], align 4
; CHECK-NEXT:    [[ADD8:%.*]] = add nsw i32 [[ADD7]], [[VEC_W_ELEM]]
; CHECK-NEXT:    [[VEC_RET_GEP:%.*]] = getelementptr i32, ptr [[VEC_RET]], i32 [[INDEX]]
; CHECK-NEXT:    store i32 [[ADD8]], ptr [[VEC_RET_GEP]], align 4
; CHECK-NEXT:    br label %simd.loop.exit
; CHECK:       simd.loop.exit:                                   ; preds = %simd.loop, %if.end6
; CHECK-NEXT:    [[INDVAR:%.*]] = add nsw i32 [[INDEX]], 1
; CHECK-NEXT:    [[EXIT_COND:%.*]] = icmp eq i32 [[INDVAR]], [[VL]]
; CHECK-NEXT:    br i1 [[EXIT_COND]], label %return, label %simd.loop, !llvm.loop !9
; CHECK:       return:                                           ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[VEC_RET11:%.*]] = call <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0(ptr [[VEC_RET]], <vscale x 2 x i1> [[MASK]], i32 [[VL]])
; CHECK-NEXT:    ret <vscale x 2 x i32> [[VEC_RET11]]

; CHECK-LABEL: @_ZGVENk2vvv_simple_control_flow_with_allocas(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VSCALE:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i32 [[VSCALE]], 2
; CHECK-NEXT:    [[ASSUME_COND:%.*]] = icmp ule i32 [[VL:%.*]], [[TMP1]]
; CHECK-NEXT:    call void @llvm.assume(i1 [[ASSUME_COND]])
; CHECK-NEXT:    [[VEC_X:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0(<vscale x 2 x i32> [[X:%.*]], ptr [[VEC_X]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    [[VEC_Y:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0(<vscale x 2 x i32> [[Y:%.*]], ptr [[VEC_Y]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    [[VEC_W:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i32.p0(<vscale x 2 x i32> [[W:%.*]], ptr [[VEC_W]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    [[VEC_RET:%.*]] = alloca <vscale x 2 x i32>, align 8
; CHECK-NEXT:    [[VL_CHECK:%.*]] = icmp uge i32 [[VL]], 0
; CHECK-NEXT:    br i1 [[VL_CHECK]], label %simd.loop, label %return
; CHECK:       simd.loop:                                        ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, %entry ], [ [[INDVAR:%.*]], %simd.loop.exit ]
; CHECK-NEXT:    [[VEC_X_GEP5:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_X]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM6:%.*]] = load i32, ptr [[VEC_X_GEP5]], align 4
; CHECK-NEXT:    [[CMP:%.*]] = icmp sgt i32 [[VEC_X_ELEM6]], 4
; CHECK-NEXT:    br i1 [[CMP]], label %if.then, label %if.else
; CHECK:       if.then:                                          ; preds = %simd.loop
; CHECK-NEXT:    [[VEC_X_GEP4:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_X]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM4:%.*]] = load i32, ptr [[VEC_X_GEP4]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[VEC_X_ELEM4]], 4
; CHECK-NEXT:    [[VEC_X_GEP3:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_X]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    store i32 [[ADD]], ptr [[VEC_X_GEP3]], align 4
; CHECK-NEXT:    [[VEC_Y_GEP8:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_Y]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM12:%.*]] = load i32, ptr [[VEC_Y_GEP8]], align 4
; CHECK-NEXT:    [[ADD1:%.*]] = add nsw i32 [[VEC_Y_ELEM12]], 5
; CHECK-NEXT:    [[VEC_Y_GEP7:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_Y]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    store i32 [[ADD1]], ptr [[VEC_Y_GEP7]], align 4
; CHECK-NEXT:    br label %if.end6
; CHECK:       if.else:                                          ; preds = %simd.loop
; CHECK-NEXT:    [[VEC_Y_GEP6:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_Y]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM10:%.*]] = load i32, ptr [[VEC_Y_GEP6]], align 4
; CHECK-NEXT:    [[CMP2:%.*]] = icmp sgt i32 [[VEC_Y_ELEM10]], 7
; CHECK-NEXT:    br i1 [[CMP2]], label %if.then3, label %if.end
; CHECK:       if.then3:                                         ; preds = %if.else
; CHECK-NEXT:    [[VEC_X_GEP2:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_X]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM2:%.*]] = load i32, ptr [[VEC_X_GEP2]], align 4
; CHECK-NEXT:    [[ADD4:%.*]] = add nsw i32 [[VEC_X_ELEM2]], 5
; CHECK-NEXT:    [[VEC_X_GEP1:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_X]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    store i32 [[ADD4]], ptr [[VEC_X_GEP1]], align 4
; CHECK-NEXT:    [[VEC_W_GEP10:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_W]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_W_ELEM16:%.*]] = load i32, ptr [[VEC_W_GEP10]], align 4
; CHECK-NEXT:    [[ADD5:%.*]] = add nsw i32 [[VEC_W_ELEM16]], 5
; CHECK-NEXT:    [[VEC_W_GEP9:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_W]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    store i32 [[ADD5]], ptr [[VEC_W_GEP9]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:                                           ; preds = %if.then3, %if.else
; CHECK-NEXT:    br label %if.end6
; CHECK:       if.end6:                                          ; preds = %if.end, %if.then
; CHECK-NEXT:    [[VEC_X_GEP:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_X]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM:%.*]] = load i32, ptr [[VEC_X_GEP]], align 4
; CHECK-NEXT:    [[VEC_Y_GEP:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_Y]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM:%.*]] = load i32, ptr [[VEC_Y_GEP]], align 4
; CHECK-NEXT:    [[ADD7:%.*]] = add nsw i32 [[VEC_X_ELEM]], [[VEC_Y_ELEM]]
; CHECK-NEXT:    [[VEC_W_GEP:%.*]] = getelementptr <vscale x 2 x i32>, ptr [[VEC_W]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_W_ELEM:%.*]] = load i32, ptr [[VEC_W_GEP]], align 4
; CHECK-NEXT:    [[ADD8:%.*]] = add nsw i32 [[ADD7]], [[VEC_W_ELEM]]
; CHECK-NEXT:    [[VEC_RET_GEP:%.*]] = getelementptr i32, ptr [[VEC_RET]], i32 [[INDEX]]
; CHECK-NEXT:    store i32 [[ADD8]], ptr [[VEC_RET_GEP]], align 4
; CHECK-NEXT:    br label %simd.loop.exit
; CHECK:       simd.loop.exit:                                   ; preds = %if.end6
; CHECK-NEXT:    [[INDVAR:%.*]] = add nsw i32 [[INDEX]], 1
; CHECK-NEXT:    [[EXIT_COND:%.*]] = icmp eq i32 [[INDVAR]], [[VL]]
; CHECK-NEXT:    br i1 [[EXIT_COND]], label %return, label %simd.loop, !llvm.loop !10
; CHECK:       return:                                           ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[VEC_RET11:%.*]] = call <vscale x 2 x i32> @llvm.vp.load.nxv2i32.p0(ptr [[VEC_RET]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    ret <vscale x 2 x i32> [[VEC_RET11]]

attributes #0 = { "_ZGVEMk2vvv_simple_control_flow" "_ZGVENk2vvv_simple_control_flow" }
attributes #1 = { "_ZGVEMk2vvv_simple_control_flow_with_allocas" "_ZGVENk2vvv_simple_control_flow_with_allocas" }
