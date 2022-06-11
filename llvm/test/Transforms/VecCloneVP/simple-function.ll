; RUN: opt -S --vec-clone-vp %s | FileCheck %s

define i64 @simple_function(i64 %x, i64 %y) #0 {
entry:
  %x.addr = alloca i64, align 8
  %y.addr = alloca i64, align 8
  store i64 %x, ptr %x.addr, align 8
  store i64 %y, ptr %y.addr, align 8
  %0 = load i64, ptr %x.addr, align 8
  %1 = load i64, ptr %y.addr, align 8
  %add = add nsw i64 %0, %1
  ret i64 %add
}

define i64 @simple_function_without_allocas(i64 %x, i64 %y) #1 {
entry:
  %add = add nsw i64 %x, %y
  ret i64 %add
}

; CHECK-LABEL: @_ZGVEMk1vv_simple_function(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VSCALE:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[ASSUME_COND:%.*]] = icmp ule i32 [[VL:%.*]], [[VSCALE]]
; CHECK-NEXT:    call void @llvm.assume(i1 [[ASSUME_COND]])
; CHECK-NEXT:    [[ZEXT_MASK:%.*]] = zext <vscale x 1 x i1> [[MASK:%.*]] to <vscale x 1 x i64>
; CHECK-NEXT:    [[VEC_MASK:%.*]] = alloca <vscale x 1 x i64>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv1i64.p0(<vscale x 1 x i64> [[ZEXT_MASK]], ptr [[VEC_MASK]], <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> poison, i1 true, i32 0), <vscale x 1 x i1> poison, <vscale x 1 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    [[VEC_X:%.*]] = alloca <vscale x 1 x i64>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv1i64.p0(<vscale x 1 x i64> [[X:%.*]], ptr [[VEC_X]], <vscale x 1 x i1> [[MASK]], i32 [[VL]])
; CHECK-NEXT:    [[VEC_Y:%.*]] = alloca <vscale x 1 x i64>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv1i64.p0(<vscale x 1 x i64> [[Y:%.*]], ptr [[VEC_Y]], <vscale x 1 x i1> [[MASK]], i32 [[VL]])
; CHECK-NEXT:    [[VEC_RET:%.*]] = alloca <vscale x 1 x i64>, align 8
; CHECK-NEXT:    [[VL_CHECK:%.*]] = icmp uge i32 [[VL]], 0
; CHECK-NEXT:    br i1 [[VL_CHECK]], label %simd.loop, label %return
; CHECK:       simd.loop:                                        ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, %entry ], [ [[INDVAR:%.*]], %simd.loop.exit ]
; CHECK-NEXT:    [[MASK_GEP:%.*]] = getelementptr i64, ptr [[VEC_MASK]], i32 [[INDEX]]
; CHECK-NEXT:    [[MASK_PARAM:%.*]] = load i64, ptr [[MASK_GEP]], align 8
; CHECK-NEXT:    [[MASK_VALUE:%.*]] = icmp ne i64 [[MASK_PARAM]], 0
; CHECK-NEXT:    br i1 [[MASK_VALUE]], label %simd.loop.then, label %simd.loop.exit
; CHECK:       simd.loop.then:                                   ; preds = %simd.loop
; CHECK-NEXT:    [[VEC_X_GEP:%.*]] = getelementptr <vscale x 1 x i64>, ptr [[VEC_X]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM:%.*]] = load i64, ptr [[VEC_X_GEP]], align 8
; CHECK-NEXT:    [[VEC_Y_GEP:%.*]] = getelementptr <vscale x 1 x i64>, ptr [[VEC_Y]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM:%.*]] = load i64, ptr [[VEC_Y_GEP]], align 8
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i64 [[VEC_X_ELEM]], [[VEC_Y_ELEM]]
; CHECK-NEXT:    [[VEC_RET_GEP:%.*]] = getelementptr i64, ptr [[VEC_RET]], i32 [[INDEX]]
; CHECK-NEXT:    store i64 [[ADD]], ptr [[VEC_RET_GEP]], align 8
; CHECK-NEXT:    br label %simd.loop.exit
; CHECK:       simd.loop.exit:                                   ; preds = %simd.loop, %simd.loop.then
; CHECK-NEXT:    [[INDVAR]] = add nsw i32 [[INDEX]], 1
; CHECK-NEXT:    [[EXIT_COND:%.*]] = icmp eq i32 [[INDVAR]], [[VL]]
; CHECK-NEXT:    br i1 [[EXIT_COND]], label %return, label %simd.loop, !llvm.loop !0
; CHECK:       return:                                           ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[VEC_RET_1:%.*]] = call <vscale x 1 x i64> @llvm.vp.load.nxv1i64.p0(ptr [[VEC_RET]], <vscale x 1 x i1> [[MASK]], i32 [[VL]])
; CHECK-NEXT:    ret <vscale x 1 x i64> [[VEC_RET_1]]
;

; CHECK-LABEL: @_ZGVENk2vv_simple_function(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VSCALE:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i32 [[VSCALE]], 2
; CHECK-NEXT:    [[ASSUME_COND:%.*]] = icmp ule i32 [[VL:%.*]], [[TMP1]]
; CHECK-NEXT:    call void @llvm.assume(i1 [[ASSUME_COND]])
; CHECK-NEXT:    [[VEC_X:%.*]] = alloca <vscale x 2 x i64>, align 16
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i64.p0(<vscale x 2 x i64> [[X:%.*]], ptr [[VEC_X]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    [[VEC_Y:%.*]] = alloca <vscale x 2 x i64>, align 16
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i64.p0(<vscale x 2 x i64> [[Y:%.*]], ptr [[VEC_Y]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    [[VEC_RET:%.*]] = alloca <vscale x 2 x i64>, align 16
; CHECK-NEXT:    [[VL_CHECK:%.*]] = icmp uge i32 [[VL]], 0
; CHECK-NEXT:    br i1 [[VL_CHECK]], label %simd.loop, label %return
; CHECK:       simd.loop:                                        ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, %entry ], [ [[INDVAR:%.*]], %simd.loop.exit ]
; CHECK-NEXT:    [[VEC_X_GEP:%.*]] = getelementptr <vscale x 2 x i64>, ptr [[VEC_X]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM:%.*]] = load i64, ptr [[VEC_X_GEP]], align 8
; CHECK-NEXT:    [[VEC_Y_GEP:%.*]] = getelementptr <vscale x 2 x i64>, ptr [[VEC_Y]], i32 0, i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM:%.*]] = load i64, ptr [[VEC_Y_GEP]], align 8
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i64 [[VEC_X_ELEM]], [[VEC_Y_ELEM]]
; CHECK-NEXT:    [[VEC_RET_GEP:%.*]] = getelementptr i64, ptr [[VEC_RET]], i32 [[INDEX]]
; CHECK-NEXT:    store i64 [[ADD]], ptr [[VEC_RET_GEP]], align 8
; CHECK-NEXT:    br label %simd.loop.exit
; CHECK:       simd.loop.exit:                                   ; preds = %simd.loop
; CHECK-NEXT:    [[INDVAR]] = add nsw i32 [[INDEX]], 1
; CHECK-NEXT:    [[EXIT_COND:%.*]] = icmp eq i32 [[INDVAR]], [[VL]]
; CHECK-NEXT:    br i1 [[EXIT_COND]], label %return, label %simd.loop, !llvm.loop !8
; CHECK:       return:                                           ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[VEC_RET_1:%.*]] = call <vscale x 2 x i64> @llvm.vp.load.nxv2i64.p0(ptr [[VEC_RET]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    ret <vscale x 2 x i64> [[VEC_RET_1]]
;

; CHECK-LABEL: @_ZGVEMk1vv_simple_function_without_allocas(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VSCALE:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[ASSUME_COND:%.*]] = icmp ule i32 [[VL:%.*]], [[VSCALE]]
; CHECK-NEXT:    call void @llvm.assume(i1 [[ASSUME_COND]])
; CHECK-NEXT:    [[ZEXT_MASK:%.*]] = zext <vscale x 1 x i1> [[MASK:%.*]] to <vscale x 1 x i64>
; CHECK-NEXT:    [[VEC_MASK:%.*]] = alloca <vscale x 1 x i64>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv1i64.p0(<vscale x 1 x i64> [[ZEXT_MASK]], ptr [[VEC_MASK]], <vscale x 1 x i1> shufflevector (<vscale x 1 x i1> insertelement (<vscale x 1 x i1> poison, i1 true, i32 0), <vscale x 1 x i1> poison, <vscale x 1 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    [[VEC_X:%.*]] = alloca <vscale x 1 x i64>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv1i64.p0(<vscale x 1 x i64> [[X:%.*]], ptr [[VEC_X]], <vscale x 1 x i1> [[MASK]], i32 [[VL]])
; CHECK-NEXT:    [[VEC_Y:%.*]] = alloca <vscale x 1 x i64>, align 8
; CHECK-NEXT:    call void @llvm.vp.store.nxv1i64.p0(<vscale x 1 x i64> [[Y:%.*]], ptr [[VEC_Y]], <vscale x 1 x i1> [[MASK]], i32 [[VL]])
; CHECK-NEXT:    [[VEC_RET:%.*]] = alloca <vscale x 1 x i64>, align 8
; CHECK-NEXT:    [[VL_CHECK:%.*]] = icmp uge i32 [[VL]], 0
; CHECK-NEXT:    br i1 [[VL_CHECK]], label %simd.loop, label %return
; CHECK:       simd.loop:                                        ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, %entry ], [ [[INDVAR:%.*]], %simd.loop.exit ]
; CHECK-NEXT:    [[MASK_GEP:%.*]] = getelementptr i64, ptr [[VEC_MASK]], i32 [[INDEX]]
; CHECK-NEXT:    [[MASK_PARAM:%.*]] = load i64, ptr [[MASK_GEP]], align 8
; CHECK-NEXT:    [[MASK_VALUE:%.*]] = icmp ne i64 [[MASK_PARAM]], 0
; CHECK-NEXT:    br i1 [[MASK_VALUE]], label %simd.loop.then, label %simd.loop.exit
; CHECK:       simd.loop.then:                                   ; preds = %simd.loop
; CHECK-NEXT:    [[VEC_X_GEP:%.*]] = getelementptr i64, ptr [[VEC_X]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM:%.*]] = load i64, ptr [[VEC_X_GEP]], align 8
; CHECK-NEXT:    [[VEC_Y_GEP:%.*]] = getelementptr i64, ptr [[VEC_Y]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM:%.*]] = load i64, ptr [[VEC_Y_GEP]], align 8
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i64 [[VEC_X_ELEM]], [[VEC_Y_ELEM]]
; CHECK-NEXT:    [[VEC_RET_GEP:%.*]] = getelementptr i64, ptr [[VEC_RET]], i32 [[INDEX]]
; CHECK-NEXT:    store i64 [[ADD]], ptr [[VEC_RET_GEP]], align 8
; CHECK-NEXT:    br label %simd.loop.exit
; CHECK:       simd.loop.exit:                                   ; preds = %simd.loop, %simd.loop.then
; CHECK-NEXT:    [[INDVAR]] = add nsw i32 [[INDEX]], 1
; CHECK-NEXT:    [[EXIT_COND:%.*]] = icmp eq i32 [[INDVAR]], [[VL]]
; CHECK-NEXT:    br i1 [[EXIT_COND]], label %return, label %simd.loop, !llvm.loop !10
; CHECK:       return:                                           ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[VEC_RET_1:%.*]] = call <vscale x 1 x i64> @llvm.vp.load.nxv1i64.p0(ptr [[VEC_RET]], <vscale x 1 x i1> [[MASK]], i32 [[VL]])
; CHECK-NEXT:    ret <vscale x 1 x i64> [[VEC_RET_1]]
;

; CHECK-LABEL: @_ZGVENk2vv_simple_function_without_allocas(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VSCALE:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i32 [[VSCALE]], 2
; CHECK-NEXT:    [[ASSUME_COND:%.*]] = icmp ule i32 [[VL:%.*]], [[TMP1]]
; CHECK-NEXT:    call void @llvm.assume(i1 [[ASSUME_COND]])
; CHECK-NEXT:    [[VEC_X:%.*]] = alloca <vscale x 2 x i64>, align 16
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i64.p0(<vscale x 2 x i64> [[X:%.*]], ptr [[VEC_X]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    [[VEC_Y:%.*]] = alloca <vscale x 2 x i64>, align 16
; CHECK-NEXT:    call void @llvm.vp.store.nxv2i64.p0(<vscale x 2 x i64> [[Y:%.*]], ptr [[VEC_Y]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    [[VEC_RET:%.*]] = alloca <vscale x 2 x i64>, align 16
; CHECK-NEXT:    [[VL_CHECK:%.*]] = icmp uge i32 [[VL]], 0
; CHECK-NEXT:    br i1 [[VL_CHECK]], label %simd.loop, label %return
; CHECK:       simd.loop:                                        ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, %entry ], [ [[INDVAR:%.*]], %simd.loop.exit ]
; CHECK-NEXT:    [[VEC_X_GEP:%.*]] = getelementptr i64, ptr [[VEC_X]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_X_ELEM:%.*]] = load i64, ptr [[VEC_X_GEP]], align 8
; CHECK-NEXT:    [[VEC_Y_GEP:%.*]] = getelementptr i64, ptr [[VEC_Y]], i32 [[INDEX]]
; CHECK-NEXT:    [[VEC_Y_ELEM:%.*]] = load i64, ptr [[VEC_Y_GEP]], align 8
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i64 [[VEC_X_ELEM]], [[VEC_Y_ELEM]]
; CHECK-NEXT:    [[VEC_RET_GEP:%.*]] = getelementptr i64, ptr [[VEC_RET]], i32 [[INDEX]]
; CHECK-NEXT:    store i64 [[ADD]], ptr [[VEC_RET_GEP]], align 8
; CHECK-NEXT:    br label %simd.loop.exit
; CHECK:       simd.loop.exit:                                   ; preds = %simd.loop
; CHECK-NEXT:    [[INDVAR]] = add nsw i32 [[INDEX]], 1
; CHECK-NEXT:    [[EXIT_COND:%.*]] = icmp eq i32 [[INDVAR]], [[VL]]
; CHECK-NEXT:    br i1 [[EXIT_COND]], label %return, label %simd.loop, !llvm.loop !11
; CHECK:       return:                                           ; preds = %simd.loop.exit, %entry
; CHECK-NEXT:    [[VEC_RET_1:%.*]] = call <vscale x 2 x i64> @llvm.vp.load.nxv2i64.p0(ptr [[VEC_RET]], <vscale x 2 x i1> shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer), i32 [[VL]])
; CHECK-NEXT:    ret <vscale x 2 x i64> [[VEC_RET_1]]
;

attributes #0 = { "_ZGVEMk1vv_simple_function" "_ZGVENk2vv_simple_function" }
attributes #1 = { "_ZGVEMk1vv_simple_function_without_allocas" "_ZGVENk2vv_simple_function_without_allocas" }
