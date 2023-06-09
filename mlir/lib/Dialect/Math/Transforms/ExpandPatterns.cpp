//===- ExpandTanh.cpp - Code to perform expanding tanh op -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements expansion of tanh op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

/// Expands tanh op into
///   1) 1-exp^{-2x} / 1+exp^{-2x}, if x => 0
///   2) exp^{2x}-1 / exp^{2x}+1  , if x < 0
static LogicalResult convertTanhOp(math::TanhOp op, PatternRewriter &rewriter) {
  auto floatType = op.getOperand().getType();
  Location loc = op.getLoc();
  auto floatOne = rewriter.getFloatAttr(floatType, 1.0);
  auto floatTwo = rewriter.getFloatAttr(floatType, 2.0);
  Value one = rewriter.create<arith::ConstantOp>(loc, floatOne);
  Value two = rewriter.create<arith::ConstantOp>(loc, floatTwo);
  Value doubledX = rewriter.create<arith::MulFOp>(loc, op.getOperand(), two);

  // Case 1: tanh(x) = 1-exp^{-2x} / 1+exp^{-2x}
  Value negDoubledX = rewriter.create<arith::NegFOp>(loc, doubledX);
  Value exp2x = rewriter.create<math::ExpOp>(loc, negDoubledX);
  Value dividend = rewriter.create<arith::SubFOp>(loc, one, exp2x);
  Value divisor = rewriter.create<arith::AddFOp>(loc, one, exp2x);
  Value positiveRes = rewriter.create<arith::DivFOp>(loc, dividend, divisor);

  // Case 2: tanh(x) = exp^{2x}-1 / exp^{2x}+1
  exp2x = rewriter.create<math::ExpOp>(loc, doubledX);
  dividend = rewriter.create<arith::SubFOp>(loc, exp2x, one);
  divisor = rewriter.create<arith::AddFOp>(loc, exp2x, one);
  Value negativeRes = rewriter.create<arith::DivFOp>(loc, dividend, divisor);

  // tanh(x) = x >= 0 ? positiveRes : negativeRes
  auto floatZero = rewriter.getFloatAttr(floatType, 0.0);
  Value zero = rewriter.create<arith::ConstantOp>(loc, floatZero);
  Value cmpRes = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE,
                                                op.getOperand(), zero);
  rewriter.replaceOpWithNewOp<arith::SelectOp>(op, cmpRes, positiveRes,
                                               negativeRes);
  return success();
}

static LogicalResult convertTanOp(math::TanOp op, PatternRewriter &rewriter) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Value operand = op.getOperand();
  Type type = operand.getType();
  Value sin = b.create<math::SinOp>(type, operand);
  Value cos = b.create<math::CosOp>(type, operand);
  Value div = b.create<arith::DivFOp>(type, sin, cos);
  rewriter.replaceOp(op, div);
  return success();
}

static LogicalResult convertCtlzOp(math::CountLeadingZerosOp op,
                                   PatternRewriter &rewriter) {
  auto operand = op.getOperand();
  auto elementTy = operand.getType();
  auto resultTy = op.getType();
  Location loc = op.getLoc();

  int bitWidth = elementTy.getIntOrFloatBitWidth();
  auto zero =
      rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(elementTy, 0));
  auto leadingZeros = rewriter.create<arith::ConstantOp>(
      loc, IntegerAttr::get(elementTy, bitWidth));

  SmallVector<Value> operands = {operand, leadingZeros, zero};
  SmallVector<Type> types = {elementTy, elementTy, elementTy};
  SmallVector<Location> locations = {loc, loc, loc};

  auto whileOp = rewriter.create<scf::WhileOp>(
      loc, types, operands,
      [&](OpBuilder &beforeBuilder, Location beforeLoc, ValueRange args) {
        // The conditional block of the while loop.
        Value input = args[0];
        Value zero = args[2];

        Value inputNotZero = beforeBuilder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, input, zero);
        beforeBuilder.create<scf::ConditionOp>(loc, inputNotZero, args);
      },
      [&](OpBuilder &afterBuilder, Location afterLoc, ValueRange args) {
        // The body of the while loop: shift right until reaching a value of 0.
        Value input = args[0];
        Value leadingZeros = args[1];

        auto one = afterBuilder.create<arith::ConstantOp>(
            loc, IntegerAttr::get(elementTy, 1));
        auto shifted =
            afterBuilder.create<arith::ShRUIOp>(loc, resultTy, input, one);
        auto leadingZerosMinusOne = afterBuilder.create<arith::SubIOp>(
            loc, resultTy, leadingZeros, one);

        afterBuilder.create<scf::YieldOp>(
            loc, ValueRange({shifted, leadingZerosMinusOne, args[2]}));
      });

  rewriter.setInsertionPointAfter(whileOp);
  rewriter.replaceOp(op, whileOp->getResult(1));
  return success();
}

void mlir::populateExpandCtlzPattern(RewritePatternSet &patterns) {
  patterns.add(convertCtlzOp);
}

void mlir::populateExpandTanPattern(RewritePatternSet &patterns) {
  patterns.add(convertTanOp);
}

void mlir::populateExpandTanhPattern(RewritePatternSet &patterns) {
  patterns.add(convertTanhOp);
}
