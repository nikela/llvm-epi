//===-- Implementation of rintl function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/rintl.h"
#include "src/__support/common.h"
#include "utils/FPUtil/NearestIntegerOperations.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(long double, rintl, (long double x)) {
  return fputil::roundUsingCurrentRoundingMode(x);
}

} // namespace __llvm_libc
