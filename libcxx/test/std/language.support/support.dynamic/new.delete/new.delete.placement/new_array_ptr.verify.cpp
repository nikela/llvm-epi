//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <new>

// void* operator new[](std::size_t, void *);

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <new>

void f() {
    char buffer[100];
    ::operator new[](4, buffer); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
