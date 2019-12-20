// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++17 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++2a %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s

namespace dr2352 { // dr2352: 10
  int **p;
  const int *const *const &f1() { return p; }
  int *const *const &f2() { return p; }
  int **const &f3() { return p; }

  const int **const &f4() { return p; } // expected-error {{reference to type 'const int **const' could not bind to an lvalue of type 'int **'}}
  const int *const *&f5() { return p; } // expected-error {{binding reference of type 'const int *const *' to value of type 'int **' not permitted due to incompatible qualifiers}}

  // FIXME: We permit this as a speculative defect resolution, allowing
  // qualification conversions when forming a glvalue conditional expression.
  const int * const * const q = 0;
  __typeof(&(true ? p : q)) x = &(true ? p : q);

  // FIXME: Should we compute the composite pointer type here and produce an
  // lvalue of type 'const int *const * const'?
  const int * const * r;
  void *y = &(true ? p : r); // expected-error {{rvalue of type 'const int *const *'}}

  // FIXME: We order these as a speculative defect resolution.
  void f(const int * const * const &r);
#if __cplusplus >= 201103L
  constexpr
#endif
  int *const *const &f(int * const * const &r) { return r; }

  // No temporary is created here.
  int *const *const &check_f = f(p);
#if __cplusplus >= 201103L
  static_assert(&p == &check_f, "");
#endif
}

namespace dr2353 { // dr2353: 9
  struct X {
    static const int n = 0;
  };

  // CHECK: FunctionDecl {{.*}} use
  int use(X x) {
    // CHECK: MemberExpr {{.*}} .n
    // CHECK-NOT: non_odr_use
    // CHECK: DeclRefExpr {{.*}} 'x'
    // CHECK-NOT: non_odr_use
    return *&x.n;
  }
#pragma clang __debug dump use

  // CHECK: FunctionDecl {{.*}} not_use
  int not_use(X x) {
    // CHECK: MemberExpr {{.*}} .n {{.*}} non_odr_use_constant
    // CHECK: DeclRefExpr {{.*}} 'x'
    return x.n;
  }
#pragma clang __debug dump not_use

  // CHECK: FunctionDecl {{.*}} not_use_2
  int not_use_2(X *x) {
    // CHECK: MemberExpr {{.*}} ->n {{.*}} non_odr_use_constant
    // CHECK: DeclRefExpr {{.*}} 'x'
    return x->n;
  }
#pragma clang __debug dump not_use_2
}

#if __cplusplus >= 201707L
// Otherwise, if the qualified-id std::tuple_size<E> names a complete class
// type **with a member value**, the expression std::tuple_size<E>::value shall
// be a well-formed integral constant expression
namespace dr2386 { // dr2386: 9
struct Bad1 { int a, b; };
struct Bad2 { int a, b; };
} // namespace dr2386
namespace std {
template <typename T> struct tuple_size;
template <> struct std::tuple_size<dr2386::Bad1> {};
template <> struct std::tuple_size<dr2386::Bad2> {
  static const int value = 42;
};
} // namespace std
namespace dr2386 {
void no_value() { auto [x, y] = Bad1(); }
void wrong_value() { auto [x, y] = Bad2(); } // expected-error {{decomposes into 42 elements}}
} // namespace dr2386
#endif

namespace dr2387 { // dr2387: 9
#if __cplusplus >= 201402L
  template<int> int a = 0;
  extern template int a<0>; // ok

  template<int> static int b = 0;
  extern template int b<0>; // expected-error {{internal linkage}}

  template<int> const int c = 0;
  extern template const int c<0>; // ok, has external linkage despite 'const'

  template<typename T> T d = 0;
  extern template int d<int>;
  extern template const int d<const int>;
#endif
}
