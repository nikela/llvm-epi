// RUN: fir-opt --boxed-procedure %s | FileCheck %s

module {
  func.func @_QPcheck(%m : i1, %args2 : !fir.boxproc<() -> ()>) {
    cf.cond_br %m, ^bb2, ^bb3(%args2: !fir.boxproc<() -> ()>)
  ^bb2:
    return
  ^bb3(%t2 : !fir.boxproc<() -> ()>):
    %t3 = fir.box_addr %t2 : (!fir.boxproc<() -> ()>) -> (() -> ())
    fir.call %t3() : () -> ()
    return
  }
}

// CHECK: module {
// CHECK-NEXT:   func.func @_QPcheck(%arg0: i1, %arg1: () -> ()) {
// CHECK-NEXT:     cf.cond_br %arg0, ^bb1, ^bb2(%arg1 : () -> ())
// CHECK-NEXT:   ^bb1:  // pred: ^bb0
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb2(%0: () -> ()):  // pred: ^bb0
// CHECK-NEXT:     %1 = fir.convert %0 : (() -> ()) -> (() -> ())
// CHECK-NEXT:     fir.call %1() : () -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }

