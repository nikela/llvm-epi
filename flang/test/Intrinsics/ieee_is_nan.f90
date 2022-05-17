! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPverify_dummy_name
function verify_dummy_name(a) result(l)
  use, intrinsic :: ieee_arithmetic, only : ieee_is_nan
  implicit none

  real(kind=8) :: a
  logical :: l

  l = ieee_is_nan(x = a)
end function verify_dummy_name

! CHECK-LABEL: func @_QPverify_elemental
function verify_elemental(a, n) result(l)
  use, intrinsic :: ieee_arithmetic, only : ieee_is_nan
  implicit none

  integer :: n
  real(kind=8) :: a(n)
  logical :: l(n)

  l = ieee_is_nan(x = a)
end function verify_elemental
