! RUN: not %python %S/test_errors.py %s %flang_fc1
! Crazy case that does not make sense but apparently all the compilers
! resolve the intrinsic return types right before the implicit-part, giving
! x a type of real.
integer(kind=kind(x)) function foo(x)
  implicit none
  !ERROR: The type of 'x' has already been implicitly declared
  integer :: x
  foo = x
end function
