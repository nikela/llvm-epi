! RUN: %python %S/test_errors.py %s %flang_fc1
! This case makes sure we resolve an intrinsic return type in a function
! once the use-association, host-association and implicity typing has
! been settled.
module moo
  implicit none
  integer, parameter :: my_kind = 4
end module moo

real(kind = my_kind) function square(x)
    use moo
    implicit none
    real(kind = my_kind) :: x

    square = x * x
end function square
