! RUN: %python %S/test_errors.py %s %flang_fc1
module my_mod

contains
  subroutine my_sub(str)
    implicit none
    character(len=*), intent(out) :: str

    character(len=1,kind=1), dimension(1:len(str)+1), target :: array_of_char
  end subroutine my_sub

end module my_mod
