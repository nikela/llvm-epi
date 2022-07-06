! RUN: bbc %s -o "-" -emit-fir | FileCheck %s

integer(1) function fct1()
end
! CHECK-LABEL: func @_QPfct1() -> i8
! CHECK:         return %{{.*}} : i8

integer(2) function fct2()
end
! CHECK-LABEL: func @_QPfct2() -> i16
! CHECK:         return %{{.*}} : i16

integer(4) function fct3()
end
! CHECK-LABEL: func @_QPfct3() -> i32
! CHECK:         return %{{.*}} : i32

integer(8) function fct4()
end
! CHECK-LABEL: func @_QPfct4() -> i64
! CHECK:         return %{{.*}} : i64

function fct()
  integer :: fct
end
! CHECK-LABEL: func @_QPfct() -> i32
! CHECK:         return %{{.*}} : i32

function fct_res() result(res)
  integer :: res
end
! CHECK-LABEL: func @_QPfct_res() -> i32
! CHECK:         return %{{.*}} : i32

integer function fct_body()
  goto 1
  1 stop
end

! CHECK-LABEL: func @_QPfct_body() -> i32
! CHECK:         cf.br ^bb1
! CHECK:       ^bb1
! CHECK:         %{{.*}} = fir.call @_FortranAStopStatement
! CHECK:         fir.unreachable

function fct_iarr1()
  integer, dimension(10) :: fct_iarr1
end

! CHECK-LABEL: func @_QPfct_iarr1() -> !fir.array<10xi32>
! CHECK:         return %{{.*}} : !fir.array<10xi32>

function fct_iarr2()
  integer, dimension(10, 20) :: fct_iarr2
end

! CHECK-LABEL: func @_QPfct_iarr2() -> !fir.array<10x20xi32>
! CHECK:         return %{{.*}} : !fir.array<10x20xi32>

logical(1) function lfct1()
end
! CHECK-LABEL: func @_QPlfct1() -> !fir.logical<1>
! CHECK:         return %{{.*}} : !fir.logical<1>

logical(2) function lfct2()
end
! CHECK-LABEL: func @_QPlfct2() -> !fir.logical<2>
! CHECK:         return %{{.*}} : !fir.logical<2>

logical(4) function lfct3()
end
! CHECK-LABEL: func @_QPlfct3() -> !fir.logical<4>
! CHECK:         return %{{.*}} : !fir.logical<4>

logical(8) function lfct4()
end
! CHECK-LABEL: func @_QPlfct4() -> !fir.logical<8>
! CHECK:         return %{{.*}} : !fir.logical<8>

real function rfct3()
end
! CHECK-LABEL: func @_QPrfct3() -> f32
! CHECK:         return %{{.*}} : f32

real(8) function rfct4()
end
! CHECK-LABEL: func @_QPrfct4() -> f64
! CHECK:         return %{{.*}} : f64

complex(4) function cplxfct3()
end
! CHECK-LABEL: func @_QPcplxfct3() -> !fir.complex<4>
! CHECK:         return %{{.*}} : !fir.complex<4>

complex(8) function cplxfct4()
end
! CHECK-LABEL: func @_QPcplxfct4() -> !fir.complex<8>
! CHECK:         return %{{.*}} : !fir.complex<8>

function fct_with_character_return(i)
  character(10) :: fct_with_character_return
  integer :: i
end
! CHECK-LABEL: func @_QPfct_with_character_return(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.char<1,10>>{{.*}}, %{{.*}}: index{{.*}}, %{{.*}}: !fir.ref<i32>{{.*}}) -> !fir.boxchar<1> {
