add_entrypoint_object(
  sched_getaffinity
  SRCS
    sched_getaffinity.cpp
  HDRS
    ../sched_getaffinity.h
  DEPENDS
    libc.include.sched
    libc.src.__support.OSUtil.osutil
    libc.src.errno.errno
)

add_entrypoint_object(
  sched_setaffinity
  SRCS
    sched_setaffinity.cpp
  HDRS
    ../sched_setaffinity.h
  DEPENDS
    libc.include.sched
    libc.src.__support.OSUtil.osutil
    libc.src.errno.errno
)

add_entrypoint_object(
  __sched_getcpucount
  SRCS
    sched_getcpucount.cpp
  HDRS
    ../sched_getcpucount.h
  DEPENDS
    libc.include.sched
)
