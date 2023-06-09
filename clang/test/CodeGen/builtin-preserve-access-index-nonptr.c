// RUN: %clang_cc1 -triple x86_64 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

#define _(x) (__builtin_preserve_access_index(x))

struct s1 {
  char a;
  int b[4];
};

int unit1(struct s1 *arg) {
  return _(arg->b[2]);
}
// CHECK: define dso_local i32 @unit1
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S1:[0-9]+]]
// CHECK: call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([4 x i32]) %{{[0-9a-z]+}}, i32 1, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[ARRAY:[0-9]+]]
//
// CHECK: ![[ARRAY]] = !DICompositeType(tag: DW_TAG_array_type
// CHECK: ![[STRUCT_S1]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1"
