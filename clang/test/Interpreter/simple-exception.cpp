// clang-format off
// This does not seem to work with a clang that is cross by default
// which suggests "target=" is not precise enough here.
// UNSUPPORTED: true
// XFAIL for arm, arm64, riscv, or running on Windows.
// XFAIL: target={{(arm|riscv).*}}, system-windows
// RUN: cat %s | clang-repl | FileCheck %s
extern "C" int printf(const char *, ...);

int f() { throw "Simple exception"; return 0; }
int checkException() { try { printf("Running f()\n"); f(); } catch (const char *e) { printf("%s\n", e); } return 0; }
auto r1 = checkException();
// CHECK: Running f()
// CHECK-NEXT: Simple exception

%quit
