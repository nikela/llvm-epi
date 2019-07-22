// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: llvm-objdump -triple armv7a-none-linux-gnueabi -d --no-show-raw-insn %t | FileCheck %s --check-prefix=DISASM
// RUN: llvm-readobj -r --symbols --sections %t | FileCheck %s
 .syntax unified
 .text
 .type foo STT_GNU_IFUNC
 .globl foo
foo:
 bx lr

 .type bar STT_GNU_IFUNC
 .globl bar
bar:
 bx lr

 .globl _start
_start:
 bl foo
 bl bar
 movw r0,:lower16:__rel_iplt_start
 movt r0,:upper16:__rel_iplt_start
 movw r0,:lower16:__rel_iplt_end
 movt r0,:upper16:__rel_iplt_end

// CHECK: Sections [
// CHECK:   Section {
// CHECK:        Section {
// CHECK:          Name: .rel.dyn
// CHECK-NEXT:     Type: SHT_REL
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x100F4
// CHECK-NEXT:     Offset: 0xF4
// CHECK-NEXT:     Size: 16
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info: 4
// CHECK:          Name: .plt
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_EXECINSTR
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x11020
// CHECK-NEXT:     Offset: 0x1020
// CHECK-NEXT:     Size: 32
// CHECK:          Index: 4
// CHECK-NEXT:     Name: .got
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x12000
// CHECK-NEXT:     Offset: 0x2000
// CHECK-NEXT:     Size: 8
// CHECK:      Relocations [
// CHECK-NEXT:   Section (1) .rel.dyn {
// CHECK-NEXT:     0x12000 R_ARM_IRELATIVE
// CHECK-NEXT:     0x12004 R_ARM_IRELATIVE
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK:        Symbol {
// CHECK:          Name: __rel_iplt_end
// CHECK-NEXT:     Value: 0x10104
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other [
// CHECK-NEXT:       STV_HIDDEN
// CHECK-NEXT:     ]
// CHECK-NEXT:     Section: .rel.dyn
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: __rel_iplt_start
// CHECK-NEXT:     Value: 0x100F4
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other [
// CHECK-NEXT:       STV_HIDDEN
// CHECK-NEXT:     ]
// CHECK-NEXT:     Section: .rel.dyn
// CHECK-NEXT:   }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: _start
// CHECK-NEXT:    Value: 0x11008
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other:
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: bar
// CHECK-NEXT:    Value: 0x11004
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: GNU_IFunc
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: foo
// CHECK-NEXT:    Value: 0x11000
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: GNU_IFunc
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }

// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: foo:
// DISASM-NEXT:    11000:      bx      lr
// DISASM: bar:
// DISASM-NEXT:    11004:      bx      lr
// DISASM: _start:
// DISASM-NEXT:    11008:      bl      #16
// DISASM-NEXT:    1100c:      bl      #28
// 1 * 65536 + 244 = 0x100f4 __rel_iplt_start
// DISASM-NEXT:    11010:      movw    r0, #244
// DISASM-NEXT:    11014:      movt    r0, #1
// 1 * 65536 + 260 = 0x10104 __rel_iplt_end
// DISASM-NEXT:    11018:      movw    r0, #260
// DISASM-NEXT:    1101c:      movt    r0, #1
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: $a:
// DISASM-NEXT:    11020:       add     r12, pc, #0, #12
// DISASM-NEXT:    11024:       add     r12, r12, #0, #20
// DISASM-NEXT:    11028:       ldr     pc, [r12, #4056]!
// DISASM: $d:
// DISASM-NEXT:    1102c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM: $a:
// DISASM-NEXT:    11030:       add     r12, pc, #0, #12
// DISASM-NEXT:    11034:       add     r12, r12, #0, #20
// DISASM-NEXT:    11038:       ldr     pc, [r12, #4044]!
// DISASM: $d:
// DISASM-NEXT:    1103c:       d4 d4 d4 d4     .word   0xd4d4d4d4

