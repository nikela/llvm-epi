//===- ValueTypes.td - ValueType definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Value types - These values correspond to the register types defined in the
// MachineValueTypes.h file.  If you update anything here, you must update it
// there as well!
//
//===----------------------------------------------------------------------===//

class ValueType<int size, int value> {
  string Namespace = "MVT";
  int Size = size;
  int Value = value;
}

def OtherVT: ValueType<0  ,  1>;   // "Other" value
def i1     : ValueType<1  ,  2>;   // One bit boolean value
def i2     : ValueType<2  ,  3>;   // 2-bit integer value
def i4     : ValueType<4  ,  4>;   // 4-bit integer value
def i8     : ValueType<8  ,  5>;   // 8-bit integer value
def i16    : ValueType<16 ,  6>;   // 16-bit integer value
def i32    : ValueType<32 ,  7>;   // 32-bit integer value
def i64    : ValueType<64 ,  8>;   // 64-bit integer value
def i128   : ValueType<128,  9>;   // 128-bit integer value

def bf16   : ValueType<16 , 10>;   // 16-bit brain floating point value
def f16    : ValueType<16 , 11>;   // 16-bit floating point value
def f32    : ValueType<32 , 12>;   // 32-bit floating point value
def f64    : ValueType<64 , 13>;   // 64-bit floating point value
def f80    : ValueType<80 , 14>;   // 80-bit floating point value
def f128   : ValueType<128, 15>;   // 128-bit floating point value
def ppcf128: ValueType<128, 16>;   // PPC 128-bit floating point value

def v1i1   : ValueType<1 ,  17>;   //   1 x i1 vector value
def v2i1   : ValueType<2 ,  18>;   //   2 x i1 vector value
def v4i1   : ValueType<4 ,  19>;   //   4 x i1 vector value
def v8i1   : ValueType<8 ,  20>;   //   8 x i1 vector value
def v16i1  : ValueType<16,  21>;   //  16 x i1 vector value
def v32i1  : ValueType<32 , 22>;   //  32 x i1 vector value
def v64i1  : ValueType<64 , 23>;   //  64 x i1 vector value
def v128i1 : ValueType<128, 24>;   // 128 x i1 vector value
def v256i1 : ValueType<256, 25>;   // 256 x i1 vector value
def v512i1 : ValueType<512, 26>;   // 512 x i1 vector value
def v1024i1: ValueType<1024,27>;   //1024 x i1 vector value

def v1i8   : ValueType<8,   28>;   //  1 x i8  vector value
def v2i8   : ValueType<16 , 29>;   //  2 x i8  vector value
def v4i8   : ValueType<32 , 30>;   //  4 x i8  vector value
def v8i8   : ValueType<64 , 31>;   //  8 x i8  vector value
def v16i8  : ValueType<128, 32>;   // 16 x i8  vector value
def v32i8  : ValueType<256, 33>;   // 32 x i8  vector value
def v64i8  : ValueType<512, 34>;   // 64 x i8  vector value
def v128i8 : ValueType<1024,35>;   //128 x i8  vector value
def v256i8 : ValueType<2048,36>;   //256 x i8  vector value

def v1i16  : ValueType<16 , 37>;   //  1 x i16 vector value
def v2i16  : ValueType<32 , 38>;   //  2 x i16 vector value
def v3i16  : ValueType<48 , 39>;   //  3 x i16 vector value
def v4i16  : ValueType<64 , 40>;   //  4 x i16 vector value
def v8i16  : ValueType<128, 41>;   //  8 x i16 vector value
def v16i16 : ValueType<256, 42>;   // 16 x i16 vector value
def v32i16 : ValueType<512, 43>;   // 32 x i16 vector value
def v64i16 : ValueType<1024,44>;   // 64 x i16 vector value
def v128i16: ValueType<2048,45>;   //128 x i16 vector value

def v1i32    : ValueType<32 , 46>;   //  1 x i32 vector value
def v2i32    : ValueType<64 , 47>;   //  2 x i32 vector value
def v3i32    : ValueType<96 , 48>;   //  3 x i32 vector value
def v4i32    : ValueType<128, 49>;   //  4 x i32 vector value
def v5i32    : ValueType<160, 50>;   //  5 x i32 vector value
def v8i32    : ValueType<256, 51>;   //  8 x i32 vector value
def v16i32   : ValueType<512, 52>;   // 16 x i32 vector value
def v32i32   : ValueType<1024,53>;   // 32 x i32 vector value
def v64i32   : ValueType<2048,54>;   // 64 x i32 vector value
def v128i32  : ValueType<4096,55>;   // 128 x i32 vector value
def v256i32  : ValueType<8182,56>;   // 256 x i32 vector value
def v512i32  : ValueType<16384,57>;  // 512 x i32 vector value
def v1024i32 : ValueType<32768,58>;  // 1024 x i32 vector value
def v2048i32 : ValueType<65536,59>;  // 2048 x i32 vector value

def v1i64  : ValueType<64 , 60>;   //  1 x i64 vector value
def v2i64  : ValueType<128, 61>;   //  2 x i64 vector value
def v4i64  : ValueType<256, 62>;   //  4 x i64 vector value
def v8i64  : ValueType<512, 63>;   //  8 x i64 vector value
def v16i64 : ValueType<1024,64>;   // 16 x i64 vector value
def v32i64 : ValueType<2048,65>;   // 32 x i64 vector value

def v1i128 : ValueType<128, 66>;   //  1 x i128 vector value

def v2f16    : ValueType<32 , 67>;    //    2 x f16 vector value
def v3f16    : ValueType<48 , 68>;    //    3 x f16 vector value
def v4f16    : ValueType<64 , 69>;    //    4 x f16 vector value
def v8f16    : ValueType<128, 70>;    //    8 x f16 vector value
def v16f16   : ValueType<256, 71>;    //   16 x f16 vector value
def v32f16   : ValueType<512, 72>;    //   32 x f16 vector value
def v2bf16   : ValueType<32 , 73>;    //    2 x bf16 vector value
def v3bf16   : ValueType<48 , 74>;    //    3 x bf16 vector value
def v4bf16   : ValueType<64 , 75>;    //    4 x bf16 vector value
def v8bf16   : ValueType<128, 76>;    //    8 x bf16 vector value
def v16bf16  : ValueType<256, 77>;    //   16 x bf16 vector value
def v32bf16  : ValueType<512, 78>;    //   32 x bf16 vector value
def v1f32    : ValueType<32 , 79>;    //    1 x f32 vector value
def v2f32    : ValueType<64 , 80>;    //    2 x f32 vector value
def v3f32    : ValueType<96 , 81>;    //    3 x f32 vector value
def v4f32    : ValueType<128, 82>;    //    4 x f32 vector value
def v5f32    : ValueType<160, 83>;    //    5 x f32 vector value
def v8f32    : ValueType<256, 84>;    //    8 x f32 vector value
def v16f32   : ValueType<512,  85>;   //   16 x f32 vector value
def v32f32   : ValueType<1024, 86>;   //   32 x f32 vector value
def v64f32   : ValueType<2048, 87>;   //   64 x f32 vector value
def v128f32  : ValueType<4096, 88>;   //  128 x f32 vector value
def v256f32  : ValueType<8182, 89>;   //  256 x f32 vector value
def v512f32  : ValueType<16384, 90>;  //  512 x f32 vector value
def v1024f32 : ValueType<32768, 91>;  // 1024 x f32 vector value
def v2048f32 : ValueType<65536, 92>;  // 2048 x f32 vector value
def v1f64    : ValueType<64, 93>;     //    1 x f64 vector value
def v2f64    : ValueType<128, 94>;    //    2 x f64 vector value
def v4f64    : ValueType<256, 95>;    //    4 x f64 vector value
def v8f64    : ValueType<512, 96>;    //    8 x f64 vector value
def v16f64   : ValueType<1024, 97>;   //   16 x f64 vector value

def nxv1i1  : ValueType<1,   98>;  // n x  1 x i1  vector value
def nxv2i1  : ValueType<2,   99>;  // n x  2 x i1  vector value
def nxv4i1  : ValueType<4,  100>;  // n x  4 x i1  vector value
def nxv8i1  : ValueType<8,  101>;  // n x  8 x i1  vector value
def nxv16i1 : ValueType<16, 102>;  // n x 16 x i1  vector value
def nxv32i1 : ValueType<32, 103>;  // n x 32 x i1  vector value

def nxv1i8  : ValueType<8,   104>;  // n x  1 x i8  vector value
def nxv2i8  : ValueType<16,  105>;  // n x  2 x i8  vector value
def nxv4i8  : ValueType<32,  106>;  // n x  4 x i8  vector value
def nxv8i8  : ValueType<64,  107>;  // n x  8 x i8  vector value
def nxv16i8 : ValueType<128, 108>;  // n x 16 x i8  vector value
def nxv32i8 : ValueType<256, 109>;  // n x 32 x i8  vector value
def nxv64i8 : ValueType<256, 110>;  // n x 64 x i8  vector value

def nxv1i16 : ValueType<16,  111>; // n x  1 x i16 vector value
def nxv2i16 : ValueType<32,  112>; // n x  2 x i16 vector value
def nxv4i16 : ValueType<64,  113>; // n x  4 x i16 vector value
def nxv8i16 : ValueType<128, 114>; // n x  8 x i16 vector value
def nxv16i16: ValueType<256, 115>; // n x 16 x i16 vector value
def nxv32i16: ValueType<512, 116>; // n x 32 x i16 vector value

def nxv1i32 : ValueType<32,  117>; // n x  1 x i32 vector value
def nxv2i32 : ValueType<64,  118>; // n x  2 x i32 vector value
def nxv4i32 : ValueType<128, 119>; // n x  4 x i32 vector value
def nxv8i32 : ValueType<256, 120>; // n x  8 x i32 vector value
def nxv16i32: ValueType<512, 121>; // n x 16 x i32 vector value
def nxv32i32: ValueType<1024,122>; // n x 32 x i32 vector value

def nxv1i64 : ValueType<64,  123>; // n x  1 x i64 vector value
def nxv2i64 : ValueType<128, 124>; // n x  2 x i64 vector value
def nxv4i64 : ValueType<256, 125>; // n x  4 x i64 vector value
def nxv8i64 : ValueType<512, 126>; // n x  8 x i64 vector value
def nxv16i64: ValueType<1024,127>; // n x 16 x i64 vector value
def nxv32i64: ValueType<2048,128>; // n x 32 x i64 vector value

// EPIVR Masks
def nxv16i4 : ValueType<64, 129>;   // n x 16 x i4 vector value
def nxv32i2 : ValueType<64, 130>;   // n x 32 x i2 vector value
def nxv64i1 : ValueType<64, 131>;   // n x 64 x i1 vector value

def nxv2f16  : ValueType<32 , 132>; // n x  2 x f16 vector value
def nxv4f16  : ValueType<64 , 133>; // n x  4 x f16 vector value
def nxv8f16  : ValueType<128, 134>; // n x  8 x f16 vector value
def nxv2bf16 : ValueType<32 , 135>; // n x  2 x bf16 vector value
def nxv4bf16 : ValueType<64 , 136>; // n x  4 x bf16 vector value
def nxv8bf16 : ValueType<128, 137>; // n x  8 x bf16 vector value
def nxv1f32  : ValueType<32 , 138>; // n x  1 x f32 vector value
def nxv2f32  : ValueType<64 , 139>; // n x  2 x f32 vector value
def nxv4f32  : ValueType<128, 140>; // n x  4 x f32 vector value
def nxv8f32  : ValueType<256, 141>; // n x  8 x f32 vector value
def nxv16f32 : ValueType<512, 142>; // n x 16 x f32 vector value
def nxv1f64  : ValueType<64,  143>; // n x  1 x f64 vector value
def nxv2f64  : ValueType<128, 144>; // n x  2 x f64 vector value
def nxv4f64  : ValueType<256, 145>; // n x  4 x f64 vector value
def nxv8f64  : ValueType<512, 146>; // n x  8 x f64 vector value

def x86mmx : ValueType<64 , 147>;   // X86 MMX value
def FlagVT : ValueType<0  , 148>;   // Pre-RA sched glue
def isVoid : ValueType<0  , 149>;   // Produces no value
def untyped: ValueType<8  , 150>;   // Produces an untyped value
def exnref : ValueType<0  , 151>;      // WebAssembly's exnref type
def token  : ValueType<0  , 248>;   // TokenTy
def MetadataVT: ValueType<0, 249>;  // Metadata

// Pseudo valuetype mapped to the current pointer size to any address space.
// Should only be used in TableGen.
def iPTRAny   : ValueType<0, 250>;

// Pseudo valuetype to represent "vector of any size"
def vAny   : ValueType<0  , 251>;

// Pseudo valuetype to represent "float of any format"
def fAny   : ValueType<0  , 252>;

// Pseudo valuetype to represent "integer of any bit width"
def iAny   : ValueType<0  , 253>;

// Pseudo valuetype mapped to the current pointer size.
def iPTR   : ValueType<0  , 254>;

// Pseudo valuetype to represent "any type of any size".
def Any    : ValueType<0  , 255>;

/// This class is for targets that want to use pointer types in patterns
/// with the GlobalISelEmitter.  Targets must define their own pointer
/// derived from this class.  The scalar argument should be an
/// integer type with the same bit size as the ponter.
/// e.g. def p0 : PtrValueType <i64, 0>;

class PtrValueType <ValueType scalar, int addrspace> :
    ValueType<scalar.Size, scalar.Value> {
  int AddrSpace = addrspace;
}
