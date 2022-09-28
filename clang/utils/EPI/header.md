# Introduction

## Vector types

An implementation of the RISC-V V-extension features 32 vector registers of length `VLEN` bits. Each vector register holds a number of elements. The wider element, in bits, that an implementation supports is called `ELEN`. 

A vector, thus, can hold `VLEN/ELEN` elements of the widest element implemented. This also means that the same vector can hold twice that number of the element is half the size. This is, a vector of floats will always hold twice the number of elements that a vector of  doubles can hold.

Vector registers in the V-extension can be grouped. Grouping can be 1 (no grouping actually), 2, 4 or 8. Grouping means larger vectors but in a smaller number (e.g. there are only 16 registers with grouping 2). Grouping is part of the state of the extension and it is called `LMUL` (length multiplier). A `LMUL` of 1 means no grouping.

In EPI `ELEN=64` so the following types are available to operate the vectors under different `LMUL` configurations.

| Vector of | LMUL=1 | LMUL=2 | LMUL=4 | LMUL=8 |
| ----- | ------ | ------ | ---- | ---- |
| `double` | `__epi_1xf64` | `__epi_2xf64` | `__epi_4xf64` | `__epi_8xf64` |
| `float` | `__epi_2xf32` | `__epi_4xf32` | `__epi_8xf32` | `__epi_16xf32` |
| `int64_t` | `__epi_1xi64` | `__epi_2xi64` | `__epi_4xi64` | `__epi_8xi64` |
| `int32_t` | `__epi_2xi32` | `__epi_4xi32` | `__epi_8xi32` | `__epi_16xi32` |
| `int16_t` | `__epi_4xi16` | `__epi_8xi16` | `__epi_16xi16` | `__epi_32xi16` |
| `int8_t` | `__epi_8xi8` | `__epi_16xi8` | `__epi_32xi8` | `__epi_64xi8` | 

The syntax of vector types is `__epi_<factor>x<ty>`.

- `factor` is the relative number of elements of the vector respect to `VLEN/ELEN`. This way `__epi_2xf32` and `__epi_2xf64` have the same number of elements but different element type.
- `ty` is the element type. This way `__epi_2xf32` and `__epi_4xf32` have a different number of elements but the same element type.

## Mask types

Mask types are unrelated to `LMUL` in that they always use a single vector register. However the `<factor>` value is still useful. The element type of a mask is `i1`.

- `__epi_1xi1` 
- `__epi_2xi1`
- `__epi_4xi1` 
- `__epi_8xi1` 
- `__epi_16xi1` 
- `__epi_32xi1` 
- `__epi_64xi1` 

For example, a relational operation between two `__epi_2x<ty>` will compute a mask of type `__epi_2xi1`.

## Tuple types

Tuple types represent a pair of vectors. Currently tuples of `LMUL=1` is implemented.

| 2 elements | 3 elements | 4 elements | 5 elements | 6 elements | 7 elements | 8 elements |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| `__epi_1xf64x2` | `__epi_1xf64x3` | `__epi_1xf64x4` | `__epi_1xf64x5` | `__epi_1xf64x6` | `__epi_1xf64x7` | `__epi_1xf64x8` |
| `__epi_2xf32x2` | `__epi_2xf32x3` | `__epi_2xf32x4` | `__epi_2xf32x5` | `__epi_2xf32x6` | `__epi_2xf32x7` | `__epi_2xf32x8` |
| `__epi_1xi64x2` | `__epi_1xi64x3` | `__epi_1xi64x4` | `__epi_1xi64x5` | `__epi_1xi64x6` | `__epi_1xi64x7` | `__epi_1xi64x8` |
| `__epi_2xi32x2` | `__epi_2xi32x3` | `__epi_2xi32x4` | `__epi_2xi32x5` | `__epi_2xi32x6` | `__epi_2xi32x7` | `__epi_2xi32x8` |
| `__epi_4xi16x2` | `__epi_4xi16x3` | `__epi_4xi16x4` | `__epi_4xi16x5` | `__epi_4xi16x6` | `__epi_4xi16x7` | `__epi_4xi16x8` |
| `__epi_8xi8x2`  | `__epi_8xi8x3`  | `__epi_8xi8x4`  | `__epi_8xi8x5`  | `__epi_8xi8x6`  | `__epi_8xi8x7`  | `__epi_8xi8x8` |

Some EPI builtins return two vectors and use tuple types of 2 elements.

To access the elements of the tuple use the fields `v0`, `v1`, ... `v7`, depending on the number of elements of the tuple type.

```cpp
__epi_1xf64x2 mytuple;

... = mytuple.v0; // __epi_1xf64
... = mytuple.v1; // __epi_1xf64
```

## Mixed types

If your code is mixing widths (e.g. vectors of `float` and `double` at the same time) there are two possible approaches:

- Underusing the registers that hold narrower registers. For instance using `__epi_1xf64` and `__epi_2xf32` but using the latter as if only had half of the elements (as if it was the nonexisting type `__epi_1xf32`). This can be achieved using a granted vector length obtained with element width 64 (i.e. the wider element). This approach is complicated if we need to convert the lower elements of `__epi_2xf32` into an `__epi_1xf64` because of the `SLEN` parameter (which need not be `VLEN`).

- Grouping registers. For instance, using `__epi_2xf64` and `__epi_2xf32`. The former type must be operated under `LMUL=2` while the latter can be operated under `LMUL=1`. The granted vector length can be requested using the wider (with `__epi_m2`) or the narrower (with `__epi_m1`) type.

## Cache flags

Some loads and store instructions allow an extra `flags` operand. This operand needs not to be constant at runtime, but its value must be either `0` or `__epi_nt`.


| Flags | Meaning |
| ----- | ------ |
| `0` | Temporal operation: the load or store will allocate the loaded/stored data in case of cache miss. |
| `__epi_nt` | Non-temporal operation: the load or store will not allocate the loaded/stored data in case of cache miss. |

# Reference
