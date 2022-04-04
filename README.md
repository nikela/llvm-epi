# EPI Project - LLVM-based compiler

This is a publicly accessible version of the [EPI project](https://www.european-processor-initiative.eu/) LLVM-based compiler developed at the Barcelona Supercomputing Center.

The main goal of this development is to foster a co-design cycle in the development of the [RISC-V Vector Extension (RVV)](https://github.com/riscv/riscv-v-spec).

## Intrinsics

This compiler implements a set of EPI-specific intrinsics of relatively low-level nature for initial exploration of the RVV extension. Reference is found at  https://repo.hca.bsc.es/gitlab/rferrer/epi-builtins-ref .

**Note**: these intrinsics are not the ones that proposed for RISC-V Vector Extension (find those at https://github.com/riscv/rvv-intrinsic-doc). It is planned to replace the EPI-specific intrinsics ones with those from the proposal in the future.

### Limitations

- Not all the instructions of RVV are currently accessible using the EPI-intrinsics and there is a bit of HPC bias to them.

## Vectorization

At the same time we are interested in pushing forward LLVM's vectorization capabilities in order to enable (semi) automatic vectorization for RVV. We are extending the `LoopVectorizer`.

LLVM has good support for SIMD-style vectorization. But currently falls short in making the most of existing features such as predication/masking (RVV, Arm SVE, Intel AVX-512, NEC SX-Aurora), vector length-agnostic vectorization (RVV and Arm SVE) and vector length-based strip-mining (RVV, NEC SX-Aurora).

We extended [Simon Moll's LLVM Vector Predication](http://www.llvm.org/docs/Proposals/VectorPredication.html) proposal which has proved useful for vectorization in RVV.

### Main approach

We use vector-length agnostic vectorization (similar to what SVE will do).

There are three main ways to vectorize a loop in RVV

1. Strip-mine to the whole register size followed by a scalar epilog.
   - This is what the `LoopVectorizer` can do now so it is relatively straightforward and does not need special LLVM IR
   - The downside is that it gives up many features of RVV.
2. Fold the epilog loop into the vector body.
   - This is done by setting the vector length in each iteration. This induces a predicate/mask over all the vector instructions of the loop (any other predicates/masks in the vector body are needed for control flow).
   - Needs special IR, here we use [LLVM Vector Predication intrinsics](http://www.llvm.org/docs/Proposals/VectorPredication.html).
   - We extended the [Vectorization Plan](http://www.llvm.org/docs/Proposals/VectorizationPlan.html) with new recipes that are vector-length aware
3. An intermediate approach in which the whole register size is used in the vector body followed by a vectorized (using set vector length) is possible.
   - We have not implemented this approach

(From the point of view of our research the approach 2 above is the most interesting to us. This is not necessarily what upstream or other RVV vendors may actually prefer).

### Limitations

- This is in very active development, so things may break or be incomplete.
- Our current cost model is a bit naive because RVV brings some challenges to the current way, see below.
- Scalable (aka vector-length agnostic) vectorization is work in progress in the LLVM community in special to enable Arm SVE.
- We tried to make the loop vectorizer as target independent as possible. However there is a single EPI intrinsic used to set the vector length (which could be easily moved as a hook to the backend). Also there may be some underlying bias towards what works best in RVV.

### Cost model

We're in the process of improving our cost model. RVV comes with some interesting differences respect to traditional SIMD which are currently embedded in LLVM's Cost Model.

- Most ISAs have the same number of registers regardless of their length. For instance there is the same number of registers in SSE, AVX-2 and AVX-512 even if they have different lengths.
  - An interesting case Arm Advanced SIMD (NEON) where there are 32 registers of 64-bit and 16 registers of 128 bit, however the compiler in the vectorizer chooses to not to model the 64-bit registers at all (after all the 128-bit ones are longer and targeting them by default is the sensible thing to do)
- RVV has 32 vector registers that can be grouped in vector groups of size 2, 4 or 8. There are 16 groups of size 2, 8 groups of size 4 and 4 groups of size 8. Vector groups can be operated as longer vectors (2x, 4x or 8x the length of a vector register). In fact it makes sense to expose them in LLVM as "super" registers. However this means that now we don't have the same number of "registers" depending on the length
  - A loop that uses very few registers (think of a SAXPY) may use very few vector registers (say 3) so a using vector groups of size 8 is ideal if we want the largest vectorization factor (= the ratio of the number of iterations of the vector loop and the number of iterations of the scalar loop) along with the smallest code size (no different instructions are needed to operate in the different groups).
  - A loop that uses many more registers may benefit from using smaller vector groups so we can control the register pressure to the vector groups/registers.
  - If we were to use the existing strategy of `LoopVectorizer`, we would report to the vectorizer that we have only 4 registers (groups of 8 are the longest ones) which doesn't seem obvious to us how it could enable the scenario shown above.

Currently the policy for RVV is kind of hard-coded in the vectorizer and remains future work to be able to integrate it better with the existing one.

# Acknowledgements

This work has been done as part of the European Processor Initiative projects EPI-SGA1 and EPI-SGA2.

The European Processor Initiative (EPI-SGA1) (FPA: 800928) has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement EPI-SGA1: 826647.

The EPI-SGA2 project has received funding from the European High Performance Computing Joint Undertaking (JU) under Framework Partnership Agreement No 800928 and Specific Grant Agreement No 101036168 EPI-SGA2. The JU receives support from the European Union’s Horizon 2020 research and innovation programme and from Croatia, France, Germany, Greece, Italy, Netherlands, Portugal, Spain, Sweden, and Switzerland.
