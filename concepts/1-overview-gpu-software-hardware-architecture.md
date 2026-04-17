```yaml
title: GPU Software/Hardware Architecture
type: concept
tags: [gpu, hardware, software-stack, cuda, compilers, operators, ai-infra]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]

```

# GPU Software/Hardware Architecture

The GPU software/hardware architecture describes the layered system from physical chip components up through high-level deep learning frameworks. Understanding this full stack is essential for AI infra engineers performing operator development, performance optimization, and precision debugging.

## Hardware Components

Modern GPUs used in AI workloads (e.g., NVIDIA A100, H100) consist of several key physical components:

- **Streaming Multiprocessors (SMs):** The primary compute units on the GPU. Each SM contains multiple CUDA cores, Tensor Cores, registers, shared memory (SRAM), and warp schedulers. All threads in a thread block execute on a single SM. One SM can hold multiple blocks simultaneously; one block always stays on one SM.

```text
GPU
└── SM (×108 on A100)
    ├── Warp Schedulers (4 per SM) — each owns 32 CUDA cores, issues 1 warp/cycle
    ├── CUDA Cores (128 per SM = 4 schedulers × 32 cores)
    ├── Tensor Cores
    ├── Register File — partitioned per thread at launch
    └── Shared Memory SRAM pool — partitioned per block, internally 32 banks
        ├── Block 0's __shared__ variables (address range A)
        └── Block 1's __shared__ variables (address range B)

```

- **Tensor Cores:** Specialized matrix-multiply-accumulate (MMA) units within each SM, designed for mixed-precision matrix operations (e.g., FP16, BF16, INT8, FP8). Tensor Cores provide the bulk of FLOPS for deep learning workloads.
- **CUDA Cores:** General-purpose floating-point and integer execution units used for scalar and element-wise operations.
- **Registers:** Per-thread fast storage allocated at kernel launch time. Register pressure directly impacts occupancy.
- **Shared Memory (SRAM):** On-chip, low-latency memory shared within a thread block. Programmers use it explicitly in CUDA kernels to stage data and avoid redundant global memory accesses. Shared memory and L1 cache share the same physical SRAM and can be partitioned by the programmer.
- **L2 Cache:** A larger on-chip cache shared across all SMs. Acts as a staging buffer between the SMs and HBM.
- **High Bandwidth Memory (HBM):** The off-chip DRAM (e.g., HBM2e on A100, HBM3 on H100). Orders of magnitude larger than SRAM but with much higher latency and limited bandwidth relative to compute throughput. Most AI kernels are memory-bandwidth-bound when data cannot be kept on-chip.
- **NVLink / NVSwitch:** High-bandwidth interconnects for GPU-to-GPU communication within a node, enabling fast all-reduce and other collective operations.
- **PCIe:** Interface between the CPU (host) and the GPU (device), used for host-device memory transfers.

### Memory Hierarchy Summary

| Level | Location | Latency | Bandwidth | Capacity |
| --- | --- | --- | --- | --- |
| Registers | Per-thread, on-SM | ~1 cycle | Very high | ~256 KB/SM |
| Shared Memory (SRAM) | Per-block, on-SM | ~5–10 cycles | Very high | 48–228 KB/SM |
| L1 Cache | Per-SM | ~20–30 cycles | High | Shared with SRAM |
| L2 Cache | On-chip, shared | ~100–200 cycles | Moderate | 40–80 MB |
| HBM (Global Memory) | Off-chip | ~400–800 cycles | ~2–3 TB/s (H100) | 40–80 GB |
| Host DRAM (via PCIe) | Host system | Very high | ~50–100 GB/s | Hundreds of GBs |

Understanding this hierarchy is central to writing efficient kernels: the goal is almost always to maximize data reuse in registers and shared memory before falling back to HBM.

---

## Execution Model

GPUs execute code through a hierarchical threading model:

- **Grid:** The top-level software concept — the set of all thread blocks in one kernel launch. A grid is mapped onto the physical SM pool by the hardware scheduler. Grid and SM are **not** one-to-one: a typical grid has thousands of blocks; an A100 has 108 SMs. The scheduler distributes blocks across SMs as they become free, in any order.
- **Thread Block (CTA — Cooperative Thread Array):** A group of threads (up to 1024) assigned to one SM. One block runs on one SM, but one SM can hold multiple blocks simultaneously (constrained by shared memory and register budgets). Threads within a block share on-chip shared memory (SRAM) and can synchronize via `__syncthreads()`. Blocks cannot communicate with each other directly — they may run on different SMs or at different times.
- **Warp:** A **scheduling concept** — a group of 32 threads the warp scheduler dispatches together. Warps have no memory of their own (memory belongs to threads via registers, and blocks via shared memory). The warp scheduler is a physical hardware unit inside each SM. An SM has multiple schedulers (4 on A100), each owning 32 CUDA cores — so up to 4 warps execute truly simultaneously per SM cycle. Within an SM, warps **interleave** to hide latency (concurrency); true parallelism is across SMs. A thread borrows a CUDA core for the duration of each instruction — it does not permanently own one.
- **Lane:** The position of a thread within its warp (0–31). `laneId = threadIdx.x % warpSize`. Used for warp-level operations like shuffle instructions.
- **Thread:** The individual execution unit. Each thread owns its own registers and computes a unique index into the data. The thread ID in a 2D block maps to a flat linear index as: `linear = threadIdx.y * blockDim.x + threadIdx.x` (x varies fastest).

### Tile — Data Layout Concept

A **tile** is a rectangular subregion of a matrix (or tensor) that is loaded into shared memory for computation. It is a **data** concept, not an execution concept:

```text
Compute concepts          Storage concepts
─────────────────         ────────────────
Thread  (execution unit)  Tile  (data subregion in shared memory)
Warp    (32 threads)
Block   (warps on one SM)
Grid    (all blocks)
```

One block is typically responsible for computing one output tile and loads the corresponding input tiles into shared memory. But the two hierarchies are independent — a 32×32 block can process a 64×64 tile (each thread handles 4 elements), or a 16×16 tile (threads over-provisioned).

**Why tiling?** HBM is slow (~500 cycle latency). Loading a tile once into shared memory and reusing it many times for computation amortizes the HBM cost — this is **data reuse**. The tile size is chosen to balance:

- **Shared memory limit** — tile must fit in ~48KB per SM
- **Warp alignment** — 32-wide tiles align with warp size for natural coalescing
- **Occupancy** — smaller tiles → more blocks resident per SM → more warps → better latency hiding (constrained by both shared memory budget and hardware block limit per SM, whichever is lower)
- **Data reuse** — larger tiles → more reuse per HBM load → higher arithmetic intensity

32×32 float tiles (4KB) are the standard: `48KB / 4KB = 12 blocks` can be resident per SM, well within the hardware block limit (32 on A100).

### Thread Layout and Memory Coalescing

CUDA defines the linear thread index as:

```text
linear = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y

```

`threadIdx.x` has no multiplier — it is the innermost term and varies fastest. This is a deliberate NVIDIA convention to match **C's row-major array layout**, where the column index varies fastest in memory. Mapping `threadIdx.x → column` means the natural 2D kernel indexing is already coalesced by default:

```cuda
float val = input[threadIdx.y * N + threadIdx.x];
//                               col=threadIdx.x → consecutive addresses → coalesced ✓

```

If `threadIdx.y` varied fastest instead, the default 2D indexing would produce strided access — a bad default for most kernels.

A warp is 32 consecutive threads by linear index — meaning all threads in a warp share the same `threadIdx.y` and differ only in `threadIdx.x`. Memory transactions are issued at the **warp level**: when a warp executes a load/store, the hardware looks at all 32 addresses and merges them into as few 128-byte HBM transactions as possible (**coalescing**). Accesses via `threadIdx.x` (column) are coalesced; accesses via `threadIdx.y` (row) are strided and cause up to 32 separate transactions — 32× bandwidth waste.

### Shared Memory (SRAM) and Bank Conflicts

Shared memory is on-chip SRAM on the SM, ~100× lower latency than HBM. Each SM has one shared memory pool, physically partitioned into **32 banks**. Blocks assigned to that SM each carve out a portion of this pool — they don't get separate banks, they all use the same 32-bank physical memory at different address ranges.

```text
SM
├── Shared Memory SRAM (one pool, internally: 32 banks)
└── Blocks assigned to this SM
    ├── Block 0's __shared__ variables → address range A within that pool
    └── Block 1's __shared__ variables → address range B within that pool

```

Shared memory is scoped to a block — threads in a block see only their own address range; other blocks cannot access it. It must be allocated before kernel execution (statically via compile-time constants/templates, or dynamically via `extern __shared__` with size as the third launch argument) so the hardware can determine how many blocks fit on the SM simultaneously.

**Banks** are the internal physical structure of shared memory — 32 parallel memory modules that can each service one 4-byte access per cycle independently. Bank assignment cycles by word: word 0 → bank 0, word 1 → bank 1, ..., word 31 → bank 31, word 32 → bank 0 again. If 32 threads each hit a different bank, all 32 accesses complete in 1 cycle. If multiple threads hit the **same bank** (different addresses), they serialize — this is a **bank conflict**. Exception: if multiple threads read the **exact same address**, the hardware broadcasts it — no conflict.

Common case: reading a column of a 2D shared memory array. In a 32-wide array, column elements are 32 words apart → all map to the same bank → 32-way conflict. Fix: add 1 column of padding (`s_mem[N][M+1]`) to shift each row by one word, spreading column accesses across different banks.

### Occupancy

Occupancy is the ratio of active warps on an SM to the maximum possible warps. Higher occupancy gives the scheduler more warps to switch between when one stalls, hiding memory latency. Occupancy is limited by:

- Register usage per thread (register file is finite per SM)
- Shared memory usage per block (more shared memory → fewer blocks fit → fewer warps on SM)
- Thread block size

Achieving high occupancy is often necessary but not sufficient for peak performance; arithmetic intensity and memory access patterns matter equally.

### Warp Scheduling and Latency Hiding

SMs are equipped with warp schedulers (2–4 per SM depending on architecture) that can issue instructions from different warps each cycle. When one warp stalls on a memory load, the scheduler switches to another ready warp. This is the primary mechanism by which GPUs hide the latency of HBM accesses — requiring sufficient in-flight warps (high occupancy) to keep the execution units busy.

### Thread Block Clusters (Hopper / H100)

The H100 architecture introduced **Thread Block Clusters** (via the SM90 target), a new level in the hierarchy between thread blocks and the grid. A cluster groups multiple thread blocks that reside on neighboring SMs and can communicate via **distributed shared memory** (DSMEM) — direct SM-to-SM SRAM access without going through HBM or L2. This is useful for operations like reductions that previously required global memory round-trips.

---

## Software Stack Layers

The GPU software stack can be understood as a series of abstraction layers:

### 1. Deep Learning Frameworks

**Examples:** PyTorch, JAX, TensorFlow

The entry point for most ML practitioners. Frameworks provide:

- Automatic differentiation (autograd)
- High-level operator APIs (`torch.matmul`, `torch.nn.Conv2d`, etc.)
- Device management and memory allocation abstractions
- Execution backends that dispatch to lower-level libraries or compiled kernels

PyTorch's dispatcher routes operator calls to the appropriate backend (CUDA, CPU, XLA, etc.) and handles autograd graph construction.

### 2. Operator Libraries

**Examples:** cuBLAS, cuDNN, CUTLASS, FlashAttention, cuSPARSE

Pre-written, highly optimized kernel libraries for common operations:

- **cuBLAS:** BLAS-level matrix operations (GEMM, TRSM, etc.), the foundation of most dense linear algebra in deep learning.
- **cuDNN:** Higher-level primitives for convolutions, attention, normalization, and RNN operations. Used heavily by frameworks.
- **CUTLASS:** NVIDIA's open-source C++ template library for GEMM and related operations, offering more flexibility than cuBLAS for custom tiling and precision configurations. CUTLASS 3.x introduces a unified CuTe layout abstraction that generalizes across SM architectures.
- **FlashAttention:** A hand-written CUDA kernel (and later Triton/CUTLASS variants) for memory-efficient scaled dot-product attention, a canonical example of kernel fusion and tiling to overcome HBM bandwidth limits.
- **cuSPARSE:** Sparse matrix operations used in pruned model inference and certain graph neural network workloads.
- **cuFFT:** GPU-accelerated Fast Fourier Transforms, used in certain convolution implementations and signal-processing workloads.

### 3. Kernel Fusion and Compilation

**Examples:** torch.compile, XLA/HLO, TensorRT, Triton

Rather than launching separate kernels for each operator, fusion combines multiple operations into a single kernel to reduce HBM round-trips:

- **torch.compile (with TorchInductor):** Uses FX graph capture and calls Triton (for GPU) or C++ codegen to generate fused kernels automatically. Introduced in PyTorch 2.0 as the primary path for eager-mode graph optimization.
- **XLA (Accelerated Linear Algebra):** JAX and TensorFlow's compiler. Operates on an HLO (High-Level Operations) IR, performs fusion and layout optimization, and emits PTX via LLVM.
- **TensorRT:** NVIDIA's inference optimizer. Accepts ONNX models, applies layer fusion, quantization, and kernel auto-tuning for deployment.
- **Triton:** A Python-embedded DSL for writing GPU kernels at a tile/block level, abstracting away warp-level details while still allowing close-to-peak performance. Widely used in custom operator development and as the backend for TorchInductor.
- **torch.export + AOTInductor:** A newer PyTorch path that exports a static computation graph for ahead-of-time compilation, producing a standalone `.so` artifact suitable for C++ deployment without the Python runtime.

### 4. CUDA / Low-Level GPU Programming

**Examples:** CUDA C++, PTX, SASS

The layer at which custom kernels are written directly:

- **CUDA C++:** NVIDIA's extension of C++ for GPU programming. Programmers specify grid/block dimensions, manage shared memory, and write device functions.
- **PTX (Parallel Thread Execution):** A virtual ISA that CUDA C++ compiles to. PTX is then compiled to SASS by the device-specific assembler (`ptxas`). Developers sometimes write or inspect PTX for fine-grained optimization.
- **SASS (Shader ASSembly):** The actual machine code running on the GPU. Tools like `nvdisasm` can disassemble SASS for low-level profiling and verification.
- **NVRTC / cuBIN:** Runtime compilation of CUDA kernels and binary formats for deployment.
- **CuTe (CUTLASS Tensor abstraction):** A composable layout and tiling library within CUTLASS that provides a unified vocabulary for describing tensor shapes, strides, and tile hierarchies across all SM generations. Increasingly used as a standalone primitive even outside full CUTLASS pipelines.

### 5. Driver and Runtime

**Examples:** CUDA Runtime API, CUDA Driver API, NCCL

Below the kernel level:

- **CUDA Runtime API:** High-level C API (`cudaMalloc`, `cudaMemcpy`, `cudaLaunchKernel`) used by most CUDA code.
- **CUDA Driver API:** Lower-level API that provides more explicit control over contexts, modules, and kernel loading. Allows loading PTX or cuBIN at runtime, useful for JIT compilation workflows.
- **NCCL (NVIDIA Collective Communications Library):** Provides collective operations (all-reduce, broadcast, all-gather) optimized for multi-GPU and multi-node topologies, used by distributed training frameworks.
- **CUDA Graphs:** A mechanism to record a sequence of GPU operations (kernel launches, memory copies) as a graph and replay it with minimal CPU overhead, reducing launch latency for repetitive workloads like inference serving.
- **Unified Memory (UM):** A CUDA feature allowing a single pointer to be accessed from both CPU and GPU, with the driver managing page migration. Convenient but can introduce unpredictable performance penalties if access patterns are not carefully managed.

---

## Operator Development

Operator development refers to the process of writing or optimizing a GPU kernel to implement a specific mathematical operation (e.g., attention, convolution, normalization, custom activation).

### Key Concepts

- **Tiling:** Decomposing large matrix/tensor operations into smaller tiles that fit in shared memory or registers, enabling data reuse and reducing HBM bandwidth.
- **Kernel Fusion:** Merging element-wise or reduction operations that follow a major compute kernel (e.g., fusing softmax into an attention kernel) to avoid intermediate HBM writes/reads.
- **Memory Coalescing:** Arranging memory accesses so that threads in a warp access contiguous addresses, enabling full utilization of HBM bandwidth (128-byte transaction granularity on modern GPUs).
- **Bank Conflicts:** Shared memory is divided into 32 banks. When multiple threads in a warp access the same bank simultaneously (at non-broadcast addresses), accesses serialize. Padding shared memory arrays is a common mitigation.
- **Occupancy vs. ILP Trade-off:** Using more registers per thread reduces occupancy but can increase instruction-level parallelism (ILP) within a warp, sometimes yielding better net throughput.
- **Async Copies (cp.async):** On Ampere and later, `cp.async` instructions allow data to be copied from global memory to shared memory without occupying registers or stalling the warp, enabling software pipelining (double- or triple-buffering of shared memory stages).
- **Tensor Core Usage:** Efficient use of Tensor Cores requires data to be in specific layouts (e.g., row-major A, column-major B for HMMA instructions) and accessed via `wmma` or `mma.sync` PTX instructions, or via CUTLASS abstractions.
- **Warp Divergence Avoidance:** Control flow that causes threads within a warp to take different branches results in both paths being executed serially with masking. Minimizing or eliminating divergent branches is important for kernels with conditional logic.
- **Software Pipelining:** Overlapping computation with data movement using double- or triple-buffered shared memory stages. The Ampere `ldmatrix` / `cp.async` and Hopper `TMA` (Tensor Memory Accelerator) instructions are key hardware primitives for this.
- **Persistent Kernels:** Instead of launching a new kernel for each tile, a persistent kernel occupies SMs for the lifetime of an operation and self-schedules work from a queue. This can reduce kernel launch overhead and improve SM utilization, but complicates code.

### Hopper-Specific Features

The H100 (SM90) architecture introduced several features relevant to operator developers:

- **Tensor Memory Accelerator (TMA):** A hardware unit that asynchronously copies multi-dimensional tiles between HBM and shared memory, decoupling data movement from compute and enabling warpgroup-level pipelining.
- **Warpgroup MMA (WGMMA):** New asynchronous MMA instructions that operate at the warpgroup (4-warp, 128-thread) granularity, providing higher throughput than the per-warp `mma.sync` instructions of prior architectures.
- **FP8 Tensor Cores:** Support for E4M3 and E5M2 FP8 formats in WGMMA, enabling ~2× higher FLOPS compared to BF16 Tensor Cores.

### Development Workflow

1. **Prototype in Triton or PyTorch:** Validate numerical correctness at a high level before investing in CUDA-level work.
2. **Profile baseline:** Use `nsys` (Nsight Systems) and `ncu` (Nsight Compute) to identify bottlenecks — is the kernel compute-bound or memory-bound? Where are warps stalling?
3. **Implement CUDA kernel:** Write the tiled, fused kernel in CUDA C++ or Triton, applying the appropriate tiling, fusion, and memory access strategies.
4. **Benchmark and compare:** Measure achieved FLOPS and memory bandwidth against theoretical roofline limits.
5. **Iterate on optimization:** Adjust tile sizes, shared memory layout, pipeline depth, warp count, and occupancy targets based on profiler feedback.
6. **Numerical validation:** Compare outputs against a reference (e.g., FP64 baseline) across a range of inputs, paying special attention to edge cases (all-zeros, large values, NaN propagation).
7. **Integrate with framework:** Register as a custom operator via PyTorch's `torch.library` API or use `torch.utils.cpp_extension.load_inline` for development-time loading.

---

## Precision and Numerical Formats

AI workloads operate across a range of numeric precisions, each with different hardware support and numerical properties:

| Format | Bits | Exponent | Mantissa | Notes |
| --- | --- | --- | --- | --- |
| FP64 | 64 | 11 | 52 | Double precision; used for reference implementations and scientific computing |
| FP32 | 32 | 8 | 23 | Default training precision; accumulator type in mixed-precision |
| TF32 | 19 | 8 | 10 | NVIDIA-internal Tensor Core format (A100+); same range as FP32, lower precision |
| BF16 | 16 | 8 | 7 | Same range as FP32, less precision than FP16; preferred for training stability |
| FP16 | 16 | 5 | 10 | Narrow dynamic range; requires loss scaling to avoid underflow in training |
| FP8 E4M3 | 8 | 4 | 3 | H100+; preferred for forward pass activations and weights |
| FP8 E5M2 | 8 | 5 | 2 | H100+; preferred for gradients due to wider range |
| INT8 | 8 | — | — | Inference only; requires quantization-aware training or post-training quantization |
