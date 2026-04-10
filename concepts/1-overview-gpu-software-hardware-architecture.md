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

- **Streaming Multiprocessors (SMs):** The primary compute units on the GPU. Each SM contains multiple CUDA cores, Tensor Cores, registers, shared memory (SRAM), and warp schedulers. All threads in a thread block execute on a single SM.
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

- **Grid:** The top-level collection of all thread blocks launched by a kernel invocation.
- **Thread Block (CTA — Cooperative Thread Array):** A group of threads (up to 1024) that are co-scheduled on the same SM and can communicate via shared memory and synchronize via `__syncthreads()`.
- **Warp:** The fundamental unit of GPU execution. A group of 32 threads that execute in lockstep (SIMT — Single Instruction Multiple Threads). Divergent control flow within a warp causes serialization.
- **Thread:** The individual execution unit. Each thread has its own registers and a unique thread ID used to index data.

### Occupancy

Occupancy is the ratio of active warps on an SM to the maximum possible warps. Higher occupancy helps hide memory latency through warp switching. Occupancy is limited by:

- Register usage per thread (register file is finite per SM)
- Shared memory usage per block
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
