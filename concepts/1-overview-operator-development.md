---
title: Operator Development
type: concept
tags: [operator, gpu, kernel, ai-infra, performance-optimization, cuda]
created: 2026-04-05
updated: 2026-04-06
sources: [1-overview.md]
---

# Operator Development

An **operator** (算子) is the smallest execution unit on a compute chip. Operators implement the parallel compute primitives of a chip and are the fundamental building blocks through which higher-level frameworks (PyTorch, TensorFlow, Megatron, etc.) express computation. Understanding and developing operators is essential for AI infra engineers performing performance optimization and operator fusion.

## Definition

- An operator maps directly to a **kernel** — a piece of code that runs on the chip's on-chip cache and compute cores, launched via `launch kernel` from host (CPU) code.
- Operators express the chip's parallel compute capability: a single operation can process large quantities of data in the same time as processing a single data point.

## Why Operators Matter for Infra

- **Operator fusion:** Combining multiple operators into a single kernel to eliminate intermediate memory reads/writes (a primary performance optimization). Requires understanding both the algorithm layer and the operator layer simultaneously.
- **Performance optimization:** Without operator knowledge, performance optimization is guesswork. The operator layer is the boundary between software and hardware execution.
- **Precision debugging:** Precision errors in chip computation are often localized to specific operator implementations. Finding these requires deep operator knowledge.
- **Parallel acceleration:** Operators define how compute parallelism is structured on the chip.

## Structure of an Operator

Operator code has two parts:

- **Host code:** Executed by the CPU; sets up parameters and issues `launch kernel` calls.
- **Kernel code:** Loaded onto on-chip cache (L1/L2) and executed on compute cores (CUDA Core, Tensor Core). Must explicitly manage registers, L1 cache, and L2 cache.

## Relationship to Compilers

- Operator development is strongly related to compiler development.
- Understanding the compiler makes the entire software stack transparent.
- For domestic (non-NVIDIA) chip developers, compilers are especially critical because they significantly affect operator behavior and because the programming model (often [[simd]] vs. [[simt]]) differs from NVIDIA's.

## Key Tools and Abstractions

- **CUDA C++ / Triton:** Primary languages for writing GPU operators
- **PTX / SASS:** Intermediate and native instruction sets that compiled kernels produce
- **cuBLAS, cuDNN:** Pre-optimized library operators for common operations (GEMM, convolution)
- **Inductor / Triton:** PyTorch's `torch.compile` path generates Triton kernels as operators
- **Nsight Compute:** Primary profiling tool for analyzing kernel performance (register usage, memory throughput)

## Skill Level in the Infra Stack

From the full-stack infra competency table:

- Operator development is Layer 5 (of 6, from top).
- Engineers need not specialize in operator development but **must understand it**; otherwise, lower layers (compiler, hardware) are black boxes and performance optimization becomes guesswork.
- True mastery through the operator layer is considered the threshold for "highly skilled" infra engineers.

## Related Concepts

- [[gpu-software-hardware-architecture]]
- [[large-model-infra]]
- [[compute-communication-overlap]]
- [[precision-convergence]]

## Sources

- [[1-overview]]

---
