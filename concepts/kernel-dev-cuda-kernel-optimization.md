```yaml
title: CUDA Kernel Optimization Techniques
type: concept
tags: [gpu, cuda, optimization, performance, hpc, operator-development]
created: 2026-04-19
updated: 2026-04-19
sources: [kernel-dev.md]
```

# CUDA Kernel Optimization Techniques

CUDA kernel optimization is the systematic application of hardware-aware techniques to maximize either **compute utilization** or **bandwidth utilization** (or both) in GPU kernels. The guiding principle is:

> **Operator = Memory Access + Computation**. All optimizations target one or both of these.

These nine techniques must be fully mastered by GPU operator developers and are frequently tested in technical interviews:

1. Shared Memory
2. Warp Reduce
3. Block and Thread partitioning
4. Coalesced memory access
5. Bank Conflict elimination
6. Ping-Pong Buffer
7. Branch reduction
8. Loop unrolling
9. Space-time tradeoffs

## Optimization Goals

- **Two targets:** Compute utilization rate and bandwidth utilization rate.
- **Decision process:**
  1. Compute the operator's **arithmetic intensity** (flops / bytes).
  2. Compare against the hardware's **balance point** ([[roofline-model]]).
  3. Determine if the operator is **bandwidth-bound** or **compute-bound**.
  4. Optimize the bottleneck → target: **≥ 80% utilization**.
- Theoretical calculation of these metrics is required knowledge — profiling tools ([[nsight]]) must also be mastered.

## Block and Thread Partitioning

- **Mapping data to threads:** Either 1 thread per element (straightforward) or 1 thread per multiple elements (when N is very large or per-thread work is too small).
- **Block size:**
  - Must be a **multiple of 32** (warp size).
  - Common sizes: 32, 64, 128, 256, 512, 1024 (CUDA maximum).
  - Shared memory consumption per block limits how many blocks fit on an SM simultaneously.
  - Register consumption per thread limits concurrent thread count on SM.
  - **Occupancy** = active warps / maximum supported warps. Higher is generally better for latency hiding, but not always (ILP matters too).
  - High arithmetic intensity → can tolerate lower occupancy. Low arithmetic intensity → needs high occupancy to hide memory latency.
- **Grid size:**
  - `Grid Size = ceil(N / B)`; check bounds in kernel if N % B ≠ 0.
  - Must be large enough to keep all SMs busy.
  - Hardware grid limit (e.g., 65535³) is rarely reached.
  - For 2D/3D data: use 2D/3D block/grid layouts for natural mapping and better coalescing.
- No universal optimum; must tune experimentally. **Autotuning** is commonly used for GEMM templates.

## Warp Reduce

- Performs a reduction (sum, max, min, etc.) entirely within a single warp (32 threads).
- Uses **Warp Shuffle** intrinsics (`__shfl_xor_sync`, `__shfl_down_sync`) for direct register-to-register data exchange — no shared memory allocation, no `__syncthreads()`.
- **Butterfly algorithm:** Pairwise exchange and accumulate over log₂(32) = 5 steps → result on lane 0.
- More efficient than shared-memory-based reduce: no bank conflicts, no synchronization overhead.
- Serves as the foundational building block for larger block-level and grid-level reductions.

## Coalesced Memory Access

- When warp threads access contiguous global memory addresses, hardware merges transactions into one or a few → high bandwidth utilization.
- Scattered or strided access → many transactions → very low bandwidth utilization.
- Ideal: consecutive `threadIdx.x` values map to consecutive memory addresses.
- Avoid accessing columns of row-major matrices directly. Solutions: shared memory transposition, loop order adjustment.

## Bank Conflict

- Shared memory is organized into banks (typically 32). Multiple threads hitting **different addresses in the same bank** → accesses serialized (N-way conflict).
- Exception: all threads hitting **the same address** in a bank → broadcast, no conflict.
- Common cause: strided access where stride shares a common factor with 32 (e.g., stride=2 → 16-way conflict).
- Solutions: address offset, **padding** (add an extra element per row to shift bank assignments).

## Ping-Pong Buffer (Double Buffering)

- Splits shared memory into two buffers (A and B) to overlap computation with data prefetch from global memory.
- While computing on buffer A (iteration i), asynchronously load data for iteration i+1 into buffer B. Swap and repeat.
- Hides global memory latency → improves SM utilization.
- Buffer sizes must be tuned so compute time and transfer time overlap effectively.

## Shared Memory as Tiling Cache

- Load a tile of global memory data into shared memory; all threads in the block reuse it from there.
- Dramatically reduces global memory (HBM) accesses, especially for data-reuse-heavy patterns like convolution and GEMM.
- Also used to reorganize data layout (e.g., transposition) to fix coalescing problems or bank conflicts.
- Synchronize with `__syncthreads()` after writes and before reads.

## Inter-Technique Relationships

These techniques are not independent and must be jointly considered:

- Block/thread partitioning determines parallel patterns and coalescing efficiency.
- Use **Shared Memory** as a tile cache to reduce global memory traffic.
- Carefully design shared memory layouts to **avoid Bank Conflict**.
- Use **Ping-Pong Buffer** to hide tile load latency.
- Use **Warp Reduce** for the most efficient intra-warp aggregation.

## Related Concepts

- [[gpu-memory-hierarchy]]
- [[shared-memory-tiling]]
- [[warp-reduce]]
- [[coalesced-memory-access]]
- [[bank-conflict]]
- [[ping-pong-buffer]]
- [[register-spill]]
- [[arithmetic-intensity]]
- [[roofline-model]]
- [[autotuning]]
- [[gemm]]

## Sources

- [[kernel-dev]]

---
