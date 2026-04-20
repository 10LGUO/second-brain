```yaml
title: Shared Memory Tiling
type: concept
tags: [gpu, cuda, shared-memory, tiling, optimization, memory-access]
created: 2026-04-19
updated: 2026-04-19
sources: [kernel-dev.md]
```

# Shared Memory Tiling

Shared Memory Tiling is the technique of loading a contiguous block ("tile") of data from global memory (HBM) into on-chip shared memory so that multiple threads within a block can reuse it, dramatically reducing the number of global memory accesses.

## Motivation

Global memory (HBM) bandwidth is the primary bottleneck in many GPU kernels. Data-reuse-heavy operations like matrix multiplication (GEMM) and convolution access the same data elements multiple times. Without tiling, each access goes to HBM; with tiling, data is loaded once per tile and reused from fast on-chip shared memory.

## Working Principle

1. Partition the input data into tiles that fit in shared memory.
2. Each thread block loads its assigned tile from global memory into a `__shared__` array.
3. Synchronize with `__syncthreads()` to ensure all threads see the complete tile.
4. Compute using data from shared memory (fast on-chip access).
5. Repeat for the next tile.

This reduces HBM traffic by a factor proportional to the tile's reuse factor.

## Benefits

- Reduces global memory access → improves bandwidth utilization.
- Increases L2 hit rate (data loaded once per tile, reused many times).
- Enables overlap with [[ping-pong-buffer]] for further latency hiding.
- Also used for layout reorganization (e.g., transpose in shared memory) to fix coalescing or bank conflict issues.

## Constraints

- Shared memory is scarce: only **tens to ~100 KB per SM**. Tile size must be chosen to fit, while also allowing sufficient blocks to occupy the SM for good **occupancy**.
- For GEMM, tile size is typically a template parameter; **autotuning** finds the optimal value.
- Must use `__syncthreads()` correctly: after writes to shared memory, before reads that depend on those writes.
- Must design access patterns carefully to avoid **bank conflicts**.

## Key Example: Matrix Transpose

- **Naïve:** Each thread reads one element from global memory (contiguous rows), writes to transposed position (column → non-contiguous) → poor coalescing on writes.
- **Tiled:** Load a tile into shared memory using row-contiguous reads (coalesced). Read back from shared memory in transposed order (fast, on-chip). Write to global memory contiguously (coalesced). Result: both global memory reads and writes are coalesced.

## Related Concepts

- [[gpu-memory-hierarchy]]
- [[cuda-kernel-optimization]]
- [[bank-conflict]]
- [[ping-pong-buffer]]
- [[coalesced-memory-access]]
- [[autotuning]]
- [[gemm]]

## Sources

- [[kernel-dev]]

---
