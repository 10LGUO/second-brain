```yaml
title: Ping-Pong Buffer (Double Buffering)
type: concept
tags: [gpu, cuda, double-buffering, latency-hiding, shared-memory, optimization]
created: 2026-04-19
updated: 2026-04-19
sources: [kernel-dev.md]
```

# Ping-Pong Buffer (Double Buffering)

The Ping-Pong Buffer (also called double buffering) is a technique for hiding global memory latency in iterative GPU kernels by overlapping computation on one buffer with asynchronous data prefetch into a second buffer, so that data is ready in shared memory before it is needed.

## Motivation

In tiled kernels (GEMM, convolution), each iteration loads a tile from global memory into shared memory before computing. This load has high latency (HBM access). Without double buffering, compute stalls waiting for data. With double buffering, compute on the current tile and prefetch of the next tile proceed simultaneously.

## Working Principle

1. Allocate **two shared memory buffers**: Buffer A and Buffer B (total shared memory usage doubles).
2. **Prologue:** Load first tile into Buffer A.
3. **Main loop:**
   - Iteration i: Compute using Buffer A; asynchronously load tile i+1 into Buffer B.
   - Iteration i+1: Compute using Buffer B; asynchronously load tile i+2 into Buffer A.
   - Buffers alternate ("ping-pong") each iteration.
4. **Epilogue:** Drain the last computation.

## Requirement: Asynchronous Memory Operations

Effective double buffering requires that the prefetch and compute truly overlap. In CUDA, this is achieved using:
- `cp.async` instructions (CUDA 11+, Ampere+) for direct async global→shared memory copies.
- Pipelining with `__pipeline_memcpy_async()` in cooperative groups.

Without async capability, the overlap may be limited.

## Tuning Considerations

- **Buffer size vs. overlap:** Buffer sizes (tile sizes) must be tuned so that compute time and data transfer time are comparable. If compute is much faster than transfer, the next tile won't be ready in time. If transfer is much faster than compute, there is no benefit.
- **Shared memory pressure:** Two buffers consume 2× the shared memory of a single-buffered approach, reducing the number of blocks that can co-reside on an SM. Must balance latency hiding against occupancy.

## Related Concepts

- [[shared-memory-tiling]]
- [[gpu-memory-hierarchy]]
- [[cuda-kernel-optimization]]
- [[autotuning]]
- [[gemm]]

## Sources

- [[kernel-dev]]

---
