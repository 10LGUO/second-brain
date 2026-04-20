```yaml
title: Coalesced Memory Access
type: concept
tags: [gpu, cuda, memory-access, bandwidth, optimization, global-memory]
created: 2026-04-19
updated: 2026-04-19
sources: [kernel-dev.md]
```

# Coalesced Memory Access

Coalesced memory access is the pattern in which threads within the same warp access contiguous addresses in global memory, enabling the hardware to merge (coalesce) multiple thread memory requests into a single or small number of memory transactions. It is the primary technique for achieving high global memory bandwidth utilization.

## Mechanism

When a warp of 32 threads issues memory requests, the memory controller examines the requested addresses. If they fall within the same aligned cache line (128 bytes), all requests are served in a single transaction. If they are scattered, each request may require a separate transaction, multiplying bandwidth consumption and latency.

- **Coalesced (ideal):** 32 threads access 32 consecutive floats (128 bytes) → 1 transaction → 100% cache line utilization.
- **Non-coalesced (worst case):** 32 threads access 32 random addresses → up to 32 transactions → effectively 1/32 bandwidth utilization.

## Ideal Pattern

Consecutive `threadIdx.x` values should map to consecutively addressed memory:

```cuda
// Coalesced: thread i accesses element i
float val = data[blockIdx.x * blockDim.x + threadIdx.x];
```

## Common Anti-Pattern: Column Access in Row-Major Matrix

```cuda
// Non-coalesced: thread i accesses element (i, col) in row-major matrix
// Adjacent threads' addresses are separated by row_width elements
float val = matrix[threadIdx.x * row_width + col];
```

Adjacent threads have addresses `row_width * sizeof(float)` apart — potentially hundreds of bytes — causing severe coalescing failure.

**Solutions:**
- **Shared memory transposition:** Load data in a coalesced pattern into shared memory, then access in the needed (non-contiguous) pattern from shared memory.
- **Loop order adjustment:** Restructure computation to access data in a coalesced order.
- **Data layout change:** Store data in column-major order if column access is the dominant pattern.

## Relationship to L2 Cache

Coalesced access also improves L2 cache utilization: a loaded 128-byte cache line is fully consumed, so no bandwidth is wasted. Non-coalesced access loads cache lines from which only a few bytes are used, polluting the cache with unused data and evicting useful data.

## Related Concepts

- [[gpu-memory-hierarchy]]
- [[cuda-kernel-optimization]]
- [[shared-memory-tiling]]
- [[bank-conflict]]

## Sources

- [[kernel-dev]]

---
