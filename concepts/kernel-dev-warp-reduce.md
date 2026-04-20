```yaml
title: Warp Reduce
type: concept
tags: [gpu, cuda, warp, reduction, optimization, warp-shuffle]
created: 2026-04-19
updated: 2026-04-19
sources: [kernel-dev.md]
```

# Warp Reduce

Warp Reduce is a reduction operation (sum, max, min, etc.) performed entirely within a single warp of 32 threads, using direct register-to-register data exchange via **Warp Shuffle** intrinsics rather than shared memory.

## Motivation

A naïve block-level reduction uses shared memory: allocate an array, write values, synchronize, then recursively reduce. This requires:
- Shared memory allocation (consumes scarce on-chip resource).
- `__syncthreads()` calls (synchronization overhead).
- Careful bank conflict avoidance.

Warp Reduce bypasses all of this by exploiting the fact that threads within a warp are **naturally synchronized** — they execute in lockstep — and can exchange register values directly via hardware-supported shuffle instructions.

## Mechanism

**Warp Shuffle intrinsics** (CUDA built-ins):
- `__shfl_down_sync(mask, val, delta)` — thread reads the value of the thread `delta` lanes below it.
- `__shfl_xor_sync(mask, val, laneMask)` — thread exchanges with the thread whose lane ID is XOR'd with `laneMask`.

**Butterfly algorithm:**
- Threads exchange and accumulate pairwise.
- After **log₂(32) = 5 steps**, the accumulated result is present on lane 0.
- No `__syncthreads()` needed at any step.

```
Step 1: Each thread exchanges with thread ±16 → partial sums
Step 2: Exchange with ±8 → partial sums
Step 3: Exchange with ±4
Step 4: Exchange with ±2
Step 5: Exchange with ±1
→ Lane 0 holds the full warp reduction result
```

## Advantages Over Shared Memory Reduce

| Property | Warp Reduce | Shared Memory Reduce |
|---|---|---|
| Shared memory used | No | Yes |
| Synchronization needed | No | Yes (`__syncthreads()`) |
| Bank conflict risk | None | Possible |
| Speed | Faster | Slower |

## Role in Larger Reductions

Warp Reduce is the **foundational building block** for block-level and grid-level reductions:
- First reduce within each warp (Warp Reduce → one result per warp).
- Write per-warp results to shared memory.
- Reduce across warps (another Warp Reduce or tree reduction on a smaller array).

## Related Concepts

- [[cuda-kernel-optimization]]
- [[gpu-memory-hierarchy]]
- [[bank-conflict]]
- [[shared-memory-tiling]]

## Sources

- [[kernel-dev]]

---
