```yaml
title: Warp
type: concept
tags: [gpu, cuda, warp, simt, parallelism, kernel-development, memory-coalescing]
created: 2026-04-07
updated: 2026-04-07
sources: [5-kernel-dev.md]
```

# Warp

A **warp** is the fundamental hardware execution unit on NVIDIA GPUs. It consists of 32 threads that are always scheduled and executed together — when the GPU issues an instruction, it issues it to all 32 threads in a warp simultaneously. This is the hardware implementation of the [[simt-programming-model]].

## Key Properties

- **Fixed size:** Always 32 threads on all NVIDIA GPU architectures.
- **Single program counter:** All 32 threads in a warp execute the same instruction at the same time.
- **Independent registers:** Each thread has its own registers and can operate on different data.
- **Scheduler unit:** The GPU warp scheduler picks warps to issue instructions each clock cycle; a Streaming Multiprocessor (SM) typically runs many warps concurrently to hide latency.

## Relationship to SIMT

Single Instruction, Multiple Threads (SIMT) is the *programming model* — you write scalar code for one thread. A warp is the *hardware mechanism* by which SIMT is implemented: the GPU groups 32 threads and executes one instruction across all of them in parallel. The programmer does not explicitly manage warps; they emerge from how the runtime maps thread blocks onto hardware.

## Memory Coalescing

The warp size directly determines memory access efficiency. When 32 threads in a warp each access memory, the hardware tries to merge (coalesce) those accesses into as few 128-byte transactions as possible:

| Access pattern | Transactions | Effective bandwidth |
| --- | --- | --- |
| 32 threads × `float` (4B), contiguous | 1 × 128B transaction | Full |
| 32 threads × `float4` (16B), contiguous | 4 × 128B transactions | Full (4× data per instruction) |
| 32 threads, strided or random | Up to 32 transactions | Severely degraded |

Vectorized memory access (`float4`, `float2`) increases bytes moved per instruction without increasing transaction count, directly improving bandwidth utilization toward the hardware ceiling.

### Why not `float8` or `float16`?

**128 bits (16 bytes) is the maximum width of a single memory load/store instruction** (`LDG.128`) on NVIDIA GPUs. `float4` (4 × 32 bits) fills it exactly — the hardware ceiling. Going wider gives nothing:

| Type | Bits | Single instruction? | Notes |
| --- | --- | --- | --- |
| `float` | 32 | Yes — underutilizes | |
| `float2` | 64 | Yes — underutilizes | |
| `float4` | 128 | Yes — maximum | |
| `float8` | 256 | No — 2 instructions | No native CUDA type; same as 2×`float4` |
| `float16` | 512 | No — 4 instructions | No native CUDA type |

The rule: **choose the vector type that gives exactly 128 bits per thread**. For `fp16` (half-precision, 16 bits each) data the right type is `half8` (8 × 16 bits = 128 bits), not `float4`. Always match vector width to data type so one instruction moves the full 128 bits.

## Warp Divergence

If threads within a warp take different branches (`if`/`else`), the warp executes both paths serially with inactive threads masked — this is **warp divergence** and reduces effective parallelism. Minimizing divergence is a key kernel optimization concern.

## Occupancy

**Occupancy** is the ratio of active warps to the maximum warps an SM can hold. Higher occupancy gives the scheduler more warps to switch between when one warp stalls on a memory load, hiding latency. Register and shared memory usage per thread constrain how many warps fit simultaneously.

## Related Concepts

- [[simt-programming-model]]
- [[arithmetic-intensity]]
- [[5-kernel-dev-ping-pong-buffer]]
- [[operator-development]]

## Sources

- [[5-kernel-dev]]
