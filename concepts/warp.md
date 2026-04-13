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

Memory transactions are issued at the **warp level**, not the thread level. When a warp executes a load/store instruction, the hardware looks at all 32 threads' addresses simultaneously and merges them into as few 128-byte HBM transactions as possible.

Each thread operates on its own data — but whether those 32 accesses cost 1 transaction or 32 depends entirely on whether the addresses are contiguous.

**Coalesced (contiguous addresses):**
```text
thread 0 → address 100
thread 1 → address 104
thread 2 → address 108  ...  thread 31 → address 224

→ one 128-byte transaction covers all 32 floats ✓
```

**Non-coalesced (strided addresses):**
```text
thread 0 → address 100
thread 1 → address 100 + N×4   (next row, far away)
thread 2 → address 100 + 2N×4  ...

→ up to 32 separate transactions, each fetching 128 bytes but using only 4 ✗
→ 32× more HBM traffic than necessary
```

| Access pattern | Transactions | Effective bandwidth |
| --- | --- | --- |
| 32 threads × `float` (4B), contiguous | 1 × 128B transaction | Full |
| 32 threads × `float4` (16B), contiguous | 4 × 128B transactions | Full (4× data per instruction) |
| 32 threads, strided or random | Up to 32 transactions | Severely degraded |

### Thread Layout Within a Block

CUDA threads are laid out in **row-major order** — `threadIdx.x` varies fastest. A warp is always 32 consecutive threads by linear index, which means all 32 threads in a warp share the same `threadIdx.y` and differ only in `threadIdx.x`.

```text
4×4 block (x=col, y=row):
(y=0,x=0) (y=0,x=1) (y=0,x=2) (y=0,x=3)  ← warp 0: same y, consecutive x
(y=1,x=0) (y=1,x=1) (y=1,x=2) (y=1,x=3)  ← warp 1
```

Consequence: indexing by `threadIdx.x` (column) produces coalesced access; indexing by `threadIdx.y` (row) produces strided access.

### `__ldg` — Load Global (Read-Only Cache)

`__ldg(&ptr)` routes a global memory load through the **read-only texture cache** instead of the regular L1. The texture cache is designed for spatial locality in 2D — it caches regions around the accessed address in both dimensions, making it better suited to strided access patterns.

```cuda
output[row * M + col] = __ldg(&input[col * N + row]);
//     coalesced write ✓        strided read — softened by texture cache
```

`__ldg` does not fix non-coalescing — it reduces the penalty by hitting cache more often for irregular access. Maps directly to the PTX `LDG` (Load Global) instruction.

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

## Lanes

Each of the 32 threads in a warp occupies a **lane** (0–31). The term comes from the SIMT hardware model — a warp executes one instruction across 32 parallel lanes simultaneously, like a 32-lane highway.

```cuda
int laneId = threadIdx.x % warpSize;   // position within the warp (0–31)
int warpId = threadIdx.x / warpSize;   // which warp within the block
```

`laneId` is more precise than `threadIdx.x` for intra-warp operations: lane 0 of warp 1 has `threadIdx.x = 32`, not 0. Using `laneId` gives the correct intra-warp position for checks like `if (laneId == 0)`.

`warpSize` is a built-in read-only constant available in any `__global__` or `__device__` function, always equal to 32.

## Warp Shuffle Instructions

Because all 32 threads in a warp execute in lockstep, values can be passed directly between lanes through registers — no shared memory, no `__syncthreads()` needed.

**`__shfl_down_sync(mask, val, offset)`**: lane `i` receives `val` from lane `i + offset`.

```cuda
// Example: warp sum reduction
for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
// After log2(32)=5 rounds, lane 0 holds the sum of all 32 lanes
```

- `mask = 0xFFFFFFFF` — all 32 lanes participate
- Strictly intra-warp — cannot cross warp boundaries
- Lanes where `lane + offset > 31` receive their own value back (undefined but irrelevant — upper lanes are discarded as the tree converges)

For cross-warp coordination, shared memory is the only option: lane 0 of each warp writes to `s_y[warpId]`, then warp 0 reads all entries after `__syncthreads()`.

## Warp Divergence

If threads within a warp take different branches (`if`/`else`), the warp executes both paths serially with inactive threads masked — this is **warp divergence** and reduces effective parallelism. Minimizing divergence is a key kernel optimization concern.

## Occupancy

**Occupancy** is the ratio of active warps to the maximum warps an SM can hold. Higher occupancy gives the scheduler more warps to switch between when one warp stalls on a memory load, hiding latency. Register and shared memory usage per thread constrain how many warps fit simultaneously.

## Related Concepts

- [[simt-programming-model]]
- [[cuda-thread-hierarchy]]
- [[cuda-host-device-memory]]
- [[arithmetic-intensity]]
- [[5-kernel-dev-ping-pong-buffer]]
- [[operator-development]]

## Sources

- [[5-kernel-dev]]
