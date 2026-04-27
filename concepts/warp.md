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

## Warp as a Scheduling Concept

A warp is a **scheduling concept**, not a memory concept. Warps have no memory of their own — memory belongs to threads (registers) and blocks (shared memory). A warp is purely the unit the hardware warp scheduler uses to dispatch work.

The **warp scheduler** is a physical hardware unit inside each SM. An SM has multiple warp schedulers (4 on A100), each owning 32 CUDA cores. Each scheduler picks one ready warp per cycle and dispatches its 32 threads to its 32 cores simultaneously:

```text
SM (A100)
├── Warp Scheduler 0 → 32 CUDA cores  ← issues 1 warp per cycle
├── Warp Scheduler 1 → 32 CUDA cores  ← issues 1 warp per cycle
├── Warp Scheduler 2 → 32 CUDA cores  ← issues 1 warp per cycle
└── Warp Scheduler 3 → 32 CUDA cores  ← issues 1 warp per cycle
                       ─────────────
                       128 CUDA cores total, 4 warps truly parallel per cycle

```

Threads don't permanently own CUDA cores — a thread borrows a core for the duration of each instruction, then releases it.

Within an SM, warps **interleave** (concurrency, not parallelism): when one warp stalls on an HBM load (~300–500 cycles), the scheduler instantly switches to another ready warp. True parallelism happens **across SMs** — each SM runs its own warp pool simultaneously and independently.

```text
SM 0: warps 0–N  interleaving (hide latency)
SM 1: warps M–P  interleaving              ← truly parallel with SM 0
SM 2: warps Q–R  interleaving              ← truly parallel

```

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

### Why 128 Bytes Per Transaction?

The 128-byte HBM transaction size is deliberately co-designed with the 32-thread warp:

```text
32 threads × 4 bytes (float) = 128 bytes = exactly 1 HBM cache line

```

In one warp cycle, all 32 threads issue their memory access simultaneously. If addresses are contiguous, the memory controller services all 32 in a single 128-byte round trip — maximizing bandwidth for that warp cycle. A smaller transaction would need multiple trips per warp cycle; a larger one would fetch data nobody requested.

### Thread Layout Within a Block

CUDA defines the linear thread index as:

```text
linear = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y

```

`threadIdx.x` has no multiplier — it varies fastest. This is a deliberate NVIDIA convention matching **C's row-major layout** where the column index varies fastest in memory. A warp is always 32 consecutive threads by linear index — so all 32 threads in a warp share the same `threadIdx.y` and differ only in `threadIdx.x`.

```text
4×4 block (x=col, y=row):
(y=0,x=0) (y=0,x=1) (y=0,x=2) (y=0,x=3)  ← warp 0: same y, consecutive x
(y=1,x=0) (y=1,x=1) (y=1,x=2) (y=1,x=3)  ← warp 1

```

**The coalescing rule: always put `threadIdx.x` on the column index** (the term without a row-stride multiplier). This ensures consecutive threads in a warp access consecutive memory addresses → 1 HBM transaction.

```cuda
// Correct — threadIdx.x on column → coalesced
float val = input[threadIdx.y * N + threadIdx.x];

// Wrong — threadIdx.x on row → strided (all warp threads jump by N floats)
float val = input[threadIdx.x * N + threadIdx.y];

```

### Coalescing in Transpose vs Normal Kernels

For most kernels (elementwise, reduction), you can make both reads and writes coalesced by putting `threadIdx.x` on the column for both:

```cuda
int idx = blockDim.x * blockIdx.x + threadIdx.x;  // column index
c[idx] = a[idx] + b[idx];  // both reads and write coalesced ✓

```

**Transpose is special** — it fundamentally cannot coalesce both reads and writes simultaneously. Reading row-by-row is coalesced but writing column-by-column is strided, and vice versa. You must choose one:

- **Coalesce writes** (chosen): `threadIdx.x` on output column → 1 write transaction per warp. Accept strided reads, mitigate with `__ldg` or shared memory tiling.
- **Coalesce reads**: strided writes — worse, because writes go directly to HBM with no cache mitigation.

Reads have more mitigation options (`__ldg` texture cache, shared memory staging) because the hardware can cache and prefetch. Non-coalesced writes hit HBM directly with no mitigation — so writes are prioritized for coalescing.

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

Both shuffle variants are strictly intra-warp (register-to-register, no shared memory, no `__syncthreads()`), and cannot cross warp boundaries.

**`__shfl_down_sync(mask, val, offset)`** — one-way read: lane `i` receives `val` from lane `i + offset`.

```cuda
// Warp sum reduction — only lane 0 gets the final result
for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
```

Mental model — one-way fold. At each step, the left half absorbs the right half and the right half is discarded:

```text
offset=16:  lanes 0..15 absorb lanes 16..31  → right half dead
offset=8:   lanes 0..7  absorb lanes 8..15   → only left quarter alive
...
offset=1:   lane 0 absorbs lane 1            → lane 0 has full sum
```

5 steps (log2(32)), only lane 0 holds the final value. Use when only one lane needs the result (e.g. writing one output element per warp).

---

**`__shfl_xor_sync(mask, val, offset)`** — symmetric exchange: lane `i` exchanges `val` with lane `i ^ offset`. Both lanes get the combined value.

```cuda
// Warp sum reduction — ALL lanes get the final result
for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
```

Mental model — two-way fold. At each step, every lane exchanges with its XOR partner — both sides survive with the combined value:

```text
offset=16:  lane 0 ↔ lane 16,  lane 1 ↔ lane 17, ...  all 32 lanes survive, each covers 2
offset=8:   lane 0 ↔ lane 8,   lane 1 ↔ lane 9,  ...  all 32 lanes survive, each covers 4
offset=4:   all 32 lanes, each covers 8
offset=2:   all 32 lanes, each covers 16
offset=1:   all 32 lanes, each covers 32 → every lane has full sum
```

5 steps (log2(32)), every lane holds the final value. Use when all lanes need the result (e.g. softmax where every thread needs `max_val` and `sum` to compute its own output).

**Why XOR gives the same pairs as +offset:**

For lanes 0..15, `N ^ 16 = N + 16` (bit 4 was 0). So the pairing is identical to `__shfl_down_sync` — same tree structure, same partners at each step. The only difference is that XOR makes the exchange symmetric (both sides get the combined value) instead of one-directional.

```text
offset=16, lane 0:   0 ^ 16 = 16   same partner as 0 + 16 = 16
offset=16, lane 16:  16 ^ 16 = 0   maps back to lane 0 (symmetric)
```

**Order of operations — no race condition:**

`__shfl_xor_sync` reads the partner's value as a snapshot **before** any lane updates its variable. So both sides see the original values:

```text
// max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
//
// lane 0 has max_val=3, lane 16 has max_val=7, offset=16:
//
// 1. shfl reads partner's max_val into a temp register (snapshot, not live reference)
//    lane 0:  temp = 7   (from lane 16)
//    lane 16: temp = 3   (from lane 0)
// 2. fmaxf compares original max_val with temp
//    lane 0:  fmaxf(3, 7) = 7
//    lane 16: fmaxf(7, 3) = 7
// 3. max_val is overwritten with result
//    both lanes now have max_val = 7
```

The shuffle is a snapshot — both lanes read originals first, then both update. This is what makes the two-way fold correct.

**When to use which:**

```text
__shfl_down_sync  →  only lane 0 needs result  (reduce then write once)
__shfl_xor_sync   →  all lanes need result      (reduce then each thread uses value)
                     avoids shared memory broadcast round-trip
```

For cross-warp coordination, shared memory is still the only option: lane 0 of each warp writes to `s_y[warpId]`, then warp 0 reads all entries after `__syncthreads()`.

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
