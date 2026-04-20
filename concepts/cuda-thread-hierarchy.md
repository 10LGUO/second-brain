```yaml
title: CUDA Thread Hierarchy (Grid, Block, Thread, Warp)
type: concept
tags: [cuda, gpu, thread, block, grid, warp, kernel-launch, parallelism]
created: 2026-04-10
updated: 2026-04-10
sources: [5-kernel-dev.md]
```

# CUDA Thread Hierarchy

CUDA organizes parallel execution into a four-level hierarchy. You write code for one thread; the hardware executes it across millions simultaneously.

```text
Grid
└── Blocks  (distributed across SMs by the hardware scheduler)
    └── Threads  (up to 1024 per block)
        └── Warps  (32 threads, the actual hardware execution unit)
```

## The Four Levels

**Grid** — the entire kernel launch. All blocks together.

**Block** (thread block) — a group of threads assigned to one Streaming Multiprocessor (SM). Threads within a block can:

- Share **shared memory** (fast on-chip L1 cache)
- Synchronize with `__syncthreads()`

Threads in *different* blocks cannot communicate directly.

**Thread** — the unit you write code for. Each thread gets unique built-in variables:

| Variable | Scope | Meaning |
| --- | --- | --- |
| `threadIdx.x` | Unique within block | Position within its block |
| `blockIdx.x` | Unique within grid | Which block this thread belongs to |
| `blockDim.x` | Same for all threads | Number of threads per block |

**Warp** — 32 threads the hardware always schedules together. See [[warp]].

## Who Decides What

A common source of confusion — block, warp, and thread sizes are determined by different parties:

| Concept | Size determined by | When |
| --- | --- | --- |
| Block size | **You** (programmer) | At kernel launch: `dim3 block(256)` |
| Warp size | **Hardware** (always 32) | Fixed — never changes |
| Warps per block | **Derived** = block_size / 32 | e.g. 256 threads → 8 warps |
| Grid size | **You** (programmer) | At kernel launch: `dim3 grid(...)` |
| Registers per thread | **Compiler** | Based on how many live variables your code needs |
| Blocks resident on SM | **Derived** at launch | `min(shared_mem_limit, register_limit, hardware_block_limit)` |
| Warps resident on SM | **Derived** = blocks_resident × warps_per_block | Determines occupancy |

The compiler's role is specifically register allocation — it analyzes your kernel and assigns registers to hold all live variables (loop indices, local arrays like `accum[TM][TN]`, intermediate values, pointers). More registers per thread → fewer threads fit on SM → fewer resident warps → lower occupancy.

The warp scheduler is a **fixed physical hardware unit** (4 on A100). It picks from the pool of resident warps each cycle. Occupancy determines how large that pool is — more resident warps gives the scheduler more choices when one stalls on memory.

```text
Programmer decides:   block size (e.g. 256 threads)
Hardware fixes:       warp size = 32 threads always
Hardware derives:     256 / 32 = 8 warps per block

Compiler decides:     registers per thread (based on code)
Hardware derives:     blocks resident on SM (see below)
Hardware derives:     warps resident = blocks_resident × warps_per_block
                      → this is occupancy
```

**Blocks resident on SM** is the key derived quantity — constrained by whichever resource runs out first:

```text
blocks_resident = min(
    floor(48KB / shared_mem_per_block),              // shared memory limit
    floor(65536 / registers_per_thread / threads_per_block),  // register file limit
    32                                               // hardware block limit (A100)
)
```

Increasing shared memory per block, registers per thread, or threads per block all reduce `blocks_resident` — and therefore reduce the number of warps the scheduler has available to hide latency.

## Kernel Launch Syntax

```cuda
kernel<<<grid_size, block_size>>>(args...);
//        ^              ^
//    # of blocks    threads per block
```

Total threads launched = `grid_size × block_size`.

## Computing Grid and Block Size

The standard pattern for a 1D array of N elements where each thread handles one element:

```cuda
int block_size = 1024;                  // threads per block (power of 2, ≤ 1024)
int grid_size  = CEIL(N, block_size);   // enough blocks to cover all N elements
kernel<<<grid_size, block_size>>>(a, N);
```

For vectorized access where each thread handles 4 elements (`float4`):

```cuda
int block_size = 1024;
int grid_size  = CEIL(CEIL(N, 4), block_size);  // CEIL(N,4) = threads needed
kernel<<<grid_size, block_size>>>(a, N);
```

`CEIL(a,b) = (a+b-1)/b` — always round **up** so no elements are skipped.

### Concrete example (N = 7, float4 kernel)

```text
CEIL(7, 4)    = 2  threads needed
CEIL(2, 1024) = 1  block needed
Launch: <<<1, 1024>>> → 1024 threads total
  thread 0: idx = 0 → processes a[0..3] ✓
  thread 1: idx = 4 → processes a[4..7], but a[7] is out of bounds → early exit
  threads 2–1023: idx ≥ 7 → early exit immediately
```

## The Bounds Check Pattern

Because threads must be launched in whole blocks, you always over-provision and guard:

```cuda
int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
if (idx >= N) return;  // extra threads exit safely
```

`CEIL` and `if (idx >= N) return` are always a pair — one ensures enough threads, the other prevents out-of-bounds memory access.

## Global Thread Index Formula

```cuda
int idx = blockDim.x * blockIdx.x + threadIdx.x;
```

- `blockDim.x * blockIdx.x` — offset to the start of this block
- `+ threadIdx.x` — offset within the block
- Each thread gets a unique `idx` across the entire grid

For `float4` (4 elements per thread), multiply by 4:

```cuda
int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
```

## Inter-Block Communication

Blocks are isolated — no shared memory between them, no direct communication within a kernel launch. The hardware scheduler places blocks on Streaming Multiprocessors (SMs) in any order; there is no guarantee two blocks are alive simultaneously.

To coordinate across blocks, you have three options:

**1. Atomics** — for simple accumulation into a single value:

```cuda
if (tid == 0) atomicAdd(output, input_s[0]);
```

All blocks race to update `output`; `atomicAdd` serializes those writes safely. Simple but contention grows with block count.

**2. End the kernel, launch another** — the gap between two kernel launches is an implicit global sync point. All blocks from the first launch are guaranteed finished before the second launch begins:

```cuda
// Stage 1: each block reduces its chunk → writes one partial sum per block
reduce<<<grid_size, block_size>>>(input, partials, N);

// implicit global sync — all blocks done, partials[] fully written

// Stage 2: one block reduces the partial sums → final answer
reduce<<<1, block_size>>>(partials, output, grid_size);
```

Cleaner than atomics for large inputs: Stage 1 is fully parallel, Stage 2 does one cheap serial reduction. Tradeoff: two kernel launches and an intermediate `partials` buffer in HBM.

**3. Cooperative Groups** — a newer CUDA feature that allows grid-wide `__syncthreads()`-style barriers within a single kernel. Requires specific hardware support and constrains occupancy.

## SM Assignment

The GPU hardware scheduler distributes blocks across SMs automatically. An SM can hold multiple blocks simultaneously to hide memory latency (see [[warp]] — occupancy). For real workloads, launch enough blocks to saturate all SMs (e.g. A100 has 108 SMs — launch thousands of blocks).

## Related Concepts

- [[warp]]
- [[simt-programming-model]]
- [[cuda-host-device-memory]]
- [[arithmetic-intensity]]

## Sources

- [[5-kernel-dev]]
