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

**Block** (thread block) — a **resource allocation unit** assigned to one SM. Its sole purpose is to group warps that share resources and can coordinate with each other:

- **Shared memory** — scoped to the block; all threads in the block share it
- **`__syncthreads()`** — barrier within the block only
- **SM assignment** — the block is the unit the hardware scheduler places on an SM

Everything else — actual execution, memory transactions, latency hiding — happens at the warp level. The block is the container that defines which warps share resources and can talk to each other. Threads in different blocks cannot communicate directly.

```text
Block = resource bundle assigned to one SM
  ├── shared memory pool  (scoped to this block)
  ├── register file slice (partitioned across threads)
  └── set of warps that can synchronize via __syncthreads()
```

**Why max 1024 threads per block?** Hardware limit — 1024 threads = 32 warps, which is the maximum the SM's per-block warp tracking supports. Beyond that, register pressure also makes it nearly impossible to have more than one block resident on the SM.

**Block shape** can be any x/y/z combination as long as the total ≤ 1024, within per-dimension hardware limits (x≤1024, y≤1024, z≤64):

```text
dim3 block(1024)      // 1D — simple array kernels
dim3 block(32, 32)    // 2D — matrix kernels, warp-aligned in x
dim3 block(16, 16)    // 2D — smaller tile, more blocks per SM
dim3 block(256)       // 1D — thread tile GEMM
dim3 block(8, 8, 8)   // 3D — 512 threads, volumetric kernels
```

The shape is a programmer convenience for index arithmetic — the hardware flattens all threads to a 1D sequence anyway. Pick the shape that makes `threadIdx.x/y/z → data index` mapping cleanest, keeping `threadIdx.x` on the column for coalescing.

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

**Inter-block communication always goes through HBM.** There is no direct register-to-register or shared-memory-to-shared-memory path between blocks. Shared memory is physically scoped to one block — even blocks resident on the same SM have completely separate shared memory pools invisible to each other:

```text
SM
├── block A  →  shared mem A  (private — only A's threads can access)
├── block B  →  shared mem B  (private — only B's threads can access)
└── block C  →  shared mem C  (private — only C's threads can access)
```

This is a deliberate design choice — it lets the hardware scheduler place blocks on any SM in any order without tracking inter-block dependencies, which is what enables CUDA's massively parallel execution model.

The consequence: every inter-block communication path goes through HBM:

```text
block A → write to HBM → block B reads from HBM
```

`__syncthreads()` only synchronizes threads **within the same block**. It has no effect across blocks.

To coordinate across blocks, you have three options:

**1. Atomics** — for simple accumulation into a single value:

```cuda
if (laneId == 0) atomicAdd(output, sum);
```

All blocks race to update `output`; `atomicAdd` serializes those writes safely. `*output` must be initialized to 0 before the kernel launches (`cudaMemset`). Simple but contention grows with block count — at large grid sizes, blocks queue up waiting for the atomic and the tail sits idle.

**2. End the kernel, launch another** — the gap between two kernel launches is an implicit global sync point. All blocks from the first launch are guaranteed finished before the second launch begins:

```cuda
// allocate partial sums buffer in HBM — one float per block
float* partial;
cudaMalloc(&partial, grid_size * sizeof(float));

// Stage 1: each block reduces its chunk → writes one partial sum to HBM
reduce_stage1<<<grid_size, block_size>>>(input, partial, N);
// implicit global sync — all blocks done, partial[] fully written in HBM

// Stage 2: one block reduces the partial sums → final answer
reduce_stage2<<<1, block_size>>>(partial, output, grid_size);
```

Cleaner than atomics for large inputs: Stage 1 is fully parallel, Stage 2 reads only `grid_size` floats (tiny, e.g. 108 on A100). Tradeoff: one extra HBM buffer and two kernel launches.

The extra HBM round trip is unavoidable — `partial[]` must live in HBM because it is the only memory visible to all SMs simultaneously.

**3. Cooperative Groups** — a newer CUDA feature that allows grid-wide `__syncthreads()`-style barriers within a single kernel. Requires specific hardware support and constrains occupancy. Threads still communicate via global memory — the feature just adds a convenient sync point around it.

## SM Assignment

The GPU hardware scheduler distributes blocks across SMs automatically. An SM can hold multiple blocks simultaneously to hide memory latency (see [[warp]] — occupancy). For real workloads, launch enough blocks to saturate all SMs (e.g. A100 has 108 SMs — launch thousands of blocks).

## Related Concepts

- [[warp]]
- [[simt-programming-model]]
- [[cuda-host-device-memory]]
- [[arithmetic-intensity]]
- [[cuda-launch-config]]

## Sources

- [[5-kernel-dev]]
