```yaml
title: GPU Execution Model
type: concept
tags: [gpu, cuda, sm, kernel, block, warp, thread, stream, shared-memory, registers, hbm, occupancy, gemm]
created: 2026-07-02
updated: 2026-07-02
sources: []
```

# GPU Execution Model

From software abstractions down to hardware limits, and how they interact during kernel execution.

## The Hierarchy

```
Software                         Hardware
────────────────────────────     ────────────────────────────
CPU (Python / PyTorch)           CPU cores
  └─ launches kernels async
Stream (command queue)           ─── no direct hardware unit ───
  └─ belongs to one GPU
Kernel (grid of blocks)          GPU
  └─ grid_dim blocks
Block (group of threads)    →    SM (Streaming Multiprocessor)
  └─ block_dim threads               one block lives on exactly one SM
Warp (32 threads)           →    warp scheduler inside SM
Thread                      →    one CUDA core / lane
```

## Software Layer

### CPU
Orchestrator. Executes Python/PyTorch code, issues kernel launch commands into streams, and continues immediately (async — does not wait for GPU). With 4 CPU cores and 16 GPUs, typical division:
- Core 0-1: PyTorch main thread, launching compute kernels
- Core 2: NCCL communication thread (All-Reduce)
- Core 3: DataLoader prefetch thread (H2D transfers)

### Stream
An async command queue belonging to one specific GPU. Ops within one stream execute serially in order. Ops across different streams on the same GPU can execute concurrently — subject to SM resource availability.

```
GPU0:
  stream_0: [kernel A] ──────── [kernel C]
  stream_1:        [kernel B]              [kernel D]
  (concurrent if SMs not fully occupied)
```

Most reliable concurrency: compute stream (uses SMs) + H2D stream (uses DMA engine) — different hardware units, no resource conflict.

### Kernel
A function launched on the GPU. Defined by a grid of blocks:
```cpp
kernel<<<grid_dim, block_dim, shared_mem_bytes>>>(args)
```
`grid_dim` determines how many blocks are created. `block_dim` determines threads per block. All blocks in the grid run the same code on different data.

### Block
A group of threads that:
- Share the same shared memory allocation
- Can synchronize with `__syncthreads()`
- Are assigned together to one SM

**Block is a software concept only.** The hardware (SM) has no notion of block during execution — it sees warps. Block is the abstraction that defines resource ownership and synchronization scope.

One block → exactly one SM (never splits).
One SM → multiple blocks concurrently (if resources allow).

### Warp
The actual hardware execution unit: 32 threads that execute the same instruction in lockstep (SIMT). The SM warp scheduler picks ready warps to execute each cycle, hiding memory latency by switching to other warps while some wait for HBM.

`block_dim` should always be a multiple of 32 — otherwise the last warp has idle lanes (wasted compute).

### Thread
One lane in a warp. Has its own registers and program counter. Identified by `threadIdx` within a block and `blockIdx` within the grid.

## Hardware Layer

### SM (Streaming Multiprocessor)
The fundamental compute unit of a GPU. A100 has 108 SMs. Each SM independently executes its assigned blocks.

Per-SM resources (A100):

| Resource | Limit | Bottleneck effect |
|---|---|---|
| Threads | 2048 | Fewer concurrent blocks |
| Registers | 65536 (32-bit) | Spill to HBM if exceeded |
| Shared memory | 164KB (configurable) | Fewer concurrent blocks |
| Blocks | 32 | Hard cap |
| Warps | 64 | Determines latency hiding ability |

**Occupancy** = actual active warps / maximum possible warps per SM. Higher occupancy → more warps available to hide memory latency → better throughput (up to a point).

### Shared Memory
On-chip SRAM inside each SM. Explicitly managed by the programmer (`__shared__`). Orders of magnitude faster than HBM (~19TB/s bandwidth vs ~2TB/s for HBM).

On A100, shared memory and L1 cache share the same 192KB physical pool — allocating more shared memory leaves less L1. Tradeoff:
- More shared memory → manual control, great for regular access patterns (e.g. tiling)
- More L1 → better for irregular/unpredictable access patterns

Shared memory is per-block — threads in the same block see the same shared memory; threads in different blocks do not.

### Registers
Per-thread private storage. Fastest memory on the GPU (no latency). Each SM has 65536 32-bit registers shared across all threads currently resident.

```
registers per thread = 65536 / (threads per block × blocks per SM)
```

If a thread uses more registers than its share, the excess **spills to local memory in HBM** — same latency as global memory, silent performance killer. Check with `nvcc --ptxas-options=-v`.

### HBM (High Bandwidth Memory)
Main GPU memory (e.g. 80GB on A100, ~2TB/s bandwidth). All tensors live here by default in PyTorch. Every `torch` op reads inputs from HBM and writes outputs back to HBM.

HBM bandwidth is the outer constraint for memory-bound kernels. No amount of occupancy tuning helps if the kernel is bottlenecked on HBM reads/writes — use the roofline model to determine which regime you're in.

## Hardware Constraints Summary

```
threads per block ≤ 1024     ← block-level hard limit (CUDA error if exceeded)
threads per SM    ≤ 2048     ← SM-level hard limit
```

The block limit (1024) is stricter than SM limit (2048) by design — forces ≥ 2 blocks per SM, providing enough warps for latency hiding.

To scale beyond 1024 threads: use more blocks, not bigger blocks.

```python
threads_per_block = 256
grid_size = (N + threads_per_block - 1) // threads_per_block
kernel<<<grid_size, threads_per_block>>>()
```

## Example: GEMM Kernel on A100

```
Problem: C = A @ B,  A: [M=4096, K=4096],  B: [K=4096, N=4096]

Launch config:
  Block tile:     [128, 128]   each block computes a 128×128 output tile
  Thread tile:    [8, 8]       each thread computes an 8×8 output sub-tile
  Threads/block:  (128/8) × (128/8) = 256 threads = 8 warps
  Grid:           (4096/128) × (4096/128) = 32 × 32 = 1024 blocks
  SMs used:       min(1024, 108) = 108 SMs, ~9 blocks per SM
```

**Shared memory per block (K-tile size = 32):**
```
A tile: [128 × 32] × 2 bytes (BF16) = 8KB
B tile: [32  × 128] × 2 bytes       = 8KB
Total:  16KB  →  164KB / 16KB = 10 blocks per SM (not the bottleneck)
```

A tile and B tile are slices of A and B loaded into shared memory for one iteration of the K loop. Each block accumulates `C += A_tile @ B_tile` over K/32 = 128 iterations, reusing each 16KB tile across all 256 threads before fetching the next slice from HBM. This is the core of shared memory tiling — pay HBM cost once, reuse many times in fast on-chip memory.

**Register usage per thread:**
```
8×8 output tile = 64 FP32 accumulators = 64 registers
+ pointers, loop variables ≈ 100 registers total
Available: 65536 / 256 threads = 256 registers per thread  →  no spill
```

**Why this kernel is compute-bound:**
Each element of A and B is loaded once but participates in 128 multiply-accumulates (reuse factor = block tile size = 128). Arithmetic intensity >> roofline ridge point → SM compute is the bottleneck, not HBM bandwidth. Tuning block/thread tile size to maximize reuse and occupancy is the right lever.

## Cross-References

- [[kernel-dev-shared-memory-tiling]] — tiling strategy in detail
- [[kernel-dev-roofline-model]] — compute-bound vs memory-bound analysis
- [[kernel-dev-register-spill]] — register spill detection and mitigation
- [[1-overview-gpu-memory-hierarchy]] — HBM, SRAM, registers from architecture perspective
- [[9-perf-opt-torch-compile-cuda-graph]] — CUDA Graph for reducing launch overhead
