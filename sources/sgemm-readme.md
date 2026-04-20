```yaml
title: CUDA SGEMM Optimization (sgemm/README.md)
type: source
tags: [cuda, gemm, sgemm, tiling, shared-memory, thread-tile, float4, double-buffer, kernel-optimization]
created: 2026-04-20
updated: 2026-04-20
sources: [CUDA_Kernel_Samples/sgemm/README.md]
```

# CUDA SGEMM Optimization

Seven progressively optimized single-precision GEMM (General Matrix-Matrix Multiplication) kernels, comparing each against cuBLAS. Based on [NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE).

Dev environment: NVIDIA GeForce GTX 1050, 5 SMs, Compute Capability 6.1, 48KB shared memory per block.

---

## Key Terms

- **Memory access volume** (访存量): amount of data read/written from global memory per kernel
- **Compute-to-memory ratio** (计算访存比): ratio of compute operations to memory accesses per iteration — higher means better latency hiding

---

## Kernel 1 — Naive (Global Memory)

One thread per output element. Each thread reads one row of A and one column of B directly from global memory.

```cpp
int id_x = blockIdx.x * blockDim.x + threadIdx.x;
int id_y = blockIdx.y * blockDim.y + threadIdx.y;
float tmp = 0.;
for (int i = 0; i < K; i++)
    tmp += A[id_y * K + i] * B[i * N + id_x];
C[id_y * N + id_x] = alpha * tmp + beta * C[id_y * N + id_x];
```

**Analysis:**

- Compute-to-memory ratio: 1 FMA per 2 global reads = **1:2**
- Total memory access: `2 × K × M × N` reads
- No reuse: same row of A read repeatedly by different threads; same column of B read repeatedly by different threads
- Performance: <1/10 of cuBLAS

---

## Kernel 2 — Shared Memory Tiling (Block Tile)

Loads BM×BK tile of A and BK×BN tile of B into shared memory. Each block computes a BM×BN patch of C.

Grid: `(CEIL(N,BN), CEIL(M,BM))` — one block per BM×BN patch of C.

**Analysis:**

- Memory access reduced to `M×N×K × (1/BM + 1/BN)` — with BM=BN=32, this is **1/32** of kernel 1
- Compute-to-memory ratio: **still 1:2** — each shared memory read is still paired with one FMA
- Bottleneck: shared memory reads are cheap (much lower latency than global), but ratio unchanged

> Increasing BM/BN further reduces global memory traffic but increases shared memory usage per block, which reduces the number of blocks resident on an SM (fewer warps → less latency hiding).

---

## Kernel 3 — 1D Thread Tile (Register Accumulation)

Each thread computes TM output elements (a column of TM rows) instead of 1. Introduces a register array `tmp[TM+1]` where `tmp[TM]` caches one Bs element to avoid repeated shared memory reads.

Block size: `BM × BN / TM` threads.

```cpp
for (int i = 0; i < BK; i++) {
    tmp[TM] = Bs[tx + i * BN];       // cache Bs element in register
    for (int j = 0; j < TM; j++)
        tmp[j] += As[(ty + j) * BK + i] * tmp[TM];  // TM FMAs per Bs read
}
```

**Analysis (TM=8):**

- Per inner iteration: 1 Bs read + 8 As reads + 8 FMAs → compute-to-memory ratio **8:9** (vs 1:2 in kernel 2)
- Global memory access: reduced to **1/64** vs kernel 1 (using 64×64 block size)
- Effectively hides memory latency via high arithmetic intensity

---

## Kernel 4 — 2D Thread Tile

Extends to TM×TN per thread (a 2D sub-tile). Each thread owns TM rows and TN columns of the block's output patch.

```text
block_row_thread = BN / TN   // threads per row of block tile
block_col_thread = BM / TM   // threads per column of block tile
thread_num = block_row_thread × block_col_thread

tx = (threadIdx.x % block_row_thread) * TN   // left edge of thread's column range
ty = (threadIdx.x / block_row_thread) * TM   // top edge of thread's row range
```

Register array: `accum[TM][TN]` — TM rows × TN cols of partial sums.

---

## Kernel 5 — 2D Thread Tile + Register Cache for Shared Memory

Same as kernel 4 but caches an entire column of `As` and row of `Bs` into registers before the inner compute loop, avoiding repeated shared memory reads.

```text
// Instead of reading As[...] and Bs[...] inside the inner loop,
// preload into register arrays frag_a[TM] and frag_b[TN],
// then compute the outer product frag_a × frag_b → accum[TM][TN]
```

This eliminates shared memory bank conflicts in the inner loop and further improves the compute-to-memory ratio.

---

## Kernel 6 — Vectorized Memory (float4)

Uses `float4` to load 4 floats per instruction instead of 1, reducing instruction count and latency for the global→shared memory copy phase.

**Key detail:** When copying A into `As`, the data is **transposed** during the copy so that subsequent reads of `As` in the compute phase are along consecutive memory addresses (coalesced / conflict-free).

```text
Copy A → As (transposed): As[col * BM + row] = A[row * K + col]
Copy B → Bs (normal):     Bs[row * BN + col] = B[row * N + col]
```

**Requirements for float4:**

1. Pointer must be 16-byte aligned
2. Data type size must be a power of 2

**Benefits:**

- Increased memory bandwidth (4× fewer load instructions)
- Reduced instruction count
- Reduced latency

**Tradeoff:** Increases register pressure → may reduce occupancy if already register-limited.

---

## Kernel 7 — Double Buffer (Prefetch)

Overlaps global→shared memory loading with computation by maintaining two shared memory buffers and two register buffers, alternating between them each iteration.

**Key insight:** On GPU, memory load units and compute units are separate hardware — they can execute in parallel. Sequential code corresponds to sequential *instruction issue*, but instructions execute concurrently in different hardware units.

```text
Without double buffer (kernel 6):
  [load tile K]  → sync → [compute tile K] → sync → [load tile K+1] → ...
  ↑ compute must wait for load each iteration

With double buffer (kernel 7):
  [load tile 0]
  loop:
    [load tile K+1 into buffer B]   ← issued first (takes time)
    [compute tile K from buffer A]  ← runs concurrently with above
    swap A ↔ B
```

**What is eliminated:** The first `__syncthreads()` (load→compute barrier) inside the loop. The second sync (compute→load barrier) cannot be eliminated because the window cannot advance until all threads finish computing.

**Two levels of double buffering:**

1. **Block level (shared memory):** 2× shared memory; hides global→shared latency
2. **Thread level (registers):** 2× register array; hides shared→register latency

**Caveat:** The first load has unavoidable latency (cold start). Only iterations 2+ benefit from overlap.

---

## Performance Progression

| Kernel | Key optimization | Approx. vs cuBLAS |
| --- | --- | --- |
| 1 | Naive global memory | <10% |
| 2 | Shared memory tiling | ~20–30% |
| 3 | 1D thread tile + register | ~40–50% |
| 4 | 2D thread tile | higher |
| 5 | Register cache for shared mem | higher |
| 6 | float4 vectorized loads | higher |
| 7 | Double buffer prefetch | closest to cuBLAS |

---

## Related Concepts

- [[cuda-gemm]]
- [[cuda-thread-hierarchy]]
- [[warp]]
- [[5-kernel-dev-arithmetic-intensity]]

## Sources

- [[CUDA_Kernel_Samples]] `sgemm/README.md`
