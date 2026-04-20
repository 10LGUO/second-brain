```yaml
title: CUDA GEMM (Matrix Multiplication)
type: concept
tags: [cuda, gemm, tiling, shared-memory, matmul, kernel-optimization]
created: 2026-04-19
updated: 2026-04-19
sources: [CUDA_Kernel_Samples]
```

# CUDA General Matrix-Matrix Multiplication (GEMM)

General Matrix-Matrix Multiplication (GEMM) computes `C = A × B` where A is M×K, B is K×N, and C is M×N:

```text
C[i][j] = Σ A[i][k] * B[k][j]   for k = 0..K-1
```

Each output element is the dot product of one row of A with one column of B. GEMM is the dominant operation in deep learning — linear layers, attention, and convolutions all reduce to it.

---

## Dimensions: M, N, K, BM, BN, BK

```text
A: M rows × K cols     (M×K)
B: K rows × N cols     (K×N)
C: M rows × N cols     (M×N)
```

The K dimension is the **reduction dimension** — it is summed over and disappears in the output. M and N are the output dimensions.

Tiled versions introduce block tile dimensions:

```text
BM = tile height for A and C  (rows per block tile)
BN = tile width  for B and C  (cols per block tile)
BK = tile depth  (cols of A tile = rows of B tile, stepped through in the K loop)
```

Each thread block computes one BM×BN tile of C by stepping through K in BK-sized chunks.

---

## Thread Assignment

One thread per output element within the tile:

```text
thread (tx, ty) owns output element C[by*BM + ty][bx*BN + tx]
```

The block at grid position `(bx, by)` handles the output tile starting at row `by*BM`, col `bx*BN`. Within that block, thread `(tx, ty)` is responsible for exactly one element of C.

---

## Pointer Initialization

Before the main loop, each pointer is moved to the tile's top-left corner so all subsequent indexing is relative:

```cuda
A = &A[(by * BM) * K];       // skip to this block's starting row in A
B = &B[bx * BN];             // skip to this block's starting col in B
C = &C[(by * BM) * N + bx * BN];  // skip to this block's output tile in C
```

After this, `A[ty * K + tx]` means "ty rows down, tx cols right from the tile origin" — no need to carry block offsets through every index.

---

## Block Tile GEMM: The K Loop

A and B are generally larger than shared memory — K can be thousands while `As` and `Bs` only hold BK columns/rows. The outer loop steps through K in BK chunks, accumulating partial dot products:

```cuda
__shared__ float As[BM * BK];  // tile of A: BM rows × BK cols
__shared__ float Bs[BK * BN];  // tile of B: BK rows × BN cols

float accum = 0.0f;  // lives in a register, persists across all iterations

for (int k = 0; k < K; k += BK) {
    // 1. Each thread copies one element from HBM → shared memory
    As[ty * BK + tx] = A[ty * K + tx];   // compact: A stride=K, As stride=BK
    Bs[ty * BN + tx] = B[ty * N + tx];

    __syncthreads();  // ALL threads must finish copying before anyone reads
                      // (without this, fast threads read stale data from slow threads)

    // 2. Advance pointers to next BK slice
    A = A + BK;        // move right BK columns in A
    B = B + BK * N;    // move down BK rows in B (each row is N elements wide)

    // 3. Each thread accumulates its partial dot product across this tile
    for (int i = 0; i < BK; i++) {
        accum += As[ty * BK + i] * Bs[i * BN + tx];
    }

    __syncthreads();  // ALL threads must finish reading before next iteration
                      // overwrites As/Bs with the next tile
}

C[ty * N + tx] = accum;  // write final result to HBM
```

**Why two `__syncthreads()`?**

```text
First  __syncthreads(): copy phase done → safe to read As/Bs
                        (all 1024 threads wrote their 1 element, tile is complete)
Second __syncthreads(): compute phase done → safe to overwrite As/Bs
                        (no thread is still reading the current tile)
```

**Why the loop is sequential per thread:**

The copy (`As[...] = A[...]`) is parallel — all 1024 threads copy simultaneously, one element each, filling the entire BM×BK tile. The inner `for (i = 0; i < BK)` loop is sequential within each thread — it reads BK elements from `As` and `Bs` to accumulate one partial dot product. The parallelism is across threads (each computing a different output element), not within a thread.

**`accum` across iterations:**

`accum` is a register variable that persists across all K/BK iterations. Each iteration adds a partial dot product from BK elements. After all iterations, `accum = Σ A[row][k] * B[k][col]` for all k — the complete dot product.

---

## Why Tiling Reduces HBM Traffic

Without tiling, each output element would require K reads from A and K reads from B — `M*N*2K` HBM reads total.

With BM×BN block tiling: each tile of A (`BM×BK`) is loaded once and reused by all BN threads computing that tile's row. Each tile of B (`BK×BN`) is loaded once and reused by all BM threads computing that tile's column.

```text
Reuse factor:
  A tile reused BN times (once per output column in the tile)
  B tile reused BM times (once per output row in the tile)
```

Larger tiles → more reuse → higher arithmetic intensity → closer to compute-bound.

---

## Mental Model Summary

```text
1. Thread assignment:  thread (tx,ty) → output element C[ty][tx] within tile
2. Pointer init:       move A, B, C pointers to this block's tile origin
3. K loop:            step through K dimension in BK chunks
   a. Copy:           all threads collectively fill As, Bs (1 element each, parallel)
   b. Sync:           wait for tile to be fully loaded
   c. Compute:        each thread accumulates BK multiply-adds into accum (sequential)
   d. Sync:           wait for all reads before overwriting tile
4. Write:             thread writes accum to C
```

---

## Related Concepts

- [[cuda-thread-hierarchy]]
- [[warp]]
- [[cuda-transpose]]
- [[5-kernel-dev-arithmetic-intensity]]
- [[5-kernel-dev-reduce-operator]]

## Sources

- [[CUDA_Kernel_Samples]] `sgemm/`
