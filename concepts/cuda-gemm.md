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
BM = rows of C that one block owns   (block tile height)
BN = cols of C that one block owns   (block tile width)
BK = depth step through K per iteration (cols of A tile = rows of B tile)
```

**BM and BN are the output ownership dimensions** — they define the BM×BN patch of C that a single block is responsible for computing end-to-end. The grid is sized to tile the full output:

```text
grid = (CEIL(N, BN), CEIL(M, BM))   // one block per BM×BN patch of C
```

BK is the inner loop step — A and B are loaded BK columns/rows at a time into shared memory (`As[BM×BK]` and `Bs[BK×BN]`), and the outer K loop steps through all K in BK-sized chunks.

```text
C (M×N full output)
  └── Block tile (BM×BN) — one block per patch, grid = CEIL(M,BM) × CEIL(N,BN)
        └── Thread tile (TM×TN) — thread tile GEMM: one thread per TM×TN sub-patch
              └── Single element (1×1) — block tile GEMM: one thread per element
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
A = &A[(by * BM) * K];            // skip to this block's starting row in A
B = &B[bx * BN];                  // skip to this block's starting col in B
C = &C[(by * BM) * N + bx * BN]; // skip to this block's output tile in C
```

After this, `A[ty * K + tx]` means "ty rows down, tx cols right from the tile origin" — no need to carry block offsets through every index.

### Why each pointer looks the way it does

All three matrices are stored **row-major**: element `[row][col]` is at `base + row * row_stride + col`.

```text
Matrix   Shape   Row stride   Block's top-left corner        Pointer init
──────   ─────   ──────────   ───────────────────────        ────────────
A        M×K     K            row = by*BM, col = 0           by*BM * K
B        K×N     N            row = 0,     col = bx*BN       0*N + bx*BN  =  bx*BN
C        M×N     N            row = by*BM, col = bx*BN       by*BM * N + bx*BN
```

**A: why K appears, why bx does not**

`K` is A's row stride (each row has K elements), so skipping `by*BM` rows costs `by*BM * K` elements. `bx` does not appear because A is not partitioned by column — this block needs **all K columns** of its BM rows to compute the full dot product.

**B: why only bx*BN, no K**

B's top-left for this block is row 0, col `bx*BN`. Row 0 means no row offset — just slide `bx*BN` columns right. K does not appear because B's row stride is N, not K.

**C: why K never appears**

C is M×N — K has been summed away and does not exist as a dimension in C. C's row stride is N, so the top-left is `by*BM * N + bx*BN`.

### Dimension alignment requirement

The K loop steps in BK increments and the grid tiles in BM/BN increments with no bounds checks. This requires:

```text
M divisible by BM
N divisible by BN
K divisible by BK
```

If not, the last tile reads out of bounds. Production kernels either assert this at the call site or pad matrices to the next multiple before launching.

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

## Thread Tile GEMM

Instead of one thread per output element, each thread owns a TM×TN sub-tile of the block's BM×BN patch. This increases arithmetic intensity by amortizing shared memory reads over more compute.

```text
TM = rows each thread owns (thread tile height)
TN = cols each thread owns (thread tile width)

block_row_thread = BN / TN   // threads across one row of the block tile
block_col_thread = BM / TM   // threads down one column of the block tile
thread_num = block_row_thread × block_col_thread  // total threads per block

tx = (threadIdx.x % block_row_thread) * TN   // left column edge of this thread's tile
ty = (threadIdx.x / block_row_thread) * TM   // top row edge of this thread's tile
```

The register accumulator becomes a 2D array:

```cuda
float accum[TM][TN] = {0.0f};

for (int i = 0; i < BK; i++) {
    // cache Bs row into register to avoid repeated shared memory reads
    float b_reg[TN];
    for (int j = 0; j < TN; j++) b_reg[j] = Bs[i * BN + tx + j];

    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
            accum[m][n] += As[(ty + m) * BK + i] * b_reg[n];
}

// write back TM×TN elements
for (int m = 0; m < TM; m++)
    for (int n = 0; n < TN; n++)
        C[(ty + m) * N + (tx + n)] = accum[m][n];
```

**Why thread tile improves arithmetic intensity:**

With TM=TN=8, one inner BK iteration reads 1 Bs element + 8 As elements and does 8 FMAs. With a 2D tile (TM=TN=8), it reads 8 Bs + 8 As and does 64 FMAs → compute-to-memory ratio 64:16 = **4:1** vs 1:2 in the naive block tile.

**Tradeoff:** Larger TM×TN → more registers per thread → fewer threads fit on SM → lower occupancy. Lower occupancy means fewer warps available to hide memory latency. The optimal tile size balances arithmetic intensity against occupancy.

---

## Vectorized Memory Access (float4)

Instead of loading one float per instruction, use `float4` to load 4 floats per instruction. This reduces instruction count and latency for the global→shared copy phase.

```cuda
float4 *A4 = reinterpret_cast<float4*>(A);
// loads 4 floats in a single instruction
float4 tmp = A4[...];
```

**Key detail — transpose during copy:** When copying A into `As`, the data is transposed so that subsequent reads of `As` in the compute loop are coalesced (consecutive memory addresses):

```text
Normal copy:      As[row * BK + col] = A[row * K + col]
Transposed copy:  As[col * BM + row] = A[row * K + col]  ← As is now col-major
```

After transposing, reading a column of `As` during compute is reading consecutive memory — no bank conflicts.

**Requirements:** pointer must be 16-byte aligned; data size must be a power of 2.

---

## Double Buffer (Prefetch)

Overlaps the global→shared memory load of tile K+1 with the computation of tile K, hiding load latency.

**Key insight:** GPU memory load units and compute units are separate hardware — they execute in parallel. Sequential code = sequential *instruction issue*, but different hardware executes loads and FMAs concurrently.

```text
Without double buffer:
  [load tile K] → sync → [compute tile K] → sync → [load tile K+1] → ...
  compute must wait for each load

With double buffer:
  [load tile 0 → buf A]
  loop:
    [load tile K+1 → buf B]   ← issued first, executes in memory unit
    [compute tile K from buf A] ← runs concurrently in compute unit
    swap A ↔ B
```

This eliminates the first `__syncthreads()` inside the loop (load→compute barrier). The second sync (compute→load) cannot be removed because the window cannot advance until all threads finish.

**Two levels:**

1. **Block level (shared memory):** 2× shared memory alternates between loading and computing
2. **Thread level (registers):** 2× register array eliminates shared→register latency

**Caveat:** The very first load has no overlap (cold start). Only from the second iteration onward is latency hidden.

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

Optimization progression:

```text
Kernel 1 (naive):        1 thread → 1 element, reads global memory directly
Kernel 2 (block tile):   1 thread → 1 element, reads via shared memory
Kernel 3 (1D tile):      1 thread → TM elements (1 col, TM rows), register accum
Kernel 4 (2D tile):      1 thread → TM×TN elements, register accum[TM][TN]
Kernel 5 (reg cache):    preload As/Bs columns into registers before inner loop
Kernel 6 (float4):       vectorized global→shared copy, transpose As during copy
Kernel 7 (double buf):   overlap load K+1 with compute K, eliminate one __syncthreads()
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
- [[sgemm-readme]]
