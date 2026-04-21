```yaml
title: AI Infra Quiz
type: quiz
tags: [cuda, gpu, quiz, self-test]
created: 2026-04-21
updated: 2026-04-21
```

# AI Infra Quiz

Self-test questions drawn from the exploration of GPU architecture and CUDA kernel optimization. Questions are grouped by topic. Answers are in collapsed sections below each question — try to answer before reading.

---

## GPU Architecture

**Q1.** What is a warp? What is its size, and why that size?

<details>
<summary>Answer</summary>

32 threads that the hardware always schedules and executes together as one unit. Size is 32 because HBM transactions are 128 bytes — 32 threads × 4 bytes (float) = 128 bytes, so one warp fills exactly one memory transaction.

</details>

---

**Q2.** An A100 has 4 warp schedulers per SM, each paired with 32 CUDA cores. How many CUDA cores does one SM have, and what does that mean for warp execution?

<details>
<summary>Answer</summary>

4 × 32 = 128 CUDA cores per SM. Each scheduler issues one warp per cycle. Multiple warps *interleave* on the same SM (concurrent scheduling), but they share the 128 cores — they do not each get their own 32 cores simultaneously unless all 4 schedulers fire different warps in the same cycle.

</details>

---

**Q3.** What is occupancy? What limits it?

<details>
<summary>Answer</summary>

Occupancy = concurrent threads on SM / max threads SM supports. Limited by whichever resource runs out first:

```text
blocks_resident = min(
    floor(shared_mem_per_SM / shared_mem_per_block),
    floor(registers_per_SM  / (registers_per_thread × threads_per_block)),
    hardware_block_limit
)
warps_resident = blocks_resident × (threads_per_block / 32)
occupancy = warps_resident / max_warps_per_SM
```

</details>

---

**Q4.** Why does low occupancy hurt performance even if your kernel is compute-bound?

<details>
<summary>Answer</summary>

The warp scheduler hides memory latency by switching to another ready warp when the current warp stalls. With few resident warps, the scheduler has fewer choices — stalls go unhidden and the SM sits idle waiting for memory. Higher occupancy = more warps to choose from = better latency hiding.

</details>

---

## Memory Hierarchy

**Q5.** List the GPU memory hierarchy from fastest to slowest, with approximate latency for each.

<details>
<summary>Answer</summary>

```text
Registers     ~1 cycle       per-thread, private
Shared memory ~20–30 cycles  per-block, SRAM on SM
L2 cache      ~200 cycles    GPU-wide
HBM (global)  ~600+ cycles   GPU-wide, off-chip
```

</details>

---

**Q6.** Shared memory is divided into 32 banks. What causes a bank conflict, and what are two ways to fix it in a transpose kernel?

<details>
<summary>Answer</summary>

A bank conflict happens when multiple threads in the same warp access different addresses in the same bank — the accesses serialize. In a transpose, reading a column of shared memory hits the same bank 32 times (32-way conflict).

Fixes:
1. **Padding** — allocate `s_mem[BLOCK_SIZE][BLOCK_SIZE + 1]`, shifting each row by 1 element so column reads land on different banks
2. **Swizzling** — XOR the column index with the row index: `s_mem[row][col ^ row]`. XOR is a bijection so no two threads map to the same cell, and column reads spread across all 32 banks

</details>

---

**Q7.** What is the coalescing rule, and why does it apply to writes more than reads?

<details>
<summary>Answer</summary>

Coalescing rule: `threadIdx.x` should index the column dimension (fastest-varying) so that adjacent threads in a warp access adjacent memory addresses, fitting in one 128-byte HBM transaction.

It applies more strictly to writes because reads have mitigation options — `__ldg` routes through the read-only texture cache and handles strided access without full coalescing. Writes have no equivalent cache.

</details>

---

## GEMM Tiling

**Q8.** In a block-tile GEMM, what do BM, BN, and BK mean? Which ones appear in the grid size?

<details>
<summary>Answer</summary>

- BM = rows of C that one block owns (block tile height)
- BN = cols of C that one block owns (block tile width)
- BK = step size through the K dimension per loop iteration

Grid: `(CEIL(N, BN), CEIL(M, BM))` — BM and BN appear; BK does not (it is internal to the K loop).

</details>

---

**Q9.** Why does the pointer init `A = &A[by * BM * K]` contain K but not N? Why does it not contain bx?

<details>
<summary>Answer</summary>

A is M×K row-major. Row stride = K (each row has K elements). Skipping `by*BM` rows = `by*BM * K` elements. N does not appear because A's row stride is K, not N.

`bx` does not appear because A is not partitioned by output column. This block needs **all K columns** of its BM rows — the column split is in B (`B = &B[bx * BN]`), not A.

</details>

---

**Q10.** Why does K never appear in the C pointer init `C = &C[by*BM*N + bx*BN]`?

<details>
<summary>Answer</summary>

C is M×N — K is the reduction dimension that is summed away. After the dot product, K disappears. C has no K axis; its row stride is N.

</details>

---

**Q11.** The K loop has two `__syncthreads()` calls. What would go wrong if you removed each one?

<details>
<summary>Answer</summary>

**Remove first sync (after copy):** Fast threads start reading As/Bs before slow threads finish writing them. They read stale/garbage data from the previous iteration.

**Remove second sync (after compute):** The next iteration overwrites As/Bs while some threads are still reading them from the current iteration. Data corruption.

</details>

---

**Q12.** What alignment requirement do M, N, K have relative to BM, BN, BK? What happens if violated?

<details>
<summary>Answer</summary>

M must be divisible by BM, N by BN, K by BK. If violated, the last tile extends past the matrix boundary — the kernel reads or writes out-of-bounds memory (undefined behavior, likely garbage results or a segfault). Production kernels either assert alignment or pad matrices before launching.

</details>

---

## Thread Tile

**Q13.** In a thread-tile GEMM with TM=TN=8, what is the compute-to-memory ratio for the inner loop? How does this compare to the block-tile (1 thread per element)?

<details>
<summary>Answer</summary>

Per inner BK iteration: load 8 elements from As (one per TM row) + 8 elements from Bs (one per TN col) = 16 shared memory reads, then 8×8 = 64 FMAs.

Ratio: 64:16 = **4:1**

Block-tile (1 thread/element): 1 FMA per 2 reads = **1:2**

Thread tile is 8× better at hiding memory latency.

</details>

---

**Q14.** Why does a larger thread tile (bigger TM×TN) lead to lower occupancy?

<details>
<summary>Answer</summary>

More output elements per thread = more live variables (the `accum[TM][TN]` array) = more registers per thread. The SM has a fixed register file. More registers per thread → fewer threads fit on the SM → fewer warps resident → lower occupancy.

</details>

---

## Kernel Optimization Progression

**Q15.** Kernel 6 transposes A during the copy to shared memory. Why?

<details>
<summary>Answer</summary>

In the compute loop, each thread reads a column of As (one element per row, stepping down). Without transposition, consecutive column elements are BK apart in memory — bank conflict. After transposing As during the copy (storing col-major), what was a column read becomes a row read — consecutive memory — no bank conflict.

</details>

---

**Q16.** Double buffer (kernel 7) eliminates one `__syncthreads()` but not the other. Which one is eliminated and why can the other not be removed?

<details>
<summary>Answer</summary>

**Eliminated:** the first sync (load→compute barrier). With double buffering, the data for iteration K was loaded during iteration K-1, so it is already ready — no need to wait.

**Cannot remove:** the second sync (compute→load barrier). The K loop window cannot advance until all threads finish reading the current tile. If any thread is still computing from the current As/Bs when the next iteration writes to the other buffer, the window pointer advances incorrectly and the kernel produces wrong results.

</details>

---

## Launch Configuration

**Q17.** What is the minimum block_size to achieve full occupancy on an RTX 3090 (max 1536 threads/SM, max 16 blocks/SM)?

<details>
<summary>Answer</summary>

`block_size ≥ 1536 / 16 = 96`

Below 96 threads per block, even filling all 16 block slots cannot reach 1536 threads on the SM.

</details>

---

**Q18.** What is a "wave" and why does the OneFlow grid formula target 32 waves?

<details>
<summary>Answer</summary>

A wave = `SM_count × max_blocks_per_SM` — the set of blocks that can run simultaneously. When a wave completes, the next wave starts.

If grid_size is not a clean multiple of wave_size, the final partial wave leaves most of the GPU idle. Targeting 32 waves means even with a partial last wave, 31/32 waves ran at full utilization — the tail effect is diluted to <3%.

</details>
