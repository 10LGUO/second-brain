```yaml
title: CUDA Kernel Development Cheat Sheet
type: concept
tags: [cuda, gpu, cheatsheet, occupancy, gemm, indexing, launch-config]
created: 2026-04-26
updated: 2026-04-26
sources: []
```

# CUDA Kernel Development Cheat Sheet

---

## What Is Always True (Hardware Invariants)

These hold across all NVIDIA GPU architectures:

```text
Warp size:               32 threads          (never changes)
Max threads per block:   1024                (never changes)
Per-dim block limits:    x ≤ 1024, y ≤ 1024, z ≤ 64
Per-dim grid limits:     x ≤ 2³¹-1, y ≤ 65535, z ≤ 65535
HBM transaction size:    128 bytes = 32 × 4B (one warp × one float)
Shared memory banks:     32 banks            (never changes)
```

---

## What Varies by Architecture

Always query at runtime with `cudaGetDeviceProperties` for portable code.

### Max threads per SM

```text
Architecture    GPU             Max threads/SM
────────────    ───             ──────────────
Kepler          GTX 780         2048
Maxwell         GTX 980         2048
Pascal          GTX 1080        2048
Volta           V100            2048
Turing          RTX 2080        1024
Ampere          A100            2048
Ada Lovelace    RTX 4090        1536
```

### Max blocks per SM

```text
Architecture    GPU             Max blocks/SM
────────────    ───             ─────────────
Kepler          GTX 780         16
Maxwell         GTX 980         32
Pascal          GTX 1080        32
Volta           V100            32
Turing          RTX 2080        16
Ampere          A100            32
Ada Lovelace    RTX 4090        24
```

Note: not monotonically increasing — Turing dropped to 16 while Ampere went back to 32.

### Shared memory per SM / per block

```text
Architecture    Shared mem/SM   Shared mem/block (default)
────────────    ─────────────   ──────────────────────────
Pascal          64KB            48KB
Volta           128KB           96KB  (configurable)
Turing          64KB            48KB
Ampere          192KB           48KB  (configurable up to 164KB)
Ada Lovelace    128KB           48KB  (configurable)
```

### Registers per SM

```text
Most architectures (Pascal and later): 65536 × 32-bit registers per SM
```

---

## Occupancy Formula

Occupancy = warps resident on SM / max warps SM supports.

```text
// [ALWAYS 32] warp size never changes
max_warps = max_threads_per_SM / 32

blocks_resident = min(
    floor(max_threads_per_SM / threads_per_block),
    //    ^ [VARIES: 2048 A100/V100, 1536 RTX4090, 1024 Turing]
    //                            ^ [YOUR KERNEL: BM*BN/(TM*TN)]

    floor(shared_mem_per_SM / shared_mem_per_block),
    //    ^ [VARIES: 192KB Ampere, 128KB Ada/Volta, 64KB Pascal/Turing]
    //                           ^ [YOUR KERNEL: (BM*BK + BK*BN)*4 bytes]

    floor(65536 / (regs_per_thread × threads_per_block)),
    //    ^ [CONSTANT: 65536 × 32-bit registers/SM, Pascal and later]
    //                 ^ [COMPILER decides: check with nvcc --ptxas-options=-v]
    //                                      ^ [YOUR KERNEL: BM*BN/(TM*TN)]

    max_blocks_per_SM
    // ^ [VARIES: 32 Ampere/Pascal/Volta, 24 Ada, 16 Turing/Kepler]
)

warps_resident = blocks_resident × (threads_per_block / 32)
//                                                      ^ [ALWAYS 32]
occupancy      = warps_resident / max_warps
```

**Example** — kernel4 on A100, BM=BN=128, BK=8, TM=TN=8:

```text
threads_per_block = BM*BN/(TM*TN) = 128*128/64 = 256          [your kernel]
shared_mem/block  = (BM*BK + BK*BN)*4 = (128*8 + 8*128)*4 = 8KB [your kernel]

thread limit:   floor(2048 / 256) = 8 blocks   [2048 = A100 chip constant]
shared limit:   floor(48KB / 8KB) = 6 blocks   [48KB = A100 default per block] ← bottleneck
register limit: compiler-dependent             [65536 regs/SM = A100 chip constant]
hardware limit: 32 blocks                      [32 = A100 chip constant]

blocks_resident = 6
warps_resident  = 6 × (256/32) = 48            [32 = always constant]
occupancy       = 48 / 64 = 75%                [64 = 2048/32 = A100 max warps]
```

Query at runtime:

```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
prop.maxThreadsPerMultiProcessor   // max threads/SM
prop.maxBlocksPerMultiProcessor    // max blocks/SM
prop.sharedMemPerMultiprocessor    // shared mem/SM
prop.regsPerMultiprocessor         // registers/SM
```

---

## Block Size Selection

```text
Rule                          Guidance
────                          ────────
Warp alignment                Must be multiple of 32
Minimum for full occupancy    ≥ max_threads_per_SM / max_blocks_per_SM
                              e.g. A100: 2048/32 = 64  RTX 4090: 1536/24 = 64
No idle SM slots              Choose divisor of max_threads_per_SM
                              Safe choices: 128, 256, 512 (common divisors of 2048/1536/1024)
High register/shared use      Prefer 128 or 256 (leave room for more blocks)
Low register/shared use       256, 512, or 1024
GEMM output tile              block_size = BM*BN / (TM*TN)  ← driven by output, not input
```

---

## Grid Size Selection

```text
Rule                          Guidance
────                          ────────
Minimum                       ≥ SM count (e.g. A100: 108 SMs)
Full utilization              integer multiple of wave_size = SM_count × max_blocks_per_SM
OneFlow formula               min(ceil(N/block_size), SM_count × tpm/block_size × 32)
GEMM grid                     (CEIL(N, BN), CEIL(M, BM)) — one block per BM×BN patch of C
```

---

## GEMM Parameter Roles

```text
Parameter   Determines          Constraint                               What's fixed vs yours
─────────   ──────────          ──────────                               ─────────────────────
BM, BN      block tile size     (BM*BK + BK*BN)*4 ≤ 48KB               48KB = chip constant (default, configurable)
BK          K-loop step         same shared memory constraint            —
TM, TN      per-thread output   register pressure: accum[TM][TN]        65536 regs/SM = chip constant
block_size  = BM*BN/(TM*TN)    ≤ 1024 [ALWAYS], multiple of 32 [ALWAYS]
grid        = (CEIL(N,BN),      M,N,K divisible by BM,BN,BK             —
               CEIL(M,BM))
```

Key distinction:

```text
Shared memory [48KB, chip constant] → governs BM, BN, BK  (input tiles As, Bs live here)
Registers     [65536, chip constant] → governs TM, TN      (output accum[TM][TN] lives here)
Thread count  [≤1024, always true]  → driven by output tile BM*BN/(TM*TN), not input tile
```

---

## Common Index Formulas

```text
Global thread index (1D):   blockIdx.x * blockDim.x + threadIdx.x
Row-major element:          row * row_stride + col
Flat → 2D:                  row = idx / width,   col = idx % width

Pointer init (GEMM):
  A (M×K, stride K):   &A[by * BM * K]              row partition, all K cols
  B (K×N, stride N):   &B[bx * BN]                  col partition, all K rows (row 0)
  C (M×N, stride N):   &C[by * BM * N + bx * BN]    both partitions, no K

Thread tile top-left:
  tx = (threadIdx.x % block_row_thread) * TN
  ty = (threadIdx.x / block_row_thread) * TM
```

---

## Related Concepts

- [[cuda-thread-hierarchy]]
- [[cuda-launch-config]]
- [[cuda-gemm]]
- [[warp]]
