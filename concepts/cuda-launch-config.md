```yaml
title: CUDA Kernel Launch Configuration (grid_size and block_size)
type: concept
tags: [cuda, kernel-launch, block-size, grid-size, occupancy, warp, sm]
created: 2026-04-21
updated: 2026-04-21
sources: []
```

# CUDA Kernel Launch Configuration

```cuda
kernel<<<Dg, Db, Ns, S>>>(...)
```

| Parameter | Type | Meaning | Default |
| --- | --- | --- | --- |
| `Dg` | `dim3` | Grid dimensions — `Dg.x * Dg.y * Dg.z` = total blocks launched | required |
| `Db` | `dim3` | Block dimensions — `Db.x * Db.y * Db.z` = threads per block | required |
| `Ns` | `size_t` | Bytes of dynamic shared memory allocated per block | 0 |
| `S` | `cudaStream_t` | CUDA stream to execute on | 0 |

---

## block_size Selection

### Hard limits

- Must be > 0 and ≤ 1024 total threads
- Per-dimension limits: x ≤ 1024, y ≤ 1024, z ≤ 64

### Rule 1 — Multiple of 32

block_size should be a multiple of 32 (warp size). A block that is not a multiple of 32 wastes slots in the last warp — those threads exist but do no useful work.

### Rule 2 — Minimum for full occupancy

Occupancy = (concurrent threads on SM) / (max threads SM supports). To reach 100% occupancy:

```text
SM theoretical threads = block_size × max_blocks_per_SM ≥ max_threads_per_SM

⇒ block_size ≥ max_threads_per_SM / max_blocks_per_SM
```

Representative values:

| GPU | max threads/SM | max blocks/SM | min block_size |
| --- | --- | --- | --- |
| V100 | 2048 | 32 | 64 |
| A100 | 2048 | 32 | 64 |
| GTX 1080 Ti | 2048 | 32 | 64 |
| RTX 3090 | 1536 | 16 | 96 |

**Practical floor: block_size ≥ 96** to be safe across architectures.

### Rule 3 — Divisor of SM max threads

Block scheduling onto an SM is atomic — all threads in a block land on the same SM together. If block_size does not divide evenly into the SM's max thread count, some thread slots are permanently idle.

Common SM max thread counts are 2048, 1536, 1024. Their shared divisors:

```text
512, 256, 128
```

These three values (128, 256, 512) are the most common block sizes seen in production CUDA libraries.

### Rule 4 — Register and shared memory pressure

Each SM has a fixed register file and shared memory pool split across all resident blocks. Per-thread budget:

```text
registers_per_thread   ≤  SM_register_limit   / (block_size × blocks_resident)
shared_mem_per_thread  ≤  SM_shared_mem_limit / (block_size × blocks_resident)
```

- **Low register / shared memory use** → can afford larger block_size (512, 1024)
- **High register / shared memory use** → use smaller block_size (128, 256) to keep more blocks resident

---

## grid_size Selection

### Hard limits

- x dimension: up to 2³¹ − 1
- y and z dimensions: up to 65535

### Rule 1 — At least one block per SM

```text
grid_size ≥ SM count
```

A100 has 108 SMs. A grid smaller than 108 blocks leaves some SMs idle entirely.

### Rule 2 — Fill complete waves

A **wave** is the set of blocks that can run simultaneously across all SMs:

```text
wave_size = SM_count × max_blocks_per_SM
```

A grid with a partial final wave (tail effect) leaves most of the GPU idle while the last few blocks finish. To avoid this:

```text
grid_size ≥ SM_count × max_blocks_per_SM × kNumWaves
```

`kNumWaves = 32` is a common default — enough waves to keep the scheduler busy and amortize the tail.

### OneFlow formula

```cpp
unsigned grid_size = std::max<int>(1,
    std::min<int64_t>(
        (n + kBlockSize - 1) / kBlockSize,           // don't over-provision for small n
        sm_count * tpm / kBlockSize * kNumWaves      // cap at kNumWaves full waves
    ));
```

| Variable | Meaning |
| --- | --- |
| `n` | number of data elements |
| `kBlockSize` | block_size |
| `sm_count` | number of SMs on the GPU |
| `tpm` | max resident threads per SM |
| `kNumWaves` | target number of waves (typically 32) |

**Logic:**

- Small n: `(n + kBlockSize - 1) / kBlockSize` wins — don't launch more blocks than elements need
- Large n: `sm_count * tpm / kBlockSize * kNumWaves` wins — launch enough blocks to fill 32 waves, ensuring high GPU utilization regardless of tail effects

---

## Quick Reference

| Concern | Guidance |
| --- | --- |
| Warp alignment | block_size is a multiple of 32 |
| Min occupancy | block_size ≥ max_threads_per_SM / max_blocks_per_SM (≥ 96 safe) |
| No idle slots | block_size ∈ {128, 256, 512} (divisors of 2048/1536/1024) |
| High register use | prefer 128 or 256 |
| Low register use | 256, 512, or 1024 |
| Grid minimum | grid_size ≥ SM count (108 for A100) |
| Full utilization | grid_size = integer multiple of wave_size, or use OneFlow formula |

---

## Related Concepts

- [[cuda-thread-hierarchy]]
- [[warp]]

## Sources

- NVIDIA CUDA C Programming Guide — Execution Configuration
- OneFlow source: `sm_count * tpm / kBlockSize * kNumWaves` grid sizing pattern
