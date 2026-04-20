```yaml
title: Bank Conflict
type: concept
tags: [gpu, cuda, shared-memory, bank-conflict, optimization]
created: 2026-04-19
updated: 2026-04-19
sources: [kernel-dev.md]
```

# Bank Conflict

A bank conflict is a performance hazard specific to GPU shared memory, caused when multiple threads in the same warp simultaneously access different addresses that map to the same memory bank, forcing the accesses to be serialized.

## Mechanism

GPU shared memory is physically divided into **multiple equal-width banks** (typically 32 banks for modern NVIDIA GPUs). The mapping from address to bank is: `bank_id = (address / word_size) % num_banks`.

- **No conflict (ideal):** Each of the 32 warp threads accesses a different bank. All 32 accesses proceed in parallel.
- **Bank conflict:** Multiple threads access different addresses within the same bank → hardware serializes the accesses → N-way conflict degrades throughput by factor N.
- **Broadcast (no conflict):** All threads access the **same address** in the same bank → hardware broadcasts the value → no serialization.

## Common Cause: Strided Access

Accessing shared memory with a stride that shares a common factor with the number of banks causes conflicts. Example: stride=2 over a 32-bank shared memory array → threads 0, 1, 2, ... map to banks 0, 2, 4, ... → only 16 banks used → 2-way conflict. Worse cases: stride=4 → 4-way conflict, etc.

## Solutions

| Problem | Solution |
|---|---|
| Strided access (stride shares factor with 32) | Change access pattern or data layout |
| Row-major 2D array with row width = multiple of 32 | Add **padding** (one extra element per row) to offset bank assignments |
| Address alignment issues | Apply address offset |

**Padding example:** For a `float shared[32][32]` array, changing to `float shared[32][33]` shifts each row's starting bank, breaking the conflict pattern.

## Impact

Bank conflicts serialize shared memory accesses, which can dramatically reduce throughput. A 16-way conflict reduces effective shared memory bandwidth by 16×. This is especially damaging in tiling (GEMM, convolution) where shared memory is heavily used.

## Related Concepts

- [[shared-memory-tiling]]
- [[cuda-kernel-optimization]]
- [[gpu-memory-hierarchy]]
- [[warp-reduce]]
- [[coalesced-memory-access]]

## Sources

- [[kernel-dev]]

---
