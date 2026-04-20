```yaml
title: GPU Memory Hierarchy
type: concept
tags: [gpu, cuda, memory, hbm, shared-memory, registers, cache, hardware]
created: 2026-04-19
updated: 2026-04-19
sources: [kernel-dev.md]
```

# GPU Memory Hierarchy

The GPU memory system is organized as a multi-level hierarchy, ranging from large but slow off-chip memory to small but extremely fast on-chip memory. Understanding this hierarchy is fundamental to writing high-performance CUDA kernels, because performance almost always depends on which level of the hierarchy data resides in and how efficiently it is accessed.

## Hierarchy (Slowest → Fastest)

**HBM / Global Memory → L2 Cache → L1 Cache → Shared Memory → Registers**

| Level | Bandwidth / Speed | Scope | Programmable? |
|---|---|---|---|
| System Memory (CPU) | PCIe ~16 GB/s | CPU-GPU transfer | Yes (host) |
| HBM / Global Memory | 80–100 GB/s | All threads globally | Yes |
| L2 Cache | Higher than HBM | All SMs | No (hardware) |
| L1 Cache | Higher than L2 | Within SM | No (hardware) |
| Shared Memory | On-chip, very fast | Within Thread Block | Yes |
| Registers | Fastest | Thread-private | Yes (implicit) |

## Programmer-Controllable Levels

Only **Shared Memory** and **Registers** are directly controllable in kernel code:

- **Shared Memory:** Declared with `__shared__`; used for intra-block communication and tiling. Scarce resource (tens to ~100 KB per SM).
- **Registers:** Compiler assigns local variables to registers. Exceeding the register limit causes **register spill** to **local memory** (a region of global/HBM memory), which is extremely slow. Over-unrolling loops is a common cause of register spill.

## L2 Cache Behavior

- Shared across all SMs; cannot be directly controlled by programmers.
- Replacement policy: **LRU** (hardware-determined).
- Cache line size: **128 bytes**.
- Access path: Warp request → L1 → (miss) L2 → (miss) HBM.
- Write behavior: On a write hit in L2, data is marked dirty and batch-flushed to HBM later, reducing HBM write traffic.

## Improving L2 Hit Rate

- **Temporal locality:** Reuse data already in cache (e.g., tiling in GEMM/convolution).
- **Spatial locality:** Access contiguous addresses so a loaded cache line is fully utilized.
- **Coalesced access:** Threads in the same warp access contiguous addresses, merging into fewer transactions.
- **Avoid large strides:** Strided access loads cache lines that are only partially used, causing high eviction rates.

## Key Properties

- **Register spill** to local memory (global memory) is a severe performance hazard — avoid over-unrolling loops.
- Shared memory is on-chip and fast but scarce; allocation decisions directly affect how many blocks can reside on an SM simultaneously (**occupancy**).
- The memory hierarchy is the central reason why techniques like [[shared-memory-tiling]], [[coalesced-memory-access]], and [[ping-pong-buffer]] exist.

## Related Concepts

- [[cuda-kernel-optimization]]
- [[shared-memory-tiling]]
- [[coalesced-memory-access]]
- [[register-spill]]
- [[bank-conflict]]
- [[ping-pong-buffer]]
- [[arithmetic-intensity]]
- [[roofline-model]]

## Sources

- [[kernel-dev]]

---
