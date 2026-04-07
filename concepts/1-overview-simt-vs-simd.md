```yaml
title: SIMT vs. SIMD
type: concept
tags: [gpu, chip, programming-model, simt, simd, domestic-chips]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# SIMT vs. SIMD

SIMT (Single Instruction, Multiple Threads) and SIMD (Single Instruction, Multiple Data) are two distinct hardware parallelism programming models used in GPU and accelerator chip design. Their difference is a root cause of the difficulty domestic Chinese chip vendors face in replicating NVIDIA's software ecosystem.

## SIMT (Single Instruction, Multiple Threads)

- Used by: NVIDIA GPUs (CUDA programming model).
- **Model:** A single instruction is issued and executed by many independent threads simultaneously. Each thread has its own register file and program counter, enabling divergent execution paths (threads can branch independently, though with performance penalties).
- **Programming:** CUDA C++ code is written from the perspective of a single thread; the hardware schedules thousands of threads in parallel (via warps).
- **Strength:** Flexible, supports irregular access patterns and conditional logic.

## SIMD (Single Instruction, Multiple Data)

- Used by: Many domestic Chinese accelerator chips.
- **Model:** A single instruction operates on a vector of data elements simultaneously. All elements execute the same operation — no per-element branching.
- **Programming:** Code is written explicitly for vector operations. No concept of independent threads.
- **Difference from SIMT:** Fundamentally different execution model — SIMT threads are more flexible (independent program counters); SIMD is more rigid (fixed vector width, no per-element divergence).

## Implications for Infra Work

- NVIDIA's entire software ecosystem (CUDA, libraries, frameworks) is built on SIMT.
- Domestic chip vendors using SIMD must rewrite or re-architect code to fit their model — they cannot simply port CUDA code.
- Operator development for domestic chips requires understanding the SIMD model deeply.
- This is a primary reason why domestic chips struggle to follow NVIDIA's feature set — it is not merely a compilation problem but a fundamental architectural mismatch.

## Related Concepts

- [[ai-chip-architecture]]
- [[llm-training-infra]]
- [[gpu-software-stack]]

## Sources

- [[1-overview]]

---
