```yaml
title: Autotuning
type: concept
tags: [gpu, cuda, optimization, gemm, template-parameters, performance]
created: 2026-04-19
updated: 2026-04-19
sources: [kernel-dev.md]
```

# Autotuning

Autotuning is the automated search over a space of kernel configuration parameters (tile sizes, block sizes, pipeline depth, etc.) to find the combination that yields optimal performance for a given operator on a given hardware target.

## Motivation

High-performance GPU kernels (especially GEMM) have many inter-dependent tunable parameters: tile dimensions (M_tile, N_tile, K_tile), threads per block, pipeline stages, vectorization width, etc. The optimal combination is hardware-specific, problem-size-specific, and cannot be determined analytically. Manual tuning is infeasible across the full parameter space.

## How It Works

CUDA GEMM operators are typically written as **C++ templates** with tile sizes and other parameters as template arguments. Autotuning:
1. Enumerates candidate parameter combinations.
2. Compiles and runs each configuration (or uses a performance model to predict performance).
3. Measures throughput (FLOP/s) for each configuration.
4. Selects the best-performing configuration.
5. Optionally, stores results in a lookup table for future launches with the same problem shape.

## Scope

- Most commonly applied to **GEMM** and **convolution**.
- Also relevant to other operators with configurable tiling parameters.
- Block size selection is a simpler form of autotuning.

## Relationship to Development Workflow

In the SJTU project's development workflow, after establishing correctness, autotuning is used to explore the optimization space efficiently, complementing manual analysis with the [[roofline-model]] and [[nsight]] profiling.

## Related Concepts

- [[cuda-kernel-optimization]]
- [[shared-memory-tiling]]
- [[ping-pong-buffer]]
- [[gemm]]
- [[roofline-model]]
- [[arithmetic-intensity]]

## Sources

- [[kernel-dev]]

---
