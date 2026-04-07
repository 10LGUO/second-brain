```markdown
---
title: SIMT Programming Model
type: concept
tags: [simt, gpu, cuda, kernel-development, parallelism]
created: 2026-04-06
updated: 2026-04-06
sources: [5-kernel-dev.md]
---

# SIMT Programming Model

SIMT (Single Instruction Multiple Threads) is the programming model underlying GPU computation, most prominently exposed through NVIDIA's CUDA framework. In SIMT, a single instruction is applied concurrently across many threads, each operating on its own data element (scalar). This abstracts away explicit data movement and vector width management from the programmer.

## Key Properties

- **No explicit data movement:** Data transfer between host and device is handled separately (e.g., via `cudaMemcpy`); inside the kernel, each thread accesses its element by index with scalar operations.
- **Scalar compute:** Each thread performs scalar arithmetic (add, multiply, transcendental functions, etc.); the hardware handles the parallelism.
- **No data alignment requirement:** Threads access arbitrary indices; alignment is not a programmer concern.
- **No manual ping-pong buffering:** The hardware and compiler manage latency hiding; shared memory double buffering can be used for advanced optimization but is not required for correctness.
- **Simpler programming model:** Compared to SIMD, writing kernels is less verbose.

## Representative Hardware
- NVIDIA GPUs (via CUDA)
- AMD GPUs (via ROCm/HIP)

## Contrast with SIMD
See [[simd-programming-model]] for a detailed comparison table.

## Special Function Units (SFU)
GPUs contain dedicated Special Function Units (SFUs) for transcendental functions (sin, cos, exp). These are separate from the main CUDA cores (1D compute) and matrix units (2D compute), and their utilization must be accounted for separately when estimating compute utilization.

## Related Concepts
- [[simd-programming-model]]
- [[arithmetic-intensity]]
- [[ping-pong-buffer]]

## Sources
- [[5-kernel-dev]]
```

---
