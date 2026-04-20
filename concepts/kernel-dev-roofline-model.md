```yaml
title: Roofline Model
type: concept
tags: [gpu, performance-modeling, arithmetic-intensity, compute-bound, bandwidth-bound]
created: 2026-04-19
updated: 2026-04-19
sources: [kernel-dev.md]
```

# Roofline Model

The Roofline Model is a visual and analytical performance model for compute kernels that identifies whether a kernel's performance is limited by compute throughput or memory bandwidth, and quantifies how far the kernel is from the hardware performance ceiling.

## Core Idea

Every kernel has two hard upper bounds on performance:
1. **Compute roof:** Peak FLOP/s of the hardware (e.g., 100 TFLOP/s for a GPU).
2. **Bandwidth roof:** Peak memory bandwidth × arithmetic intensity (e.g., 2 TB/s × kernel's FLOP/B).

The achievable performance is the **minimum** of these two: `min(Peak FLOP/s, Bandwidth × Arithmetic Intensity)`.

This creates the characteristic "roofline" shape when plotted on a log-log graph of performance vs. arithmetic intensity.

## Hardware Balance Point

The **balance point** (also called ridge point) is where the two roofs intersect:

```
Balance Point = Peak FLOP/s / Peak Memory Bandwidth
```

- Kernels with arithmetic intensity **below** the balance point are **bandwidth-bound**.
- Kernels with arithmetic intensity **above** the balance point are **compute-bound**.

## Application to Optimization

1. Compute the kernel's theoretical arithmetic intensity.
2. Locate the kernel on the roofline plot.
3. Determine bound type.
4. Apply targeted optimizations to approach the relevant roof.
5. Target: **≥ 80% of the roofline ceiling** for the limiting resource.

The Roofline Model prevents wasted effort: there is no point optimizing compute if the kernel is bandwidth-bound.

## Limitations

- Assumes perfect overlap of compute and memory operations.
- Does not account for instruction-level bottlenecks, latency, occupancy effects, or cache behavior in detail.
- Serves as a first-order guide; [[nsight]] profiling provides deeper insight.

## Related Concepts

- [[arithmetic-intensity]]
- [[cuda-kernel-optimization]]
- [[gpu-memory-hierarchy]]
- [[nsight]]
- [[gemm]]

## Sources

- [[kernel-dev]]

---
