```yaml
title: Arithmetic Intensity
type: concept
tags: [gpu, performance-modeling, roofline-model, compute-bound, bandwidth-bound]
created: 2026-04-19
updated: 2026-04-19
sources: [kernel-dev.md]
```

# Arithmetic Intensity

Arithmetic intensity (also called compute-to-bandwidth ratio) is the ratio of the number of floating-point operations performed by a kernel to the number of bytes of data transferred between compute and memory. It is the key metric for determining whether a kernel is **compute-bound** or **bandwidth-bound**, and for setting the correct optimization target.

## Definition

```
Arithmetic Intensity = FLOPs / Bytes of Memory Traffic
```

Units: FLOPs per byte (FLOP/B).

## Roofline Model

The **Roofline Model** uses arithmetic intensity to locate a kernel's performance bottleneck relative to hardware limits:

- **Hardware balance point** = Peak FLOP/s / Peak Memory Bandwidth (GB/s)
  - E.g., if peak compute = 100 TFLOP/s and peak bandwidth = 2 TB/s, balance point = 50 FLOP/B.
- If **kernel arithmetic intensity < balance point** → **bandwidth-bound**: memory bandwidth is the bottleneck.
- If **kernel arithmetic intensity > balance point** → **compute-bound**: compute throughput is the bottleneck.

## Optimization Implications

| Bound | Bottleneck | Optimization Target |
|---|---|---|
| Bandwidth-bound | Memory bandwidth | Coalescing, tiling, reducing redundant loads |
| Compute-bound | FP throughput | Reduce instruction count, increase ILP, avoid pipeline stalls |

**Target:** ≥ 80% utilization of whichever resource is the bottleneck.

**Critical practice:** Must be able to compute arithmetic intensity theoretically for any operator — cannot rely solely on profiling tools (though tools like [[nsight]] must also be used).

## Examples

- **GEMM** (large): high data reuse → high arithmetic intensity → compute-bound.
- **Element-wise operations** (ReLU, addition): minimal reuse → very low arithmetic intensity → strongly bandwidth-bound.
- **Convolution:** intermediate, depends on kernel size and input dimensions.

## Related Concepts

- [[roofline-model]]
- [[cuda-kernel-optimization]]
- [[coalesced-memory-access]]
- [[shared-memory-tiling]]
- [[gpu-memory-hierarchy]]
- [[gemm]]

## Sources

- [[kernel-dev]]

---
