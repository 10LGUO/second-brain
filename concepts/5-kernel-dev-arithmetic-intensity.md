```yaml
title: Arithmetic Intensity
type: concept
tags: [arithmetic-intensity, performance-analysis, compute-bound, bandwidth-bound, operator-optimization]
created: 2026-04-06
updated: 2026-04-07
sources: [5-kernel-dev.md]
```

# Arithmetic Intensity

Arithmetic intensity is the key metric used to classify an operator as either compute-bound or bandwidth-bound. It measures how much computation is performed per unit of memory traffic.

## Definition

```text
Arithmetic Intensity = Total FLOPs / Total Memory Access (Bytes)
```

Higher arithmetic intensity means more computation is done per byte moved — characteristic of compute-bound operations (e.g., large matrix multiplications). Lower intensity indicates bandwidth-bound operations (e.g., element-wise ops, LayerNorm, Softmax, transpose).

## Balance Point

The **balance point** is a hardware-derived threshold:

```text
Balance Point = Peak Compute Performance (FLOPs/s) / Peak Memory Bandwidth (Bytes/s)
```

Comparing an operator's arithmetic intensity against the balance point tells you which resource is your bottleneck:

| Arithmetic Intensity vs. Balance Point | Regime | Bottleneck | Right optimization |
| --- | --- | --- | --- |
| Intensity **>** balance point | Compute-bound | FPU / Tensor Core throughput | Better tiling, Tensor Core usage, higher occupancy |
| Intensity **<** balance point | Bandwidth-bound | HBM bandwidth | Operator fusion to eliminate memory round-trips |

Optimizing for the wrong regime wastes effort — this is the first calculation to make before writing a custom kernel.

## Concrete Example: NVIDIA A100

- Peak compute (FP16): ~312 TFLOPs/s
- Peak High Bandwidth Memory (HBM) bandwidth: ~2 TB/s
- **Balance point: ~156 FLOPs/byte**

A large General Matrix Multiplication (GEMM) has hundreds of FLOPs/byte → compute-bound.
LayerNorm over a hidden dimension has a handful of FLOPs/byte → bandwidth-bound; fusing it with adjacent operators is the correct optimization.

## Why This Drives Custom Kernel Development

Framework operators (PyTorch, etc.) execute as separate kernels, each reading from and writing back to HBM. For bandwidth-bound operators this memory round-tripping is the dominant cost. A custom fused kernel keeps intermediate results in registers or L1 cache, entirely bypassing HBM between operations. Without writing a custom kernel you cannot control which regime you're in.

## Related Concepts

- [[operator-fusion]]
- [[hbm-and-gpu-memory]]
- [[simt-programming-model]]
- [[mfu-model-flops-utilization]]
- [[ping-pong-buffer]]

## Sources

- [[5-kernel-dev]]
