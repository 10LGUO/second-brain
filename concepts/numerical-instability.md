```yaml
title: Numerical Instability (数值不稳定性)
type: concept
tags: [precision, floating-point, reduction, non-determinism, training, inference]
created: 2026-05-26
updated: 2026-05-26
sources: [7-accuracy-debugging.md]
```

# Numerical Instability (数值不稳定性)

Numerical instability arises because floating-point arithmetic is **non-associative**: `(a + b) + c ≠ a + (b + c)` in general. Computers represent real numbers with finite bits (FP32, FP16, BF16), introducing inherent rounding errors. The order of additions therefore changes results.

## Root Cause: Big-Eats-Small (大数吃小数)

When two numbers of vastly different magnitudes are added, the smaller number may be rounded to zero due to precision limits. The result depends on addition order. A single such error is tiny, but deep learning is filled with **reduction operations** that aggregate millions of numbers — tiny errors compound exponentially through layers.

## Reduction Operations Most Affected

- LayerNorm / RMSNorm (mean and variance over a full dimension)
- AllReduce / Reduce-Scatter (tensor parallelism and ZeRO)
- Matmul (multiply-accumulate chains)
- Softmax (sum over vocabulary dimension)

## Dynamic Batching Non-Determinism

In inference servers (e.g., SGLang), requests are dynamically grouped into variable-size batches. Batch size changes how GPU kernels split reduction work across thread blocks, which changes the floating-point addition order, which produces microscopically different outputs — **even with temperature=0 and a fixed seed**.

```
batch size changes
  → kernel launch config changes
    → reduction block split changes
      → FP addition order changes
        → result differs
```

## Solution: Batch-Invariant Kernels

**Thinking Machines Lab + SGLang (September 2025):** Implemented *batch-invariant* kernels for RMSNorm, Matmul, and attention. These kernels use a **fixed partition strategy** independent of batch size, ensuring computation order never changes. Reference: https://github.com/sgl-project/sglang/issues/10278

Other techniques:
- Fix seeds: `torch.manual_seed(seed)` + `torch.cuda.manual_seed(seed)`
- Set batch size = 1 to eliminate batching-induced non-determinism
- Generate random tensors on CPU, then `.to(device)` (so RNG differences across hardware don't matter)
- Fix topology / execution order so AllReduce accumulates in the same order every run

## Key Principle

> Don't try to eliminate floating-point imprecision (impossible in hardware). Instead, **eliminate the conditions that cause non-determinism** — fix the computation order.

## Related

- [[1-overview-precision-convergence]] — broader precision debugging methodology
- [[7-accuracy-debugging]] — full debugging guide
- [[5-kernel-dev-reduce-operator]] — reduction kernel implementation
- [[5-kernel-dev-layernorm]] — LayerNorm accumulation precision issues
