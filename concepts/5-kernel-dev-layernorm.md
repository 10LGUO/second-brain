```markdown
---
title: LayerNorm (Layer Normalization)
type: concept
tags: [layernorm, normalization, deep-learning, operator-optimization, simd]
created: 2026-04-06
updated: 2026-04-06
sources: [5-kernel-dev.md]
---

# LayerNorm (Layer Normalization)

Layer Normalization (LayerNorm) normalizes activations across the feature dimension of a neural network layer, stabilizing training and improving convergence. It is a fundamental operator in transformer architectures and large language models.

## Mathematical Definition

Given input vector x = [x_i, i=1..N]:

1. **Compute mean:** μ = (1/N) × Σx_i
2. **Compute variance:** σ² = (1/N) × Σ(x_i − μ)²
3. **Normalize:** x̂ = (x − μ) / sqrt(σ² + ε)  
   (ε is a small constant for numerical stability)
4. **Scale and shift:** y = γ·x̂ + β  
   (γ = learned scale parameter, β = learned bias parameter)

## One-Pass Trick

Mean and variance can be computed in a single pass by accumulating both Σx_i and Σx_i² simultaneously, then deriving mean and variance algebraically. **Caveat:** This introduces floating-point precision errors due to accumulation order.

## SIMD Implementation Details (AI Chip)

- **Input/output shape:** M×N
- **Thread configuration:** 12×64 threads; each thread handles one row
- **L2 cache constraint (256K):** Full M×N input cannot fit in L2; weights (γ) and biases (β) are stored entirely in L2 to amortize access cost
- **N dimension limit:** N ≤ 16384 (sufficient for current LLMs)
- **Loop unrolling:** Inner loops unrolled twice; requires data alignment to 64 elements; below 64 elements, falls back to scalar accumulation
- **Reduce strategy:** Final mean/variance reduction is performed entirely in registers — avoids writing back to L1 and eliminates `mfence` overhead
- **Remainder handling:** No padding solution found; scalar fallback used; L3 out-of-bounds consequences unknown
- **SIMD instructions used:** add, mul, mac for mean/variance and normalization computation

## Unimplemented Optimizations (Known Gaps)

- **Ping-pong buffer:** Not implemented; would improve bandwidth utilization
- **Small-data re-partitioning:** When M < 12×64, threads are idle. When N is small, per-thread work is minimal. Proposed fix: split N across multiple threads (extreme case: one thread per SIMD width), use L2 for inter-thread reduce.

## Related Concepts
- [[simd-programming-model]]
- [[softmax]]
- [[reduce-operator]]
- [[ping-pong-buffer]]
- [[arithmetic-intensity]]

## Sources
- [[5-kernel-dev]]
```

---
