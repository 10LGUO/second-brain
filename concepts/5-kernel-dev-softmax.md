```markdown
---
title: Softmax Operator
type: concept
tags: [softmax, normalization, operator-optimization, deep-learning, attention]
created: 2026-04-06
updated: 2026-04-06
sources: [5-kernel-dev.md]
---

# Softmax Operator

Softmax is an exponential normalization function that converts a vector of raw scores into a probability distribution. It is used extensively in attention mechanisms (e.g., scaled dot-product attention in transformers) and output layers of classification networks.

## Computation Steps (Kernel Implementation)

1. **Find maximum value:** Use multiple vector registers to store local maxima across chunks of the input, then reduce to find the global maximum. (Subtracting the max before exponentiation improves numerical stability.)
2. **Compute exponential sum:**
   - a) Subtract the global max from each element
   - b) Compute the exponential of each element
   - c) Accumulate exponential values into a running sum (stored in a sum vector)
   - d) Write the exponential values to output or a temporary array for subsequent normalization
3. **Normalize:** Divide each exponential value by the exponential sum.

## Numerical Stability

The subtraction of the maximum value before computing exponentials prevents overflow (very large exp values) without changing the result of the softmax, since:

```text

softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)

```text

## Classification

Softmax is typically **bandwidth-bound** for standard sequence lengths, as the arithmetic intensity is low (relatively few FLOPs per byte loaded from HBM).

## Related Concepts
- [[layernorm]]
- [[reduce-operator]]
- [[arithmetic-intensity]]
- [[simd-programming-model]]

## Sources
- [[5-kernel-dev]]
```text

---
