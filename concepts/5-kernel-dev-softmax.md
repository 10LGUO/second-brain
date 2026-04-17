```yaml
title: Softmax Operator
type: concept
tags: [softmax, normalization, operator-optimization, deep-learning, attention, cuda]
created: 2026-04-06
updated: 2026-04-13
sources: [5-kernel-dev.md]

```

# Softmax Operator

Softmax converts a vector of raw scores into a probability distribution — all outputs are in (0, 1) and sum to exactly 1.

```text
Softmax(x_i) = e^x_i / Σ e^x_j

```

Example: `[2.0, 1.0, 0.5]` → `[0.59, 0.24, 0.17]` (sum = 1.0). Larger inputs get disproportionately larger outputs (exponential amplification). Used in attention to normalize raw attention scores into weights, and in classification output layers.

## Numerical Stability

`e^x` overflows `float32` for large x (e.g. `e^1000 = inf`). The fix is to subtract the global max M before exponentiating:

```text
Softmax(x_i) = e^(x_i - M) / Σ e^(x_j - M)

```

Mathematically identical — M cancels out in numerator and denominator — but all exponents are now ≤ 0, so results stay in (0, 1].

## CUDA Three-Kernel Pipeline

Softmax requires a global max and a global sum across all N elements. Since blocks cannot communicate within a kernel launch, this is split into three separate kernels. Each kernel launch is an implicit global sync point.

```cuda
// Initialize max to -FLT_MAX (identity for max), sum to 0
setToNegativeMax<<<1,1>>>(max_val);
cudaMemset(sum, 0, sizeof(float));

// Pass 1: find global maximum M
max_kernel<<<grid, block>>>(input, max_val, N);

// implicit sync — max_val fully written before pass 2 starts

// Pass 2: compute Σ e^(xi - M)
sum_kernel<<<grid, block>>>(input, sum, max_val, N);

// implicit sync — sum fully written before pass 3 starts

// Pass 3: normalize each element
softmax_kernel<<<grid, block>>>(input, output, sum, max_val, N);

```

`max_val` and `sum` are single floats in HBM that act as the communication channel between passes.

### max_kernel

Finds the global max using warp shuffle reduction (same pattern as [[5-kernel-dev-reduce-operator]] v2, with `fmaxf` instead of `+=`). Out-of-bounds threads load `-FLT_MAX` (identity for max) so they don't corrupt the result.

```cuda
float val = (idx < N) ? input[idx] : (-FLT_MAX);
for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
// ... write warp max to shared mem, warp 0 reduces, lane 0 calls atomicMax

```

`atomicMax` for floats is not natively supported — implemented via `atomicCAS` (Compare-And-Swap) loop on the raw int bits. Values in `*address` only ever increase (fmaxf never writes smaller), so the loop always terminates with the true maximum.

### sum_kernel

Same warp shuffle structure, with `expf(input[idx] - *max_val)` as the per-thread value and `+=` as the reduction operator. `atomicAdd` accumulates each block's partial sum into the global `*sum`.

```cuda
float val = (idx < N) ? expf(input[idx] - *max_val) : 0.0f;

```

### softmax_kernel

One thread per element — no reduction needed:

```cuda
output[idx] = expf(input[idx] - *max_val) / (*sum);

```

Example (`input=[1,2,3]`, `M=3`, `sum=1.503`):

```text
thread 0: e^(1-3) / 1.503 = 0.135 / 1.503 = 0.090
thread 1: e^(2-3) / 1.503 = 0.368 / 1.503 = 0.245
thread 2: e^(3-3) / 1.503 = 1.0   / 1.503 = 0.665
sum = 1.0 ✓

```

## CUDA Function Qualifiers

| Qualifier | Runs on | Called from | Notes |
| --- | --- | --- | --- |
| `__global__` | GPU | CPU via `<<<>>>` | Kernel entry point |
| `__device__` | GPU | GPU only | Helper functions called from kernels |
| `__host__` | CPU | CPU only | Default if no qualifier |

`static` on a `__device__` function limits its visibility to the current translation unit — prevents linker conflicts if another `.cu` file defines the same function name.

## Classification

Softmax is **bandwidth-bound** — arithmetic intensity is low (few FLOPs per byte loaded from High Bandwidth Memory (HBM)). The bottleneck is the three HBM passes over the input.

## Related Concepts

- [[layernorm]]
- [[5-kernel-dev-reduce-operator]]
- [[5-kernel-dev-arithmetic-intensity]]
- [[cuda-thread-hierarchy]]
- [[warp]]

## Sources

- [[5-kernel-dev]]
