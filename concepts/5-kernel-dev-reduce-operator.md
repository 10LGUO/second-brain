```yaml
title: Reduce Operator
type: concept
tags: [reduce, operator-optimization, simd, parallelism, accumulation]
created: 2026-04-06
updated: 2026-04-06
sources: [5-kernel-dev.md]
```

# Reduce Operator

A reduce operator aggregates all elements of a tensor (or a slice thereof) into a single scalar value using an associative operation, most commonly summation. It is a foundational primitive used inside [[layernorm]], [[softmax]], attention, and many other operators.

## Implementations (Progressive Optimization)

### 1. Naïve Scalar Loop

The baseline implementation iterates over every element sequentially and accumulates into a single variable.

```cpp
float reduce_scalar(const float* data, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        acc += data[i];
    }
    return acc;
}
```

**Characteristics:**

- No parallelism exploited
- Compiler may auto-vectorize, but not guaranteed
- Memory-bound for large inputs; the bottleneck is cache line fetches

---

### 2. SIMD Vectorized Reduction

Using AVX2 (256-bit), eight `float32` values are accumulated per cycle using vector registers. Multiple accumulators are maintained to break sequential dependency chains.

```cpp
#include <immintrin.h>

float reduce_avx2(const float* data, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 16; i += 16) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(data + i));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(data + i + 8));
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    // Horizontal sum of acc0
    __m128 lo  = _mm256_castps256_ps128(acc0);
    __m128 hi  = _mm256_extractf128_ps(acc0, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum);
    // Scalar tail
    for (; i < n; i++) result += data[i];
    return result;
}
```

**Key ideas:**

- Two accumulators (`acc0`, `acc1`) allow the CPU to pipeline independent add operations, hiding FP-add latency (~4 cycles on modern x86)
- Horizontal sum at the end collapses the vector lane results
- Tail loop handles non-multiple-of-16 lengths

---

### 3. Multi-Threaded Parallel Reduction

For very large tensors (e.g., summing an entire embedding matrix), the work is partitioned across threads. Each thread reduces its local chunk, then a final scalar pass combines the partial sums.

```cpp
#include <omp.h>

float reduce_parallel(const float* data, int n) {
    int nthreads = omp_get_max_threads();
    std::vector<float> partial(nthreads, 0.0f);
    #pragma omp parallel
    {
        int tid   = omp_get_thread_num();
        int chunk = (n + nthreads - 1) / nthreads;
        int start = tid * chunk;
        int end   = std::min(start + chunk, n);
        float local = 0.0f;
        for (int i = start; i < end; i++) local += data[i];
        partial[tid] = local;
    }
    float result = 0.0f;
    for (float p : partial) result += p;
    return result;
}
```

**Characteristics:**

- Scales linearly up to the memory-bandwidth limit
- False sharing avoided by using per-thread local variables before writing to `partial`
- In practice, combine with SIMD inner loop per thread for maximum throughput

---

### 4. Online / Welford-Style Numerically Stable Reduction

Plain summation of large arrays of floats suffers from catastrophic cancellation. Kahan compensated summation or Welford's online algorithm can be used when numerical accuracy matters (e.g., computing variance inside [[layernorm]]).

```cpp
float reduce_kahan(const float* data, int n) {
    float sum  = 0.0f;
    float comp = 0.0f;  // compensation
    for (int i = 0; i < n; i++) {
        float y = data[i] - comp;
        float t = sum + y;
        comp = (t - sum) - y;
        sum  = t;
    }
    return sum;
}
```

**Trade-off:** Roughly 2× the scalar work; only worthwhile when precision loss would meaningfully affect downstream results (e.g., very long sequences or mixed-precision pipelines).

---

## Reduction Variants

| Variant | Operation | Use Case |
| --- | --- | --- |
| Sum | `acc += x` | Mean, LayerNorm, Softmax denominator |
| Max | `acc = max(acc, x)` | Softmax numerical stability shift |
| Sum-of-squares | `acc += x*x` | Variance in LayerNorm |
| Argmax | track index | Greedy decoding |
| LogSumExp | numerically stable log-sum | Log-space softmax |

---

## Performance Considerations

- **Memory bandwidth bound:** For large tensors, the bottleneck is reading from RAM/cache, not arithmetic. Ensure data is contiguous and cache-aligned (64-byte aligned for AVX2).
- **Accumulator count:** Using 4–8 independent accumulators is typically optimal on modern OOO CPUs to saturate FP pipelines.
- **Fused kernels:** Wherever possible, fold the reduce into an adjacent operator (e.g., compute mean and variance in a single pass for [[layernorm]]) to halve memory traffic.
- **Floating-point associativity:** Compilers will not reorder FP additions by default (`-ffast-math` enables this). Explicit unrolling with multiple accumulators is the portable alternative.

---

## Role in Larger Operators

- **[[layernorm]]** – requires mean (sum reduce) and variance (sum-of-squares reduce) over the hidden dimension
- **[[softmax]]** – requires max reduce (for numerical shift) and sum reduce (for normalization denominator)
- **Attention** – row-wise softmax involves per-row reduce operations over sequence-length dimension
- **[[cross-entropy-loss]]** – requires LogSumExp over the vocabulary dimension

---

## Related Concepts

- [[simd-vectorization]]
- [[memory-bandwidth]]
- [[layernorm]]
- [[softmax]]
- [[operator-fusion]]
- [[parallelism]]

## Sources

- [[5-kernel-dev]]
