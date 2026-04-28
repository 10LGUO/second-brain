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

---

## CUDA GPU Reduction (Progressive Implementations)

GPU reduction follows the same tree structure but exploits the thread hierarchy. Each version builds on the previous.

### v1: Shared Memory Tree Reduction

Each block loads its chunk into on-chip shared memory, then performs a tree reduction. Only thread 0 writes the block's partial sum.

```cuda
__global__ void reduce_v1(float* input, float* output, int N) {
    __shared__ float s[BLOCK_SIZE];  // on-chip, block-scoped
    int tid = threadIdx.x;
    int n   = blockDim.x * blockIdx.x + tid;

    s[tid] = (n < N) ? input[n] : 0.0f;  // load HBM → shared mem
    __syncthreads();  // wait for all threads in block to load

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) s[tid] += s[tid + offset];
        __syncthreads();  // sync after each round
    }
    if (tid == 0) atomicAdd(output, s[0]);  // accumulate across blocks
}
```

Each element is read from HBM exactly once; all `log2(BLOCK_SIZE)` reduction rounds use shared memory (~100× faster than HBM). `atomicAdd` handles the cross-block race.

### v2: Warp Shuffle Reduction

Replaces shared memory for intra-warp reduction with `__shfl_down_sync` — values are passed directly between thread registers. Only one `__syncthreads()` is needed (between Phase 1 and Phase 2), versus `log2(BLOCK_SIZE)` in v1.

```cuda
__global__ void reduce_v2(float* input, float* output, int N) {
    __shared__ float s_y[32];  // one slot per warp (max 1024/32 = 32 warps)

    int idx    = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;   // which warp within this block
    int laneId = threadIdx.x % warpSize;   // position within the warp (0–31)

    float val = (idx < N) ? input[idx] : 0.0f;

    // Phase 1: intra-warp reduction via shuffle — no __syncthreads() needed (lockstep)
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);  // lane i += lane i+offset

    // Lane 0 of each warp writes its partial sum to shared memory
    if (laneId == 0) s_y[warpId] = val;
    __syncthreads();  // only sync point: wait for all warps to write

    // Phase 2: warp 0 reduces the per-warp partial sums
    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;  // number of warps in this block
        val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (laneId == 0) atomicAdd(output, val);
    }
}
```

**Why no sync within warp?** All 32 threads in a warp execute in lockstep — same instruction, same clock cycle. By the time the next line runs, every lane's shuffle is already done.

**`__shfl_down_sync(mask, val, offset)`**: lane `i` receives `val` from lane `i + offset`. Strictly intra-warp — cannot cross warp boundaries. `s_y` is the bridge between warps.

**`warpSize`**: a built-in read-only constant, always 32 on all current NVIDIA hardware.

**`laneId` vs `threadIdx.x`**: lane 0 of warp 1 has `threadIdx.x = 32`, not 0. `laneId = threadIdx.x % warpSize` gives the correct intra-warp position for checks like `if (laneId == 0)`.

### v3: Warp Shuffle + float4

Combines v2 with `float4` vectorized loads — each thread processes 4 elements per 128-bit LDG.128 instruction, reducing HBM transactions 4×. Grid size is divided by 4 accordingly.

```cuda
int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
float4 tmp = FLOAT4(input[idx]);
val += tmp.x + tmp.y + tmp.z + tmp.w;
// ... same warp shuffle reduction as v2
```

### Performance Comparison

| Version | Sync calls per block | Shared mem writes | HBM loads per element |
| --- | --- | --- | --- |
| v1 (shared mem tree) | `log2(BLOCK_SIZE)` | `log2(BLOCK_SIZE)` × all threads | 1 |
| v2 (warp shuffle) | 1 | 1 per warp | 1 |
| v3 (shuffle + float4) | 1 | 1 per warp | 0.25 (4 per transaction) |

---

## SGEMV — Matrix-Vector Reduction (one warp per row)

Single-precision General Matrix-Vector Multiplication (SGEMV): `y = A × x`, A is M×K, x is K×1, y is M×1. Each output element `y[row]` is the dot product of one row of A with x — a reduction over K.

```cuda
dim3 grid(M);   // one block per output row
dim3 block(32); // one warp per block

__global__ void sgemv(float* A, float* x, float* y, int M, int K) {
    int laneId = threadIdx.x % warpSize;
    int row = blockIdx.x;
    if (row >= M) return;

    // Phase 1: each lane accumulates a partial dot product
    // 32 lanes split K columns — lane i handles cols i, i+32, i+64, ...
    float res = 0.0f;
    for (int i = 0; i < CEIL(K, warpSize); i++) {
        int col = i * warpSize + laneId;
        res += (col < K) ? A[row * K + col] * x[col] : 0.0f;
    }

    // Phase 2: warp shuffle reduction — fold 32 partial sums into lane 0
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
        res += __shfl_down_sync(0xFFFFFFFF, res, offset);

    // Phase 3: lane 0 writes the final dot product
    if (laneId == 0) y[row] = res;
}
```

One warp per row parallelises the K dot product across 32 lanes, then reduces with warp shuffle. No shared memory needed — shuffle operates entirely within registers.

---

## Inter-Block Communication and HBM

`__syncthreads()` only synchronises threads **within one block**. It has no effect across blocks.

**All inter-block communication must go through HBM.** Shared memory is scoped to one block — even blocks resident on the same SM have completely separate, mutually invisible shared memory pools. There is no SM-wide shared memory.

```text
SM
├── block A → shared mem A  (private to A)
├── block B → shared mem B  (private to B)
└── block C → shared mem C  (private to C)
```

Every inter-block communication pattern involves an HBM write + read:

**Option 1 — atomicAdd:** each block's lane 0 atomically adds its partial sum to a single output location. Simple, but contention serialises blocks at large grid sizes.

```cuda
// output must be zeroed before launch: cudaMemset(output, 0, sizeof(float))
if (laneId == 0) atomicAdd(output, sum);
```

**Option 2 — two-kernel:** Stage 1 writes partial sums to an HBM buffer; Stage 2 reduces the buffer. The kernel launch gap is an implicit global sync — all Stage 1 blocks are done before Stage 2 starts.

```cuda
float* partial;
cudaMalloc(&partial, grid_size * sizeof(float));  // lives in HBM

reduce_stage1<<<grid_size, block_size>>>(input, partial, N);
// implicit global sync

reduce_stage2<<<1, block_size>>>(partial, output, grid_size);
```

Stage 2 reads only `grid_size` floats (e.g. 108 on A100) — trivially cheap. The extra HBM buffer is unavoidable because HBM is the only memory visible to all SMs simultaneously.

**Tradeoff:**

```text
atomicAdd    simple, no buffer, but serialises at large grid_size
two-kernel   extra HBM allocation + 2 launches, but Stage 1 fully parallel
```

## Related Concepts

- [[simd-vectorization]]
- [[memory-bandwidth]]
- [[layernorm]]
- [[softmax]]
- [[operator-fusion]]
- [[parallelism]]

## Sources

- [[5-kernel-dev]]
