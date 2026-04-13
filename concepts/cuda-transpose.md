```yaml
title: CUDA Matrix Transpose
type: concept
tags: [cuda, transpose, memory-coalescing, ldg, bandwidth, kernel-optimization]
created: 2026-04-13
updated: 2026-04-13
sources: [CUDA_Kernel_Samples]
```

# CUDA Matrix Transpose

Transposing an M×N matrix (`output[col][row] = input[row][col]`) is a classic bandwidth-bound kernel. The challenge: contiguous rows become contiguous columns after transposition, so either reads or writes must be strided. Naive implementations waste HBM bandwidth through non-coalesced access.

## Naive Implementation

```cuda
__global__ void transpose_naive(float* input, float* output, int M, int N) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N)
        output[col * M + row] = input[row * N + col];
}
```

Threads in a warp share the same `threadIdx.y`, varying only in `threadIdx.x`.

- **Read** `input[row * N + col]`: `col = ... + threadIdx.x` → consecutive addresses → **coalesced** ✓
- **Write** `output[col * M + row]`: `row = ... + threadIdx.y` (same for whole warp) → same column, different rows → **strided** ✗

One warp issues up to 32 separate write transactions instead of 1.

## Coalesced Write (with `__ldg`)

Swap which dimension maps to `threadIdx.x` so writes are coalesced. Accept strided reads and soften them with `__ldg`:

```cuda
__global__ void transpose_coalesced_write(float* input, float* output, int M, int N) {
    // row/col now index into output space
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < M)
        output[row * M + col] = __ldg(&input[col * N + row]);
        //     coalesced ✓              strided, but through read-only cache
}
```

- **Write** `output[row * M + col]`: `col = ... + threadIdx.x` → consecutive → **coalesced** ✓
- **Read** `input[col * N + row]`: `row = ... + threadIdx.y` (same for whole warp) → strided ✗ → mitigated by `__ldg`

### Why prioritize coalesced writes?

Writes go directly to L2/HBM — non-coalesced writes cause full 32× bandwidth waste with no mitigation. Reads can be cached; `__ldg` routes through the read-only texture cache which handles 2D spatial locality better than L1, reducing the strided read penalty.

## Why Transpose is Hard to Coalesce

The fundamental issue: a matrix stored in row-major order has rows contiguous in memory. Reading row-by-row is coalesced; writing to transposed positions (column-by-column) is strided. You cannot make both reads and writes coalesced simultaneously without reordering data in between — which is what shared memory tiling achieves (a further optimization not shown here).

## Memory Coalescing Recap

Since a warp issues memory transactions collectively, whether 32 accesses cost 1 or 32 transactions depends on address contiguity:

| Access | Cost | Reason |
| --- | --- | --- |
| `threadIdx.x` → column index | Coalesced | x varies within warp → consecutive addresses |
| `threadIdx.y` → row index | Strided | y is same for whole warp → addresses jump by row stride |

## Related Concepts

- [[warp]]
- [[cuda-thread-hierarchy]]
- [[5-kernel-dev-arithmetic-intensity]]

## Sources

- [[CUDA_Kernel_Samples]] `transpose/transpose.cu`
