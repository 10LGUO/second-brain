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

## Shared Memory Tiling (Fully Coalesced)

The best solution: use shared memory as a staging buffer. Load a 32×32 tile with coalesced reads, transpose within shared memory (cheap, on-chip), then write with coalesced writes.

```cuda
template <const int BLOCK_SIZE>
__global__ void transpose_tiled(float* input, float* output, int M, int N) {
    __shared__ float s_mem[BLOCK_SIZE][BLOCK_SIZE + 1];  // +1 padding: avoids bank conflicts on column reads
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;

    // Phase 1: coalesced read from HBM → shared mem
    int x1 = bx + threadIdx.x;  // input col
    int y1 = by + threadIdx.y;  // input row
    if (x1 < N && y1 < M)
        s_mem[threadIdx.y][threadIdx.x] = input[y1 * N + x1];
    //   row=y,col=x → coalesced read (threadIdx.x varies in warp) ✓
    __syncthreads();

    // Phase 2: coalesced write from shared mem → HBM
    int x2 = by + threadIdx.x;  // output col (was input row)
    int y2 = bx + threadIdx.y;  // output row (was input col)
    if (x2 < M && y2 < N)
        output[y2 * M + x2] = s_mem[threadIdx.x][threadIdx.y];
    //                                ↑ swapped indices = transpose
    //   threadIdx.x varies in warp → x2 consecutive → coalesced write ✓
}
```

**Why `BLOCK_SIZE + 1` padding?**

Phase 2 reads `s_mem[threadIdx.x][threadIdx.y]` — all 32 threads in a warp share the same `threadIdx.y` and have `threadIdx.x = 0..31`. This reads 32 elements down a column. In a 32-wide array, column elements are 32 words apart → all map to the same bank → 32-way bank conflict.

With `+1` padding (33 words per row), column elements are 33 words apart → spread across 33 different banks → no conflict:

```text
Without padding:  s_mem[0][k]=word k,  s_mem[1][k]=word 32+k  → both bank k  ✗
With padding:     s_mem[0][k]=word k,  s_mem[1][k]=word 33+k  → banks k and k+1 ✓
```

**Bank assignment:** shared memory has 32 banks, cycling every 4 bytes. `word i → bank (i % 32)`.

**Template parameter `<BLOCK_SIZE>`:** required because `__shared__ float s_mem[BLOCK_SIZE][BLOCK_SIZE+1]` must be sized at compile time — static shared memory allocation happens before any thread runs so the hardware can compute how many blocks fit on each SM.

**`dim3` launch:**
```cuda
dim3 block(32, 32);                      // 2D block: 1024 threads, x varies fastest
dim3 grid(CEIL(N, 32), CEIL(M, 32));     // enough tiles to cover M×N matrix
transpose_tiled<32><<<grid, block>>>(input, output, M, N);
```

`dim3` is CUDA's 3D dimension struct (unused dimensions default to 1). There is no `dim2` or `dim4` — `dim3` covers all cases; the hardware flattens everything to 1D anyway. A 4th dimension would provide no additional hardware capability.

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
