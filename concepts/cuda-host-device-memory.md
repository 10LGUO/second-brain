```yaml
title: CUDA Host-Device Memory Model
type: concept
tags: [cuda, gpu, memory, cudaMalloc, cudaMemcpy, host, device, hbm, pcie]
created: 2026-04-10
updated: 2026-04-10
sources: [5-kernel-dev.md]
```

# CUDA Host-Device Memory Model

Central Processing Unit (CPU) and GPU have **separate memory spaces**. Data must be explicitly allocated on the GPU and transferred across the PCIe bus before a kernel can use it. This boundary is a major performance concern — transfers are slow relative to High Bandwidth Memory (HBM) bandwidth.

## The Three Steps

```text
CPU (host)                          GPU (device)
──────────────────────────────────────────────────
malloc()          →  cudaMalloc()       allocate
cudaMemcpy(H→D)   →  [kernel runs]      transfer in
cudaMemcpy(D→H)   ←  [results ready]   transfer out
```

## Key Functions

### `cudaMalloc`

Allocates memory on the GPU:

```cuda
float* a_d = nullptr;
cudaMalloc((void**)&a_d, N * sizeof(float));
```

- `a_d` (convention: `_d` suffix = "device") holds a GPU memory address
- The CPU **cannot** read or write this address directly
- Must be freed with `cudaFree(a_d)` when done

### `cudaMemcpy`

Copies data between CPU and GPU:

```cuda
// CPU → GPU (before kernel): send inputs
cudaMemcpy(a_d, a_h, N * sizeof(float), cudaMemcpyHostToDevice);

// GPU → CPU (after kernel): retrieve results
cudaMemcpy(c_h, c_d, N * sizeof(float), cudaMemcpyDeviceToHost);
```

- Crosses the PCIe bus — typically 16–32 GB/s, much slower than HBM (~2 TB/s on A100)
- Synchronous by default — CPU waits until transfer completes
- Minimizing transfers is a key optimization; keep data on GPU across multiple kernels

### `cudaCheck` (error handling pattern)

Every CUDA function returns an error code. A common helper macro wraps each call:

```cuda
#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n",
               file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Usage:
cudaCheck(cudaMalloc((void**)&a_d, N * sizeof(float)));
```

Without this, a failed `cudaMalloc` silently returns a null pointer, causing a crash later with no useful context. Always wrap CUDA calls in production code.

## Pointer Conventions

| Suffix | Meaning | Lives in |
| --- | --- | --- |
| `a_h` | host | CPU RAM |
| `a_d` | device | GPU HBM |

The CPU can read/write `a_h` freely. It cannot dereference `a_d` — any attempt is undefined behavior (segfault or silent corruption).

## Full Example Flow

```cuda
// 1. Allocate on CPU
float* a_h = (float*)malloc(N * sizeof(float));

// 2. Allocate on GPU
float* a_d;
cudaCheck(cudaMalloc((void**)&a_d, N * sizeof(float)));

// 3. Fill CPU array, transfer to GPU
for (int i = 0; i < N; i++) a_h[i] = i;
cudaCheck(cudaMemcpy(a_d, a_h, N * sizeof(float), cudaMemcpyHostToDevice));

// 4. Launch kernel — operates entirely in GPU HBM
my_kernel<<<grid, block>>>(a_d, N);

// 5. Transfer results back to CPU
cudaCheck(cudaMemcpy(a_h, a_d, N * sizeof(float), cudaMemcpyDeviceToHost));

// 6. Use results on CPU
printf("%f\n", a_h[0]);
```

## Why Separate Memory Spaces?

GPU HBM is physically on the GPU die (or package), connected to compute cores with massive bandwidth (~2 TB/s). CPU RAM is connected via PCIe (~32 GB/s). Making them a unified space would bottleneck GPU memory accesses to PCIe speeds. The explicit transfer model lets you batch transfers and keep data GPU-side across many kernel launches.

## Related Concepts

- [[cuda-thread-hierarchy]]
- [[hbm-high-bandwidth-memory]]
- [[arithmetic-intensity]]
- [[warp]]

## Sources

- [[5-kernel-dev]]
