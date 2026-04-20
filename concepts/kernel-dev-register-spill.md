```yaml
title: Register Spill
type: concept
tags: [gpu, cuda, registers, performance, memory-hierarchy]
created: 2026-04-19
updated: 2026-04-19
sources: [kernel-dev.md]
```

# Register Spill

Register spill is a GPU performance hazard that occurs when a CUDA kernel requires more registers per thread than the hardware allocates, causing the compiler to redirect the overflow variables to **local memory** — a region of global memory (HBM) — which is orders of magnitude slower than registers.

## Mechanism

Each SM has a fixed total register file (e.g., 65536 32-bit registers on modern NVIDIA GPUs). These are shared among all active threads on the SM. If a thread requires too many registers, the compiler spills the excess to **local memory**, which is physically located in HBM and accessed via the same slow path as global memory.

```cuda
__global__ void kernel() {
    float reg_var = 0.0f;       // Stored in register (fast)
    float local_arr[100];       // Likely spills to local memory (slow, in HBM)
}
```

## Causes

- **Large local arrays:** Variable-length or large arrays cannot fit in registers and are placed in local memory.
- **Excessive loop unrolling:** Unrolling a loop `N` times creates `N` copies of the loop body's register variables → N× register pressure → potential spill.
- **Too many simultaneously live variables:** Deep computation graphs with many intermediate values.

## Performance Impact

Register spill dramatically degrades performance because:
- Local memory (HBM) is 100–1000× slower than registers.
- Spill introduces extra global memory traffic, increasing effective memory bandwidth consumption.
- Can turn a compute-bound kernel into a bandwidth-bound one unexpectedly.

## Detection and Mitigation

- **Detection:** Use `--ptxas-options=-v` in nvcc to see register and local memory usage per kernel; [[nsight]] profiler shows local memory traffic.
- **Mitigation:**
  - Reduce per-thread live variable count.
  - Limit loop unrolling (the document explicitly warns against over-unrolling).
  - Use `__launch_bounds__` to guide the compiler's register allocation.
  - Restructure computation to reduce register pressure.

## Related Concepts

- [[gpu-memory-hierarchy]]
- [[cuda-kernel-optimization]]
- [[autotuning]]

## Sources

- [[kernel-dev]]

---
