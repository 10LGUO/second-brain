```yaml
title: GPU Memory Hierarchy
type: concept
tags: [gpu, memory, hbm, gddr, nvlink, dma, on-chip-cache, hardware]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# GPU Memory Hierarchy

The GPU memory system is divided into multiple levels extending outward from the compute cores. Levels closer to the compute cores offer higher bandwidth but smaller capacity, while levels farther away offer larger capacity but lower bandwidth. Understanding this hierarchy is fundamental to kernel development, video memory optimization, and memory access bandwidth analysis.

## Memory Hierarchy (Nearest to Farthest)

### 1. Registers

- **Scope:** Per-thread; private to each CUDA thread (or shader invocation)
- **Capacity:** Typically 256 KB – 512 KB of register file per Streaming Multiprocessor (SM)
- **Bandwidth:** Highest of any memory level; single-cycle latency
- **Latency:** ~1 cycle
- **Persistence:** Lifetime of the thread
- **Notes:** Register spilling (when a kernel uses more registers than available) causes values to be pushed to local memory (off-chip DRAM), incurring significant latency penalties. Compiler flags such as `--maxrregcount` control register usage.

---

### 2. Shared Memory / L1 Cache

- **Scope:** Per-thread-block (CTA); all threads in a block share the same bank
- **Capacity:** Configurable; typically 48 KB – 228 KB per SM depending on architecture and configuration (e.g., NVIDIA Hopper allows up to 228 KB shared memory per SM)
- **Bandwidth:** Very high; ~100–160 TB/s aggregate across all SMs on high-end GPUs
- **Latency:** ~20–40 cycles
- **Persistence:** Lifetime of the thread block
- **Key concept — bank conflicts:** Shared memory is divided into 32 banks (on modern NVIDIA GPUs). When multiple threads in a warp access the same bank simultaneously (but different addresses), the accesses are serialized, reducing effective bandwidth. Padding arrays or restructuring access patterns can eliminate bank conflicts.
- **Configurable split:** On many architectures (Volta, Turing, Ampere, Hopper), the SM has a unified data cache that can be partitioned between L1 cache and shared memory using `cudaFuncSetAttribute` with `cudaFuncAttributePreferredSharedMemoryCarveout`.
- **Related concept:** [[warp-execution]] — the warp scheduler benefits heavily from low-latency shared memory to hide arithmetic latency.

---

### 3. L2 Cache

- **Scope:** GPU-wide (shared across all SMs)
- **Capacity:** Typically 4 MB – 60 MB depending on GPU generation (e.g., NVIDIA A100: 40 MB; H100: 50 MB)
- **Bandwidth:** ~5–12 TB/s on high-end datacenter GPUs
- **Latency:** ~200–400 cycles
- **Persistence:** Not guaranteed; managed by hardware replacement policy
- **Notes:** L2 cache acts as the primary staging buffer between on-chip memory and device (HBM/GDDR) memory. Cache hit rates can be monitored with hardware performance counters (e.g., via `nsight-compute`). Persistent kernel techniques and stream-ordered memory allocations can improve L2 residency.

---

### 4. Device Memory (HBM / GDDR)

- **Scope:** GPU-wide; accessible by all SMs and (with appropriate flags) by the CPU via PCIe/NVLink
- **Capacity:** 8 GB – 192 GB depending on GPU SKU (e.g., H100 SXM: 80 GB HBM3; H200: 141 GB HBM3e)
- **Bandwidth:** 900 GB/s – 3.35 TB/s (HBM3e on H200)
- **Latency:** ~600–800 cycles
- **Variants:**
  - **HBM (High Bandwidth Memory):** Stacked DRAM dies connected via silicon interposer; used in datacenter GPUs (A100, H100, MI300X). Very high bandwidth, lower capacity per stack.
  - **GDDR6 / GDDR6X:** Used in consumer and workstation GPUs (RTX 30/40 series). Lower bandwidth than HBM but cheaper and available in larger configurations.
- **Memory types within device memory:**
  - **Global memory:** The default large heap; accessible by all threads
  - **Local memory:** Physically located in global memory; used for register spill and stack frames; private to each thread
  - **Texture memory:** Cached path through a dedicated texture cache; optimized for 2D spatial locality
  - **Constant memory:** 64 KB read-only region with a dedicated cache; broadcast-friendly for uniform accesses across a warp

---

### 5. NVLink / NVSwitch (Peer GPU Memory)

- **Scope:** Multi-GPU systems; allows one GPU to directly read/write another GPU's device memory
- **Bandwidth:** NVLink 4.0 (H100): 900 GB/s bidirectional per GPU; NVSwitch fabric in DGX H100: up to 3.6 TB/s all-to-all
- **Latency:** Higher than local device memory; typically adds ~1–2 µs over baseline
- **Persistence:** As long as the peer-access mapping is established (`cudaDeviceEnablePeerAccess`)
- **Notes:** NVLink bypasses PCIe, providing dramatically higher bandwidth for collective operations (AllReduce, AllGather) used in distributed training. Related concept: [[nvlink]].

---

### 6. Host Memory (CPU DRAM) via PCIe

- **Scope:** System-wide; CPU and GPU share access via PCIe bus
- **Bandwidth:** PCIe 5.0 x16: ~64 GB/s bidirectional; PCIe 4.0 x16: ~32 GB/s bidirectional
- **Latency:** Highest in the hierarchy; microseconds to tens of microseconds depending on transfer type
- **Access modes:**
  - **Pinned (page-locked) memory:** Registered with the OS to prevent paging; required for DMA transfers; enables `cudaMemcpyAsync` overlap with kernel execution
  - **Pageable memory:** Default CPU allocation; copies require a staging bounce buffer in pinned memory, adding latency
  - **Unified Memory (UM):** CUDA abstraction that provides a single pointer accessible from both CPU and GPU; the runtime handles page migration and prefetching via the Page Migration Engine (PME). Performance depends heavily on access patterns and prefetch hints (`cudaMemPrefetchAsync`).
  - **Zero-copy / mapped memory:** GPU directly accesses host memory over PCIe without explicit copies; useful for data that is read once or when GPU memory is insufficient, but bandwidth-limited by PCIe
- **Related concepts:** [[dma-transfers]], [[unified-memory]]

---

## Summary Table

| Level | Scope | Typical Capacity | Approx. Bandwidth | Approx. Latency |
| --- | --- | --- | --- | --- |
| Registers | Per-thread | ~256–512 KB / SM | Highest (on-chip) | ~1 cycle |
| Shared Memory / L1 | Per-thread-block | 48–228 KB / SM | ~100+ TB/s aggregate | ~20–40 cycles |
| L2 Cache | GPU-wide | 4–60 MB | ~5–12 TB/s | ~200–400 cycles |
| Device Memory (HBM/GDDR) | GPU-wide | 8–192 GB | 900 GB/s – 3.35 TB/s | ~600–800 cycles |
| Peer GPU Memory (NVLink) | Multi-GPU | Remote GPU capacity | Up to 900 GB/s per link | µs range |
| Host Memory (PCIe) | System-wide | Hundreds of GB | ~32–64 GB/s | µs – tens of µs |

---

## Key Design Principles

### Coalesced Memory Access

Global memory transactions are served in 128-byte cache lines. When threads in a warp access consecutive addresses, all accesses are satisfied by a single (or minimal number of) transaction(s). Strided or random access patterns require multiple transactions, reducing effective bandwidth. Kernel layouts should be designed so that thread index maps linearly to memory address wherever possible.

### Occupancy and the Latency Hiding Model

GPUs hide memory latency by context-switching between warps. Higher occupancy (more active warps per SM) gives the warp scheduler more candidates to switch to while waiting for memory. However, occupancy is constrained by shared memory usage and register count per thread. The [[roofline-model]] provides a framework for reasoning about whether a kernel is compute-bound or memory-bandwidth-bound.

### Memory Access Patterns and Caching

- **Streaming (non-temporal):** Data that will not be reused should bypass L2 using cache bypass hints (`__cacheop` intrinsics or PTX `.cs` / `.cg` modifiers) to avoid polluting the cache.
- **Reuse-friendly:** Data accessed multiple times per kernel should be explicitly staged in shared memory to exploit spatial and temporal locality.
- **Read-only data:** Marking kernel pointer arguments as `const __restrict__` allows the compiler to route accesses through the read-only data cache (texture cache path), which is separate from L1/L2 and can improve throughput for broadcast patterns.

### Avoiding Shared Memory Bank Conflicts

Shared memory is organized into 32 banks of 4-byte (or 8-byte on some configs) width. Accesses from threads within a warp to the same bank are serialized unless all threads access the same address (broadcast). Common mitigation: pad array dimensions by one element (e.g., `__shared__ float tile[32][33]`).

### Tiling Strategy

Explicit tiling into shared memory is the canonical technique for reuse-heavy kernels such as matrix multiplication (GEMM). The standard pattern is:

1. Each thread block loads a tile of A and a tile of B from global memory into shared memory using coalesced reads.
2. All threads in the block compute partial dot products from the shared memory tiles.
3. The block advances through the K-dimension one tile at a time, accumulating into a register-resident accumulator.
4. The final result is written back to global memory.

This pattern reduces global memory traffic by a factor proportional to the tile size, replacing high-latency global memory accesses with low-latency shared memory accesses. See [[tensor-core-utilization]] for how this pattern extends to Tensor Core (WMMA / warpgroup MMA) usage in CUTLASS and cuBLAS.

### Unified Memory Caveats

Unified Memory simplifies programming but can introduce unexpected page-fault overhead. Best practices:

1. Use `cudaMemPrefetchAsync` to move pages to the GPU before kernels launch.
2. Use `cudaMemAdvise` to set read-mostly or preferred-location hints.
3. Avoid CPU and GPU concurrently accessing the same pages without explicit synchronization.
4. On systems without hardware page migration support (pre-Pascal or non-NVLink configurations), UM falls back to full device synchronization on page faults, which can severely degrade performance.

### Asynchronous Data Movement and Compute-Transfer Overlap

CUDA streams allow concurrent execution of memory copies and kernels on separate engines. The GPU exposes dedicated copy engines (typically one host-to-device and one device-to-host) that operate independently of the SM compute pipeline. Best practices:

1. Allocate pinned memory for all host buffers involved in async transfers.
2. Use at least two streams with double-buffered staging to keep the copy engines and SMs busy simultaneously.
3. Use CUDA events (`cudaEventRecord`, `cudaStreamWaitEvent`) to express precise ordering constraints between streams without blocking the CPU.
4. On Hopper and later, the `cudaMemcpyAsync` path can be further accelerated by the hardware DMA engine (TMA — Tensor Memory Accelerator) for bulk tile copies directly into shared memory, bypassing L2.

---

## Architecture-Specific Notes

### NVIDIA Volta (V100)

- 16 GB or 32 GB HBM2
- 6 MB L2 cache
- Up to 96 KB shared memory per SM (configurable vs. L1)
- NVLink 2.0: 300 GB/s bidirectional per GPU
- First architecture with independent thread scheduling (threads within a warp can diverge at finer granularity than previous SIMT model)
- First architecture with Tensor Cores (FP16 matrix multiply-accumulate in FP32)

### NVIDIA Turing (RTX 20 series / T4)

- GDDR6 (consumer) or 16 GB GDDR6 (T4)
- 4–6 MB L2 cache
- Up to 64 KB shared memory per SM
- No NVLink on consumer cards; NVLink 2.0 on Quadro variants
- Added INT8 and INT4 Tensor Core modes for inference workloads

### NVIDIA Ampere (A100)

- 40 MB or 80 MB HBM2e
- 40 MB L2 cache (significantly larger than Volta's 6 MB; substantially improves reuse for large batch workloads)
- Up to 164 KB shared memory per SM
- NVLink 3.0: 600 GB/s bidirectional per GPU
- Multi-Instance GPU (MIG): hardware partitioning of GPC slices, memory, and cache into up to 7 isolated GPU instances
- Asynchronous memory copy intrinsics (`cp.async`) allow direct global-to-shared memory DMA from within the SM without staging through registers, improving pipeline efficiency

### NVIDIA Hopper (H100)

- 80 GB HBM3 (SXM variant); 80 GB HBM2e (PCIe variant)
- 50 MB L2 cache
- Up to 228 KB shared memory per SM
- NVLink 4.0: 900 GB/s bidirectional per GPU; NVSwitch 3.0 in DGX H100
- Thread Block Clusters: programming model abstraction grouping up to 8 thread blocks into a cluster; blocks within a cluster can access each other's shared memory via Distributed Shared Memory (DSM) using `cluster.sync()` and explicit `mapa` / `ld.shared::cluster` instructions
- Tensor Memory Accelerator (TMA): hardware unit that performs bulk async copies between global memory and shared memory with descriptor-based addressing, decoupling data movement from compute warp scheduling
- FP8 Tensor Core support for further inference throughput scaling
- Confidential Computing support with hardware memory encryption

### NVIDIA H200

- 141 GB HBM3e
- Memory bandwidth: 3.35 TB/s (vs. 3.35 TB/s on H100 SXM5 with HBM3; H200 increases capacity more than peak bandwidth relative to H100 SXM5)
- Otherwise identical to H100 SXM5 in SM architecture, cache hierarchy, and NVLink configuration
- Primarily benefits workloads that are capacity-bound or bandwidth-bound on large models (e.g., LLM inference with large KV caches)

### AMD CDNA2 (MI250X)

- 128 GB HBM2e across two dies (64 GB each), connected via Infinity Fabric
- 8 MB L2 cache per die (16 MB total)
- Infinity Fabric (xGMI): up to 800 GB/s peer bandwidth between dies; up to 400 GB/s between separate GPUs
- 64 KB LDS (Local Data Share) per CU — functionally equivalent to NVIDIA shared memory
- 16 KB L1 vector cache per CU
- ROCm programming model mirrors CUDA: LDS maps to shared memory, wavefronts (64 threads) map to warps

### AMD CDNA3 (MI300X)

- 192 GB HBM3 across 8 stacks
- Memory bandwidth: ~5.3 TB/s
- Unified CPU+GPU die package; 13 chiplets (3 CPU dies + 8 GPU compute dies + 4 IO dies)
- 256 MB Infinity Cache (L3 die cache shared across all compute dies)
- Targets LLM inference and training workloads where model size exceeds single-GPU HBM capacity of competing products

---

## Common Bottlenecks and Diagnostics

| Symptom | Likely Cause | Mitigation |
| --- | --- | --- |
| Low memory bandwidth utilization | Uncoalesced global memory access | Restructure data layout; transpose tiles via shared memory |
| High shared memory bank conflicts | Strided shared memory access pattern | Pad shared memory arrays to avoid bank conflicts |
