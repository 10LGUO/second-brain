```yaml
title: AI Chip Architecture
type: concept
tags: [gpu, chip, hardware, hbm, cuda-core, tensor-core, memory, nvlink, dma]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# AI Chip Architecture

AI chip architecture describes the internal hardware organization of accelerators (primarily GPUs) used for training and inference of large models. It spans compute cores, on-chip cache hierarchy, off-chip memory (VRAM), and inter-chip interconnects. Understanding this architecture is foundational to all performance optimization work in LLM infrastructure.

## Key Components

### Compute Cores

Modern AI accelerators contain two broad classes of execution units:

- **CUDA Cores (FP32 / FP64 ALUs):** General-purpose floating-point and integer execution units. Each streaming multiprocessor (SM) contains many CUDA cores. Used for element-wise ops, activation functions, and any work that does not map cleanly to matrix multiply.
- **Tensor Cores:** Specialized matrix-multiply-accumulate (MMA) units introduced in NVIDIA Volta (V100). Operate on small tiles (e.g., 16×16×16) in mixed precision (FP16×FP16→FP32, BF16, TF32, INT8, FP8). A single Tensor Core operation completes in one clock cycle what would take many CUDA Core cycles. The overwhelming majority of FLOPS in transformer workloads flow through Tensor Cores.

Each SM is a self-contained compute cluster that includes both core types, a warp scheduler, register file, and L1/shared memory. The full chip is composed of tens to hundreds of SMs.

### Memory Hierarchy

Memory access latency and bandwidth are the primary bottlenecks for large model workloads. The hierarchy from fastest/smallest to slowest/largest:

| Level | Location | Capacity (H100 SXM) | Bandwidth |
| --- | --- | --- | --- |
| Registers | Per thread, on SM | ~256 KB per SM | ~100s TB/s effective |
| L1 / Shared Memory | Per SM, configurable split | 228 KB per SM | ~33 TB/s |
| L2 Cache | On-die, shared across SMs | 50 MB | ~12 TB/s |
| HBM (VRAM) | Off-chip, on package | 80 GB (HBM3) | ~3.35 TB/s |
| NVLink / NVSwitch | Inter-GPU | — | ~900 GB/s (H100 NVLink 4.0) |
| PCIe / Host DRAM | CPU-GPU link | — | ~64 GB/s (PCIe 5.0 ×16) |

The ratio between compute throughput (FLOPS) and memory bandwidth defines the **arithmetic intensity** ceiling. Operations below this ceiling are **memory-bound**; above it they are **compute-bound**.

### HBM (High Bandwidth Memory)

HBM stacks DRAM dies vertically and connects them to the GPU die via a wide interposer. Key properties:

- **Wide bus:** 1024–8192-bit interface vs. 256–384-bit for GDDR.
- **High bandwidth:** HBM3 delivers ~3.35 TB/s on the H100, vs. ~1 TB/s for GDDR6X.
- **Lower power per bit** than GDDR due to shorter signal paths.
- **Limited capacity:** Stacking constrains total VRAM (typically 40–192 GB), which is the binding constraint for maximum model size without offloading or parallelism.

HBM capacity and bandwidth are often the first-order constraints when fitting and serving large models. See [[vram-management]] and [[kv-cache]].

### On-Chip SRAM and Shared Memory

L1 cache and shared memory occupy the same physical SRAM bank per SM, with a software-configurable split. Shared memory is explicitly managed by the programmer (or compiler) and is crucial for:

- **Tiling:** Loading a tile of a matrix into shared memory so threads reuse data many times before evicting.
- **Reductions:** Warp-level and block-level reductions used in softmax, layer norm, etc.
- **Avoiding bank conflicts:** Careful padding and access patterns prevent serialization of reads.

Efficient use of shared memory is the central technique in high-performance CUDA kernels and the foundational insight of [[flash-attention]].

### Warp Execution and SIMT

GPUs execute threads in groups of 32 called **warps**. All threads in a warp execute the same instruction each cycle (SIMT — Single Instruction, Multiple Thread). Implications:

- **Divergence:** If threads in a warp take different branches, both paths execute serially with masking, halving (or worse) effective throughput.
- **Occupancy:** The SM scheduler hides memory latency by switching between resident warps. Higher occupancy (more active warps per SM) enables better latency hiding. Occupancy is limited by register and shared memory pressure.
- **Memory coalescing:** For maximum bandwidth, threads in a warp should access contiguous memory addresses so accesses merge into one transaction.

### Tensor Core Operation Model

Tensor Cores perform a **D = A × B + C** operation on small matrix tiles in a single fused step. Software exposes these through:

- **PTX / WMMA API:** Low-level warp-level matrix multiply intrinsics.
- **CUTLASS:** NVIDIA's open-source library of high-performance GEMM templates.
- **cuBLAS / cuDNN:** Vendor-optimized libraries that select optimal Tensor Core kernels automatically.
- **Triton / custom kernels:** Allow researchers to write Tensor Core-utilizing kernels with higher-level abstractions.

To reach peak Tensor Core utilization, matrix dimensions must align with tile sizes (multiples of 8, 16, or 64 depending on precision), and data must be in the correct layout (row-major vs. column-major).

### DMA and Memory Transfer Engines

The **DMA (Direct Memory Access)** engine moves data between host DRAM and device HBM asynchronously, without stalling compute SMs. Key points:

- Transfers are initiated via `cudaMemcpyAsync` and overlap with kernel execution using CUDA streams.
- **Copy engines** are separate hardware units on the GPU; the H100 has dedicated copy engines for bidirectional host↔device and peer transfers.
- Effective pipeline design stages data transfers to hide PCIe latency behind compute. This is critical for inference serving where prompt tokens or weights may be paged.

### NVLink and NVSwitch

For multi-GPU systems, NVIDIA provides high-bandwidth interconnects beyond PCIe:

- **NVLink:** Point-to-point links between GPUs. H100 NVLink 4.0 provides 900 GB/s total bidirectional bandwidth between two GPUs.
- **NVSwitch:** A crossbar switch chip that allows any GPU in a node to communicate with any other at full NVLink bandwidth. An 8-GPU DGX H100 node uses NVSwitch to form an all-to-all, non-blocking fabric.
- Used to implement **tensor parallelism** and **pipeline parallelism** efficiently within a node, where collective operations (AllReduce, AllGather) are the bottleneck. See [[model-parallelism]].

### PCIe and Host Interface

When NVLink is unavailable (consumer or cloud multi-GPU setups), GPUs communicate via PCIe:

- PCIe 4.0 ×16: ~32 GB/s; PCIe 5.0 ×16: ~64 GB/s (bidirectional ~128 GB/s).
- Much slower than NVLink; tensor parallelism across PCIe-connected GPUs incurs significant overhead.
- CPU↔GPU transfers for data loading, gradient checkpointing restores, and KV cache paging all share this bus.

---

## Arithmetic Intensity and the Roofline Model

The **roofline model** characterizes whether a kernel is limited by compute or memory bandwidth:

```text
Achievable FLOPS/s = min(Peak FLOPS/s, Bandwidth × Arithmetic Intensity)
```

- **Arithmetic intensity** = FLOPs performed / bytes read from memory.
- Large GEMMs (e.g., large batch matmuls) have high arithmetic intensity → compute-bound → Tensor Cores dominate.
- Attention with small batch sizes, element-wise ops, layer norm → low arithmetic intensity → memory-bound → HBM bandwidth dominates.
- Kernel fusion (e.g., [[flash-attention]]) raises effective arithmetic intensity by keeping intermediate values in SRAM rather than writing to HBM.

### Worked Example: GEMM vs. Softmax

For a GEMM with dimensions M=4096, N=4096, K=4096 in BF16:

- FLOPs: 2 × 4096³ ≈ 137 GFLOPs
- Bytes read (weights + activations): ~134 MB
- Arithmetic intensity: ~1024 FLOPs/byte → compute-bound on H100

For a softmax over a 4096-length sequence:

- FLOPs: ~3 × 4096 ≈ 12K FLOPs (exp, sum, divide)
- Bytes read/written: 2 × 4096 × 2 bytes ≈ 16 KB
- Arithmetic intensity: ~0.75 FLOPs/byte → heavily memory-bound

This contrast motivates fusing softmax with adjacent operations and keeping intermediates on-chip whenever possible.

---

## Precision Formats

| Format | Bits | Notes |
| --- | --- | --- |
| FP64 | 64 | Scientific computing; rarely used in DL |
| FP32 | 32 | Standard training accumulator |
| TF32 | 19 | NVIDIA's "fast FP32" on Tensor Cores; same exponent range, reduced mantissa |
| BF16 | 16 | Wide dynamic range (same as FP32 exponent); preferred for training |
| FP16 | 16 | Smaller dynamic range; requires loss scaling; common for inference |
| INT8 | 8 | Quantized inference; requires calibration |
| FP8 (E4M3/E5M2) | 8 | H100+ training and inference; two variants trade range vs. precision |
| FP4 | 4 | Blackwell (B100/B200) generation; extreme throughput, aggressive quantization |

Mixed-precision training typically keeps a master copy of weights in FP32 and performs forward/backward passes in BF16 or FP16. See [[quantization]] and [[mixed-precision-training]].

### Precision and Throughput Scaling

NVIDIA Tensor Core throughput roughly doubles with each halving of precision on the same generation of hardware:

- H100 SXM: FP32 ~67 TFLOPS, TF32 ~989 TFLOPS, BF16/FP16 ~1979 TFLOPS, FP8 ~3958 TFLOPS (dense)
- Sparsity (2:4 structured sparsity) doubles effective throughput again at the cost of requiring sparse weight formats

---

## Streaming Multiprocessor (SM) Deep Dive

Each SM on a modern NVIDIA GPU contains:

- **Warp schedulers (4 per SM on Hopper):** Issue instructions to warps each cycle. Multiple schedulers allow issuing to different warps in the same cycle, hiding pipeline latency.
- **Register file:** ~65,536 32-bit registers per SM, partitioned among resident warps and threads. Register pressure is the most common cause of reduced occupancy.
- **Tensor Core units:** 4 sets of Tensor Cores per SM on Hopper (one per warp scheduler quadrant). Each processes one warp-level MMA operation per cycle.
- **CUDA Core ALUs:** 128 FP32 CUDA Cores per SM on H100, used for non-GEMM compute.
- **Special Function Units (SFUs):** Handle transcendental ops (sin, cos, exp, rsqrt) at lower throughput. A bottleneck for activation functions like GELU if not approximated.
- **Load/Store Units (LSUs):** Move data between register file, shared memory, and L1/L2/HBM.
- **L1 Cache / Shared Memory:** 228 KB unified SRAM; programmer-controlled split between L1 cache and explicitly managed shared memory.

### Occupancy and Latency Hiding

GPU throughput depends on keeping the warp schedulers busy. When one warp stalls waiting for memory, the scheduler switches to another resident warp immediately (zero-cost context switch). The number of warps that can reside simultaneously is bounded by:

- **Register file capacity:** More registers per thread → fewer resident threads.
- **Shared memory capacity:** Larger shared memory allocations per block → fewer concurrent blocks.
- **Maximum warps per SM:** H100 supports up to 64 warps (2048 threads) per SM.

Achieving high occupancy is not always optimal — sometimes using more registers or shared memory per thread enables algorithmic improvements (e.g., larger tiles) that outweigh reduced occupancy.

---

## Multi-GPU Topology and Scaling

### Within a Node

An 8-GPU DGX H100 server provides:

- 8× H100 SXM GPUs, each with 80 GB HBM3
- 4× NVSwitch 3.0 chips creating a fully non-blocking NVLink fabric
- 900 GB/s NVLink bandwidth per GPU
- All-to-all AllReduce at ~450 GB/s effective bisection bandwidth

This topology is ideal for tensor parallelism (TP) across all 8 GPUs, with AllReduce operations completing in microseconds rather than milliseconds.

### Across Nodes

Multi-node training requires a network fabric:

- **InfiniBand (HDR/NDR):** 200–400 Gb/s per link; used in high-end clusters (e.g., NVIDIA DGX SuperPOD, AWS P4/P5).
- **RoCE (RDMA over Converged Ethernet):** Lower-cost alternative; latency slightly higher than IB.
- **EFA (Elastic Fabric Adapter):** AWS's custom fabric for P3/P4/P5 instances.

Inter-node bandwidth is 10–100× lower than intra-node NVLink bandwidth, fundamentally shaping parallelism strategy: pipeline parallelism (PP) is preferred across nodes (fewer, larger messages) while tensor parallelism is preferred within nodes (frequent small messages). See [[model-parallelism]].

### Topology-Aware Collective Operations

NCCL (NVIDIA Collective Communications Library) automatically detects the NVLink/PCIe/IB topology and selects algorithms (ring, tree, all-to-all) to maximize bandwidth utilization. Poor topology awareness can leave significant bandwidth on the table.

---

## The Transformer Engine (Hopper+)

The H100 introduced the **Transformer Engine**, a hardware/software co-design feature targeting transformer workloads specifically:

- **FP8 mixed precision:** The Transformer Engine dynamically selects FP8 (E4M3 for forward pass, E5M2 for gradient) per-tensor, computing scaling factors on the fly to maintain accuracy.
- **Per-tensor scaling:** Avoids the static calibration required for INT8 quantization; scaling factors are updated each step.
- **Hardware acceleration:** Dedicated hardware paths for FP8 GEMM accumulating into FP32/BF16.
- **Integration with frameworks:** PyTorch `torch.float8` and the `transformer-engine` library expose this to model code with minimal changes.

The Transformer Engine effectively doubles throughput for transformer forward passes on H100 vs. BF16 without requiring offline calibration, making FP8 training practical at scale.

---

## Generational Overview (NVIDIA)

| GPU | Arch | Tensor Core Gen | HBM | Peak BF16 TFLOPS | NVLink | Key Advance |
| --- | --- | --- | --- | --- | --- | --- |
| V100 | Volta | 1st gen | HBM2, 900 GB/s | 125 | NVLink 2.0 (300 GB/s) | First Tensor Cores |
| A100 | Ampere | 3rd gen | HBM2e, 2 TB/s | 312 | NVLink 3.0 (600 GB/s) | TF32, sparsity, MIG |
| H100 | Hopper | 4th gen | HBM3, 3.35 TB/s | 1,979 | NVLink 4.0 (900 GB/s) | FP8, Transformer Engine |
