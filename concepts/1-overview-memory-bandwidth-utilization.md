```yaml
title: Memory Bandwidth Utilization
type: concept
tags: [performance, memory, bandwidth, gpu, hardware, optimization]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# Memory Bandwidth Utilization

**Memory bandwidth utilization** (访存利用率) measures how efficiently a chip uses its available off-chip memory (HBM) bandwidth. It is a critical performance metric in AI inference and training, complementary to compute utilization ([[mfu-model-flops-utilization]]).

## Key Properties

- **Definition**: The ratio of observed (effective) memory bandwidth to the peak theoretical memory bandwidth of the hardware.
  - `MBU = (Effective Bandwidth Used) / (Peak Memory Bandwidth)`
  - Expressed as a percentage; e.g., 80% MBU on an A100 (2 TB/s peak) means ~1.6 TB/s is being used.
- **Off-chip vs. on-chip**: MBU typically refers to off-chip DRAM (HBM) bandwidth. On-chip SRAM (shared memory, L1/L2 cache) accesses are not counted, as they are orders of magnitude faster and not the usual bottleneck.
- **Memory-bound vs. compute-bound**: Operations are classified based on their arithmetic intensity (FLOPs per byte of memory access):
  - **Memory-bound**: Low arithmetic intensity — the chip is waiting on memory, not compute. High MBU, low MFU. Examples: element-wise ops, [[layer-normalization]], autoregressive decoding.
  - **Compute-bound**: High arithmetic intensity — the chip is saturating compute units. Low MBU, high MFU. Examples: large matrix multiplications in training.
- **Arithmetic intensity**: The key quantity determining which regime an operation falls in. Defined as FLOPs / bytes of memory traffic. The "roofline" crossover point separating memory-bound from compute-bound is: `Peak FLOPs / Peak Bandwidth` (measured in FLOPs/byte).

## Why It Matters

In modern AI workloads, **inference is often memory-bandwidth-bound**, especially during:

- **Autoregressive generation** in large language models ([[transformer-architecture]]): each decode step loads the full model weights once per token generated but performs relatively few FLOPs.
- **KV-cache access**: loading cached key/value tensors from HBM during attention.
- Small batch sizes, which do not amortize memory loads across enough parallel computation.

Because weights must be loaded from HBM on every forward pass, a model with billions of parameters quickly saturates bandwidth even at modest throughput targets. This makes MBU the primary limiter of tokens-per-second in many production LLM serving scenarios.

## Relationship to MFU

MBU and [[mfu-model-flops-utilization]] are complementary but typically inversely dominant:

| Regime | MFU | MBU | Typical scenario |
| --- | --- | --- | --- |
| Memory-bound | Low | High | Inference, small batch |
| Compute-bound | High | Low | Training, large batch matmuls |
| Balanced | Moderate | Moderate | Carefully tuned kernels |

A fully efficient system would simultaneously maximize both, but hardware roofline constraints make this impossible except at the exact crossover arithmetic intensity.

## Roofline Model

The **roofline model** provides a visual and analytical framework for understanding MBU and MFU together:

- The achievable performance (FLOPs/s) is bounded by the minimum of:
  1. Peak compute: `P_compute` (FLOPs/s)
  2. Memory-bandwidth-limited compute: `Arithmetic Intensity × Peak Bandwidth`
- Operations to the left of the ridge point are memory-bound; those to the right are compute-bound.
- Kernel optimization aims to push operations toward or beyond the ridge point.
- The roofline ceiling is a hard physical limit; no amount of software optimization can exceed peak bandwidth or peak compute independently.
- Multiple roofline ceilings can be modeled (e.g., L2 cache bandwidth vs. HBM bandwidth) to capture hierarchical memory effects.

## Practical Targets

- Well-optimized memory-bound kernels on modern GPUs (A100, H100) achieve **60–85% MBU**.
- Poorly optimized kernels (e.g., naive PyTorch ops with excessive kernel launches, poor fusion) may achieve only 10–30% MBU.
- [[flash-attention]] is a key example of a technique that dramatically improves effective MBU by reducing HBM reads/writes through tiling and on-chip reuse.
- In production LLM serving, end-to-end MBU at the system level is typically lower than kernel-level MBU due to scheduling overhead, CPU-GPU synchronization, and tokenization latency.

## Optimization Techniques

Several techniques improve memory bandwidth utilization or reduce the bandwidth required:

- **Operator fusion**: Combining multiple elementwise or memory-bound ops into a single kernel reduces the number of HBM round-trips. See [[kernel-fusion]].
- **[[flash-attention]]**: Rewrites the attention computation to tile the operation across SRAM, avoiding materialization of large intermediate attention matrices in HBM. FlashAttention-2 and FlashAttention-3 extend this with improved parallelism and warp specialization.
- **Quantization** ([[model-quantization]]): Reducing weight precision (e.g., FP16 → INT8 → INT4) directly reduces bytes loaded per parameter, effectively multiplying available bandwidth by the compression ratio. Weight-only quantization (e.g., GPTQ, AWQ) is particularly impactful for memory-bound inference since activations remain in higher precision while weights are dequantized on the fly.
- **Speculative decoding**: Increases arithmetic intensity per decode step by batching candidate tokens, reducing the memory-bound nature of autoregressive generation. See [[speculative-decoding]].
- **Continuous batching**: Aggregates requests to increase effective batch size, improving compute-to-memory ratio and amortizing weight loads across more concurrent users.
- **Weight streaming / prefetching**: Overlapping HBM reads with computation to hide memory latency, particularly relevant on hardware with deep memory pipelines.
- **Tensor parallelism**: Distributing weights across multiple devices, so each device loads a smaller fraction of weights per step; effective bandwidth scales with device count for weight-load-bound scenarios.
- **Paged attention**: Manages KV-cache memory in non-contiguous pages (as in [[vllm]]), reducing memory fragmentation and enabling higher effective batch sizes, indirectly improving overall bandwidth utilization.
- **Activation checkpointing**: During training, trades recomputation for reduced memory bandwidth pressure on large activation tensors; can shift bottlenecks between compute and memory.

## Hardware Context

| Hardware | Peak HBM Bandwidth | Peak FP16 Compute | Ridge Point |
| --- | --- | --- | --- |
| NVIDIA A100 (80 GB SXM) | ~2.0 TB/s | ~312 TFLOP/s | ~156 FLOPs/byte |
| NVIDIA H100 SXM | ~3.35 TB/s | ~989 TFLOP/s (FP16 TC) | ~295 FLOPs/byte |
| NVIDIA H100 NVL | ~3.9 TB/s | ~989 TFLOP/s | ~254 FLOPs/byte |
| NVIDIA H200 SXM | ~4.8 TB/s | ~989 TFLOP/s | ~206 FLOPs/byte |
| Google TPU v4 | ~1.2 TB/s | ~275 TFLOP/s | ~229 FLOPs/byte |
| Google TPU v5e | ~1.6 TB/s | ~393 TFLOP/s | ~246 FLOPs/byte |
| AMD MI300X | ~5.3 TB/s | ~1,307 TFLOP/s (FP16) | ~247 FLOPs/byte |

Higher ridge points (H100 vs. A100) mean a larger class of operations remain memory-bound, increasing the importance of bandwidth optimization as compute scales faster than memory bandwidth. Notably, the H200 improves over the H100 primarily through higher HBM3e bandwidth rather than additional compute, explicitly targeting memory-bound inference workloads.

## Measurement

MBU can be measured using:

- **Hardware performance counters**: Tools like NVIDIA Nsight Compute report HBM read/write throughput directly, broken down by kernel. Key metrics: `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` and related counters for global memory loads/stores.
- **Empirical estimation**: Profiling kernel wall time and computing effective bandwidth from known tensor sizes.
  - `Effective BW = (bytes_read + bytes_written) / kernel_time_seconds`
  - This requires accurate accounting of all tensors read and written, including intermediate buffers.
- **Model-level estimation**: For a known model and batch configuration, theoretical minimum bytes accessed can be computed from parameter counts and activation sizes, then compared against observed throughput.
  - For autoregressive decoding: minimum bytes ≈ `2 × num_parameters × bytes_per_param` (read weights once) plus KV-cache volume.
- **roofline profiling tools**: NVIDIA's Nsight Systems and Nsight Compute together provide roofline charts placing each kernel on the compute vs. memory-bound spectrum automatically.
- **PyTorch Profiler**: Can expose kernel-level timing and, with CUPTI integration, hardware counters for bandwidth estimation within Python-native workflows.

## Common Pitfalls

- **Conflating peak and sustained bandwidth**: Manufacturers quote peak burst bandwidth; sustained bandwidth under realistic access patterns (non-sequential, strided, or scattered) is often 10–20% lower.
- **Ignoring cache effects**: A kernel that fits in L2 cache may appear to have very high "effective HBM bandwidth" when measured naively, because the actual HBM traffic is low — which is a good thing, not a measurement of HBU.
- **Batch size dependence**: MBU is not a fixed property of a model but varies with batch size, sequence length, and hardware configuration. Reported numbers should always state these conditions.
- **Double-counting in fused kernels**: When operators are fused, the intermediate tensor reads/writes that would have occurred in separate kernels do not hit HBM; naive byte-counting of logical operations will overestimate actual memory traffic.

## Variants and Related Metrics

- **DRAM bandwidth utilization**: Equivalent term used outside GPU contexts (e.g., CPU inference, edge devices).
- **Cache hit rate / L2 utilization**: On-chip bandwidth metrics; high L2 reuse reduces HBM pressure and can make a kernel appear more compute-bound than its logical arithmetic intensity suggests.
- **Memory-bound fraction** (Amdahl-style): The fraction of total inference time dominated by memory-bound operations, useful for estimating the ceiling benefit of quantization or fusion across a full pipeline.
- **Bytes per parameter per forward pass**: A derived metric useful for comparing model architectures on memory efficiency; for a standard dense transformer decode step with batch size 1, this is approximately 2 bytes/param in FP16.
- **Memory bandwidth efficiency (MBE)**: Sometimes used interchangeably with MBU; occasionally defined more narrowly as bandwidth utilized relative to bandwidth theoretically required by the algorithm (not the hardware peak), making it a measure of algorithmic efficiency rather than hardware utilization.
- **Bandwidth-compute balance ratio**: The ratio of a model's operational arithmetic intensity to the hardware ridge point, indicating how far into the memory-bound regime a given workload falls.

## Worked Example: LLM Decode Step

For a 70B parameter model in FP16 (2 bytes/param = 140 GB of weights) running on a single H100 (3.35 TB/s peak):

- **Minimum time per token** (batch size 1, ignoring KV-cache): `140 GB / 3,350 GB/s ≈ 42 ms/token` → ~24 tokens/second maximum.
- At 80% MBU: effective rate ≈ `0.8 × 3,350 = 2,680 GB/s` → `140 / 2,680 ≈ 52 ms/token` → ~19 tokens/second.
- With INT4 quantization (0.5 bytes/param effective): weight load drops to 35 GB → `35 / 2,680 ≈ 13 ms/token` → ~77 tokens/second at the same MBU.

This illustrates why quantization has an outsized impact on memory-bound inference: it directly multiplies the effective bandwidth without requiring hardware changes.

## Related Concepts

- [[mfu-model-flops-utilization]]
- [[flash-attention]]
- [[kernel-fusion]]
- [[model-quantization]]
- [[transformer-architecture]]
- [[arithmetic-intensity]]
- [[roofline-model]]
- [[kv-cache]]
- [[speculative-decoding]]
- [[layer-normalization]]
- [[vllm]]
- [[continuous-batching]]
- [[tensor-parallelism]]

## Sources

- [[1-overview.md]]
