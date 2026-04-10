```yaml
title: Large Model Infra (AI Infrastructure for Large Models)
type: concept
tags: [ai-infra, large-models, training, inference, distributed-systems, optimization]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# Large Model Infra (AI Infrastructure for Large Models)

Large model infra refers to the engineering discipline of optimizing the **compute, storage, communication, performance, and precision** of large-scale AI model training and inference systems. Its primary goal is to reduce the cost of training and serving large language models (LLMs) and other large generative models by maximizing hardware utilization, minimizing memory usage, and ensuring numerical correctness at scale.

## Key Properties

- **Scale**: Systems must handle models with billions to trillions of parameters, often spread across thousands of accelerators (GPUs, TPUs, or custom ASICs).
- **Hardware utilization**: Achieving high MFU (Model FLOP Utilization) is a primary KPI — wasted compute cycles directly translate to cost.
- **Memory efficiency**: GPU HBM (High Bandwidth Memory) is scarce and expensive; techniques to reduce memory footprint are central to the discipline.
- **Communication efficiency**: At scale, inter-device bandwidth (NVLink, InfiniBand, RoCE) is a frequent bottleneck; minimizing communication overhead is critical.
- **Numerical stability**: Mixed-precision and low-precision arithmetic introduce rounding errors that can destabilize training; infra must ensure correctness.
- **Fault tolerance**: Long training runs spanning days or weeks must recover gracefully from hardware failures without losing significant progress.
- **Reproducibility**: Stochastic parallelism can introduce non-determinism; infra must allow debugging and reproducible experimentation.

## Core Problem Areas

### 1. Parallelism Strategies

Large models do not fit on a single device, requiring decomposition across many devices. The main strategies are:

- **Data Parallelism (DP)**: Each device holds a full copy of the model; batches are split across devices and gradients are averaged. Simple but requires the model to fit on one device.
- **Tensor Parallelism (TP)**: Individual weight matrices are sharded across devices (e.g., splitting attention heads). Enables fitting larger models per layer but requires frequent all-reduce communication within layers.
- **Pipeline Parallelism (PP)**: Different layers are assigned to different devices; micro-batches flow through the pipeline. Reduces memory per device but introduces pipeline bubbles (idle time).
- **Sequence Parallelism (SP)**: The sequence dimension of activations is sharded across devices, complementing tensor parallelism by reducing activation memory.
- **Expert Parallelism**: For Mixture-of-Experts (MoE) models, different experts reside on different devices; routing logic dispatches tokens to the appropriate device.
- **Fully Sharded Data Parallelism (FSDP / ZeRO)**: Model parameters, gradients, and optimizer states are sharded across data-parallel ranks; each rank only holds a shard and gathers parameters on demand. Pioneered by DeepSpeed's ZeRO optimizer and adopted in PyTorch FSDP.

Practical systems combine these strategies (e.g., TP + PP + DP, sometimes called "3D parallelism") to balance memory, communication, and compute.

### 2. Memory Optimization

GPU HBM is often the binding constraint. Key techniques include:

- **Gradient checkpointing (activation recomputation)**: Discard intermediate activations during the forward pass and recompute them during backprop, trading compute for memory.
- **Mixed precision training**: Store weights in FP16 or BF16 but maintain a master copy in FP32 for optimizer updates (loss scaling handles underflow in FP16).
- **Optimizer state offloading**: Move optimizer states (e.g., Adam's first and second moments) to CPU RAM or NVMe, freeing GPU memory at the cost of PCIe bandwidth.
- **Paged attention / KV cache management**: During inference, KV caches grow with sequence length; paged memory allocation (as in vLLM's PagedAttention) avoids fragmentation and enables higher batch sizes.
- **Weight tying and embedding sharing**: Reduce parameter count by sharing weights between input embeddings and output projection.
- **Activation offloading**: Temporarily move activations to CPU memory during the forward pass and stream them back during the backward pass, providing an additional memory–bandwidth trade-off on top of gradient checkpointing.
- **Memory pooling and fragmentation reduction**: Custom allocators (e.g., PyTorch's caching allocator, or arena-based schemes) reduce fragmentation and allocation overhead in long-running workloads.

### 3. Compute Optimization

- **Kernel fusion**: Fusing multiple elementwise operations into a single GPU kernel reduces memory bandwidth pressure and kernel launch overhead. Tools: Triton, CUDA custom kernels, torch.compile.
- **FlashAttention**: A memory-efficient, IO-aware exact attention algorithm that tiles computation to avoid materializing the full attention matrix in HBM, dramatically reducing memory usage and improving throughput for long sequences. FlashAttention-2 and FlashAttention-3 further improve parallelism and hardware utilization.
- **Operator scheduling and graph compilation**: Frameworks like XLA (used in JAX/TPU), torch.compile (TorchDynamo + TorchInductor), and TensorRT optimize computation graphs by fusing, reordering, and eliminating redundant operations.
- **Overlapping compute and communication**: Pipelining gradient all-reduces with the backward pass (e.g., bucketing gradients in DDP) hides communication latency behind compute.
- **Continuous batching**: In inference serving, rather than waiting for a full batch, new requests are inserted into ongoing forward passes as slots free up, maximizing GPU utilization (pioneered by Orca/vLLM).
- **Warp-level and tensor core utilization**: Writing kernels that maximize use of NVIDIA Tensor Cores (or AMD Matrix Cores) requires careful tile sizing, data layout (row-major vs. column-major), and warp scheduling to avoid stalls.
- **Microbatch scheduling in pipelines**: Techniques such as 1F1B (one-forward-one-backward) scheduling and interleaved pipeline stages reduce pipeline bubble fractions compared to naive GPipe-style scheduling.

### 4. Quantization and Precision

Reducing numerical precision reduces memory and increases throughput at the cost of potential accuracy loss:

- **Post-training quantization (PTQ)**: Quantize weights (and optionally activations) after training. Techniques include GPTQ (weight-only INT4/INT8 via second-order Hessian approximation) and AWQ (activation-aware weight quantization).
- **Quantization-aware training (QAT)**: Simulate quantization during training using straight-through estimators to recover accuracy degraded by PTQ.
- **FP8 training**: Emerging hardware (H100, MI300X) supports FP8 natively; frameworks are developing FP8 training recipes with per-tensor or per-block scaling. Transformer Engine (NVIDIA) provides drop-in FP8 layers with automatic scaling.
- **BF16 vs FP16**: BF16 has the same exponent range as FP32, making it more stable for training (less need for loss scaling); FP16 has higher precision in the mantissa but is prone to overflow/underflow.
- **Speculative decoding**: At inference time, a small draft model generates candidate tokens that are verified in parallel by the large model, increasing effective throughput without changing output distribution.
- **INT8 activation quantization**: Quantizing both weights and activations to INT8 (e.g., LLM.int8() / SmoothQuant) requires handling outlier activation channels that resist quantization without per-channel scaling corrections.
- **Block-wise and grouped quantization**: Dividing weight matrices into small blocks and computing separate scale factors per block (as in GGUF/llama.cpp formats) significantly improves accuracy at low bit-widths such as Q4_K and Q5_K.

### 5. Storage and Checkpointing

- **Checkpoint frequency and cost**: Saving full model + optimizer state for trillion-parameter models is expensive in time and storage. Async checkpointing (writing to CPU memory while training continues) and incremental checkpointing reduce overhead.
- **Distributed checkpointing**: Each rank saves its shard independently; a global metadata file allows reconstruction. PyTorch Distributed Checkpoint (DCP) and Megatron-LM implement this pattern, enabling parallel writes that scale with cluster size.
- **Storage systems**: Fast distributed file systems (Lustre, GPFS, WekaFS) or object stores (S3, GCS, Azure Blob) with high aggregate bandwidth are required to avoid checkpoint I/O becoming a bottleneck. Checkpointing to DRAM (host memory) first and flushing asynchronously to persistent storage further reduces training pause time.
- **Checkpoint compression**: Lossless compression (LZ4, Zstandard) applied to checkpoint shards can reduce storage costs and network transfer time, particularly for optimizer states that often contain low-entropy FP32 values.
- **Selective checkpointing**: Saving only model weights (not optimizer states) at intermediate intervals, with full checkpoints less frequently, balances recovery granularity against storage cost.

### 6. Inference Serving

Inference introduces different constraints from training:

- **Latency vs. throughput trade-off**: Interactive applications (chatbots) prioritize time-to-first-token (TTFT) and token generation latency; batch jobs prioritize throughput (tokens per second per dollar).
- **KV cache**: Attention keys and values from prior tokens are cached to avoid recomputation; managing this cache efficiently (paging, prefix caching, radix attention as in SGLang) is a major focus.
- **Disaggregated prefill/decode**: Separating the compute-intensive prefill phase from the memory-bandwidth-bound decode phase onto different hardware pools (as explored by Splitwise and PD disaggregation approaches) can improve both latency and utilization simultaneously.
- **Model parallelism for inference**: TP and PP are used at inference time to fit large models; tensor parallelism increases per-token throughput at the cost of communication overhead, creating an optimal TP degree that depends on sequence length and batch size.
- **Batching strategies**: Static batching wastes GPU cycles waiting for slow requests; continuous batching (iteration-level scheduling) and dynamic batching with SLA-aware schedulers maximize utilization while respecting latency budgets.
- **Chunked prefill**: Breaking long prompt prefills into chunks interleaved with decode steps prevents long prefill requests from blocking decode throughput and improves tail latency.
- **Speculative execution and lookahead decoding**: Beyond speculative decoding with a draft model, techniques like Medusa (multiple decoding heads) and EAGLE (feature-level draft models) reduce latency without requiring a separate model.
- **Prefix caching**: Reusing KV cache entries for shared prompt prefixes across requests (e.g., system prompts) dramatically reduces redundant computation in production serving scenarios.

### 7. Networking and Collective Communication

- **Collective operations**: Training at scale relies on all-reduce (gradient synchronization), all-gather (parameter gathering in FSDP), reduce-scatter, and all-to-all (MoE expert routing). Each has different bandwidth and latency profiles.
- **NCCL and RCCL**: NVIDIA NCCL (and AMD's RCCL fork) implement optimized collective algorithms (ring-allreduce, tree-allreduce, recursive halving-doubling) on top of NVLink, InfiniBand, and RoCE fabrics.
- **Topology-aware communication**: Modern clusters have hierarchical networks (NVLink within a node, InfiniBand between nodes). Communication libraries and frameworks must exploit this hierarchy (intra-node vs. inter-node collectives) to minimize expensive inter-node transfers.
- **RDMA and GPUDirect**: Remote Direct Memory Access allows NICs to read/write GPU memory directly without CPU involvement, reducing latency and CPU overhead for inter-node transfers.
- **Network congestion and fairness**: At scale, many concurrent collective operations compete for bandwidth. Techniques such as adaptive routing, explicit congestion notification (ECN), and careful job placement on the network fabric are required to prevent hot spots.
- **Communication-computation overlap**: Frameworks implement gradient bucketing, double-buffering, and asynchronous collective launches to overlap network transfers with ongoing compute, hiding a significant fraction of communication overhead.

### 8. Fault Tolerance and Reliability

- **Hardware failure rates at scale**: In clusters of thousands of GPUs running for weeks, hardware failures (GPU ECC errors, NIC failures, host OS crashes) are not exceptional events but expected occurrences. Infra must handle them gracefully.
- **Checkpoint-restart**: The baseline approach — roll back to the last checkpoint on failure. Minimizing checkpoint overhead (see §5) directly reduces the expected wasted compute per failure.
- **In-memory redundancy**: Techniques such as redundant in-memory checkpoints (C3, Gemini) replicate recent state in peer GPU memory, enabling fast recovery from single-node failures without reading from slow persistent storage.
- **Elastic training**: Frameworks that can resize the training job (add or remove nodes) dynamically reduce the impact of partial failures without requiring full restarts.
- **Anomaly detection**: Silent data corruption (SDC) from GPU compute errors may not raise hardware exceptions but can silently corrupt model state. Infra must include loss spike detection, gradient norm monitoring, and periodic numerical consistency checks.
- **Straggler mitigation**: Slow nodes (due to thermal throttling, memory errors, or network congestion) can bottleneck synchronous training. Techniques include straggler replication, backup workers, and asynchronous or semi-synchronous training variants.

## Key Variants and Subfields

| Subfield | Focus |
| --- | --- |
| **Training infra** | Maximizing MFU, minimizing time-to-convergence for large pre-training runs |
| **Fine-tuning infra** | Parameter-efficient fine-tuning (LoRA, QLoRA), RLHF pipelines, DPO at scale |
| **Inference serving** | Latency, throughput, cost-per-token optimization for production serving |
| **Evaluation infra** | Efficient harnesses for running benchmarks across model checkpoints at scale |
| **Data pipeline infra** | High-throughput tokenization, shuffling, deduplication, and streaming of web-scale datasets |
| **Networking / interconnect** | Optimizing collective communication (NCCL, RCCL) over InfiniBand, RoCE, NVLink |
| **Compiler and kernel engineering** | Writing and auto-generating high-performance GPU kernels via Triton, CUTLASS, or torch.compile backends |
| **Reliability engineering** | Fault detection, checkpoint strategies, elastic training, and silent data corruption mitigation |

## Key Metrics

- **MFU (Model FLOP Utilization)**: Fraction of theoretical peak FLOP/s actually used for useful model computation. Values of 30–60% are typical for large pre-training runs; higher is better.
- **Tokens per second (TPS)**: Throughput of a training run or inference server, often normalized per GPU or per dollar.
- **Time-to-first-token (TTFT)**: Latency from request arrival to first generated token (inference); critical for interactive applications.
- **Inter-token latency (ITL)**: Latency between successive generated tokens during the decode phase; determines perceived generation speed.
- **GPU memory utilization**: Fraction of HBM in use; high utilization is generally desirable but leaves no headroom for activation spikes or cache growth.
- **Cost per million tokens**: Business-level metric for inference serving efficiency, encompassing hardware, networking, and operational costs.
- **Pipeline bubble fraction**: The fraction of pipeline stages sitting idle due to pipeline fill/drain; lower is better for pipeline parallelism efficiency.
- **Communication-to-compute ratio**: The fraction of step time spent in collective communication vs. compute; high ratios indicate communication bottlenecks.
- **Checkpoint overhead**: Time or fraction of training wall-clock time spent saving checkpoints; should be minimized via async or distributed checkpointing.

## Key Frameworks and Tools

- **[[megatron-lm]]**: NVIDIA's library for efficient tensor and pipeline parallelism; widely used for large-scale pre-training runs on GPU clusters. Provides reference implementations of 3D parallelism and sequence parallelism.
- **[[deepspeed]]**: Microsoft's library implementing ZeRO optimizer stages (1/2/3), pipeline parallelism, and inference optimizations including DeepSpeed-Inference and DeepSpeed-FastGen.
- **[[pytorch-fsdp]]**: PyTorch's native fully sharded data parallelism, offering ZeRO-3-equivalent sharding with tight integration into the PyTorch ecosystem.
- **[[flashattention]]**: Memory-efficient exact attention kernel by Tri Dao et al.; now standard in most training and inference stacks. FlashAttention-2 and FlashAttention-3 extend support to newer hardware and improve parallelism.
- **[[v
