```yaml
title: LLM Training Infrastructure
type: concept
tags: [llm-infra, distributed-training, gpu, performance, precision]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# LLM Training Infrastructure

LLM training infrastructure refers to the full software and hardware stack required to train large language models at scale — spanning thousand-card to ten-thousand-card GPU/accelerator clusters. The core challenge is simultaneously managing compute efficiency, memory usage, inter-device communication, numerical precision, and fault tolerance, all at a scale where even tiny per-device failure rates compound into frequent aggregate failures.

## Key Properties

### Scale

Modern frontier model training runs use thousands to tens of thousands of accelerators (GPUs or TPUs) operating in concert. A single training run for a frontier model may consume millions of GPU-hours and petabytes of training data. This scale means that infrastructure decisions — parallelism strategy, precision format, interconnect topology — have enormous impact on both cost and feasibility.

### Parallelism Strategies

Because no single device can hold or process a large model alone, training is distributed across devices using one or more parallelism strategies:

- **Data Parallelism (DP):** Each device holds a full model replica and processes a different mini-batch. Gradients are synchronized (all-reduced) across replicas after each step. Simple but memory-inefficient for very large models.
- **Tensor Parallelism (TP):** Individual weight matrices are sharded across devices. Each device computes a partial result; results are combined via communication collectives within a layer. Reduces per-device memory but requires high-bandwidth intra-node interconnects (e.g., NVLink).
- **Pipeline Parallelism (PP):** The model is split into sequential stages, each assigned to a different device or group of devices. Micro-batches flow through the pipeline. Introduces pipeline bubbles (idle time) that must be minimized via scheduling strategies (e.g., 1F1B, interleaved schedules).
- **Sequence Parallelism (SP):** Activations along the sequence dimension are distributed across devices, reducing activation memory during attention and layer-norm operations. Often used in conjunction with tensor parallelism.
- **Expert Parallelism (EP):** Used in Mixture-of-Experts (MoE) architectures; different experts are placed on different devices and tokens are routed to the appropriate device.
- **Fully Sharded Data Parallelism (FSDP / ZeRO):** Parameters, gradients, and optimizer states are sharded across data-parallel workers. Devices gather shards only when needed for computation, dramatically reducing per-device memory. ZeRO stages 1/2/3 progressively shard more state.

Real training runs typically combine multiple strategies (e.g., DP + TP + PP = "3D parallelism") to balance memory, compute, and communication costs.

### Memory Management

Training a large model consumes memory for:

- **Parameters** (model weights)
- **Gradients** (one gradient tensor per parameter)
- **Optimizer states** (e.g., Adam maintains first and second moment estimates — 2× parameter count in FP32)
- **Activations** (intermediate values stored for the backward pass)
- **Temporary buffers** (communication buffers, workspace for kernels)

Techniques to reduce memory pressure include:

- **Activation checkpointing (gradient checkpointing):** Discard activations during the forward pass and recompute them during backward. Trades compute for memory.
- **Mixed precision training:** Store weights in FP16 or BF16; keep a master copy in FP32 for optimizer updates.
- **Offloading:** Move optimizer states or parameters to CPU RAM or NVMe storage (e.g., ZeRO-Infinity).
- **Paged attention / memory-efficient attention:** Reduce activation memory during attention computation (relevant at inference but also useful during training).

### Numerical Precision

Precision choice affects both memory usage and training stability:

- **FP32 (single precision):** Historically the default. Numerically stable but memory-intensive and slower on modern hardware than lower-precision formats.
- **FP16 (half precision):** 2× memory savings and faster tensor core throughput. Requires loss scaling to avoid underflow in gradients. Can be numerically unstable for some operations.
- **BF16 (bfloat16):** Same exponent range as FP32 but reduced mantissa. More numerically stable than FP16 for training; preferred on TPUs, A100, H100, and later hardware.
- **FP8 (e4m3 / e5m2):** Emerging format supported on H100/H200. Enables further throughput gains; requires careful per-tensor or per-block scaling. Active research area.
- **Mixed precision:** Compute in low precision (FP16/BF16); accumulate and store master weights in FP32. Standard practice today.

### Interconnect and Communication

Communication between devices is often the bottleneck in distributed training:

- **Intra-node:** NVLink (NVIDIA) provides high-bandwidth, low-latency GPU-to-GPU communication within a node (e.g., 900 GB/s bidirectional on NVLink 4.0).
- **Inter-node:** InfiniBand (HDR/NDR) or high-speed Ethernet connects nodes. Bandwidth is typically much lower than NVLink (e.g., 400 Gb/s per link with NDR InfiniBand vs. several TB/s intra-node NVLink).
- **Collective operations:** All-reduce, all-gather, reduce-scatter are the dominant communication patterns. Libraries like NCCL (NVIDIA) and RCCL (AMD) implement these efficiently.
- **Topology-aware scheduling:** Parallelism strategies must be mapped to the physical network topology — e.g., tensor parallelism is placed within a node (NVLink), pipeline parallelism crosses nodes (InfiniBand).

### Fault Tolerance

At scale, hardware failures (GPU errors, network faults, storage failures) are near-certain over the course of a long training run:

- **Checkpointing:** Periodically save model weights, optimizer states, and data loader position to distributed storage. Enables resuming from the last checkpoint after failure. Checkpoint frequency is a trade-off between storage I/O overhead and lost computation on failure.
- **Asynchronous checkpointing:** Write checkpoints in the background to minimize blocking of training.
- **Elastic training:** Ability to add or remove workers during a run (complex; less common in frontier training).
- **Automatic restart:** Infrastructure-level tooling (e.g., Kubernetes, Slurm) to detect failures and relaunch jobs automatically.
- **Shadow / redundant workers:** Some systems maintain spare devices ready to replace failed ones.
- **MTTR (Mean Time To Recovery):** Key operational metric. Long recovery times waste expensive GPU time.

### Compute Efficiency Metrics

- **MFU (Model FLOP Utilization):** Fraction of theoretical peak FLOP/s actually used for useful model computation. A high MFU (e.g., 40–60% on A100s) indicates efficient infrastructure. Losses come from communication overhead, memory bottlenecks, and pipeline bubbles.
- **Hardware FLOP/s:** Theoretical peak throughput of an accelerator (e.g., 312 TFLOP/s BF16 on A100, 989 TFLOP/s BF16 on H100 SXM with sparsity).
- **Throughput (tokens/sec):** End-to-end training speed; directly determines time and cost to train.
- **Batch size and gradient accumulation:** Larger global batch sizes improve throughput but can affect convergence. Gradient accumulation allows effective large batches with limited memory.

## Key Components

### Hardware

- **GPU clusters:** NVIDIA A100, H100, H200 are the dominant training accelerators as of 2024–2025. AMD MI300X is an emerging competitor.
- **TPUs:** Google's custom accelerators, used internally and available via Google Cloud. Tightly integrated with JAX/XLA.
- **Networking:** InfiniBand (Mellanox/NVIDIA) for inter-node; NVLink/NVSwitch for intra-node.
- **Storage:** High-throughput distributed filesystems (Lustre, GPFS, WekaIO) for checkpoint storage and data loading.
- **Power and cooling:** Frontier clusters consume tens to hundreds of megawatts; power delivery and liquid cooling are active infrastructure concerns.

### Software Frameworks

- **[[pytorch]]:** The dominant research and production training framework. Supports FSDP, DDP, and integrates with libraries like DeepSpeed and Megatron.
- **[[deepspeed]]:** Microsoft library implementing ZeRO optimizer, pipeline parallelism, and mixed precision. Widely used for large-scale training.
- **[[megatron-lm]]:** NVIDIA's framework for tensor and pipeline parallelism, optimized for transformer training on NVIDIA hardware.
- **[[jax]] / XLA:** Google's functional array computation framework. Used for TPU training and increasingly for GPU training. Strong support for compilation and sharding via `jax.sharding`.
- **[[triton]]:** OpenAI's GPU kernel language enabling custom high-performance ops (e.g., FlashAttention) without writing raw CUDA.
- **[[nccl]]:** NVIDIA's collective communication library; foundational for all GPU-distributed training.

### Key Algorithmic Techniques

- **[[flashattention]]:** Memory-efficient attention algorithm that reorders attention computation to minimize HBM reads/writes. Dramatically reduces activation memory and increases throughput for long sequences.
- **[[gradient-checkpointing]]:** Trade compute for memory by recomputing activations during the backward pass.
- **[[mixed-precision-training]]:** See precision section above.
- **[[muon-optimizer]] / [[adam-optimizer]]:** Optimizer choices affect convergence and memory. Adam (AdamW) is standard; newer optimizers like Muon aim for better scaling.
- **[[learning-rate-scheduling]]:** Warmup + cosine decay or Warmup + Stable + Decay (WSD) schedules are common. Schedule shape affects final model quality.

## Variants and Configurations

### Single-Node Multi-GPU

Used for smaller models or during development. All GPUs connected via NVLink; no inter-node networking required. Simpler to manage; full NVLink bandwidth available.

### Multi-Node GPU Cluster

The standard setting for frontier training. Requires careful orchestration of intra- and inter-node communication, job scheduling (Slurm, Kubernetes, custom), and distributed storage.

### TPU Pods

Google's TPU infrastructure organizes chips into pods with dedicated high-bandwidth interconnects. Training is typically done with JAX; the programming model differs significantly from GPU clusters.

### Cloud vs. On-Premises

- **Cloud (AWS, GCP, Azure, CoreWeave):** Elastic capacity, lower upfront cost, easier to scale. Higher per-GPU-hour cost at sustained usage; potential for noisy neighbors, network variability.
- **On-premises:** High upfront CapEx; lower marginal cost for sustained use; full control over hardware and networking. Preferred by large labs (OpenAI, Anthropic, Google DeepMind, Meta) for frontier runs.

## Common Failure Modes and Challenges

- **Loss spikes / divergence:** Sudden increases in training loss, often caused by bad data batches, numerical instability, or hardware errors producing NaN/Inf values. Mitigated by gradient clipping, loss scaling, and data quality filtering.
- **GPU memory OOM (out of memory):** Occurs when memory pressure exceeds device capacity. Resolved by reducing batch size, enabling activation checkpointing, or adjusting sharding strategy.
- **Communication bottlenecks:** All-reduce or all-gather operations dominating step time. Addressed by topology-aware parallelism placement and overlapping compute with communication.
- **Checkpoint corruption:** Partially written checkpoints can be corrupted on failure. Mitigated by atomic writes, checksum validation, and keeping multiple checkpoint generations.
- **Stragglers:** Slow devices holding back synchronous training. Caused by hardware degradation, thermal throttling, or network issues. Detected via profiling; failing devices may need to be excluded.
- **Pipeline bubbles:** Idle time in pipeline-parallel training between micro-batches. Reduced by increasing the number of micro-batches or using interleaved pipeline schedules.
- **Data loading bottlenecks:** When storage throughput cannot keep up with GPU compute, GPUs stall waiting for data. Mitigated by prefetching, caching hot data in RAM, and using high-throughput distributed filesystems.
- **Numerical instability with low precision:** FP16 and FP8 training can encounter gradient underflow or overflow, particularly in early training or with aggressive learning rates. Mitigated by dynamic loss scaling, careful initialization, and per-tensor scaling factors (FP8).
- **Deadlocks in collective operations:** Misconfigured communication groups or asymmetric collective calls across ranks can cause hangs. Detected by watchdog timers; avoided through careful framework-level abstraction and testing.
- **Token routing imbalance (MoE):** In Mixture-of-Experts models, uneven token routing can cause some expert devices to be overloaded while others are idle, reducing MFU. Mitigated by auxiliary load-balancing losses and capacity factors.

## Observability and Profiling

Effective infrastructure management requires continuous visibility into training health:

- **Loss and metric curves:** Training loss, validation loss, gradient norm, and learning rate are logged continuously (e.g., via Weights & Biases, TensorBoard, or custom dashboards). Anomalies (spikes, plateaus) trigger investigation.
- **GPU utilization and MFU tracking:** Per-device utilization, memory usage, and achieved FLOP/s are monitored. Low utilization signals a bottleneck.
- **Communication profiling:** Tools like NCCL's built-in logging, NVIDIA Nsight Systems, and PyTorch Profiler can identify collective operations dominating step time.
- **Kernel-level profiling:** NVIDIA Nsight Compute provides per-kernel occupancy, memory bandwidth, and arithmetic intensity, enabling identification of inefficient custom kernels.
- **Timeline tracing:** Distributed traces across all ranks allow visualization of synchronization points and straggler effects.
- **Health checks:** Automated pre-run health checks validate GPU memory, NVLink bandwidth, and InfiniBand connectivity before launching a training job, avoiding wasted time discovering hardware failures mid-run.

## Cost Considerations

Training frontier models is extraordinarily expensive; infrastructure choices have direct financial consequences:

- **Compute cost:** At cloud spot or reserved rates, a large training run (e.g., 10,000 H100s for several months) can cost tens to hundreds of millions of dollars. MFU improvements directly reduce this cost.
- **Storage cost:** Checkpoints for a large model (hundreds of GB to multiple TB per checkpoint, written frequently) accumulate significant storage costs. Checkpoint retention policies balance recovery capability against storage spend.
- **Networking cost:** In cloud environments, inter-node data transfer can incur additional charges. Topology-aware placement reduces unnecessary cross-zone or cross-region traffic.
- **Energy cost:** Power consumption at scale is substantial. Frontier labs increasingly track and optimize for power usage effectiveness (PUE) and total energy consumption, both for cost and environmental reasons.
- **Opportunity cost of failures:** Every hour of GPU idle time due to a crash, straggler, or checkpoint recovery represents direct financial loss. Fault tolerance investment has a measurable ROI at scale.

## Emerging Trends

- **FP8 training:** H100/H200 hardware support for FP8 compute is being adopted by frontier labs. Libraries like Transformer Engine (NVIDIA) provide plug-in FP8 layers with automatic scaling. Expected to become standard as tooling matures.
- **Longer context training:** Training on sequences of 128K tokens or more strains memory and communication infrastructure. Techniques like ring attention (distributing sequence chunks across devices) and sparse attention patterns are active areas of development.
- **MoE scaling:** Mixture-of-Experts architectures allow scaling parameter count without proportionally scaling compute, but introduce expert parallelism and routing complexity. Infrastructure support for efficient MoE training is rapidly improving.
- **Custom silicon:** Beyond NVIDIA GPUs and Google TPUs, custom accelerators from Amazon (Trainium), Microsoft (Maia), Meta (MTIA), and startups (Cerebras, Groq, Tenstorrent) are entering the training landscape. Each requires adapted software stacks.
- **Interconnect advances:** NVLink 5.0, CXL-based memory pooling, and co-packaged optics for inter-node networking are on hardware roadmaps and will shift the compute/communication trade-off.
- **Automated parallelism search:** Tools that automatically search for optimal parallelism configurations (e.g., Alpa, FlexFlow) reduce the manual tuning burden, particularly important as model architectures evolve rapidly.
- **Continuous / online training:** Moving beyond static pre-training runs toward continuously updated models introduces new infrastructure requirements around data versioning, incremental checkpointing, and online evaluation.

## Related Concepts
