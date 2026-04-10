```yaml
title: LLM Inference Infrastructure
type: concept
tags: [llm-infra, inference, distributed-inference, slo, kv-cache, pd-separation]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# LLM Inference Infrastructure

LLM inference infrastructure refers to the systems and frameworks used to serve large language model responses to end users at scale. The primary goals are high performance (throughput and latency) and high accuracy (numerical precision). Single-card inference has few engineering challenges; distributed inference across multiple devices introduces substantial complexity.

## Key Properties

### Performance Dimensions

- **Throughput**: Number of tokens generated per second across all concurrent requests. High throughput is critical for cost efficiency.
- **Latency**: Time-to-first-token (TTFT) and time-per-output-token (TPOT) are the primary latency metrics. Together they determine perceived responsiveness.
- **Service Level Objectives (SLOs)**: Production systems must meet contractual or internal latency and availability targets. Infrastructure design choices are often driven by SLO requirements.

### Accuracy Dimensions

- **Numerical precision**: FP16, BF16, INT8, and INT4 quantization formats trade accuracy for speed and memory efficiency.
- **Attention correctness**: Approximations such as sparse or linear attention must not degrade output quality beyond acceptable thresholds.
- **Consistency**: Deterministic outputs are desirable for debugging and reproducibility, though batching and parallel decoding can introduce non-determinism.

---

## Core Components

### KV Cache

The key-value (KV) cache stores intermediate attention keys and values computed during prefill so they do not need to be recomputed during each decode step. It is the dominant consumer of GPU memory during inference.

- **Capacity constraints**: KV cache size scales with batch size × sequence length × number of layers × hidden dimension. Memory pressure limits maximum batch size and context length.
- **Paged attention**: Inspired by virtual memory, paged attention (as implemented in [[vllm]]) stores KV cache in non-contiguous memory blocks, reducing fragmentation and enabling higher effective batch sizes.
- **Prefix caching**: Shared prompt prefixes can be cached and reused across requests, reducing redundant computation and TTFT for common system prompts.
- **Offloading**: KV cache can be offloaded to CPU memory or SSDs when GPU memory is exhausted, at the cost of higher latency.

### Prefill vs. Decode Separation (PD Separation)

Inference has two distinct computational phases with very different hardware profiles:

- **Prefill phase**: Processes the entire input prompt in parallel. This phase is compute-bound and benefits from high arithmetic intensity (FLOP-heavy operations). It is analogous to a large matrix multiplication workload.
- **Decode phase**: Generates one token at a time autoregressively. This phase is memory-bandwidth-bound because it reads the full model weights and KV cache for each token generated.

**PD separation** (also called disaggregated prefill-decode) routes prefill and decode work to separate pools of hardware optimized for each workload. Prefill nodes may be configured for high compute throughput; decode nodes may be optimized for memory bandwidth. This separation can improve SLO attainment and hardware utilization simultaneously.

Challenges of PD separation include:

- KV cache transfer between prefill and decode nodes adds latency and network bandwidth pressure.
- Load balancing becomes more complex with two separate scheduling domains.
- Systems such as [[distserve]] and [[splitwise]] have explored this architecture.

### Batching Strategies

- **Static batching**: A fixed batch of requests is processed together. Simple but leads to wasted compute when requests finish at different times.
- **Continuous batching** (iteration-level scheduling): Requests are added to and removed from the batch at each decode step. This dramatically improves GPU utilization and is the default in modern serving frameworks such as [[vllm]] and [[tgi]].
- **Chunked prefill**: Long prefill sequences are broken into smaller chunks interleaved with decode steps, reducing head-of-line blocking and improving TTFT for queued decode requests.

### Tensor Parallelism and Pipeline Parallelism

Distributed inference splits model weights across multiple devices:

- **Tensor parallelism (TP)**: Individual weight matrices are sharded across devices. Each device computes a partial result; an all-reduce synchronization step combines results. TP reduces per-device memory and latency but requires high-bandwidth interconnects (e.g., NVLink).
- **Pipeline parallelism (PP)**: The model is split layer-by-layer across devices. Each device holds a subset of layers and passes activations to the next stage. PP reduces interconnect bandwidth requirements but introduces pipeline bubbles that lower throughput.
- **Sequence parallelism**: Activations along the sequence dimension are sharded, complementing tensor parallelism for long-context workloads.
- **Expert parallelism**: For mixture-of-experts (MoE) models, individual experts are placed on different devices, requiring all-to-all communication for token routing.

### Speculative Decoding

Speculative decoding uses a small draft model to propose multiple future tokens in parallel, which a larger verifier model then accepts or rejects in a single forward pass. When acceptance rates are high, this reduces the number of serial decode steps and improves effective throughput without changing output distribution.

Variants include:

- **Self-speculative / medusa heads**: Draft tokens are generated by auxiliary heads attached to the main model.
- **Token tree speculation**: Multiple candidate token sequences are proposed as a tree and verified simultaneously.
- **Retrieval-based speculation**: Draft tokens are proposed by looking up likely continuations from a corpus rather than running a neural model, reducing draft generation cost to near zero.

---

## Scheduling and Resource Management

### Request Scheduling

The inference scheduler must balance:

- **Fairness**: Avoiding starvation of low-priority or long requests.
- **SLO compliance**: Prioritizing requests at risk of missing latency targets.
- **Memory management**: Preempting or swapping requests when KV cache memory is exhausted.

Preemption policies include recomputation (re-run prefill from scratch) and swapping (move KV cache to CPU memory). Recomputation wastes compute; swapping wastes memory bandwidth.

Advanced schedulers may use:

- **Earliest deadline first (EDF)**: Prioritize requests whose SLO deadline is soonest.
- **Shortest job first (SJF)**: Prioritize requests with the fewest remaining decode steps, improving average latency at the cost of potential starvation for long requests.
- **Work-conserving scheduling**: Ensure no GPU cycle is wasted when runnable requests exist, even if this means temporarily violating strict priority orderings.

### Memory Management

- **Memory pooling**: Pre-allocating a large contiguous GPU memory pool for KV cache avoids fragmentation and allocation overhead.
- **Garbage collection**: Completed requests must have their KV cache memory reclaimed promptly.
- **Radix attention**: Extends paged attention with a radix tree structure to enable fine-grained prefix sharing across requests with common prefixes of varying length.
- **KV cache eviction policies**: When memory is full, least-recently-used (LRU) or scored eviction policies determine which cached prefixes to discard. Evicted entries must be recomputed on next access.

### Autoscaling

Production serving systems must scale the number of inference replicas dynamically in response to traffic:

- **Horizontal scaling**: Add or remove full model replicas. Limited by replica startup time (model loading can take minutes for large models).
- **Vertical scaling**: Adjust tensor or pipeline parallelism degree. More disruptive but allows finer-grained resource adjustment.
- **Predictive autoscaling**: Forecast traffic patterns (e.g., diurnal cycles) and proactively provision capacity before demand spikes.
- **Spot/preemptible instances**: Cloud spot instances offer lower cost but require checkpoint/restore or request migration on preemption.

---

## Key Challenges

### Heterogeneous Hardware

Production deployments often span multiple GPU generations or mixed GPU/CPU environments. Inference systems must abstract hardware differences while exploiting vendor-specific features (e.g., CUDA graphs, FlashAttention, Tensor Core variants). Hardware-aware kernel selection and autotuning (as in [[triton]] and [[tvm]]) are essential for peak performance across platforms.

### Long-Context Inference

Contexts of 100K+ tokens dramatically increase KV cache memory consumption and attention computation cost. The attention mechanism scales quadratically with sequence length in both time and memory under standard implementations. Solutions include:

- **Sliding window attention**: Each token attends only to a fixed window of recent tokens, capping memory and compute at O(window_size × sequence_length).
- **KV cache compression and eviction**: Selectively drop or compress KV entries for tokens deemed less relevant, trading accuracy for memory.
- **Ring attention**: Distributes the sequence across devices in a ring topology, enabling attention over sequences longer than any single device's memory capacity.
- **FlashAttention**: Tile-based fused attention kernel that reduces HBM memory bandwidth usage and avoids materializing the full attention matrix, enabling longer contexts within a fixed memory budget.

### Multi-Modal Inference

Serving vision-language models or audio-language models requires preprocessing pipelines for non-text modalities and careful integration of modality-specific encoders with the language model backbone. Key considerations include:

- **Encoder-decoder latency asymmetry**: Vision encoders may add significant TTFT overhead if not pipelined efficiently with prefill.
- **Variable-length visual tokens**: Image resolution affects the number of visual tokens, making batching more complex.
- **Caching visual embeddings**: For repeated queries over the same image, caching the encoded visual representation avoids redundant encoder computation.

### Cost Efficiency

Hardware accelerators (GPUs, TPUs) are expensive. Inference infrastructure must maximize utilization (measured in MFU — model FLOP utilization) while meeting latency SLOs. Underutilization and over-provisioning are the primary cost levers. Key strategies include:

- **Request batching**: Increasing batch size amortizes weight-loading memory bandwidth across more tokens, improving arithmetic intensity and MFU during decode.
- **Quantization**: Reducing weight precision from FP16 to INT8 or INT4 reduces memory footprint and bandwidth requirements, enabling larger batch sizes or lower-cost hardware.
- **Model distillation**: Smaller distilled models can serve many use cases at a fraction of the inference cost of the full-size model.
- **Caching at the application layer**: Semantic caching of common queries avoids running inference at all for frequently repeated prompts.

### Cold Start Latency

Loading a large model from storage to GPU memory is slow (potentially many seconds to minutes). Strategies to mitigate cold start include:

- **Warm pools**: Maintain a minimum number of pre-loaded replicas at all times.
- **Model sharding with fast loading**: Store model weights in formats optimized for fast deserialization (e.g., safetensors, memory-mapped files).
- **Lazy loading**: Begin accepting requests before all weights are loaded by prioritizing early layers.

### Numerical Stability and Quantization Artifacts

Low-precision quantization (INT4, INT8) can introduce accuracy regressions that are difficult to detect without comprehensive evaluation. Challenges include:

- **Outlier activations**: Transformer activations often contain large outliers that degrade uniform quantization quality; methods such as SmoothQuant and GPTQ address this.
- **Per-tensor vs. per-channel quantization**: Finer-grained quantization scales improve accuracy but add overhead.
- **KV cache quantization**: Quantizing the KV cache to INT8 or INT4 reduces memory consumption further but requires careful handling of precision loss in long-context settings.

---

## Variants and Deployment Patterns

| Pattern | Description | Use Case |
| --- | --- | --- |
| Single-node, single-GPU | Model fits on one GPU; no distribution needed | Small models, development |
| Single-node, multi-GPU (TP) | Tensor parallel across GPUs on one machine | Medium/large models |
| Multi-node distributed | TP + PP across multiple machines | Very large models (100B+) |
| Disaggregated prefill-decode | Separate clusters for prefill and decode | High-throughput production |
| Serverless / spot inference | Autoscaling on cloud spot instances | Variable traffic, cost-sensitive |
| Edge / on-device inference | Model runs on consumer hardware (laptop, phone) | Privacy-sensitive, offline use cases |
| Hybrid CPU-GPU inference | Weights partially offloaded to CPU RAM | Memory-constrained environments |

---

## Observability and Debugging

Reliable inference infrastructure requires comprehensive observability:

- **Metrics**: Token throughput, TTFT, TPOT, queue depth, KV cache utilization, GPU memory utilization, MFU, request success/error rates.
- **Tracing**: Distributed tracing (e.g., OpenTelemetry) across preprocessing, scheduling, prefill, decode, and postprocessing stages to identify latency bottlenecks.
- **Logging**: Request-level logs with input/output lengths, assigned batch, device, and timing data to enable post-hoc analysis and SLO auditing.
- **Profiling**: GPU kernel profiling (e.g., NVIDIA Nsight, PyTorch Profiler) to identify inefficient operations and guide optimization.
- **Shadow traffic**: Replaying production traffic against new versions of the inference stack to validate correctness and performance before rollout.

---

## Benchmarking and Evaluation

Standard benchmarks for LLM inference infrastructure include:

- **ShareGPT traces**: Real-world conversation length distributions used to simulate production traffic patterns.
- **Synthetic Poisson arrivals**: Controlled load testing with configurable request arrival rates and length distributions.
- **MLPerf Inference**: Industry-standard benchmark suite covering throughput and latency targets across model sizes.
- **Metrics reported**: Tokens per second (TPS), P50/P90/P99 TTFT and TPOT, SLO attainment rate at various load levels, cost per million tokens.

Comparing systems requires careful attention to workload assumptions; results are highly sensitive to input/output length distributions, request concurrency, and hardware configuration.

---

## Related Concepts

- [[kv-cache]]
- [[paged-attention]]
- [[speculative-decoding]]
- [[tensor-parallelism]]
- [[continuous-batching]]
- [[pd-separation]]
- [[transformer-architecture]]
- [[quantization]]
- [[mixture-of-experts]]
- [[flashattention]]
- [[chunked-prefill]]
- [[radix-attention]]
- [[ring-attention]]
- [[model-distillation]]
- [[autoscaling]]

## Related Entities and Projects

- [[vllm]] — Open-source LLM serving engine; introduced paged attention and continuous batching.
- [[tgi]] — HuggingFace Text Generation Inference server.
- [[tensorrt-llm]] — NVIDIA's optimized inference library with quantization and fused kernel support.
- [[distserve]] — Research system exploring disaggregated prefill-decode serving.
- [[splitwise]] — Microsoft Research system for PD separation across heterogeneous hardware.
- [[sarathi-serve]] — System introducing chunked prefill to reduce decode interference and improve SLO attainment.
- [[orca]] — Early academic system proposing iteration-level continuous batching.
- [[triton]] — OpenAI's GPU kernel programming language used to implement custom inference kernels.
- [[flashattention]] — Memory-efficient attention kernel enabling longer contexts and higher throughput.
- [[deepspeed-inference]] — Microsoft's inference optimization library with ZeRO-Inference for large model serving.

## Sources

- [[1-overview.md]]
