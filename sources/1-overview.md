```yaml
title: "Lecture 1: General Overview"
type: source
tags: [ai-infra, large-models, gpu, inference, training, distributed-systems, compilers, operators]
created: 2026-04-05
updated: 2026-04-05
sources: []
```

# Lecture 1: General Overview

**Original title:** 1.总体介绍  
**Last modified (source):** December 4, 2025  
**Knowledge base:** Inference-related knowledge base  

---

## 1. Large Model Development and Infra

The rapid growth of large language models (LLMs) and other large-scale AI models has driven an equally rapid evolution in the infrastructure required to train and serve them. Modern large models involve billions to trillions of parameters, requiring specialized hardware, software stacks, and systems design to operate efficiently.

Key trends in large model development include:

- **Scale**: Models have grown from millions to hundreds of billions of parameters within a few years (GPT-2 → GPT-3 → GPT-4, LLaMA series, etc.).
- **Hardware dependency**: Training and inference are heavily dependent on GPU and accelerator hardware (NVIDIA A100, H100, TPUs, etc.).
- **Infrastructure complexity**: Running large models at scale requires orchestration of distributed compute, memory management, networking, and storage.

---

## 2. Training vs. Inference

Large model infrastructure is typically divided into two major workloads:

### Training

- Involves forward passes, loss computation, backpropagation, and weight updates.
- Requires storing activations, gradients, and optimizer states — extremely memory-intensive.
- Typically uses techniques like mixed-precision training (FP16/BF16), gradient checkpointing, and distributed parallelism.

### Inference

- Involves only forward passes to generate predictions or outputs.
- Memory requirements are lower than training but latency and throughput constraints are strict.
- Key challenges include KV-cache management, batching strategies, and quantization.
- Inference optimization is the primary focus of this lecture series.

---

## 3. The GPU and Accelerator Ecosystem

GPUs are the dominant hardware platform for both training and inference of large models. Key concepts:

- **CUDA cores / Tensor cores**: Tensor cores (introduced with Volta architecture) accelerate matrix multiply-accumulate (MMA) operations, which are the dominant operation in transformer models.
- **Memory hierarchy**: Registers → L1/shared memory → L2 cache → HBM (High Bandwidth Memory). Understanding this hierarchy is critical for writing efficient kernels.
- **PCIe vs. NVLink**: Inter-GPU communication bandwidth significantly impacts distributed training and inference performance.
- **FLOPS vs. memory bandwidth**: Many operations in large models are memory-bandwidth-bound rather than compute-bound, making memory access patterns a key optimization target.

Related: [[gpu-architecture]], [[cuda-programming-model]]

---

## 4. The Software Stack

The software stack for large model infra spans multiple layers:

| Layer | Examples |
| --- | --- |
| Model definition | PyTorch, JAX, TensorFlow |
| Compiler / graph optimization | torch.compile, XLA, TensorRT, TVM |
| Operator libraries | cuBLAS, cuDNN, CUTLASS, FlashAttention |
| Serving frameworks | vLLM, TensorRT-LLM, Triton Inference Server |
| Orchestration | Kubernetes, Ray, SLURM |

Each layer introduces abstractions and optimization opportunities. Understanding where bottlenecks arise at each layer is essential for infra engineers.

Related: [[compiler-stack]], [[operator-optimization]], [[serving-frameworks]]

---

## 5. Distributed Systems for Large Models

Because large models exceed the memory capacity of a single GPU, distributed strategies are necessary:

- **Data Parallelism (DP)**: Each GPU holds a full model copy; data is split across GPUs. Gradients are synchronized after each step.
- **Tensor Parallelism (TP)**: Individual weight matrices are split across GPUs. Requires frequent inter-GPU communication within a layer.
- **Pipeline Parallelism (PP)**: Model layers are split across GPUs in stages. Introduces pipeline bubbles but reduces communication frequency.
- **Sequence Parallelism**: Attention computation is distributed along the sequence dimension.
- **Expert Parallelism**: Used for Mixture-of-Experts (MoE) models; different experts reside on different GPUs.

In practice, large deployments use combinations of the above (e.g., 3D parallelism = DP + TP + PP).

Related: [[tensor-parallelism]], [[pipeline-parallelism]], [[data-parallelism]], [[mixture-of-experts]]

---

## 6. Operators and Kernel Optimization

At the lowest level, model computations are expressed as operators (kernels) that execute on GPU hardware:

- **GEMM (General Matrix Multiplication)**: The core operation in linear layers and attention. Heavily optimized via cuBLAS and CUTLASS.
- **Attention kernels**: Naive attention is quadratic in sequence length and memory-intensive. [[flash-attention]] fuses operations and tiles computation to stay within SRAM, drastically reducing HBM traffic.
- **Fused kernels**: Combining multiple operations (e.g., LayerNorm + linear) into a single kernel reduces memory round-trips.
- **Custom CUDA / Triton kernels**: Engineers write custom kernels when library implementations are suboptimal for specific shapes or access patterns.

Related: [[flash-attention]], [[triton-lang]], [[cutlass]]

---

## 7. Inference-Specific Challenges

This lecture series focuses specifically on inference. Key challenges include:

- **KV Cache**: During autoregressive generation, keys and values from past tokens are cached to avoid recomputation. Cache size grows with sequence length and batch size, becoming a major memory bottleneck.
- **Batching**: Combining multiple requests into a single batch improves GPU utilization but introduces complexity (different sequence lengths, different generation states).
- **Continuous batching**: Unlike static batching, continuous (iteration-level) batching allows new requests to join mid-generation, improving throughput.
- **Quantization**: Reducing weight and/or activation precision (INT8, INT4, FP8) to reduce memory footprint and increase throughput, at the cost of potential accuracy degradation.
- **Speculative decoding**: Using a small draft model to propose token sequences that are then verified by the large model in parallel, reducing latency.

Related: [[kv-cache]], [[continuous-batching]], [[quantization]], [[speculative-decoding]]

---

## 8. Key Metrics

When evaluating large model infra, the following metrics are commonly used:

| Metric | Description |
| --- | --- |
| **Latency (TTFT)** | Time to First Token — how long until the first output token is produced |
| **Latency (TPOT)** | Time Per Output Token — average time between successive output tokens |
| **Throughput** | Tokens generated per second across all requests |
| **MFU** | Model FLOP Utilization — fraction of theoretical peak FLOPS achieved |
| **Memory efficiency** | How effectively GPU HBM is utilized |

There is typically a latency-throughput tradeoff: larger batches improve throughput but increase per-request latency.

---

## 9. Overview of Lecture Series

This lecture series covers the full stack of large model inference infrastructure, including:

1. **General overview** (this lecture) — scale, hardware, software stack, key challenges
2. **GPU architecture and CUDA programming model**
3. **Operator optimization** — GEMM, attention, fused kernels
4. **Compiler and graph optimization**
5. **Quantization and low-precision inference**
6. **Distributed inference** — tensor parallelism, pipeline parallelism
7. **Serving systems** — KV cache management, batching, scheduling
8. **Advanced topics** — speculative decoding, MoE inference, long-context inference

---

## Related Pages

- [[gpu-architecture]]
- [[flash-attention]]
- [[kv-cache]]
- [[continuous-batching]]
- [[quantization]]
- [[speculative-decoding]]
- [[tensor-parallelism]]
- [[pipeline-parallelism]]
- [[mixture-of-experts]]
- [[serving-frameworks]]
- [[operator-optimization]]
- [[compiler-stack]]
- [[triton-lang]]
