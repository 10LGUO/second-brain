```yaml
title: AI Infra General Overview
type: concept
tags: [ai-infra, llm, training, inference, distributed-systems, optimization]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# AI Infra General Overview

AI Infra (AI Infrastructure) refers to the engineering domain that systematically optimizes around large model training and inference across five dimensions: **compute, memory, communication, performance, and numerical precision**. Its core objectives are to reduce training and inference costs and improve resource utilization. It sits between algorithmic research and underlying hardware, responsible for transforming algorithm prototype code into production systems that run efficiently on large-scale clusters.

## Core Optimization Dimensions

| Dimension | Training-Side Focus | Inference-Side Focus |
| ----------- | -------------------- | -------------------- |
| Stability | Automatic suspend/recovery for thousand-GPU/ten-thousand-GPU cluster failures | Service high availability; SLO guarantees |
| Numerical Precision | NaN/hang/loss divergence; domestic chip bugs | Quantization accuracy; domestic chip inference precision |
| Performance Optimization | Scaling efficiency (multi-GPU vs. single-GPU); communication hiding | MFU (compute utilization); bandwidth utilization; TTFT/TPOT |
| Memory Optimization | Run model with minimum VRAM, use remaining capacity for data parallelism | Reduce memory footprint → increase batch size → improve throughput |

## Algorithm and Infra Collaboration Workflow

1. **Algorithm Engineers**: Build model architecture, write prototype code in Python + PyTorch
2. **Infra Engineers**: Receive model structure, run it through with dummy data, and execute the following optimizations:
   - Parallelism optimization
   - Operator fusion
   - Low-precision training/inference (FP8/FP16/quantization)
   - Communication optimization
3. After optimization is complete, launch formal model training

## Skill Stack (High to Low Level)

| Layer | Content | Importance |
| ------- | --------- | ----------- |
| Algorithm & Models | LLM principles, KV Cache, etc. | Familiarity required |
| Training/Inference Frameworks | Megatron, DeepSpeed, vLLM, SGLang | Must master |
| PyTorch Framework | Underlying principles (view/permute/compile/autograd) | Must master |
| Distributed Communication Libraries | Compute-communication fusion, communication-computation hiding | Must master |
| Operator (Kernel) Development | Chip parallel computing; operator fusion; parallel acceleration | Need to master |
| Compilers | Especially relevant for domestic chips | Awareness sufficient |

## Key Performance Metrics (Inference)

- **TTFT (Time to First Token)**: Latency to first token
- **TPOT (Time per Output Token)**: Per-token inference latency
- **Total Latency = TTFT + TPOT × number of output tokens**
- **MFU (Model FLOPS Utilization)**: Compute utilization rate
- **Memory Bandwidth Utilization = Actual memory bandwidth / HBM peak bandwidth**

## Related Concepts

- [[gpu-software-stack]]
- [[gpu-memory-hierarchy]]
- [[llm-inference-metrics]]
- [[distributed-training]]
- [[mfu]]
- [[kv-cache]]

## Related Entities

- [[megatron]]
- [[deepspeed]]
- [[vllm]]
- [[sglang]]
- [[nccl]]
- [[flash-attention]]
- [[nvlink]]
