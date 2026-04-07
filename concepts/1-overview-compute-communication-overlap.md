```markdown
---
title: Compute-Communication Overlap (通算融合 / 通信计算隐藏)
type: concept
tags: [distributed-systems, performance-optimization, ai-infra, training, inference, communication]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
---

# Compute-Communication Overlap (通算融合 / 通信计算隐藏)

Compute-communication overlap is the technique of asynchronously executing communication operations (data transfer between GPUs or nodes) and compute operations (matrix multiplications, attention, etc.) simultaneously, so that communication latency is hidden behind compute time. It is one of the most important performance optimization techniques in large-scale distributed training and inference.

## Motivation

In multi-card (multi-GPU) distributed training and inference, computation and communication are both necessary but their sequential execution wastes time. Communication that **cannot be fully hidden** directly reduces the [[speedup-ratio]], making scale-out less effective.

Under the von Neumann architecture perspective used in AI infra:
- **Compute flow (计算流):** Fill compute units with work at all times.
- **Data flow (数据流):** Move data as fast as possible, minimizing resource usage.
- The goal is to **overlap data flow and compute flow** via async execution so that each hides the latency of the other.

## How It Works

- Computation and communication are dispatched to separate hardware units (compute cores vs. NVLink/network interconnects) and issued asynchronously (e.g., via different CUDA streams).
- While the GPU computes on one partition of the data, it simultaneously sends/receives another partition's activations or gradients over [[nvlink]] or InfiniBand.
- Synchronization only occurs at necessary dependency points.

## Variants and Terminology

- **通算融合 (compute-communication fusion):** Fusing communication and compute at the operator level — the communication is embedded in the operator itself rather than being a separate step.
- **通信计算隐藏 (communication-compute hiding):** The broader technique of scheduling communication to overlap with compute, reducing exposed latency.

## Importance

- Dominant optimization direction in current large-scale LLM training.
- Much of the complexity in frameworks like Megatron and DeepSpeed is devoted to implementing correct and efficient compute-communication overlap.
- Distributed communication libraries ([[nccl]]) are a mandatory skill for infra engineers specifically because of compute-communication overlap work.

## Speedup Ratio Impact

```

Speedup ratio = Multi-card performance / Single-card performance

```text
- If communication is fully hidden: speedup ratio approaches the theoretical maximum.
- If communication is not hidden: exposed communication time reduces effective compute time, lowering the speedup ratio.
- When a model fits on one card, distributed is not used because communication overhead reduces rather than improves performance.

## Related Concepts

- [[large-model-infra]]
- [[gpu-software-hardware-architecture]]
- [[speedup-ratio]]
- [[operator-development]]
- [[hbm-high-bandwidth-memory]]

## Sources

- [[1-overview]]
```

---
