```yaml
title: Operator Fusion
type: concept
tags: [llm-infra, performance, kernel, gpu, optimization]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# Operator Fusion

Operator fusion is a GPU performance optimization technique that merges multiple separate compute operations (operators) into a single GPU kernel, reducing kernel launch overhead, memory round-trips, and improving overall throughput. It is one of the primary tools available to LLM infrastructure engineers for improving training and inference performance.

## Key Properties

- **What it does:** Instead of launching N separate kernels (each reading/writing to HBM), a fused kernel performs all N operations in a single pass, keeping data in on-chip cache (L1/L2 or registers) between operations.
- **Why it matters:** HBM bandwidth is the bottleneck for many LLM operations. Avoiding unnecessary HBM reads/writes directly improves performance.
- **Requires model understanding:** To fuse operators correctly, the engineer must understand the model architecture (e.g., which operations are always sequentially composed). This is why algorithm knowledge is a prerequisite for infra work.
- **At internet companies:** Most straightforward fusion schemes may already be implemented; the remaining gains require creative or operation-specific approaches.
- **Relationship to on-chip cache:** Fusion is only possible because on-chip cache is explicitly managed in kernel code — fused operations share data in registers/L1 rather than spilling to HBM.
- **Tooling:** Custom fused kernels can be written in CUDA C++ or Triton (used by PyTorch's Inductor compiler).

## Key Example

- **[[flash-attention]]** is a famous example of operator fusion applied to the attention mechanism — it fuses the QK^T matmul, softmax, and AV matmul into a single kernel to avoid materializing the full attention matrix in HBM.

## Variants

- **Graph-level fusion:** Performed by graph compilers (XLA, Inductor) automatically.
- **Manual kernel fusion:** Performed by infra engineers writing custom CUDA/Triton kernels.

## Related Concepts

- [[gpu-software-stack]]
- [[ai-chip-architecture]]
- [[compute-communication-overlap]]
- [[llm-training-infra]]
- [[llm-inference-infra]]
- [[flash-attention]]

## Sources

- [[1-overview]]

---
