```yaml
title: TransformerEngine (TE)
type: entity
tags: [llm-infra, fp8, operator-fusion, compute-communication-overlap, nvidia, library]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# TransformerEngine (TE)

TransformerEngine (TE) is NVIDIA's library for accelerating transformer model training and inference. It provides FP8 precision support, compute-communication hiding (overlap), and operator acceleration. It is used within Megatron and is now the preferred replacement for the older Apex library.

## Why It Matters to This Wiki

TE represents the current state of the art for low-precision and overlap optimizations in LLM training infrastructure. It is directly integrated into the must-master Megatron framework.

## Key Works, Products, or Contributions

- **FP8 training support:** Enables 8-bit floating point training, reducing memory and increasing compute throughput.
- **Compute-communication hiding:** Implements overlap of communication and computation to improve distributed training speedup ratio.
- **Operator acceleration library:** Provides optimized kernels for transformer operations.
- **Replaces Apex:** TE overlaps in functionality with the older Apex library; TE is now the preferred choice within Megatron.

## Related Entities and Concepts

- [[megatron]]
- [[compute-communication-overlap]]
- [[operator-fusion]]
- [[llm-training-infra]]

## Sources

- [[1-overview]]

---
