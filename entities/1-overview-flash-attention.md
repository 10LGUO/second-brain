```yaml
title: Flash-Attention
type: entity
tags: [llm-infra, attention, operator-fusion, inference, training, library]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# Flash-Attention

Flash-Attention is a highly optimized attention kernel that uses operator fusion and memory-aware tiling to compute the attention mechanism significantly faster and with lower memory usage than the standard implementation.

## Why It Matters to This Wiki

Flash-Attention is identified as a must-master library for LLM infrastructure engineers. It is a canonical example of operator fusion — the fused kernel avoids materializing the full attention matrix in HBM, dramatically reducing memory bandwidth consumption. It is used in both training (Megatron integrates it) and inference.

## Key Works, Products, or Contributions

- Fuses QK^T matmul, softmax, and AV matmul into a single GPU kernel.
- Uses tiling to keep intermediate results in on-chip SRAM (L1/L2), avoiding HBM round-trips.
- Reduces memory complexity of attention from O(N²) to O(N) in practice.
- Enables training and inference on longer sequences than standard attention allows.
- Integrated into Megatron and most modern LLM inference engines.

## Related Entities and Concepts

- [[operator-fusion]]
- [[kv-cache]]
- [[llm-training-infra]]
- [[llm-inference-infra]]
- [[megatron]]
- [[ai-chip-architecture]]

## Sources

- [[1-overview]]

---
