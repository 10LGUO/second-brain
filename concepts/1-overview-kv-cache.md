---
title: KV Cache
type: concept
tags: [llm-infra, inference, attention, memory, optimization]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
---

# KV Cache

KV Cache (Key-Value Cache) is a fundamental optimization technique for LLM inference that caches the key and value tensors computed during the attention mechanism's prefill phase, so they do not need to be recomputed during the autoregressive decode phase. It is one of the most impactful single techniques in LLM inference optimization.

## Key Properties

- **What it does:** During inference, at each decode step, the model's attention layers need to attend over all previously generated tokens. Without caching, keys and values for all prior tokens would be recomputed at every step. KV cache stores these tensors in VRAM so they can be reused.
- **Why it matters:** Eliminates redundant computation during decoding, dramatically reducing TPOT (Time Per Output Token) and improving throughput.
- **Memory cost:** KV cache consumes VRAM proportional to batch size × sequence length × number of layers × number of heads × head dimension × precision. Memory optimization of KV cache is a major area of inference infra work.
- **Requires model understanding:** The inventors of KV cache needed to understand the transformer model architecture — cited in the source as an example of why infra engineers must understand algorithms.
- **Distributed KV cache:** At scale, KV cache management becomes a distributed systems problem. See [[pd-separation]].

## Relationship to PD Separation

- In PD (Prefill-Decode) separation architectures, the KV cache generated during prefill must be transferred between prefill nodes and decode nodes. Distributed KV cache management systems (lmcache for vLLM, Mooncake for SGLang) handle this.

## Related Concepts

- [[pd-separation]]
- [[llm-inference-infra]]
- [[flash-attention]]
- [[gpu-software-stack]]

## Sources

- [[1-overview]]

---
