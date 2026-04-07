```yaml
title: vLLM
type: entity
tags: [framework, inference, llm, distributed-inference, ai-infra]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# vLLM

vLLM is a mainstream LLM inference engine and one of the two primary inference frameworks (alongside [[sglang]]) that AI infra engineers must master. It supports high-throughput and low-latency LLM serving.

## Why It Matters to This Wiki

vLLM is a primary reference implementation for distributed LLM inference. It implements PagedAttention (KV cache management), continuous batching, and other inference optimization techniques. Together with [[sglang]], it represents the current convergence point of LLM inference framework technology.

## Key Features

- High-throughput LLM serving with efficient KV cache management
- Continuous batching for improved GPU utilization
- Distributed inference support

## PD Separation Component

- **lmcache:** vLLM's [[prefill-decode-separation]] component; focuses on distributed [[kv-cache]] management between prefill and decode workers.

## Skill Classification

- Layer 2 in the AI infra full-stack skill hierarchy (training/inference framework layer)
- Must master at least one of vLLM or SGLang for inference infra engineering

## Related Entities

- [[sglang]]
- [[megatron]]
- [[nccl]]
- [[flash-attention]]

## Related Concepts

- [[large-model-infra]]
- [[prefill-decode-separation]]
- [[kv-cache]]
- [[inference-latency-metrics]]

## Sources

- [[1-overview]]

```text

---
