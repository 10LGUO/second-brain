```yaml
title: SGLang
type: entity
tags: [framework, inference, llm, distributed-inference, ai-infra]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# SGLang

SGLang is a mainstream LLM inference engine and one of the two primary inference frameworks (alongside [[1-overview-vllm]]) that AI infra engineers must master. It is particularly notable for its structured generation capabilities and its PD separation component, Mooncake.

## Why It Matters to This Wiki

SGLang represents the current convergence of LLM inference framework technology alongside vLLM. Its PD separation approach via Mooncake is a notable architectural pattern for scalable inference deployment.

## Key Features

- High-performance LLM serving
- Structured generation support
- Distributed inference

## PD Separation Component

- **Mooncake:** SGLang's [[1-overview-prefill-decode-separation]] component; focuses on distributed [[1-overview-kv-cache]] management between prefill and decode workers.

## Skill Classification

- Layer 2 in the AI infra full-stack skill hierarchy (training/inference framework layer)
- Must master at least one of SGLang or vLLM for inference infra engineering

## Related Entities

- [[1-overview-vllm]]
- [[1-overview-megatron]]
- [[1-overview-nccl]]
- [[1-overview-flash-attention]]

## Related Concepts

- [[1-overview-large-model-infra]]
- [[1-overview-prefill-decode-separation]]
- [[1-overview-kv-cache]]
- [[1-overview-slo-metrics-ttft-tpot]]

## Sources

- [[1-overview]]

```text

---
