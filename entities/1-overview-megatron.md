```yaml
title: Megatron
type: entity
tags: [framework, training, distributed, llm, ai-infra]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# Megatron

Megatron (developed by NVIDIA Research) is a distributed training framework for large language models that contains virtually all mainstream LLM distributed training techniques. It is considered one of the two essential training frameworks (alongside [[deepspeed]]) that AI infra engineers must master.

## Why It Matters to This Wiki

Megatron is a primary reference implementation for distributed LLM training techniques. Both internet companies and domestic chip vendors adapt and extend Megatron for their own use (forking to develop proprietary features). It integrates with [[transformerengine]] for FP8 support and compute-communication overlap, and with [[apex]] for operator acceleration (though TransformerEngine is now preferred over Apex).

## Key Features

- Contains virtually all mainstream LLM distributed training features
- Pipeline parallelism, tensor parallelism, data parallelism
- Integrates TransformerEngine (TE) for FP8 training and compute-communication overlap
- Integrates Apex for operator acceleration (legacy; TE preferred now)
- Used with [[nccl]] for distributed communication

## Skill Classification

- Layer 2 in the AI infra full-stack skill hierarchy (training/inference framework layer)
- Must master for training infra engineering

## Related Entities

- [[deepspeed]]
- [[nccl]]
- [[flash-attention]]
- [[transformerengine]]
- [[vllm]]
- [[sglang]]

## Related Concepts

- [[large-model-infra]]
- [[compute-communication-overlap]]

## Sources

- [[1-overview]]

```text

---
