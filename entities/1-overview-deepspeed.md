```yaml
title: DeepSpeed
type: entity
tags: [llm-infra, distributed-training, framework, microsoft]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# DeepSpeed

DeepSpeed is Microsoft's open-source deep learning optimization library and distributed training framework for large language models. It is one of the two essential training frameworks for LLM infrastructure engineers (alongside Megatron).

## Why It Matters to This Wiki

DeepSpeed is identified as a must-master framework for LLM training infrastructure. It is used by both internet companies (forked and customized for internal use) and domestic chip vendors (adapted to non-NVIDIA hardware).

## Key Works, Products, or Contributions

- Implements ZeRO (Zero Redundancy Optimizer) for memory-efficient distributed training.
- Supports pipeline parallelism, tensor parallelism, and data parallelism.
- Widely used in production LLM training at internet companies alongside or as an alternative to Megatron.
- Target for adaptation by domestic chip vendors porting their hardware to the LLM ecosystem.

## Related Entities and Concepts

- [[megatron]]
- [[nccl]]
- [[llm-training-infra]]
- [[transformer-engine]]

## Sources

- [[1-overview]]

---
