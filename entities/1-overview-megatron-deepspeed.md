```yaml
title: Megatron and DeepSpeed
type: entity
tags: [framework, training, distributed-training, llm, ai-infra]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# Megatron and DeepSpeed

**Megatron-LM** (developed by NVIDIA) and **DeepSpeed** (developed by Microsoft) are the two dominant open-source frameworks for large-scale distributed LLM training. Together they are considered to contain "almost all mainstream LLM distributed training features" and are marked as **must-master** tools for AI infra engineers.

## Why It Matters to This Wiki

Megatron and DeepSpeed are the primary training framework layer in the AI infra stack. Understanding their internals is required for performance optimization, precision debugging, and adapting training to new hardware including domestic chips. Both internet companies and chip vendors fork and modify these frameworks for internal use.

## Key Works, Products, and Contributions
