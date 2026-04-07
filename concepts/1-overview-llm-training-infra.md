```yaml
title: LLM Training Infrastructure
type: concept
tags: [llm-infra, distributed-training, gpu, performance, precision]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# LLM Training Infrastructure

LLM training infrastructure refers to the full software and hardware stack required to train large language models at scale — spanning thousand-card to ten-thousand-card GPU/accelerator clusters. The core challenge is simultaneously managing compute efficiency, memory usage, inter-device communication, numerical precision, and fault tolerance, all at a scale where even tiny per-device failure rates compound into frequent aggregate failures.

## Key Properties
