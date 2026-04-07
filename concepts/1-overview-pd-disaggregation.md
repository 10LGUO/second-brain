```yaml
---
title: PD Disaggregation (Prefill-Decode Disaggregation)
type: concept
tags: [inference, llm-serving, architecture, kv-cache, distributed-inference]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# PD Disaggregation (Prefill-Decode Disaggregation)

**PD disaggregation** (PD分离) refers to the architectural pattern of separating the two phases of LLM inference — the **prefill phase** and the **decode phase** — onto different hardware or service instances, allowing each to be independently scaled and optimized.

## Key Properties
