---
title: LLM Inference Infrastructure
type: concept
tags: [llm-infra, inference, distributed-inference, slo, kv-cache, pd-separation]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
---

# LLM Inference Infrastructure

LLM inference infrastructure refers to the systems and frameworks used to serve large language model responses to end users at scale. The primary goals are high performance (throughput and latency) and high accuracy (numerical precision). Single-card inference has few engineering challenges; distributed inference across multiple devices introduces substantial complexity.

## Key Properties
