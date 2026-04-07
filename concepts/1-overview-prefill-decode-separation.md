```yaml
title: Prefill-Decode Separation (PD分离)
type: concept
tags: [inference, llm, vllm, sglang, kv-cache, distributed-inference, ai-infra]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# Prefill-Decode Separation (PD分离)

Prefill-Decode (PD) separation is an architectural pattern in LLM inference that separates the **prefill phase** (processing the input prompt to generate the initial KV cache) from the **decode phase** (autoregressive token generation) into distinct compute stages, often running on separate hardware or services. This separation enables independent scaling and optimization of each phase.

## Background

LLM inference consists of two distinct phases with very different compute characteristics:

- **Prefill:** Processes the entire input prompt in parallel (compute-bound); generates the KV cache for all input tokens.
- **Decode:** Generates output tokens one at a time autoregressively (memory-bandwidth-bound); reads the KV cache on every step.

Because these phases have different bottlenecks, running them on the same hardware simultaneously leads to resource contention and suboptimal utilization.

## How PD Separation Helps

- Allows prefill and decode to run on hardware optimized for their respective bottlenecks.
- Enables independent scaling: more prefill capacity for high-input-volume workloads, more decode capacity for high-output-volume workloads.
- Improves [[slo]] metrics ([[ttft]] and [[tpot]]) by reducing interference between phases.
- Requires efficient **distributed KV cache** management: the KV cache generated during prefill must be transferred to the decode workers.

## Implementations

| Inference Engine | PD Separation Component | Focus |
| --- | --- | --- |
| **vLLM** | **lmcache** | Distributed KV cache |
| **SGLang** | **Mooncake** | Distributed KV cache |

Both lmcache and Mooncake focus on efficiently managing and transferring the distributed KV cache between prefill and decode workers.

## Related Concepts

- [[large-model-infra]]
- [[kv-cache]]
- [[ttft]]
- [[tpot]]
- [[slo]]

## Related Entities

- [[vllm]]
- [[sglang]]

## Sources

- [[1-overview]]

```text

---
