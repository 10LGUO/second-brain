```yaml
title: PD Separation (Prefill-Decode Separation)
type: concept
tags: [llm-infra, inference, distributed-inference, architecture, kv-cache]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# PD Separation (Prefill-Decode Separation)

PD Separation is a distributed LLM inference architecture pattern in which the **prefill phase** (processing the input prompt) and the **decode phase** (autoregressive token generation) are handled by separate groups of machines/nodes, each optimized for their respective compute characteristics.

## Key Properties

- **Motivation:** Prefill is compute-bound (processes many tokens in parallel); decode is memory-bandwidth-bound (generates one token at a time, repeatedly reading KV cache). Running them on the same hardware forces suboptimal trade-offs for both.
- **Architecture:** Dedicated prefill nodes handle prompt processing and generate the initial KV cache; dedicated decode nodes handle autoregressive generation, consuming the KV cache.
- **Distributed KV cache:** A critical component — KV cache generated on prefill nodes must be efficiently transferred to and managed on decode nodes. This is a major distributed systems engineering challenge.
- **Implementations:**
  - [[vllm]] + **lmcache**: lmcache is vLLM's PD-separation component focused on distributed KV cache management.
  - [[sglang]] + **Mooncake**: Mooncake is SGLang's PD-separation component with the same focus.

## Related Concepts

- [[kv-cache]]
- [[llm-inference-infra]]
- [[vllm]]
- [[sglang]]

## Sources

- [[1-overview]]

---
