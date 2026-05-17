```yaml
title: SLO Metrics for LLM Inference (TTFT, TPOT)
type: concept
tags: [inference, slo, latency, performance, llm-serving, transformer, kv-cache]
created: 2026-04-05
updated: 2026-05-17
sources: [1-overview.md]
```

# SLO Metrics for LLM Inference (TTFT, TPOT)

**SLO (Service Level Objective)** metrics define the service quality targets for LLM inference systems. The primary metrics are TTFT and TPOT, which together describe the end-to-end latency experienced by users.

## Key Properties

- **TTFT (Time To First Token):** Wall-clock time from when a request is received to when the first output token is returned to the user. Determines perceived responsiveness.
- **TPOT (Time Per Output Token):** Average time to generate each subsequent token after the first. Determines streaming smoothness.
- Together they govern the full user experience: a low TTFT feels responsive; a low TPOT feels fast during generation.

## Relationship to Transformer Architecture

TTFT and TPOT are a direct consequence of how Transformer-based models generate text autoregressively. The inference pipeline has two structurally distinct phases:

### Prefill → TTFT

The entire input prompt (N tokens) is processed in a single parallel forward pass through all Transformer layers. Every token attends to every other token simultaneously (full self-attention over the input). The result is:

1. The KV (Key-Value) cache is populated for all input tokens.
2. The first output token is produced.

The time this takes is **TTFT**. Because the full sequence is processed in parallel, prefill is **compute-bound** — throughput scales with arithmetic intensity (FLOP/s). Longer prompts increase TTFT proportionally.

### Decode → TPOT

After prefill, generation proceeds one token at a time. At each step:

1. The new token is passed through all Transformer layers.
2. Attention is computed between the new token and all prior tokens using the stored KV cache — no recomputation of past keys/values.
3. The next token is sampled and appended to the KV cache.

Each decode step generates exactly one token, but must read the **entire model weight matrix** from HBM (High-Bandwidth Memory) to compute even a single token's forward pass. This makes decode **memory-bandwidth-bound**: the GPU spends most of its time waiting for data to arrive from HBM, not computing. The average time per step is **TPOT**.

### Why the bottlenecks differ

| Phase | Tokens processed per step | Compute per byte loaded | Bottleneck |
|---|---|---|---|
| Prefill | N (all prompt tokens) | High (many tokens share the same weights) | Compute (FLOP/s) |
| Decode | 1 | Very low (one token, all weights still loaded) | Memory bandwidth (HBM GB/s) |

Increasing batch size during decode amortizes the weight-loading cost across more tokens per step, raising arithmetic intensity and improving TPOT at the cost of higher TTFT for individual requests.

## KV Cache Connection

[[1-overview-kv-cache]] is the mechanism that keeps TPOT tractable. Without it, each decode step would re-run the full self-attention over all prior tokens, making TPOT grow linearly with sequence length. With KV cache, only the current token's attention query is computed against cached keys and values — constant compute per step, but growing memory pressure.

## Latency vs. Throughput Trade-off

- Small batch size → low TTFT and TPOT for individual requests; low aggregate throughput.
- Large batch size → higher per-request latency; much higher aggregate tokens/second.
- **Continuous batching** (iteration-level scheduling) dynamically manages this trade-off by adding and removing requests at each decode step rather than waiting for a full batch to complete.

## Related Concepts

- [[1-overview-kv-cache]]
- [[1-overview-llm-inference-infra]]
- [[1-overview-pd-separation]]
- [[1-overview-memory-bandwidth-utilization]]
- [[1-overview-flash-attention]]

## Sources

- [[1-overview]]
