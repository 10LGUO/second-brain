```yaml
title: LLM Inference — Prefill and Decode Phases
type: concept
tags: [inference, prefill, decode, kv-cache, continuous-batching, generation, llm, memory-bandwidth, compute-bound]
created: 2026-06-03
updated: 2026-06-03
sources: [10-nanovllm.md]
```

# LLM Inference — Prefill and Decode Phases

LLM autoregressive generation has two fundamentally different phases with different compute characteristics and optimization strategies.

## The Two Phases

### Prefill

- **Input:** Full prompt (P tokens) processed in one forward pass
- **Output:** Logits for the first output token + KV cache populated for all P positions
- **Compute character:** **Compute-bound** — large matrix multiplications (Q, K, V projections over P tokens) saturate Tensor Cores
- **Bottleneck:** FLOP/s

```
Input:  [token_1, token_2, ..., token_P]   (all at once)
Output: logits + KV cache[0..P-1]
```

### Decode

- **Input:** Single new token (the previously generated one)
- **Output:** Logits for the next token + KV cache extended by one position
- **Compute character:** **Memory-bandwidth-bound** — one token means tiny matmuls (batch=1, seq=1) that cannot saturate Tensor Cores; the bottleneck is reading all model weights and the KV cache from HBM each step
- **Bottleneck:** HBM bandwidth

```
Step 1: [token_P+1]  → logits, cache grows to P+1
Step 2: [token_P+2]  → logits, cache grows to P+2
...
```

## Why Decode is Memory-Bandwidth-Bound

At each decode step, the GPU must read:
- All model weights (~2× param_count bytes for BF16)
- Full KV cache for all previous positions

For a 7B parameter model in BF16: ~14 GB of weights per step. At H100 HBM bandwidth of 3.35 TB/s, this takes ~4 ms — regardless of batch size 1 or 32, because the weights are the same. The arithmetic work (matmul with a single token vector) completes in microseconds.

**Implication:** Increasing batch size during decode is nearly free until the batch is large enough to make the matmuls compute-bound. Production engines maximize batch size precisely to amortize the weight reads across many concurrent requests.

## KV Cache Growth

The KV cache grows by one position per decode step per layer:

```
After prefill (P tokens):
  cache shape: [num_layers, 2, batch, num_kv_heads, P, head_dim]

After T decode steps:
  cache shape: [num_layers, 2, batch, num_kv_heads, P+T, head_dim]
```

Memory per token in cache (one layer):
```
2 × num_kv_heads × head_dim × bytes_per_element
= 2 × 8 × 128 × 2 bytes (BF16) = 4096 bytes = 4 KB per layer
```

For 32 layers: 128 KB per token. At 4096 max tokens and batch=32: 32 × 4096 × 128 KB = 16 GB — often the dominant memory consumer, leaving little room for model weights.

## Sampling Strategies

After computing logits, the next token is sampled:

| Strategy | Formula | Notes |
|---|---|---|
| Greedy | `argmax(logits)` | Deterministic, no diversity |
| Temperature | `sample(softmax(logits / T))` | T<1 sharpens, T>1 flattens |
| Top-k | Sample from top-k probability mass | Limits vocabulary |
| Top-p (nucleus) | Sample from smallest set with cumulative prob ≥ p | Adapts to distribution shape |

Temperature is applied before top-k/top-p filtering.

## Generation Loop Skeleton

```python
def generate(model, input_ids, max_new_tokens, temperature=1.0, top_p=0.9):
    past_len = 0
    kv_cache = None

    # Prefill
    logits, kv_cache = model(input_ids, past_len=0, kv_cache=None)
    next_token = sample(logits[:, -1, :], temperature, top_p)
    past_len = input_ids.shape[1]
    generated = [next_token.item()]

    # Decode
    for _ in range(max_new_tokens - 1):
        logits, kv_cache = model(
            next_token.unsqueeze(0), past_len=past_len, kv_cache=kv_cache
        )
        next_token = sample(logits[:, -1, :], temperature, top_p)
        generated.append(next_token.item())
        past_len += 1
        if next_token.item() == EOS_TOKEN_ID:
            break

    return generated
```

## Production Extensions

NanoVLLM's simple loop becomes a full engine by adding:

- **Continuous batching:** Requests join and leave mid-generation; the batch is never stalled waiting for the slowest sequence to finish. Maximizes GPU utilization.
- **Paged attention:** KV cache stored in fixed-size non-contiguous pages (like OS virtual memory); eliminates fragmentation and allows fine-grained memory sharing.
- **Speculative decoding:** A small draft model proposes K tokens; the large model verifies all K in one parallel forward pass. Speeds up memory-bandwidth-bound decode by increasing tokens-per-step.
- **Chunked prefill:** Interleave prefill chunks with decode steps to reduce TTFT while keeping GPU busy.

## Cross-References

- [[10-nanovllm]] — full lecture with code
- [[10-nanovllm-model-loading]] — model architecture and weight loading
- [[1-overview-kv-cache]] — paged attention and KV cache management
- [[1-overview-slo-metrics-ttft-tpot]] — TTFT / TPOT and why prefill/decode distinction matters
- [[1-overview-memory-bandwidth-utilization]] — why decode is memory-bandwidth-bound
- [[8-memory-opt-flash-attention]] — efficient attention for both phases
