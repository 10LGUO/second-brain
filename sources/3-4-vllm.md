```yaml
title: "Lecture 3/4 — Key Features of the vLLM Inference Engine"
type: source
tags: [inference, llm, vllm, continuous-batching, kv-cache, paged-attention, quantization, speculative-decoding, chunked-prefill, scheduling, tensor-parallelism, pipeline-parallelism]
created: 2026-06-22
updated: 2026-06-22
sources: [3:4 - vllm.pdf]
```

# Lecture 3/4 — Key Features of the vLLM Inference Engine

Source: SJTU AI Infra Team, lecture series.

This lecture covers the key systems-level features that make vLLM a high-throughput LLM inference engine: continuous batching, KV cache management, PagedAttention, scheduling, quantization, speculative decoding, and multi-node/multi-GPU serving.

---

## 1. Continuous Batching

### Background

LLM inference has two phases:
- **Prefill**: process the input prompt, computing the KV cache for all tokens at once — compute-intensive.
- **Decode**: generate token by token, one token per step — memory-bandwidth-intensive.

Traditional **static batching**: pack a batch of requests and wait until all of them finish before accepting the next batch. Problem: sequence lengths differ, so once short sequences finish the GPU idles waiting for the long ones — low utilization.

### How Continuous Batching Works

**Continuous batching** (also called iteration-level batching): after each decode step, immediately replace finished requests with new ones instead of waiting for the whole batch to complete.

```
Step 1: [Req A, Req B, Req C, Req D]
Step 2: [Req A, Req B, Req C, Req E]  ← D finished, E inserted
Step 3: [Req A, Req F, Req C, Req E]  ← B finished, F inserted
```

Effect: the GPU stays fully loaded, greatly improving throughput (the vLLM paper reports ~23× over FasterTransformer).

### The Prefill/Decode Conflict

When a batch mixes prefill (large compute) and decode (small compute) requests, the prefill request preempts the GPU and increases decode latency (the tension between Time to First Token and Inter-Token Latency). Chunked Prefill (below) addresses this.

---

## 2. Scheduling

### Scheduling Goals

- Maximize GPU utilization
- Bound request latency (SLA constraint)
- Prevent KV cache memory OOM

### vLLM Scheduler Behavior

vLLM uses a **First-Come-First-Served (FCFS)** policy, combined with admission control based on the number of available KV cache blocks:

1. A new request enters the waiting queue.
2. The scheduler checks whether enough KV cache blocks remain.
3. If enough, the request is moved to the running queue.
4. If not, a lower-priority request is **preempted**, freeing its KV cache blocks.
5. The preempted request returns to the waiting queue and re-does prefill when scheduled again.

**Swap**: preempted KV cache can be swapped to CPU memory first to avoid re-prefilling, but swap bandwidth cost is high and results are mixed in practice.

### Scheduling Granularity

- Before **Chunked Prefill**: a prefill request is a single scheduling unit as a whole.
- After **Chunked Prefill**: prefill can be split into multiple chunks and interleaved with decode to balance latency.

---

## 3. KV Cache Management

### KV Cache Memory Pressure

KV cache size per token:

```
2 × num_layers × num_heads × head_dim × sizeof(dtype)
```

For Llama-3 8B (BF16):
- 32 layers × 32 heads × 128 head_dim × 2 bytes × 2 (K+V) = **512 KB/token**
- 4096-token context = **2 GB**
- 100 concurrent requests = **200 GB**

### Fragmentation Problem

Different requests have different sequence lengths. The traditional approach **pre-allocates contiguous memory for the maximum length** per request:
- Internal fragmentation: space a request doesn't use is wasted.
- External fragmentation: non-contiguous memory can't be given to new requests.
- No sharing: requests with the same prompt store the KV redundantly.

---

## 4. PagedAttention

### Core Idea

Borrowing the operating system's **virtual memory paging** mechanism, split the KV cache into fixed-size **blocks** that need not be physically contiguous.

- **Block size**: typically 16 or 32 tokens.
- **Block table**: each request maintains a logical-block-number → physical-block-number mapping.
- **Physical block pool**: a pool of physical blocks pre-allocated in GPU memory.

```
Logical KV:  [Block 0] [Block 1] [Block 2] [Block 3]
                ↓         ↓         ↓         ↓
Physical:    [Slot 7]  [Slot 2]  [Slot 9]  [Slot 1]   ← non-contiguous
```

### Advantages

| Problem | Traditional approach | PagedAttention |
|---|---|---|
| Internal fragmentation | Pre-allocate max length, heavy waste | Allocate blocks on demand, waste ≤ 1 block |
| External fragmentation | Contiguous allocation, ~20-30% fragmentation | Blocks reused freely, fragmentation ≈ 0 |
| Prompt sharing | Each request stores independently | Same-prompt blocks shared via Copy-on-Write |

### Copy-on-Write

Multiple requests share the same prompt's physical blocks. When a request needs to write (generate a new token), CoW triggers: copy the block to a new physical block, then write.

Use cases:
- **Parallel sampling**: multiple outputs from one prompt (beam search, best-of-N).
- **Shared system prompt**: multiple requests share the same system prompt.

### PagedAttention Kernel

A standard attention kernel assumes contiguous KV storage; PagedAttention needs a custom CUDA kernel that addresses memory indirectly through the block table. vLLM implemented two versions:
- `paged_attention_v1`: direct implementation.
- `paged_attention_v2`: a divide-and-conquer version for long sequences.

---

## 5. Chunked Prefill

### Problem

In continuous batching, one long prefill request (e.g. 8K tokens) monopolizes a step, causing inter-token latency (ITL) spikes for the decode requests in the same batch.

### Solution

Split prefill into fixed-size chunks (e.g. 512 tokens); each step processes only one chunk, packed together with decode tokens:

```
Step 1: [prefill chunk 0~511, decode A, decode B, decode C]
Step 2: [prefill chunk 512~1023, decode A, decode B, decode C]
...
```

Effect:
- Decode requests are no longer blocked by a long prefill; ITL is more stable.
- Compute density stays high (chunk + decode tokens merged into one forward pass).
- Cost: the prefill's TTFT (Time to First Token) increases.

---

## 6. Quantization

### Why Quantize

- Reduce model weight memory (BF16 → INT8 saves 50%, INT4 saves 75%).
- Reduce KV cache memory (FP8 KV cache).
- Increase compute throughput (INT8/FP8 Tensor Core peak throughput is higher).

### Common Schemes

| Scheme | Precision | Target | Notes |
|---|---|---|---|
| AWQ | INT4 | weights | Protects salient weights, small accuracy loss |
| GPTQ | INT4/INT8 | weights | Hessian-based layer-wise quantization |
| SmoothQuant | INT8 | weights + activations | Shifts activation quantization difficulty onto weights |
| FP8 (W8A8) | FP8 | weights + activations | Native H100 support, accuracy close to BF16 |
| FP8 KV Cache | FP8 | KV cache | Reduces KV memory, small accuracy loss |

### KV Cache Quantization

Quantizing the KV cache to FP8 halves KV memory, enabling longer context or larger batches. vLLM supports both per-tensor and per-channel FP8 KV cache quantization.

---

## 7. Speculative Decoding

### Principle

LLM decode is **memory-bandwidth-bound** (only 1 token generated per step, much GPU compute idle). Speculative decoding uses a small **draft model** to quickly generate several candidate tokens, then a large **target model** verifies them in one pass:

```
Draft model:  token₁, token₂, token₃, token₄, token₅
Target model: one forward pass verifies all 5 tokens
Accept token₁~token₄, reject token₅, resample token₅'
Net effect: 1 target forward ≈ 4 tokens produced
```

### Acceptance Rate and Speedup

Let the average acceptance rate of draft tokens be α, with k draft tokens generated per speculation:

```
expected tokens per step ≈ (1 - αᵏ⁺¹) / (1 - α)
```

Speedup depends on:
- α
- draft model speed
- target model batch size (speculation gains fall off at large batch)

### Draft Model Sources

- **Independent small model**: e.g. Llama 3.2 1B drafting for Llama 3.1 70B.
- **EAGLE / Medusa**: add lightweight draft heads inside the target model, sharing the KV cache — near-zero draft overhead.
- **Ngram lookup**: find repeated ngrams in already-generated text to use as drafts.

---

## 8. Tensor Parallelism & Pipeline Parallelism

### Tensor Parallelism (TP)

Split a single layer's weight matrices by column/row across multiple GPUs; each GPU computes a partial result, merged via AllReduce.

- TP=4: 4 GPUs each hold 1/4 of the attention heads and FFN weights.
- One AllReduce per layer (actually 2: after attention + after FFN).
- Latency grows with TP; usually TP ≤ GPUs per node (to avoid cross-node AllReduce).

### Pipeline Parallelism (PP)

Split the model by depth; different GPUs handle different layers.

- PP=4: GPU0 handles layers 0-7, GPU1 handles layers 8-15, …
- Low communication volume, suitable across nodes.
- Downside: pipeline bubbles — GPUs incur waiting time.
- Combine with micro-batching to reduce bubbles.

### Practical Combinations

Typical large-model configs cover all GPUs with TP × PP:

| Model | Typical config |
|---|---|
| 70B, 8×A100 | TP=8, PP=1 |
| 405B, 32×H100 | TP=8, PP=4 |

---

## 9. Prefix Caching

For multiple requests sharing the same prefix (e.g. a system prompt), vLLM can reuse the already-computed KV cache blocks:

- Compute a hash of each block's token sequence as the cache key.
- When a new request arrives, look up block hashes; on a hit, reuse directly and skip prefill.
- An LRU eviction policy manages the cache.

Effect: when the system prompt is a large fraction of total length (e.g. RAG, long system prompts), it substantially lowers TTFT and compute.

---

## 10. Key Metrics

| Metric | Full name | Meaning |
|---|---|---|
| TTFT | Time to First Token | Time from request to first output token; affected by prefill |
| ITL / TPOT | Inter-Token Latency / Time Per Output Token | Time to generate each token during decode |
| Throughput | — | Total output tokens per second across the system |
| Goodput | — | Effective throughput that meets SLA constraints |

There is a tradeoff between TTFT and ITL: larger batch size raises throughput but increases ITL; Chunked Prefill mitigates the impact of TTFT on ITL.

---

## Summary

vLLM's core design philosophy is to **manage GPU memory the way an operating system manages memory**:

1. **Continuous batching** — eliminates the waiting waste of static batching
2. **PagedAttention** — eliminates KV cache fragmentation and enables sharing
3. **Chunked Prefill** — balances TTFT and ITL
4. **Prefix caching** — reuses common prefixes, cutting redundant compute
5. **Speculative decoding** — uses idle compute to accelerate memory-bound decode
6. **Quantization** — reduces memory and bandwidth pressure

Stacked together, these techniques give vLLM a 10-30× throughput improvement over a naive implementation.
