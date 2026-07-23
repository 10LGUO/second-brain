```yaml
title: KV Cache Quantization Project Walkthrough
type: source
tags: [kv-cache, quantization, int8, vllm, sglang, pageattention, inference]
created: 2026-07-04
updated: 2026-07-04
sources: []
```

# KV Cache Quantization Project Walkthrough

Practical implementation notes (from the SJTU AI Infra team / snow.liao) walking through how to actually build INT8 KV cache quantization into a vLLM/SGLang-style serving engine, covering both dynamic (validation) and static (production) quantization.

## 1. KV Cache Recap

- KV cache is one big contiguous tensor per engine, shape `[block, block_size, 2, head, head_dim]`, shared by all requests. `2` = K and V. `block_size` is tokens-per-block — too large wastes memory (internal fragmentation), too small hurts pageattention's access locality (block boundaries force discontiguous reads). Typical values: 32/64/128/256.
- Number of blocks is whatever HBM remains after fixed costs (model weights, CUDA graph buffers) — bigger blocks-worth of HBM directly means more concurrent requests the service can support, which is the commercial lever for an inference provider.
- Where KV cache is touched in the vLLM/SGLang pipeline:
  - **Prefill**: K/V produced by weight matmuls, written into the cache via a `reshape_and_cache`-style transpose op. The prefill attention compute itself reads straight from K/V (not from the cache) unless chunked prefill is enabled, in which case it also needs to read prior chunks' KV back out of the cache.
  - **PD-merged** setups pass KV cache straight to decode on the same machine/GPU. **PD-separated** setups must transfer KV cache across the network from prefill to decode nodes — quantizing KV cache shrinks that transfer, which is one of the concrete wins independent of compute speed.
  - **Decode**: reads KV cache via the pageattention kernel (this is where quantization's biggest performance win comes from — lower-precision cache means less memory traffic and more compute throughput), and writes newly-generated tokens' K/V back into the cache each step.
  - Speculative decoding adds a draft model's own KV cache to manage separately.

## 2. Dynamic Quantization (precision validation path)

- Goal: validate whether INT8 KV cache is accurate enough *before* investing in a real CUDA kernel. Do this at the PyTorch level (no compilation needed) rather than writing a kernel first.
- Method: get a golden fp16/bf16 PyTorch-equivalent implementation of pageattention (recovered from vLLM's kernel source + its own unit tests — the unit tests are the "gold" reference for what a fused kernel is supposed to compute), verify it matches, then progressively modify it into an INT8-equivalent PyTorch implementation, validate that end-to-end on a model, and only then write the real INT8 kernel against that validated reference.
- Quantize both `q` and `kvcache`: `q_int8, q_scale = int8_quant(q)`, `kvcache_int8, kvcache_scale = int8_quant(kvcache)`.
- Quantization granularity choice: **per-channel** (scale shape `[head, head_dim]`) is the practical sweet spot — finer than per-tensor/per-head (less quant error) while, unlike per-token, its scale doesn't depend on token count, so it can be precomputed/frozen for static quantization. Per-tensor risks a single outlier wrecking the whole cache's precision; per-token is fine for pure dynamic quant but can't be made static and hurts performance since every new token needs its own quant pass.
- Since PyTorch's `aten` ops have no native int8 arithmetic (only fp32/fp16/bf16, and eventually fp8), the dynamic-quant math must be *simulated* by dequantizing to float and multiplying scales back in — e.g. `q_int8.float() * k_int8.float() * q_scale * kvcache_scale` — while keeping the same numeric behavior a true int8 kernel would eventually produce. Softmax output stays fp32 (fp16 can overflow); the final `p @ v` step similarly simulates a fp32×int8 product with its own scale.
- **Gotcha**: per-channel scales cannot be pulled out as a common factor across the whole `q·k` dot product (unlike per-tensor scales), so naively multiplying `q_int8.float() * k_int8.float() * q_scale * kvcache_scale` is not mathematically equivalent to quantizing q and k independently per-channel. Fix: multiply q by k's scale first, then quantize the *product*, so the math stays equivalent.
- Recommended validation: run both per-head and per-channel variants, compare accuracy against un-quantized baseline on real benchmarks (aime, human-eval style, gpqa) using a small Qwen model (1.5B–7B) on a single GPU.

## 3. Static Quantization (production path)

- Why: dynamic quantization still stores a full fp16 KV cache and re-quantizes it into a second int8 buffer inside the pageattention kernel on every call — that's an extra half-a-cache's worth of peak memory and repeated quantization overhead. Static quantization instead makes the cache's native dtype int8 from initialization, so the engine gets ~2x the blocks (and thus ~2x the concurrency) for the same HBM, plus removes the runtime quantization pass entirely.
- vLLM framework changes required:
  - **(a)** Change the KV cache tensor's dtype to int8 in source.
  - **Prefill**: quantize K/V (`perchannel_quant`) before `reshape_and_cache` writes them into the int8 cache (a real kernel would fuse quantize + reshape_and_cache so this adds no extra latency). If chunked prefill is enabled, reading back prior chunks' KV requires a matching *dequantize* step so attention math still runs in fp16.
  - **Decode**: quantize `q` dynamically per step (q is never statically quantized — only the cache is), call `int8_pageattention_pytorch(q_int8, q_scale, kvcache_int8, scale)`; newly-generated K/V get quantized and written back the same way as prefill.
- Precision validation loop: (1) get a set of static quant params by running a single-GPU pass over a clean numeric dataset (aime recommended — pure math, low noise) and dumping the cache's value distribution to pick scale(s) (e.g. max value or a percentile); (2) wire those static params into the framework changes above and run full benchmarks. If accuracy craters, debug by substituting the *dynamic*-quant code path's live quantization with the *static* dumped scale — if that alone reproduces the accuracy drop, the static scale itself is wrong; if not, the framework-level change has a bug.
- Where the actual performance win comes from: prefill-side changes (reshape_and_cache fused with quantize, int8 cache transfer over the network in PD-separated setups) are close to a wash — they land on TPOT not TTFT since first-token generation doesn't depend on the cache write. The **decode-side int8 pageattention kernel** (int8 memory traffic + int8 tensor-core compute) is where essentially all the throughput win comes from, and it scales with sequence length — longer contexts benefit more. A full int8 gemm path additionally needs a dynamic in-kernel quantization of the post-softmax `p` before the `p @ v` matmul, so tensor cores can be used there too.
- Full production rollout also has to cover: (a) how quant params are dumped under distributed parallelism (data/tensor/expert parallel all affect what "the" per-channel scale should be, and how you pull it back out — max vs percentile — matters); (b) CUDA graph compatibility (does capturing a graph need extra handling given the changed cache dtype/shape?); (c) speculative decoding (should the draft model's KV cache be quantized too, and does it need isolating from the target model's cache in vLLM source, and what's the effect on acceptance rate?); (d) validating both PD-merged and PD-separated precision, plus how int8 KV cache shifts TTFT/TPOT under high concurrency in PD-separated mode; (e) whether a w16a16c8 (weight fp16, activation fp16, cache int8) setup is compatible with also moving to w8a8 or w4a8 weight/activation quantization, and why.

## Key Takeaways

- The whole validate-then-implement methodology: prove precision viability cheaply at the PyTorch level using existing unit tests as ground truth, *before* investing in a real fused CUDA kernel — this is the reusable pattern, not just a KV-cache-specific trick.
- Per-channel quantization is the practical default for KV cache because it's the finest granularity that can still be frozen into a static (token-count-independent) scale.
- The dominant performance lever is the decode-path pageattention kernel; prefill-side changes are mostly about memory footprint and (in PD-separated deployments) network transfer size, not raw speed.

## Related Concepts

- [[1-overview-kv-cache]]
- [[attention-mechanism]]
- [[1-overview-pd-separation]]
- [[1-overview-precision-convergence]]

## Related Entities

- [[1-overview-vllm]]
- [[1-overview-sglang]]
