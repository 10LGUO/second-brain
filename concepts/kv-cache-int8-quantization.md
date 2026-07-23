```yaml
title: KV Cache INT8 Quantization
type: concept
tags: [kv-cache, quantization, int8, inference, pageattention]
created: 2026-07-04
updated: 2026-07-04
sources: [kv_cache_quantization_example.md]
```

# KV Cache INT8 Quantization

Storing and computing [[1-overview-kv-cache]] in INT8 instead of fp16/bf16, to cut its memory footprint (more concurrency headroom) and speed up the decode-phase pageattention kernel (less memory traffic, int8 tensor-core compute). Comes in two flavors: dynamic (quantize a live fp16 cache on the fly, used to validate precision cheaply) and static (cache is int8-native from allocation, used in production for the memory and latency win).

## Key Properties

- **Granularity**: per-channel (scale shape `[head, head_dim]`) is the practical default — finer than per-tensor/per-head so quant error is small, and unlike per-token its scale doesn't depend on token count, so it can be frozen ("static") ahead of time. Per-tensor risks one outlier degrading the whole cache; per-token can't be static and adds per-step quant overhead.
- **Validation-before-kernel methodology**: prove INT8 precision is viable in a plain PyTorch simulation (dequantize-and-multiply, since `aten` has no native int8 ops) before writing a real fused CUDA kernel — anchored against the fp16 kernel's own unit tests as ground truth.
- **Scale non-associativity gotcha**: with per-channel scales, `q_int8.float() * k_int8.float() * q_scale * kvcache_scale` is *not* equivalent to independently quantizing q and k per-channel, because the scale can't be factored out of the dot product the same way a per-tensor scale can. Fix: multiply q by k's scale first, then quantize that product.
- **Where the speed win comes from**: almost entirely the decode-side int8 pageattention kernel (int8 memory + int8 compute), which scales with sequence length. Prefill-side changes mostly save memory/network transfer (relevant for PD-separated deployments), not raw prefill speed.
- **Static quantization's memory win**: making the cache int8-native from init (rather than fp16-with-runtime-quant) roughly doubles the number of cache blocks that fit in HBM, directly increasing max concurrency.

## Related Concepts

- [[quantization-fundamentals]]
- [[1-overview-kv-cache]]
- [[attention-mechanism]]
- [[1-overview-pd-separation]]
- [[1-overview-precision-convergence]]

## Related Entities

- [[1-overview-vllm]]
- [[1-overview-sglang]]

## Sources

- [[kv_cache_quantization_example]]
