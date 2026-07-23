```yaml
title: Quantization Fundamentals
type: concept
tags: [quantization, int8, inference, precision, kv-cache, tensor-cores]
created: 2026-07-19
updated: 2026-07-19
sources: [kv_cache_quantization_example.md]
```

# Quantization Fundamentals

Quantization maps floating-point values to low-bit integers (typically int8) so tensors are cheaper to store and move, and so integer tensor-core paths can be used for compute. A fp value `x` becomes an integer code `q = round(x / scale)` (plus a zero-point in the asymmetric case); dequantization recovers `q * scale ≈ x`. The `round()` step is lossy and its loss is permanent — everything downstream (granularity choice, calibration, kernel design) is about keeping that loss small enough not to hurt model quality, while actually cashing in the memory/compute win.

## Symmetric vs Asymmetric

- **Symmetric**: `q = clamp(round(x / scale), -128, 127)`, `scale = max|x| / 127`. The representable range is centered on zero. Simplest math — no cross-terms in matmuls — and the standard choice for KV cache and most LLM inference work. See the round trip at `paged_attention_ref.py:45-46`:

  ```python
  scale = x.abs().amax(dim=0, keepdim=True).clamp_min(1e-8) / 127.0
  return ((x / scale).round().clamp(-128, 127) * scale).to(x.dtype)
  ```

- **Asymmetric**: adds a zero-point `z` so `q = round(x / scale) + z`, using the full [-128, 127] range for skewed distributions (e.g. post-ReLU activations, all ≥ 0). More accurate for one-sided data, but the zero-point introduces extra cross-terms in integer matmul, so symmetric is preferred when distributions are roughly zero-centered — which K/V activations are.
- int8 range is [-128, 127]; symmetric schemes usually target ±127 so the range is mirror-symmetric.
- `clamp_min(1e-8)` on the scale guards the all-zeros group — otherwise `x / scale` divides by zero.

## Granularity

Granularity = which slice of the tensor shares one scale. Finer granularity → less quant error but more scale metadata and more constraints on kernels and static freezing.

| Granularity | Scale shape (KV cache `[block, block_size, 2, head, head_dim]`) | Notes |
|---|---|---|
| per-tensor | scalar | One outlier anywhere wrecks precision everywhere |
| per-token | `[tokens]` | Fine, but scale count grows with sequence — cannot be frozen statically |
| per-head | `[head]` | Coarse; scales factor cleanly out of dot products |
| per-channel | `[head, head_dim]` | Practical KV-cache default: fine, and token-count-independent so it can be frozen |
| per-group | e.g. groups of 64–128 channels | Weight-quant staple (GPTQ/AWQ style); between per-channel and per-tensor |

**How to choose — the outlier-ratio method** (`analyze_kv_granularity.py:52-57`): for each candidate axis, group the tensor by the dims that would get their own scale, take max-abs per group, and compute `max(group-max) / median(group-max)`.

- High ratio → groups along that axis have very different magnitudes, so a shared scale forces small groups to share range with an outlier group. Specializing a scale per group on that axis buys real precision.
- Low ratio → groups already look alike; splitting that axis gains little.
- Per-tensor is the trivial single-group baseline (ratio ≡ 1.0).

For Qwen2.5-7B, K's outlier ratio is channel-dominated, which is why `int8_roundtrip_per_channel` in `paged_attention_ref.py:38-46` shares one scale across tokens (dim 0) and specializes per `(head, head_dim)` — the axis with outlier structure gets its own scale; the axis without one shares.

## Dynamic vs Static

- **Dynamic**: scale computed at runtime from the live data (`x.abs().amax(...)` right before quantizing). Best possible accuracy per granularity — the scale always fits the data — but costs a runtime pass, and for KV cache it implies the cache still lives in fp16 and gets re-quantized into a side buffer at every attention call.
- **Static**: scale frozen ahead of time from calibration data. Slightly worse accuracy (calibration may not cover the deployment distribution), but no runtime quant pass, and — crucially for KV cache — the cache tensor can be int8-native from allocation, roughly doubling the number of blocks (and thus max concurrency) that fit in HBM. This is why granularity must be token-count-independent for the static path: a per-token scale can't be frozen. See [[kv-cache-int8-quantization]].
- Standard methodology: validate with dynamic quant first (cheap, no frozen params to get wrong), then freeze to static for production.

## What Gets Quantized: Weights vs Activations vs KV Cache

- **Weights** are easiest: fixed at deployment, so scales can be chosen offline per-channel/per-group with unlimited calibration effort (GPTQ, AWQ). No runtime distribution shift.
- **Activations** are harder: values depend on the input, and transformer activations have well-documented outlier channels — a few channels with magnitudes 10–100x the rest — which blow up per-tensor scales.
- **KV cache** is an activation store with a twist: it persists across the whole generation and dominates memory at long context, so quantizing it pays twice (capacity + bandwidth). K in particular inherits the activation outlier-channel structure, which is what the outlier-ratio analysis measures.

## Quantized Matmul Math

Whether scales "factor out" of a dot product decides how cheap the integer kernel can be:

- **Per-tensor / per-head scales** are constant along the contracted (head_dim) axis: `q·k = (s_q s_k) Σ_d q_int[d] k_int[d]` — the whole sum runs in pure int8×int8→int32, and one fp multiply at the end applies the scales.
- **Per-channel scales along the contracted dim do not factor out**: `Σ_d s_q[d] q_int[d] s_k[d] k_int[d]` has a different scale on every term of the sum. Naively computing `q_int.float() * k_int.float() * q_scale * k_scale` as if the scales were common factors is silently wrong — the **scale non-associativity** gotcha from [[kv-cache-int8-quantization]].
- **Fix — fold one scale into the other operand**: compute `q' = q * k_scale` in fp first, then quantize `q'`. The per-channel factor is absorbed before quantization, the integer dot product regains a single common scale, and the math is exactly equivalent.
- A full int8×int8→int32 path requires **both** operands quantized. In attention that means q must be dynamically quantized per step even when only the cache is static, and the post-softmax probability matrix `p` needs its own in-kernel quantization before `p @ v` if that matmul is to hit int8 tensor cores too.

## Fake Quantization

Fake quant = quantize→dequantize round trip in fp, injected where the real int8 storage/compute would sit. The tensor experiences exactly the quantization loss a real kernel would produce, but every op stays fp — no custom kernels, runs in plain PyTorch (whose `aten` ops have no int8 arithmetic anyway). This is the de-risking tool: `paged_attention_ref.py` gates a per-channel int8 round trip on the gathered K/V (`kv_quant_mode="int8"`, lines 90-92) inside a PyTorch reproduction that is first verified to match vLLM's real `flash_attn_varlen_func` kernel bit-for-tolerance. Any end-to-end accuracy drop measured this way is attributable purely to quantization loss — validated before a single line of CUDA is written. The same validate-then-implement pattern appears in [[1-overview-precision-convergence]].

## Calibration

Choosing static scales from representative data:

- **Max calibration**: scale = max|x| over the calibration set. Zero clipping error, but one rare outlier permanently stretches the scale and wastes resolution on the 99.9% of values that are small.
- **Percentile clipping**: scale from e.g. the 99.9th percentile of |x|; outliers beyond it saturate at ±127. Trades a little clipping error on rare values for much finer resolution on common ones — usually the better deal when the outlier ratio is high.
- Calibration data should be clean and representative (the KV-cache walkthrough recommends pure-math sets like AIME for low-noise value distributions). Under tensor/expert parallelism, "the" per-channel scale must be dumped consistently across shards.

## Hardware Context: Where the Win Comes From

- **Compute**: int8 paths (dp4a instructions, imma/int8 tensor cores) deliver roughly 2x the fp16 tensor-core throughput on NVIDIA GPUs. But decode-phase attention is memory-bound (see [[5-kernel-dev-arithmetic-intensity]]), so the compute win is secondary there.
- **Bandwidth**: halving bytes-per-value halves HBM traffic for reading the cache — the dominant decode-side win, and it grows with sequence length.
- **Capacity**: an int8-native cache doubles the KV blocks that fit in leftover HBM → ~2x concurrent requests for the same GPU. In PD-separated serving it also halves the prefill→decode network transfer.
- Rule of thumb: memory-bound ops (decode attention) win from bandwidth/capacity; compute-bound ops (prefill GEMMs) win from int8 tensor cores.

## Common Pitfalls

- **Calling `.float()` on int8 codes without applying the scale** — the codes are meaningless integers until multiplied by their scale; forgetting it produces outputs off by ~scale⁻¹.
- **Outlier-dominated per-tensor scale** — one large value forces every other value into a handful of int8 codes. Diagnose with the outlier-ratio method before picking granularity.
- **Quantizing the softmax path carelessly** — softmax must run in fp32 (fp16 can overflow; see `paged_attention_ref.py:100,110` keeping score accumulation and softmax in fp32). If `p` is quantized for the `p @ v` matmul it needs its own dynamic in-kernel scale; never feed int8 codes into exp.
- **Treating per-channel scales as factorable** — the scale non-associativity trap above; results look plausible but are not equivalent to the real kernel's math.
- **Forgetting `clamp_min` on the scale** — an all-zero channel yields scale 0 and NaNs on the divide.
- **Validating quant loss against an unverified baseline** — first prove the fp reproduction matches the real kernel, then inject fake quant; otherwise reproduction bugs and quant loss are confounded.

## Related Concepts

- [[kv-cache-int8-quantization]]
- [[1-overview-kv-cache]]
- [[attention-mechanism]]
- [[1-overview-precision-convergence]]
- [[numerical-instability]]
- [[5-kernel-dev-arithmetic-intensity]]
- [[1-overview-memory-bandwidth-utilization]]

## Sources

- [[kv_cache_quantization_example]]
- Hands-on implementations: `/Users/gzn/playground/qwen_int8_quantization/paged_attention_ref.py` (fake-quant paged attention reference), `/Users/gzn/playground/qwen_int8_quantization/analyze_kv_granularity.py` (outlier-ratio granularity analysis)
