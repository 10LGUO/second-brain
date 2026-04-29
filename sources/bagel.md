```yaml
title: BAGEL — Bytedance Multimodal Foundation Model
type: source
tags: [multimodal, llm, diffusion, inference, optimization, flash-attention, mixture-of-experts, training]
created: 2026-04-29
updated: 2026-04-29
sources: [https://github.com/bytedance-seed/BAGEL]
```

# BAGEL

**BAGEL** (Bytedance-seed) is an open-source unified multimodal foundation model supporting visual understanding, text-to-image generation, and image editing in a single architecture.

- **Parameters:** 14B total, 7B active (Mixture of Tokens)
- **License:** Apache 2.0
- **HuggingFace:** `ByteDance-Seed/BAGEL-7B-MoT`
- **Repo:** https://github.com/bytedance-seed/BAGEL

---

## Architecture

**Decoder-only unified architecture** — both understanding and generation share the same transformer backbone.

```text
Input (text + image tokens, interleaved)
  └── Dual image encoders
        ├── Pixel-level encoder   (low-level visual features)
        └── Semantic encoder      (high-level understanding)
  └── Mixture-of-Tokens (MoT) Transformer
        ├── 14B total parameters
        └── 7B active per forward pass (sparse activation)
  └── Output
        ├── Understanding: text tokens
        └── Generation: diffusion decoder (denoising, ~50 steps)
```

**Mixture of Tokens (MoT):** Similar in spirit to Mixture of Experts but operating at the token level — enables 7B active parameters from a 14B total model. Different tokens route to different parameter subsets.

**Training objective:** "Next Group of Token Prediction" — predicts groups of tokens rather than one at a time, enabling both autoregressive text and diffusion-style image generation within one model.

**Training:** Three phases:
1. Pre-training on trillions of interleaved multimodal tokens (text, image, video, web)
2. Continued training to expand capacity
3. Supervised fine-tuning for task-specific alignment

---

## Hardware Requirements

| Mode | VRAM | Notes |
| --- | --- | --- |
| Full precision | 32GB+ | Single A100 80GB or 2×A100 40GB |
| INT8 quantization | 22–32GB | Not recommended (quality loss) |
| NF4 quantization | 12–32GB | Recommended for consumer GPUs |
| DF11 compressed | <32GB | Bytedance-specific quantization |

**Critical dependency:** `flash_attn==2.5.8` — required for efficient attention. Must be compiled for your CUDA version.

---

## Inference Pipeline

For image generation (text-to-image / editing):

```text
1. Encode text prompt → token embeddings
2. Encode input image (if editing) → pixel + semantic features
3. MoT transformer forward pass (understanding + conditioning)
4. Diffusion decoder: ~50 denoising timesteps
5. Decode latents → output image
```

Key inference parameters:

| Parameter | Range | Effect |
| --- | --- | --- |
| `cfg_text_scale` | 4.0–8.0 | Text prompt adherence |
| `cfg_image_scale` | 1.0–2.0 | Input image detail preservation |
| `cfg_interval` | [0.4, 1.0] | Window where CFG is applied |
| `num_timesteps` | ~50 | Denoising steps (quality vs speed) |
| `cfg_renorm_type` | global/channel/text_channel | Guidance normalization strategy |

**CFG-Renorm:** Custom guidance normalization that improves image editing quality by renormalizing the classifier-free guidance signal.

---

## Performance

| Benchmark | BAGEL | Comparison |
| --- | --- | --- |
| MME | 2388 | Outperforms Qwen2.5-VL, InternVL-2.5 |
| MMBench | 85.0% | State of the art open-source |
| MM-Vet | 67.2% | Competitive with proprietary models |
| GenEval | 0.82 | Competitive with FLUX-1-dev, SD3 |
| Image editing | — | Superior to open-source alternatives |

---

## Likely Optimization Targets

For GPU performance optimization work:

**1. Flash Attention** — already required, but the attention mechanism is the dominant cost in the transformer backbone. Profile attention kernel time vs total step time.

**2. Diffusion sampling loop** — 50 timesteps × full forward pass each. High arithmetic intensity. Potential targets:
- Reduce timesteps (distillation)
- Batch multiple denoising steps
- CUDA graph capture of the static diffusion loop

**3. MoT routing** — token routing adds overhead (scatter/gather operations). Profile the routing cost vs compute savings.

**4. Dual encoder** — two image encoders run per forward pass. Check if they can be parallelized or fused.

**5. Cross-modal attention** — attention between text and image tokens. Profile with `ncu` for memory vs compute bound.

---

## Profiling Approach

```bash
# 1. System-level profile (nsys)
nsys profile -o bagel_profile python inference.py

# 2. Kernel-level on the slowest operator (ncu)
ncu --set full -k <kernel_name> -o kernel_profile python inference.py

# 3. torch.profiler for Python-level view
```

Start with `nsys` to find the biggest time sinks (attention? diffusion loop? data loading?), then drill into specific kernels with `ncu`.

---

## Practical Notes

- Full inference requires 32GB+ VRAM — need A100 80GB or 2×40GB
- For optimization practice, start with a single transformer block or single diffusion step on a proxy model before running full BAGEL
- NF4 quantization reduces VRAM to ~12GB but changes kernel behavior (dequantization overhead becomes significant)

---

## Related Concepts

- [[gpu-perf-codelab]]
- [[cuda-gemm]]
- [[warp]]
- [[5-kernel-dev-arithmetic-intensity]]
- [[cuda-cheatsheet]]

## Sources

- https://github.com/bytedance-seed/BAGEL
- https://arxiv.org/abs/2505.14683
