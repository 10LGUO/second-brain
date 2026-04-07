```yaml
title: LLM Infrastructure Skill Stack
type: concept
tags: [llm-infra, learning-path, skills, career, operator-development, compiler]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# LLM Infrastructure Skill Stack

The LLM infrastructure skill stack describes the layered set of technical competencies required to be an effective LLM infrastructure engineer, from high-level framework usage down to chip-level hardware understanding.

## Learning Philosophy

- Infra is **experience-driven**, not purely talent/intelligence-driven.
- More hands-on practice with hardware → more encountered scenarios → more accumulated knowledge → higher expertise.
- Large-scale distributed systems are vast; every node has extensive knowledge that interacts with other nodes.
- Tech stack updates quickly → follow technical blogs, track hot topics.

> **Overall learning path = Knowledge System + Practice (Optimization)**

## Full-Stack Knowledge Layers (Top to Bottom)

| Layer | Content | Notes |
| --- | --- | --- |
| **Algorithm/Model Layer** | LLM algorithms, Stable Diffusion algorithms | Must understand to perform operator fusion; e.g., KV cache inventors required model understanding |
| **Training/Inference Framework Layer** | Megatron/DeepSpeed (training); vLLM/SGLang (inference) | Must master |
| **PyTorch/TensorFlow/JAX Layer** | Internal mechanisms; graph compilation falls here | Must know internals: `view`/`permute` vs. `clone`, autograd, `compile` |
| **Distributed Communication Library** | NCCL, distributed communication primitives, physical topology | Essential; many optimizations involve compute-communication overlap |
| **Operator Development** | Minimum execution unit on chip; operator fusion, parallel acceleration | Can choose not to specialize but must understand; otherwise optimization is blind |
| **Compiler** | Operator-level compilation; closest to chip hardware | Can choose not to specialize but should understand; full stack transparency |

> "Mastering down to operator development makes you quite strong; mastering the full stack makes you exceptional."

## Key Skills by Dimension

- **Compute:** Understand parallelism, CUDA/Tensor cores, how to keep compute streams busy.
- **Storage:** Memory hierarchy (registers → L1 → L2 → HBM), avoid unnecessary copies (`view`/`permute` not `clone`), memory optimization to reduce VRAM usage.
- **Communication:** Compute-communication overlap, NCCL, NVLink, distributed topology.
- **Precision:** Understand low-precision formats (FP8/FP16/BF16), numerical stability, debugging `nan`/`inf`/loss divergence; systematic precision convergence methodology for domestic chips.
- **Performance:** Systemic optimization requiring global perspective — MFU, bandwidth utilization, operator profiling (Nsight).

## Related Concepts

- [[llm-training-infra]]
- [[llm-inference-infra]]
- [[gpu-software-stack]]
- [[ai-chip-architecture]]
- [[operator-fusion]]
- [[compute-communication-overlap]]

## Sources

- [[1-overview]]

---
