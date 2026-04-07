```markdown
---
title: HBM (High Bandwidth Memory)
type: concept
tags: [hardware, memory, gpu, ai-infra, hbm, vram]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
---

# HBM (High Bandwidth Memory)

HBM (High Bandwidth Memory) is the high-speed, high-bandwidth off-chip memory standard used in AI compute chips (GPUs and AI accelerators). It is the primary memory backing for model parameters, activations, gradients, and KV cache during large model training and inference. In common usage, **VRAM (显存)** refers to HBM capacity.

## Properties

- Located **outside** the compute chip die (the surrounding modules in a GPU chip diagram)
- High bandwidth compared to alternatives (e.g., GDDR)
- **Training requires HBM** — GDDR is too slow for the bandwidth demands of large model training
- **Inference can use GDDR** for lower-cost deployments, but HBM is preferred for high-throughput scenarios
- PyTorch tensors created on `device` reside in HBM; OOM (Out-of-Memory) errors occur when HBM capacity is exceeded

## Memory Access Mechanics

- **DMA (Direct Memory Access):** Hardware engine that transfers data from HBM to on-chip cache and registers. Controlled by the chip vendor's driver software.
- **Memory access bandwidth (访存带宽):** The actual data transfer rate from HBM into the chip interior. Determined by both HBM-side and chip-side (DMA engine) capability.
- **Memory access utilization (访存利用率)** = actual bandwidth / rated HBM bandwidth. Always < 1.0 due to protocol overhead (analogous to CRC overhead in network communications).
  - Poor chips: low utilization due to bad DMA hardware design or inadequate software.
  - Good chips/drivers: utilization approaches rated bandwidth closely.

## Supply Constraints

- HBM production is capacity-constrained (dominated by a small number of manufacturers).
- Not all chip vendors can procure HBM, which affects the ability to tape out and mass-produce competitive AI chips.
- This is a significant competitive factor for domestic (non-NVIDIA) chip vendors.

## Relationship to AI Infra Optimization

- **Memory optimization** is one of the core axes of AI infra work: use less HBM → fit larger batches → higher throughput → lower cost per token.
- OOM errors (exceeding HBM) are a primary engineering challenge in both training and inference.
- **NVLink** (NVIDIA's chip-to-chip interconnect) enables direct HBM-to-HBM data transfer between chips without CPU involvement, achieving high intra-node communication bandwidth.

## Related Concepts

- [[gpu-software-hardware-architecture]]
- [[large-model-infra]]
- [[operator-development]]
- [[nvlink]]

## Sources

- [[1-overview]]
```

---
