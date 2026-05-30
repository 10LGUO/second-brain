```yaml
title: CPU Offloading for GPU Memory
type: concept
tags: [training, memory, offloading, cpu, deepspeed, zero, optimization, gpu]
created: 2026-05-30
updated: 2026-05-30
sources: [8-memory-optimization.md]
```

# CPU Offloading for GPU Memory

CPU offloading moves tensors that are temporarily idle from GPU HBM to CPU DRAM, freeing GPU memory at the cost of PCIe bandwidth.

## Motivation

Adam optimizer states (first + second moments) consume 8 bytes per parameter — for a 7B parameter model that is 56 GB of optimizer state alone, before parameters and activations. These states are only accessed once per optimizer step, making them good offload candidates.

PCIe 4.0 bandwidth: ~32 GB/s. HBM bandwidth: ~2–3 TB/s. Offloading is viable because optimizer states are accessed infrequently relative to activations.

## ZeRO-Offload (DeepSpeed)

ZeRO-Offload (Ren et al., 2021) offloads optimizer states and gradients to CPU:

**Data flow during a training step:**

```
Forward pass:    GPU (parameters in HBM) → GPU compute
Backward pass:   GPU → gradients accumulate in HBM
Gradient reduce: All-Reduce across GPUs (GPU-GPU)
Offload:         Gradients transferred CPU (PCIe)
Optimizer step:  CPU Adam update (on CPU cores, parallel to next GPU forward)
Prefetch:        Updated parameters transferred back to GPU (PCIe)
```

The CPU optimizer step runs in parallel with the next GPU forward pass, hiding most of the latency.

**Effective memory saving:** Optimizer states (8 B/param) and gradients (2–4 B/param) moved to CPU. For a 7B model: saves ~70–84 GB of HBM.

**Throughput cost:** PCIe round-trip adds latency; effective throughput reduction depends on model size and PCIe bandwidth. For large models where GPU would otherwise OOM, offloading enables training at all.

## ZeRO-3 + Offload

ZeRO-3 shards parameters across GPUs; combined with offload, even parameters can live on CPU and be gathered to GPU only when needed for a forward/backward step. This enables training models whose parameter count far exceeds total GPU HBM across all nodes.

## Activation Offloading

For long-context training where activations dominate:
1. After a layer's forward pass, immediately transfer its activations to CPU DRAM.
2. Before the backward pass reaches that layer, prefetch activations back to GPU.

Requires careful overlap scheduling: prefetch must complete before backprop needs the data. PyTorch 2.x `activation_checkpointing` with custom storage callbacks can implement this.

## When to Use

| Scenario | Recommended approach |
|---|---|
| Model barely fits, optimizer states are the bottleneck | ZeRO-Offload (optimizer state offload) |
| Model doesn't fit even with ZeRO-3 sharding | ZeRO-3 + full CPU offload |
| Long context, activations dominate | Activation offloading + FlashAttention |
| Inference only | Quantization is usually more effective than offloading |

## Cross-References

- [[8-memory-opt-gradient-checkpointing]] — activation memory reduction before resorting to offloading
- [[8-memory-opt-flash-attention]] — eliminates attention matrix memory first
- [[1-overview-gpu-memory-hierarchy]] — PCIe bandwidth to host memory
- [[8-memory-optimization]] — full lecture context
