```yaml
title: Gradient Checkpointing
type: concept
tags: [training, memory, optimization, backpropagation, activations, gpu]
created: 2026-05-30
updated: 2026-05-30
sources: [8-memory-optimization.md]
```

# Gradient Checkpointing

Gradient checkpointing (also called *activation recomputation* or *rematerialization*) trades extra forward-pass computation for reduced activation memory during training.

## The Problem

Standard backpropagation retains all intermediate activations produced in the forward pass so they can be used during the backward pass (to compute gradients). For a model with L layers, this requires O(L) activation memory — for large models at large batch sizes this dominates GPU memory usage.

For a transformer layer: activations per layer ≈ `2 × batch_size × seq_len × hidden_dim` bytes. At batch=32, seq=2048, hidden=4096, FP16: ~1 GB per layer. A 96-layer model would require ~96 GB of activation memory alone.

## How It Works

With gradient checkpointing:

1. **Forward pass:** Only selected tensors ("checkpoints") are retained in HBM. All other intermediate activations are discarded immediately after use.
2. **Backward pass:** When backprop reaches a segment whose activations were discarded, the segment is **re-run** (forward) from its checkpoint to regenerate the needed activations, then the backward pass proceeds.

**Memory vs. compute trade-off:**

| Strategy | Memory | Extra FLOPs |
|---|---|---|
| No checkpointing | O(L) | 0 |
| Checkpoint every layer | O(1) | ~100% (2× forward) |
| Checkpoint every √L layers | O(√L) | ~33% (one extra partial forward) |

The √L strategy is optimal: with uniform checkpoints every √L layers, at most √L segments need recomputation, each of length √L.

## PyTorch API

```python
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

# Wrap a single function/module forward
output = checkpoint(my_module, input_tensor)

# Wrap a sequential of modules
output = checkpoint_sequential(layers, segments=4, input=x)
```

`checkpoint` by default does **not** save any activation tensors within the wrapped function. During backward, it replays the forward computation.

**Important:** Random operations (dropout) inside a checkpointed region require special handling — the RNG state must be saved and restored during recomputation:

```python
# PyTorch handles this automatically with use_reentrant=False (recommended)
output = checkpoint(my_module, input_tensor, use_reentrant=False)
```

## Practical Usage Patterns

**Per-layer checkpointing (most common):**
```python
class CheckpointedTransformer(nn.Module):
    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

**Selective checkpointing:** Checkpoint only expensive layers (attention) while retaining cheap ones (MLP) — tunable based on memory budget vs. throughput goals.

**Integration with FlashAttention:** FlashAttention already avoids storing the O(N²) attention matrix, so checkpointing the attention block saves only the remaining activations (QKV projections, output projection). The two techniques are complementary.

## Memory Savings in Practice

For a GPT-3-scale model (175B params) with standard mixed-precision training:
- Without checkpointing: activation memory can exceed 100 GB for moderate batch sizes
- With per-layer checkpointing: activation memory drops to O(1) layers worth of activations

Throughput cost is typically 20–35% in practice (depends on model, hardware, and how expensive the forward pass is relative to memory bandwidth).

## Cross-References

- [[8-memory-opt-flash-attention]] — eliminates the O(N²) attention matrix; reduces checkpointing need for attention
- [[8-memory-optimization]] — full lecture context including offloading and quantization
- [[1-overview-gpu-memory-hierarchy]] — HBM capacity constraints that motivate checkpointing
