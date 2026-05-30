```yaml
title: FlashAttention
type: concept
tags: [attention, memory, gpu, hbm, io-complexity, training, inference, optimization]
created: 2026-05-30
updated: 2026-05-30
sources: [8-memory-optimization.md]
```

# FlashAttention

FlashAttention is an exact, IO-aware implementation of scaled dot-product attention that dramatically reduces HBM (High Bandwidth Memory) traffic by keeping the attention computation in SRAM (on-chip shared memory).

## The Problem with Standard Attention

For a sequence of length N, standard attention:

1. Computes Q·Kᵀ → writes N×N matrix S to HBM (O(N²) memory)
2. Reads S, computes softmax(S) → writes P to HBM
3. Reads P, computes P·V → writes output O to HBM

HBM I/O cost: O(N²·d) reads + O(N²) writes, where d is head dimension.

For N=4096, d=128: the attention matrix alone requires 4096² × 2 bytes ≈ 32 MB per head per layer — a major bottleneck for long contexts.

## FlashAttention Algorithm

**Key insight:** The softmax denominator (ℓ) and the weighted sum (O) can be computed *incrementally* over tiles of K and V using the numerically stable online softmax identity:

```
softmax([x₁, x₂]) = normalize([exp(x₁-m), exp(x₂-m)])
where m = max(x₁, x₂)
```

When processing tile j after tile i, update the running max and log-sum-exp:

```
m_new = max(m_old, max(S_j))
ℓ_new = exp(m_old - m_new) · ℓ_old + sum(exp(S_j - m_new))
O_new = (exp(m_old - m_new) · O_old · ℓ_old + exp(S_j - m_new) · V_j) / ℓ_new
```

This allows the full attention output to be computed block-by-block, with only a small tile of K, V, and Q in SRAM at any time.

**Algorithm:**

```
for each Q tile q:
    initialize m = -∞, ℓ = 0, O = 0
    for each (K, V) tile (k, v):
        S = q · kᵀ / √d
        m_new = max(m, rowmax(S))
        P_tilde = exp(S - m_new)      # in SRAM
        ℓ_new = exp(m - m_new) · ℓ + rowsum(P_tilde)
        O = (exp(m - m_new) · O · ℓ + P_tilde · v) / ℓ_new
        m, ℓ = m_new, ℓ_new
    write O to HBM
```

The N×N matrix S is never materialized in HBM.

## Complexity Comparison

| Implementation | HBM Memory | HBM I/O | FLOPs |
|---|---|---|---|
| Standard attention | O(N²) | O(N²·d) | O(N²·d) |
| FlashAttention | O(N) | O(N·d²/M) | O(N²·d) |

M = SRAM size per SM. HBM I/O reduction factor ≈ N/d (often 4–16× for typical configs).

FLOPs are identical — FlashAttention is not an approximation.

## Performance Impact

- **Memory:** Eliminates the O(N²) attention matrix allocation. For Llama-3 70B at seq len 8192, this frees tens of GB per batch.
- **Speed:** 2–4× wall-clock speedup for long sequences, as the kernel becomes memory-bandwidth-bound rather than compute-bound.
- **Enables longer contexts:** Context length is no longer limited by the attention matrix size in HBM.

## Versions

| Version | Year | Key additions |
|---|---|---|
| FlashAttention-1 | 2022 | Original tiled algorithm; CUDA kernel for A100 |
| FlashAttention-2 | 2023 | Better parallelism across sequence; reduces non-matmul FLOPs; supports GQA |
| FlashAttention-3 | 2024 | H100 TMA + warp specialization; FP8 support; overlapped compute/data movement |

## Usage in PyTorch

PyTorch 2.0+ includes a fused attention kernel that calls FlashAttention under the hood via `F.scaled_dot_product_attention`:

```python
import torch.nn.functional as F

# Automatically uses FlashAttention when on CUDA + compatible shapes
out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
```

Explicit usage via the `flash-attn` package:

```python
from flash_attn import flash_attn_func

# q, k, v: (batch, seqlen, nheads, headdim), dtype bfloat16 or float16
out = flash_attn_func(q, k, v, causal=True)
```

## Cross-References

- [[8-memory-opt-gradient-checkpointing]] — complementary technique; FlashAttention eliminates remat need for attention
- [[1-overview-gpu-memory-hierarchy]] — SRAM vs. HBM hierarchy that makes this possible
- [[5-kernel-dev-shared-memory-tiling]] — tiling principle underlying FlashAttention
- [[5-kernel-dev-softmax]] — online softmax algorithm
