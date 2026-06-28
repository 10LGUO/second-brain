```yaml
title: Attention Mechanism
type: concept
tags: [attention, transformer, kv-cache, multi-head, flash-attention, paged-attention, inference, embeddings]
created: 2026-06-26
updated: 2026-06-26
sources: [sample_code/attention_implementation.py]
```

# Attention Mechanism

## Input Layout and Transpose

Q, K, V arrive as `[batch_size, seq_len, num_heads, head_dim]` — tokens first, heads second. Before computing attention, transpose to `[batch_size, num_heads, seq_len, head_dim]` so that each (batch, head) pair is an independent item:

```python
Q = Q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
```

Without this, matmul would mix heads together. Each head must compute its own independent attention pattern.

## Core Computation

```
scores = Q @ K^T / sqrt(head_dim)    # [batch, heads, seq_len, seq_len]
weights = softmax(scores, dim=-1)
output = weights @ V                  # [batch, heads, seq_len, head_dim]
```

Three distinct roles:
- **Q, K**: determine *where* to attend (matching)
- **V**: determines *what* to retrieve (content)
- **scores**: not the output — they're weights for blending V vectors

## Why Scale by 1/sqrt(head_dim)

The dot product sums `head_dim` terms. If Q and K have unit variance per dimension, the sum has variance `head_dim`. Without scaling, large magnitudes push softmax toward a one-hot distribution → vanishing gradients.

```python
scale = 1 / math.sqrt(head_dim)  # head_dim = Q.shape[-1]
```

This is a property of the dot product dimensionality, not Q specifically — `K.shape[-1]` or `V.shape[-1]` would give the same value.

The unit-variance assumption holds approximately because:
1. Weights are initialized with unit variance (Xavier/Kaiming)
2. Layer norm before attention keeps activations normalized

## Why Multiple Heads

A single head produces one weighted blend of all V vectors — it can learn one relationship pattern. Multiple heads each project into a different subspace via learned `W_Q`, `W_K`, `W_V`, learning different patterns simultaneously.

Example — token "bank" in *"I deposited money at the river bank"*:
- Head A attends to "money", "deposited" → financial context
- Head B attends to "river" → geographical context
- Head C attends to adjacent tokens → syntactic structure

The concatenated output carries all three contexts at once.

`d_model = num_heads * head_dim` — total model width is preserved, each head works on a `head_dim`-sized slice of the representation.

## Linear Projections (W_Q, W_K, W_V)

Rather than splitting the embedding vector into chunks per head, each head projects via a learned matrix:

```
W_Q: [d_model, head_dim]  per head
```

**Splitting** gives each head a fixed slice of dimensions — head 0 always sees dims 0–127.  
**Projecting** lets each head learn a blend of all `d_model` dimensions — far more expressive.

Total projection parameters: `4 * d_model²` (W_Q + W_K + W_V + W_O output projection).

```
num_heads * (d_model * head_dim) = num_heads * d_model * (d_model/num_heads) = d_model²
```

## From Embeddings to Prediction

**Token → embedding**: lookup in a `[vocab_size, d_model]` table indexed by token ID. Positional encoding is added to inject position information.

**Through transformer blocks** (repeated `num_layers` times):
1. Attention → output projection W_O → residual add → layer norm
2. FFN (expand + contract) → residual add → layer norm

**Final prediction**:
- Final layer norm
- LM head: `nn.Linear(d_model, vocab_size)` → logits `[batch, seq_len, vocab_size]`
- Softmax → probabilities → sample next token

**Weight tying**: many models share `W_embed = W_lm_head.T`, halving parameter count.

## nn.Linear

Stores `W: [out, in]` and `b: [out]`. Forward pass:

```python
output = input @ W.T + b
```

## Block Attention vs Flash Attention

**Block attention** (Q-tiled only): processes Q in blocks of `block_size`, but computes full `q_block @ K^T` each iteration — K is never tiled. Score slice is `[batch*heads, block_size, seq_len]`, still O(N²) in memory across iterations. K and V are reloaded from HBM `seq_len/block_size` times.

**Flash Attention**: tiles *both* Q and KV blocks. Score tile is `[block_Q, block_K]` — O(block²) regardless of seq_len. Online softmax (running m, l statistics) allows correct softmax without materializing full scores. K and V each loaded from HBM exactly once.

In raw PyTorch, intermediate tensors always round-trip through HBM. The "no score matrix in HBM" property requires a fused CUDA kernel (`F.scaled_dot_product_attention` with Flash backend, or hand-written CUDA).

## `torch.compile` vs CUDA Graphs

| | `torch.compile()` | CUDA Graphs |
|---|---|---|
| Eliminates | Redundant HBM round-trips (op fusion) | CPU kernel launch overhead |
| How | Fuses elementwise ops into adjacent kernels | Records launch sequence; replays with one CPU call |
| Matmul boundary | Cannot fuse across matmul boundaries | N/A |
| Shape constraint | Handles varying shapes | Requires fixed shapes |
| Use case | Compute efficiency | Decode step (always `q_len=1`) |

Score matrix `S = QK^T` still materializes in HBM under `torch.compile` — two matmuls cannot be fused.

## Paged Attention

KV cache stored as a shared page pool: `[num_pages, page_size, num_kv_heads, head_dim]`. A page holds `page_size` tokens (not one token). Each sequence has a `block_table` mapping logical pages to physical pages.

**Gather step** (what makes it paged): before computing attention for sequence `b`, use `block_table[b]` to gather only that sequence's pages. Without this, attention would mix tokens from different sequences.

**Decode only** (`q_len=1`): one new token attends over all cached pages. Prefill runs standard attention since `q_len = prompt_len`.

## Cross-References

- [[8-memory-opt-flash-attention]] — Flash Attention algorithm detail
- [[10-nanovllm-scheduler-paged-attention]] — Block manager, prefix cache, scheduler
- [[9-perf-opt-torch-compile-cuda-graph]] — CUDA Graph for decode
- [[1-overview-gpu-memory-hierarchy]] — HBM vs SRAM
