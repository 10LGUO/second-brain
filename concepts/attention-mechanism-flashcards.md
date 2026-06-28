```yaml
title: Attention Mechanism — Flashcards
type: flashcard
tags: [attention, transformer, flash-attention, paged-attention, embeddings, inference]
created: 2026-06-26
updated: 2026-06-26
source: attention-mechanism
```

# Attention Mechanism — Flashcards

---

**Q: Why transpose Q from `[batch, seq_len, heads, head_dim]` to `[batch, heads, seq_len, head_dim]` before attention?**

So each (batch, head) pair is an independent batch item. Without the transpose, matmul would mix heads together — each head must compute its own attention pattern independently.

---

**Q: What are the roles of Q, K, V?**

- Q, K → determine *where* to attend (matching via dot product)
- V → determines *what* to retrieve (content blended by attention weights)

`output = softmax(QK^T / sqrt(d)) @ V`

---

**Q: Why scale scores by `1/sqrt(head_dim)`?**

The dot product sums `head_dim` terms. With unit-variance inputs, the sum has variance `head_dim`. Without scaling, softmax becomes one-hot → vanishing gradients.

`1/sqrt(head_dim)` normalizes variance back to 1.

Note: uses `Q.shape[-1]` for convenience — same value as `K.shape[-1]`.

---

**Q: What maintains the unit-variance assumption that makes the scale valid?**

1. Xavier/Kaiming weight initialization at training start
2. Layer norm before attention keeps activations normalized throughout training

---

**Q: Why multiple attention heads instead of one?**

One head = one weighted blend of V vectors = one relationship pattern. Multiple heads each project into different subspaces via learned W_Q, W_K, W_V, learning different patterns simultaneously.

Example — "bank" in *"I deposited money at the river bank"*:
- Head A: attends to "money", "deposited" → financial context
- Head B: attends to "river" → geographical context

---

**Q: Why use linear projection (W_Q, W_K, W_V) instead of just splitting the embedding vector per head?**

Splitting: each head sees a fixed slice of dimensions (rigid).  
Projecting: each head learns a blend of *all* `d_model` dimensions (expressive).

The projection matrices are where heads learn *what to look for*.

---

**Q: How many parameters do Q/K/V/O projections add? Show the math.**

`4 * d_model²`

```
Per matrix: num_heads * (d_model * head_dim)
          = num_heads * d_model * (d_model / num_heads)
          = d_model²
```
× 4 for W_Q, W_K, W_V, W_O.

---

**Q: How is a token converted to an embedding vector?**

1. Tokenizer maps token string → integer ID
2. Lookup row `ID` from embedding table `[vocab_size, d_model]`
3. Add positional encoding (same position = same embedding without it)

The table is learned during training via backprop.

---

**Q: What does `nn.Linear(in, out)` compute?**

```python
output = input @ W.T + b   # W: [out, in], b: [out]
```

---

**Q: What is weight tying?**

Sharing `W_embed = W_lm_head.T` — the embedding table and the final LM head use the same weight matrix (transposed). Halves parameter count for those layers.

---

**Q: Trace a token from input to next-token prediction.**

```
token string
  → tokenizer → token ID
  → embedding table lookup → [d_model] vector
  → + positional encoding
  → × num_layers: [attention → W_O → residual+LN → FFN → residual+LN]
  → final layer norm
  → LM head: Linear(d_model, vocab_size) → logits
  → softmax → probabilities → sample next token
```

---

**Q: What is `d_model` and how does it relate to heads?**

`d_model` = embedding/hidden dimension, the width of every token representation throughout the transformer.

`d_model = num_heads * head_dim`

Typical values: 768 (125M), 2048 (1.3B), 4096 (7B), 8192 (70B).

---

**Q: What does `unsqueeze(dim)` do?**

Inserts a new dimension with size = 1 at position `dim`. No data copied — pure metadata change.

```python
x = torch.zeros(4, 6)    # [4, 6]
x.unsqueeze(0)            # [1, 4, 6]
x.unsqueeze(1)            # [4, 1, 6]
```

Used to make tensor rank match before broadcasting.

---

**Q: What is `torch.bmm`?**

Batched matrix multiply for 3D tensors:
```python
torch.bmm(A, B)  # A: [B, n, m], B: [B, m, p] → [B, n, p]
```
Equivalent to `A[i] @ B[i]` for each `i` in parallel. Stricter than `matmul` — requires exactly 3D inputs.

---

**Q: What is the difference between block attention and Flash Attention?**

| | Block attention | Flash Attention |
|---|---|---|
| Q tiling | Yes (blocks of `block_size`) | Yes |
| KV tiling | No (full K each block) | Yes |
| Score tile size | `[block_Q, seq_len]` | `[block_Q, block_K]` |
| HBM for scores | Still O(N²) total | Never written |
| K/V HBM reads | `seq_len/block_size` times | Once |

Flash Attention requires a fused CUDA kernel — raw PyTorch always materializes intermediates in HBM.

---

**Q: Does `torch.compile()` prevent the score matrix from going to HBM?**

No. `torch.compile()` fuses elementwise ops but cannot fuse across matmul boundaries. `S = QK^T` (matmul) and `output = weights @ V` (matmul) remain separate kernels — `S` must be written to HBM between them.

Only a hand-written fused CUDA kernel (Flash Attention) avoids this.

---

**Q: What is the difference between `torch.compile()` and CUDA Graphs?**

- `torch.compile()`: reduces HBM traffic by fusing ops into fewer kernels (compute side)
- CUDA Graphs: eliminates CPU kernel launch overhead by recording and replaying a fixed launch sequence (dispatch side)

CUDA Graphs require fixed input shapes — used for decode (`q_len=1`), not prefill (variable length).

---

**Q: In paged attention, what does one page contain?**

`page_size` tokens worth of KV vectors — shape `[page_size, num_kv_heads, head_dim]` for K (same for V). Not one token, not one sequence — a fixed-size chunk of any sequence's KV history.

---

**Q: What is the gather step in paged attention and why is it required for correctness?**

Before computing attention for sequence `b`:
1. Look up `block_table[b]` → physical page IDs for this sequence
2. Gather those pages from the shared pool → contiguous KV for sequence `b`
3. Compute standard attention over gathered KV

Without gathering, attention over the raw pool mixes tokens from different sequences — a correctness bug, not a performance issue.

---

**Q: Why does paged attention only work for decode (`q_len=1`)?**

During decode, one new token attends over its full cached KV history (many pages). The paged attention kernel is designed for this single-query case.

Prefill processes `q_len = prompt_len` tokens at once and runs standard attention — the full prompt is available contiguously, so paging provides no benefit.
