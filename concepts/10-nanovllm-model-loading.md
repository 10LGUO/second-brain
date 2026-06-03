```yaml
title: LLM Model Loading (Safetensors, GQA, RoPE)
type: concept
tags: [inference, model-loading, safetensors, huggingface, gqa, rope, rmsnorm, swiglu, llm, architecture]
created: 2026-06-03
updated: 2026-06-03
sources: [10-nanovllm.md]
```

# LLM Model Loading

Loading an LLM for inference involves three steps: reading weight files, parsing the architecture from config, and initializing the model with those weights.

## Safetensors Format

Safetensors is the standard weight format for modern HuggingFace models, replacing PyTorch pickle (`.bin`). Advantages:
- **Safe:** No arbitrary code execution on load (unlike pickle)
- **Fast:** Memory-mapped — large models load in seconds via mmap, not full copies
- **Zero-copy:** Tensors can be accessed directly from disk without buffering

```python
from safetensors import safe_open

weights = {}
with safe_open("model.safetensors", framework="pt", device="cuda") as f:
    for key in f.keys():
        weights[key] = f.get_tensor(key)
```

For sharded models, `model.safetensors.index.json` maps parameter names to shard filenames. Load only the shards containing needed parameters.

## config.json Architecture Fields

| Field | Meaning |
|---|---|
| `hidden_size` | Embedding / residual stream dimension |
| `intermediate_size` | MLP hidden dimension (typically 2.7× hidden_size) |
| `num_hidden_layers` | Number of transformer blocks |
| `num_attention_heads` | Number of query heads |
| `num_key_value_heads` | Number of KV heads (< query heads → GQA) |
| `max_position_embeddings` | Maximum sequence length |
| `rope_theta` | RoPE base frequency |
| `vocab_size` | Tokenizer vocabulary size |

## Grouped Query Attention (GQA)

When `num_key_value_heads < num_attention_heads`, the model uses GQA. Each KV head is shared by `num_attention_heads / num_key_value_heads` query heads.

**Memory impact:** GQA reduces KV cache size proportionally. With 32 Q heads and 8 KV heads, the KV cache is 4× smaller than Multi-Head Attention (MHA) with equal heads.

**Implementation:** Expand KV heads before computing attention:

```python
# k, v: [batch, num_kv_heads, seq_len, head_dim]
num_q_per_kv = num_attention_heads // num_key_value_heads  # e.g., 4

k = k.repeat_interleave(num_q_per_kv, dim=1)  # → [batch, 32, seq_len, head_dim]
v = v.repeat_interleave(num_q_per_kv, dim=1)
```

MQA (Multi-Query Attention) is the extreme case: `num_key_value_heads = 1`.

## RMSNorm

Modern LLMs (Llama, Qwen, Mistral) replace LayerNorm with RMSNorm — simpler, slightly faster, no mean subtraction:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
```

## SwiGLU MLP

Standard MLP in modern LLMs uses SwiGLU activation (Swish-gated linear unit):

```python
class SwiGLUMLP(nn.Module):
    def forward(self, x):
        # gate_proj and up_proj in parallel, then element-wise gate
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

Requires 3 weight matrices (gate, up, down) vs. 2 in standard MLP — but better empirical performance.

## Rotary Position Embedding (RoPE)

RoPE encodes position by rotating Q and K vectors in 2D subspaces. Each dimension pair `(2i, 2i+1)` is rotated by angle `pos × θ_i` where `θ_i = rope_theta^(-2i/d)`.

Properties:
- **Relative position:** The dot product `Q·Kᵀ` depends only on relative position `m - n`, not absolute positions
- **Extrapolation:** Scaled RoPE variants (YaRN, LongRoPE) extend to longer sequences than trained on

```python
def apply_rope(q, k, cos, sin):
    # cos, sin: precomputed for each position, shape [seq_len, head_dim]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)
```

## Cross-References

- [[10-nanovllm]] — full lecture context
- [[1-overview-kv-cache]] — how GQA reduces KV cache memory
- [[8-memory-opt-flash-attention]] — efficient attention implementation
- [[1-overview-precision-convergence]] — BF16 loading and inference precision
