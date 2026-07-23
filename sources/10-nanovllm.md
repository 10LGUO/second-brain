```yaml
title: "Lecture 10 — Building an LLM Inference Engine from Scratch"
type: source
tags: [inference, llm, vllm, nanovllm, kv-cache, model-loading, safetensors, tokenizer, huggingface, continuous-batching, forward-pass, serving]
created: 2026-06-03
updated: 2026-06-03
sources: [10-Nanovllm.pdf]
```

# Lecture 10 — Building an LLM Inference Engine from Scratch

Source: SJTU AI Infra Team, lecture series.

NanoVLLM is a minimal LLM inference engine built from scratch to teach the internals of production engines like vLLM. The goal is to understand each component by implementing it, rather than treating the engine as a black box.

---

## Introduction

Production LLM inference engines (vLLM, SGLang, TGI) are complex systems. NanoVLLM strips them down to the essential pieces:

1. Model loading (weights + architecture)
2. Tokenization
3. Forward pass (prefill + decode)
4. KV cache management
5. Serving loop

Understanding NanoVLLM provides the mental model needed to read and contribute to production engines.

---

## 1. Model Loading

### 1.1 Weight Files and Safetensors

Modern LLMs are distributed as **safetensors** files — a safe, fast alternative to PyTorch `.bin` / pickle format. Safetensors stores tensors with a header describing shapes and dtypes, enabling zero-copy mmap loading.

Typical model directory layout:

```
model/
  config.json            # architecture hyperparameters
  tokenizer.json         # tokenizer vocabulary and rules
  tokenizer_config.json
  model.safetensors      # weights (small models, single file)
  model-00001-of-00004.safetensors  # weights (large models, sharded)
  model.safetensors.index.json      # shard index: param name → file
```

Example model footprint (Qwen 7B):

| | Parameters | File size | Precision | VRAM |
|---|---|---|---|---|
| Qwen2.5-7B | 7.5 B | ~15 GB | BF16 | ~16 GB |

Loading weights with safetensors:

```python
from safetensors import safe_open

weights = {}
with safe_open("model.safetensors", framework="pt", device="cuda") as f:
    for key in f.keys():
        weights[key] = f.get_tensor(key)
```

For sharded models, use the index file to find which shard contains each parameter, then load shards on demand.

### 1.2 Weight Name Mapping

HuggingFace weight names follow a convention (e.g., `model.layers.0.self_attn.q_proj.weight`). A custom engine may use different internal names. A mapping dict translates between them:

```python
HF_NAME_MAP = {
    "model.embed_tokens.weight": "embed_tokens",
    "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.attn.q",
    "model.layers.{i}.mlp.gate_proj.weight": "layers.{i}.mlp.gate",
    # ...
}
```

---

## 2. HuggingFace Model Architecture

### 2.1 Reading config.json

`config.json` defines the model architecture. Key fields for a transformer:

```json
{
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,      // GQA: fewer KV heads than Q heads
  "max_position_embeddings": 32768,
  "rope_theta": 500000.0,
  "vocab_size": 152064,
  "torch_dtype": "bfloat16"
}
```

`num_key_value_heads < num_attention_heads` indicates **Grouped Query Attention (GQA)** — multiple query heads share one KV head, reducing KV cache size.

### 2.2 Model Initialization

A minimal transformer layer in NanoVLLM:

```python
class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = GroupedQueryAttention(config)
        self.mlp  = SwiGLUMLP(config)
        self.input_layernorm     = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

    def forward(self, x, kv_cache, position_ids):
        x = x + self.attn(self.input_layernorm(x), kv_cache, position_ids)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
```

**RMSNorm** (used by Llama, Qwen, Mistral — replacing LayerNorm):

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
```

**SwiGLU MLP** (standard in modern LLMs):

```python
class SwiGLUMLP(nn.Module):
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))
```

**Rotary Position Embedding (RoPE):** Applies a position-dependent rotation to Q and K vectors before the dot product. Encodes relative position without learned embeddings. Supports length extrapolation via scaled RoPE (YaRN, LongRoPE).

```python
def apply_rope(q, k, cos, sin):
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
```

### 2.3 Grouped Query Attention (GQA)

With GQA, `num_key_value_heads` (e.g., 8) < `num_attention_heads` (e.g., 32). Each KV head is shared by `num_attention_heads / num_key_value_heads` = 4 query heads:

```python
# Expand KV heads to match Q heads before attention
k = k.repeat_interleave(self.num_q_per_kv, dim=1)  # [B, 32, S, D]
v = v.repeat_interleave(self.num_q_per_kv, dim=1)
```

GQA reduces the KV cache size by `num_q_per_kv`× — a major memory saving at long context.

---

## 3. Tokenizer

Use HuggingFace tokenizers directly:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Encode
input_ids = tokenizer.encode("Hello, world!", return_tensors="pt").cuda()

# Decode
text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

Key tokenizer concepts:
- **BPE (Byte Pair Encoding):** Merges frequent byte pairs iteratively; used by GPT-family, Qwen
- **Special tokens:** `<|im_start|>`, `<|im_end|>` (Qwen chat format), `<bos>`, `<eos>`
- **Chat template:** `tokenizer.apply_chat_template(messages, tokenize=False)` formats a conversation into the model's expected prompt format

---

## 4. Forward Pass

### 4.1 Prefill vs. Decode

LLM generation has two phases:

| Phase | Input | Output | Compute character |
|---|---|---|---|
| **Prefill** | Full prompt (many tokens) | First output token + KV cache | Compute-bound (large matmul) |
| **Decode** | One new token | One output token (KV cache updated) | Memory-bandwidth-bound (small matmul) |

The decode phase reads the entire KV cache from HBM every step — this is why KV cache size directly limits throughput.

### 4.2 KV Cache Implementation

Pre-allocate a fixed KV cache buffer before generation:

```python
# Shape: [num_layers, 2, batch, num_kv_heads, max_seq_len, head_dim]
kv_cache = torch.zeros(
    num_layers, 2,
    batch_size, num_kv_heads,
    max_seq_len, head_dim,
    dtype=torch.bfloat16, device="cuda"
)
```

During each decode step, append new K and V at position `cur_len`:

```python
def update_kv_cache(layer_idx, k_new, v_new, kv_cache, cur_len):
    kv_cache[layer_idx, 0, :, :, cur_len] = k_new  # K
    kv_cache[layer_idx, 1, :, :, cur_len] = v_new  # V

def get_kv(layer_idx, kv_cache, cur_len):
    k = kv_cache[layer_idx, 0, :, :, :cur_len+1]
    v = kv_cache[layer_idx, 1, :, :, :cur_len+1]
    return k, v
```

### 4.3 Causal Attention Mask

During prefill, apply a causal mask so token `i` only attends to tokens `≤ i`:

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```

During decode, no mask needed — the single new token attends to the full cached context.

### 4.4 Generation Loop

```python
def generate(model, tokenizer, prompt, max_new_tokens=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    seq_len = input_ids.shape[1]

    # Prefill
    logits, kv_cache = model(input_ids, past_len=0)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated = [next_token.item()]

    # Decode loop
    for _ in range(max_new_tokens - 1):
        logits, kv_cache = model(next_token, past_len=seq_len, kv_cache=kv_cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        seq_len += 1
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated)
```

Sampling strategies (instead of `argmax`):
- **Greedy:** `argmax` — deterministic, no creativity
- **Temperature sampling:** `softmax(logits / T)` — higher T = more random
- **Top-p (nucleus) sampling:** Sample from the smallest set of tokens whose cumulative probability ≥ p
- **Top-k sampling:** Sample from the k highest-probability tokens

---

## 5. Launching the Inference Service

A minimal HTTP serving loop wrapping the generate function:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def handle():
    data = request.json
    prompt = data["prompt"]
    output = generate(model, tokenizer, prompt)
    return jsonify({"output": output})

app.run(host="0.0.0.0", port=8000)
```

Production engines add:
- **Continuous batching:** Merge requests mid-generation so the GPU is never idle waiting for a batch to finish. New requests join as slots free up.
- **Paged attention:** Non-contiguous KV cache blocks (like virtual memory) to eliminate fragmentation and support variable-length sequences.
- **Prefix caching:** Cache KV for shared system prompts; skip prefill for repeated prefixes.
- **Tensor parallelism:** Shard model weights across GPUs for models that don't fit on one card.

See [[1-overview-llm-inference-infra]], [[1-overview-kv-cache]].

---

## Summary

NanoVLLM component map:

```
config.json + safetensors
        ↓
  Model init (TransformerLayer × N)
        ↓
  Tokenizer (encode prompt → input_ids)
        ↓
  Prefill (full prompt → logits + KV cache populated)
        ↓
  Decode loop (one token at a time, KV cache grows)
        ↓
  Detokenize → text output
        ↓
  HTTP server (Flask / FastAPI)
```

Key insights from building NanoVLLM:
1. **Prefill is compute-bound, decode is memory-bandwidth-bound** — they have fundamentally different optimization targets
2. **KV cache is the dominant memory consumer** — its size determines max batch size and context length
3. **GQA dramatically reduces KV cache** — 4× fewer KV heads = 4× less cache memory
4. **The generation loop is simple** — complexity in production engines comes from batching, scheduling, and memory management, not the forward pass itself

---

## Key Cross-References

- [[10-nanovllm-model-loading]] — safetensors, weight mapping, GQA architecture
- [[1-overview-kv-cache]] — KV cache concepts and paged attention
- [[1-overview-llm-inference-infra]] — production inference engine components
- [[8-memory-opt-flash-attention]] — efficient attention implementation used in practice
- [[1-overview-precision-convergence]] — BF16 / FP8 for inference
- [[1-overview-slo-metrics-ttft-tpot]] — TTFT / TPOT metrics that inference engines optimize for
