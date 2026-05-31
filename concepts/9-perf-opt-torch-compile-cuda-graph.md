```yaml
title: torch.compile and CUDA Graph
type: concept
tags: [torch-compile, cuda-graph, performance, gpu, optimization, static-graph, kernel-fusion, training, inference]
created: 2026-05-31
updated: 2026-05-31
sources: [9-profile-optimization.md]
```

# torch.compile and CUDA Graph

Both `torch.compile` and CUDA Graph target the same problem — reducing CPU overhead in the training/inference loop — via different mechanisms. They are complementary and often used together.

## torch.compile (PyTorch 2.0+)

`torch.compile` captures the Python computation graph using **TorchDynamo** (a Python bytecode interceptor), then lowers it to an optimized backend.

### How it works

```
Python model code
      ↓ TorchDynamo (bytecode capture, graph break on unsupported ops)
FX Graph (symbolic computation graph)
      ↓ AOTAutograd (ahead-of-time joint forward+backward graph)
      ↓ Inductor backend (default)
Triton kernels (GPU) or C++ kernels (CPU)
```

**Inductor optimizations applied:**
- Operator fusion (elementwise chains, pointwise + reduction)
- Memory layout selection (NHWC vs. NCHW, stride permutations)
- Constant folding
- Loop tiling and vectorization for the generated Triton code

### Usage

```python
import torch

model = MyModel().cuda()
model = torch.compile(model)           # default: mode="default"
model = torch.compile(model, mode="reduce-overhead")   # prioritize launch overhead reduction
model = torch.compile(model, mode="max-autotune")      # auto-tune tile sizes (slower compile)
model = torch.compile(model, dynamic=True)             # support dynamic shapes
```

**Compilation happens on the first call** with a given input shape. Subsequent calls with the same shape hit the compiled path.

### Graph breaks

When TorchDynamo encounters unsupported Python constructs (data-dependent control flow, certain Python builtins), it inserts a "graph break" — falling back to eager Python for that segment. Graph breaks defeat fusion and compilation. Diagnose with:

```python
torch._dynamo.explain(model)(inputs)   # shows graph break locations
```

### When torch.compile helps most

- Models with many small operators (elementwise chains, LayerNorm, activations)
- Models with predictable shapes
- Inference loops with repeated identical calls

### When torch.compile helps less

- Large compute-bound kernels (GEMM) — already at hardware peak; fusion can't help much
- Models with heavy dynamic control flow — many graph breaks
- First-call latency is unacceptable (compilation takes seconds to minutes)

---

## CUDA Graph

CUDA Graph records a sequence of CUDA kernel launches into an immutable graph object. Replaying the graph issues all the launches with a single API call, eliminating per-step driver overhead entirely.

### Mechanism

```
Normal execution:
  CPU: [launch K1] [launch K2] [launch K3] ...   ← 5-20µs overhead per launch
  GPU:          [K1][K2][K3]...

CUDA Graph execution:
  CPU: [graph.replay()]                           ← single call replays all
  GPU: [K1][K2][K3]...
```

### Requirements

CUDA Graph requires:
1. **Static shapes** — tensor sizes must not change between replay calls
2. **Static memory addresses** — all tensors must be pre-allocated; no dynamic allocation inside the graph
3. **No CPU-GPU sync inside the graph** — no `.item()`, no conditional branches on GPU values

### Usage

```python
model = model.cuda()
static_input = torch.zeros(batch_size, seq_len, hidden).cuda()
static_output = torch.zeros(batch_size, seq_len, vocab_size).cuda()

# Warmup runs (allow caching allocator to settle)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_output = model(static_input)
torch.cuda.current_stream().wait_stream(s)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay (copy new data into static_input before each replay)
def run(new_input):
    static_input.copy_(new_input)
    g.replay()
    return static_output.clone()
```

### torch.compile + CUDA Graph together

`mode="reduce-overhead"` in `torch.compile` automatically applies CUDA Graph capture after compilation:

```python
model = torch.compile(model, mode="reduce-overhead")
```

This is the recommended path — `torch.compile` handles fusion and lowering; CUDA Graph handles launch overhead.

### Speedup profile

| Model type | torch.compile gain | + CUDA Graph gain |
|---|---|---|
| Many small ops (e.g., transformer w/ small batch) | 1.5–3× | Additional 1.5–3× |
| Large compute-bound (big GEMM) | 1.05–1.2× | Additional 1.1–1.3× |
| Inference with static shapes | High | High |
| Training with dynamic batch | Moderate | Low (shapes change) |

### Limitations

- **CUDA Graph:** Does not support dynamic shapes, in-graph allocations, or Python callbacks. Incompatible with certain distributed patterns that require CPU involvement mid-step.
- **torch.compile:** Compilation time (first call) can be 30 seconds to several minutes for large models. Use `torch._dynamo.config.cache_size_limit` and `torch.compiler.reset()` to manage the cache.

---

## Cross-References

- [[9-perf-opt-host-device]] — understanding why launch overhead matters
- [[9-perf-opt-profiling]] — identifying launch overhead in nsys traces
- [[1-overview-operator-fusion]] — fusion techniques that torch.compile applies
- [[2-pytorch-computational-graph]] — PyTorch's eager vs. compiled execution models
