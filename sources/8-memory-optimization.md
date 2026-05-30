```yaml
title: "Lecture 8 — Memory Optimization (存储优化)"
type: source
tags: [gpu, memory, training, flash-attention, gradient-checkpointing, offloading, quantization, pruning, profiling, optimization]
created: 2026-05-30
updated: 2026-05-30
sources: [8-memory optimization.pdf]
```

# Lecture 8 — Memory Optimization (存储优化)

Source: 上交大 AI infra 团队 (SJTU AI Infra Team), lecture series.

---

## 一、GPU Memory Overview (显存行化)

GPU memory (显存) during LLM training is consumed by four main categories:

| Category | Size (per parameter) | Notes |
|---|---|---|
| Model parameters | 2 B (FP16) or 4 B (FP32) | Weights |
| Gradients | 2 B (FP16) or 4 B (FP32) | Same shape as parameters |
| Optimizer states | 8 B (FP32 Adam) | First moment + second moment, each FP32 |
| Intermediate activations | Variable | Depends on batch size, seq len, depth |

**Rule of thumb for Adam mixed-precision training:** ~16–18 bytes per parameter minimum (parameters + gradients + optimizer states), before activations.

Memory flows between CPU DRAM and GPU HBM (High Bandwidth Memory) over PCIe. Within the GPU, the compute pipeline reads from HBM through L2 → L1/shared memory → registers.

---

## 二、Memory Optimization Methods (显存优化方法)

### 2.1 Formally Tracking Memory Usage (形式化追踪存储消耗)

Use PyTorch built-ins to inspect GPU memory state:

```python
# Summary of current allocations
print(torch.cuda.memory_summary())

# Reserved vs. allocated (in bytes)
torch.cuda.memory_reserved()   # total reserved from OS
torch.cuda.memory_allocated()  # actually in use by tensors

# Peak memory since last reset
torch.cuda.max_memory_allocated()
torch.cuda.reset_peak_memory_stats()
```

PyTorch memory snapshot (introduced in PyTorch 2.x) records a timeline of every allocation and free, allowing visualization of which operators consume the most memory.

### 2.2 FlashAttention

Standard attention computes the full N×N attention score matrix and writes it to HBM, requiring O(N²) memory and O(N²) HBM I/O.

FlashAttention (Dao et al., 2022) rewrites attention to:
1. Tile Q, K, V into blocks that fit in SRAM (on-chip shared memory).
2. Compute the softmax and weighted sum within each tile using the online softmax trick (log-sum-exp accumulation).
3. Never materialize the full N×N matrix in HBM.

Results:
- **Memory:** O(N²) → O(N) HBM footprint
- **HBM I/O:** reduced by a factor of ~N/block\_size relative to standard attention
- **Speed:** 2–4× faster wall-clock time for long sequences despite equivalent FLOPs
- **Exact:** numerically identical to standard attention (not an approximation)

FlashAttention-2 (2023) further improves parallelism across sequence positions and reduces non-matrix-multiply FLOPs. FlashAttention-3 (2024) targets H100 with asynchronous warp specialization and FP8 support.

See [[8-memory-opt-flash-attention]] for full derivation.

### 2.3 Gradient Checkpointing (梯度检查点)

Normal backprop retains all forward-pass intermediate activations in memory until the backward pass completes — O(L) memory for a model of L layers.

Gradient checkpointing trades compute for memory:
- During the forward pass, only *checkpoint* tensors (at strategic boundaries) are saved.
- During backprop, each segment between checkpoints is **re-computed** from the nearest checkpoint before the local backward pass runs.
- Memory: O(√L) with optimal uniform checkpoint placement.
- Compute overhead: ~33% extra FLOPs (one extra forward pass over non-checkpointed segments).

PyTorch API:

```python
from torch.utils.checkpoint import checkpoint

# Wrap a module or function call
output = checkpoint(my_function, *inputs)

# Entire transformer layer:
output = checkpoint(transformer_layer, hidden_states, attention_mask)
```

`checkpoint` by default does not save any tensors from the wrapped function's forward pass; it replays the forward computation during backward.

For activation memory that dominates (large batch, long sequence), checkpointing is often the first intervention before more aggressive techniques.

See [[8-memory-opt-gradient-checkpointing]] for details.

### 2.4 Memory Distribution Across GPUs (显卡之间存储分布)

**ZeRO (Zero Redundancy Optimizer)** — Microsoft DeepSpeed:

| Stage | What is sharded | Memory saving |
|---|---|---|
| ZeRO-1 | Optimizer states | ~4× vs. DDP |
| ZeRO-2 | + Gradients | ~8× |
| ZeRO-3 | + Parameters | ~64× (N GPUs) |

In vanilla DDP (Distributed Data Parallel), every GPU holds a full copy of parameters, gradients, and optimizer states — pure redundancy. ZeRO eliminates this redundancy by assigning each GPU a shard; tensors are gathered on demand during forward/backward and immediately discarded.

**Tensor Parallelism / Pipeline Parallelism** also distribute parameters across GPUs but require model architecture changes. ZeRO is architecture-agnostic.

---

## 三、Profiling Tools (检测优化工具)

### PyTorch Memory Snapshot

```python
import torch.cuda.memory as mem

# Start recording
mem._record_memory_history(max_entries=100000)

# ... run model ...

# Save snapshot
mem._dump_snapshot("memory_snapshot.pickle")
mem._record_memory_history(enabled=None)  # stop
```

Visualize at `pytorch.org/memory_viz` — produces a flame-graph-style allocation timeline showing which callsites allocated the largest tensors and when they were freed.

### Nsight Systems / Compute

`nsys profile` captures GPU kernel timelines and memory transfer events. `ncu` (Nsight Compute) profiles individual kernels including L2 hit rates and HBM bandwidth. See [[8-memory-opt-profiling-tools]].

---

## 四、Memory Optimization Techniques (存储优化工具)

### 4.1 Offloading (卸载)

Move tensors that are not immediately needed from GPU HBM to CPU DRAM (or NVMe) and bring them back when required. The cost is CPU↔GPU PCIe bandwidth (~32–64 GB/s) vs. saved GPU memory.

**DeepSpeed ZeRO-Offload:**
- Offloads optimizer states (and optionally gradients) to CPU during forward/backward pass.
- CPU Adam optimizer update runs on CPU cores while GPU continues with next iteration.
- Enables training larger models on fewer GPUs at the cost of throughput.

**Activation offloading:**
- Offload activation tensors to CPU immediately after they are produced in the forward pass.
- Prefetch them back to GPU just before they are needed in the backward pass.
- Requires careful pipelining to hide PCIe latency.

### 4.2 Quantization (量化)

Reduce model weight precision at load time to save HBM:

| Format | Memory vs. FP16 | Tool |
|---|---|---|
| INT8 | 0.5× | `bitsandbytes`, TensorRT |
| INT4 / NF4 | 0.25× | `bitsandbytes` (QLoRA), GPTQ, AWQ |
| FP8 | 0.5× | H100 native, Transformer Engine |

**bitsandbytes** library:

```python
from transformers import AutoModelForCausalLM
import bitsandbytes as bnb

# 8-bit loading
model = AutoModelForCausalLM.from_pretrained("model", load_in_8bit=True)

# 4-bit loading (NF4 + double quantization)
model = AutoModelForCausalLM.from_pretrained(
    "model",
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

**QLoRA** combines 4-bit quantized base weights + FP16 LoRA adapters, enabling fine-tuning of large models on single GPUs.

Key trade-off: quantization saves memory at the cost of precision degradation. Lower bit-width → more aggressive quantization error. See [[1-overview-precision-convergence]] for convergence implications.

### 4.3 Pruning (剪枝)

Remove redundant weights to reduce model size and, potentially, memory footprint.

**Unstructured pruning:** Zero out individual weights below a magnitude threshold. Sparse weights require sparse compute support (e.g., NVIDIA 2:4 sparsity on Ampere+) to realize actual speedups.

**Structured pruning:** Remove entire attention heads, channels, or layers. Hardware-friendly — results in a smaller dense model. Performance degrades more rapidly per parameter removed.

**Magnitude pruning pipeline:**
1. Train to convergence.
2. Rank weights by magnitude (or gradient signal).
3. Zero/remove the lowest-ranked fraction.
4. Fine-tune to recover accuracy.
5. Repeat (iterative pruning).

Pruning is most effective when the target deployment is inference-only; it is rarely used during training due to the complexity of maintaining sparse gradients.

---

## Key Cross-References

- [[8-memory-opt-flash-attention]] — FlashAttention algorithm and I/O complexity
- [[8-memory-opt-gradient-checkpointing]] — gradient checkpointing memory/compute trade-off
- [[8-memory-opt-cpu-offloading]] — CPU offloading with ZeRO-Offload
- [[1-overview-gpu-memory-hierarchy]] — GPU memory levels (registers → shared → L2 → HBM)
- [[1-overview-hbm-high-bandwidth-memory]] — HBM bandwidth and capacity specs
- [[1-overview-precision-convergence]] — FP16/BF16/FP8 training precision
- [[5-kernel-dev-reduce-operator]] — reduction kernel internals relevant to attention
