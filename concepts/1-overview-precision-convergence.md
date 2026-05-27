```yaml
title: Precision Convergence (精度收敛)
type: concept
tags: [precision, domestic-chips, ai-infra, debugging, training, inference]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# Precision Convergence (精度收敛)

Precision convergence refers to the challenge of ensuring that a model running on a given hardware/software stack produces numerical outputs that match expected results — typically validated against a reference implementation (usually NVIDIA GPU). It encompasses both algorithmic correctness and hardware/chip system correctness. Precision is a foundational requirement: **"Only when precision is correct can you discuss performance."**

## Why It Matters

- **Prerequisite for everything else:** No performance optimization is meaningful if the model produces incorrect results.
- **Business-critical for domestic chip vendors:** A single unlocated precision error can cost billions in contracts. If the source of the error (software vs. hardware) cannot be identified, next-generation chip hardware may inherit the defect — propagating the problem indefinitely.
- **Systematic challenge:** Requires a dedicated methodology; cannot be solved by ad hoc debugging alone.

## Sources of Precision Issues

### Operator-Level

- **Incorrect computation:** wrong result from a kernel bug.
- **Missing synchronization:** a block should sync before writing results but doesn't; produces non-deterministic or wrong outputs.
- **Memory trampling (内存踩踏):** kernel writes out of bounds in HBM, corrupting adjacent tensors. Unit tests often miss this; manifests only in full model runs.
- **Accumulation precision / overflow:** low-precision accumulator where FP32 is required (classic: LayerNorm sum in FP16 overflows or loses mantissa bits).

### Distributed / Communication

- Cascaded-update bugs: distributed comm completes but dependent state not updated.
- Memory alignment: some RDMA drivers require aligned buffers; misalignment silently corrupts data.

### Numerical Instability

Floating-point is non-associative; large-scale reduction operations (LayerNorm, AllReduce, Matmul) amplify tiny rounding differences exponentially. See [[numerical-instability]].

### Low-Precision Quantization

Coarser quantization granularity → greater precision loss.

### Multi-Stream / CUDA Graph

Missing stream sync, or CUDA Graph capturing incorrect state, causes hard-to-reproduce errors.

---

## Debugging Methodology

### Establish a Baseline

- Training: single-card loss curve as baseline for parallelism variants (tensor parallel, pipeline parallel, data parallel are all mathematically equivalent to single-card).
- Inference: existing working run before optimization.
- Domestic chip: use GPU run as gold baseline (cross-validate GPU baseline itself).

### Fix Randomness / Non-Determinism

- `torch.manual_seed(seed)` + `torch.cuda.manual_seed(seed)`
- Load same checkpoint; or generate weights on CPU with fixed seed then `.to(device)`
- **Generate all random tensors on CPU then `.to(device)`** — each hardware has its own RNG; this ensures domestic chip and GPU sequences align.
- Disable dropout and other stochastic ops when diagnosing.
- Set batch size = 1 to eliminate dynamic-batching non-determinism.

### Numerical Comparison

| Scope | What to compare |
|---|---|
| Training | Forward activations, logits, backward gradients (via hooks), post-optimizer params & optimizer state |
| Inference | Forward activations, logits, KV cache |

Run 3–5 steps (more steps risk false negatives from accumulated drift). Metrics: cosine similarity (domestic chip threshold ≥ 0.98), element-wise relative/absolute error. Compare at operator granularity — if all operator inputs/outputs match, final result will match.

---

## Related

- [[numerical-instability]] — floating-point non-associativity and batch-invariant kernels
- [[7-accuracy-debugging]] — detailed lecture notes on precision debugging
- [[5-kernel-dev-reduce-operator]] — reduction kernel internals
- [[5-kernel-dev-layernorm]] — LayerNorm accumulation
