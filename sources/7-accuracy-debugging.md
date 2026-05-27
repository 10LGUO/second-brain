```yaml
title: "Lecture 7 — Accuracy Debugging (精度调试)"
type: source
tags: [precision, debugging, training, inference, gpu, domestic-chips, numerical-instability, quantization]
created: 2026-05-26
updated: 2026-05-26
sources: [7-Accuracy Debugging.pdf]
```

# Lecture 7 — Accuracy Debugging (精度调试)

Source: 上交大 AI infra 团队 (SJTU AI Infra Team), lecture series. Last modified Dec 4, 2025.

---

## 一、Precision Fundamentals (精度相关介绍)

### 1. Numeric Precision Types

**Floating-point formats:**

| Type | Bits | Sign | Exp | Mantissa | Range | Decimal precision | Memory |
|---|---|---|---|---|---|---|---|
| FP64 | 64 | 1 | 11 | 52 | ±1.8×10³⁰⁸ | 15–17 digits | 8 B |
| FP32 | 32 | 1 | 8 | 23 | ±3.4×10³⁸ | 6–9 digits | 4 B |
| BF16 | 16 | 1 | 8 | 7 | ±3.4×10³⁸ | 1–2 digits | 2 B |
| FP16 | 16 | 1 | 5 | 10 | ±6.6×10⁴ | 3–4 digits | 2 B |
| FP8 E4M3 | 8 | 1 | 4 | 3 | ±240 | 1–2 digits | 1 B |
| FP8 E5M2 | 8 | 1 | 5 | 2 | ±57344 | 0–1 digits | 1 B |

**BF16 vs FP16:** BF16 keeps the same exponent range as FP32 (avoids overflow) but sacrifices mantissa precision; FP16 has higher mantissa precision but very limited range — prone to overflow.

**Integer formats:** INT64, INT32, INT16, INT8, UINT8 — exact integers within their range.

### 2. Precision Metrics

**Training:** Compare loss curves and grad norm curves against a baseline. Full alignment with the baseline is the standard.

**Inference:** (1) Run benchmarks (e.g., `evalscope` with HumanEval, AIME24, AIME25, GPQA); (2) manually inspect outputs for gibberish / logical errors.

### 3. Common Precision Scenarios

#### GPU Precision Optimization

- **Training:** AI infra engineers optimize (operator fusion, low-precision quantization, comm-compute overlap, async optimizations). The optimized model must align precision with the original PyTorch FP32 baseline.
- **Inference:** New features (paged attention, quantization, speculative decoding) must not degrade precision — though some applications tolerate precision loss for throughput (e.g., creative writing), others do not (e.g., math/physics problems).

#### Domestic Chip Precision (国产芯片精度)

Domestic chips have far smaller teams (100–1000 people) compared to NVIDIA. They cannot validate every operator across all use cases. Precision issues often arise from hardware and software bugs. Precision is the **reflection of the correctness of the entire hardware-software computing system**.

Goals:
- Training: align loss/grad-norm curves with GPU
- Inference: align outputs with GPU baseline

### 4. Numerical Instability (数值不稳定性)

Floating-point arithmetic is inherently non-associative: `(a + b) + c ≠ a + (b + c)` in general. The root cause is the **big-eats-small** phenomenon — when adding numbers of vastly different magnitude, the smaller number may be rounded away entirely.

**Reduction operations** (RMSNorm, LayerNorm, AllReduce, Reduce-Scatter, matmul) aggregate many small errors. Layer-by-layer propagation amplifies tiny discrepancies exponentially, ultimately changing the sampled token.

**Dynamic batching non-determinism:** Inference servers (e.g., SGLang) group requests into variable-size batches. Different batch sizes change how GPU kernels split reduction tasks across thread blocks, which changes the floating-point addition order, which produces microscopically different results. Setting temperature=0 and fixing seeds is insufficient — the output remains non-deterministic across batches.

**Solution (Thinking Machines Lab + SGLang, Sep 2025):** Implement *batch-invariant* kernels for RMSNorm, Matmul, and attention — kernels that use a fixed partition strategy regardless of batch size. Reference: https://github.com/sgl-project/sglang/issues/10278

Key insight: **don't eliminate floating-point imprecision (impossible in hardware); instead eliminate the conditions that cause non-determinism** — fix the computation order.

---

## 二、GPU Precision Debugging

### 1. Scenario

Arises when an infra optimization (new feature) is added and precision needs to be verified against the pre-optimization baseline.

### 2. Training Precision Debugging

**Step 1 — Establish baseline:**
- Tensor parallelism vs. single-card (mathematically equivalent)
- Pipeline parallelism vs. single-card (ensure identical inputs)
- Data parallelism (N-card multi-batch) vs. single-card (same total data)

**Step 2 — Fix randomness/non-determinism:**
- Fix seeds: `torch.manual_seed(seed)` + `torch.cuda.manual_seed(seed)`
- Load the same checkpoint (or generate weights on CPU with fixed seed, then `.to(device)`)
- Fix inputs

**Step 3 — Numerical comparison:**
Run baseline and target code side-by-side, comparing at operator granularity. Compare:
- Forward pass intermediate activations
- Final logits
- Backward gradients
- Post-optimizer-step parameters and optimizer state

Recommended: 3–5 steps. Too many steps accumulates floating-point drift that may cause false negatives.

**Metrics:** cosine similarity, element-wise relative error, element-wise absolute error.

### 3. Inference Precision Debugging

**Step 1 — Baseline:** Existing working inference run (before the new optimization).

**Step 2 — Fix randomness:**
- Sampling algorithms introduce randomness; fix with seeds.
- Non-determinism from dynamic batching or topology changes (AllReduce order): **set batch size = 1** to eliminate.
- Fix inputs.

**Step 3 — Numerical comparison:**
Compare forward activations, logits, KV cache across many steps. If the final token matches at each step, inference is correct. Multi-step testing also validates KV cache correctness.

---

## 三、Domestic Chip Precision Debugging

### 1. Scenarios

- **Model porting:** Make CUDA-ecosystem models run on domestic chip.
- **Infra optimization:** Chip-specific hacks; validate precision is preserved.

### 2. Training

**Baseline:** Use GPU run as the gold baseline (cross-validate to ensure GPU baseline is itself correct). Alternatively use a single-card run on the same chip.

**Fixing randomness:** Each hardware has its own random number generator (RNG); even different Intel CPU models produce different random sequences. **Best practice: generate random numbers on CPU, then `.to(device)`** — this makes domestic chip and GPU random sequences align.

**Numerical comparison:** Compare every tensor in every TensorList (convert to float32 before comparison). Use cosine similarity ≥ 0.98 as the passing threshold (this threshold should be validated experimentally). Run 3–5 steps.

### 3. Inference

Baseline and randomness-fixing same as GPU. Numerical comparison same as GPU.

---

## 四、Common Precision Problems (常见精度问题)

1. **Operator bugs:** Incorrect computation, or missing synchronization (block should sync but doesn't).

2. **Memory trampling (内存踩踏):** Operator kernel writes out of bounds into HBM, corrupting adjacent tensors. Unit tests often miss this; it only manifests during full model runs.

3. **Accumulation precision / overflow:** Using low-precision accumulators where FP32 is needed (e.g., LayerNorm accumulation in low precision overflows or loses precision).

4. **Distributed communication bugs:**
   - Cascaded-update bugs: distributed comm completes but downstream state not updated.
   - Memory alignment: some RDMA drivers require aligned buffers (NCCL does not, but custom comm layers might).

5. **Low-precision quantization:** Coarser quantization granularity → greater precision loss.

6. **Multi-stream race conditions:** Missing stream synchronization.

7. **CUDA Graph bugs:** Can cause strange, hard-to-reproduce precision errors.

**General rule:** Precision debugging is laborious and requires combining knowledge of each specific technology feature. Always iterate experimentally.

---

## Key Cross-References

- [[1-overview-precision-convergence]] — precision fundamentals
- [[numerical-instability]] — floating-point non-associativity, big-eats-small
- [[1-overview-compute-communication-overlap]] — overlap and stream sync issues
- [[5-kernel-dev-reduce-operator]] — reduction kernel internals
- [[5-kernel-dev-layernorm]] — LayerNorm accumulation precision
