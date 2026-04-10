```yaml
title: MFU — Model FLOPS Utilization
type: concept
tags: [performance, compute, gpu, training, inference, metrics]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# MFU — Model FLOPS Utilization

**MFU (Model FLOPS Utilization)** is a metric that reflects how efficiently a model's training or inference workload utilizes the theoretical peak compute throughput (FLOPS) of the hardware. It is one of the two primary performance metrics in AI infra, alongside [[memory-bandwidth-utilization]].

## Key Properties

- **Definition**: MFU = (Achieved FLOPS) / (Peak Hardware FLOPS). A value of 1.0 (100%) means every floating-point operation the hardware is capable of in a given time window is being used productively by the model computation.
- **Range in practice**: Real workloads rarely exceed 50–60% MFU. Well-optimized large-model training runs (e.g., PaLM, Megatron-LM) have reported 38–57% MFU on TPUs and high-end GPU clusters. Values below ~20% are generally considered poor and warrant investigation.
- **Compute-bound vs. memory-bound**: MFU is most meaningful when a workload is compute-bound — i.e., the bottleneck is arithmetic throughput rather than data movement. When a workload is memory-bandwidth-bound, [[memory-bandwidth-utilization]] (MBU) is the more diagnostic metric. Large batch sizes and large matrix dimensions push workloads toward being compute-bound and thus toward higher attainable MFU.
- **Hardware peak FLOPS depends on dtype**: Peak FLOPS differs substantially by precision. An H100 SXM5 delivers ~67 TFLOPS (FP64), ~134 TFLOPS (FP32), ~989 TFLOPS (BF16/FP16 with sparsity off), and ~1979 TFLOPS with structured sparsity. MFU calculations must specify which peak is used as the denominator; the convention is typically to use the dense (non-sparsity) tensor-core peak for the training dtype (usually BF16 or FP16).
- **Model FLOPS vs. hardware FLOPS**: The numerator is the number of floating-point operations the model actually requires per second, computed from the model architecture (e.g., 6ND for a transformer with N parameters and D tokens per second for forward+backward). This is distinct from the hardware's raw counter of FP ops, which may include overhead operations.

## Formula

```text
MFU = (Model FLOPS per second) / (Hardware peak FLOPS)

Model FLOPS per second = (FLOPS per token) × (tokens per second)
```

For a transformer, a common approximation for training FLOPS per token (forward + backward) is:

```text
FLOPS per token ≈ 6 × N
```

where N is the number of model parameters. This approximation holds for large models where the embedding and attention FLOPS are dominated by the dense feed-forward matrix multiplications. A more precise estimate also adds the attention FLOPS:

```text
FLOPS per token ≈ 6N + 12 × n_layers × n_heads × d_head × seq_len
```

The factor of 6 in the leading term arises from: 2 FLOPS per multiply-accumulate (MAC), multiplied by the forward pass (factor of 2 for multiply + add) and backward pass (factor of ~2× the forward cost for gradient computation with respect to weights and inputs).

## Why MFU Matters

- **Infrastructure efficiency signal**: MFU tells operators whether they are getting good value from expensive GPU/TPU compute. A cluster running at 15% MFU is effectively wasting ~85% of its hardware investment on overhead, idle time, communication, or poor kernel utilization.
- **Scaling cost estimation**: When projecting the cost to train a model of a given size on a given number of tokens, the estimated wall-clock time depends directly on the achievable MFU. Underestimating MFU leads to optimistic compute budgets; overestimating leads to wasted reservation spend.
- **Comparison across hardware generations**: MFU normalizes throughput against hardware peak, allowing meaningful comparison of training efficiency across GPU generations (A100 vs. H100) or hardware types (GPU vs. TPU), independent of raw clock speed differences.
- **Optimization target**: Teams working on kernels, communication libraries, and distributed training strategies use MFU as their primary objective function. Improvements in [[flash-attention]], [[tensor-parallelism]], and overlap of compute with [[all-reduce]] communication all show up as MFU gains.
- **Accountability for system software**: Because MFU measures the gap between theoretical and realized throughput, it naturally assigns accountability to the software stack — compiler, runtime, communication library, and kernel implementation — rather than the hardware vendor.

## Common Causes of Low MFU

| Cause | Description |
| --- | --- |
| Small batch size / short sequences | Matrix multiplications become too small to saturate tensor cores; arithmetic intensity drops below the compute-bound threshold. |
| Memory bandwidth bottleneck | Attention with long sequences, embedding lookups, or frequent small operations cause the GPU to stall waiting for data. See [[memory-bandwidth-utilization]]. |
| Communication overhead | In multi-GPU/multi-node training, all-reduce, all-gather, or pipeline bubble time subtracts from productive compute time. |
| Kernel launch overhead | Many small sequential ops (e.g., non-fused activation functions, layer norms) create CPU-side scheduling latency and GPU idle gaps. |
| Load imbalance | In [[pipeline-parallelism]], uneven stage compute times cause "bubbles" where stages sit idle. |
| Checkpointing / recomputation cost | Activation checkpointing re-runs forward passes, adding FLOPS that do not count toward model progress and diluting effective MFU. |
| Mixed-precision inefficiency | Unintentional FP32 ops (e.g., certain loss computations, optimizer states) reduce the fraction of time spent on fast tensor-core paths. |
| CPU bottlenecks | Data loading, tokenization, or host-side preprocessing that cannot keep pace with GPU throughput leads to GPU starvation. |
| Suboptimal memory layout | Non-contiguous tensors or poor stride patterns force the GPU to perform gather operations instead of streaming sequential memory reads, increasing effective latency. |
| Framework overhead | Dynamic graph frameworks with high per-step Python overhead (e.g., un-compiled eager-mode PyTorch on small models) introduce scheduling latency that limits throughput. |

## Relationship to MBU

MFU and [[memory-bandwidth-utilization]] (MBU) are complementary diagnostics:

- A workload cannot simultaneously be at 100% MFU and 100% MBU — one resource is always the bottleneck.
- **Compute-bound workloads** (large matmuls, large batch): MFU is the binding constraint. Improving memory bandwidth will not help unless the workload crosses the roofline into memory-bound territory.
- **Memory-bound workloads** (small batch inference, attention with very long seq, embedding lookup): MBU is the binding constraint. Optimizing kernels for compute efficiency has diminishing returns until bandwidth pressure is relieved.
- The **roofline model** is the formal framework for determining which regime a given operation sits in, based on its arithmetic intensity (FLOPS / bytes accessed).
- In practice, a single training step contains a mix of compute-bound operations (large dense matrix multiplications in feed-forward layers) and memory-bound operations (layer normalization, softmax, elementwise activations). The overall MFU is a weighted average of the utilization of each operation, which means a few slow memory-bound kernels can drag down the aggregate metric significantly even if the matmuls are well-optimized.

See [[roofline-model]] for the full treatment.

## Practical Benchmarks and Reference Points

| System | Model | Reported MFU |
| --- | --- | --- |
| TPU v4 pod | PaLM 540B | ~57% (Google, 2022) |
| A100 80GB SXM | Megatron-LM GPT-3 175B | ~38–46% |
| H100 SXM5 | LLaMA-class models (optimized) | ~45–55% (community reports) |
| A100 (unoptimized baseline) | Medium transformers | ~20–30% |
| Consumer GPU (RTX 4090) | Small models, small batch | ~5–15% |
| H100 SXM5 | Mistral / Mixtral with expert routing | ~30–40% (MoE routing overhead reduces MFU vs. dense) |

These numbers are illustrative; actual MFU depends heavily on sequence length, batch size, parallelism strategy, and software stack maturity. MoE (mixture-of-experts) architectures typically achieve lower MFU than dense models of equivalent parameter count because expert routing introduces load imbalance and additional communication.

## Improving MFU

- **Use fused kernels**: [[flash-attention]] fuses the attention computation to avoid multiple passes over large activations, significantly improving both MFU and MBU for attention-heavy workloads.
- **Increase batch size**: Larger batches amortize fixed overheads (kernel launches, communication setup) and improve arithmetic intensity of matrix multiplications.
- **Overlap communication with compute**: Libraries like NCCL with async operations, or frameworks that support compute-communication pipelining, reduce the fraction of time GPUs are idle waiting for gradients.
- **Use BF16/FP16 training**: Mixed-precision training routes most work through tensor cores, which have 8–16× higher throughput than FP32 cores on modern hardware.
- **Tune parallelism strategies**: Choosing the right combination of [[data-parallelism]], [[tensor-parallelism]], and [[pipeline-parallelism]] for a given model size and cluster topology can substantially reduce communication overhead.
- **Profile and eliminate small ops**: Tools like Nsight Systems or PyTorch Profiler can identify sequences of small operations that are serialized and un-fused, revealing opportunities for custom kernels or `torch.compile`.
- **Use `torch.compile` or XLA**: Ahead-of-time or JIT graph compilation can fuse elementwise operations, eliminate redundant memory traffic, and reduce Python overhead, typically yielding 10–30% MFU improvement on medium-sized models.
- **Sequence packing**: For variable-length workloads (e.g., instruction fine-tuning), packing multiple short sequences into a single fixed-length context window eliminates padding waste and improves effective tokens-per-second, boosting MFU.
- **Prefetching and async data loading**: Ensuring the data pipeline saturates GPU memory with the next batch before the current step finishes prevents GPU starvation, which is a common low-MFU culprit in fine-tuning workflows.

## Measuring MFU in Practice

Computing MFU requires three quantities:

1. **Hardware peak FLOPS**: Obtained from the hardware vendor's datasheet. Must match the dtype and sparsity mode actually used (e.g., BF16 dense tensor core peak for a BF16 training run).
2. **Model FLOPS per token**: Estimated analytically from the model architecture using the formulas above, or measured using a FLOP-counting utility such as `torch.profiler` with `with_flops=True`, `fvcore`, or `calflops`.
3. **Tokens per second**: Measured empirically during the training run, typically as `(global_batch_size × sequence_length) / step_time_seconds`.

A minimal implementation in PyTorch pseudocode:

```python
# Estimate model FLOPS per token (simplified, dense transformer)
flops_per_token = 6 * num_parameters  # forward + backward approximation

# Measure tokens per second
tokens_per_second = (global_batch_size * seq_len) / step_time

# Compute MFU
model_flops_per_second = flops_per_token * tokens_per_second
mfu = model_flops_per_second / hardware_peak_flops_per_second
```

Common pitfalls:

- **Using the wrong dtype peak**: If training in BF16 but dividing by FP32 peak, MFU will appear artificially low.
- **Forgetting the backward pass**: Using 2N instead of 6N for the FLOPS estimate leads to a 3× underestimate.
- **Including recomputation in the numerator**: If activation checkpointing is enabled, the re-run forward pass adds real FLOPS but does not advance training. Whether to include this in the numerator is a matter of convention; the HFU variant includes it, MFU typically does not.
- **Averaging over a step that includes non-compute time**: If step time is measured end-to-end including optimizer steps, logging, and evaluation, MFU will be lower than the compute-only value.

## Variants and Related Metrics

- **HFU (Hardware FLOPS Utilization)**: Sometimes used as a synonym for MFU, but can also refer to a version that counts all hardware FP operations (including recomputation from activation checkpointing) rather than only "useful" model FLOPS. The distinction matters when comparing runs with and without gradient checkpointing. Introduced formally in the Megatron-LM and PaLM papers to distinguish hardware-centric from model-centric accounting.
- **Achieved TFLOPS**: The raw numerator of MFU (useful FLOPS per second), reported directly without normalizing against hardware peak. Useful when comparing across runs on identical hardware.
- **Tokens per second (TPS)**: A throughput metric that is hardware-agnostic and workload-specific. Less directly comparable across model sizes but easier to reason about for end-to-end system capacity planning.
- **FLOPS per dollar**: Extends MFU to include cost, useful for comparing cloud instance types where price-per-hour differs. Computed as `(MFU × hardware_peak_FLOPS) / cost_per_second`.
- **Samples per second / images per second**: Domain-specific throughput metrics used in vision and multimodal training that play the same role as tokens-per-second but for non-text modalities.
- **GPU utilization (%)**: A coarser metric reported by tools like `nvidia-smi` that measures the fraction of time the GPU has at least one kernel active. A GPU can show 99% utilization while running at 10% MFU if it is executing many small, inefficient kernels. GPU utilization is therefore a poor proxy for MFU and should not be used as a primary performance indicator.

## Historical Context

The term "Model FLOPS Utilization" was popularized in the **PaLM paper** (Chowdhery et al., 2022), which reported 57.5% MFU on TPU v4 pods as a headline efficiency result. Prior work (e.g., Megatron-LM) reported similar metrics under slightly different names. The metric gained broad adoption across the LLM training community as a standardized way to compare training efficiency across systems, complementing the Chinchilla scaling laws work which focused on the relationship between compute budget (total FLOPS) and model quality rather than the efficiency with which that budget is spent.

The practical upper bound of ~50–60% for large-scale training reflects fundamental constraints: even with perfect kernel efficiency, communication in distributed training, optimizer steps, and data loading impose irreducible overhead. Research into communication-free or communication-overlapped training architectures aims to push this ceiling higher.

## Related Concepts

- [[memory-bandwidth-utilization]] — the complementary metric for memory-bound workloads
- [[roofline-model]] — formal framework for determining whether a workload is compute-bound or memory-bound
- [[flash-attention]] — key technique that improves both MFU and MBU for attention
- [[tensor-parallelism]] — parallelism strategy that affects inter-device communication overhead and thus MFU
- [[pipeline-parallelism]] — introduces pipeline bubbles that reduce effective MFU
- [[data-parallelism]] — gradient all-reduce overhead impacts MFU at scale
- [[mixed-precision-training]] — prerequisite for achieving high MFU on tensor cores
- [[arithmetic-intensity]] — the per-operation property that determines whether an op is compute-bound or memory-bound
- [[activation-checkpointing]] — trades recomputation FLOPS for reduced memory, affecting MFU accounting
- [[all-reduce]] — collective communication primitive whose latency and bandwidth cost directly subtracts from MFU in data-parallel training
- [[chinchilla-scaling-laws]] — governs how total compute budget should be allocated between model size and token count; MFU determines how quickly that budget is consumed

## Sources

- [[1-overview.md]]
