```yaml
title: Parallelism Strategies
type: concept
tags: [distributed-training, data-parallelism, tensor-parallelism, pipeline-parallelism, expert-parallelism, moe, all-reduce, all-to-all]
created: 2026-07-02
updated: 2026-07-02
sources: []
```

# Parallelism Strategies

Four parallelism strategies are commonly combined in large model training. Each targets a different bottleneck and uses a different communication pattern.

## Data Parallelism (DP)

**What is split**: data (each GPU holds a full model replica, processes a different mini-batch)

**How it works**:
1. Each GPU runs forward + backward on its mini-batch
2. All-Reduce gradients across all GPUs (average)
3. Every GPU applies the same averaged gradient → replicas stay in sync

Averaged gradient is mathematically equivalent to computing on the full concatenated batch of size `N × B`.

**Communication**: All-Reduce on gradients after each backward pass. Can be overlapped with backward computation via gradient bucketing — bucket fills up → trigger All-Reduce while backward continues on earlier layers.

**Memory**: Each GPU stores a full model replica. Does not help with model size, only throughput.

**Best for**: models that fit on a single GPU, scaling throughput.

## Tensor Parallelism (TP)

**What is split**: individual weight matrices, partitioned across GPUs

```
W: [d_model, 4*d_model] split column-wise across 2 GPUs:
  GPU0: W[:, :2*d_model]
  GPU1: W[:, 2*d_model:]
```

Every token's computation is distributed — all GPUs participate in computing each layer's output, then results are merged via All-Reduce or All-Gather.

**Communication**: All-Reduce or All-Gather after each layer. Communication happens on the critical path (cannot be overlapped with compute for that layer).

**Memory**: Each GPU stores a fraction of each weight matrix — reduces per-GPU memory proportionally to TP degree.

**Best for**: dense layers that are too large for a single GPU. Requires high-bandwidth interconnect (NVLink) because communication is on the critical path.

## Pipeline Parallelism (PP)

**What is split**: layers — different GPUs hold different layers of the model

```
GPU0: layers 0-7
GPU1: layers 8-15
GPU2: layers 16-23
GPU3: layers 24-31
```

Data flows through GPUs sequentially. Micro-batching reduces the idle "bubble" — while GPU1 processes micro-batch 1, GPU0 starts on micro-batch 2.

**Communication**: point-to-point (activations passed between adjacent GPUs). Low communication volume but introduces pipeline bubble overhead.

**Memory**: Each GPU stores only its assigned layers — reduces memory proportionally to PP degree.

**Best for**: very deep models. Bubble overhead limits efficiency; micro-batch size tuning is critical.

## Expert Parallelism (EP)

**What is split**: experts within a MoE layer — different GPUs hold different complete experts (each expert is a full FFN)

```
Layer i (MoE, 256 experts):
  GPU0: expert 0-7    (8 complete FFNs)
  GPU1: expert 8-15
  ...
  GPU31: expert 248-255
```

Each token is routed by the router to top-k experts. Tokens travel to whichever GPU holds their assigned expert, are computed there, then return.

**Communication**: All-to-All twice per MoE layer (dispatch tokens to expert GPUs, combine results back). Unlike TP, GPUs do not collaborate on the same token — each token goes to specific GPUs only.

**Memory**: Each GPU stores only its fraction of experts per MoE layer. Dense layers (attention, non-MoE FFN) still need separate parallelism.

**Best for**: MoE models where the number of experts exceeds what fits on one GPU.

## MoE Parameter Count

For a standard FFN with W parameters, a MoE layer with N experts does not simply have `N × W` parameters — in practice each expert is made smaller:

```
Standard FFN:    hidden_dim = 4 * d_model,  params = W
MoE (N experts): hidden_dim = 4 * d_model / k (per expert),  total params = N × W/k
```

Design goal: more total parameters (capacity) without increasing FLOPs — achieved by activating only top-k experts (sparse activation) while keeping each expert smaller than the original FFN.

Example (DeepSeek-V3): 256 experts, top-2 activation, each expert ~1/8 the size of a standard FFN → total MoE params = 32W, activated params per token = W/4.

## Comparison

| | Data Parallel | Tensor Parallel | Pipeline Parallel | Expert Parallel |
|---|---|---|---|---|
| What is split | Data | Weight matrices | Layers | Experts (within MoE layer) |
| Communication | All-Reduce (gradients) | All-Reduce / All-Gather (activations) | P2P (activations) | All-to-All (tokens) |
| On critical path | No (overlappable) | Yes | Partially (bubble) | Yes |
| Memory saving | No | Yes (per layer) | Yes (per layer group) | Yes (MoE layers only) |
| Applies to | Any model | Dense layers | Any model | MoE layers |
| Scales with | Batch size | Layer width | Model depth | Number of experts |

## Combining Strategies in Practice

Large MoE models typically combine all four:

```
DP: replicate across node groups for throughput
TP: split attention + dense FFN within a node (NVLink bandwidth)
PP: split layers across nodes (fewer cross-node communications)
EP: split experts within MoE layers across GPUs
```

The choice of degrees (DP=8, TP=4, PP=4, EP=32 etc.) is determined by model architecture, hardware topology, and communication bandwidth at each level.

## Cross-References

- [[collective-communication]] — All-Reduce, All-to-All, Ring All-Reduce mechanics
- [[1-overview-llm-training-infra]] — training infrastructure overview
- [[1-overview-compute-communication-overlap]] — overlapping All-Reduce with backward pass
