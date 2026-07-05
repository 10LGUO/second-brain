```yaml
title: Collective Communication
type: concept
tags: [distributed-training, data-parallelism, all-reduce, ring-all-reduce, nccl, gradients]
created: 2026-07-02
updated: 2026-07-02
sources: []
```

# Collective Communication

Collective operations coordinate data across multiple GPUs. They are the foundation of data parallelism — after each forward/backward pass, gradients must be synchronized across all GPUs so every GPU updates its weights identically.

## The Four Operations

| Operation | Sender | Aggregation | Receiver | Result |
|---|---|---|---|---|
| Scatter | 1 node splits and sends | None | Each node gets one chunk | Data distributed |
| Reduce-Scatter | All nodes | Sum per chunk | Each node keeps one chunk | Partial sums distributed |
| All-Gather | All nodes | None | All nodes get full data | Full data replicated |
| All-Reduce | All nodes | Sum | All nodes get full sum | Full sum replicated |

```
Scatter:
  GPU0[A,B,C,D] → GPU0:A, GPU1:B, GPU2:C, GPU3:D

Reduce-Scatter:
  GPU0[A0,B0,C0,D0]
  GPU1[A1,B1,C1,D1]  →  GPU0: ΣA,  GPU1: ΣB,  GPU2: ΣC,  GPU3: ΣD
  GPU2[A2,B2,C2,D2]
  GPU3[A3,B3,C3,D3]

All-Gather:
  GPU0:A, GPU1:B, GPU2:C, GPU3:D  →  every GPU: [A,B,C,D]

All-Reduce:
  GPU0[A0,B0], GPU1[A1,B1]  →  every GPU: [ΣA, ΣB]
  = Reduce-Scatter + All-Gather
```

## Why All-Reduce for Data Parallelism

Each GPU holds a full model replica and processes a different mini-batch. After backward pass, each GPU has gradients computed on its own data. Without synchronization, each GPU would update weights in a different direction and the replicas would diverge.

All-Reduce averages the gradients across all GPUs. This is mathematically equivalent to computing gradients over the full concatenated batch:

```
avg(grad_GPU0, grad_GPU1, ..., grad_GPUN)
= (1/N) × Σ [(1/B) × Σ loss_i per GPU]
= (1/N*B) × Σ loss_i over all samples
= gradient of full batch of size N*B
```

After All-Reduce every GPU applies the same gradient to the same weights — replicas stay in sync.

**Side effect**: effective batch size becomes `N × B`. Larger batches reduce gradient noise but may require learning rate scaling (common rule: multiply LR by N when batch size multiplies by N) and can hurt generalization (less noise → sharper minima).

## Ring All-Reduce

Naive centralized approach — all GPUs send to one master node, master reduces and broadcasts back — creates a bandwidth bottleneck at the master that grows linearly with N.

Ring All-Reduce distributes the work evenly across all links:

```
GPU0 → GPU1 → GPU2 → GPU3 → GPU0
```

### Phase 1: Reduce-Scatter (N-1 rounds)

Each round every GPU simultaneously sends one chunk to its right neighbor and receives one chunk from its left neighbor, accumulating as it goes:

```
Round 1:  GPU0 sends D0→GPU1,  GPU1 sends A1→GPU2,  GPU2 sends B2→GPU3,  GPU3 sends C3→GPU0
Round 2:  GPU1 sends D0+D1→GPU2,  GPU2 sends A1+A2→GPU3, ...
...
Result:   GPU0: ΣA,  GPU1: ΣB,  GPU2: ΣC,  GPU3: ΣD
```

Note: the name "Reduce-Scatter" describes the final effect, not execution order. In the Ring implementation, reduction and forwarding are interleaved each round — there is no separate "reduce phase" followed by a "scatter phase."

### Phase 2: All-Gather (N-1 rounds)

Each GPU broadcasts its fully-reduced chunk around the ring:

```
Round 1:  GPU0 sends ΣA→GPU1,  GPU1 sends ΣB→GPU2, ...
...
Result:   every GPU holds [ΣA, ΣB, ΣC, ΣD]
```

### Communication Volume

Each GPU sends and receives `2(N-1)/N × P × bytes` ≈ `2P × bytes` total, where P = number of parameters. This is **independent of N** — adding more GPUs does not increase per-GPU communication volume. This is the key scalability advantage of Ring All-Reduce over centralized approaches.

The absolute communication cost scales with model size (P), not batch size. Larger batches amortize this fixed cost over more computation, improving the compute-to-communication ratio.

## Cross-References

- [[1-overview-compute-communication-overlap]] — hiding all-reduce latency behind compute
- [[1-overview-llm-training-infra]] — data parallelism, tensor parallelism, pipeline parallelism
