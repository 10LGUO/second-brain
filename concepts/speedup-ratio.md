```yaml
title: Speedup Ratio
type: concept
tags: [distributed-training, performance, parallelism, amdahl, efficiency]
created: 2026-05-17
updated: 2026-05-17
sources: []
```

# Speedup Ratio

The **speedup ratio** (加速比) measures how much faster a parallelized or optimized system runs compared to a baseline:

```
S = T₁ / Tₙ
```

where T₁ is baseline time (e.g., single-GPU) and Tₙ is the optimized time (e.g., N-GPU). A speedup of S = 6.7 on 8 GPUs means the system is 6.7× faster, not the ideal 8×.

## Amdahl's Law

Amdahl's Law gives the theoretical upper bound when only a fraction of the workload can be parallelized:

```
S = 1 / (f_serial + (1 - f_serial) / N)
```

- `f_serial`: fraction of execution time that is inherently serial (cannot be parallelized)
- `N`: number of parallel workers

Even with infinite workers, the maximum speedup is `1 / f_serial`. A 5% serial fraction caps the speedup at 20× regardless of how many GPUs are added.

## Why Linear Speedup Is Rarely Achieved in LLM Training

[[1-overview-llm-training-infra]] identifies the main sources of sub-linear speedup:

| Source | Mechanism |
|---|---|
| Inter-node communication | All-Reduce / All-Gather across InfiniBand; time grows with scale |
| Pipeline bubbles | Idle stages waiting for micro-batches in pipeline parallelism |
| MoE token imbalance | Some expert devices overloaded, others idle |
| Checkpoint I/O | Periodic blocking writes to distributed storage |
| Stragglers | One slow device holds back the entire synchronous step |

The aggregate effect is captured by **MFU (Model FLOP Utilization, 模型浮点利用率)**: actual useful FLOP/s divided by peak hardware FLOP/s. A high MFU (40–60% on A100s) indicates the parallelization overhead is well-managed.

## Relationship to Compute-Communication Overlap

[[1-overview-compute-communication-overlap]] frames the problem directly: communication that cannot be hidden behind compute reduces the effective speedup ratio. When communication is fully overlapped with computation, the speedup ratio approaches its theoretical maximum; when communication is exposed (serialized), it appears as `f_serial` in Amdahl's Law and caps the benefit of adding more devices.

## Related Concepts

- [[1-overview-llm-training-infra]]
- [[1-overview-compute-communication-overlap]]
- [[1-overview-mfu-model-flops-utilization]]
- [[1-overview-operator-fusion]]

## Sources

- Derived from discussion of distributed LLM training infrastructure.
