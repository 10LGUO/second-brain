```yaml
title: NVIDIA Nsight
type: entity
tags: [gpu, cuda, profiling, tools, nvidia, performance-analysis]
created: 2026-04-19
updated: 2026-04-19
sources: [kernel-dev.md]
```

# NVIDIA Nsight

NVIDIA Nsight is a suite of GPU profiling and debugging tools produced by NVIDIA, used to analyze the performance of CUDA kernels running on NVIDIA GPUs.

## Who/What It Is

Nsight is NVIDIA's primary developer toolchain for GPU performance analysis. It includes:
- **Nsight Compute:** Kernel-level profiler providing detailed hardware counter data (memory throughput, compute utilization, warp efficiency, occupancy, bank conflicts, etc.).
- **Nsight Systems:** System-level timeline profiler showing CPU-GPU interaction, API calls, kernel launch timing, and data transfers.
- **Nsight Graphics:** For graphics workloads (less relevant to operator development).

## Why It Matters to This Wiki

Nsight is the essential measurement tool for GPU operator development. The SJTU kernel development guide explicitly names it as a required tool:
- Used to **identify performance bottlenecks** after an initial correct implementation is produced.
- Measures actual utilization (compute utilization, bandwidth utilization) against theoretical roofline limits.
- Guides the iterative optimization process: measure → identify bottleneck → apply technique → measure again.
- Must be used **alongside** theoretical analysis ([[arithmetic-intensity]], [[roofline-model]]) — theoretical understanding is necessary to interpret profiling results.

## Key Capabilities in Operator Development Context

- Report actual memory bandwidth utilization vs. peak.
- Report compute throughput vs. peak.
- Identify [[register-spill]] via local memory traffic metrics.
- Identify [[bank-conflict]] via shared memory bank conflict counters.
- Measure [[coalesced-memory-access]] efficiency via L2 sector hit/miss rates and global load/store efficiency.
- Show occupancy, active warp counts, and stall reasons per instruction.

## Related Entities and Concepts

- [[cuda-kernel-optimization]]
- [[roofline-model]]
- [[arithmetic-intensity]]
- [[register-spill]]
- [[bank-conflict]]

## Sources

- [[kernel-dev]]

---
