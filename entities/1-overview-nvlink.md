```yaml
title: NVLink
type: entity
tags: [gpu, interconnect, hardware, nvidia, communication]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# NVLink

NVLink is NVIDIA's proprietary high-bandwidth chip-to-chip interconnect. It enables direct HBM-to-HBM communication between GPU chips without routing data through the CPU or PCIe bus, achieving significantly higher inter-GPU bandwidth than PCIe.

## Why It Matters to This Wiki

NVLink is the physical interconnect over which NCCL collective operations run for intra-node GPU communication in large-scale LLM training. Its bandwidth directly affects the achievable compute-communication overlap ratio and distributed training efficiency.

## Key Works, Products, or Contributions

- Direct GPU-to-GPU communication (bypasses CPU and PCIe).
- Much higher bandwidth than PCIe (e.g., NVLink 4.0: 900 GB/s bidirectional per GPU in NVSwitch configurations).
- Enables efficient AllReduce, AllGather, and other collective operations within a node.
- Used together with [[nccl]] for intra-node communication in distributed training.

## Related Entities and Concepts

- [[nccl]]
- [[ai-chip-architecture]]
- [[compute-communication-overlap]]
- [[llm-training-infra]]

## Sources

- [[1-overview]]

---
