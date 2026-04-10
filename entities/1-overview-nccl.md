```yaml
title: NCCL (NVIDIA Collective Communications Library)
type: entity
tags: [library, distributed-communication, gpu, training, inference, ai-infra]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# NCCL (NVIDIA Collective Communications Library)

NCCL (NVIDIA Collective Communications Library) is a distributed communication library providing multi-GPU and multi-node collective communication primitives — most notably AllReduce — optimized for NVIDIA GPU topologies. It is a mandatory skill for AI infra engineers and is used in both distributed training (gradient synchronization) and distributed inference.

## Why It Matters to This Wiki

NCCL is the foundational communication layer underlying virtually all large-scale distributed deep learning workloads. Understanding NCCL is essential for reasoning about training throughput, scaling efficiency, and infrastructure bottlenecks in any serious AI system. It surfaces repeatedly in discussions of [[distributed-training]], [[model-parallelism]], [[data-parallelism]], and [[tensor-parallelism]], and is directly relevant to the performance characteristics of frameworks like [[pytorch]], [[megatron-lm]], and [[deepspeed]].

## Key Primitives

- **AllReduce** — Aggregates (sums, averages) tensors across all participating GPUs and returns the result to every rank. The most commonly used primitive in data-parallel gradient synchronization.
- **AllGather** — Each rank contributes a tensor; all ranks receive the concatenation of all tensors. Used heavily in tensor parallelism to reconstruct sharded activations.
- **ReduceScatter** — Reduces tensors across ranks and scatters disjoint shards back to each rank. Paired with AllGather in ring-based tensor parallelism (e.g., Megatron-style).
- **Broadcast** — Sends a tensor from one root rank to all other ranks.
- **Reduce** — Like AllReduce but the result lands only on a designated root rank.
- **Send/Recv (point-to-point)** — Unicast primitives added in later NCCL versions, enabling pipeline parallelism implementations.

## Topology Awareness

NCCL automatically detects and exploits the physical interconnect topology of the system:

- **NVLink** — High-bandwidth GPU-to-GPU links within a node (e.g., NVLink 4.0 on H100 offers ~900 GB/s bidirectional per GPU). NCCL prefers NVLink rings or trees when available.
- **NVSwitch** — A switch fabric connecting all GPUs in a node (e.g., DGX H100), enabling all-to-all bandwidth at full NVLink speed without bottlenecking through the CPU.
- **PCIe** — Fallback path when NVLink is absent; significantly lower bandwidth (~64 GB/s per slot, shared).
- **InfiniBand / RoCE** — Inter-node communication via RDMA-capable network fabrics. NCCL integrates with these through the **NCCL Net plugin** and leverages **GPUDirect RDMA** to transfer data directly from GPU memory to the NIC, bypassing the CPU and host memory.

NCCL uses a ring-based or tree-based algorithm depending on message size and topology, automatically selecting the most efficient strategy.

## Algorithm Selection

| Message Size | Preferred Algorithm | Reason |
| --- | --- | --- |
| Small | Tree (recursive halving/doubling) | Lower latency, fewer steps |
| Large | Ring AllReduce | Higher bandwidth utilization |
| Mixed | Hybrid / auto-tuned | NCCL internal heuristics |

The ring AllReduce algorithm divides bandwidth cost evenly: for N GPUs, each GPU sends and receives `(N-1)/N` of the data volume, making it bandwidth-optimal.

## Integration Points

- **PyTorch DDP** (`torch.distributed`) — Uses NCCL as the default backend for GPU collective communication.
- **PyTorch FSDP** — Relies on NCCL AllGather and ReduceScatter for sharded parameter and gradient communication.
- **Megatron-LM** — Uses NCCL directly for tensor and pipeline parallelism communication.
- **DeepSpeed** — Uses NCCL (or optionally MPI) for ZeRO-stage gradient and parameter communication.
- **JAX / XLA** — Can use NCCL through the XLA collective ops backend on NVIDIA hardware.

## Key Environment Variables

NCCL exposes a large number of environment variables for tuning and debugging:

- `NCCL_DEBUG=INFO` / `WARN` / `TRACE` — Controls verbosity of NCCL logging. Essential for debugging hangs and performance issues.
- `NCCL_IB_DISABLE=1` — Forces NCCL to avoid InfiniBand, falling back to socket transport.
- `NCCL_SOCKET_IFNAME` — Specifies the network interface for socket-based transport.
- `NCCL_P2P_DISABLE=1` — Disables peer-to-peer GPU communication (useful for debugging topology issues).
- `NCCL_ALGO` — Forces a specific algorithm (e.g., `Ring`, `Tree`).
- `NCCL_PROTO` — Forces a specific protocol (`Simple`, `LL`, `LL128`). `LL` and `LL128` are low-latency protocols using inline data in flag words.
- `NCCL_NTHREADS` — Controls the number of CUDA threads per NCCL communication block.
- `NCCL_BUFFSIZE` — Sets the size of the communication buffer per channel.

## Common Failure Modes

- **Hangs / deadlocks** — Occur when ranks do not call collectives in the same order, or when one rank crashes mid-collective. Always ensure symmetric collective calls across all ranks.
- **NCCL timeout** — Default watchdog timeout kills jobs after a collective stalls for too long. Configurable but masking the root cause is dangerous.
- **Bus ID mismatches** — Misconfigured GPU visibility (`CUDA_VISIBLE_DEVICES`) causes NCCL to build an incorrect topology graph.
- **InfiniBand GID index errors** — Common in multi-tenant clusters where the RoCE GID table index differs across nodes; fix by setting `NCCL_IB_GID_INDEX`.
- **GPUDirect RDMA failures** — Kernel module mismatches (`nvidia-peermem` not loaded) silently fall back to slower paths; verify with `NCCL_DEBUG=INFO`.

## Performance Mental Model

The effective throughput of an AllReduce across N nodes with bandwidth B (per link) in a ring is approximately:

```text
Effective bandwidth ≈ B × (N - 1) / N
```

For large N this approaches B, meaning ring AllReduce is bandwidth-optimal. The practical bottleneck is almost always the inter-node link (InfiniBand or Ethernet), not NVLink within a node.

For tensor parallelism within a node (common in LLM inference), NCCL AllReduce over NVLink is fast enough that compute typically dominates, not communication — a key reason tensor parallelism is usually confined within a single node.

## Related Concepts

- [[distributed-training]]
- [[data-parallelism]]
- [[tensor-parallelism]]
- [[pipeline-parallelism]]
- [[model-parallelism]]
- [[gradient-synchronization]]
- [[nvlink]]
- [[infiniband]]
- [[gpudirect-rdma]]
- [[pytorch]]
- [[megatron-lm]]
- [[deepspeed]]

## Sources

- [[1-overview.md]]
