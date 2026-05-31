```yaml
title: "Lecture 9 — Performance Optimization (性能优化讲解)"
type: source
tags: [gpu, profiling, performance, operator-fusion, quantization, cuda-graph, torch-compile, streams, pinned-memory, distributed, static-graph, dynamic-graph, optimization]
created: 2026-05-31
updated: 2026-05-31
sources: [9-profile optimization.pdf]
```

# Lecture 9 — Performance Optimization (性能优化讲解)

Source: 上交大 AI infra 团队 (SJTU AI Infra Team), lecture series.

---

## 1. Performance Optimization Overview (性能优化概述)

Performance optimization is fundamentally about improving GPU utilization. The goal is to maximize the use of hardware compute and memory bandwidth, eliminate idle time, and reduce unnecessary CPU↔GPU interaction.

### 1.1 Profiling (profile)

Always start with a coarse-grained profile before optimizing. The standard workflow:

1. Run `nsys` (Nsight Systems) or PyTorch Profiler to capture a timeline of host + device activity.
2. Identify the bottleneck: is time spent in kernels, in CPU overhead (kernel launch), in data transfers, or in communication?
3. Zoom into the bottleneck operator or phase.

**Tools:**

| Tool | Purpose |
|---|---|
| `nsys profile` | Timeline of CPU/GPU events, kernel durations, memory transfers, stream usage |
| `ncu` (Nsight Compute) | Per-kernel roofline analysis, memory throughput, occupancy |
| PyTorch Profiler + TensorBoard | Python-level operator timing, CUDA kernel attribution, memory timeline |
| `torch.profiler.profile` | Integrated profiler, outputs Chrome trace or TensorBoard |

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")
```

Profile → identify bottleneck → target that one thing → re-profile. Do not optimize blindly.

See [[9-perf-opt-profiling]] for the full profiling workflow.

---

### 1.2 Common Performance Optimization Points (常见的性能优化点)

#### 1.2.1 Operator-Level Optimization (算子角度)

**Operator vectorization:** Profile to find which kernels consume the most time. Check whether the operator is using the hardware's vector units efficiently. Shape mismatches (e.g., sizes that are not multiples of 16 or 128) can dramatically reduce throughput by misaligning memory accesses or wasting warp lanes.

**Operator fusion:** Merge multiple small operators into a single kernel to eliminate intermediate HBM round-trips. Examples:
- Fused LayerNorm (compute mean + variance + normalize in one pass)
- Fused Adam (parameter update + moment update in one kernel)
- FlashAttention (Q·Kᵀ + softmax + ·V without materializing the attention matrix)

PyTorch 2.x `torch.compile` performs automatic fusion. Custom CUDA kernels allow manual fusion.

**Shape and format optimization:** Certain tensor shapes and memory layouts (e.g., NHWC vs. NCHW, row-major vs. column-major) are better suited to specific hardware paths (Tensor Cores, cuDNN algorithms). Profile across representative shapes and pick the layout that maximizes throughput.

**Quantization / low-precision:** The most impactful single optimization. Lowering precision (FP16 → INT8 → INT4 → FP8) reduces:
- Memory footprint (fewer bytes per weight)
- Memory bandwidth demand (fewer bytes to transfer)
- Compute cost (Tensor Cores have higher throughput at lower precision)

Quantization is the highest-leverage performance optimization available.

See [[1-overview-operator-fusion]], [[1-overview-precision-convergence]].

#### 1.2.2 Host/Device Optimization (host和device优化)

**Minimize CPU-GPU synchronization points.** Every `.item()`, `.numpy()`, `torch.cuda.synchronize()`, or scalar loss check forces the CPU to wait for the GPU to finish — stalling the entire pipeline.

**Avoid many small kernel launches.** Each CUDA kernel launch has ~5–20 µs of CPU-side overhead (driver dispatch). A model with thousands of small operators accumulates significant launch overhead. Solutions:
- `torch.compile` / `torch.jit.script`: fuse operators, reduce launch count
- CUDA Graph: record the launch sequence once and replay it, eliminating per-step overhead

**Async execution model:**
```
CPU                    GPU
 |---launch kernel A-->|
 |---launch kernel B-->| ...kernel A running...
 |---launch kernel C-->| ...kernel B running...
 |                     | ...kernel C running...
```
The CPU should always be ahead of the GPU in the launch queue, never blocking.

See [[9-perf-opt-host-device]].

#### 1.2.3 Distributed Communication Optimization (分布式通信优化)

**Buffer aggregation:** Rather than issuing many small AllReduce / AllGather calls, accumulate gradients and communicate larger buffers at once. Fewer calls amortize the per-call latency.

**Compute-communication overlap (通信运算并行):** The key principle — never let the GPU sit idle waiting for a communication to complete.

- Use non-blocking collectives (`dist.all_reduce(..., async_op=True)`)
- Bucket gradients: start AllReduce on a bucket as soon as all its gradients are ready, while backward continues computing gradients for earlier layers
- Pipeline parallelism uses send/recv across pipeline stages on separate streams

**Reduce-Scatter / AllGather vs. AllReduce:** In ZeRO-3, replace AllReduce (=Reduce-Scatter + AllGather) with separate primitives, allowing the AllGather for the next layer to overlap with the compute of the current layer.

See [[1-overview-compute-communication-overlap]].

#### 1.2.4 CUDA Streams (stream)

A CUDA stream is a sequence of operations that execute in order on the GPU. Operations on *different* streams can execute concurrently.

Use streams to overlap:
- Compute kernel (stream 0) + data prefetch H→D (stream 1)
- AllGather comm (stream 1) + forward compute (stream 0)
- Backward compute (stream 0) + AllReduce of earlier buckets (stream 2)

```python
compute_stream = torch.cuda.Stream()
comm_stream = torch.cuda.Stream()

with torch.cuda.stream(compute_stream):
    output = model(input)

with torch.cuda.stream(comm_stream):
    dist.all_reduce(grad_bucket, async_op=True)

# Synchronize when needed
torch.cuda.current_stream().wait_stream(comm_stream)
```

Key rule: events (`.record_event()` / `.wait_event()`) express dependencies between streams without blocking the CPU.

#### 1.2.5 torch.compile and CUDA Graph

**`torch.compile` (PyTorch 2.0+):**
- Traces the computation graph using TorchDynamo (Python bytecode capture)
- Applies graph-level optimizations: operator fusion, constant folding, layout selection
- Lowers to efficient backends (Inductor → Triton kernels, or nvFuser)
- Eliminates Python overhead for the traced region

```python
model = torch.compile(model)  # wraps model; compilation happens on first call
```

**CUDA Graph:**
- Records a sequence of CUDA kernel launches into a graph object
- Replays the graph on subsequent steps with a single `graph.replay()` call
- Eliminates all per-step CPU kernel-launch overhead (~5–20 µs per kernel)
- Requires static shapes and static memory addresses (no dynamic allocation)

```python
# Warmup
for _ in range(3):
    y = model(x)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_y = model(static_x)

# Replay
static_x.copy_(new_input)
g.replay()
output = static_y  # already updated in place
```

**Small-model effect:** For models with many small kernels, `torch.compile` + CUDA Graph can give 2–5× speedup purely by eliminating launch overhead, with no algorithmic change.

**Large-model effect:** For compute-bound large models, gains are more modest (~5–20%) since kernel time dominates launch overhead.

See [[9-perf-opt-torch-compile-cuda-graph]].

#### 1.2.6 Compute and Storage Buffer Optimization (计算和存储缓冲优化)

Pre-allocate output buffers and reuse them across iterations to avoid repeated `malloc` / `cudaMalloc` calls. Use memory pools (`torch.cuda.memory.CUDAPluggableAllocator` or PyTorch's built-in caching allocator).

Cache intermediate tensors across steps when they are reused (e.g., static KV cache in inference, fixed position embeddings). Avoid redundant allocations in the hot path.

#### 1.2.7 Pinned Memory (pin_memory)

CPU-to-GPU transfers are faster when the CPU buffer is **pinned** (page-locked), because the GPU's DMA engine can access pinned memory directly without a kernel bounce-copy.

```python
# Pinned DataLoader
loader = DataLoader(dataset, pin_memory=True, num_workers=4)

# Manual pinning
tensor = torch.zeros(1024).pin_memory()
tensor_gpu = tensor.cuda(non_blocking=True)  # async H→D, no CPU stall
```

`non_blocking=True` combined with pinned memory allows the `.cuda()` transfer to run concurrently with CPU work on a separate stream.

**Caution:** Pinned memory is a limited OS resource; allocating too much degrades system performance. Use selectively for large, frequently-transferred buffers.

#### 1.2.8 Chip-Specific Optimization (根据芯片特性的优化)

Every hardware has architectural quirks that affect optimal kernel design:
- NVIDIA Ampere: `cp.async` for direct global→shared memory DMA; TF32 for GEMM
- NVIDIA Hopper: TMA for bulk async tile copies; FP8 Tensor Cores; Thread Block Clusters
- Domestic AI chips: Vary widely; operator implementations must be validated for each hardware's ISA and memory system

Profile on the actual target hardware. Roofline analysis (arithmetic intensity vs. peak FLOP/s and memory bandwidth) determines whether a kernel is compute-bound or memory-bound — and thus which hardware properties to exploit.

---

### 1.3 Summary (总结)

Performance optimization priority order (roughly):

1. **Profile first** — identify the actual bottleneck before touching code
2. **Quantization** — highest-leverage; reduces memory, bandwidth, and compute simultaneously
3. **Operator fusion** — eliminates intermediate HBM traffic; use `torch.compile` or custom kernels
4. **Compute-communication overlap** — for distributed training; overlap is free throughput
5. **Host/device sync elimination** — remove `.item()`, use CUDA Graph for static workloads
6. **Stream-level parallelism** — overlap compute with prefetch, comm with compute
7. **Memory layout / shape alignment** — ensure tensor shapes are hardware-friendly
8. **Pinned memory + async transfers** — eliminate PCIe transfer stalls

---

## 2. Static Graph Performance Optimization (静态图性能优化)

### 2.1 Optimization Properties of Static Graphs (静态图拥有的优化特性)

Static graphs (shapes fixed at compile time) unlock optimizations that are impossible in fully dynamic execution:

- **Shape specialization:** The compiler knows exact tensor dimensions and can generate kernels tuned for those shapes (e.g., tile sizes, loop unrolling factors).
- **Padding for alignment:** Pad sequence lengths and batch sizes to multiples of 8, 16, or 128 to ensure Tensor Core alignment. `torch.nn.functional.pad` + fixed `max_seq_len` is the standard approach.
- **Memory planning:** With known shapes, the runtime can pre-assign non-overlapping memory regions to all intermediate tensors, eliminating dynamic allocation.
- **Kernel fusion across boundaries:** The compiler can fuse operators across module boundaries that dynamic dispatch cannot see.

**`torch.compile` with static shapes:**

```python
model = torch.compile(model, dynamic=False)  # assume fixed shapes
```

Alternatively, set `torch._dynamo.config.assume_static_by_default = True`.

With static shapes, `torch.compile` can fully capture the computation graph and apply the full suite of Inductor optimizations.

### 2.2 Other Static Graph Features (其它特性)

- `torch.jit.script` / `torch.jit.trace` (older approach): export to TorchScript for deployment without Python overhead
- `torch.export` (PyTorch 2.x): stricter graph capture for AOT (ahead-of-time) compilation and deployment
- TensorRT integration: convert static-shape PyTorch models to optimized TensorRT engines for inference

---

## 3. Dynamic Graph Performance Optimization (动态图性能优化)

Dynamic graphs (variable shapes, control flow) are harder to optimize but not hopeless:

- `torch.compile` with `dynamic=True`: handles symbolic shapes; generates shape-generic kernels with some overhead vs. fully static
- Shape bucketing: quantize input shapes into a small set of buckets; compile one kernel per bucket; eliminates recompilation on every new shape
- Avoid Python-level loops over sequence elements — push looping into compiled regions where possible
- Profile shape distributions in production to choose bucket boundaries

---

## Key Cross-References

- [[9-perf-opt-profiling]] — profiling workflow and tool selection
- [[9-perf-opt-host-device]] — host/device sync points and launch overhead
- [[9-perf-opt-torch-compile-cuda-graph]] — torch.compile and CUDA Graph in depth
- [[1-overview-compute-communication-overlap]] — distributed comm-compute overlap
- [[1-overview-operator-fusion]] — operator fusion techniques
- [[1-overview-precision-convergence]] — quantization and precision trade-offs
- [[kernel-dev-roofline-model]] — roofline model for identifying compute vs. memory bottlenecks
- [[8-memory-opt-flash-attention]] — FlashAttention as an example of operator-level memory optimization
- [[1-overview-gpu-memory-hierarchy]] — memory hierarchy context for optimization decisions
