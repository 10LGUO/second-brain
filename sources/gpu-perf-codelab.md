```yaml
title: GPU Performance Optimization Codelab (Lesson 11)
type: source
tags: [cuda, gpu, profiling, optimization, nsight, torch-profiler, multi-model, training]
created: 2026-04-29
updated: 2026-04-29
sources: [11. GPU performance optimization codelab.pdf]
```

# GPU Performance Optimization Codelab

**第十一课：基于多模型训练场景的GPU性能优化实践**
Lesson 11: GPU Performance Optimization Practice Based on Multi-Model Training Scenarios

---

## 1. Overview of Training Environment (概览训练环境)

Environment setup includes inspection of GPU hardware, CUDA version, and driver state. Key commands:

```bash
nvidia-smi          # GPU status, memory usage, utilization
nvcc --version      # CUDA toolkit version
nvidia-smi topo -m  # GPU topology / NVLink connections
```

---

## 2. Common Tools (常见基元/工具)

### torch.profiler — Basic Profiling Steps

The fundamental workflow for profiling a PyTorch training loop:

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, data in enumerate(dataloader):
        train_step(data)
        prof.step()
```

Key parameters:
- `wait` — steps to skip at the start
- `warmup` — steps to warm up (recorded but discarded)
- `active` — steps to actually profile
- `on_trace_ready` — callback when trace is ready (e.g. write to TensorBoard)
- `record_shapes` — record tensor shapes
- `with_stack` — record Python call stacks

### torch.profiler — Reading the Trace

The profiler output (viewed in TensorBoard or Chrome `chrome://tracing`) shows:
- CPU ops timeline
- CUDA kernel timeline
- Memory allocations
- Operator call stacks

Key things to look for:
- **Gaps between GPU kernels** — CPU is bottlenecking (not enough work queued)
- **Long kernels** — potential optimization targets
- **Memory copies** — H2D/D2H transfers interrupting computation

---

## 3. nsight Tools (nsight相关工具)

### Nsight Systems (`nsys`)

System-level profiler — shows the big picture: CPU/GPU timeline, CUDA streams, memory transfers, NCCL communication.

```bash
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas \
     --capture-range=cudaProfilerApi \
     --stop-on-range-end=true \
     --cudabacktrace=true \
     -x true \
     -o my_profile \
     python train.py
```

Output: `.nsys-rep` file, viewed in Nsight Systems GUI.

### Nsight Compute (`ncu`)

Kernel-level profiler — deep metrics for individual CUDA kernels: memory throughput, compute utilization, occupancy, warp efficiency, cache hit rates.

```bash
ncu --set full -o profile_output python train.py
```

Key metrics reported:
- **SM Throughput** — how close to peak compute utilization
- **Memory Throughput** — how close to peak HBM bandwidth
- **Occupancy** — warps resident vs maximum
- **Warp Efficiency** — fraction of warp slots doing useful work (vs divergence)
- **L1/L2 hit rate** — cache effectiveness

### NVTX Markers

Add custom annotations to the profiler timeline:

```python
import torch.cuda.nvtx as nvtx

nvtx.range_push("forward pass")
output = model(input)
nvtx.range_pop()

nvtx.range_push("backward pass")
loss.backward()
nvtx.range_pop()
```

Ranges appear as colored bands in Nsight Systems timeline, making it easy to identify which phase of training each kernel belongs to.

---

## 4. Multi-Model Training Optimization (多模型训练优化)

### Identifying Bottlenecks

The profiling workflow for multi-model scenarios:

1. **Profile baseline** — identify where time is actually spent
2. **Check GPU utilization** — is the GPU idle waiting for CPU?
3. **Check communication** — is NCCL/NVLink saturated?
4. **Check memory** — are there excessive allocations or H2D copies?

### Common Issues and Fixes

**CPU bottleneck (GPU idle):**
- Use `torch.compile()` to reduce Python overhead
- Use CUDA Graphs to replay static computation graphs
- Increase DataLoader `num_workers`

**Memory bottleneck:**
- Use `torch.cuda.memory_summary()` to audit allocations
- Enable `gradient_checkpointing` to trade compute for memory
- Use mixed precision (`torch.autocast`)

**Communication bottleneck (multi-GPU):**
- Overlap computation with communication (`overlap_comm=True` in FSDP)
- Check NVLink topology with `nvidia-smi topo -m`

---

## 5. Key Profiling Metrics Reference

| Metric | Tool | What it means |
| --- | --- | --- |
| GPU Utilization | nvidia-smi | % of time SM has at least one active warp |
| SM Throughput | ncu | compute ops / peak compute ops |
| Memory Throughput | ncu | HBM bandwidth used / peak bandwidth |
| Occupancy | ncu | resident warps / max warps per SM |
| Warp Efficiency | ncu | active threads / (active warps × 32) |
| L2 Hit Rate | ncu | fraction of memory requests served from L2 |

---

## 6. Roofline Model (屋顶线模型)

The roofline model determines whether a kernel is compute-bound or memory-bound:

```text
arithmetic intensity = FLOPs / bytes accessed from HBM

if intensity > ridge_point:  compute-bound  → optimize compute
if intensity < ridge_point:  memory-bound   → optimize memory access
```

Ridge point = peak FLOPS / peak memory bandwidth. For A100:
```text
peak FLOPS  = 312 TFLOPS (FP16)
peak BW     = 2 TB/s
ridge point = 312e12 / 2e12 = 156 FLOPs/byte
```

Kernels below the ridge point benefit from tiling, vectorized loads, and reducing HBM traffic. Kernels above it benefit from algorithmic improvements.

---

## Related Concepts

- [[cuda-thread-hierarchy]]
- [[cuda-gemm]]
- [[cuda-cheatsheet]]
- [[5-kernel-dev-arithmetic-intensity]]
- [[warp]]

## Sources

- 11. GPU performance optimization codelab.pdf (Lesson 11, SJTU AI Infra)
