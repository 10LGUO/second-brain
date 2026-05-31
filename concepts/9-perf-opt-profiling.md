```yaml
title: GPU Performance Profiling Workflow
type: concept
tags: [profiling, performance, nsys, nsight, pytorch-profiler, gpu, optimization, debugging]
created: 2026-05-31
updated: 2026-05-31
sources: [9-profile-optimization.md]
```

# GPU Performance Profiling Workflow

Profiling is the mandatory first step before any performance optimization. Optimizing without profiling is guesswork; profiling reveals where time is actually spent.

## The Profiling Loop

```
Profile → Identify bottleneck → Optimize that one thing → Re-profile
```

Never skip re-profiling after a change — the bottleneck shifts as you fix things.

## Tool Selection

| Tool | Granularity | Best for |
|---|---|---|
| `nsys profile` (Nsight Systems) | System-level timeline | Finding idle GPU time, sync points, transfer overlap, stream usage |
| `ncu` (Nsight Compute) | Per-kernel | Memory bandwidth utilization, occupancy, roofline position |
| PyTorch Profiler | Python operator → CUDA kernel | Attributing GPU time to Python-level ops |
| `py-spy` / `cProfile` | Python CPU only | Finding CPU bottlenecks outside GPU dispatch |

Start coarse (nsys), then zoom in (ncu) only after identifying which kernel to optimize.

## Nsight Systems (nsys)

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=profile_run \
  python train.py

nsys-ui profile_run.nsys-rep   # open GUI
```

The timeline view shows:
- **CPU rows:** Python execution, kernel launch calls
- **GPU rows:** Kernel execution per stream, memory copies
- **Gap analysis:** Idle GPU time indicates sync points or insufficient overlap

Key patterns to look for:
- CPU row always ahead of GPU row → healthy pipeline
- GPU row stalls waiting for CPU → launch overhead or sync point
- Long transfers on PCIe → data loading bottleneck or unnecessary H↔D copies
- Single stream with no overlap → opportunity to add stream-level parallelism

## PyTorch Profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        with record_function("forward"):
            out = model(batch)
        with record_function("backward"):
            loss.backward()
        prof.step()

# Table output
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

`record_function` labels regions in the trace. `with_stack=True` attributes kernel time to Python call sites.

**Memory timeline:**
```python
with profile(profile_memory=True) as prof:
    model(x)

print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
```

## NVTX Annotations

Insert NVTX ranges in code to label regions in the nsys timeline:

```python
import torch.cuda.nvtx as nvtx

nvtx.range_push("attention_layer")
out = attention(q, k, v)
nvtx.range_pop()
```

Or use the `@nvtx.annotate` decorator. These annotations appear as labeled bands in the nsys timeline, making it easy to correlate Python-level operations with GPU kernel activity.

## Reading the Roofline

After identifying the hot kernel with nsys, use `ncu` to place it on the roofline:

```bash
ncu --set full --target-processes all -o kernel_profile python script.py
```

Key metrics:
- **SM active cycles / total cycles** → occupancy
- **DRAM bandwidth achieved / peak** → memory bandwidth utilization
- **FLOP/s achieved / peak** → compute utilization
- **Arithmetic intensity** (FLOPs / bytes) → where the kernel sits on the roofline

If arithmetic intensity < roofline knee: kernel is **memory-bound** → optimize data layout, increase reuse, fuse operators.
If arithmetic intensity ≥ roofline knee: kernel is **compute-bound** → maximize Tensor Core utilization, reduce wasted FLOPs.

See [[kernel-dev-roofline-model]].

## Common Bottleneck Patterns

| Observation | Root cause | Fix |
|---|---|---|
| GPU idle between kernels | CPU launch overhead | torch.compile, CUDA Graph |
| GPU idle at step boundaries | CPU-GPU sync (`.item()`, `.numpy()`) | Remove sync; use async ops |
| Large PCIe transfers | Data on CPU, model on GPU; or explicit `.cpu()` calls | Keep data on GPU; use pinned memory + non_blocking |
| One stream, no overlap | No comm-compute overlap | Add streams; async collectives |
| Single kernel dominates | Memory-bound kernel | Fuse with adjacent ops; improve coalescing |
| Many tiny kernels | No fusion | torch.compile or manual kernel fusion |

## Cross-References

- [[9-perf-opt-host-device]] — eliminating sync points and launch overhead
- [[9-perf-opt-torch-compile-cuda-graph]] — eliminating launch overhead at scale
- [[kernel-dev-roofline-model]] — interpreting ncu output
- [[1-overview-compute-communication-overlap]] — profiling distributed overlap
