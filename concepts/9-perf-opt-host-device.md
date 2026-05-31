```yaml
title: Host/Device Optimization (Minimizing CPU-GPU Overhead)
type: concept
tags: [gpu, cpu, performance, launch-overhead, synchronization, cuda, optimization, training, inference]
created: 2026-05-31
updated: 2026-05-31
sources: [9-profile-optimization.md]
```

# Host/Device Optimization

The CPU (host) and GPU (device) run asynchronously. The CPU issues CUDA kernel launches and immediately continues; the GPU executes them on its own timeline. Performance breaks down when this pipeline stalls — either the CPU waits on the GPU or the GPU waits on the CPU.

## The Ideal Execution Model

```
CPU:  [launch A][launch B][launch C][launch D]...
GPU:            [   A   ][   B   ][   C   ][   D   ]...
```

The CPU should always be ahead of the GPU, keeping the GPU's launch queue full. The GPU should never be idle waiting for the CPU to issue the next kernel.

## CPU-GPU Synchronization Points

Any operation that forces the CPU to wait for GPU completion stalls the pipeline:

| Operation | Why it syncs |
|---|---|
| `tensor.item()` | Transfers scalar to CPU; requires GPU to finish producing it |
| `tensor.cpu()` / `.numpy()` | Transfers data to CPU synchronously |
| `torch.cuda.synchronize()` | Explicit full barrier |
| `print(tensor)` | Calls `.item()` internally |
| Logging a GPU tensor value | Same as above |
| Python `if tensor > 0:` | Forces `.item()` |

**Fix:** Defer any operation that needs a CPU-side value. For loss logging, accumulate on GPU and only sync at the end of an epoch, not every step.

```python
# Bad — syncs every step
if loss.item() < best_loss:
    save_checkpoint()

# Better — sync only occasionally
if step % 100 == 0:
    torch.cuda.synchronize()
    if loss.item() < best_loss:
        save_checkpoint()
```

## Kernel Launch Overhead

Each CUDA kernel launch has ~5–20 µs of CPU-side dispatch overhead (driver validation, argument marshaling, queue submission). A model with 1000 small operators per step accumulates 5–20 ms of pure launch overhead — significant at high throughput targets.

**Symptoms in nsys:** Many thin kernel bars with visible gaps between them; CPU row is the bottleneck, not the GPU.

**Solutions:**

1. **torch.compile:** Fuses operators, reducing the number of kernel launches. The compiled Inductor backend generates fewer, larger kernels.
2. **CUDA Graph:** Records the entire launch sequence once and replays it with a single API call. See [[9-perf-opt-torch-compile-cuda-graph]].
3. **Custom CUDA kernels:** Manually fuse small operators into one kernel. See [[1-overview-operator-fusion]].

## Non-Blocking Transfers

CPU→GPU transfers block by default. Use `non_blocking=True` with pinned memory for async transfers:

```python
# Allocate pinned memory on host
host_tensor = torch.zeros(batch_size, dim).pin_memory()

# Async transfer — returns immediately; GPU DMA runs concurrently
gpu_tensor = host_tensor.cuda(non_blocking=True)

# ... do other work on CPU ...

# GPU will use gpu_tensor only after the transfer completes
# (GPU kernel ordering guarantees this automatically within a stream)
```

The transfer runs on CUDA's copy engine concurrently with GPU compute on the default stream, as long as they are on separate streams.

## torch.cuda.synchronize() as a Debugging Tool

During debugging, `torch.cuda.synchronize()` before timing measurements gives accurate wall-clock times. In production, remove it — the synchronization itself is the bottleneck.

```python
# Correct timing
start = time.perf_counter()
torch.cuda.synchronize()  # wait for GPU before starting timer
output = model(input)
torch.cuda.synchronize()  # wait for GPU before stopping timer
end = time.perf_counter()
```

## Stream Parallelism

Use separate CUDA streams to overlap independent operations:

```python
stream_compute = torch.cuda.Stream()
stream_prefetch = torch.cuda.Stream()

with torch.cuda.stream(stream_prefetch):
    next_batch = next_batch.cuda(non_blocking=True)

with torch.cuda.stream(stream_compute):
    output = model(current_batch)

# Ensure prefetch is done before next compute step uses it
stream_compute.wait_stream(stream_prefetch)
```

## Cross-References

- [[9-perf-opt-torch-compile-cuda-graph]] — eliminating launch overhead with graph capture
- [[9-perf-opt-profiling]] — identifying sync points and launch gaps in profiles
- [[1-overview-gpu-memory-hierarchy]] — PCIe host memory bandwidth
- [[1-overview-compute-communication-overlap]] — stream-level overlap for distributed training
