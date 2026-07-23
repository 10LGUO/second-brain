```yaml
title: PyTorch Dispatcher and Eager Execution
type: concept
tags: [pytorch, dispatcher, dispatch-key, custom-op, cuda, async, cuda-graph, sync, tensor-core, kv-cache]
created: 2026-07-22
updated: 2026-07-22
sources: [kv_cache_quantization_example.md]
```

# PyTorch Dispatcher and Eager Execution

In eager PyTorch, every operator call is independently routed to a registered kernel based on the properties of its input tensors. The **dispatcher** is that routing table. Understanding it explains three practical things: how a fused custom kernel (e.g. paged attention) gets plugged in, when the CPU stalls waiting on the GPU, and why some implementations can be captured into a CUDA graph while others cannot.

## Dispatch Keys — what determines routing

When you call `a + b`, the same line of code runs a different kernel depending on where the tensors live:

```python
a = torch.tensor([1., 2.]); b = torch.tensor([3., 4.])
a + b          # dispatch key = CPU  → CPU add kernel
a = a.cuda(); b = b.cuda()
a + b          # dispatch key = CUDA → CUDA add kernel
```

`add` is an abstract operator; `CPU`/`CUDA` are keys; the dispatcher looks up `(op, key)` in a registry and calls the matching kernel. Keys come from two sources:

1. **From the tensors** (their `DispatchKeySet`):
   - **device → backend key** (`CPU`, `CUDA`, `XPU`…) — the main "where it runs" key
   - **layout key** (`Dense`/strided, `Sparse`, `MkldnnCPU`)
2. **From thread-local context** (functionality keys, processed before the backend):
   - **Autograd** — whether grad mode is on (insert backward-graph recording)
   - **Autocast** — whether inside `torch.autocast` (auto dtype casting)

The final key set is the union across all input tensors plus context; the dispatcher processes keys in priority order (autograd → autocast → … → backend).

**`dtype` is NOT a dispatch key.** The dispatcher picks the CUDA kernel by device; dtype is resolved *inside* that kernel via an `AT_DISPATCH_FLOATING_TYPES(scalar_type, ...)` switch that instantiates the bf16/fp32 template. So: **device decides which backend kernel; dtype switches inside it.**

## Custom / fused operators — how they plug in

A fused kernel (whole attention in one launch) uses the same dispatcher — you just register it into the table first. Two steps: declare the schema, register an implementation per key.

```cpp
// declare the op exists + its signature (device/dtype-agnostic)
TORCH_LIBRARY(mylib, m) {
  m.def("fused_attn(Tensor q, Tensor k, Tensor v) -> Tensor");
}
// register the CUDA implementation
TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
  m.impl("fused_attn", &fused_attn_cuda_kernel);
}
```

```python
out = torch.ops.mylib.fused_attn(q, k, v)   # q.device=CUDA → fused_attn_cuda_kernel
```

`m.def` adds "this op exists"; `m.impl(op, KEY, fn)` binds a kernel to a key. One op can have several impls: `CUDA`, `CPU`, and crucially `Meta` (compute output shape only, no data — for `torch.compile` shape inference) and `Autograd` (backward). vLLM registers this way: the cache-write op is `torch.ops._C_cache_ops.reshape_and_cache_flash` (a `TORCH_LIBRARY` C++ extension), and the whole attention is wrapped as `unified_attention_with_output` via `direct_register_custom_op`, with a companion `_fake`/Meta version for shape inference — so `torch.compile` treats it as one opaque graph node. This is the path a hand-written INT8 attention kernel takes to replace a pure-PyTorch reference.

## Async execution and synchronization points

CUDA kernel launches are **async by default** — the CPU enqueues the kernel into a CUDA stream and continues immediately; it does not wait for the GPU. A long chain of tensor ops is queued rapidly while the GPU works through it behind the CPU. You do not "trigger" async; it is the default. What you can explicitly control is concurrency (multiple `torch.cuda.Stream`), async H2D/D2H copies (`.to(device, non_blocking=True)` with pinned memory), replay (CUDA graphs), and forced waiting (`torch.cuda.synchronize()`).

The CPU only stalls at a **sync point** — and the distinction between reading a tensor's *value* vs its *metadata* is key:

| Reading… | Syncs? | Examples |
|---|---|---|
| **value** (must wait for GPU to compute) | **yes** | `.item()`, `.tolist()`, `.cpu()`, `.numpy()`, `if tensor:` |
| **metadata** (known on CPU immediately) | **no** | `.shape`, `.size()`, `.dtype`, `.device`, `.stride()` |

Shape/dtype/device are CPU-side bookkeeping tracked symbolically — free to read, no GPU involvement. Only pulling actual numbers back forces the CPU to block until the GPU catches up.

## Accumulation width — where hardware protects precision

A GPU-specific precision fact: tensor cores use **low-precision inputs with high-precision accumulation**. `bf16 × bf16 → fp32 accumulate`; `int8 × int8 → int32 accumulate`. So bf16 storage does not wreck a matmul — the running dot-product sum is kept in fp32 registers and only cast back at the end. This is part of why low-precision storage/compute is safe to "risk" inside a kernel. (See [[1-overview-precision-convergence]] for the fp32-accumulator requirement in reductions, and bf16's equal-exponent / lower-mantissa tradeoff vs fp16.)

## CUDA graph capture and its constraints

A CUDA graph records the exact sequence of kernel launches with their exact parameters (grid/block, pointers, sizes) into a static graph, then replays it — amortizing per-launch overhead, which dominates decode with many tiny kernels. Capture requires: a **fixed** kernel sequence and launch params, **no CPU sync** during capture, **stable addresses**, and **no CPU-side control flow that depends on GPU results**.

A pure-PyTorch paged-attention reference that loops over sequences in Python violates all of these (concretely, from the qwen INT8 project's `pytorch_paged_ref.py`):

- `.tolist()` on `cu_seqlens_q`/`seqused_k` is a **sync** and pulls lengths to CPU
- the Python `for seq in ...` loop has a **data-dependent trip count** (num_seqs) and per-iteration work driven by `q_len`/`kv_len` **values**
- slices like `key_cache[blocks].reshape(-1,...)[:kv_len]` and `torch.arange(kv_len)` have **data-dependent shapes**

Hence such a reference must run with `enforce_eager=True` (CUDA graphs disabled). By contrast, the production fused varlen kernel **is** capturable: it takes `block_table` and `seqused_k` as **GPU tensor arguments**, handles variable lengths with a GPU-side loop **inside one kernel**, never reads them to CPU, and pads batches to fixed capture sizes. The lesson for kernel work: **variable length must be consumed inside the kernel, not unrolled by a host-side Python loop.**

## Related Concepts

- [[gpu-execution-model]] — streams, SM/warp/block hierarchy, the hardware the dispatcher targets
- [[2-pytorch-computational-graph]] — eager vs graph, how ops chain
- [[9-perf-opt-torch-compile-cuda-graph]] — CUDA Graph and torch.compile for launch-overhead reduction
- [[1-overview-precision-convergence]] — fp32 accumulation, bf16 vs fp16 range/precision tradeoff
- [[1-overview-compute-communication-overlap]] — explicit multi-stream concurrency
- [[kv-cache-int8-quantization]] — the project this was grounded in (INT8 paged attention kernel path)
- [[pytorch-tensor]] — tensor metadata (device/dtype/shape) and views

## Sources

- [[kv_cache_quantization_example]] — hands-on grounding via the qwen INT8 KV cache project
```
