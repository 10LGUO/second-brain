AI Infra pillars: compute, memory, communication, precision, performance

Memory: HBM (high bandwidth, low latency), HBM3e (higher bandwidth), GDDR (lower bandwidth, higher latency)

---

## Attention

**Q/K decide where to look; V decides what to retrieve**
scores = QK^T are just weights, not the output. The actual information content lives in V. softmax(scores) @ V is the step that converts "how much attention" into "what blend of content". Without V, you know where to look but have nothing to take from there.

**The scale 1/sqrt(head_dim) is about variance control, not specific to Q**
The dot product sums head_dim terms. With unit-variance inputs, the sum has variance head_dim. Dividing by sqrt(head_dim) normalizes it back to 1, preventing softmax from becoming one-hot and causing vanishing gradients. Using Q.shape[-1] is just a convenient way to read head_dim — K and V have the same head_dim.

**Multiple heads: the same token attends in multiple ways simultaneously**
Not a division of labor across tokens — a single token's representation needs to pull different types of information from different positions at the same time. One head produces one weighted average; multiple heads let the model retain multiple "perspectives" for the same position.

**Linear projection vs splitting the embedding**
Splitting is a hard partition — each head sees a fixed slice of dimensions. Projection lets each head learn a linear combination of all d_model dimensions. W_Q/W_K/W_V are themselves the carrier of "what pattern each head is learning to look for."

**Why prefill parallelizes but decode does not**
Prefill: the full prompt is known upfront, all tokens can compute QK^T simultaneously → large matmul → compute-bound.
Decode: autoregressive dependency means step t's output is step t+1's input, steps cannot overlap → one vector×matrix per step → memory-bound (compute sits idle, bottleneck is KV cache read bandwidth).

## Paged Attention

**A page holds page_size tokens, not one token**
Analogous to OS virtual memory: a page holds many bytes, not one. page_size is a management granularity tradeoff — smaller pages reduce internal fragmentation but increase page table overhead.

**The gather step is a correctness requirement, not an optimization**
The page pool is shared physical memory across all sequences. Computing attention directly over the raw pool without gathering would mix tokens from different sequences — the result would be completely wrong, not just inefficient.

**Paged attention is decode-only**
During prefill the full prompt can be stored contiguously; non-contiguous page management only has value during decode, where one token is appended per step.

## Speculative Decoding

**The real bottleneck is HBM reads per token, not arithmetic dependency**
The autoregressive dependency (step t+1 needs token t's embedding to construct Q) is a mathematical constraint, but it's not the core problem — once a model iteration finishes, the next token is known and Q can be constructed immediately. The real bottleneck is that KV cache lives in HBM (too large for SRAM: a single sequence at 1K tokens already exceeds total SRAM on an A100), so every decode step pays the full HBM bandwidth cost regardless. Speculative decoding improves the ratio of tokens produced per HBM read: naive decode is 1:1, speculative decode approaches λ:1 by using draft tokens to enable parallel prefill-style verification.

**Why you can't hold KV cache in SRAM across steps**
Not an engineering choice — KV cache for a single sequence (e.g. Qwen3-0.6B, seq_len=1024) is ~58MB, exceeding total SRAM across all SMs on an A100 (~40MB). KV cache must live in HBM by physical necessity.

**How draft tokens are verified in parallel**

Target model takes the full draft sequence as a prefill input:
```
[context | draft_1 | draft_2 | ... | draft_λ]
```
Causal mask encodes the dependency — position t+i can only attend to context + draft_1..draft_{i-1}. All positions are computed in one forward pass in parallel; the mask handles the sequential dependency at the matrix level without needing serial execution.

Accept/reject condition: `random() < P_target(x) / P_draft(x)`
Derived from rejection sampling. If P_target > P_draft, target is more confident than draft → high accept rate. If P_target < P_draft, draft was overconfident → reject with proportional probability. Mathematically guarantees the accepted distribution is exactly P_target. `random()` returns a uniform sample from [0, 1) — the ratio directly becomes the acceptance probability:
```
ratio = 1.0  → 100% accept (target and draft equally confident)
ratio = 0.8  → 80% accept
ratio > 1.0  → clipped to 1.0, always accept (target more confident than draft)
```
When draft and target distributions are identical, all tokens are accepted and speedup approaches λ.

If draft_i is rejected, draft_{i+1} and beyond are thrown away regardless of their computed values — they were based on a wrong prefix. Target model resamples at the rejection point. The wasted compute on discarded positions is acceptable because prefill is cheap (high parallelism) and the saved HBM reads from accepted tokens outweigh the waste. Higher draft acceptance rate → fewer discarded positions → better efficiency.

**The two verify approaches**
Method 1 `[B, H, λ, t+λ]`: processes all draft tokens as a prefill with causal mask — position i attends to original context plus draft tokens 1..i-1. Strictly equivalent to target model's distribution.
Method 2 `[B×λ, t+1, D]`: treats each draft token as an independent batch item attending only to original context + itself. Under sequential acceptance, the conditioning prefix is actually the same as method 1 — the difference shows up in reject/truncation handling, not in the attention computation itself.

## CUDA Graph

**CUDA Graph: what it actually is and why it has constraints**
CUDA Graph pre-records a sequence of kernel dispatches into a static DAG. The benefit is eliminating dispatch bubbles between small kernels — for large kernels, dispatch overhead is negligible relative to compute time so the gain doesn't matter. The cost is that all decisions (kernel order, input memory addresses) must be statically determined at record time.

The kernel execution order inside a graph is actually fully determined by the DAG edges — it's not "unknown." The real reason CPU-side sync is forbidden is that the CPU has already exited the scheduling loop after issuing a single "replay this graph" command. In eager mode the CPU acts as the conductor, inserting sync points and reading return values between kernel launches. During graph replay there is no CPU in the loop to receive values or make decisions — so anything requiring runtime CPU involvement (sync, D2H reads, dynamic shapes, control flow) cannot be expressed in the graph.

Workaround for control flow: capture multiple graphs (one per branch), CPU selects which to replay at runtime. This is how vLLM/NanoVLLM handles variable batch sizes — capture graphs for a few fixed batch sizes, pad inputs to the nearest size, replay the matching graph.

**Debugging CUDA Graph captured flows (no print available)**
1. Dump the kernel list — verify the expected kernels were recorded and executed in the right order (use `nsys` or `cuda graph debug dump`)
2. Check for illegal memory access — HBM addresses are fixed at capture time; if the buffer was freed or reallocated between capture and replay, you get silent corruption or illegal access. Run with `compute-sanitizer`
3. Dump values to a pre-allocated inspection buffer — allocate a fixed HBM buffer before capture, write intermediate values into it inside the graph, then D2H copy after replay to inspect

## GPU Execution Model

**Mental model: Stream / CPU / GPU / SM / Kernel / Block relationships**

```
CPU cores     — orchestrators: execute Python, launch kernels into streams (async)
GPUs          — workers: execute kernels
Streams       — async command queues belonging to a specific GPU; ops within one stream are serial, ops across streams on the same GPU can be concurrent
SMs           — execution units inside a GPU (A100: 108 SMs); a block runs on one SM, never split across SMs
Blocks        — unit of work dispatched to an SM; an SM can hold multiple blocks if resources allow
Kernels       — a grid of blocks launched together; occupies as many SMs as blocks can be distributed across
```

**How many SMs a kernel occupies**
Determined by launch config (`grid_dim` = number of blocks) and per-block resource usage (shared memory, registers, thread count). If a block uses 82KB of 164KB SM shared memory, one SM can hold at most 2 blocks. Large kernels with high occupancy crowd out other streams; small kernels or low-occupancy kernels leave room for concurrent streams.

**Multi-stream concurrency on one GPU**
Multiple streams can run concurrently on the same GPU only if SM resources are not fully claimed. The most reliable use case: compute stream (uses SMs) + H2D transfer stream (uses DMA engine) — different hardware units, true parallelism with no resource conflict.

**CPU launch is async**
CPU puts a kernel launch command into the stream queue and immediately continues. GPU executes asynchronously. This is why CUDA Graph helps small kernels — many small launches accumulate CPU dispatch overhead; replaying a graph issues them all in one CPU call.

**Hardware constraints when designing kernel shapes**

Each SM has fixed resources — exceeding any one limit reduces the number of concurrent blocks (occupancy drops):

| Resource | A100 per SM limit | Impact if exceeded |
|---|---|---|
| Threads | 2048 | Fewer concurrent blocks per SM |
| Registers | 65536 (32-bit) | Spill to local memory (HBM) → slow |
| Shared memory | 164KB (configurable) | Fewer concurrent blocks per SM |
| Blocks | 32 | Hard cap regardless of other resources |

**Register spill**: if a thread uses more registers than available (65536 / num_threads), the excess spills to local memory in HBM — same latency as global memory, kills performance silently.

**Shared memory vs L1 cache tradeoff**: on A100, shared memory and L1 cache share the same 192KB physical pool. Allocating more shared memory (up to 164KB) leaves less for L1 cache — useful when access pattern is regular and manually managed, harmful when irregular access benefits more from L1.

**HBM bandwidth as the outer constraint**: even with perfect occupancy, memory-bound kernels are capped by HBM bandwidth (~2TB/s on A100). The roofline model determines whether a kernel is compute-bound (FLOPs/byte > ridge point) or memory-bound — kernel shape tuning only helps if you're on the compute-bound side.

**Warp size is fixed at 32**: block_dim should be a multiple of 32, otherwise the last warp has idle threads. Common choices: 128, 256, 512 threads per block.

**Example: typical GEMM kernel shape (A100)**

```
Problem: C = A @ B,  A: [M=4096, K=4096],  B: [K=4096, N=4096]

Block tile:     [128, 128]  (each block computes a 128×128 output tile)
Thread tile:    [8, 8]      (each thread computes an 8×8 output sub-tile)
Threads/block:  (128/8) × (128/8) = 16 × 16 = 256 threads = 8 warps

Grid:           (4096/128) × (4096/128) = 32 × 32 = 1024 blocks
SMs used:       min(1024, 108) = 108 SMs, each holding ~9 blocks

Shared memory per block:
  A tile: 128 × 32 × 2 bytes (BF16) = 8KB
  B tile: 32  × 128 × 2 bytes       = 8KB
  Total:  16KB  →  164KB / 16KB = 10 blocks per SM (shared memory not the bottleneck)

Register per thread:
  8×8 output tile = 64 accumulators (FP32) = 64 registers
  + pointers, loop vars ≈ 100 registers total
  65536 / 256 threads = 256 registers available per thread  →  no spill
```

A tile and B tile are chunks of A and B loaded into shared memory for one K-loop iteration. Each block accumulates `C += A_tile @ B_tile` over K/32 = 128 iterations, reusing each loaded tile across all 256 threads before fetching the next slice from HBM.

**Block-SM relationship and thread limits**

One block is always assigned to exactly one SM — a block never splits across SMs. But one SM can hold multiple blocks concurrently (if resources allow):

```
Block → SM:  one-to-one (a block lives on exactly one SM)
SM → Blocks: one-to-many (an SM can run multiple blocks concurrently)
```

Thread limits stack as hard constraints enforced at launch time (A100):

```
threads per block ≤ 1024    (block-level hard limit, CUDA error if exceeded)
threads per SM    ≤ 2048    (SM-level hard limit)
```

The block limit (1024) is stricter than the SM limit (2048) by design — forces at least 2 blocks per SM, giving the SM enough warps to hide memory latency through warp switching.

To scale beyond 1024 threads, use more blocks (larger grid), not larger blocks:

```python
threads_per_block = 256
grid_size = (N + threads_per_block - 1) // threads_per_block
kernel<<<grid_size, threads_per_block>>>()
```

**Block is a software abstraction; the hardware only knows warps and SMs**

Block (and grid) are software concepts defined in kernel launch config. The hardware execution pipeline is:

```
GigaThread Engine (GPU scheduler) → distributes blocks to SMs
SM → breaks each block into warps of 32 threads → executes warps
```

The SM has no concept of "block" during execution — it only sees warps. Block is an abstraction for: grouping threads that share the same shared memory, partitioning work across SMs (grid_dim = block count), and scoping `__syncthreads()` synchronization. Below the block level, hardware takes over.

## Memory and Compute

**In PyTorch eager mode, every tensor lives in HBM**
Every torch op materializes its output as an HBM tensor. Intermediate results always round-trip through HBM. "No HBM write" only holds inside a fused CUDA kernel (Flash Attention or hand-written) — torch.compile cannot fuse across matmul boundaries.

**torch.compile and CUDA Graphs solve different layers of overhead**
- torch.compile: reduces HBM round-trips by fusing elementwise ops into adjacent kernels (compute side)
- CUDA Graphs: reduces CPU kernel launch overhead by recording and replaying a fixed launch sequence (dispatch side)
They are complementary but neither replaces Flash Attention's avoidance of materializing the score matrix in HBM.

**Where block attention (Q-tiled only) actually saves over naive**
The only real saving: the score slice does not need to round-trip through HBM between softmax and @ V — both can happen within one kernel pass. However K/V are still fully reloaded from HBM on every block iteration, making total HBM reads worse than naive. In raw PyTorch even this saving disappears since each op is a separate kernel.