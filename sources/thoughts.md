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

## Memory and Compute

**In PyTorch eager mode, every tensor lives in HBM**
Every torch op materializes its output as an HBM tensor. Intermediate results always round-trip through HBM. "No HBM write" only holds inside a fused CUDA kernel (Flash Attention or hand-written) — torch.compile cannot fuse across matmul boundaries.

**torch.compile and CUDA Graphs solve different layers of overhead**
- torch.compile: reduces HBM round-trips by fusing elementwise ops into adjacent kernels (compute side)
- CUDA Graphs: reduces CPU kernel launch overhead by recording and replaying a fixed launch sequence (dispatch side)
They are complementary but neither replaces Flash Attention's avoidance of materializing the score matrix in HBM.

**Where block attention (Q-tiled only) actually saves over naive**
The only real saving: the score slice does not need to round-trip through HBM between softmax and @ V — both can happen within one kernel pass. However K/V are still fully reloaded from HBM on every block iteration, making total HBM reads worse than naive. In raw PyTorch even this saving disappears since each op is a separate kernel.