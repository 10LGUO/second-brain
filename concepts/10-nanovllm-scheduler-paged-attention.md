```yaml
title: NanoVLLM — Scheduler and Paged Attention
type: concept
tags: [inference, scheduler, paged-attention, kv-cache, prefix-cache, preemption, continuous-batching, nanovllm]
created: 2026-06-08
updated: 2026-06-08
sources: [10-nanovllm.md]
```

# NanoVLLM — Scheduler and Paged Attention

NanoVLLM implements a scheduler and block manager that together handle KV cache memory management, continuous batching, and prefix caching for LLM inference.

## Sequence

A `Sequence` represents one request from arrival to completion. Key fields:

| Field | Meaning |
|---|---|
| `token_ids` | Full token sequence — prompt + generated tokens so far |
| `num_prompt_tokens` | Length of the original prompt, fixed |
| `num_cached_tokens` | Tokens whose KV is already in cache (from prefix cache or prior steps) |
| `num_scheduled_tokens` | Tokens to process in the current batch step |
| `block_table` | List of physical KV cache block IDs assigned to this sequence |
| `is_prefill` | Whether the sequence is currently in the prefill phase |

State machine: `WAITING → RUNNING → FINISHED`. Preemption sends a sequence back to `WAITING` with `block_table` cleared.

`__getstate__` serializes differently for prefill vs decode: prefill sends full `token_ids` (model needs the whole prompt), decode sends only `last_token` (only one token is fed per step), minimizing inter-process communication in tensor parallel setups.

## Paged Attention

KV cache is managed as fixed-size **blocks** (default 256 tokens each). One global KV cache tensor is allocated at startup:

```
kv_cache: (2, num_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
```

Each sequence has a `block_table` — a CPU-side list of physical block IDs pointing into this shared tensor. Blocks are allocated/freed at block granularity, eliminating fragmentation.

**Block states:**
- **used**: currently occupied by a sequence (`ref_count > 0`)
- **free**: not currently occupied, but may still contain valid KV data if hash is intact
- **complete**: a full block whose tokens won't change — eligible for prefix cache hashing
- **tail block**: the last block of a sequence, still being written — not hashed

## Prefix Cache

Complete blocks are hashed using a chained scheme: each block's hash includes the previous block's hash as a prefix. This means a hash uniquely identifies the entire token history up to that block, not just the block itself.

`can_allocate` checks for cache hits before allocating:
1. Computes hash of each complete block (skipping the last, partial block)
2. Looks up in `hash_to_block_id`
3. Verifies token content to guard against hash collisions
4. Breaks on first miss — only **contiguous prefix hits** count

Returns `num_cached_blocks`: the number of leading blocks whose KV can be reused. The scheduler uses this to skip those tokens during prefill (`num_tokens = seq.num_tokens - num_cached_blocks * block_size`).

`hash_blocks` (called in `postprocess`) registers newly completed blocks into the hash map after each forward pass — this is the only place hashing happens, ensuring blocks are only registered once they're fully written.

## Scheduler

Two phases per `schedule()` call, tried in order:

### Prefill (WAITING → RUNNING)

Processes sequences from `waiting` queue. For each sequence:
1. `can_allocate` — checks prefix cache hits and free block availability; returns `-1` if OOM
2. If OOM, `break` (not `continue`) — waiting queue is FIFO, later sequences are unlikely to fit either
3. Compute `num_tokens` using `num_cached_blocks` to skip cached prefix
4. Chunked prefill: only the first sequence in a batch can be chunked across multiple steps
5. `allocate` — pins the cached blocks (incrementing `ref_count`) and allocates fresh blocks for the remainder

Returns immediately if any prefill sequences were scheduled (prefill and decode never mix in one batch).

### Decode (append one token per sequence)

For each running sequence, checks if a new KV slot is available (`can_append`). If not, preempts to free space:

- **Other sequences exist in running**: preempt the lowest-priority one (from the tail of `running`)
- **Only current sequence left**: self-preempt — release this sequence's KV, push it back to `waiting`, and call `self.schedule()` recursively. The freed blocks allow prefill to proceed immediately.

`can_append` returns `len(free_block_ids) >= (len(seq) % block_size == 1)` — a new block is needed only when the next token starts a new block (Python bool coerces to 0/1 for the comparison).

`may_append` allocates a new block only when needed (first slot of a new block). The tail block is never hashed until it's complete.

## Preemption

`preempt(seq)` releases all of a sequence's KV blocks (`deallocate`) and returns it to the front of `waiting`. There is no swap to CPU — blocks are simply freed. If prefix cache entries survive (not evicted by subsequent allocations), the sequence can recover cached blocks on re-prefill without recomputation.

## Model Runner Integration

Before each forward pass, `prepare_prefill` builds:
- `input_ids` / `positions`: only the scheduled (non-cached) tokens
- `cu_seqlens_q`: cumulative query lengths (scheduled tokens only)
- `cu_seqlens_k`: cumulative key lengths (cached + scheduled tokens)
- `slot_mapping`: physical HBM slot for each scheduled token's KV write
- `block_tables` (only when `cu_seqlens_k > cu_seqlens_q`, i.e. prefix cache hit): padded 2D tensor of block IDs sent to the attention kernel so it can read historical KV

The attention kernel uses `block_tables` to look up `block_tables[seq_idx, token_pos // block_size]` and find the physical block to read from.

For decode, CUDA Graphs are used to replay a pre-captured kernel sequence, avoiding kernel launch overhead. Prefill always runs eagerly since its input shape varies.

## Cross-References

- [[10-nanovllm-prefill-decode]] — prefill vs decode compute characteristics
- [[10-nanovllm-model-loading]] — model architecture and tensor parallel setup
- [[1-overview-kv-cache]] — KV cache fundamentals
- [[8-memory-opt-flash-attention]] — attention kernel that consumes block_tables
- [[9-perf-opt-torch-compile-cuda-graph]] — CUDA Graph for decode acceleration
