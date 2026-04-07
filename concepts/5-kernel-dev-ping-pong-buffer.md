```markdown
---
title: Ping-Pong Buffer (Double Buffering)
type: concept
tags: [ping-pong-buffer, double-buffering, compute-communication-hiding, kernel-optimization]
created: 2026-04-06
updated: 2026-04-06
sources: [5-kernel-dev.md]
---

# Ping-Pong Buffer (Double Buffering)

A ping-pong buffer (also called double buffering) is an optimization technique that hides memory latency by overlapping computation with data transfer. Two alternating buffers are used: while one buffer's data is being processed by compute units, the other buffer is being filled with the next chunk of data from memory.

## Motivation

On AI hardware, the data transfer engine (DMA) and compute units are **separate hardware resources** that can operate in parallel. Without double buffering, compute units sit idle while waiting for data to arrive from HBM. With double buffering, this idle time is eliminated.

## How It Works

1. Load chunk 0 into buffer[0].
2. Begin computing buffer[0].
3. Simultaneously load chunk 1 into buffer[1].
4. When compute on buffer[0] finishes, switch: compute buffer[1] while loading chunk 2 into buffer[0].
5. Repeat, alternating between buffers.

## Example Code Pattern (CUDA-style)

```c
void compute_kernel(float *out, float *in) {
    __shared__ float buffer[2][TILE_SIZE];
    load_data(buffer[0], in);
    for(int tile=0; tile<TILES; tile++) {
        __syncthreads();
        process(buffer[tile%2]);
        if(tile < TILES-1) {
            async_load(buffer[(tile+1)%2], in + (tile+1)*TILE_SIZE);
        }
    }
}
```text

## SIMD vs. SIMT

- **SIMD:** Ping-pong buffers are generally **required** and must be implemented manually by the programmer.
- **SIMT (GPU):** Generally not needed at the kernel code level; hardware and compiler handle latency hiding, though advanced CUDA kernels may use shared memory double buffering explicitly.

## Applicability in Reduce Example

The ping-pong buffer optimization was identified as an enhancement to the SIMD reduce implementations (scalar → SIMD basic → multiple accumulators → ping-pong), overlapping loads with accumulation to maximize throughput.

## Related Concepts

- [[simd-programming-model]]
- [[simt-programming-model]]
- [[arithmetic-intensity]]

## Sources

- [[5-kernel-dev]]

```text

---
