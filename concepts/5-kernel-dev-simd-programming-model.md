```markdown
---
title: SIMD Programming Model
type: concept
tags: [simd, kernel-development, ai-chip, parallelism, operator-optimization]
created: 2026-04-06
updated: 2026-04-06
sources: [5-kernel-dev.md]
---

# SIMD Programming Model

SIMD (Single Instruction Multiple Data) is a parallel programming model in which a single instruction operates on multiple data elements simultaneously using wide vector registers. It is the dominant programming model for many domestic Chinese AI chips and is also realized in CPUs via extensions such as AVX-512.

## Key Properties

- **Explicit data movement:** The programmer must manually write code to move data between memory hierarchy levels (e.g., HBM → L1), specifying transfer sizes and indexing.
- **Data alignment required:** Data must be aligned to the SIMD vector width (e.g., 1024-element alignment). Under-filled vectors waste compute capacity.
- **Vector compute:** One instruction processes an entire vector of elements simultaneously (e.g., 16 floats at a time with `__m512`).
- **Ping-pong buffers required:** To hide memory latency, programmers must manually implement double buffering — compute one buffer while loading the other.
- **More complex code:** Compared to SIMT, SIMD code is more verbose and requires explicit management of data flow.

## Contrast with SIMT

| Dimension | SIMD | SIMT |
| --- | --- | --- |
| Data flow | Must write explicitly | Not required |
| Data alignment | Required | Not required |
| Compute mode | Vector (multiple data per instruction) | Scalar |
| Ping-pong buffer | Generally required | Generally not needed |
| Representative hardware | Domestic AI chips | GPU |

## AVX-512 Intrinsics (Common)

| Intrinsic | Description |
| --- | --- |
| `__m512` | 512-bit SIMD register type (holds 16 floats) |
| `_mm512_setzero_ps()` | Zero-initialize a 512-bit register |
| `_mm512_load_ps(ptr)` | Load 16 floats from aligned memory |
| `_mm512_add_ps(a, b)` | Add two 512-bit float registers |
| `_mm512_reduce_add_ps(v)` | Horizontal reduction sum of a 512-bit register |

## Optimization Techniques Specific to SIMD

- **Loop unrolling:** Manually unrolling inner loops (e.g., 2× or 4×) reduces loop overhead and increases instruction-level parallelism (ILP).
- **Multiple accumulators:** Using several independent accumulator registers breaks dependency chains and improves throughput.
- **FMA (Fused Multiply-Add):** Combines multiply and add into a single instruction, reducing instruction count.
- **Register-only reduce:** Perform final reductions in registers rather than writing back to L1, avoiding `mfence` overhead.

## Related Concepts
- [[simt-programming-model]]
- [[ping-pong-buffer]]
- [[arithmetic-intensity]]
- [[layernorm]]
- [[reduce-operator]]

## Sources
- [[5-kernel-dev]]
```

---
