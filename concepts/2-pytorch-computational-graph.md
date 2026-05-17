```yaml
title: Computational Graph (PyTorch)
type: concept
tags: [pytorch, computational-graph, autograd, deep-learning, backpropagation, operator-fusion, tpu, xla]
created: 2026-04-05
updated: 2026-05-17
sources: [2-pytorch.md]
```

# Computational Graph

## Definition

A **computational graph** is a directed acyclic graph (DAG) in which nodes represent
operations or values (tensors) and edges represent data flow (tensor dependencies).
PyTorch builds this graph **dynamically** (at runtime, during the forward pass) —
this is called **define-by-run** or **eager execution**, in contrast to static graph
frameworks like early TensorFlow.

## Dynamic vs. Static Graphs

| Property | PyTorch (Dynamic) | TensorFlow 1.x / JAX jit (Static) |
| --- | --- | --- |
| Graph construction | At runtime (each forward pass) | Before execution (compile step) |
| Debugging | Standard Python debugger works | Requires special tools |
| Control flow | Native Python `if`/`for` | Requires `tf.cond`, `tf.while_loop` / `jax.lax.cond` |
| Flexibility | High | Lower |
| Performance optimization | JIT via `torch.compile` | Built-in at graph level |

## Operator Behavior in Each Mode

### Dynamic graph (eager execution)

Each operator call immediately dispatches an independent GPU kernel:

```python
x = torch.relu(x)      # launches relu kernel → result written to HBM
x = x @ weight         # launches matmul kernel → reads from HBM, writes to HBM
x = torch.softmax(x)   # launches softmax kernel → reads from HBM
```

Every intermediate tensor is materialized in HBM (High-Bandwidth Memory). With N operators, there are N kernel launches and up to 2N HBM round-trips (one read + one write per operator). The Python interpreter sees each operator independently; no cross-operator optimization is possible.

### Static graph (compiled execution)

The compiler receives the entire graph before any kernel runs. It can perform **operator fusion**: merge multiple consecutive operators into a single kernel that keeps intermediate results in on-chip registers or L1 cache rather than writing them back to HBM.

```
relu → matmul → softmax   →   compiled into one fused kernel
                               (intermediates stay in registers, never touch HBM)
```

This is the mechanism behind [[1-overview-flash-attention]], which fuses the QKᵀ matmul, softmax, and AV matmul into one kernel, avoiding materializing the full N×N attention matrix in HBM. See [[1-overview-operator-fusion]] for the general principle.

### Bridging the gap: `torch.compile`

PyTorch 2.0 introduced `torch.compile` (backed by TorchInductor) to capture a static graph from eager code and apply compiler-level fusion:

1. FX graph capture traces a Python execution and extracts a static operator graph.
2. TorchInductor generates fused Triton or C++ kernels.
3. Subsequent calls reuse the compiled artifact.

The typical production pattern: **develop with eager (dynamic), deploy with `torch.compile` (static)**.

## TPU and XLA: Enforced Static Graph

TPUs use JAX + XLA (Accelerated Linear Algebra) as their software stack and operate exclusively in static graph mode:

- `@jax.jit` (just-in-time compilation) is required for any non-trivial computation; without it the code runs in a slow interpreter mode.
- XLA compiles the full graph from an HLO (High-Level Operations) intermediate representation, applies fusion and memory layout optimization, then emits native TPU instructions via LLVM.
- Control flow that depends on runtime tensor values (e.g., data-dependent `if`) cannot be represented in the static graph and must be replaced with `jax.lax.cond` / `jax.lax.scan`.

Consequence: TPU code must be written in a "compiler-friendly" style from the start — Python dynamism is a compilation error, not just a performance cost. This is the primary ergonomic difference from writing GPU code with PyTorch.

| | GPU + PyTorch | TPU + JAX |
|---|---|---|
| Default mode | Dynamic (eager) | Static (`jax.jit` required) |
| Operator fusion | Opt-in (`torch.compile`) | Always on (XLA fuses automatically) |
| First run | Fast (no compile) | Slow (XLA compilation) |
| Repeated runs | Slower (per-kernel overhead) | Faster (fused kernel reuse) |
| Dynamic control flow | Native Python | `jax.lax.cond` / `jax.lax.scan` |

## Structure

- **Leaf nodes:** Tensors created by the user (e.g., model parameters, input data).
  If `requires_grad=True`, they accumulate gradients in `.grad`.
- **Interior nodes:** Tensors produced by operations. They hold a `grad_fn` reference
  to the operation that created them.
- **Root node:** Typically the scalar loss value from which `.backward()` is called.

## Execution

```text
Input → Linear → ReLU → Linear → Softmax → CrossEntropyLoss → scalar loss
                                                                     ↓
                                                              .backward()
                                                    (gradients flow right-to-left)
```

## Related Concepts

- [[pytorch-autograd]]
- [[pytorch-tensor]]
- [[pytorch-nn-module]]
- [[1-overview-operator-fusion]]
- [[1-overview-flash-attention]]
- [[1-overview-hbm-high-bandwidth-memory]]

## Sources

- [[2-pytorch]]
