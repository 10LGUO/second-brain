```yaml
title: Computational Graph (PyTorch)
type: concept
tags: [pytorch, computational-graph, autograd, deep-learning, backpropagation]
created: 2026-04-05
updated: 2026-04-05
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

| Property | PyTorch (Dynamic) | TensorFlow 1.x (Static) |
| --- | --- | --- |
| Graph construction | At runtime (each forward pass) | Before execution (compile step) |
| Debugging | Standard Python debugger works | Requires special tools |
| Control flow | Native Python `if`/`for` | Requires `tf.cond`, `tf.while_loop` |
| Flexibility | High | Lower |
| Performance optimization | JIT via `torch.jit` | Built-in at graph level |

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

```text

## torch.jit and TorchScript

For production deployment, PyTorch offers `torch.jit.script` and `torch.jit.trace`
to convert dynamic graphs into static, optimizable representations (TorchScript).

## Related Concepts

- [[autograd]]
- [[pytorch-tensor]]
- [[pytorch-nn-module]]

## Sources

- [[2-pytorch.md]]
```

---
