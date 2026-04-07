```yaml
title: PyTorch Tensor
type: concept
tags: [pytorch, tensor, deep-learning, linear-algebra]
created: 2026-04-05
updated: 2026-04-05
sources: [2-pytorch.md]
```

# PyTorch Tensor

## Definition

A **tensor** in PyTorch is the fundamental n-dimensional array data structure used for all
numerical computation in the framework. Tensors are analogous to NumPy `ndarray` objects
but with additional capabilities: they can be moved to GPU memory for accelerated computation
and they participate in PyTorch's automatic differentiation system ([[autograd]]).

## Key Properties

- **dtype:** Data type of elements (e.g., `torch.float32`, `torch.int64`, `torch.bool`)
- **shape:** Dimensions of the tensor, accessible via `.shape` or `.size()`
- **device:** Where the tensor lives — `cpu` or `cuda:N` for GPU N
- **requires_grad:** Boolean flag; if `True`, operations on this tensor are tracked for
  automatic differentiation
- **Immutable metadata, mutable data:** Shape is fixed after creation; values can be
  modified in-place (operations suffixed with `_`, e.g., `add_()`)

## Creation Methods

| Method | Description |
| --- | --- |
| `torch.tensor(data)` | Create from Python list/NumPy array |
| `torch.zeros(shape)` | All-zero tensor |
| `torch.ones(shape)` | All-one tensor |
| `torch.rand(shape)` | Uniform random in [0, 1) |
| `torch.randn(shape)` | Standard normal random |
| `torch.arange(start, end, step)` | 1D range tensor |
| `torch.empty(shape)` | Uninitialized tensor |

## Key Operations

- **Arithmetic:** `+`, `-`, `*`, `/`, `torch.matmul()` / `@`, `torch.dot()`
- **Reshaping:** `.view()`, `.reshape()`, `.squeeze()`, `.unsqueeze()`, `.permute()`
- **Indexing:** Standard Python/NumPy-style slicing; advanced indexing with boolean masks
- **Reduction:** `.sum()`, `.mean()`, `.max()`, `.min()`, `.argmax()`
- **Moving devices:** `.to(device)`, `.cuda()`, `.cpu()`
- **NumPy bridge:** `.numpy()` (CPU tensor → ndarray), `torch.from_numpy()`

## Relationship to Autograd

When `requires_grad=True`, every operation producing a new tensor records a
`grad_fn` (gradient function). Calling `.backward()` on a scalar output tensor
traverses this computation graph and accumulates gradients in `.grad` attributes
of leaf tensors. See [[autograd]].

## Related Concepts

- [[autograd]]
- [[neural-network]]
- [[pytorch-nn-module]]
- [[computational-graph]]

## Sources

- [[2-pytorch.md]]

```text

---
