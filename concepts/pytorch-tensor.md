```yaml
title: PyTorch Tensor
type: concept
tags: [pytorch, tensor, deep-learning, linear-algebra, gpu]
created: 2026-04-05
updated: 2026-04-05
sources: [2-pytorch.md]
```

# PyTorch Tensor

## Definition

A **PyTorch Tensor** (`torch.Tensor`) is the fundamental n-dimensional array data
structure in [[pytorch-framework]]. It is analogous to NumPy's `ndarray` but with
two critical additions: it can reside on a GPU for accelerated computation, and it
can participate in automatic differentiation via [[autograd]].

## Key Properties

| Property | Description |
| --- | --- |
| `.shape` | Dimensions of the tensor (a `torch.Size` object) |
| `.dtype` | Data type: `torch.float32`, `torch.int64`, `torch.bool`, etc. |
| `.device` | Where the tensor lives: `cpu`, `cuda:0`, etc. |
| `.requires_grad` | Whether gradients should be tracked for this tensor |

## Creation Methods

```python
torch.tensor([1.0, 2.0, 3.0])       # from Python list
torch.zeros(3, 4)                    # zeros matrix
torch.ones(2, 2)                     # ones matrix
torch.rand(5, 5)                     # uniform random [0, 1)
torch.randn(3, 3)                    # standard normal
torch.arange(0, 10, step=2)         # range
torch.linspace(0, 1, steps=100)     # linear spacing
```

## Core Operations

- **Arithmetic:** `+`, `-`, `*`, `/`, `@` (matrix multiply), `torch.matmul()`
- **Indexing/Slicing:** Standard Python / NumPy-style indexing
- **Reshaping:** `.view()` (contiguous memory required), `.reshape()` (flexible),
  `.squeeze()`, `.unsqueeze()`, `.permute()`
- **Reduction:** `.sum()`, `.mean()`, `.max()`, `.min()`, `.argmax()`
- **In-place ops:** Suffixed with `_`: `.add_()`, `.fill_()` — modifies tensor
  directly; cannot be used on tensors that require grad

## Moving Between Devices

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)        # move tensor
x = x.cuda()            # shorthand for GPU
x = x.cpu()             # move back to CPU
```

## Relationship with NumPy

```python
# Tensor → NumPy (shares memory on CPU)
arr = tensor.numpy()

# NumPy → Tensor (shares memory)
tensor = torch.from_numpy(arr)
```

## Variants

- **Sparse Tensors:** `torch.sparse_coo_tensor()` — for high-dimensional sparse data
- **Quantized Tensors:** Reduced precision for inference efficiency
- **Complex Tensors:** `dtype=torch.complex64`

## Related Concepts

- [[autograd]] — gradient tracking is built on top of tensors
- [[neural-network-module]] — layers operate on tensors
- [[gpu-acceleration]] — tensors move to GPU via `.to(device)`

## Sources

- [[2-pytorch.md]] `[inferred — re-verify on re-ingestion]`

```text

---
