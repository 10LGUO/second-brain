```yaml
title: PyTorch Autograd (Automatic Differentiation)
type: concept
tags: [pytorch, autograd, automatic-differentiation, backpropagation, deep-learning]
created: 2026-04-05
updated: 2026-04-05
sources: [2-pytorch.md]
```

# PyTorch Autograd

## Definition

**Autograd** is PyTorch's automatic differentiation engine. It implements
**reverse-mode automatic differentiation** (backpropagation) by dynamically building
a directed acyclic computational graph as forward-pass operations are executed, then
traversing that graph in reverse to compute gradients. This allows gradients of arbitrary
scalar-valued functions with respect to any tensor to be computed without manual derivation.

## How It Works

1. **Forward pass:** Operations on tensors with `requires_grad=True` record themselves
   in a computation graph. Each output tensor holds a reference to its `grad_fn`.
2. **Backward pass:** Calling `.backward()` on a scalar loss tensor triggers reverse
   traversal of the graph. The chain rule is applied at each node.
3. **Gradient accumulation:** Gradients are accumulated (summed) into the `.grad`
   attribute of each leaf tensor.
4. **Graph disposal:** By default the graph is freed after `.backward()` to save memory.
   Pass `retain_graph=True` to keep it.

## Key API

| API | Purpose |
| --- | --- |
| `tensor.requires_grad_(True)` | Enable gradient tracking in-place |
| `tensor.backward()` | Compute gradients from this tensor |
| `tensor.grad` | Accumulated gradient after backward |
| `tensor.detach()` | Return new tensor detached from graph |
| `torch.no_grad()` | Context manager: disable gradient tracking |
| `torch.enable_grad()` | Re-enable gradient tracking inside `no_grad` block |
| `tensor.grad_fn` | The function that created this tensor (None for leaf) |
| `tensor.is_leaf` | True if tensor is a leaf node |

## Gradient Zeroing

Gradients accumulate by default. Before each optimizer step, call
`optimizer.zero_grad()` (or `tensor.grad.zero_()` manually) to reset gradients,
preventing incorrect accumulation across iterations.

## Common Patterns

```python
# Typical training step
optimizer.zero_grad()
output = model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

## Related Concepts

- [[pytorch-tensor]]
- [[computational-graph]]
- [[neural-network]]
- [[pytorch-optimizer]]

## Sources

- [[2-pytorch.md]]

```text

---
