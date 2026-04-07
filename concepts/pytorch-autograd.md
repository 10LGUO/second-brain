```yaml
title: PyTorch Autograd (Automatic Differentiation)
type: concept
tags: [pytorch, autograd, backpropagation, gradient, computational-graph]
created: 2026-04-05
updated: 2026-04-05
sources: [2-pytorch.md]
```

# PyTorch Autograd

## Definition

**Autograd** is PyTorch's automatic differentiation engine. It records operations
performed on tensors with `requires_grad=True` into a **dynamic computational graph**
(define-by-run), then traverses this graph in reverse during `.backward()` to compute
gradients via the chain rule. This is the foundation that makes gradient-based
optimization of neural networks practical in [[pytorch-framework]].

## Core Mechanism
