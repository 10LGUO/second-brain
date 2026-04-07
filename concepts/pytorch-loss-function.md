```yaml
title: PyTorch Loss Functions
type: concept
tags: [pytorch, loss, objective-function, training, deep-learning]
created: 2026-04-05
updated: 2026-04-05
sources: [2-pytorch.md]
```

# PyTorch Loss Functions

## Definition

**Loss functions** (also called objective functions or criteria) in
[[pytorch-framework]] quantify the discrepancy between a model's predictions and the
ground-truth targets. They return a scalar tensor on which `.backward()` is called
to initiate gradient computation. All are available under `torch.nn`.

## Common Loss Functions
