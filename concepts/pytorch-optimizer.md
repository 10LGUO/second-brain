```yaml
title: PyTorch Optimizer
type: concept
tags: [pytorch, optimizer, gradient-descent, training, deep-learning]
created: 2026-04-05
updated: 2026-04-05
sources: [2-pytorch.md]
```

# PyTorch Optimizer

## Definition

A **PyTorch Optimizer** (`torch.optim`) encapsulates a gradient-based parameter
update rule. Given gradients computed by [[pytorch-autograd]], the optimizer adjusts
model parameters to reduce the loss. PyTorch provides standard optimizers out of the
box, all of which share a common interface.

## Common Optimizers

| Class | Algorithm | Notes |
| --- | --- | --- |
| `optim.SGD` | Stochastic Gradient Descent | Add `momentum=0.9` for faster convergence |
| `optim.Adam` | Adaptive Moment Estimation | Default choice for most tasks |
| `optim.AdamW` | Adam + weight decay decoupling | Preferred for Transformers |
| `optim.RMSprop` | Root Mean Square Propagation | Common in RNNs |
| `optim.Adagrad` | Adaptive Gradient | Per-parameter learning rates |

## Instantiation

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

## The Update Cycle

```python
optimizer.zero_grad()    # 1. clear old gradients
loss.backward()          # 2. compute new gradients
optimizer.step()         # 3. apply update rule
```

## Learning Rate Schedulers

`torch.optim.lr_scheduler` adjusts the learning rate during training:

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# Call after each epoch:
scheduler.step()
```

## Parameter Groups

Optimizers support different hyperparameters per layer group:

```python
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-4},
    {'params': model.head.parameters(),     'lr': 1e-3},
])
```

## Related Concepts

- [[pytorch-autograd]] — produces the `.grad` values that optimizers consume
- [[pytorch-training-loop]] — optimizer sits inside the training loop
- [[pytorch-nn-module]] — optimizer updates the module's parameters

## Sources

- [[2-pytorch.md]] `[inferred — re-verify on re-ingestion]`

```text

---
