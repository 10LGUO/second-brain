```yaml
title: PyTorch Optimizers (torch.optim)
type: concept
tags: [pytorch, optimizer, gradient-descent, deep-learning, training]
created: 2026-04-05
updated: 2026-04-05
sources: [2-pytorch.md]
```

# PyTorch Optimizers

## Definition

The `torch.optim` module provides **optimization algorithms** that update model parameters
using computed gradients. Optimizers implement variants of gradient descent and are
the bridge between the loss gradient (computed by [[autograd]]) and the parameter update
that reduces that loss.

## Common Optimizers

| Optimizer | Description |
| --- | --- |
| `optim.SGD(params, lr)` | Stochastic Gradient Descent; supports momentum, weight decay |
| `optim.Adam(params, lr)` | Adaptive Moment Estimation; default choice for most tasks |
| `optim.AdaGrad` | Accumulates squared gradients for per-parameter LR scaling |
| `optim.RMSprop` | Moving average of squared gradients |
| `optim.AdamW` | Adam with decoupled weight decay (preferred over Adam + L2) |

## Usage Pattern

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()        # 1. Clear old gradients
        outputs = model(inputs)      # 2. Forward pass
        loss = criterion(outputs, labels)  # 3. Compute loss
        loss.backward()              # 4. Backward pass
        optimizer.step()             # 5. Update parameters
```

## Learning Rate Scheduling

`torch.optim.lr_scheduler` provides schedulers to adjust learning rate during training:

- `StepLR`: Decay by `gamma` every N steps
- `CosineAnnealingLR`: Cosine annealing schedule
- `ReduceLROnPlateau`: Reduce when metric stops improving

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# After optimizer.step():
scheduler.step()
```

## Related Concepts

- [[autograd]]
- [[pytorch-nn-module]]
- [[loss-function]]
- [[pytorch-tensor]]

## Sources

- [[2-pytorch.md]]

```text

---
