```yaml
title: PyTorch Training Loop
type: concept
tags: [pytorch, training, optimization, deep-learning, workflow]
created: 2026-04-05
updated: 2026-04-05
sources: [2-pytorch.md]
```

# PyTorch Training Loop

## Definition

The **training loop** is the iterative procedure in [[pytorch-framework]] by which a
model's parameters are updated to minimize a loss function over a dataset. It
coordinates the model, data loader, loss function, and optimizer into a repeating
cycle.

## Standard Pattern

```python
model.train()                              # set training mode
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device) # move to GPU if available

        # 1. Zero gradients from previous step
        optimizer.zero_grad()

        # 2. Forward pass
        predictions = model(X)

        # 3. Compute loss
        loss = criterion(predictions, y)

        # 4. Backward pass — compute gradients
        loss.backward()

        # 5. Update parameters
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch}: loss = {running_loss / len(train_loader):.4f}")
```

## The Five-Step Cycle (per batch)

| Step | Call | Purpose |
| --- | --- | --- |
| 1 | `optimizer.zero_grad()` | Clear stale gradients |
| 2 | `pred = model(X)` | Forward pass, build graph |
| 3 | `loss = criterion(pred, y)` | Measure error |
| 4 | `loss.backward()` | Backprop, fill `.grad` |
| 5 | `optimizer.step()` | Apply gradient update |

## Validation Loop

```python
model.eval()                               # disable dropout / adjust BN
with torch.no_grad():                      # no gradient tracking needed
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        val_loss += criterion(pred, y).item()
```

## Common Additions

- **Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`
  — prevents exploding gradients
- **LR scheduling:** `scheduler.step()` after each epoch
- **Checkpointing:** Save `model.state_dict()` when validation loss improves
- **Mixed precision:** Wrap with `torch.cuda.amp.autocast()` and use `GradScaler`

## Related Concepts

- [[pytorch-nn-module]] — the model being trained
- [[pytorch-autograd]] — powers the backward pass
- [[pytorch-optimizer]] — executes the parameter update
- [[pytorch-loss-function]] — quantifies prediction error
- [[pytorch-dataloader]] — supplies batches of data

## Sources

- [[2-pytorch.md]] `[inferred — re-verify on re-ingestion]`

```text

---
