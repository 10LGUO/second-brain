```yaml
title: PyTorch nn.Module
type: concept
tags: [pytorch, neural-network, nn-module, deep-learning, model-building]
created: 2026-04-05
updated: 2026-04-05
sources: [2-pytorch.md]
```

# PyTorch nn.Module

## Definition

`torch.nn.Module` is the **base class for all neural network models** in PyTorch.
Every custom model, layer, or network component is defined by subclassing `nn.Module`.
It provides parameter management, device movement, serialization, and hooks infrastructure
that make building and training deep learning models systematic and composable.

## Key Subclasses / Built-in Layers

| Layer | Description |
| --- | --- |
| `nn.Linear(in, out)` | Fully connected (dense) layer |
| `nn.Conv2d(in_ch, out_ch, kernel)` | 2D convolution |
| `nn.ReLU()`, `nn.Sigmoid()`, `nn.Tanh()` | Activation functions |
| `nn.BatchNorm2d(num_features)` | Batch normalization |
| `nn.Dropout(p)` | Dropout regularization |
| `nn.Embedding(num, dim)` | Embedding lookup table |
| `nn.LSTM(...)`, `nn.GRU(...)` | Recurrent layers |
| `nn.Sequential(*layers)` | Ordered container of modules |

## Defining a Custom Module

```python
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
```

## Key Methods

| Method | Purpose |
| --- | --- |
| `.forward(x)` | Define the computation (called via `model(x)`) |
| `.parameters()` | Iterator over all learnable parameters |
| `.named_parameters()` | Iterator over `(name, param)` pairs |
| `.state_dict()` | Ordered dict of all parameters and buffers |
| `.load_state_dict(dict)` | Restore parameters from a state dict |
| `.to(device)` | Move all parameters/buffers to device |
| `.train()` | Set to training mode (affects Dropout, BatchNorm) |
| `.eval()` | Set to evaluation mode |
| `.zero_grad()` | Zero all parameter gradients |

## Model Saving and Loading

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = MyNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

## Related Concepts

- [[pytorch-tensor]]
- [[autograd]]
- [[pytorch-optimizer]]
- [[pytorch-dataloader]]
- [[loss-function]]

## Sources

- [[2-pytorch.md]]

```text

---
