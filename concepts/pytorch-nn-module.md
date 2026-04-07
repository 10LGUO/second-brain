```yaml
title: PyTorch nn.Module
type: concept
tags: [pytorch, neural-network, module, layers, deep-learning]
created: 2026-04-05
updated: 2026-04-05
sources: [2-pytorch.md]
```

# PyTorch `nn.Module`

## Definition

`torch.nn.Module` is the **base class for all neural network components** in
[[pytorch-framework]]. Any custom model, layer, or reusable block should subclass
`nn.Module`. It provides parameter management, device movement, serialization, and
hooks as built-in capabilities.

## Minimal Custom Model Pattern

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = MyModel()
output = model(input_tensor)   # calls forward() automatically
```

## Built-in Layers
