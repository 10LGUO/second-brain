```yaml
title: PyTorch DataLoader and Dataset
type: concept
tags: [pytorch, dataloader, dataset, data-pipeline, deep-learning]
created: 2026-04-05
updated: 2026-04-05
sources: [2-pytorch.md]
```

# PyTorch DataLoader and Dataset

## Definition

`torch.utils.data.Dataset` and `torch.utils.data.DataLoader` form PyTorch's **data
loading and preprocessing pipeline**. `Dataset` defines how to access individual data
samples; `DataLoader` wraps a dataset to provide batching, shuffling, and parallel loading
via multiple worker processes.

## Dataset

A custom dataset is created by subclassing `Dataset` and implementing three methods:

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        """Return total number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return one sample and its label."""
        return self.data[idx], self.labels[idx]
```
