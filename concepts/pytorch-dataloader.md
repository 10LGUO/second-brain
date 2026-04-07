```yaml
title: PyTorch DataLoader and Dataset
type: concept
tags: [pytorch, data, dataloader, dataset, preprocessing, pipeline]
created: 2026-04-05
updated: 2026-04-05
sources: [2-pytorch.md]
```

# PyTorch DataLoader and Dataset

## Definition

PyTorch's `torch.utils.data` module provides a standardized pipeline for feeding
data to models. The two core abstractions are:

- **`Dataset`** — defines *what* the data is (individual samples)
- **`DataLoader`** — defines *how* to serve it (batching, shuffling, parallelism)

## `Dataset` — Abstract Base Class

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)           # total number of samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # one sample
```

Two methods **must** be implemented: `__len__` and `__getitem__`.

## `DataLoader` — Batching Engine

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,         # randomize order each epoch
    num_workers=4,        # parallel data loading subprocesses
    pin_memory=True,      # faster CPU→GPU transfer
    drop_last=False,      # drop incomplete final batch if True
)

for X, y in loader:
    ...   # X.shape = [32, ...]
```

## Built-in Datasets (`torchvision.datasets`)

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.MNIST(root='./data', train=True,
                            download=True, transform=transform)
```

Available: MNIST, FashionMNIST, CIFAR-10, CIFAR-100, ImageNet, COCO, etc.

## `torchvision.transforms`

Common preprocessing operations:

- `transforms.ToTensor()` — PIL/NumPy → float tensor, scale [0,255]→[0,1]
- `transforms.Normalize(mean, std)` — standardize per channel
- `transforms.Resize(size)`, `transforms.CenterCrop(size)`
- `transforms.RandomHorizontalFlip()`, `transforms.RandomCrop(size)` — augmentation
- `transforms.Compose([...])` — chain transforms

## `torch.utils.data.random_split`

```python
train_set, val_set = random_split(full_dataset, [50000, 10000])
```

## Related Concepts

- [[pytorch-training-loop]] — DataLoader is iterated in the training loop
- [[pytorch-nn-module]] — model receives batches from the loader

## Sources

- [[2-pytorch.md]] `[inferred — re-verify on re-ingestion]`

```text

---
