```yaml
title: PyTorch (Framework)
type: entity
tags: [pytorch, deep-learning, framework, facebook, meta, open-source]
created: 2026-04-05
updated: 2026-04-05
sources: [2-pytorch.md]
```

# PyTorch

## What It Is

**PyTorch** is an open-source deep learning framework developed primarily by
**Meta AI Research (FAIR)** and released publicly in 2016. It is built on top of the
Torch library (originally in Lua) and provides a Python-first, imperative programming
model for building and training neural networks. As of the mid-2020s it is the dominant
framework in academic deep learning research and increasingly in production.

## Why It Matters to This Wiki

PyTorch is the primary implementation framework for deep learning work referenced
throughout this wiki. Understanding its API — tensors, autograd, nn.Module, optimizers,
and data loading — is prerequisite to understanding most neural network implementation
details discussed in related concept and source pages.

## Key Components / Products

| Component | Description |
| --- | --- |
| `torch` | Core tensor library + autograd engine |
| `torch.nn` | Neural network layers and loss functions |
| `torch.optim` | Optimization algorithms |
| `torch.utils.data` | Dataset and DataLoader infrastructure |
| `torchvision` | Computer vision datasets, models, transforms |
| `torchaudio` | Audio processing tools |
| `torchtext` | NLP utilities |
| `torch.jit` | TorchScript for model serialization / production |
| `torch.distributed` | Distributed training across multiple GPUs/nodes |
| `torch.cuda` | CUDA GPU acceleration interface |

## Key Design Philosophy

- **Pythonic and imperative:** Models are ordinary Python classes; control flow
  uses standard Python constructs
- **Define-by-run:** Computation graph built dynamically each forward pass
- **Research-first:** Prioritizes flexibility and debuggability over deployment
  optimization (though TorchScript and TorchDeploy address production use)

## Versioning Context

The document is part of a numbered series suggesting a structured course or curriculum.
PyTorch's API has been largely stable since 1.x; major additions include
`torch.compile` (2.0+) for graph-level optimization.

## Related Entities

- [[meta-ai-research]]

## Related Concepts

- [[pytorch-tensor]]
- [[autograd]]
- [[pytorch-nn-module]]
- [[pytorch-optimizer]]
- [[pytorch-dataloader]]
- [[computational-graph]]

## Sources

- [[2-pytorch.md]]

```text

---
