```yaml
title: Operator and Operator Fusion
type: concept
tags: [operators, kernel, gpu, performance, optimization, ai-infra]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# Operator and Operator Fusion

An **operator** (算子) is the smallest execution unit on a chip; it implements a specific compute operation (e.g., matrix multiply, activation function, normalization). Operators are implemented as **kernels** — code that runs on the chip's compute cores. Understanding and writing operators is a foundational skill in AI infra work.

## Key Properties
