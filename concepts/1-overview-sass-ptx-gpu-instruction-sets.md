```yaml
title: SASS and PTX — GPU Instruction Sets
type: concept
tags: [gpu, isa, compiler, hardware, nvidia, low-level]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# SASS and PTX — GPU Instruction Sets

NVIDIA GPUs use a two-level instruction set architecture: **PTX** (a virtual, portable instruction set) and **SASS** (the actual native machine code). Understanding these levels is important for deep compiler and operator work, especially for anyone working at the bottom layers of the AI infra stack.

## Key Properties
