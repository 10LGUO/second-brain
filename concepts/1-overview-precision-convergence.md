```yaml
title: Precision Convergence (精度收敛)
type: concept
tags: [precision, domestic-chips, ai-infra, debugging, training, inference]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
```

# Precision Convergence (精度收敛)

Precision convergence refers to the challenge of ensuring that a model running on a given hardware/software stack produces numerical outputs that match expected results — typically validated against a reference implementation (usually NVIDIA GPU). It encompasses both algorithmic correctness and hardware/chip system correctness. Precision is a foundational requirement: **"Only when precision is correct can you discuss performance."**

## Why It Matters

- **Prerequisite for everything else:** No performance optimization is meaningful if the model produces incorrect results.
- **Business-critical for domestic chip vendors:** A single unlocated precision error can cost billions in contracts. If the source of the error (software vs. hardware) cannot be identified, next-generation chip hardware may inherit the defect — propagating the problem indefinitely.
- **Systematic challenge:** Requires a dedicated methodology; cannot be solved by ad hoc debugging alone.

## Sources of Precision Issues
