```markdown
---
title: Reduce Operator
type: concept
tags: [reduce, operator-optimization, simd, parallelism, accumulation]
created: 2026-04-06
updated: 2026-04-06
sources: [5-kernel-dev.md]
---

# Reduce Operator

A reduce operator aggregates all elements of a tensor (or a slice thereof) into a single scalar value using an associative operation, most commonly summation. It is a foundational primitive used inside LayerNorm, Softmax, attention, and many other operators.

## Implementations (Progressive Optimization)
