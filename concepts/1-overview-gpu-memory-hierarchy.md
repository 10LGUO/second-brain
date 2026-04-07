---
title: GPU Memory Hierarchy
type: concept
tags: [gpu, memory, hbm, gddr, nvlink, dma, on-chip-cache, hardware]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
---

# GPU Memory Hierarchy

The GPU memory system is divided into multiple levels extending outward from the compute cores. Levels closer to the compute cores offer higher bandwidth but smaller capacity, while levels farther away offer larger capacity but lower bandwidth. Understanding this hierarchy is fundamental to kernel development, video memory optimization, and memory access bandwidth analysis.

## Memory Hierarchy (Nearest to Farthest)
