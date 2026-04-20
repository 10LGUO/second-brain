```yaml
title: "GPU Operator Development Introduction (kernel_dev)"
type: source
tags: [gpu, cuda, hpc, operator-development, memory-hierarchy, optimization]
created: 2026-04-19
updated: 2026-04-19
sources: []
```

# GPU Operator Development Introduction

**Organization:** Shanghai Jiao Tong University (SJTU) AI Infra Team
**Author/Path:** snowliao > GPU Operator Development Introduction
**Project:** High-Performance Operator Development Project
**Document Type:** Internal wiki / technical guide
**Last Modified:** March 18 (year unspecified)

---

## Overview

This document is an internal technical guide for the SJTU AI Infra team's high-performance GPU operator development project. It covers GPU memory architecture, a complete taxonomy of CUDA kernel optimization techniques, optimization goals and decision frameworks, and a recommended operator development workflow. It is written as a practical reference for engineers learning and interviewing in GPU operator development.

---

## 1. GPU Memory Architecture
