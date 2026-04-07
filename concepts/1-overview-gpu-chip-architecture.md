---
title: AI 芯片构成（GPU 架构）
type: concept
tags: [gpu, hardware, hbm, cuda-core, tensor-core, nvlink, dma, memory-hierarchy]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
---

# AI 芯片构成（GPU 架构）

## 定义

AI 芯片（以 GPU 为典型代表）由计算核心、片上缓存、片外存储（显存）和芯片互联单元组成，各层次之间通过带宽差异显著的数据通路连接，共同决定模型训练与推理的性能上限。

## 关键属性
