```yaml
title: "Lecture 3/4 — vLLM 推理引擎关键特性 (Key Features of the vLLM Inference Engine)"
type: source
tags: [inference, llm, vllm, continuous-batching, kv-cache, paged-attention, quantization, speculative-decoding, chunked-prefill, scheduling, tensor-parallelism, pipeline-parallelism]
created: 2026-06-22
updated: 2026-06-22
sources: [3:4 - vllm.pdf]
```

# Lecture 3/4 — vLLM 推理引擎关键特性

Source: 上交大 AI Infra 团队 (SJTU AI Infra Team), lecture series.

This lecture covers the key systems-level features that make vLLM a high-throughput LLM inference engine: continuous batching, KV cache management, PagedAttention, scheduling, quantization, speculative decoding, and multi-node/multi-GPU serving.

---

## 1. 连续批处理 (Continuous Batching)

### 背景

LLM 推理分两个阶段：
- **Prefill**：处理输入 prompt，一次性计算所有 token 的 KV cache，计算密集
- **Decode**：逐 token 生成，每步只生成一个 token，内存带宽密集

传统**静态批处理 (static batching)**：将一批请求打包，等所有请求都完成才接受下一批。问题：序列长度不一，短序列完成后 GPU 空转等待长序列，利用率低。

### 连续批处理原理

**连续批处理 (continuous batching)**，也叫 iteration-level batching：每个 decode step 结束后，立刻将已完成的请求替换为新请求，不等待整批完成。

```
Step 1: [Req A, Req B, Req C, Req D]
Step 2: [Req A, Req B, Req C, Req E]  ← D 完成，E 插入
Step 3: [Req A, Req F, Req C, Req E]  ← B 完成，F 插入
```

效果：GPU 始终满载，吞吐量大幅提升（vLLM 论文报告相比 FasterTransformer 提升约 23×）。

### Prefill 与 Decode 的冲突

同批中混合 prefill（大计算量）和 decode（小计算量）请求时，prefill 请求会抢占 GPU 导致 decode 延迟增加（Time to First Token 与 Inter-Token Latency 的矛盾）。后续 Chunked Prefill 解决此问题。

---

## 2. 调度策略 (Scheduling)

### 调度目标

- 最大化 GPU 利用率（吞吐优先）
- 控制请求延迟（SLA 约束）
- 防止 KV cache 内存 OOM

### vLLM 调度器行为

vLLM 使用**先来先服务 (First-Come-First-Served, FCFS)** 策略，配合 KV cache 可用块数做准入控制：

1. 新请求进入等待队列
2. 调度器检查当前 KV cache 剩余块是否足够
3. 足够则将请求移入运行队列（running）
4. 不足则将低优先级（最晚到达）的请求**抢占 (preempt)**，释放其 KV cache 块
5. 被抢占的请求回到等待队列，下次调度重新做 prefill

**Swap**（可选）：被抢占的 KV cache 可先 swap 到 CPU 内存，避免重新 prefill，但 swap 带宽开销大，实践中效果参差。

### 调度粒度

- **Chunked Prefill** 启用前：prefill 请求整体作为一个调度单元
- **Chunked Prefill** 启用后：prefill 可以切成多个 chunk，与 decode 交替执行，平衡延迟

---

## 3. KV Cache 管理

### KV Cache 的内存压力

每个 token 的 KV cache 大小：

```
2 × num_layers × num_heads × head_dim × sizeof(dtype)
```

对于 Llama-3 8B（BF16）：
- 32 层 × 32 头 × 128 head_dim × 2 字节 × 2（K+V）= **512 KB/token**
- 4096 token 上下文 = **2 GB**
- 同时跑 100 个请求 = **200 GB**（远超单卡）

### 碎片化问题

不同请求的序列长度不同，传统方式为每个请求**预先分配最大长度**的连续内存：
- 内部碎片：请求未用完的空间浪费
- 外部碎片：内存不连续，无法给新请求使用
- 无法共享：相同 prompt 的请求重复存储 KV

---

## 4. PagedAttention

### 核心思想

借鉴操作系统**虚拟内存分页 (virtual memory paging)** 机制，将 KV cache 切分为固定大小的**块 (block)**，不要求物理连续。

- **Block size**：通常 16 或 32 tokens
- **Block table**：每个请求维护一张逻辑块号 → 物理块号的映射表
- **Physical block pool**：GPU 显存中预先分配好的物理块池

```
Logical KV:  [Block 0] [Block 1] [Block 2] [Block 3]
                ↓         ↓         ↓         ↓
Physical:    [Slot 7]  [Slot 2]  [Slot 9]  [Slot 1]   ← 非连续
```

### 优势

| 问题 | 传统方案 | PagedAttention |
|---|---|---|
| 内部碎片 | 预分配最大长度，浪费严重 | 按需分配块，最多浪费 1 块 |
| 外部碎片 | 连续分配，碎片率约 20-30% | 块可任意复用，碎片接近 0 |
| Prompt 共享 | 每个请求独立存储 | 相同 prompt 的块 Copy-on-Write 共享 |

### Copy-on-Write (写时复制)

多个请求共享同一 prompt 的物理块（引用计数 > 1）。当某个请求需要写入（生成新 token）时，触发 CoW：复制该块到新物理块，再写入。

应用场景：
- **Parallel sampling**：同一 prompt 生成多个输出（beam search、best-of-N）
- **Shared system prompt**：多请求共享相同系统提示

### PagedAttention Kernel

标准 attention kernel 假设 KV 连续存储；PagedAttention 需要自定义 CUDA kernel，通过 block table 间接寻址。vLLM 实现了两个版本：
- `paged_attention_v1`：直接实现
- `paged_attention_v2`：针对长序列的分治版本

---

## 5. Chunked Prefill

### 问题

Continuous batching 中，一个长 prefill 请求（如 8K tokens）会独占一个 step，导致同批 decode 请求的 inter-token latency (ITL) 出现尖峰。

### 解决方案

将 prefill 切成固定大小的 chunk（如 512 tokens），每个 step 只处理一个 chunk，和 decode token 一起打包执行：

```
Step 1: [prefill chunk 0~511, decode A, decode B, decode C]
Step 2: [prefill chunk 512~1023, decode A, decode B, decode C]
...
```

效果：
- decode 请求不再被长 prefill 阻塞，ITL 更稳定
- 计算密度保持高（chunk + decode token 合并成一个 forward pass）
- 代价：prefill 的 TTFT (Time to First Token) 变长（被分多步完成）

---

## 6. 量化 (Quantization)

### 为什么量化

- 减小模型权重内存占用（BF16 → INT8 省 50%，INT4 省 75%）
- 减小 KV cache 内存（FP8 KV cache）
- 提高计算吞吐（INT8/FP8 Tensor Core 峰值算力更高）

### 常见方案

| 方案 | 精度 | 对象 | 特点 |
|---|---|---|---|
| AWQ | INT4 | 权重 | 保护显著权重，精度损失小 |
| GPTQ | INT4/INT8 | 权重 | 基于 Hessian 的逐层量化 |
| SmoothQuant | INT8 | 权重+激活 | 将激活的难量化性转移到权重 |
| FP8 (W8A8) | FP8 | 权重+激活 | H100 原生支持，精度接近 BF16 |
| FP8 KV Cache | FP8 | KV cache | 减少 KV 显存，精度损失小 |

### KV Cache 量化

KV cache 量化为 FP8 可将 KV 显存减半，支持更长上下文或更大批次。vLLM 支持 per-tensor 和 per-channel 两种 FP8 KV cache 量化方式。

---

## 7. 投机推理 (Speculative Decoding)

### 原理

LLM decode 是**内存带宽瓶颈**（每步只生成 1 token，GPU 算力大量闲置）。投机推理用一个小**草稿模型 (draft model)** 快速生成多个候选 token，再用大**目标模型 (target model)** 一次验证：

```
Draft model: token₁, token₂, token₃, token₄, token₅  （5步）
Target model: 一次 forward pass 验证全部 5 个 token
接受 token₁~token₄，拒绝 token₅，重采样 token₅'
净效果: 1 次 target forward ≈ 产出 4 个 token
```

### 接受率与加速比

设草稿 token 的平均接受率为 α，每次投机生成 k 个草稿 token：

```
期望每步产出 token 数 ≈ (1 - αᵏ⁺¹) / (1 - α)
```

加速比取决于：
- α（草稿模型与目标模型的分布匹配程度）
- draft model 速度（越快越好）
- 目标模型 batch 大小（大 batch 时投机收益下降）

### Draft Model 来源

- **独立小模型**：如用 Llama 3.2 1B 给 Llama 3.1 70B 打草稿
- **EAGLE / Medusa**：在目标模型内部加轻量草稿头，共享 KV cache，draft 接近零开销
- **Ngram lookup**：从已生成文本中查找重复 ngram 作为草稿（适合长文档生成）

---

## 8. 张量并行与流水线并行 (Tensor Parallelism & Pipeline Parallelism)

### 张量并行 (Tensor Parallelism, TP)

将单层的权重矩阵按列/行切分到多张 GPU，每张 GPU 计算部分结果，通过 AllReduce 合并。

- TP=4：4 张 GPU 各持有 1/4 的 attention heads 和 FFN 权重
- 每层结束做一次 AllReduce（2 次：attention 后 + FFN 后）
- 延迟随 TP 增大（通信开销），通常 TP ≤ 单节点 GPU 数（避免跨节点 AllReduce）

### 流水线并行 (Pipeline Parallelism, PP)

将模型层按深度切分，不同 GPU 处理不同层。

- PP=4：GPU0 处理 layer 0-7，GPU1 处理 layer 8-15，…
- 通信量小（只传激活值），适合跨节点
- 缺点：流水线气泡（pipeline bubble），GPU 存在等待时间
- 配合 micro-batching 减少气泡

### 实践搭配

大模型典型配置：TP × PP 覆盖全部 GPU：

| 模型 | 典型配置 |
|---|---|
| 70B，8×A100 | TP=8, PP=1 |
| 405B，32×H100 | TP=8, PP=4 |

---

## 9. 前缀缓存 (Prefix Caching)

对于有相同前缀（如 system prompt）的多个请求，vLLM 可以复用已计算的 KV cache 块：

- 对每个 block 的 token 序列计算哈希值作为 cache key
- 新请求到来时，先查找 block hash，命中则直接复用，跳过 prefill
- LRU 淘汰策略管理 cache

效果：在 system prompt 占总长度比例大时（如 RAG、long system prompt），可大幅降低 TTFT 和计算量。

---

## 10. 关键指标 (Key Metrics)

| 指标 | 全称 | 含义 |
|---|---|---|
| TTFT | Time to First Token | 从请求到收到第一个输出 token 的时间，受 prefill 影响 |
| ITL / TPOT | Inter-Token Latency / Time Per Output Token | decode 阶段每生成一个 token 的时间 |
| Throughput | — | 系统每秒总输出 token 数 |
| Goodput | — | 满足 SLA 约束的有效吞吐量 |

TTFT 和 ITL 之间存在权衡：增大 batch size 提高吞吐但增加 ITL；Chunked Prefill 缓解 TTFT 对 ITL 的影响。

---

## 总结

vLLM 的核心设计理念是**把 GPU 内存当操作系统管内存一样管**：

1. **Continuous batching** — 消除静态批处理的等待浪费
2. **PagedAttention** — 消除 KV cache 的内存碎片，实现共享
3. **Chunked Prefill** — 平衡 TTFT 和 ITL
4. **Prefix caching** — 复用公共前缀，降低重复计算
5. **Speculative decoding** — 利用闲置算力加速 memory-bound decode
6. **Quantization** — 降低内存和带宽压力

这些技术叠加使 vLLM 相比朴素实现吞吐量提升 10-30×。
