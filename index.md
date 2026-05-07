# Index

Content-oriented catalog of everything in this wiki. Updated on every ingest.

---

## Concepts

### GPU Architecture & Hardware
- [GPU Software & Hardware Architecture](concepts/1-overview-gpu-software-hardware-architecture.md) — SM structure, execution model, warp scheduling, memory hierarchy
- [GPU Memory Hierarchy](concepts/1-overview-gpu-memory-hierarchy.md) — HBM, SRAM, registers, cache hierarchy
- [HBM (High Bandwidth Memory)](concepts/1-overview-hbm-high-bandwidth-memory.md) — HBM architecture, bandwidth, A100 specs
- [AI Chip Architecture](concepts/1-overview-ai-chip-architecture.md) — GPU chip design, tensor cores, interconnects
- [SASS / PTX GPU Instruction Sets](concepts/1-overview-sass-ptx-gpu-instruction-sets.md) — GPU ISA, PTX virtual ISA, SASS native ISA
- [Warp](concepts/warp.md) — 32-thread SIMT execution unit, scheduling, shuffle instructions, coalescing
- [CUDA Thread Hierarchy](concepts/cuda-thread-hierarchy.md) — Grid, block, warp, thread; occupancy; inter-block communication
- [CUDA Host-Device Memory](concepts/cuda-host-device-memory.md) — cudaMalloc, cudaMemcpy, unified memory
- [CUDA Launch Configuration](concepts/cuda-launch-config.md) — block_size and grid_size selection rules, occupancy formula
- [CUDA Cheat Sheet](concepts/cuda-cheatsheet.md) — Hardware invariants, per-arch specs, occupancy formula, GEMM parameter roles

### CUDA Kernel Programming
- [CUDA GEMM](concepts/cuda-gemm.md) — Block tile, thread tile, K loop, pointer init, float4, double buffer
- [CUDA Transpose](concepts/cuda-transpose.md) — Naive, coalesced-write + __ldg, shared memory tiling, bank conflict padding/swizzling
- [SIMT Programming Model](concepts/5-kernel-dev-simt-programming-model.md) — SIMT vs SIMD, warp divergence, lockstep execution
- [SIMD Programming Model](concepts/5-kernel-dev-simd-programming-model.md) — Vectorized CPU operations
- [SIMT vs SIMD](concepts/1-overview-simt-vs-simd.md) — Comparison of GPU SIMT and CPU SIMD models
- [Arithmetic Intensity](concepts/5-kernel-dev-arithmetic-intensity.md) — FLOPs/byte, roofline model, compute vs memory bound
- [Roofline Model](concepts/kernel-dev-roofline-model.md) — Performance bound analysis, ridge point
- [Reduce Operator](concepts/5-kernel-dev-reduce-operator.md) — Shared memory tree, warp shuffle, float4+shuffle, SGEMV, inter-block reduction
- [Softmax](concepts/5-kernel-dev-softmax.md) — Three-kernel pipeline, numerical stability, atomicMax CAS, CUDA qualifiers
- [LayerNorm](concepts/5-kernel-dev-layernorm.md) — Online/Welford normalization, fused CUDA implementation
- [Shared Memory Tiling](concepts/kernel-dev-shared-memory-tiling.md) — Tiling strategy, As/Bs tiles, bank conflicts
- [Bank Conflict](concepts/kernel-dev-bank-conflict.md) — 32 shared memory banks, conflict detection, padding and swizzling
- [Coalesced Memory Access](concepts/kernel-dev-coalesced-memory-access.md) — 128B HBM transactions, threadIdx.x on column rule
- [Register Spill](concepts/kernel-dev-register-spill.md) — Spill to local memory, nvcc --ptxas-options=-v, impact on occupancy
- [Warp Reduce](concepts/kernel-dev-warp-reduce.md) — __shfl_down_sync, __shfl_xor_sync, tree reduction
- [Ping-Pong Buffer](concepts/5-kernel-dev-ping-pong-buffer.md) — Double buffer / prefetch to overlap compute and memory
- [GPU Memory Hierarchy (kernel-dev)](concepts/kernel-dev-gpu-memory-hierarchy.md) — Registers, shared memory, L2, HBM from kernel perspective
- [CUDA Kernel Optimization](concepts/kernel-dev-cuda-kernel-optimization.md) — Optimization strategies and patterns
- [Autotuning](concepts/kernel-dev-autotuning.md) — Parameter search for kernel configuration
- [Arithmetic Intensity (kernel-dev)](concepts/kernel-dev-arithmetic-intensity.md) — Kernel-level FLOPs/byte analysis

### LLM Infrastructure
- [AI Infra Overview](concepts/1-overview-ai-infra-overview.md) — LLM infra landscape, training vs inference, key challenges
- [LLM Training Infrastructure](concepts/1-overview-llm-training-infra.md) — Distributed training, parallelism strategies, frameworks
- [LLM Inference Infrastructure](concepts/1-overview-llm-inference-infra.md) — Serving, batching, KV cache, latency optimization
- [Large Model Infra](concepts/1-overview-large-model-infra.md) — Model parallelism, memory optimization, FSDP, ZeRO
- [LLM Infra Skill Stack](concepts/1-overview-llm-infra-skill-stack.md) — Skills required for LLM infrastructure engineering
- [KV Cache](concepts/1-overview-kv-cache.md) — Key-value cache for autoregressive inference
- [MFU (Model FLOPs Utilization)](concepts/1-overview-mfu-model-flops-utilization.md) — Training efficiency metric
- [Memory Bandwidth Utilization](concepts/1-overview-memory-bandwidth-utilization.md) — Inference efficiency metric
- [Compute-Communication Overlap](concepts/1-overview-compute-communication-overlap.md) — Hiding communication latency behind compute
- [Prefill-Decode Separation](concepts/1-overview-prefill-decode-separation.md) — PD separation architecture for LLM serving
- [PD Separation](concepts/1-overview-pd-separation.md) — Alternative page on prefill-decode separation
- [SLO Metrics (TTFT / TPOT)](concepts/1-overview-slo-metrics-ttft-tpot.md) — Time to first token, time per output token
- [Precision & Convergence](concepts/1-overview-precision-convergence.md) — FP16/BF16/FP8 training stability
- [GPU Software Stack](concepts/1-overview-gpu-software-stack.md) — CUDA, cuDNN, cuBLAS, driver stack
- [Operator Development](concepts/1-overview-operator-development.md) — Custom CUDA kernel development workflow
- [Operator Fusion](concepts/1-overview-operator-fusion.md) — Fusing kernels to reduce memory traffic

### PyTorch
- [PyTorch Tensor](concepts/pytorch-tensor.md) — Tensor operations, memory layout, strides
- [PyTorch Autograd](concepts/pytorch-autograd.md) — Automatic differentiation, computational graph
- [PyTorch nn.Module](concepts/pytorch-nn-module.md) — Module system, parameters, forward pass
- [PyTorch Optimizer](concepts/pytorch-optimizer.md) — SGD, Adam, optimizer step
- [PyTorch DataLoader](concepts/pytorch-dataloader.md) — Data loading, num_workers, prefetching
- [PyTorch Loss Function](concepts/pytorch-loss-function.md) — Cross-entropy, MSE, custom losses
- [PyTorch Training Loop](concepts/pytorch-training-loop.md) — Forward, backward, optimizer step pattern
- [Computational Graph](concepts/2-pytorch-computational-graph.md) — Dynamic graph construction in PyTorch
- [PyTorch Framework](entities/pytorch-framework.md) — PyTorch deep learning framework overview

---

## Entities

### Organizations & Frameworks
- [NVIDIA](entities/1-overview-nvidia.md) — GPU hardware, CUDA ecosystem, A100/H100
- [Megatron-LM](entities/1-overview-megatron.md) — NVIDIA's large model training framework
- [Megatron-DeepSpeed](entities/1-overview-megatron-deepspeed.md) — Combined Megatron + DeepSpeed framework
- [DeepSpeed](entities/1-overview-deepspeed.md) — Microsoft distributed training library, ZeRO optimizer
- [vLLM](entities/1-overview-vllm.md) — PagedAttention-based LLM serving framework
- [SGLang](entities/1-overview-sglang.md) — Structured generation LLM serving framework
- [NCCL](entities/1-overview-nccl.md) — NVIDIA Collective Communications Library
- [NVLink](entities/1-overview-nvlink.md) — NVIDIA high-bandwidth GPU interconnect
- [Flash Attention](entities/1-overview-flash-attention.md) — Memory-efficient attention algorithm
- [Transformer Engine](entities/1-overview-transformer-engine.md) — NVIDIA FP8 training library
- [PyTorch Framework](entities/pytorch-framework.md) — PyTorch deep learning framework
- [Nsight](entities/kernel-dev-nsight.md) — NVIDIA profiling tools (nsys, ncu)
- [pybind11](entities/kernel-dev-pybind11.md) — C++/Python binding library
- [CUDA Kernel Samples](entities/5-kernel-dev-cuda-kernel-samples.md) — CUDA_Kernel_Samples repo reference
- [Zebu](entities/5-kernel-dev-zebu.md) — FPGA-based hardware simulator

---

## Sources

- [1. General Overview — Lecture 1](sources/1-overview.md) — LLM infrastructure overview: training & inference challenges, GPU chip architecture, software stack, key frameworks, skill stack
- [2. PyTorch Detailed Explanation](sources/2-pytorch.md) — PyTorch framework tutorial: tensors, autograd, nn.Module, optimizers, DataLoader, computational graph
- [5. Kernel Dev](sources/5-kernel-dev.md) — Kernel operator development: SIMD/SIMT, LayerNorm/Softmax/Reduce, AI chip performance optimization
- [GPU Operator Development (kernel_dev)](sources/kernel-dev.md) — CUDA kernel optimization: memory hierarchy, warp reduce, shared memory tiling, bank conflict, ping-pong buffer, roofline model
- [CUDA SGEMM Optimization](sources/sgemm-readme.md) — Seven progressive SGEMM kernels: naive → shared memory tiling → 1D/2D thread tile → float4 → double buffer
- [GPU Performance Optimization Codelab](sources/gpu-perf-codelab.md) — Profiling workflow: torch.profiler, nsys, ncu, NVTX markers, roofline model, multi-model training bottlenecks
- [BAGEL](sources/bagel.md) — Bytedance 14B multimodal model (7B active MoT); architecture, VRAM requirements, inference pipeline, optimization targets

---

## Self-Test

- [Quiz](quiz.md) — 18 self-test questions: GPU architecture, memory hierarchy, GEMM tiling, thread tile, kernel optimization, launch config
