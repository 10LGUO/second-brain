---
title: GPU Software Stack
type: concept
tags: [gpu, software-stack, cuda, driver, compiler, ptx, sass, runtime, libraries]
created: 2026-04-05
updated: 2026-04-05
sources: [1-overview.md]
---

# GPU Software Stack

The GPU software stack is the full hierarchy of software layers that sit between application code (e.g., PyTorch training scripts) and the physical GPU hardware. Understanding each layer is essential for LLM infrastructure engineers, particularly for operator development, performance optimization, and debugging.

## Full Stack Hierarchy

```text
Chip → OS → Driver → Runtime → Operators/Communication Libraries
→ PyTorch/TensorFlow/PaddlePaddle → LLM Frameworks (Megatron, DeepSpeed, vLLM, SGLang)
```

## Layer 1: Hardware Instruction Set (ISA Level)

- **SASS (Shader Assembly):** Native GPU machine code (binary); directly controls CUDA Core/Tensor Core and memory units. Hardware-specific.
- **PTX (Parallel Thread Execution):** NVIDIA virtual instruction set (text/binary); hardware-architecture-independent; provides cross-generation compatibility.
- **Compilation flow:** CUDA C++ → PTX → JIT compiled by GPU driver → SASS.
- Example instructions:
  - `LDG (Load Global)`: loads data from global memory.
  - `FFMA (Fused Multiply-Add)`: executes multiply-add operation.

## Layer 2: Driver Layer

- **GPU Kernel Driver:** OS kernel module; manages GPU hardware resources (VRAM allocation, interrupt handling, power management).
- **UMD (User Mode Driver):** Provides API interface (e.g., NvAPI); manages CUDA context (multi-process/multi-thread isolation); command submission queue.
- **DMA:** Driver controls GPU direct access to host memory (e.g., PCIe BAR space).
- Error handling: captures GPU hardware errors (e.g., illegal memory access) and reports them to the application.

## Layer 3: Runtime Layer

- **CUDA Runtime API:** `cudaMalloc`, `cudaMemcpyAsync`, `cudaLaunchKernel`; manages CUDA Streams and Events. Synchronous and asynchronous variants.
- **CUDA Driver API (lower-level):** `cuMemAlloc`, `cuLaunchKernel`; allows fine-grained control (e.g., load PTX directly); used by framework developers — PyTorch calls Driver API directly.
- **UVA (Unified Virtual Addressing):** CPU and GPU memory share a unified address space.
- **Pinned Memory:** Accelerates host-device data transfer via `cudaMallocHost`.
- **Multi-stream parallelism:** Kernels in different streams can execute concurrently; synchronized via `cudaEvent`.
- **Multi-GPU communication:** `cudaDeviceEnablePeerAccess` enables direct GPU-to-GPU access.

## Layer 4: Compiler & Libraries
