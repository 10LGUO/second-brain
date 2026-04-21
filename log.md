# Log

Append-only chronological record of all wiki activity.

Format: `## [YYYY-MM-DD] <operation> | <title/description>`

Operations: `ingest`, `query`, `lint`, `create`, `update`
Tag `[Code Change]` for changes to wiki infrastructure (ingest.py, schema.md, .markdownlint.json, lint.md) rather than knowledge content.

---

## [2026-04-05] create | Wiki initialized

## [2026-04-05] [Code Change] create | ingest.py — chunked PDF ingestion pipeline with OCR fallback (pdfplumber + Tesseract chi_sim+eng)

## [2026-04-05] ingest | 1. General Overview — OCR fallback used (scanned image PDF)

## [2026-04-05] ingest | 2. PyTorch Detailed Explanation

## [2026-04-05] ingest | 1. General Overview — re-ingested with higher token limits and English output

## [2026-04-05] update | Translated all Chinese/pinyin content to English across wiki; renamed zongti-jieshao → overview in all filenames and content

## [2026-04-05] [Code Change] update | ingest.py — OCR auto-detection, English-only output instruction, increased token limits (8K/16K)

## [2026-04-06] ingest | 5. Kernel Dev — Operator Development (scanned PDF, OCR used)

## [2026-04-06] [Code Change] create | .markdownlint.json — lint config (MD013, MD025, MD041 disabled)

## [2026-04-06] lint | Fixed markdown lint across all pages: frontmatter ```yaml fences, MD040/MD060/MD031/MD032/MD047

## [2026-04-06] lint | Audit: duplicates, orphans, broken wikilinks, missing pages, index gaps — see findings in lint.md

## [2026-04-06] [Code Change] create | lint.md — reusable wiki health-check prompt

## [2026-04-07] query | Why customize CUDA kernels? — balance point, arithmetic intensity, warp mechanics, vectorized access

## [2026-04-07] update | Expanded concepts/5-kernel-dev-arithmetic-intensity.md — balance point interpretation, A100 example, optimization strategy table

## [2026-04-07] create | concepts/warp.md — warp definition, SIMT relationship, memory coalescing, float4 vs float8 instruction width limit

## [2026-04-07] update | concepts/5-kernel-dev-simt-programming-model.md — added Warp section, fixed frontmatter

## [2026-04-07] lint | Completed 12 truncated concept/entity/source pages; deleted 18 empty stubs and superseded duplicates; fixed all remaining lint errors

## [2026-04-10] query | CUDA kernel anatomy (add.cu) — grid/block/thread hierarchy, host-device memory, float4, FLOAT4 macro, CEIL, bounds check

## [2026-04-10] create | concepts/cuda-thread-hierarchy.md, concepts/cuda-host-device-memory.md

## [2026-04-13] query | CUDA reduction kernels (sum.cu) — shared memory tree reduction, warp shuffle reduction, inter-block communication, atomicAdd, atomicMax via CAS loop

## [2026-04-13] query | Softmax CUDA kernel pipeline — three-pass design (max→sum→normalize), numerical stability, __global__/__device__/static qualifiers

## [2026-04-13] update | concepts/cuda-thread-hierarchy.md — inter-block communication section (atomics, two-kernel pattern, Cooperative Groups)

## [2026-04-13] update | concepts/5-kernel-dev-reduce-operator.md — CUDA GPU reduction section: shared memory tree (v1), warp shuffle (v2), float4+shuffle (v3), laneId, __shfl_down_sync, warpSize

## [2026-04-13] update | concepts/5-kernel-dev-softmax.md — rewrote page: three-kernel pipeline, numerical stability example, atomicMax CAS explanation, CUDA function qualifiers table

## [2026-04-13] update | concepts/warp.md — added lanes section (laneId, warpSize), warp shuffle instructions (__shfl_down_sync), cross-warp coordination pattern

## [2026-04-13] query | Matrix transpose — coalescing at warp level, threadIdx.x vs threadIdx.y layout, __ldg (Load Global) read-only texture cache

## [2026-04-13] update | concepts/warp.md — expanded memory coalescing section: warp-level transaction batching, thread layout within block, __ldg explanation

## [2026-04-13] create | concepts/cuda-transpose.md — naive vs coalesced-write transpose, why writes prioritized, __ldg tradeoff

## [2026-04-14] query | Shared memory tiling transpose — bank conflicts, padding, dim3, static vs dynamic shared memory, grid-SM many-to-many mapping, 2D thread linear index

## [2026-04-14] update | concepts/1-overview-gpu-software-hardware-architecture.md — expanded execution model: grid-SM mapping, lane definition, thread layout/coalescing, shared memory SRAM/bank conflicts with examples

## [2026-04-14] update | concepts/cuda-transpose.md — added shared memory tiling section with bank conflict/padding explanation, dim3, template parameter rationale

## [2026-04-15] query | Bank structure — banks are internal physical subdivisions of SM's shared memory pool; blocks share the same 32 banks at different address ranges

## [2026-04-15] query | Warp as scheduling concept — warp scheduler hardware, 4 schedulers × 32 cores = 128 cores on A100, interleaving vs parallelism, threads borrow cores

## [2026-04-15] query | Coalescing rule — threadIdx.x always on column; why transpose is special; reads vs writes mitigation options (__ldg, shared memory tiling)

## [2026-04-15] query | 128-byte HBM transaction co-designed with 32-thread warp; threadIdx.x varies fastest by NVIDIA convention to match C row-major

## [2026-04-15] update | concepts/warp.md — added warp-as-scheduling-concept section (warp scheduler, 4×32 cores, interleave vs parallel), 128B transaction rationale, coalescing rule with transpose comparison

## [2026-04-15] update | concepts/1-overview-gpu-software-hardware-architecture.md — added full SM hardware diagram, clarified warp as scheduling concept in execution model

## [2026-04-16] lint | Fixed MD031/MD022 errors across warp.md, softmax.md, transpose.md, architecture.md, log.md

## [2026-04-18] query | GEMM block tile kernel — pointer initialization, K loop, A/B pointer advancement

## [2026-04-19] query | GEMM tiling mechanics — thread assignment, __syncthreads() purpose, accum register, copy parallelism vs compute sequentiality, why tiling reduces HBM traffic

## [2026-04-19] create | concepts/cuda-gemm.md — GEMM thread assignment, M/N/K/BM/BN/BK relationships, K loop with two sync points, tiling reuse analysis, mental model summary

## [2026-04-19] ingest | kernel_dev

## [2026-04-20] query | BM/BN meaning — block tile output ownership dimensions; grid sized CEIL(M,BM)×CEIL(N,BN)

## [2026-04-20] update | concepts/cuda-gemm.md — expanded BM/BN/BK section: output ownership semantics, grid sizing formula, block→thread→element hierarchy diagram

## [2026-04-20] update | CUDA_Kernel_Samples sgemm/kernel2.cu, kernel3.cu — added comments explaining BM/BN/BK/TM ownership semantics and grid/block sizing rationale

## [2026-04-20] ingest | sgemm/README.md — seven progressive SGEMM kernels (naive, shared memory, 1D/2D thread tile, float4, double buffer)

## [2026-04-20] create | sources/sgemm-readme.md — full source summary of CUDA SGEMM README

## [2026-04-20] update | concepts/cuda-gemm.md — added thread tile, float4, double buffer sections; optimization progression table

## [2026-04-21] ingest | grid_size/block_size selection rules — occupancy formula, wave sizing, OneFlow grid formula

## [2026-04-21] create | concepts/cuda-launch-config.md — block_size rules (warp alignment, occupancy floor, SM divisors, register pressure), grid_size rules (wave sizing, OneFlow formula)

## [2026-04-21] update | concepts/cuda-gemm.md — expanded pointer init section: per-matrix explanation of why K/N/bx appear or not, dimension alignment requirement

## [2026-04-21] create | quiz.md — 18 self-test questions covering GPU architecture, memory hierarchy, GEMM tiling, thread tile, kernel optimization, and launch configuration
