```yaml
title: pybind11
type: entity
tags: [tools, python, cpp, cuda, binding, pytorch]
created: 2026-04-19
updated: 2026-04-19
sources: [kernel-dev.md]
```

# pybind11

pybind11 is a lightweight, header-only C++ library that enables seamless interoperability between C++ (and CUDA) code and Python, by exposing C++ classes and functions as Python modules.

## Who/What It Is

pybind11 is an open-source C++ library (originally inspired by Boost.Python) that generates Python bindings for C++ code with minimal boilerplate. It supports CUDA/C++ GPU operator code and integrates naturally with PyTorch's extension mechanism.

## Why It Matters to This Wiki

In the SJTU high-performance operator development project, pybind11 is the recommended binding layer for:
- Exposing custom CUDA kernels (host code + kernel code) to Python as PyTorch-compatible operators.
- Enabling **integration testing** via Python test cases rather than pure C++ unit tests.
- Allowing comparison of custom kernel outputs against native PyTorch reference implementations in Python.

This workflow (CUDA kernel → pybind11 → Python import → pytest/script comparison) is the project's chosen development and testing methodology, favored for learning efficiency even though it is heavier than pure C++ unit tests.

The document notes that pure C++ unit tests are the industry standard for production systems (minimal dependencies, CI-suitable), while pybind11-based Python integration tests are used here for convenience and rapid iteration.

## Key Works / Products

- GitHub: `pybind/pybind11`
- Used extensively in PyTorch's `torch.utils.cpp_extension` module for custom operator development.

## Related Entities and Concepts

- [[cuda-kernel-optimization]]
- [[gpu-memory-hierarchy]]

## Sources

- [[kernel-dev]]

---
