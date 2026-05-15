# Atlas-TQ1-BitNet
Bit-exact C++ implementation of the Atlas BitNet inference engine.

## Overview
Atlas is a high-performance inference engine optimized for 1.58-bit (ternary) quantized models. It focuses on extreme memory efficiency and CPU-level optimization using AVX2 and OpenMP.

## Performance Benchmarks (2026-05-15)
Results measured on Intel Core i7-13700T (13th Gen) with DDR4-3200 (Single-Channel):

| Model | Size | Speed (Old PC) | Speed (New PC) | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Falcon3-7B** | 7B | 5.8s / token | **2.3s / token** | ~2.5x |
| **Falcon3-10B** | 10B | 7.5s / token | **3.3s / token** | ~2.3x |

*Note: Load times for 10B reduced from ~20s to **6.4s** due to NVMe/DDR4 optimizations.*

## Features
- **Bit-Exact Kernel:** Validated deterministic output across MSVC and Clang builds.
- **Ternary Quantization:** Native support for TQ1_0 (BitNet 1.58b) packing.
- **Advanced Sampling:** Temperature-based sampling to prevent repetition loops and handle EOS tokens.
- **SIMD Optimized:** Hand-crafted AVX2 kernels for weight unpacking and dot products.

## How to Run
1. **Compile:** `clang++ -O3 -march=native -fopenmp atlas_falcon3.cpp -o atlas.exe`
2. **Inference:**
   `python ask.py "Your prompt here" --10b --temp 0.4`

## Roadmap
- [x] AVX2 & OpenMP Support
- [x] Temperature Sampling
- [ ] Dual-Channel RAM Optimization (Target: < 2.0s for 10B)
- [ ] RAG Integration for CORE-X CMS
