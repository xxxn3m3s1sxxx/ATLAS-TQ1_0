<p align="center">
  <img src="atlas_banner.svg" alt="ATLAS Banner" width="100%">
</p>

# ATLAS — TQ1.0 Ternary Inference Engine

CPU inference engine for Falcon3 BitNet b1.58 ternary-quantized models. Repacks HuggingFace safetensors into **TQ1.0** format (5 ternary trits/byte, Base-3) and runs fast inference via C++ DLL/SO + Python. **Windows + Linux x86-64**, no GPU, 8-16 GB RAM.

## Supported Models

| Model | Atlas Size | Layers | Hidden | Intermediate | Heads | KV Heads |
|-------|-----------|--------|--------|-------------|-------|----------|
| Falcon3-1B-Instruct | 1.22 GB | 18 | 2048 | 8192 | 8 | 4 |
| Falcon3-3B-Instruct | 1.96 GB | 22 | 3072 | 9216 | 12 | 4 |
| Falcon3-7B-Instruct | 2.75 GB | 28 | 3072 | 23040 | 12 | 4 |
| Falcon3-10B-Instruct | 3.28 GB | 40 | 3072 | 23040 | 12 | 4 |

All use `head_dim=256`, `rope_theta=1000042`, GQA. v5 `.atlas` format embeds tokenizer — no external files needed.

## Quick Start

```bash
pip install numpy safetensors transformers
```

### Pack → Infer

```python
from atlas_infer import AtlasModel

# Load and generate (hybrid mode is default)
model = AtlasModel('falcon3-10b-tq1.atlas')
print(model.generate_c("What is the capital of France?", temperature=0.0))

# Deterministic sampling
model.set_seed(42)
print(model.generate_c("What is the capital of France?", temperature=0.7, top_k=40, top_p=0.9))
```

### Build from source

**Windows:**
```bash
compile.bat
```

**Linux:**
```bash
chmod +x compile-linux.sh && ./compile-linux.sh
```

Requires Clang (Windows) or GCC (Linux) with OpenMP, AVX2+FMA.

## Performance

Measured on **i5-1235U** (Alder Lake, 2P+8E, 8 OMP threads, 16 GB DDR4). All benchmarks via `m.generate_c()` at 30 tokens, warm (after 1+ runs).

### v1.3.2 — Three Matmul Modes

| Mode | 1B | 3B | 7B | 10B |
|------|----|----|----|-----|
| **Hybrid** (FFN int8 + QKV packed) | **14.9 tok/s** ~1.4 GB | **4.6 tok/s** ~4 GB | **7.4 tok/s** ~7 GB | **4.8 tok/s** ~9.3 GB |
| **Int8** (full cache) | 13.6 tok/s ~1.9 GB | 5.0 tok/s ~4.7 GB | 3.2 tok/s ~8.3 GB | 5.4 tok/s ~10.8 GB |
| **Packed** (no cache) | — | — | — | 1.5 tok/s ~7.5 GB |

**Coherence verified**: All modes produce identical argmax at T=0 (`"The capital of France is Paris."`) for 7B/10B. 1B requires sampling (T≥0.7) — greedy degenerates due to model-inherent distribution, not engine.

### Hybrid Mode (default, v1.3.2)

Best speed/RAM balance. FFN projections (gate/up/down) run as decompressed int8 — they dominate compute. QKV/O projections stay in TQ1-packed format — they're memory-bound and benefit from 5× fewer bytes read.

For 1B (hidden=2048), the f32 bypass activates automatically to eliminate activation quantization noise.

### Int8 Cache

On first load, a `.i8` companion file is created (decompressed int8 tensors). Subsequent loads mmap it directly — sub-second startup:

| Model | Cold Load | Warm Load |
|-------|:--------:|:---------:|
| 1B | 4.5s | 0.8s |
| 3B | 10.1s | 1.4s |

### Packed Mode (v1.3.1, `use_packed_matmul=True`)

Decodes TQ1→int8 on-the-fly per matmul. No cache file needed. Loads in **1.6s** (10B), uses **7.5 GB RAM**. ~3× slower than hybrid but fits in 8 GB systems.

## Architecture

```
safetensors → atlas_packer.py → .atlas file → atlas_infer.py → atlas.dll / libatlas.so
                                                         |
                                                    atlas_forward (fused N layers, C++)
                                                         |
                                              ┌──────────┼──────────┐
                                          RMSNorm  Attention  FFN(SiLU)
                                         (C++)      (C++)     (int8/packed)
                                                         |
                                               Final RMSNorm + LM head GEMV (int8)
```

### Pipeline

1. **Packer** (`atlas_packer.py`): De-interleaves BitNet's 4-row-packed uint8 → Base-3 TQ1. 5 trits per byte, padded with ternary-0.
2. **v5 file format**: 64-byte header (magic, hyperparams, tokenizer offset), 12-byte tensor directory, name block, data, embedded tokenizer (tokenizer.json + config).
3. **C++ library** (`atlas_api.cpp`): Single source for Win/Linux. Loads, maps, runs fused forward (all layers in one C call). Ping-pong buffers, no Python round-trips between layers. `atlas_generate()` — single FFI call for full decode loop.
4. **Matmul modes**: int8 (`vpmaddubs_epi16`), f32 bypass (`vfmadd231ps`), ternary-add (`vpsignb`), TQ1-packed (chunked decode + SIMD). Switched at runtime.
5. **Sampling**: Xoshiro256** PRNG + Gumbel-max top-k/top-p. T=0 → argmax.

## Files

| File | Purpose |
|------|---------|
| `atlas_packer.py` | safetensors → TQ1 v5 (embedded tokenizer) |
| `atlas_infer.py` | Python inference engine |
| `atlas_api.cpp` | C++ library (load, forward, matmul, attention, norms) |
| `atlas_ffi.h` | C API contract |
| `atlas.dll` / `libatlas.so` | Prebuilt binaries |
| `falcon3-{1,3,7,10}b-tq1.atlas` | Packed models |

## License

Code: Apache 2.0. BitNet b1.58: Microsoft Research. Falcon3: TII (subject to [TII Falcon License 1.0](https://falconllm.tii.ae/)).
