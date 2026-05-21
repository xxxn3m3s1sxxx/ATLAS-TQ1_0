<p align="center">
  <img src="atlas_banner.svg" alt="ATLAS Banner" width="100%">
</p>

# ATLAS — TQ1.0 Ternary Inference Engine

CPU inference engine for Falcon3 BitNet b1.58 ternary-quantized models. Repacks HuggingFace safetensors into **TQ1.0** format (5 ternary trits/byte, Base-3) and runs fast inference via C++ DLL/SO + Python. **Windows + Linux x86-64**, no GPU, 8-16 GB RAM.

> ⚡ **Architecture Scope:** ATLAS v2.0.1 is a hyper-optimized, dependency-free inference engine specifically tailored for **Falcon3 Ternary (1.58-bit) architectures** using an interleaved RoPE pattern and a fused SwiGLU loop.

## Supported Models

| Model | Atlas Size | Layers | Hidden | Intermediate | Heads | KV Heads |
|-------|-----------|--------|--------|-------------|-------|----------|
| Falcon3-1B-Instruct | 1.22 GB | 18 | 2048 | 8192 | 8 | 4 |
| Falcon3-3B-Instruct | 1.96 GB | 22 | 3072 | 9216 | 12 | 4 |
| Falcon3-7B-Instruct | 2.75 GB | 28 | 3072 | 23040 | 12 | 4 |
| Falcon3-10B-Instruct | 3.28 GB | 40 | 3072 | 23040 | 12 | 4 |

All use `head_dim=256`, `rope_theta=1000042`, GQA. v5/v6 `.atlas` format embeds tokenizer (v6: binary pool-lookup decode, no external deps) — no external files needed.

## Quick Start

```bash
# Runtime only (inference):
pip install numpy
# For repacking models from safetensors:
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

Measured on **Intel Core i7-7700T** (Kaby Lake, 4C/8T @ 2.9 GHz, 8 MB L3, 16 GB DDR4). All benchmarks via `m.generate_c()` at 30 tokens, warm (after 1+ runs). **All modes produce "Paris" at T=0** across all 4 models (coherence verified).

### v1.4.0 — Three Matmul Modes

| Mode | 1B | 3B | 7B | 10B |
|------|:--:|:--:|:--:|:---:|
| **Hybrid** (FFN int8 + QKV packed) | 17.6 tok/s | 5.3 tok/s | 9.2 tok/s | **11.0 tok/s** |
| **Int8** (full cache) | 17.4 tok/s | **5.5 tok/s** | **14.5 tok/s** | 8.8 tok/s |
| **Packed** (decode on-the-fly) | 15.8 tok/s | 3.3 tok/s | 10.2 tok/s | 6.4 tok/s |

#### Architectural Analysis

These numbers reflect the interplay of SIMD throughput, cache pressure, and memory bandwidth for each model size:

- **1B (f32 bypass class):** Mode is irrelevant (15.8–17.6 tok/s). The f32 bypass (`hidden ≤ 2048`) eliminates activation quantization entirely. Matmuls are small enough that the CPU processes them in register — neither memory bandwidth nor cache eviction is a bottleneck.

- **7B (int8 dominance):** Int8 reaches **14.5 tok/s**, outperforming hybrid (9.2) by 58%. The intermediate dimension (23040 = 7.5× hidden) creates enormous FFN matmuls that benefit from full `vpmaddubs_epi16` SIMD throughput on already-paged-in data. Hybrid's packed QKV save memory but add decode overhead that doesn't offset the FFN compute gain.

- **10B (hybrid dominance):** Hybrid leads at **11.0 tok/s** vs int8 (8.8). On the i7-7700T's 8 MB L3 cache, the full int8 path causes constant cache evictions across 40 layers. Hybrid's packed QKV slashes memory traffic by 5×, keeping the FFN-heavy pipeline fed and avoiding L3 thrashing.

### Hybrid Mode (default, v1.3.2+)

Best speed/RAM balance. FFN projections (gate/up/down) run as decompressed int8 — they dominate compute. QKV/O projections stay in TQ1-packed format — they're memory-bound and benefit from 5× fewer bytes read.

For 1B (hidden=2048), the f32 bypass activates automatically to eliminate activation quantization noise.

### Int8 Cache

On first load, a `.i8` companion file is created (decompressed int8 tensors). Subsequent loads mmap it directly — sub-second startup:

| Model | Cold Load (decompress + save) | Warm Load (mmap) |
|-------|:--------:|:---------:|
| 1B | 4.5s | 0.8s |
| 3B | 10.1s | 1.4s |

### Packed Mode (v1.3.1, `use_packed_matmul=True`)

Decodes TQ1→int8 on-the-fly per matmul. No cache file needed. Loads in **1.2s** (1B) to **11.9s** (10B). ~2-3× slower than hybrid but uses less RAM and requires no disk cache.

### Load Times (First Load, including lm_head quantization + cache creation)

| Model | Hybrid | Int8 (fresh) | Packed |
|-------|:------:|:------------:|:------:|
| 1B | 7.2s | 4.5s | 1.2s |
| 3B | 13.6s | 10.1s | 2.3s |
| 7B | 83.6s | 26.3s | 20.8s |
| 10B | 119.2s | 25.0s | 11.9s |

First load is dominated by decompression (TQ1→int8), lm_head quantization, and disk write. Subsequent loads with `.i8` cache mmap in under 2s.

### 🛠️ Windows Runtime Troubleshooting

If you compile using the LLVM-MinGW toolchain and encounter a `FileNotFoundError` or activation error when loading `atlas.dll` via ctypes, ensure the following MinGW runtime DLLs are placed in your working directory (next to `atlas.dll`):
- `libunwind.dll`
- `libwinpthread-1.dll`
- `libc++.dll`
- `libomp.dll` (OpenMP parallelization backend)

These are shipped with the LLVM-MinGW distribution under `x86_64-w64-mingw32\bin\`. Copy them next to `atlas.dll` or add that directory to your `PATH`.

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
2. **v5/v6 file format**: v5: 64-byte header (magic `"ATLAS"`, version=5, model hyperparameters, tokenizer offset/size), 12-byte tensor directory, name block, data, embedded tokenizer.json. v6: same structure + binary tokenizer block (128-byte header, pool offsets/lengths, BPE merges, byte_encoder, special tokens) — enables C++ pool-lookup decode without `transformers` or `tokenizers` libraries. C API `atlas_get_tokenizer()` exposes v5 JSON or v6 binary block. `AtlasModel('model.atlas')` suffices, no external model directory.
3. **C++ library** (`atlas_api.cpp`, single source for Windows + Linux): Loads the atlas file into memory. TQ1 tensors are decompressed to int8 with per-tensor `valloc`/`vfree` (`VirtualAlloc` on Windows, `mmap` on Linux). `atlas_forward` runs all N layers in one fused C++ call — RMSNorm + 7× int8 matmul (Q/K/V/O/gate/up/down) + fused GQA attention (RoPE + cache + causal softmax + weighted sum) + fused FFN (gate+up in one OMP region, SiLU+mul+quantize fused into down projection) — with no Python round-trips between layers. Ping-pong buffers avoid per-layer copies.
4. **Tokenizer (v6)**: Python `tokenizers` for encode (Falcon3's byte_encoder is fundamentally incompatible with C++ byte-level pre-encode — 191/256 byte tokens overwritten by special tokens). C++ pool-lookup for decode (`O(1)` per token). `_apply_chat_template` renders Falcon3 Jinja2 format without `transformers`.
5. **Matmul modes**: int8 (`vpmaddubs_epi16`), f32 bypass (`vfmadd231ps`), ternary-add (`vpsignb`), TQ1-packed (chunked decode + SIMD). Switched at runtime.
6. **Sampling**: Xoshiro256** PRNG + Gumbel-max top-k/top-p. T=0 → argmax. Optimized survivor-list collection (top-k prunes to ~40 candidates, top-p operates on survivors only — eliminates O(V log V) sort).

## Bugfix Chronology

Ten critical bugs were discovered and fixed during development. Any one of them would cause the model to produce garbage output (correlation near zero with reference activations) or crash.

### Bug 1 [FIXED]: `fseek` 32-bit overflow

The ATLAS file for Falcon3-7B is 2.74 GB. Tensors beyond offset ~2 GB were being read from the wrong file position because `fseek` (32-bit) truncated the offset. Fixed by replacing with `_fseeki64` (Windows) / `fseeko` (POSIX) via a `FSEEK` macro.

**Symptoms**: Layer-0 projections correct, deeper layers produce NaN or garbage.

### Bug 2 [FIXED]: 2-bit packing vs Base-3 unpacking

HuggingFace BitNet safetensors store 2-bit packed ternary values: `byte = v0 + v1*4 + v2*16 + v3*64`. The original packer decoded them with `%3` and `//3` (Base-3), producing incorrect ternary values. Fixed by using `& 3`, `>> 2`, etc.

**Symptoms**: Weight values off by ~5% per element, correlation still measurable (~0.5) but never reaching 1.0.

### Bug 3 [FIXED]: Row ordering (interleaved vs stride)

BitNet stores weights in a row-aware interleaved format: uint8 row `ur` contains columns for output rows `4*ur+0` through `4*ur+3`. The C++ matmul output is in this interleaved order (`ur*4+q`). But the reference HuggingFace `unpack_weights` produces stride-order output (`q*rows_packed+ur`). Without reordering, every projected tensor had correlation near 0 despite correct ternary values.

**Fix**: `out.reshape(batch, rows_packed, 4).transpose(0, 2, 1).reshape(batch, rows)`.

### Bug 4 [FIXED]: K/V cache swap

In `atlas_forward_layer`, K was written to `buf_hidden` but the attention copy read from `buf_up`; V was written to `buf_up` but read from `buf_hidden`. Fixed by swapping the copy destinations so K→buf_up and V→buf_hidden.

### Bug 5 [FIXED]: `_rmsnorm` weight truncation (create_string_buffer)

`ctypes.create_string_buffer()` treats the input as a C-string and truncates at the first NULL byte (`\x00`). FP16 value `1.0` = bytes `\x00\x3C` (little-endian), so RMSNorm weights containing many values ≈1.0 got truncated at the first such value, zeroing most norm outputs.

**Fix**: Cache the DLL's raw `ctypes.POINTER(c_uint8)` directly instead of converting to `bytes` → `create_string_buffer`.

### Bug 6 [FIXED]: Snap buffer overflow (batch resize)

Debug snapshot buffers were allocated once with the initial batch size and never resized. Prefill (B=12) after decode warmup (B=1) wrote past the end, causing access violation.

### Bug 7 [FIXED]: Activation buffer overflow on non-aligned TQ1 dimensions

TQ1 packing rounds dimensions up to multiples of 5: a projection with `inter_dim=8192` produces `packed_cols = ceil(8192/5) = 1639`, so the activation buffer must hold `1639 × 5 = 8195` floats per batch. But `max_aligned` was computed from the raw `inter_dim` (8192), rounding to 8192.

**Fix**: 7 bytes of extra padding before alignment to accommodate any TQ1 rounding.

### Bug 8 [FIXED]: Int8 cache corruption (five root causes)

The `.i8` mmap cache had five independent defects:
1. **Duplicate file offsets**: Precompute all offsets into a `std::vector<int64_t>`, write once.
2. **GQA scale over-read**: Only cache ttype==3 (int8-decoded) tensors.
3. **Inflated cache entries**: Only cache int8-decoded tensors.
4. **Missing prefetch**: Always call `atlas_prefetch_int8` regardless of cache source.
5. **Short fwrite on 70 MB+ tensors**: Wrap `fwrite` in retry loop with 64 KB chunks.

### Bug 8.6 [FIXED v1.0.8]: Cache Short-Write Protection

Disk space check via `GetDiskFreeSpaceExA`, `setvbuf` unbuffered writes, retry on short writes, file size validation on load.

### Bug 9 [FIXED]: Ping-pong buffer analysis

Even `n_layers` (all Falcon3 models: 18, 22, 28, 40) means `buf_a == hidden_states` after loop — no copy needed. The `if (n_layers % 2 == 1)` fix is correct for all cases.

### Bug 10 [FIXED]: KV cache pointer mismatch in forward_layer

Per-layer `forward_layer` passed full K/V caches but C++ always offset from index 0. Fixed by offsetting pointer per layer.

### Bug 11 [FIXED v1.2.0]: 10B tokenizer_offset int32 overflow

`int` (32-bit) overflow bei >2 GB Dateigröße. 10B Offset bei ~3.3 GB → negativ als int32. Fix: `uint32_t` + `ptrdiff_t` cast.

### Bug 12 [FIXED v1.4.0]: Stack overflow from alloca in forward_layer_internal

Four `alloca(B * qd * sizeof(float))` calls used ~2.9 MB stack at B=60+, exceeding 1 MB default Windows stack. Floated all 4 buffers to heap via `attn_ws` struct field, allocated in `ensure_buffers`. Also moved `scores` alloca outside per-batch loop. Total stack now ~210 KB max.

### Bug Re-Analysis v1.0.8: 1B Coherence False Alarm

Corr=0.23 test failure traced to **two bugs in Python test script**, not engine:
- **RMSNorm in-place corruption**: `_rmsnorm` modified input in-place, corrupting residual path.
- **Shared quantization gap**: C++ fused FFN uses one shared scale for gate+up; Python per-layer uses separate scales (0.3% expected variance).

**Result with fixes**: corr=0.9967, max_diff=4.0. Engine correct.

### Verification

All four Falcon3 models (1B, 3B, 7B, 10B) pass coherence: "The capital of France is Paris." at T=0 across all 3 matmul modes (12/12 test passes). The 1B model requires sampling (T ≥ 0.7) — greedy decoding degenerates due to model-inherent distribution (`"` with p=0.43), not engine.

## Files

| File | Purpose |
|------|---------|
| `atlas_packer.py` | safetensors → TQ1 v5/v6 (embedded tokenizer + optional binary tokenizer block) |
| `add_v6_block.py` | Append v6 binary tokenizer block to existing v5 files (fast migration) |
| `atlas_infer.py` | Python inference engine |
| `atlas_api.cpp` | C++ library (load, forward, matmul, attention, norms, binary tokenizer) |
| `atlas_ffi.h` | C API contract |
| `atlas.dll` / `libatlas.so` | Prebuilt binaries |
| `falcon3-{1,3,7,10}b-tq1.atlas` | Packed models |

## Version History

| Version | Key Changes |
|---------|-------------|
| **v2.0.1** | `scores` alloca → heap (stack fully sterile) |
| **v2.0.0** | C++ binary tokenizer (v6 format) — no `transformers` dependency at runtime |
| **v1.4.0** | Stack overflow fix, survivor-list sampling optimization |
| **v1.3.2** | Hybrid mode (FFN int8 + QKV packed), per-tensor dispatch |
| **v1.3.1** | Direct TQ1-packed matmul, `atlas_set_num_threads` |
| **v1.3.0** | Ternary-add kernel (`vpsignb`), eliminates row_sum correction |
| **v1.2.0** | C++ sampling (Xoshiro256**, Gumbel-max), `atlas_generate` |
| **v1.1.0** | AllocHdr-based valloc/vfree, production hardening |
| **v1.0.0** | Initial TQ1.0 inference engine |

## License

Code: Apache 2.0. BitNet b1.58: Microsoft Research. Falcon3: TII (subject to [TII Falcon License 1.0](https://falconllm.tii.ae/)).
