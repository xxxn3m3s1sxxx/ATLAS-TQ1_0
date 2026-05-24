<p align="center">
  <img src="atlas_banner.svg" alt="ATLAS Banner" width="100%">
</p>

# ATLAS — TQ1.0 Ternary Inference Engine

CPU inference engine for BitNet b1.58 ternary-quantized models (Falcon3, Bonsai/Qwen3). Repacks HuggingFace safetensors into **TQ1.0** format (5 ternary trits/byte, Base-3) and runs fast inference via C++ DLL/SO + Python. **Windows + Linux x86-64**, no GPU, 8-16 GB RAM.

> ⚡ **v2.4.1**: Full Bonsai/Qwen3 support — YaRN RoPE, QK-Norm, Tie Embeddings, per-row block-scaled TQ1 (g128), 5 critical C++ bugs fixed, 10× Bonsai speedup via int8 cache.

## Supported Models

| Model | Atlas Size | Layers | Hidden | Intermediate | Heads | KV Heads | Vocab |
|-------|-----------|--------|--------|-------------|-------|----------|-------|
| Falcon3-1B-Instruct | 1.22 GB | 18 | 2048 | 8192 | 8 | 4 | 131072 |
| Falcon3-3B-Instruct | 1.96 GB | 22 | 3072 | 9216 | 12 | 4 | 131072 |
| Falcon3-7B-Instruct | 2.75 GB | 28 | 3072 | 23040 | 12 | 4 | 131080 |
| Falcon3-10B-Instruct | 3.28 GB | 40 | 3072 | 23040 | 12 | 4 | 131072 |
| Ternary-Bonsai-1.7B-unpacked | 0.86 GB | 28 | 2048 | 6144 | 16 | 8 | 151669 |
| Ternary-Bonsai-4B-unpacked | 1.45 GB | 36 | 2560 | 9728 | 32 | 8 | 151669 |
| Ternary-Bonsai-8B-unpacked | ~3 GB | 36 | 4096 | 12288 | 32 | 8 | 151669 |

Falcon3: `head_dim=256`, `rope_theta=1000042`, GQA.  
Ternary-Bonsai/Qwen3: `head_dim=128`, `rope_theta=1M` (1.7B/8B) or `5M` (4B), YaRN factor=4.0, Tie Embeddings, QK-Norm, SwiGLU.  
All v5/v6 `.atlas` format — embeds tokenizer (v6: binary pool-lookup decode, no external deps).

### Model Sources

| Model (source) | HF Repo |
|----------------|---------|
| Ternary-Bonsai | `prism-ml/Ternary-Bonsai-*-unpacked` (FP16 safetensors, needs repacking) |
| Ternary-Bonsai | `prism-ml/Ternary-Bonsai-*-gguf` (GGUF, for llama.cpp) |
| Ternary-Bonsai | `prism-ml/Ternary-Bonsai-*-mlx-2bit` (MLX, for Apple Silicon) |
| Falcon3 | `tiiuae/Falcon3-*-Instruct` (must be ternarized via ATLAS packer) |

All are Apache 2.0 licensed.

## Quick Start

```bash
# Runtime only (inference):
pip install numpy
# For repacking models from safetensors:
pip install numpy safetensors transformers
```

### Generate (C++ core)

```python
from atlas_infer import AtlasModel

model = AtlasModel('falcon3-10b-tq1.atlas')

# Deterministic generation
print(model.generate_c("What is the capital of France?", temperature=0.0))

# Sampling with repetition penalty
model.set_seed(42)
print(model.generate_c("Tell me about Paris", temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.1))

# Streaming
for chunk in model.generate_stream("Write a short poem", max_new_tokens=100):
    print(chunk, end="", flush=True)

# Chat with system prompt
model.set_system_prompt("You are a helpful assistant.")
messages = [
    {"role": "user", "content": "What is the capital of France?"}
]
print(model.generate_c(messages, temperature=0.7))
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

### Repacking from Safetensors

```bash
# Falcon3: download from tiiuae/Falcon3-*-Instruct
python atlas_pack.py path/to/falcon3-model-dir

# Bonsai: download from prism-ml/Ternary-Bonsai-*-unpacked
python atlas_pack.py path/to/bonsai-model-dir
```

The CLI autodetects model family from `config.json` and generates the output filename automatically (e.g. `falcon3-10b-tq1.atlas` or `bonsai-4b-tq1-g128.atlas`). Requires `transformers` + `torch` for tokenizer config (install via `pip install -r requirements-dev.txt`).

## Performance

### v2.4.1 — Current (Bonsai + Bugfix Release)

Measured on **Intel Core i7-7700T** (Kaby Lake, 4C/8T @ 2.9 GHz). `generate_c()` at T=0, 30 tokens, warm. All models produce correct output at T=0.

| Model | Mode | tok/s | Quality (T=0) |
|-------|------|:-----:|---------------|
| **Falcon3-3B** | hybrid+int8 | 4.3 | "Paris. Paris is a city in France." |
| **Bonsai-1.7B** | f32 bypass | **13.0** | "The capital of France is **Paris**." |
| **Bonsai-1.7B** | hybrid+int8 | 19.2 | "The capital of France is Paris." |
| **Bonsai-4B** | hybrid+int8 | **15.2** | "The capital of France is Paris." |

Bonsai-1.7B defaults to f32 bypass (hidden=2048). Quantized hybrid yields higher throughput with minor quantization noise.

### Hybrid Mode (default, v1.3.2+)

FFN projections (gate/up/down) run as decompressed int8 — they dominate compute. QKV/O projections stay in TQ1-packed format (5× fewer bytes read). Int8 KV-cache with per-position scaling (v2.3.0). For small models (hidden≤2048), f32 bypass auto-activates to eliminate activation quantization noise.

### Int8 Weight Cache

On first load, a `.i8` companion file is created (decompressed int8). Subsequent loads mmap it directly for sub-second startup. Bonsai models cache 196–252 tensors (~1.4–3.6 GB).

### Int8 KV-Cache (v2.3.0, Internal)

The KV-cache is int8-quantized with dynamic per-position scaling (one float32 scale per KV-head per position). No manual allocation or management needed — `atlas_forward` calls `ensure_cache(max_seq_len)` internally. API signatures no longer carry `k_cache`/`v_cache` parameters.

**RAM savings**: 10B@4K context: 320 MB (fp16) → 173 MB (int8). SIMD in-flight dequantization in attention hotpath adds no measurable overhead.

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
                                         (C++)   (int8 KV)   (int8/packed)
                                                         |
                                               Final RMSNorm + LM head GEMV (int8)
```

### Pipeline

1. **Packer** (`atlas_packer.py`): De-interleaves BitNet's 4-row-packed uint8 → Base-3 TQ1. 5 trits per byte, padded with ternary-0.
2. **v5/v6 file format**: v5: 64-byte header (magic `"ATLAS"`, version=5, model hyperparameters, tokenizer offset/size), 12-byte tensor directory, name block, data, embedded tokenizer.json. v6: same structure + binary tokenizer block (128-byte header, pool offsets/lengths, BPE merges, byte_encoder, special tokens) — enables C++ pool-lookup decode without `transformers` or `tokenizers` libraries. C API `atlas_get_tokenizer()` exposes v5 JSON or v6 binary block. `AtlasModel('model.atlas')` suffices, no external model directory.
3. **C++ library** (`atlas_api.cpp`, single source for Windows + Linux): Loads the atlas file into memory. TQ1 tensors are decompressed to int8 with per-tensor `valloc`/`vfree` (`VirtualAlloc` on Windows, `mmap` on Linux). `atlas_forward` runs all N layers in one fused C++ call — RMSNorm + 7× int8 matmul (Q/K/V/O/gate/up/down) + fused GQA attention (RoPE + int8 KV-cache + causal softmax + weighted sum) + fused FFN (gate+up in one OMP region, SiLU+mul+quantize fused into down projection) — with no Python round-trips between layers. Ping-pong buffers avoid per-layer copies. KV-cache is int8-quantized with per-position scaling (`v2.3.0`), fully internal — no manual cache management needed.
4. **Int8 KV-Cache (v2.3.0)**: FP16→int8 quantization with dynamic scale per (KV-head, position). 10B@4K: 320 MB → 173 MB RAM. SIMD in-flight dequantization in attention hotpath. Cache is fully encapsulated in the C++ model struct — no k_cache/v_cache parameters in `atlas_forward` or Python API.
5. **Tokenizer (v6)**: Python `tokenizers` for encode (Falcon3's byte_encoder is fundamentally incompatible with C++ byte-level pre-encode — 191/256 byte tokens overwritten by special tokens). C++ pool-lookup for decode (`O(1)` per token). `_apply_chat_template` renders Falcon3 Jinja2 format without `transformers`.
6. **Matmul modes**: int8 (`vpmaddubs_epi16`), f32 bypass (`vfmadd231ps`), ternary-add (`vpsignb`), TQ1-packed (chunked decode + SIMD). Switched at runtime.
7. **Sampling**: Softmax multinomial with top-k/top-p (v2.0.4, replaced Gumbel-max). Xoshiro256** PRNG. T=0 → argmax. Optimized survivor-list collection (top-k prunes to ~40 candidates, top-p operates on survivors only — eliminates O(V log V) sort).
8. **Streaming** (v2.1.0): `atlas_generate_stream` callback C API, Python `generate_stream` generator. `set_system_prompt()` for context injection. Chat history via `list[dict]` messages with `_apply_chat_template`.
9. **Repetition penalty** (v2.1.1): Applied in C-core before top-k pruning, exposed in `generate_c`/`generate_stream`.

## Bugfix Chronology

Twenty bugs were discovered and fixed during development. Any one of them would cause the model to produce garbage output (correlation near zero with reference activations) or crash.

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

### v2.0.x Bugfix Summary

Twelve additional bugs found and fixed during the v2.0.x cycle:

| Bug | Severity | Fix | Version |
|-----|----------|-----|---------|
| Memory leak: `__del__` fehlte | HIGH | `atlas_free` in Python destructor | v2.0.2 |
| KV-cache overflow: `pos < max_seq_len` ungeprüft | HIGH | Defense-in-depth clamp in generate + attention | v2.0.2 |
| Stale `.i8` cache loading | MEDIUM | File-size validation + tensor shape check | v2.0.2 |
| Thread-unsafe static vectors | MEDIUM | `thread_local` + `std::call_once` | v2.0.2 |
| `atlas_set_seed(0)` → garbage | LOW | Pass seed 0 directly | v2.0.2 |
| `n_input >= max_seq_len` vor Prefill | CRITICAL | Early return with -1 | v2.0.3 |
| `scores_buf` null-deref bei OOM | LOW | Guard + early return | v2.0.3 |

### Verification

All four Falcon3 models (1B, 3B, 7B, 10B) and Bonsai models (1.7B, 4B) pass coherence: "The capital of France is Paris." at T=0. The 1B model requires sampling (T ≥ 0.7) — greedy decoding degenerates due to model-inherent distribution.

## Files

| File | Purpose |
|------|---------|
| `atlas_packer.py` | Falcon3 safetensors → TQ1 v5/v6 (embedded + optional binary tokenizer) |
| `atlas_packer_bonsai.py` | Bonsai/Qwen3 safetensors → g128 block-scaled TQ1 (ttype=5, per-row per-block fp16 scales) |
| `add_v6_block.py` | Append v6 binary tokenizer block to existing v5 files (fast migration) |
| `atlas_infer.py` | Python inference engine |
| `atlas_api.cpp` | C++ library (load, forward, matmul, attention, norms, binary tokenizer, int8 KV-cache) |
| `atlas_ffi.h` | C API contract |
| `falcon3-{1,3,7,10}b-tq1.atlas` | Packed Falcon3 models |
| `bonsai-{1.7,4}b-tq1-g128.atlas` | Packed Bonsai/Qwen3 models (g128 block-scaled) |
| `atlas_pack.py` | Unified CLI — autodetects model family, dispatches to correct packer |

## Version History

| Version | Key Changes |
|---------|-------------|
| **v2.4.1** | Static analysis bughunt (5 C++ bugs), ttype=5 int8 decompress for Bonsai (10× speedup), unified packer CLI, `generate()` chat template fix, repo cleanup |
| **v2.4.0** | Qwen3/Bonsai-4B TQ1.0 support — head_dim=128, QK-Norm, YaRN RoPE, Tie Embeddings, dynamic vocab |
| **v2.3.1** | Windows packer hotfix (`out.flush()` before `seek`), 7B v6 repair |
| **v2.3.0** | Int8 KV-Cache (fp16→int8, dynamic scaling, internal `ensure_cache()`), 10B@4K: 320→173 MB |
| **v2.2.2** | F16C in attention score + weighted sum (batch `_mm256_cvtph_ps` + FMA), 10B +47%, 3B +5.7% |
| **v2.2.1** | BPE-PQ priority queue in tokenizer merge (O(n²)→O(n log n)), 1401 tokens in 24ms |
| **v2.2.0** | TQ1-LUT in decompression (replace %3//3 with lookup), F16C (`_mm256_cvtph_ps`) for fp16→fp32 in RMSNorm + scalar, ~30% throughput |
| **v2.1.1** | Repetition penalty in C-core (before top-k), exposed in Python API |
| **v2.1.0** | Streaming (`atlas_generate_stream` callback C API, Python `generate_stream` generator), `set_system_prompt`, chat history via `list[dict]` messages |
| **v2.0.4** | Softmax sampling (replace Gumbel-max), thread_local→static revert, default T=0.7 |
| **v2.0.3** | n_input ≥ max_seq_len guard (CRITICAL), scores_buf OOM guard (Bug 9) |
| **v2.0.2** | Memory leak fix (`__del__`), KV-cache overflow clamp, stale .i8 cache validation, thread-local statics, seed=0 pass-through |
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
