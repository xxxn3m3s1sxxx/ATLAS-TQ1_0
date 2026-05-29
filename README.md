<p align="center">
  <img src="atlas_banner.svg" alt="ATLAS Banner" width="100%">
</p>

# ATLAS — TQ1.0 Ternary Inference Engine

CPU inference engine for BitNet b1.58 ternary-quantized models (Falcon3, Bonsai/Qwen3). Repacks HuggingFace safetensors into **TQ1.0** format (5 ternary trits/byte, Base-3) and runs fast inference via C++ DLL/SO + Python. **Windows + Linux x86-64**, no GPU, 8-16 GB RAM.

> ⚡ **v2.8.0**: Load-time int4 FFN quantization (18-26% faster 7B/10B). All 6 models correct: Falcon3, Bonsai, BitNet-2B4T.

## Supported Models

| Model | Atlas Size | Layers | Hidden | Intermediate | Heads | KV Heads | Vocab |
|-------|-----------|--------|--------|-------------|-------|----------|-------|
| BitNet-2B4T-b1.58 | 1.03 GB | 30 | 2560 | 6912 | 20 | 5 | 128256 |
| Falcon3-1B-Instruct | 1.22 GB | 18 | 2048 | 8192 | 8 | 4 | 131072 |
| Falcon3-3B-Instruct | 1.96 GB | 22 | 3072 | 9216 | 12 | 4 | 131072 |
| Falcon3-7B-Instruct | 2.75 GB | 28 | 3072 | 23040 | 12 | 4 | 131080 |
| Falcon3-10B-Instruct | 3.28 GB | 40 | 3072 | 23040 | 12 | 4 | 131072 |
| Ternary Bonsai-1.7B | 0.86 GB | 28 | 2048 | 6144 | 16 | 8 | 151669 |
| Ternary Bonsai-4B | 1.45 GB | 36 | 2560 | 9728 | 32 | 8 | 151669 |
| Ternary Bonsai-8B | 2.47 GB | 28 | 4096 | 14400 | 32 | 8 | 151669 |
| TriLM-1.1B | 0.53 GB | 24 | 1792 | 5120 | 28 | 28 | 50432 |

Falcon3: `head_dim=256`, `rope_theta=1000042`, GQA.  
BitNet-2B4T: `head_dim=128`, `rope_theta=500000`, SubLN (attn_sub_norm, ffn_sub_norm), **ReLU²** activation, Tie Embeddings.  
Bonsai/Qwen3: `head_dim=128`, `rope_theta=1M` (1.7B) or `5M` (4B) or `10M` (8B), YaRN factor=4.0, Tie Embeddings, QK-Norm, SwiGLU.  
TriLM: LLaMA architecture, MHA, SwiGLU, RoPE, natively ternary (QAT).  
All v5/v6 `.atlas` format — embeds tokenizer (v6: binary pool-lookup decode, no external deps).

### Model Sources

| Model | HF Repo |
|-------|---------|
| BitNet-2B4T | `microsoft/bitnet-b1.58-2B-4T` (U8 pre-quantized, use `--packed` flag) |
| Ternary Bonsai-1.7B | `prism-ml/Ternary-Bonsai-1.7B-unpacked` |
| Ternary Bonsai-4B | `prism-ml/Ternary-Bonsai-4B-unpacked` |
| Ternary Bonsai-8B | `prism-ml/Ternary-Bonsai-8B-unpacked` |
| Falcon3-1B-Instruct | `tiiuae/Falcon3-1B-Instruct` |
| Falcon3-3B-Instruct | `tiiuae/Falcon3-3B-Instruct` |
| Falcon3-7B-Instruct | `tiiuae/Falcon3-7B-Instruct` |
| Falcon3-10B-Instruct | `tiiuae/Falcon3-10B-Instruct` |
| TriLM-1.1B | `SpectraSuite/TriLM_1.1B_Unpacked` |

All are Apache 2.0 licensed.

## Quick Start

```bash
# Runtime only (inference):
pip install numpy
# For repacking models from safetensors:
pip install numpy safetensors transformers
# SSE web server (optional):
pip install fastapi uvicorn
```

### C API

```c
#include "atlas_ffi.h"
#include <stdio.h>

int main() {
    void* model = atlas_load("falcon3-10b-tq1.atlas");
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    // Tokenized prompt (use atlas_tokenizer_preencode + atlas_tokenizer_merge)
    int input_ids[] = { 193, 6764, 13, 426, 16874, 30, 399, 16874, 8221 };
    int n_input = sizeof(input_ids) / sizeof(input_ids[0]);

    int output_ids[256];
    int n_out = atlas_generate(model, input_ids, n_input,
                               4096, 200,          // max_seq_len, max_new_tokens
                               0.7f, 40, 0.9f,     // temp, top_k, top_p
                               1.1f,               // repetition_penalty
                               output_ids);
    if (n_out > 0) {
        printf("Generated %d tokens\n", n_out);
        // Decode via atlas_tokenizer_decode(...)
    }

    atlas_free(model);
    return 0;
}
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
# BitNet b1.58 (microsoft/bitnet-b1.58-2B-4T, pre-quantized U8):
python atlas_packer_bitnet.py path/to/bitnet-b1.58-2B-4T --packed

# Falcon3 (tiiuae/Falcon3-1B-Instruct / -3B / -7B / -10B):
python atlas_packer.py path/to/falcon3-3b-instruct

# Bonsai (prism-ml/Ternary-Bonsai-1.7B-unpacked / -4B / -8B):
python atlas_packer_bonsai.py path/to/ternary-bonsai-1.7b

# TriLM-1.1B (SpectraSuite/TriLM_1.1B_Unpacked):
python atlas_packer.py path/to/trilm-1.1b
```

Each packer autodetects model family from `config.json` and generates the output filename automatically (e.g. `bitnet-2b4t-u8-v2.atlas`, `falcon3-10b-tq1.atlas`, or `bonsai-4b-tq1-g128.atlas`). Requires `transformers` + `torch` for tokenizer config (install via `pip install -r requirements-dev.txt`).

## Python API

| Method | Description |
|--------|-------------|
| `AtlasModel(path)` | Load `.atlas` model. Optional `model_dir` for tokenizer config fallback. |
| `generate_c(text, ...)` | Generate text. Accepts string or `list[dict]` messages. Returns string. |
| `generate_stream(text, ...)` | Generator yielding int token IDs (caller decodes via `_cpp_decode`). |
| `set_system_prompt(text)` | Set system prompt for chat mode. |
| `set_seed(seed)` | Seed the RNG (default: random). |
| `set_num_threads(n)` | Set OpenMP thread count. |
| `set_use_f32_matmul(bool)` | Toggle f32 bypass mode (auto-enabled for hidden≤2048, rope_theta≥3M, or SubLN BitNet). |
| `set_use_hybrid_matmul(bool)` | Toggle hybrid FFN-int8 + QKV-packed mode (default). |
| `set_use_packed_matmul(bool)` | Toggle full TQ1-packed mode (all matmuls, no decompress). |
| `set_base_seq_len(int)` | Set trained context length for NTK scaling (v2.5.0). |
| `set_max_seq_len(int)` | Set ring buffer window size per session (v2.5.0). |
| `reset_cache()` | Zero KV cache — start fresh conversation (v2.6.0). |

## Performance

### v2.8.0 — Current (int4 FFN Quantization)

Measured on **Intel Core i7-7700T** (Kaby Lake, 4C/8T @ 2.9 GHz). Warm, `generate_c()` at T=0.7 / top_k=40 for sampling models, T=0 for deterministic. Times shown as sustained gen tok/s where available, otherwise total (prefill+gen).

| Model | Mode | tok/s | Quality |
|-------|------|:-----:|---------|
| **BitNet-2B4T** | f32 bypass | **2.8** | T=0: "The capital of France is Paris." Degenerates after 10-15 tokens. |
| **Falcon3-3B** | hybrid+int8 | **7.1** | Correct at T=0.7, 200 tokens (no EOS). |
| **Falcon3-1B** | f32 bypass | **10.1** | Gumbel-EOS at ~24 tokens. |
| **Bonsai-1.7B** | f32 bypass | **13.0** | "The capital of France is Paris." |
| **Bonsai-1.7B** | hybrid+int8 | 19.2 | "The capital of France is Paris." (minor quant noise) |
| **Bonsai-4B** | hybrid+int8 | **17.4** | "The capital of France is Paris." (YaRN 8K ctx) |
| **Bonsai-8B** | f32 bypass | **1.8** | T=0.7 coherent reasoning. |
| **Falcon3-7B (int8)** | hybrid+int8 | **2.5** | 61 tokens (sampling-dependent). |
| **Falcon3-7B (int4)** | hybrid+int8 | **3.15** | +26% vs int8, 200 tokens (no EOS). |
| **Falcon3-10B (int8)** | hybrid+int8 | **1.9** | 29 tokens (sampling-dependent). |
| **Falcon3-10B (int4)** | hybrid+int8 | **2.25** | +18% vs int8, 29 tokens (sampling-dependent). |

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
safetensors → atlas_packer*.py → .atlas file → atlas_infer.py → atlas.dll / libatlas.so
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

1. **Packer** (`atlas_packer.py`, `atlas_packer_bonsai.py`, `atlas_packer_bitnet.py`): Converts HF safetensors to TQ1 format. Falcon3/TriLM use `atlas_packer.py` (2-bit packed uint8 → Base-3 TQ1), Bonsai/Qwen3 use `atlas_packer_bonsai.py` (block-scaled g128 ttype=5), BitNet uses `atlas_packer_bitnet.py` (Microsoft U8 I2_S → TQ1).
2. **v5/v6 file format**: v5: 64-byte header (magic `"ATLAS"`, version=5, model hyperparameters, tokenizer offset/size), 12-byte tensor directory, name block, data, embedded tokenizer.json. v6: same structure + binary tokenizer block (128-byte header, pool offsets/lengths, BPE merges, byte_encoder, special tokens) — enables C++ pool-lookup decode without `transformers` or `tokenizers` libraries. C API `atlas_get_tokenizer()` exposes v5 JSON or v6 binary block. `AtlasModel('model.atlas')` suffices, no external model directory.
3. **C++ library** (`atlas_api.cpp`, single source for Windows + Linux): Loads the atlas file into memory. TQ1 tensors are decompressed to int8 with per-tensor `valloc`/`vfree` (`VirtualAlloc` on Windows, `mmap` on Linux). `atlas_forward` runs all N layers in one fused C++ call — RMSNorm + 7× int8 matmul (Q/K/V/O/gate/up/down) + fused GQA attention (RoPE + int8 KV-cache + causal softmax + weighted sum) + fused FFN (gate+up in one OMP region, SiLU+mul+quantize fused into down projection) — with no Python round-trips between layers. Ping-pong buffers avoid per-layer copies. KV-cache is int8-quantized with per-position scaling (`v2.3.0`), fully internal — no manual cache management needed.
4. **Int8 KV-Cache (v2.3.0)**: FP16→int8 quantization with dynamic scale per (KV-head, position). 10B@4K: 320 MB → 173 MB RAM. SIMD in-flight dequantization in attention hotpath. Cache is fully encapsulated in the C++ model struct — no k_cache/v_cache parameters in `atlas_forward` or Python API.
5. **Tokenizer (v6)**: Python `tokenizers` for encode (Falcon3's byte_encoder is fundamentally incompatible with C++ byte-level pre-encode — 191/256 byte tokens overwritten by special tokens). C++ pool-lookup for decode (`O(1)` per token). `_apply_chat_template` renders Falcon3 Jinja2 format without `transformers`.
6. **Matmul modes**: int8 (`vpmaddubs_epi16`), f32 bypass (`vfmadd231ps`), ternary-add (`vpsignb`), TQ1-packed (chunked decode + SIMD). Switched at runtime.
7. **Sampling**: Softmax multinomial with top-k/top-p (v2.0.4, replaced Gumbel-max). Xoshiro256** PRNG. T=0 → argmax. Optimized survivor-list collection (top-k prunes to ~40 candidates, top-p operates on survivors only — eliminates O(V log V) sort).
8. **Streaming** (v2.1.0): `atlas_generate_stream` callback C API, Python `generate_stream` generator. `set_system_prompt()` for context injection. Chat history via `list[dict]` messages with `_apply_chat_template`.
9. **Repetition penalty** (v2.1.1): Applied in C-core before top-k pruning, exposed in `generate_c`/`generate_stream`.

## Bugfix Chronology

20+ bugs were discovered and fixed during development. See [BUGS.md](BUGS.md) for the full chronology — `fseek` 32-bit overflow, Base-3 vs 2-bit packing, K/V cache swap, RMSNorm truncation, stack overflow, I2_S bit order/layout (BitNet), `stored_scale` formula (BitNet), and 15+ more.

All supported models (Falcon3 1B–10B, Bonsai 1.7B–8B, BitNet-2B4T, TriLM-1.1B) pass coherence at T=0.

## Files

| File | Purpose |
|------|---------|
| `atlas_api.cpp` | C++ library (load, forward, matmul, attention, norms, binary tokenizer, int8 KV-cache) — single source for Windows + Linux |
| `atlas_ffi.h` | C API contract |
| `atlas_infer.py` | Python `AtlasModel` class — `generate_c()`, `generate_stream()`, chat, caching |
| `atlas_server.py` | FastAPI SSE Web-Server — `/v1/chat/completions`, `/health`, `/reset` (v2.6.0) |
| `atlas_packer.py` | Falcon3 safetensors → TQ1 v5/v6 (embedded + optional binary tokenizer) |
| `atlas_packer_bonsai.py` | Bonsai/Qwen3 safetensors → g128 block-scaled TQ1 (ttype=5, per-row per-block fp16 scales) |
| `atlas_packer_bitnet.py` | BitNet b1.58: BF16 → TQ1.0 (per-tensor absmean) OR U8 pre-quantized → TQ1.0 (`--packed`) |
| `add_v6_block.py` | Append v6 binary tokenizer block to existing v5 files (fast migration) |
| `tests/` | pytest suite, CI smoke test, regression tests |
| `.github/workflows/build.yml` | CI Pipeline — GitHub Actions build on Ubuntu/Windows/macOS |

## Version History

| Version | Key Changes |
|---------|-------------|
| **v2.8.0** | **Load-time int4 FFN quantization (18-26% faster 7B/10B)**: New `atlas_matmul_i4_f32` AVX2 kernel — nibble-unpack + `(nibble^8)-8` sign-extension + `vpmaddubsw`. `atlas_quantize_ffn_to_i4()` converts int8→int4 at load time, halves FFN memory bandwidth. ttype=8 dispatch with `use_f32_matmul` guard for hybrid safety. Lane-permute fix. All 6 models verified. |
| **v2.7.9** | **Fix duplicate attn_sub_norm (BitNet collapse)**: Merge artifact in `forward_layer_internal` — sub-norm was applied twice to attention output. BitNet-2B4T: `/ / / / /` → `"The capital of France is Paris."`. `data_size` formula fixed for ttype=5 (`row_dim * n_blocks * 2`). |
| **v2.7.7** | **BitNet b1.58 Packing Fixes**: Fixed I2_S bit order/layout (Microsoft stores row 0 in high bits, was reading from low). Fixed `stored_scale` formula (`127/g→1/g`). Added `/127.0f` in ttype=5 decompress. Pipeline verified: 210/210 tensors at correlation 1.000000. U8 `--packed` path recommended. |
| **v2.7.6** | **BitNet b1.58 Final Fixes**: Correct dimensions (30L/2560H/6912I/20/5 heads). ReLU² confirmed (Microsoft `hidden_act: "relu2"`). `--packed` flag for U8 pre-quantized weights. Correct chat template (`Role: content<|eot_id|>`), correct EOS (128009). |
| **v2.7.5** | **ttype=5 Decompress + f32_bypass everywhere**: All ttype=5 tensors decompressed to int8 at load. f32_bypass forced for block-scaled models (rope_theta≥3M or hidden≤2048). Bonsai-8B: 0.2→1.6-2.2 tok/s. |
| **v2.6.4** | **min_new_tokens + persistent KV cache**: Logit clamp (-1e9f) suppresses EOS during warmup (default 20 tokens). `cache_offset` skips prefill for cached tokens in multi-turn. Full C API update for both `atlas_generate` and `atlas_generate_stream`. |
| **v2.6.3** | **BitNet ARCH_BITNET + Repo Cleanup**: `atlas_ensure_layer_idx()` C API fixes Python `forward()` path for BitNet models. f32_bypass forced for SubLN architectures (activation quantization destroys ~0.01× SubLN weights). 91 scratch files deleted, dead packers removed, `.gitignore` hardened. |
| **v2.6.2** | Safe `decompress_ttype5` dispatch (try/except guard) |
| **v2.6.1** | ttype=5 decompress + f32_bypass for all block-scaled models |
| **v2.6.0** | **SSE Web-Server + Prompt-Caching + CI Pipeline**: `atlas_server.py` — FastAPI `/v1/chat/completions` (SSE streaming), `atlas_reset_cache()` C-API + Python wrapper, `asyncio.Lock()`-serialized cache, `.github/workflows/build.yml` CI Pipeline (Ubuntu/Windows/macOS). Ring buffer + NTK context extension validated (8K Falcon3, 16K Bonsai-4B). |
| **v2.5.0** | **Context Window Extension**: Ring buffer KV cache (modulo `max_seq_len`), NTK-aware frequency scaling (`ctx_scale = max_seq_len / base_seq_len`), `set_base_seq_len()` API, dynamic `max_seq_len` per generate call. |
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

Code: Apache 2.0. BitNet b1.58: Microsoft Research. Falcon3: TII ([TII Falcon License 1.0](https://falconllm.tii.ae/)). Ternary-Bonsai: PrismML (Apache 2.0).
