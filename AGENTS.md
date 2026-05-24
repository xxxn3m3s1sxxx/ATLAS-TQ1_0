# ATLAS — Falcon3 TQ1.0 Inference Engine

CPU inference engine for BitNet b1.58 ternary-quantized models (Falcon3, Bonsai/Qwen3). Repacks HuggingFace safetensors into **TQ1.0** format (5 ternary trits/byte, Base-3) and runs fast inference via C++ DLL/SO + Python. **Windows + Linux x86-64**, no GPU, 8-16 GB RAM.

## Architecture

- **TQ1.0 format**: 5 ternary trits per byte (Base-3 encoding), ~1.58 bits/weight
- **TQ1.0 g128** (ttype=5): Per-row per-block fp16 scales (block_size=128), 16/128 = 0.125 b/w overhead. Used by Bonsai/Qwen3 models. Packed via `matmul_tq1_block_reorder`.
- **4 matmul modes**: int8 (default, `vpmaddubs` SIMD), f32 bypass (reference, no activation quant), ternary (`vpsignb` pure sign), TQ1-packed (chunked decode + SIMD)
- **Hybrid mode** (default since v1.3.2): FFN tensors decompressed to int8, QKV/O stay TQ1-packed. Per-tensor dispatch.
- **f32 bypass**: Auto-enabled for `hidden <= 2048` (1B, Bonsai-1.7B). Eliminates activation quantization noise.
- **C++ binary tokenizer** (v6 format, v2.0.0): No `transformers` dependency at runtime. `tokenizers` lib for encode, C++ pool-lookup for decode.
- **128*row_sum correction**: Required for all uint8×int8 matmuls (activation quantization adds +128 bias).

## Supported Models

| Model | Atlas Size | Layers | Hidden | Intermediate | Heads | KV Heads | Vocab |
|-------|-----------|--------|--------|-------------|-------|----------|-------|
| Falcon3-1B-Instruct | 1.22 GB | 18 | 2048 | 8192 | 8 | 4 | 131072 |
| Falcon3-3B-Instruct | 1.96 GB | 22 | 3072 | 9216 | 12 | 4 | 131072 |
| Falcon3-7B-Instruct | 2.75 GB | 28 | 3072 | 23040 | 12 | 4 | 131080 |
| Falcon3-10B-Instruct | 3.28 GB | 40 | 3072 | 23040 | 12 | 4 | 131072 |
| Bonsai-1.7B-Chat | 0.86 GB | 28 | 2048 | 6144 | 16 | 8 | 151669 |
| Bonsai-4B-Chat | 1.45 GB | 36 | 2560 | 9728 | 32 | 8 | 151669 |

Falcon3: `head_dim=256`, `rope_theta=1000042`, GQA.  
Bonsai/Qwen3: `head_dim=128`, `rope_theta=1M` (1.7B) or `5M` (4B), YaRN factor=4.0, Tie Embeddings, QK-Norm, SwiGLU.

## Build

**Windows (clang):**
```bash
clang++ -fopenmp -O2 -mavx2 -mfma -mf16c -ffast-math -std=c++17 -shared -o atlas.dll atlas_api.cpp "C:\Program Files\LLVM\lib\libomp.lib"
```

**Linux (GCC/Clang):**
```bash
clang++ -fopenmp -O2 -mavx2 -mfma -mf16c -ffast-math -std=c++17 -fPIC -shared -o libatlas.so atlas_api.cpp -lgomp
```

**Env**: `KMP_DUPLICATE_LIB_OK=TRUE` on Windows for MKL compat.

## Performance

Measured on **Intel Core i7-7700T** (Kaby Lake, 4C/8T @ 2.9 GHz, 8 MB L3). Warm (model loaded and cached). `generate_c()` at T=0.7, top_k=40, 200 max tokens. Includes prefill of ~22 token prompt. Times shown as total tok/s (prefill + gen) and pure gen tok/s where sustained.

| Model | Hybrid tok/s (total) | Hybrid tok/s (pure gen) | Sustained gen tokens |
|-------|:--------------------:|:-----------------------:|:--------------------:|
| **3B** | **4.1** | **4.5** | 200 (no EOS) |
| **1B** | **7.4** | **10.1** | 24 (Gumbel-EOS) |
| **7B** | **1.9** | — | 61 (sampling-dependent) |
| **10B** | **1.3** | — | 29 (sampling-dependent) |

**10B/7B early EOS**: With T=0.7 sampling, Gumbel noise occasionally pushes EOS token ahead of natural continuation, limiting sustained gen length. This is Gumbel-max sampling behavior, not an engine limitation.

**T=0 argmax behavior** (deterministic mode):  
- **10B/7B**: Clean output ("The capital of France is Paris."), EOS after answer.  
- **3B**: Correct answer + newline collapse after completion (model-inherent, 22L insufficient calibration).  
- **1B**: Pure newline collapse (18L/2048H too small for stable argmax path).  

🚀 **Default recommendation**: Always use `T=0.7, top_k=40` for any model below 7B. T=0 is only reliable for 7B+.

### Bonsai Benchmarks (v2.4.1+)

| Model | Default Mode | tok/s | Quality (T=0) |
|-------|-------------|:-----:|---------------|
| **Bonsai-1.7B** | f32 bypass | **13.0** | "The capital of France is Paris." |
| **Bonsai-4B** | hybrid+int8 | **15.2** | "The capital of France is Paris." |

Bonsai-1.7B f32 bypass auto-enabled (hidden=2048). Quantized hybrid mode yields 19.2 tok/s.

### Architecture Notes

- **1B**: f32 bypass (`hidden ≤ 2048`) eliminates activation quantization. 10 tok/s pure gen.
- **3B vs 7B/10B**: Same hidden (3072) but intermediate scales from 9216 (3B) to 23040 (7B/10B). FFN matmul is 2.5× wider on 7B/10B, dominating the per-token cost.
- **10B**: 40 layers mean 1.8× more memory traffic per token than 7B (28 layers), despite same per-layer weight size.
- **Bonsai-4B vs -1.7B**: Same 36 layers (4B) vs 28 layers (1.7B). 4B has wider hidden (2560→2048) and intermediate (9728→6144). Bonsai-4B achieves higher tok/s due to better int8/AVX2 utilization per layer.

## Sampling

- **Gumbel-max**: `argmax_i(logits[i] + Gumbel(0,1))` samples from `softmax(logits)` — no softmax needed for top-k-only path.
- **Survivor-list optimization** (v1.5.0): After top_k pruning, softmax/heap/Gumbel operate only on survivors (~40 tokens) instead of full V=131072 vocab.
- top_k+p overhead ≈ top_k overhead.

## C API

```c
void* atlas_load(const char* path);
void atlas_free(void* model);
int atlas_generate(void* model, const int* input_ids, int n_input,
    int max_seq_len, int max_new_tokens,
    float temperature, int top_k, float top_p,
    float repetition_penalty,
    int* output_ids);
void atlas_set_seed(uint64_t seed);
void atlas_set_num_threads(int n);
void atlas_set_use_hybrid_matmul(void* model, int enable);
void atlas_set_use_packed_matmul(void* model, int enable);
void atlas_set_use_f32_matmul(void* model, int enable);
const char* atlas_get_tokenizer(void* model, int* size);
```

See `atlas_ffi.h` for full API.

## Roadmap

### v2.4.0 — Qwen3/Bonsai-Okosystem-Upgrade (AKTIV)
- **Packer (`atlas_packer_qwen.py`)**: Tensor-Mapping (Qwen3→ATLAS), Skalierungsfaktor-Extraktion (`max(abs(w))`), Ternarisierung (`round(w/scale)`), 5-Trit-Packing.
- **head_dim=128**: Alle Attention-Pfade (RoPE, Scores, Weighted Sum, KV-Cache) auf variablen head_dim umstellen.
- **QK-Norm**: Zwei neue RMSNorm-Tensoren pro Layer (`q_norm`, `k_norm`) im Attention-Hotpath.
- **Dynamisches Vocab**: `vocab_size` aus Datei-Header statt hardcoded 131072 (Bonsai: 151669). EOS/PAD-IDs aus Header.
- **YaRN RoPE**: Frequenz-Skalierung mit NTK-Approximation für rope_theta=5M, factor=4.0.
- **Tie Word Embeddings**: `lm_head` = `embed_tokens` (shared weights). Int8-Quantisierung des Embedding-Tensors.
- **SwiGLU-Hotpath**: `gate`/`up` parallel berechnet, SiLU fusioniert — identisch zu Falcon3, kein Umbau nötig.
- **Target**: Bonsai-4B TQ1.0 ~1.5 GB. Kompatibilität mit Qwen3 Familie.

### v2.5.0 — Context Window Extension
- **RoPE NTK-scaling**: Interpolation factor im C++-Core (`rope_theta` adjust + freq scaling)
- **Ring Buffer KV Cache**: Zirkuläres Überschreiben der ältesten Positionen, dynamisches `max_seq_len` pro `atlas_generate`-Aufruf
- **Target**: 8K Kontext für 10B bei ~346 MB Cache (2× heutige Reichweite bei ~gleichem RAM)
- Abhängigkeit: int8 KV-Cache (v2.3.0) — liefert die RAM-Reserve für die Verdopplung

### Deferred
- **F16C-Rester**: Diminishing returns (heiße Pfade bereits erledigt)
- **SSE Web-Server**: ~50 Zeilen FastAPI/SSE-Wrapper, jederzeit nachrüstbar

## Version History

| Version | Key Changes |
|---------|-------------|
| **v2.4.1** | **Static Analysis Bug Hunt**: 5 C++ bugs fixed (strict aliasing `*(uint16_t*)(odd_addr)`→memcpy, negative memset, unaligned AVX2 cast→pre-decoded float scales, thread-unsafe `static` buffers→`thread_local`). `atlas_decompress_all` now handles ttype=5 (g128 block-scaled) tensors — enables int8 cache for Bonsai models (10× speedup: 0.58→5.78 tok/s). Python `generate()` uses `_apply_chat_template()` for all paths (fixes Bonsai v5 template error). |
| **v2.4.0** | **Qwen3/Bonsai-4B**: head_dim=128, QK-Norm, YaRN RoPE 5M+f4, Tie Embeddings, dyn. Vocab (151669), EOS/PAD aus Header, Bonsai-4B Packer (`atlas_packer_qwen.py`). C++: `rope_scale`/`layer_stride`-Setters, `ensure_layer_idx` mit stride-11-Detektion, QK-Norm in `atlas_attention_f32`, NTK-YaRN in RoPE-Schleife. ✅ Falcon3-1B Regression, ✅ Bonsai-4B 100 Tokens (kein Crash). |
| **v2.3.1** | **Windows MSVCRT File-Buffer Hotfix**: `out.flush()` vor `out.seek(64)` im `atlas_packer.py` hinzugefügt. Verhindert Directory-Korruption bei Modellen >2 GB (7B v6). 7B v6 lädt nun fehlerfrei und generiert korrekt. |
| **v2.3.0** | **Int8 KV-Cache Quantisierung**: FP16→int8 mit dynamischer Skalierung pro (KV-Head, Position). Cache aus API-Signaturen entfernt, vollständig intern im `AtlasModel`-Struct via `ensure_cache()`. SIMD-In-Flight-Dequantisierung im Attention-Hotpath. 10B@4K: 320 MB → 173 MB RAM. Python-Schnittstelle bereinigt (kein manuelles Cache-Array-Management mehr). |
| v2.2.2 | F16C in attention score + weighted sum (batch _mm256_cvtph_ps + FMA), 10B +47%, 3B +5.7% |
| v2.2.1 | BPE-PQ priority queue in tokenizer merge (O(n²)→O(n log n)), 1401 tokens in 24ms |
| v2.2.0 | TQ1-LUT in decompression (replace %3//3 with lookup), F16C (_mm256_cvtph_ps) for fp16→fp32 in RMSNorm + scalar, ~30% throughput gain on 3B/10B |
| v2.1.1 | Repetition penalty in C-core (before top-k), exposed in Python generate_c/generate_stream |
| v2.1.0 | Streaming `atlas_generate_stream` callback C API, Python `generate_stream` generator, `set_system_prompt`, chat history via `list[dict]` messages |
| v2.0.4 | softmax sampling (replace Gumbel-max), thread_local→static revert, AGENTS.md benchmarks corrected, default T=0.7 |
| v2.0.3 | thread_local buffers, cache validation, std::call_once, seq clamp, seed fix |
| v2.0.1 | Task 0: scores alloca → heap (stack fully sterile) |
| v2.0.0 | C++ binary tokenizer (v6 format, no transformers dep) |
| v1.4.0 | Stack overflow fix (attn_ws heap alloc), survivor-list sampling |
| v1.3.2 | Hybrid mode (FFN int8 + QKV packed), per-tensor dispatch |
| v1.3.1 | Direct TQ1-packed matmul + atlas_set_num_threads |
| v1.3.0 | Ternary-add kernel (_mm256_sign_epi8), eliminates row_sum correction |
| v1.2.0 | C++ sampling (Xoshiro256**, Gumbel-max), atlas_generate |
| v1.1.0 | AllocHdr-based valloc/vfree, production hardening |
| v1.0.0 | Initial release — TQ1.0 inference engine |

## File Layout

- `atlas_api.cpp` — Full engine: AVX2 kernels, attention, RMSNorm, sampling, generate loop
- `atlas_infer.py` — Python `AtlasModel` class with `generate_c()`
- `atlas_ffi.h` — C API declarations (v6 header layout)
- `atlas_packer.py` — v5/v6 format writer, repacks safetensors → .atlas
- `add_v6_block.py` — Append v6 binary tokenizer block to existing v5 files

## Technical Details

- **v5 format**: `[header:64] [dir:n*12] [name_block] [token_data...] [tokenizer_block]`. Header bytes 29-32: tokenizer_size, 33-36: tokenizer_offset.
- **v6 format**: v5 + binary tokenizer block (128-byte header, offsets/lengths/pool, BPE merges, byte_encoder, special tokens).
- **Chat template**: `<|role|>\n{content}\n` — NO `<|im_end|>` tokens. Generation prompt: `<|assistant|>\n`.
- **Sampling overhead**: 1B top_k=40+p: ~3 tok/s (survivor-list makes top_p ≈ free after top_k).
- **Prefill**: All prompt tokens processed in single batched `atlas_forward` call (B=prompt_len).
- **Cache**: `.i8` cache auto-generated on first full int8 decompress, mmap'd on reload.
