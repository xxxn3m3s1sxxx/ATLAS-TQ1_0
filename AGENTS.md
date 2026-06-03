# ATLAS — Falcon3 TQ1.0 Inference Engine

CPU inference engine for BitNet b1.58 ternary-quantized models (Falcon3, Bonsai/Qwen3). Repacks HuggingFace safetensors into **TQ1.0** format (5 ternary trits/byte, Base-3) and runs fast inference via C++ DLL/SO + Python. **Windows + Linux x86-64**, no GPU, 8-16 GB RAM.

## Architecture

- **TQ1.0 format**: 5 ternary trits per byte (Base-3 encoding), ~1.58 bits/weight
- **TQ1.0 g128** (ttype=5): Per-row per-block fp16 scales (block_size=128), 16/128 = 0.125 b/w overhead. Used by Bonsai/Qwen3 models. Packed via `matmul_tq1_block_reorder`.
- **5 matmul modes**: int8 (default, `vpmaddubs` SIMD), int4 (v2.8.0, nibble-unpack + `vpmaddubsw`, halves FFN memory bandwidth), f32 bypass (reference, no activation quant), ternary (`vpsignb` pure sign), TQ1-packed (chunked decode + SIMD)
- **ttype=8 (int4 packed)**: 2 int4 weights per byte, sign-extension via `(nibble ^ 8) - 8`. Converted at load time from existing int8 FFN tensors — no packer changes required. `[fp16_scale:2][packed_weights:rows*packed_cols][row_sums:rows*4]`. Guarded by `use_f32_matmul` to avoid activation quantization for hybrid architectures.
- **Hybrid mode** (default since v1.3.2): FFN tensors decompressed to int8, QKV/O stay TQ1-packed. Per-tensor dispatch.
- **f32 bypass**: Auto-enabled for `hidden <= 2048` (1B, Bonsai-1.7B), `rope_theta >= 3M` (Bonsai-8B), or `ARCH_BITNET` (SubLN models). Eliminates activation quantization noise — required for SubLN architectures where weights are ~0.01× and u8+128 activation quant destroys signal.
- **C++ binary tokenizer** (v6 format, v2.0.0): No `transformers` dependency at runtime. `tokenizers` lib for encode, C++ pool-lookup for decode.
- **v6 added_tokens**: Up to 256 extra tokens (IDs ≥ V) stored in binary tokenizer block. Encoded as `(offs[10], offs[11], offs[12], len_specials)` in 128-byte header. `offs[10]` = special_pool_offset, `offs[11]` = special_pool_len, `offs[12]` = special_map_offset. Preencode scans longest-first via `memcmp`. Decode looks up by ID.
- **`rope_interleaved_set` flag** (v2.10.3): Tracks whether config.json set `rope_interleaved`. If yes, Heuristik in `ensure_layer_idx` überschreibt nicht. Default: `true` (interleaved). Config override möglich (Llama/BitNet: half-split, `false`).
- **128*row_sum correction**: Required for all uint8×int8 matmuls (activation quantization adds +128 bias).

## Supported Models

| Model | Atlas Size | Layers | Hidden | Intermediate | Heads | KV Heads | Vocab | Arch |
|-------|-----------|--------|--------|-------------|-------|----------|-------|------|
| BitNet-2B4T-b1.58 | 1.03 GB | 30 | 2560 | 6912 | 20 | 5 | 128256 | SubLN, ReLU² |
| Falcon3-1B-Instruct | 1.22 GB | 18 | 2048 | 8192 | 8 | 4 | 131072 | Falcon3 |
| Falcon3-3B-Instruct | 1.96 GB | 22 | 3072 | 9216 | 12 | 4 | 131072 | Falcon3 |
| Falcon3-7B-Instruct | 2.75 GB | 28 | 3072 | 23040 | 12 | 4 | 131080 | Falcon3 |
| Falcon3-10B-Instruct | 3.28 GB | 40 | 3072 | 23040 | 12 | 4 | 131072 | Falcon3 |
| Ternary Bonsai-1.7B | 0.86 GB | 28 | 2048 | 6144 | 16 | 8 | 151669 | Qwen3 (QK-Norm) |
| Ternary Bonsai-4B | 1.45 GB | 36 | 2560 | 9728 | 32 | 8 | 151669 | Qwen3 (QK-Norm) |
| Ternary Bonsai-8B | 3.72 GB | 36 | 4096 | 12288 | 32 | 8 | 151669 | Qwen3 (QK-Norm) |
| TriLM-1.1B | 0.53 GB | 24 | 1792 | 5120 | 28 | 28 | 50432 | SubLN (head_dim=64) |
| TriLM-1.5B | 0.65 GB | 24 | 2048 | 6144 | 32 | 32 | 50432 | SubLN (head_dim=64) |
| TriLM-2.4B | 0.88 GB | 30 | 2304 | 7680 | 36 | 36 | 50304 | SubLN (head_dim=64) |
| Llama3-8B-1.58-100B-tokens | 4.11 GB | 32 | 4096 | 14336 | 32 | 8 | 131072 | Llama3 (GQA, QK-Norm) |
| BitCPM-CANN-1B | 0.83 GB | 28 | 2048 | 6144 | 16 | 2 | 73448 | Llama (LongRoPE) |
| BitCPM-CANN-3B | 1.35 GB | 32 | 2560 | 10240 | 32 | 2 | 73448 | Llama (LongRoPE) |

Falcon3: `head_dim=256`, `rope_theta=1000042`, GQA.  
BitNet-2B4T: `head_dim=128`, `rope_theta=500000`, SubLN (attn_sub_norm, ffn_sub_norm), **ReLU²** activation, Tie Embeddings.  
Bonsai/Qwen3: `head_dim=128`, `rope_theta=1M` (1.7B) or `5M` (4B) or `10M` (8B), YaRN factor=4.0, Tie Embeddings, QK-Norm, SwiGLU.  
TriLM (≤2.4B): `head_dim=64`, `rope_theta=10K`, SubLN, MHA (no GQA), SwiGLU.  
TriLM 3.9B (⚠️ not yet packed): `head_dim=128`, standard Llama arch, **NO SubLN** — different arch than smaller TriLMs!
Llama3 (Base Model): `head_dim=128`, `rope_theta=500000`, GQA (8 KV heads), QK-Norm, Tie Embeddings. V=131072. No chat template (Base model). 256 added tokens (IDs 128000-128255) stored in v6 binary tokenizer.
BitCPM-CANN-1B: `head_dim=128`, `rope_theta=10000`, LongRoPE (theta=100M for pos≥2048), Llama arch, SwiGLU. Tie Embeddings, V=73448. MiniCPM tokenizer (v5 embedded). Chat template: `<|role|>\n{content}\n`. 9.7 tok/s on i7-7700T (28L/2048H).
BitCPM-CANN-3B: `head_dim=128`, `rope_theta=10000`, LongRoPE (same factors), Llama arch, SwiGLU. Tie Embeddings, V=73448. MiniCPM tokenizer (v5 embedded). Hybrid path. T=0: "Paris." coherent, T=0.7: coherent.

### HF Alignment Check — 22 Models Verified

`scripts/verify_hf_alignment.py` auto-checks all models against HF Hub:

| Family | Count | Status |
|--------|-------|--------|
| Falcon3 (1B/3B/7B/10B) | 4 | ✅ All PASS |
| Bonsai (1.7B/4B/8B) | 3 | ✅ All PASS |
| BitNet-2B4T | 1 | ⏭️ SKIP (restricted repo) |
| TriLM (99M→3.9B, all sizes) | 10 | ✅ 9 PASS, ⏭️ 1 SKIP (2.3B private) |
| Qwen reference (Qwen2.5/3/QwQ) | 4 | ✅ All PASS |

**Critical finding**: TriLM family is **internally inconsistent** — ≤2.4B uses SubLN (head_dim=64), 3.9B uses standard Llama (head_dim=128, no SubLN). `derive_arch` auto-detects this via head_dim threshold.

## Build

**Windows (clang):**
```bash
clang++ -fopenmp -O2 -mavx2 -mfma -mf16c -ffast-math -std=c++17 -shared -o atlas.dll atlas_api.cpp
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
| **3B** | **7.1** | — | 200 (no EOS) |
| **1B** | **7.4** | **10.1** | 24 (Gumbel-EOS) |
| **2B (BitNet)** | **2.8** (f32 bypass) | — | T=0: Correct short answers ("The capital of France is Paris."), degenerates into repetition after 10-15 tokens. T=0.7: Coherent first sentence, then degrades. f32_bypass required (SubLN signal destroyed by u8+128 activation quant). U8-packed path (Microsoft pre-quantized) strongly preferred over BF16 ternarization. Two bugs fixed: I2_S bit order/layout (k*B+ur) + stored_scale formula (127/g→1/g). |
| **7B (int8)** | **2.5** | — | 61 (sampling-dependent) |
| **7B (int4)** | **3.15** | — | 200 (no EOS, +26%) |
| **10B (int8)** | **1.9** | — | 29 (sampling-dependent) |
| **10B (int4)** | **2.25** | — | 29 (sampling-dependent, +18%) |

**10B/7B early EOS**: With T=0.7 sampling, Gumbel noise occasionally pushes EOS token ahead of natural continuation, limiting sustained gen length. This is Gumbel-max sampling behavior, not an engine limitation.

**int4 FFN quantization** (v2.8.0): Load-time int8→int4 conversion halves FFN memory bandwidth. 7B achieves **3.15 tok/s** (+26% vs int8), 10B achieves **2.25 tok/s** (+18% vs int8). Automatic skip for f32_bypass models (Bonsai, BitNet, 1B).

**T=0 argmax behavior** (deterministic mode):  
- **10B/7B**: Clean output ("The capital of France is Paris."), EOS after answer.  
- **3B**: Correct answer + newline collapse after completion (model-inherent, 22L insufficient calibration).  
- **1B**: Pure newline collapse (18L/2048H too small for stable argmax path).  
- **2B (BitNet)**: Real English words (U8 path) but degenerates into repetition after 3-5 tokens (30L/2560H ternary-quantized, model-inherent). T=0.7 yields diverse token salad.  
- **8B (Bonsai)**: "The capital of France is Paris." at T=0, degenerates into repetition after 50+ tokens. T=0.7 yields coherent short answers.  

🚀 **Default recommendation**: Always use `T=0.7, top_k=40` for any model below 7B. T=0 is only reliable for 7B+.

### Bonsai Benchmarks (v2.5.0+)

| Model | Default Mode | tok/s | Context Window | Quality (T=0) |
|-------|-------------|:-----:|:--------------:|---------------|
| **Bonsai-1.7B** | f32 bypass | **13.0** | 4K (f32) / 8K (NTK) | "The capital of France is Paris." |
| **Bonsai-4B** | hybrid+int8 | **17.4** | 8K (YaRN) / 16K (NTK) | "The capital of France is Paris." |
| **Bonsai-8B** | f32 bypass | **2.2** | 8K (YaRN) / 16K (NTK) | "The capital of France is Paris.", T=0.7 coherent |

Bonsai-1.7B f32 bypass auto-enabled (hidden=2048). Bonsai-8B rope_theta=1M (<3M threshold, f32 not auto-triggered), use `AtlasModel(..., use_f32_matmul=True)`. Hybrid path degenerates to garbage — needs f32_bypass despite hidden=4096. Quantized hybrid mode yields 19.2 tok/s (1.7B only).

### Architecture Notes

- **1B**: f32 bypass (`hidden ≤ 2048`) eliminates activation quantization. 10 tok/s pure gen.
- **3B vs 7B/10B**: Same hidden (3072) but intermediate scales from 9216 (3B) to 23040 (7B/10B). FFN matmul is 2.5× wider on 7B/10B, dominating the per-token cost.
- **10B**: 40 layers mean 1.8× more memory traffic per token than 7B (28 layers), despite same per-layer weight size.
- **Bonsai-4B vs -1.7B**: Same 36 layers (4B) vs 28 layers (1.7B). 4B has wider hidden (2560→2048) and intermediate (9728→6144). Bonsai-4B achieves higher tok/s due to better int8/AVX2 utilization per layer.
- **Bonsai-8B**: 36L/4096H/12288I, rope_theta=1M with YaRN 4.0. Requires `use_f32_matmul=True`. The i8 cache (6.9 GB) reduces load time from decompression (TTFP from 40s to 1s). Generation is memory-bandwidth bound at 2.2 tok/s on DDR4.

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
void atlas_set_base_seq_len(void* model, int seq_len);   // v2.5.0: NTK context base
void atlas_reset_cache(void* model);                     // v2.6.0: Zero KV cache
void atlas_set_rope_interleaved(void* model, int enable);// v2.9.2: Toggle interleaved/half-split RoPE
void atlas_set_rope_theta(void* model, float theta);     // v2.9.2: Override RoPE theta frequency
const char* atlas_get_tokenizer(void* model, int* size);
```

See `atlas_ffi.h` for full API.

## Roadmap

### v2.10.4 ✅ — BitCPM-CANN-1B/3B Support + Debug Print Fixes (ABGESCHLOSSEN)
- **BitCPM-CANN-1B TQ1.0**: 28L/2048H/6144I/16:2 heads/128hdim/73448vocab, Llama-Architektur (LongRoPE). 0.83 GB, MiniCPM v5 Tokenizer eingebettet. Chat template: `<|role|>\n{content}\n`.
- **BitCPM-CANN-3B TQ1.0**: 32L/2560H/10240I/32:2 heads/128hdim/73448vocab, Llama-Architektur (LongRoPE). 1.35 GB, MiniCPM v5 Tokenizer. Hybrid path. T=0/T=0.7 beide kohärent ("The capital of France is Paris. Paris is the most populous city in France...").
- **Bug fix: Unconditional `logits[96944]` OOB Read**: Debug-Print in `atlas_generate` (Prefill top-5) las `logits[V-1]` mit V=73448. Der Print `logits[96944]` lag 586 Bytes über der Allokation → sporadischer Crash bei `0x...EAE0` je nach Heap-Layout. Fix: Alle unconditional Debug-Prints entfernt (attn_raw/attn_sft, DECLM/PRELM, LMHEAD, Prefill top-5, Decode top-5, before/after barriers).
- **`[ACTDBG] max_val=` Spam entfernt**: Debug-Print in `matmul_tq2_f32` (lines 3490-3501) produziert >1000 Zeilen pro Forward. Entfernt.
- **`ATLAS_DLL` env var fix**: `atlas_infer.py` prüfte `ATLAS_DLL`-Umgebungsvariable nur wenn `atlas.dll` nicht existierte. Fix: `ATLAS_DLL` überschreibt immer. Erleichtert Debug-DLL-Switching via `$env:ATLAS_DLL="C:\atlas\atlas_d.dll"`.
- **Packing fix: ZIP-based pytorch_model.bin**: CANN-3B uses modern PyTorch ZIP serialization (not safetensors). Packer loads lazily via `torch.load` — no OOM.
- **Performance**: 9.7 tok/s auf i7-7700T (CANN-1B, T=0.7, top_k=40, 30 Tokens).
- **43/43 Mock-Tests grün**: 4 zusätzliche BitCPM-Tests. Keine Regression auf 7 Architekturen.

### v2.10.3 ✅ — Bug Hunt Round 2+3 + Llama3-Support (ABGESCHLOSSEN)
- **12 Bugs gefunden & gefixt**: Bug #1 (Falcon3 BPE Vocab cutoff), Bug #2 (Llama3 stops in generate_c), Bug #3 (Llama3 special token suppression), Bug #4 (unaligned `*(uint32_t*)ap` → memcpy), Bug #5 (rope_interleaved Heuristik überschreibt config.json), Bug #6 (xoshiro_state global → thread_local), Bug #7 (g_has_avx512_vnni Init-Race), Bug #8 (EOS sentinel=0 konfligiert mit Token-ID 0), Bug #9 (fehlende BitNet-Stops in generate_c/_cpp_decode/generate), Bug #10 (silent except:pass → warn).
- **Llama3-8B-1.58-100B-tokens**: 32L/4096H/14336I, GQA (8 KV heads), QK-Norm, Tie Embeddings, V=131072. 256 added tokens (IDs 128000-128255) in v6 binary tokenizer. Base Model (no chat template). T=0 argmax: Prompt repetition. T=0.7: coherent.
- **v6 added_tokens Support**: C++ preencode scannt longest-first via sortierte `added_specs`. Decode per ID-Nachschlag. Preencode integriert in `tokenize_to_ids()`.
- **`rope_interleaved_set` Flag**: Neues Struct-Feld. Config-JSON setzt es → Heuristik in `ensure_layer_idx` feuert nicht. Default `true` (interleaved). Qwen3/Bonsai setzen auf `false` (half-split).
- **39/39 Mock-Tests grün**: 7 Architekturen (Falcon3, falcon3-ttype0, Qwen3, BitNet, TurboQuant, Llama3, Bonsai). E2E v6 tokenizer roundtrip mit added_tokens.
- **8/8 HF-Modelle regression-getestet**: Falcon3 (1B/3B/7B/10B), Bonsai (1.7B/4B/8B), BitNet-2B4T — alle heruntergeladen, inferiert, gelöscht. Keine Regression.

### v2.10.2 ✅ — Consistent Naming + HF-Repos + Bugfixes (ABGESCHLOSSEN)
- **HF-Repos konsistent**: Falcon3 Modelle umbenannt zu `Falcon3-*-1.58bit-ATLAS`. Bonsai-8B in Performance-Tabelle ergänzt.
- **`matmul_reorder_deq` pre-divide**: Scale-Division (÷127) zur Pack-Zeit, runtime nur noch Mul — minimale Optimierung.
- **v2.10.2 getaggt und gepusht** ✅.

### v2.10.1 ✅ — BitNet EOS Fix + HF-Deployment (ABGESCHLOSSEN)
- **BitNet EOS Token Fix**: `build_tokenizer_binary` Pattern-Matching priority fix. `eos_token_id=128001` (BOS) → `128009` (`<|eot_id|>`). Generiert `"Paris.<|eot_id|>"`.
- **HF-Deployment**: [`xxxn3m3s1sxxx/BitNet-2B4T-b1.58-ATLAS`](https://huggingface.co/xxxn3m3s1sxxx/BitNet-2B4T-b1.58-ATLAS) (1.04 GB, MIT). [`xxxn3m3s1sxxx/Ternary-Bonsai-*-ATLAS`](https://huggingface.co/models?search=xxxn3m3s1sxxx/Ternary-Bonsai) (3 Größen, Apache-2.0).
- **v2.10.1 getaggt und gepusht** ✅.

### v2.10.0 ✅ — Unified Packer + BF16 Weight-Scale Fix (ABGESCHLOSSEN)
- **Unified `pack_to_atlas.py`**: Single packer replaces all individual packers (`atlas_packer.py`, `atlas_packer_g128.py`, `atlas_packer_bitnet.py`). Architecture auto-detection via `config.json`. Single pipeline for Falcon3/Qwen3/BitNet/TriLM/Llama.
- **BF16 weight_scale Fix**: `weight_scale`-Tensoren sind bfloat16 dtype → `get_tensor_np()` mit `framework="np"` wirft Exception → `scales.get(sname, 1.0)` liefert immer 1.0. Fix: Fallback auf `reader.get_bf16_manual(tname)` in `pack_to_atlas.py:510-519`. Betrifft Falcon3, BitNet und alle Modelle mit BF16 weight_scale.
- **4 Falcon3 Modelle gepackt**: 1B (1.22 GB) ✅ "Paris." korrekt, 3B (1.97 GB) ✅ "Paris." korrekt, 7B (2.75 GB), 10B (3.28 GB).
- **Scale-Formel Analyse**: `matmul_reorder_deq` und `matmul_f32_reorder` dividieren durch scale (`sum / scale`) statt zu multiplizieren (`scale * sum`). Da dies ein konstanter Faktor auf alle Logits ist, hebt er sich in argmax/softmax auf → kein Einfluss auf Output-Qualität. Fix wäre `deq_scale = scale / 127.0f` statt `1/(127 * scale)`, aber nicht notwendig für korrekte Generierung.
- **TQ2-Pfad korrekt**: Multipliziert explizit mit `scale` in Zeile 3495 (`sf = fp16_to_fp32(sptr[j]) * scale`).
- **22/22 Mock-Tests**: CI grün. 5 Architekturen (Falcon3, falcon3-ttype0, Qwen3, BitNet, TurboQuant).
- **Realistischer ttype=0 Mock**: `atlas_mock_model.py` mit `falcon3-ttype0` Arch, `pack_tq1_per_tensor()` für echten TQ1-Dispatch-Pfad.
- **release_to_hf.py** unterstützt jetzt `--atlas-path` für Pre-Packed Files + HF-Upload mit korrektem YAML-Frontmatter.
- **BitNet EOS Token Fix**: `build_tokenizer_binary` in `pack_to_atlas.py` hatte zwei Bugs: (1) `tid == 0` Fallback vor Pattern-Matching setzte `eos=0` zu früh, (2) config.json `eos_token_id=128001` (BOS) statt korrektem 128009 (`<|eot_id|>`). Fix: Pattern-Matching zuerst, `tid == 0` Fallback entfernt, Header-EOS bevorzugt `tokenizer_config.json` String-Lookup via `tokenizer.token_to_id()`. BitNet generiert jetzt `"The capital of France is Paris.<|eot_id|>"` — korrektes Stoppen.
- **BitNet HF-Deployment**: [`xxxn3m3s1sxxx/BitNet-2B4T-b1.58-ATLAS`](https://huggingface.co/xxxn3m3s1sxxx/BitNet-2B4T-b1.58-ATLAS) — 1.04 GB, `base_model: microsoft/bitnet-b1.58-2B-4T`, `license: mit`, Tags: `ternary`, `quantized`, `atlas`, `tq1`, `cpu-optimized`, `bitnet`, `cpu-inference`.
- **Bonsai HF-Deployment**: Alle 3 Bonsai-Modelle (1.7B/4B/8B) auf [`xxxn3m3s1sxxx/Ternary-Bonsai-*-ATLAS`](https://huggingface.co/models?search=xxxn3m3s1sxxx/Ternary-Bonsai) deployed — `base_model: prism-ml/Ternary-Bonsai-*-unpacked`, `license: apache-2.0`, Tags: `ternary`, `quantized`, `atlas`, `tq1`, `cpu-optimized`, `bonsai`, `llm`, `cpu-llm`, `edge-ai`, `no-gpu`, `efficient-inference`.
- **bitnet_2b4t Mock-Modell**: `tests/atlas_mock_model.py` mit 2L/2560H/6912I/20/5 heads/128256 vocab, SubLN arch="bitnet", `f32_bypass`-Corridor-Test. 655 MB Datei, load/forward korrekt.
- **`release_to_hf.py` BitNet-Unterstützung**: `_read_atlas_arch()`, `"2B"` Size-Detection, `is_bitnet`-Branch in README-Gen (MIT license, korrektes Prompt-Template).
- **DLL+CLI Build OK**: Release-Build kompiliert sauber.
- **v2.10.0 getaggt und gepusht** ✅.

### v2.9.3 ✅ — AKI-Bug-Fix + HF-Alignment-Check + CI-Hardening (ABGESCHLOSSEN)
- **AKI-Bug fix**: `row_dim==vocab_size` Heuristik in `atlas_api.cpp:707-710` durch Name-Guard abgesichert (`embed_tokens`/`token_embd` substring check via `m->tensor_names[i]`). Schützt vor Fehlklassifikation von 1D-Norm-Tensoren (`model.norm.weight`) als 2D-Embedding bei `hidden_dim == vocab_size`. In keinem Produktionsmodell getriggert (8/8 HF-Modelle OK), aber defensiv notwendig.
- **HF-Alignment-Verifikation**: `scripts/verify_hf_alignment.py` — Zero-Download-Check gegen HuggingFace Hub (config.json + model.safetensors.index.json). **18 Modelle: 16 PASS, 0 FAIL, 2 SKIP** (restricted). Coverage: 4 Falcon3 + 3 Bonsai + 10 TriLM (99M→3.9B) + 1 BitNet (restricted). Auto-Discovery via `--discover` scannt alle bekannten HF-Orgs.
- **TriLM 3.9B Blindspot entdeckt**: TriLM-Familie ist intern inkonsistent! ≤2.4B = SubLN (head_dim=64, 11 Tensoren), 3.9B = Standard-Llama (head_dim=128, 9 Tensoren, KEIN SubLN). `derive_arch` erkennt via head_dim-Schwelle (`trilm` vs `trilm_nosubln`). Packer muss beim Packen entsprechend triggern.
- **Coverage gefixt**: Stale gcda/gcno-mismatch durch `del /Q *.gcda *.gcno` vor Rebuild. `compile.bat` läuft jetzt alle 44 Tests (inkl. OMP-Stress + E2E-Pipeline) unter Coverage. `--gcov-ignore-parse-errors` für Robustheit.
- **CI gehärtet**: `build.yml` läuft jetzt `verify_hf_alignment.py` als eigenen Step. Coverage auf Linux+Windows inkl. OMP+E2E-Tests.
- **Qwen-Arch-Differenzierung**: `qwen25` (ohne QK-Norm, `Qwen2ForCausalLM`) vs `qwen3` (mit QK-Norm, `Qwen3ForCausalLM`) — Arch-Detektion via `model_type`, nicht via Modell-ID.
- **EOS-Fix aus tokenizer**: `eos_id` von `m->tok.special[0]` statt `header_guess(11)`. TriLM 1.5B jetzt korrektes EOS.
- **2 neue C-APIs**: `atlas_set_rope_interleaved`, `atlas_set_rope_theta`.
- **`tests/atlas_mock_model.py`**: Synthetische v8.8-Modelle (200-300 KB) mit echtem TQ1-Packing für 3 Architekturen. `pack_tq1_g128` aus Produktions-Packer.
- **`tests/test_mock_model.py`**: 9 parametrisierte pytest-Tests (load/forward/batch × Falcon3/Qwen3/BitNet) in 1.14s.
- **Regression**: Falcon3-3B "Paris" ✓, Bonsai-8B "Paris" ✓, TriLM-1.5B coherent ✓.

### v2.8.0 ✅ — Load-Time int4 FFN Quantization (ABGESCHLOSSEN)
- **New AVX2 kernel `atlas_matmul_i4_f32`**: Nibble-unpack + `(nibble^8)-8` sign-extension + `vpmaddubsw` — 64 elements per iteration.
- **`atlas_quantize_ffn_to_i4()`**: Load-time int8→int4 conversion for gate/down/up FFN tensors. Clip to [-8,7], pack 2/byte. Guarded by `use_f32_matmul` for hybrid architecture safety.
- **ttype=8 dispatch**: Gate+up and down projection branches in `forward_layer_internal`, inserted before ternary/f32/fallback dispatch.
- **Lane-permute fix**: `_mm256_unpack*_epi8` per-128-bit-lane behavior corrected via `_mm256_permute2f128_si256`.
- **Python reorder**: `set_use_f32_matmul()` called before `quantize_ffn_to_i4()` to ensure f32_bypass models skip int4 conversion.
- **Target erreicht**: 7B +26% (2.5→3.15 tok/s), 10B +18% (1.9→2.25 tok/s). Alle 6 Modelle (3B/7B/10B/Bonsai-1.7B/Bonsai-4B/BitNet) korrekt.

### v2.4.0 ✅ — Qwen3/Bonsai-Okosystem-Upgrade (ABGESCHLOSSEN)
- **Packer (`atlas_packer_bonsai.py`)**: Tensor-Mapping (Qwen3→ATLAS), Skalierungsfaktor-Extraktion (`max(abs(w))`), Ternarisierung (`round(w/scale)`), 5-Trit-Packing.
- **head_dim=128**: Alle Attention-Pfade (RoPE, Scores, Weighted Sum, KV-Cache) auf variablen head_dim umstellen.
- **QK-Norm**: Zwei neue RMSNorm-Tensoren pro Layer (`q_norm`, `k_norm`) im Attention-Hotpath.
- **Dynamisches Vocab**: `vocab_size` aus Datei-Header statt hardcoded 131072 (Bonsai: 151669). EOS/PAD-IDs aus Header.
- **YaRN RoPE**: Frequenz-Skalierung mit NTK-Approximation für rope_theta=5M, factor=4.0.
- **Tie Word Embeddings**: `lm_head` = `embed_tokens` (shared weights). Int8-Quantisierung des Embedding-Tensors.
- **SwiGLU-Hotpath**: `gate`/`up` parallel berechnet, SiLU fusioniert — identisch zu Falcon3, kein Umbau nötig.
- **Target**: Bonsai-4B TQ1.0 ~1.5 GB. Kompatibilität mit Qwen3 Familie.

### v2.5.0 ✅ — Context Window Extension (ABGESCHLOSSEN)
- **Ring Buffer KV Cache**: Zirkuläres Überschreiben der ältesten `max_seq_len` Positionen. `seq_now` kann `max_seq_len` überschreiten — der Cache wickelt modulo `max_seq_len` und überschreibt die ältesten Einträge.
- **NTK Context Extension**: `ctx_scale = max_seq_len / base_seq_len` kompoundiert mit `rope_scale` für NTK-aware Frequenzanpassung. Erlaubt sanfte Skalierung über die trainierte Kontextlänge hinaus (z.B. 4K→8K für Falcon3, 8K→16K für Bonsai-4B).
- **`set_base_seq_len()` API**: Neue C-API + Python-Methode. `base_seq_len` = trainierte Kontextlänge (z.B. 4096 Falcon3, 2048 Bonsai-1.7B, 8192 Bonsai-4B). NTK-Scaling wird automatisch angewandt wenn `max_seq_len > base_seq_len`.
- **Dynamisches `max_seq_len`**: Pro `atlas_generate`-Aufruf konfigurierbar (Python: `generate_c(..., max_seq_len=8192)`).
- **Kein RAM-Wachstum**: Cache bleibt `n_layers × n_kv_heads × max_seq_len × head_dim` — keine lineare Skalierung mit `seq_now`.
- **Target erreicht**: 8K Kontext auf Falcon3-3B getestet, 16K auf Bonsai-4B getestet. Ring Buffer für 128→200+ Token Wrapping validiert.

### v2.6.0 ✅ — Pipeline: SSE Web-Server + Prompt-Caching (ABGESCHLOSSEN)
- **SSE Web-Server**: `atlas_server.py` — FastAPI/SSE-Wrapper für HTTP-Streaming. `/v1/chat/completions` Endpoint, `StreamingResponse` für token-by-token SSE.
- **Prompt-Caching**: KV-Cache persistiert über `generate_c`-Aufrufe hinweg. `asyncio.Lock()` serialisiert Zugriff. `POST /reset` zum manuellen Cache-Leeren.
- **`atlas_reset_cache()` C-API**: Neue C-Funktion + Python `AtlasModel.reset_cache()`. Zeros KV-Cache-Daten, Allokation bleibt erhalten.
- **CI Pipeline**: `.github/workflows/build.yml` — GitHub Actions Build-Test auf Ubuntu/Windows/macOS mit Clang/LLVM. Automatischer Build bei Push auf main.
- **Target**: 10B Chat-Client mit sub-second Prefill für kurze Folgefragen.

### Deferred
- **F16C-Rester**: Diminishing returns (heiße Pfade bereits erledigt)

## Version History

| Version | Key Changes |
|---------|-------------|
| **v2.10.4** | **BitCPM-CANN-1B Support + Debug Print Crash Fix**: 28L/2048H/6144I/16:2 heads/73448vocab, Llama-Architektur (LongRoPE), MiniCPM v5 Tokenizer. Unconditional `logits[96944]` OOB Read in Debug-Print gefixt (586 Bytes über Allokation bei V=73448). Alle unconditional Debug-Prints entfernt. `ATLAS_DLL` env var fix (überschreibt jetzt immer). 9.7 tok/s auf i7-7700T. |
| **v2.10.3** | **Bug Hunt Round 2+3 + Llama3-Support**: 12 Bugs gefixt (Falcon3 BPE Vocab cutoff, Llama3 stops/specials, unaligned memcpy, rope_interleaved Heuristik, xoshiro_state thread_local, VNNI Init-Race, EOS sentinel=0→None, BitNet-Stops, silent except→warn). v6 added_tokens Support (IDs 128000-128255, longest-first preencode). `rope_interleaved_set` Flag. Llama3-8B-1.58-100B-tokens Base Model (32L/4096H/14336I). 39/39 Tests, 7 Architekturen. |
| **v2.10.2** | **Consistent Naming + HF-Repos**: Falcon3 HF-Modelle umbenannt zu `Falcon3-*-1.58bit-ATLAS`. `matmul_reorder_deq` pre-divide Optimierung. Bonsai-8B in Performance-Tabelle. |
| **v2.10.1** | **BitNet EOS Fix + HF-Push**: `build_tokenizer_binary` Pattern-Matching priority fix. BitNet-2B4T + Bonsai (3 Größen) auf HF Hub deployed. `eos_token_id=128001→128009`. |
| **v2.10.0** | **Unified Packer + BF16 Weight_Scale Fix**: `pack_to_atlas.py` ersetzt alle Einzel-Packer. BF16 weight_scale Fallback. 4 Falcon3 Modelle gepackt. BitNet EOS Token Fix. CLI Build OK. 22/22 Mock-Tests. |
| **v2.9.2** | **Mock-CI-Infrastruktur + Bugfixes**: 3 Bugs gekillt (ttype=1 data_size heuristic, ensure_buffers Q-buffer overflow, _cache_indices BitNet stride). 2 neue C-APIs (atlas_set_rope_interleaved, atlas_set_rope_theta). `tests/atlas_mock_model.py` generiert synthetische v8-Modelle (200-300 KB) für 3 Architekturen (Falcon3/Qwen3/BitNet) mit echtem TQ1-Packing. `tests/test_mock_model.py`: 9 parametrisierte pytest-Tests (load/forward/batch) in 1.14s. EOS-fix aus tokenizer special[0] statt header-guess. Regression: Falcon3-3B/Bonsai-8B/TriLM-1.5B — 3/3 pass. |
| **v2.9.1** | **Hardening-Release**: Windows UTF-8 argv über `CommandLineToArgvW`+`WideCharToMultiByte` — Umlaute/Akzente korrekt. 6 Argument-Guards (NaN/Overflow/Sektor 2). CI/CD Smoke-Test (`tests/test_mock_model.py`). Proaktiver CPUID-AVX2-Check (`check_avx2()`) mit Fehlermeldung statt SIGILL. cross-platform release.yml mit shell32. |
| **v2.9.0** | **Standalone C++ CLI** (`atlas_cli.cpp`): 575 Zeilen, `LoadLibrary`/`dlopen` dynamisches DLL-Binding, interaktiver `/reset`-Modus, Chat-Template-Detection (Falcon3/BitNet/Qwen3), vollständiges Arg-Parsing. `compile.bat` baut jetzt `atlas.dll` + `atlas.exe`. GitHub Auto-Release (`release.yml`) mit Windows/Linux Zip/Tar + LLVM-Runtime-DLLs. README komplett umgeschrieben — Community-Framing. |
| **v2.8.0** | **Load-time int4 FFN quantization (18-26% faster)**: New `atlas_matmul_i4_f32` AVX2 kernel — nibble-unpack + sign-extension via `(nibble^8)-8` + `vpmaddubsw`. `atlas_quantize_ffn_to_i4()` converts int8→int4 at load time, halves FFN memory bandwidth. ttype=8 dispatch in `forward_layer_internal` with `use_f32_matmul` guard for hybrid safety. 7B: 2.5→3.15 tok/s (+26%), 10B: 1.9→2.25 tok/s (+18%). Lane-permute bug fixed: `_mm256_unpack*_epi8` per-128-bit-lane issue patched via `_mm256_permute2f128_si256`. |
| **v2.7.9** | **Fix duplicate attn_sub_norm (BitNet collapse)**: Merge-Artefakt in `forward_layer_internal` — sub-norm wurde zweimal auf Attention-Output angewandt. BitNet-2B4T: `/ / / / /` → `"The capital of France is Paris."`. `data_size`-Formel für ttype=5 korrigiert (`row_dim * n_blocks * 2`). |
| **v2.7.7** | **BitNet Packing Fixes**: Fixed U8 bit ordering (Microsoft I2_S stores row 0 in high bits, was reading from low). Switched BF16 to per-tensor absmean + `weight_scale` loading. Fixed `data_size` header calc for ttype=5. U8 `--packed` path recommended. |
| **v2.7.6** | **BitNet b1.58 Final Fixes**: AGENTS.md dimensions corrected to 30L/2560H/6912I/20/5 heads. ReLU² confirmed correct (Microsoft `hidden_act: "relu2"`). `--packed` flag for U8 pre-quantized Microsoft weights. Python BitNet detection (`_is_bitnet` via attn_sub_norm), correct chat template (`Role: content<|eot_id|>`), correct EOS (128009). Misidentification as Qwen3 fixed. |
|---------|-------------|
| **v2.7.5** | **ttype=5 Decompress + f32_bypass everywhere**: Reverted fused-kernel-only approach. All ttype=5 tensors decompressed to int8 at load. `f32_bypass` forced for block-scaled models (rope_theta≥3M or hidden≤2048) — no uint8+128 activation quant, no signal collapse. Bonsai-8B: 0.2→1.6-2.2 tok/s with perfect T=0.7 coherence. |
| **v2.6.3** | **BitNet ARCH_BITNET + Repo Cleanup**: `atlas_ensure_layer_idx()` C API für Python `forward()`-Pfad. f32_bypass forced für SubLN-Architekturen (u8+128 Activation Quant zerstört ~0.01× SubLN-Signal). 91 Scratch-Dateien gelöscht, tote Packer entfernt, `.gitignore` gehärtet. |
| **v2.6.2** | Safe decompress_ttype5 dispatch (try/except guard) |
| **v2.6.1** | ttype=5 decompress + f32_bypass for all block-scaled models |
| **v2.6.0** | **SSE Web-Server + Prompt-Caching**: `atlas_server.py` — FastAPI/SSE `/v1/chat/completions`, `atlas_reset_cache()` C-API + Python wrapper, `asyncio.Lock()`-serialisierter Cache, `.github/workflows/build.yml` CI Pipeline (Ubuntu/Windows/macOS). |
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
- `atlas_cli.cpp` — Standalone C++ CLI (LoadLibrary/dlopen, Arg-Parsing, Chat-Template, Interactive)
- `atlas_infer.py` — Python `AtlasModel` class with `generate_c()`
- `atlas_ffi.h` — C API declarations (v6 header layout)
- `pack_to_atlas.py` — **Unified packer** (v2.10.0): auto-detects architecture from config.json, single pipeline for Falcon3/Qwen3/BitNet/TriLM/Llama. Deprecates individual packers.
- `atlas_packer_mappings.py` — Architecture definitions used by `pack_to_atlas.py` (tensor name patterns, quantization rules, flags per arch).
- `atlas_packer.py` — [DEPRECATED] v5/v6 format writer for Falcon3 models. Use `pack_to_atlas.py`.
- `atlas_packer_g128.py` — [DEPRECATED] Block-scaled g128 packer for Bonsai/Qwen3. Use `pack_to_atlas.py`.
- `atlas_packer_bitnet.py` — [DEPRECATED] BitNet b1.58 packer. Use `pack_to_atlas.py`.
- `atlas_server.py` — FastAPI SSE Web-Server mit Prompt-Caching (v2.6.0)
- `scripts/release_to_hf.py` — Deployment-Wrapper: packt Modell via `pack_to_atlas.py` + optionaler HF-Hub-Upload (`--push`). Generiert YAML-Frontmatter (base_model, license, tags per Architektur-Familie). Keine Netzwerk-Abhängigkeit im Kern-Packer.
- `add_v6_block.py` — Append v6 binary tokenizer block to existing v5 files
- `compile.bat` — Windows Build-Script (DLL + optional CLI)
- `tests/atlas_mock_model.py` — Minimales Mock-Modell für CI-Smoke-Tests (7 Architekturen, TQ1-Packing)
- `tests/test_mock_model.py` — CI Smoke-Test (39 parametrisierte Tests)
- `.github/workflows/build.yml` — CI Pipeline: Build-Test auf Ubuntu/Windows/macOS
- `.github/workflows/release.yml` — Auto-Release bei v*-Tag (Windows/Linux Zip/Tar)

## Technical Details

- **v5 format**: `[header:64] [dir:n*12] [name_block] [token_data...] [tokenizer_block]`. Header bytes 29-32: tokenizer_size, 33-36: tokenizer_offset.
- **v6 format**: v5 + binary tokenizer block (128-byte header, offsets/lengths/pool, BPE merges, byte_encoder, special tokens).
- **v6 added_tokens**: Up to 256 extra tokens (IDs ≥ V) stored in binary tokenizer block. Encoded as `(offs[10], offs[11], offs[12], len_specials)` in 128-byte header. `offs[10]` = special_pool_offset, `offs[11]` = special_pool_len, `offs[12]` = special_map_offset. Preencode scans longest-first via `memcmp`. Decode looks up by ID.
- **Chat template**: `<|role|>\n{content}\n` — NO `<|im_end|>` tokens. Generation prompt: `<|assistant|>\n`.
  BitNet: `{Role}: {content}<|eot_id|>\n` — generation prompt: `Assistant: `. EOS token `<|eot_id|>` = 128009.
  Llama3 (Base Model): No chat template. Generation: `{prompt}`. Stopp-Tokens: `<|eot_id|>`, `<|start_header_id|>`, `<|end_header_id|>`.
- **Sampling overhead**: 1B top_k=40+p: ~3 tok/s (survivor-list makes top_p ≈ free after top_k).
- **Prefill**: All prompt tokens processed in single batched `atlas_forward` call (B=prompt_len).
- **Cache**: `.i8` cache auto-generated on first full int8 decompress, mmap'd on reload.
