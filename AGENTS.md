# ATLAS — TQ1.0 Ternary Inference Engine

**Status**: v2.17.0, Juli 2026.
**Repo**: `xxxn3m3s1sxxx/ATLAS-TQ1_0` auf GitHub.

MLA (Multi-head Latent Attention) und MoE (Mixture-of-Experts) für DeepSeek-V2 ab v2.17.0. TQ1.0-Format und Packer bleiben das Kernartefakt.

## Schnellstart

```bash
compile.bat                # Release: atlas.dll + atlas.exe (Windows, clang++)
compile.bat debug          # Debug: atlas_d.dll (-O0 -g)
compile.bat test           # Release + fixture-Tests
compile.bat coverage       # Coverage-Build + Tests + HTML-Report

# Linux
clang++ -fopenmp -O2 -mavx2 -mfma -mf16c -ffast-math -std=c++17 -fPIC -shared -o libatlas.so atlas_api.cpp atlas_vnni.cpp -lgomp

# macOS ARM64 — Homebrew LLVM (nicht Apple Clang)
brew install llvm libomp
$LLVM_PREFIX/bin/clang++ -fopenmp -O2 -march=armv8.2-a+dotprod -ffast-math -std=c++17 -fPIC -shared -D__aarch64__ -o libatlas.so atlas_api.cpp atlas_kernel_arm64.cpp -L$LLVM_PREFIX/lib -lomp
```

## Tests

```bash
pytest tests/test_mock_model.py          # Schnelle CI-Smokes (kein Modell nötig)
pytest tests/test_fixtures.py            # Golden-Regression mit echten Modellen
pytest tests/test_fixtures.py -k "golden" # Nur Golden-Regenerierung
pytest tests/test_e2e_pipeline.py -v     # Prompt-Caching + Streaming
pytest tests/test_omp_stress.py -v       # OMP-Thread-Stress
```

Fixture-Tests brauchen eine `atlas.dll` im Repo-Root. Mock-Tests generieren synthetische `.atlas`-Modelle (200-300 KB) automatisch in `mock/`. `KMP_DUPLICATE_LIB_OK=TRUE` auf Windows setzen für MKL-Kompatibilität.

## File Layout

| File | Role |
|------|------|
| `atlas_api.cpp` | Engine: AVX2-Kernels, Attention, MLA, MoE, RoPE, Sampling, Generate-Loop |
| `atlas_kernel_arm64.cpp` | NEON-Äquivalente (8 Kernels, v2.16.1) |
| `atlas_vnni.cpp` | VNNI-Kernel (x86, AVX-VNNI falls verfügbar) |
| `atlas_cli.cpp` | Standalone-CLI (LoadLibrary/dlopen, Chat-Template) |
| `atlas_infer.py` | Python `AtlasModel`-Klasse mit `generate_c()` |
| `atlas_server.py` | FastAPI/SSE `/v1/chat/completions` mit Prompt-Caching |
| `atlas_ffi.h` | C-API-Contract + v6-Header-Layout |
| `pack_to_atlas.py` | Unified Packer: HF safetensors → TQ1.0 (auto-detect) |
| `atlas_packer_mappings.py` | Architektur-Definitionen pro Modell-Familie |
| `compile.bat` | Windows-Build |
| `compile-linux.sh` | Linux-Build |
| `scripts/release_to_hf.py` | Pack + optional HF-Upload |

## Critical Gotchas

- **f32_bypass**: Auto-enabled für `hidden <= 2048`, `rope_theta >= 3M`, oder `ARCH_BITNET`. Hybrid/int4-Pfad erzeugt "achuachuachu"-Garbage bei rope_theta >= 3M (Bonsai-4B/8B, CANN 3B/8B). Bei Bonsai-8B (`rope_theta=1M` < 3M) nicht auto-triggered → `AtlasModel(..., use_f32_matmul=True)` setzen.
- **`ATLAS_DLL` env**: Überschreibt den DLL-Pfad. `$env:ATLAS_DLL="C:\atlas\atlas_d.dll"` für Debug-Switching.
- **int4 FFN**: Nur für memory-bandwidth-gebundene Modelle (7B +26%, 10B +18%). Für compute-gebundene (Bonsai-4B) ist int8 optimal. Automatisch geskippt für f32_bypass-Modelle.
- **Chat Templates pro Arch**:
  - Falcon3: `<|role|>\n{content}\n`, Generation: `<|assistant|>\n`
  - Qwen3/Bonsai/CANN: ChatML (`<|im_start|>role\n...<|im_end|>\n`)
  - BitNet: `{Role}: {content}<|eot_id|>\n`, EOS=128009
  - Llama3 Instruct: `<|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\n{content}<|eot_id|>`
  - Llama3 Base: Kein Template anwenden!
  - DeepSeek-V2: ChatML (identisch zu Qwen3/CANN) + MLA compressed KV cache
- **CANN dual EOS**: Sowohl `</s>` (ID 2) als auch `<|im_end|>` (ID 73440) terminieren.
- **C++ binary tokenizer**: v6-Format, kein transformers-Dependency zur Runtime. `tokenizers`-Lib nur für Python-seitiges Encoding nötig.
- **ARM64 Port**: `atlas_kernel_arm64.cpp` wird bei x86-Build nicht kompiliert. Build mit `-D__aarch64__`. `__arm64__` statt `__aarch64__` unter Homebrew LLVM → die Portabilitäts-Block mapped auch `_M_ARM64`.
- **KV Cache Reset**: `model.reset_cache()` nach Kontext-Wechsel. Ring-Buffer überschreibt älteste Positionen bei `seq_now > max_seq_len`.
- **MLA layer_stride**: Dynamisch: `6 + 3*n_shared + 2` — indices: [0]ln1 [1]q [2]kv_a [3]kv_b [4]o [5]ln2 [6..6+3*n_shared-1]shared_experts [6+3*n_shared]kv_a_layernorm [6+3*n_shared+1]router
- **MLA compressed KV**: DeepSeek-V2 nutzt komprimierte KV-Cache (kv_lora_rank + qk_rope_head_dim) pro Position. `compressed_kv_stride = kv_lora_rank + qk_rope_head_dim`.
- **Shared Expert Names**: C++ nutzt `mlp.shared_experts.gate_proj.weight` (Plural, Dot-separated) — nicht `shared_expert`.
- **Buffer Aliasing MoE**: `tmp_down = buf_hidden + b*H` — darf nicht `output` oder `buf_gate` aliasen (wird bei MoE-Dispatch überschrieben).
- **moe_expert_tidx**: Flat vector `[layer * n_experts * 3 + expert * 3 + proj]` → tensor index. proj: 0=gate, 1=up, 2=down. Populated in `atlas_repack_experts()`.
- **Mock Model FFN Dimensions**: `_shape_of()` muss für MLA-Architekturen `moe_intermediate_size` statt `cfg["inter"]` für alle FFN-Projektionen verwenden (dense + MoE). Die C++ Engine setzt `inter_dim = moe_intermediate_size` (JSON override in `atlas_load`). Wenn Mock `gate_proj.row_dim = cfg["inter"]` nutzt, überschreibt `matmul_f32_reorder` den `buf_gate`-Buffer → Access Violation.
- **`atlas_load_cache` mmap_base**: `atlas_load_cache` (line ~2094) speichert `mmap_base`/`mmap_handle`/`mmap_file`/`mmap_size` UNCONDITIONAL vor `if (replaced > 0)`. Bei Cache-Miss bleibt `mmap_base` gesetzt → `atlas_save_cache` scheitert an Sharing-Violation (nur `FILE_SHARE_READ`).
- **Kein alloca/VLA auf dem heißen Pfad**: `alloca` und VLAs sind auf dem Inferenzpfad (`atlas_attention_mla`, `atlas_moe_forward`, `forward_layer_internal_mla`) strikt verboten — `alloca` gibt Speicher erst beim Verlassen der Funktion frei, nicht pro Schleifeniteration. Bei32K Kontext → Stack Overflow. Al temporärer Workspace muss über pre-kalkulierte Heap-Offsets (`attn_ws`) alloziert und recycelt werden.
- **attn_ws Layout (MLA)**: `attn_ws` dient als wiederverwendbares Ring-Scratchpad für die gesamte MLA-Inferenz:
  ```
  [0, B·qd)              q_full
  [B·qd, 2·B·qd)         attn_out
  [2·B·qd, +B·rope)       k_pe_all (→ danach x_safe reusen)
  [+, +kv_out)             kv_out (per-position)
  [+, +lora)               c_kv_f32 (per-position)
  [+, +rope)               k_pe_f32 (per-position)
  ```
  `x_safe` (MoE) reusert `attn_ws[0..B·H)` — q_full ist tot nach Attention. Lebenszyklen überlappen zu keinem Zeitpunkt.

## Skills (in .opencode/skills/)

- `atlas-build`: Build-Prozess (Release/Debug/Coverage/ARM64)
- `atlas-kernel`: AVX2-Matmul, NEON, Kernel-Entwicklung
- `atlas-hashline`: Hash-anchored Editing für atlas_api.cpp
- `atlas-benchmark`: Performance-Messung (tok/s, Ladezeit)

## TQ1.0 Format Summary

- 5 ternäre Trits pro Byte (Base-3), ~1.58 bits/weight
- v6 Header: 64 Bytes, gefolgt von Tensor-Directory, Name-Block, Tensor-Daten, Binary-Tokenizer
- ttype=0: TQ1 packed, ttype=3: int8 dekomprimiert, ttype=5: TQ1 g128 block-scaled, ttype=8: int4 packed, ttype=11: int8 row-scaled, ttype=10: TQ2 universal
- `.i8` Cache: Auto-generiert bei erstem int8-Decompress, mmap'd beim Reload
