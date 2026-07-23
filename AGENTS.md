# ATLAS — TQ1.0 Ternary Inference Engine

**Status**: v2.17.0, Juli 2026.
**Repo**: `xxxn3m3s1sxxx/ATLAS-TQ1_0` auf GitHub.

Keine Weiterentwicklung. Das TQ1.0-Format und der Packer sind das eigentliche Artefakt — die Engine ist durch DDR4-Bandbreite physikalisch begrenzt (~20 GB/s → ~3 tok/s für 7B).

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
| `atlas_api.cpp` | Engine: AVX2-Kernels, Attention, RoPE, Sampling, Generate-Loop |
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
- **CANN dual EOS**: Sowohl `</s>` (ID 2) als auch `<|im_end|>` (ID 73440) terminieren.
- **C++ binary tokenizer**: v6-Format, kein transformers-Dependency zur Runtime. `tokenizers`-Lib nur für Python-seitiges Encoding nötig.
- **ARM64 Port**: `atlas_kernel_arm64.cpp` wird bei x86-Build nicht kompiliert. Build mit `-D__aarch64__`. `__arm64__` statt `__aarch64__` unter Homebrew LLVM → die Portabilitäts-Block mapped auch `_M_ARM64`.
- **KV Cache Reset**: `model.reset_cache()` nach Kontext-Wechsel. Ring-Buffer überschreibt älteste Positionen bei `seq_now > max_seq_len`.

## Skills (in .opencode/skills/)

- `atlas-build`: Build-Prozess (Release/Debug/Coverage/ARM64)
- `atlas-kernel`: AVX2-Matmul, NEON, Kernel-Entwicklung
- `atlas-hashline`: Hash-anchored Editing für atlas_api.cpp
- `atlas-benchmark`: Performance-Messung (tok/s, Ladezeit)

## TQ1.0 Format Summary

- 5 ternäre Trits pro Byte (Base-3), ~1.58 bits/weight
- v6 Header: 64 Bytes, gefolgt von Tensor-Directory, Name-Block, Tensor-Daten, Binary-Tokenizer
- ttype=0: TQ1 packed, ttype=3: int8 dekomprimiert, ttype=5: TQ1 g128 block-scaled, ttype=8: int4 packed, ttype=10: TQ2 universal
- `.i8` Cache: Auto-generiert bei erstem int8-Decompress, mmap'd beim Reload
