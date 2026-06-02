<p align="center">
  <img src="atlas_banner.svg" alt="ATLAS Banner" width="100%">
</p>

# ATLAS — Run 10B LLMs on a 8GB RAM Laptop

**No GPU needed.** ATLAS runs Falcon3, Bonsai/Qwen3, and BitNet b1.58 models on CPU using ternary-quantized **TQ1.0** format (~1.58 bits/weight). Run a 3B model on a 4 GB laptop, or a 10B model on 8 GB RAM — all at 2-17 tokens/second on CPU.

> 🚀 **v2.10.0 — Falcon3 + Bonsai + BitNet TQ1.0 Series**: All 4 Falcon3 models (1B/3B/7B/10B), 3 Bonsai models (1.7B/4B/8B), and **BitNet b1.58-2B-4T** packaged, verified T=0 correct, and deployed to HuggingFace. Unified pack_to_atlas.py with architecture auto-detection replaces all individual packers. BF16 weight_scale fix. BitNet EOS token fix (128009 <|eot_id|>). 35/35 mock tests, CI green on Windows + Linux.

## Quickstart

**Download** the latest release zip and unzip into one folder:

```
atlas.exe   ← CLI binary
atlas.dll   ← engine DLL
libomp.dll  ← OpenMP runtime
```

**Run a model:**

```bash
# Download a pre-packed model from Hugging Face (see "Model Sources" below):
# e.g. https://huggingface.co/xxxn3m3s1sxxx/Falcon3-3B-Instruct-ATLAS
#      https://huggingface.co/xxxn3m3s1sxxx/Ternary-Bonsai-4B-ATLAS

# Chat with it:
atlas.exe falcon3-3B-Instruct-tq1.atlas "What is the capital of France?"
# → "The capital of France is Paris."

# Interactive chat:
atlas.exe falcon3-3B-Instruct-tq1.atlas -i

# Adjust sampling:
atlas.exe model.atlas "Tell me a story" --temp 0.9 --max-new 500
```

**Requirements:** Windows x86-64, AVX2 CPU (Intel Haswell 2013+ / AMD Excavator 2015+), 4-8 GB RAM. Works without Python.

## Supported Models

| Model | Download Size | RAM Needed | Layers | Heads | Best tok/s |
|-------|:------------:|:----------:|:------:|:-----:|:----------:|
| Falcon3-3B-Instruct | **2.0 GB** | **4 GB** | 22 | 12 | **7.1** |
| Falcon3-1B-Instruct | 1.2 GB | 3 GB | 18 | 8 | **10.1** |
| Falcon3-7B-Instruct | 2.8 GB | 6 GB | 28 | 12 | **3.15** |
| Falcon3-10B-Instruct | 3.3 GB | 8 GB | 40 | 12 | **2.25** |
| **Bonsai (Qwen3)** |||||
| Bonsai-1.7B | 0.9 GB | 3 GB | 28 | 16 | **13.0** |
| Bonsai-4B | 1.5 GB | 5 GB | 36 | 32 | **17.4** |
| Bonsai-8B | 2.5 GB | 8 GB | 36 | 32 | **1.8** |
| **TriLM** |||||
| TriLM-830M | 0.4 GB | 2 GB | 24 | 28 | — |
| TriLM-1.1B | 0.5 GB | 3 GB | 24 | 28 | — |
| TriLM-1.5B | 0.7 GB | 3 GB | 24 | 32 | — |
| TriLM-2.4B | 0.9 GB | 4 GB | 30 | 36 | — |
| TriLM-3.9B ⚠️ | 1.2 GB | 5 GB | 32 | 64 | — |
| **Other** |||||
| BitNet-2B4T-b1.58 | 1.0 GB | 4 GB | 30 | 20 | **2.8** |

"Best tok/s" on Intel Core i7-7700T (4C/8T @ 2.9 GHz). Your speed depends on CPU.
- **3B+ models** work well on 4-8 GB RAM laptops
- **7B/10B models** need 6-8 GB RAM (faster with int4 FFN mode)
- **1B models** are fast but less capable
- **TriLM 3.9B ⚠️**: Uses **standard Llama** architecture (head_dim=128, no SubLN) — different from smaller TriLMs (SubLN, head_dim=64). Auto-detected.

### Model Sources

| Family | HuggingFace Repo | License | Status |
|--------|-----------------|:-------:|:------:|
| **Falcon3** (1B/3B/7B/10B) | [`tiiuae/Falcon3-*-Instruct`](https://huggingface.co/tiiuae) | TII Falcon License 2.0 | ✅ |
| **Falcon3 ATLAS** (1B/3B/7B/10B) | [`xxxn3m3s1sxxx/Falcon3-*-Instruct-ATLAS`](https://huggingface.co/models?search=xxxn3m3s1sxxx/Falcon3) | TII Falcon License 2.0 | ✅ |
| **Bonsai** (1.7B/4B/8B) | [`prism-ml/Ternary-Bonsai-*-unpacked`](https://huggingface.co/prism-ml) | Apache 2.0 | ✅ |
| **Bonsai ATLAS** (1.7B/4B/8B) | [`xxxn3m3s1sxxx/Ternary-Bonsai-*-ATLAS`](https://huggingface.co/models?search=xxxn3m3s1sxxx/Ternary-Bonsai) | Apache 2.0 | ✅ |
| **BitNet b1.58** (2B) | [`microsoft/bitnet-b1.58-2B-4T`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) | MIT | ✅ |
| **BitNet b1.58 ATLAS** (2B) | [`xxxn3m3s1sxxx/BitNet-2B4T-b1.58-ATLAS`](https://huggingface.co/xxxn3m3s1sxxx/BitNet-2B4T-b1.58-ATLAS) | MIT | :white_check_mark: |
| **TriLM** (99M→3.9B, 10 sizes) | [`SpectraSuite`](https://huggingface.co/SpectraSuite) | Apache 2.0 | ✅ |
| Falcon-Edge (1B/3B) | [`tiiuae`](https://huggingface.co/collections/tiiuae/falcon-edge-series-6804fd13344d6d8a8fa71130) | TII Falcon License 2.0 | 🚧 |
| OLMo-BitNet-1B | [`NousResearch/OLMo-Bitnet-1B`](https://huggingface.co/NousResearch/OLMo-Bitnet-1B) | Apache 2.0 | 🚧 |
| Llama3-8B-1.58 | [`HF1BitLLM/Llama3-8B-1.58-100B-tokens`](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens) | Llama 3 | 🚧 |

✅ — Supported and tested. 🚧 — Experimental, needs packer work.

**TriLM ⚠️**: TriLM 3.9B uses **standard Llama** (head_dim=128, no SubLN). Smaller TriLMs (≤2.4B) use SubLN (head_dim=64). Auto-detected by `derive_arch`.

### Converting Models to .atlas Format

```bash
# Requires Python + transformers:
pip install safetensors transformers numpy

# All architectures auto-detected (Falcon3, Bonsai, Qwen3, BitNet, TriLM, Llama):
python pack_to_atlas.py path/to/model-directory
```

The packer autodetects the model and generates a `.atlas` file ready for the CLI.

## CLI Usage

```bash
atlas.exe <model.atlas> [prompt] [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--temp <f>` | 0.7 | Temperature (0 = deterministic) |
| `--top-k <n>` | 40 | Top-k sampling (0 = off) |
| `--top-p <f>` | 0.9 | Top-p nucleus sampling |
| `--max-new <n>` | 200 | Max tokens to generate |
| `--max-seq <n>` | 4096 | Context window size |
| `--rep-penalty <f>` | 1.0 | Repetition penalty (1.0=off) |
| `--seed <n>` | random | RNG seed for reproducible output |
| `--threads <n>` | auto | CPU threads to use |
| `--raw` | off | Send prompt as-is (no chat template) |
| `-i` | off | Interactive chat mode |

**Interactive mode** (`-i`) supports:
- Type messages and get responses (KV cache persists across turns)
- `/reset` — clear conversation context
- `/exit` or `/quit` — exit

## Performance Tips

| Setting | When to Use |
|---------|-------------|
| `--temp 0.0` | Factual QA, deterministic answers |
| `--temp 0.7` + `--top-k 40` | Creative writing, conversation |
| `--threads 4` | On battery, or shared CPU |
| `--threads <P-cores>` | **Hybrid CPUs** (Alder Lake+): set to physical P-core count. E-cores slow synchronized matmul barriers. |
| `--max-seq 2048` | Lower RAM usage (shorter context) |

- **7B+ models**: Use `--max-new 100` for faster responses
- **1B/3B models**: Good at `--max-new 200-500`
- **T=0 (deterministic)**: Best for 7B+ models. Smaller models may repeat or collapse at T=0 — use T=0.7 instead.

## Build from Source

**Windows (clang):**
```bash
compile.bat
```

**Linux:**
```bash
chmod +x compile-linux.sh && ./compile-linux.sh
```

Requires Clang with OpenMP, AVX2+FMA, C++17.

The CLI (`atlas.exe`) is optional — it's built by `compile.bat` but only the DLL is required for Python/API usage.

## Performance (Technical)

Measured on Intel Core i7-7700T (4C/8T @ 2.9 GHz). Warm cache, `generate_c()` at T=0.7 / top_k=40.

| Model | Mode | tok/s | Notes |
|-------|------|:-----:|-------|
| Bonsai-1.7B | f32 bypass | **13.0** | Fastest model, good quality |
| Bonsai-1.7B | hybrid+int8 | 19.2 | Higher throughput, minor quant noise |
| Bonsai-4B | hybrid+int8 | **17.4** | Best perf/size tradeoff |
| Falcon3-3B | hybrid+int8 | **7.1** | Good general-purpose model |
| Falcon3-1B | f32 bypass | **10.1** | Fast but shallow |
| Falcon3-7B (int4) | hybrid+int8 | **3.15** | 26% faster than int8 |
| Falcon3-10B (int4) | hybrid+int8 | **2.25** | 18% faster than int8 |
| Bonsai-8B | f32 bypass | **1.8** | Most capable, slowest |
| BitNet-2B4T | f32 bypass | **2.8** | Experimental |
| TriLM-830M | f32 bypass | — | SubLN, head_dim=64 |
| TriLM-1.1B | f32 bypass | — | SubLN, head_dim=64 |
| TriLM-1.5B | f32 bypass | — | SubLN, head_dim=64 |
| TriLM-2.4B | f32 bypass | — | SubLN, head_dim=64 |
| TriLM-3.9B ⚠️ | hybrid+int8 | — | Standard Llama, no SubLN |

### How It Works

1. **TQ1.0 format**: Repacks HuggingFace safetensors into 5 ternary trits per byte (Base-3 encoding). ~1.58 bits/weight.
2. **Hybrid mode** (default): FFN projections run as decompressed int8 (dominate compute), QKV/O stay TQ1-packed (5× less memory reads).
3. **Int8 KV-cache**: Per-position int8 quantization with dynamic scaling halves KV cache RAM. 10B@4K: 320 MB → 173 MB.
4. **Int4 FFN**: Load-time conversion halves FFN memory bandwidth (7B: +26%). Auto-enabled.
5. **C++ binary tokenizer**: No Python dependencies at runtime. Encode via preencode + BPE merge, decode via pool lookup.

### Matmul Modes

| Mode | How | When |
|------|-----|------|
| int8 (default) | `vpmaddubs` SIMD | Best general speed |
| int4 (v2.8.0) | nibble-unpack + `vpmaddubsw` | 7B/10B FFN (18-26% faster) |
| f32 bypass | `vfmadd231ps` | Small models (1B, Bonsai-1.7B) |
| TQ1-packed | chunked decode + SIMD | Models with no decompress |

## Python API

```python
from atlas_infer import AtlasModel

model = AtlasModel("falcon3-3B-Instruct-tq1.atlas")
response = model.generate_c("What is the capital of France?")
print(response)
```

| Method | Description |
|--------|-------------|
| `AtlasModel(path)` | Load model |
| `generate_c(text)` | Generate text (returns string) |
| `generate_stream(text)` | Streaming generator (yields token IDs) |
| `set_system_prompt(text)` | Set system prompt |
| `set_seed(seed)` | Set RNG seed |
| `set_num_threads(n)` | Set CPU thread count |
| `reset_cache()` | Clear conversation context |

## Architecture

```
safetensors → atlas_packer*.py → .atlas file → atlas.exe / atlas_infer.py
                                                      |
                                                 atlas.dll / libatlas.so
                                                      |
                                            atlas_forward (fused layers)
                                                      |
                                            +----+----+----+----+
                                          Norm  Attn  FFN  LM Head
```

### Files

| File | Purpose |
|------|---------|
| `atlas_cli.cpp` | Standalone CLI (`atlas.exe` — no Python needed) |
| `atlas_api.cpp` | C++ inference engine — AVX2 kernels, attention, norms, tokenizer |
| `atlas_vnni.cpp` | AVX-512 VNNI matmul kernel (separate TU, `target("avx10.2")`) |
| `atlas_ffi.h` | C API contract |
| `atlas_infer.py` | Python bindings (`AtlasModel` class) |
| `atlas_server.py` | SSE web server (FastAPI, `/v1/chat/completions`) |
| `atlas_packer*.py` | Model converters (safetensors → .atlas) |
| `compile.bat` | Windows build script |
| `scripts/check_api_parity.py` | API parity scanner (C → Python) |
| `scripts/check_coverage_thresholds.py` | Coverage gates in CI |
| `tests/test_fuzz.py` | Fuzz tests (head_dim edge-cases) |
| `tests/test_e2e_pipeline.py` | End-to-end pipeline tests |
| `tests/test_omp_stress.py` | OMP stress tests |
| `tests/test_bonsai8b_tq2.py` | Bonsai-8B TQ2 format tests |
| `tests/generate_test_fixtures.py` | Test fixture generator |
| `BUGS.md` | Known issues & limitations |
| `.github/workflows/` | CI + auto-release pipelines |

## Version History

| Version | What's New |
|---------|------------|
| **v2.10.0** | **Unified Packer + BF16 weight_scale Fix + Falcon3 TQ1.0 Series**. Single `pack_to_atlas.py` replaces all individual packers — architecture auto-detection from `config.json`. BF16 weight_scale fix (`get_bf16_manual` fallback). All 4 Falcon3 models (1B/3B/7B/10B) packaged as TQ1.0, verified T=0 correct, deployed to Hugging Face. 22/22 mock tests, CI green. |
| **v2.9.3** | AKI-Bug-Fix + HF alignment (16/18 PASS) + TriLM blindspot. TQ2 P2: OMP scale-decode, batch stores, 2× unrolled matmul (+91%: 0.58→1.11 tok/s). AVX-512 VNNI kernel via `atlas_vnni.cpp` with CPUID dispatch + clang 19+ guard.
| **v2.9.2** | Synthetic mock CI suite (9 tests, 3 archs, 1.14s). Bugfix: ttype=1 data_size, EOS fallback, Q-buffer overflow, BitNet stride. New APIs: set_rope_interleaved, set_rope_theta. TriLM-1.5B/TriLM-2.4B support. |
| **v2.8.0** | int4 FFN quantization (7B +26%, 10B +18%). CLI binary. |
| **v2.7.9** | BitNet fix: duplicate sub-norm collapse resolved. |
| **v2.7.5** | ttype=5 decompress. Bonsai-8B: 0.2→1.8 tok/s. |
| **v2.6.0** | SSE web server, prompt caching, CI pipeline. |
| **v2.5.0** | Ring buffer KV cache, NTK context extension (8K-16K). |
| **v2.4.0** | Bonsai/Qwen3 support, QK-Norm, YaRN RoPE, dynamic vocab. |
| **v2.3.0** | Int8 KV-cache (10B@4K: 320→173 MB). |
| **v2.2.0** | F16C SIMD, TQ1-LUT decompress (~30% faster). |
| **v2.1.0** | Streaming generation, repetition penalty. |
| **v2.0.0** | C++ binary tokenizer (no transformers dependency). |
| **v1.0.0** | Initial TQ1.0 inference engine. |

## License

Code: Apache 2.0. Falcon3: TII Falcon License 2.0 (derivative TQ1.0 models at [`xxxn3m3s1sxxx/Falcon3-*-Instruct-ATLAS`](https://huggingface.co/xxxn3m3s1sxxx) carry the same TII Falcon License 2.0). Bonsai/Qwen3: PrismML (Apache 2.0) — derivative ATLAS models at [`xxxn3m3s1sxxx/Ternary-Bonsai-*-ATLAS`](https://huggingface.co/models?search=xxxn3m3s1sxxx/Ternary-Bonsai). BitNet b1.58: Microsoft (MIT). TriLM: SpectraSuite (Apache 2.0).
