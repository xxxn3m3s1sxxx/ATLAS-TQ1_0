# ATLAS — Falcon3 TQ1.0 Inference Engine

## Goal
- TQ1.0 (Base-3, 5 ternary trits/byte) CPU inference engine for Falcon3 models on i5, end-to-end text generation via C++ DLL/SO + Python (Windows + Linux x86-64).

## Constraints & Preferences
- Windows 11 / Linux, 8-16 GB RAM, no GPU. i5-1235U (Alder Lake, 8 OMP threads, AVX2+FMA).
- Windows: `clang++ -fopenmp -O2 -mavx2 -mfma -mf16c -ffast-math -std=c++17` + `libomp.dll`.
- Linux: GCC or Clang, `-fPIC`, `libomp-dev` or `libgomp`.
- `KMP_DUPLICATE_LIB_OK=TRUE` for MKL compat (Windows only).
- **KOHÄRENZ MUSS IMMER ERHALTEN BLEIBEN (0.999999)**.
- All models: `head_dim=256`, `rope_theta=1000042`, GQA. 1B/3B/10B `vocab_size=131072`, 7B `131080`.
- 10B: 40L, 3072H, 23040I, 12/4 heads. 7B: 28L, 3072H, 23040I. 3B: 22L, 3072H, 9216I. 1B: 18L, 2048H, 8192I, 8/4 heads.

## Progress
### Done
- **v5 format with embedded tokenizer**: All 4 models repacked as `.atlas` v5 with tokenizer.json + tokenizer_config.json embedded. `generate()` no longer needs `model_dir`. C API `atlas_get_tokenizer()` exposes raw bytes.
- **Bug 8.6 fixed (Cache Short-Write Protection)**: `atlas_save_cache` checks disk space via `GetDiskFreeSpaceExA`, uses `setvbuf(IONBF)` unbuffered writes, retries on short writes, deletes corrupt partial cache on any failure. `atlas_load_cache` validates file size against header offsets before mapping.
- **f32 matmul bypass added**: `atlas_set_use_f32_matmul(AtlasModel*, int)` — skips activation quantization, uses direct AVX2 f32x i8 FMA. Enabled for `hidden <= 2048` (1B model). `matmul_f32_reorder` + f32 gate+up FFN kernel.
- **1B coherence analyzed**: Greedy degenerates (`,` p=0.43) due to model-inherent distribution. Sampling (`T=1.0, top_k=40, top_p=0.9`) produces correct output. Engine exact — f32 bypass produces identical argmax.
- **Bug 9 revisited (Ping-Pong Buffer)**: Pointer semantics correctly analyzed — even `n_layers` means `buf_a == hidden_states` after loop (correct). The `if (n_layers % 2 == 1)` copy is correct. No engine bug.
- **False Alarm corr=0.23**: Two bugs in Python test reference (not engine): `_rmsnorm` in-place residual corruption + shared quantization gap. Fixed reference gives **corr=0.9967**, max_diff=4.0.
- **v1.0.9 Memory Audit — 4 bugs fixed**: AllocHdr fix (Linux munmap leak), is_mapped guard before vfree (mmap double-free), proper mmap_size tracking (Windows), Linux fd close.
- **v1.1.0 Production Hardening**: AllocHdr-based valloc/vfree, is_mapped guards, fd close on Linux, int8 quant clip fix in Python.
- **v1.2.0 (C++ Sampling + generate)**: `atlas_set_seed` (Xoshiro256**), `atlas_sample` (Gumbel-max top-k/top-p, O(V)), `atlas_generate` (single C-call decode loop with embed lookup + rmsnorm + lm_head GEMV + sample + EOS). Python `generate_c()` wraps it — 1 FFI call per generation instead of per-token. Cached layer index array in `AtlasModel::layer_idx_cache`.
- **v1.2.1 Patch Release**: `AtlasModel.set_seed()` as clean Python method. Version bump. Released as GitHub latest.
- **v1.3.0 — Ternary-Add Kernel**: `matmul_ternary_add_reorder` with `_mm256_sign_epi8` — pure sign-based ternary dot product. Eliminates 128*row_sum correction. Fused FFN ternary gate+up path. `atlas_set_use_ternary_matmul()` C API, `AtlasModel.set_use_ternary_matmul()` Python. Identical argmax to int8 path (10B T=0). "The capital of France is Paris." ✅.
- **v1.3.1 — Direct TQ1-Packed Matmul + num_threads**: `matmul_tq1_packed_reorder` with chunked decode + `vpmaddubs` SIMD int8 matmul. Pre-computed Base-3 LUT (`tq1_decode[256][5]`, 1280 bytes, L1-resident). Per-thread decode buffer via `malloc`/`free` inside OMP. Correct 128*row_sum activation bias correction verified. `atlas_set_use_packed_matmul()` C API. `atlas_set_num_threads()` C API + `AtlasModel.set_num_threads(n)` Python. T=0 coherence: packed=Paris, int8=Paris ✅. Builds clean with `-Wall -Wextra -Wpedantic` (0 warnings). **Pushed as v1.3.1 release on GitHub.**

### In Progress
- **v1.3.2 — Mixed-Precision**: ttype=4 for int4 protected layers (down_proj, first 2-3 FFN).
- **v2.0.0 — C++ BPE Tokenizer**: Python-autarkic. Single FFI call, no PreTrainedTokenizerFast dependency.

### Fixed
- **Bug 11 [10B tokenizer_offset int32 overflow]**: `int` (signed 32-bit) fur `tokenizer_offset` overflowt bei >2 GB Dateigroese. 10B Offset bei ~3.3 GB -> negativ als int32. Fix: `uint32_t` + `ptrdiff_t` cast. Einziger Bug in v1.2.0-pre (behoben in v1.2.0).

## Key Decisions
- **v5 format**: `[header:64] [dir:n*12] [name_block] [token_data...] [tokenizer_block]`. Tokenizer stored as separate raw JSON block (no merge — avoids tokenizers Rust parser corruption). Header bytes 29-32: tokenizer_size, 33-36: tokenizer_offset.
- **f32 bypass bleibt drin**: Eliminates engine quantization noise. Serves as numerical reference path.
- **Ternary-add kernel via vpsignb**: `_mm256_sign_epi8(act_i8, w_i8)` — pure sign operation, replaces `vpmaddubs_epi16(act_u8, w_i8)`. No 128*row_sum correction needed. Identical argmax.
- **Four matmul modes**: int8 (default, vpmaddubs), f32 (reference, fmadd with float activations), ternary (v1.3.0, vpsignb with quantized activations), TQ1-packed (v1.3.1, chunked decode + SIMD).
- **`atlas_forward` seq_now**: Must be actual sequence length, NOT layer count.
- **Shared gate+up quantization**: C++ fused path = single shared scale. Python per-layer = separate scales. 0.3% gap is EXPECTED.

## Next Steps
1. ~~#3 Tokenizer + safetensors overhead killen~~ **(DONE — v5 embedded tokenizer)**
2. ~~**Linux native testen**~~ **(DONE — v1.2.0 compile-linux.sh validated)**
3. ~~**10B coherence test**~~ **(DONE — "The capital of France is Paris." ✅)**
4. ~~**README finalisieren**~~ **(DONE)**

## Critical Context
- **v1.3.1 latest** (TQ1-Packed Matmul: `matmul_tq1_packed_reorder` with chunked decode + SIMD).
- **10B model only** on disk (1B/3B/7B removed to free space for 10B testing).
- **`.i8` cache**: Auto-generated on first load, mmap'd on subsequent loads.
- **Prefill is already batched**: `forward()` passes all prompt tokens as B=prompt_len in single `atlas_forward` call. `atlas_attention_f32` handles causal masking per-token within the batch.
- **Engine correctness**: corr=0.9967 with fixed Python reference. Remaining 0.3% = shared quantization gap.
- **1B f32 bypass**: Auto-enabled for `hidden <= 2048`. Confirms u8 path is exact (identical argmax).
- **Benchmarks (v1.3.1, i5-1235U, 8 OMP threads, 30 tok, Python generate, warm)**:
  - **10B int8 cache**: load=7.5s, gen=~5.4 tok/s, 10.8 GB RAM
  - **10B packed**: load=1.6s, gen=1.2-3.3 tok/s, 7.5 GB RAM
- **Coherence**: T=0 identical argmax between packed and int8 ("The capital of France is Paris."). Verified: the 128*row_sum correction is REQUIRED because uint8 activations have a +128 bias, regardless of whether weights are ternary or full-range int8.
- **Memory**: Loading via file mmap (zero-copy). Packed mode: ~200 MB Python + ~3.3 GB C++ (TQ1 mmap only).

## Custom Skills (.opencode/skills/)
- `atlas-build`: Compile the C++ DLL/SO with correct flags (Windows clang++ / Linux GCC). Debug/release build + common linking fixes.
- `atlas-kernel`: AVX2 int8 matmul, f32 bypass, attention, RMSNorm, sampling kernels. Correctness expectations (corr>0.996) and memory model.
- `atlas-benchmark`: tok/s measurement, load times, f32-bypass comparison. Known benchmarks table. Coherence verification protocol.
- `atlas-hashline`: Hash-anchored editing with `hashline.py` — LINE+HASH anchors statt str_replace. SHA-256 content hashes, 4 edit ops, anchor validation, multi-file support.

## Hashline Editing
- `C:\opencode-tools\monorepo\hashline.py`: Hash-anchored editing tool — SHA-256 content hashes, 4 edit ops (`+`, `<`, `=`, `-`), anchor validation, multi-file support.
- `.opencode/skills/atlas-hashline/SKILL.md`: Skill fur Hashline-Workflow.
- Usage: `python hashline.py read <file>` -> Anker sehen, dann `python hashline.py edit <file> <diff>`.
- Also: `python hashline.py replace <file> --file-old <old> --file-new <new>`.
- Set `$env:PYTHONIOENCODING="utf-8"` auf Windows.
- OpenCode Plugin: `hashline_edit` (drop-in fur edit) + `hashline_patch` (raw diffs) via `C:\opencode-tools\plugins\`.

## OpenCode Tools (separate repo)
- **Monorepo**: `C:\opencode-tools\monorepo\` - 13 tools, 366 tests, 10 TS plugins.
- **Plugins**: `C:\opencode-tools\plugins\` - registered in `opencode.json` for auto-load.
- **Usage**: `impact def X`, `trace Y --down`, `verify file --contains text`, `graph file --out`.
- **Install**: `C:\opencode-tools\monorepo\install.bat` (Windows) or `install.sh` (Linux).
- **Test**: `cd C:\opencode-tools\monorepo && python test_impact.py`

## Relevant Files
- `C:\atlas\atlas_api.cpp`: `atlas_get_tokenizer()` C API (line 441), v5 header parsing (bytes 29-36), AllocHdr valloc/vfree, is_mapped guards, int8 matmul kernel, f32 bypass, Xoshiro256** PRNG, Gumbel-max sample, `atlas_generate()`, `matmul_ternary_add_reorder` (v1.3.0 vpsignb kernel).
- `C:\atlas\atlas_infer.py`: `AtlasModel` — embedded tokenizer loading via `atlas_get_tokenizer()`, `generate_c()` wraps `atlas_generate` (v1.2.1). `set_seed()` Python method.
- `C:\atlas\atlas_ffi.h`: Full C API — v5 layout, `atlas_get_tokenizer`, `atlas_set_seed`, `atlas_sample`, `atlas_generate`.
- `C:\atlas\atlas_packer.py`: v5 format writer — appends tokenizer block after tensor data, stores offset in header.
- `C:\atlas\falcon3-10b-tq1.atlas`: **v5** packed 10B model file with embedded tokenizer.
- `C:\models\Falcon3-10B-Instruct-1.58bit\`: model config, optional safetensors (only needed for repacking).
- `C:\opencode-tools\monorepo\`: opencode-tools monorepo (separate repo)
