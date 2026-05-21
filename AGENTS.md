# ATLAS — Falcon3 TQ1.0 Inference Engine

## Goal
- TQ1.0 (Base-3, 5 ternary trits/byte) CPU inference engine for Falcon3 models on i5, end-to-end text generation via C++ DLL/SO + Python (Windows + Linux x86-64).
- v2.0.0: Remove `transformers` dependency at runtime — C++ binary tokenizer replaces PreTrainedTokenizerFast. `tokenizers` lib still used for encode (Falcon3 byte_encoder is fundamentally broken for C++ encode), C++ pool-lookup handles decode.

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
- **v1.3.2 — Hybrid Mode (FFN int8 + QKV packed)**: `atlas_set_use_hybrid_matmul()` + `atlas_decompress_ffn()` C APIs. Per-tensor ttype-based dispatch (ttype=0→packed, ttype=3→int8). FFN (gate/up/down) decompressed to int8, QKV/O stay packed. **6.6 tok/s on 10B** (warm) — faster than full int8 cold start because no 9.5 GB cache file to load. ~9.3 GB RAM. All 4 models packed + tested.
- **v1.3.2 Full Portfolio**: All 4 Falcon3 models packed and benchmarked on i5-1235U:
  - **1B hybrid**: 14.9 tok/s, ~1.4 GB, f32 bypass auto-enabled
  - **3B hybrid**: 4.6 tok/s, ~4 GB
  - **7B hybrid**: 7.4 tok/s, ~7 GB
  - **10B hybrid**: 4.8 tok/s (warm), ~9.3 GB
- **1B hybrid fix**: When hidden<=2048 in hybrid mode, decompresses ALL tensors (not just FFN) to ensure f32 bypass fires deterministically. Packed→f32 dispatch gap (ttype==0 checked before use_f32_matmul) fixed by bypassing packed path entirely for small models.
- **3B `generate_c` coherence verified — FALSE ALARM**: Extensive debugging showed the C path (`atlas_generate`) produces identical correct output ("Paris") at T=0 for 3B. The earlier perception of a bug was caused by comparing against Python `generate()` which has a different sampling implementation (numpy multinomial vs Xoshiro256** Gumbel-max) and uses `_rmsnorm`/`_lmhead_gemv` Python wrappers that produce slightly different numerics from the C path. Verified with 0xCC garbage-filled decode buffers (no uninitialized tail issue). All 4 models: **1B/3B/7B/10B all produce correct answers at T=0 via `generate_c()`**.
- **NULL-check added**: `matmul_tq1_packed_reorder` decode_buf malloc now has NULL check before use.
- **v2.0.0 — C++ Binary Tokenizer (Phase 1-4)**: v6 format with binary tokenizer block (offsets/lengths/pool/merges/byte_encoder). C++ decode via pool lookup (O(1)). Python `tokenizers` for encode (Falcon3 byte_encoder fundamentally broken — 191/256 byte tokens replaced by special tokens at IDs 0-255). `_apply_chat_template` renders Falcon3 Jinja2 format without transformers. **No `transformers` dependency at runtime.**
- **1B/7B v6 packed**: Version 6 header, binary tokenizer block ~3.22 MB. Roundtrip encode→C++ decode verified correct. 7B coherence: "Paris" at T=0 via `generate_c` ✅.
- **`add_v6_block.py`**: Fast v6 block appender to existing v5 `.atlas` files. Reads `tokenizer.json` directly, builds binary block, updates header. Avoids full 10+ min repack for v5→v6 migration.
- **Chat template bug fixed**: `_apply_chat_template` was using `<|im_end|>` format. Falcon3 actual template: `<|user|>\n{content}\n<|assistant|>\n`. Fix restored 7B coherence.

### In Progress
- *(none)*

### Fixed
- **Bug 12 [Stack overflow from alloca in forward_layer_internal]**: Four `alloca(B * qd * sizeof(float))` calls in attention section used ~2.9 MB stack at B=60+, exceeding the 1 MB default Windows stack. Floated all 4 buffers to heap via `attn_ws` field in struct, allocated in `ensure_buffers`. Also moved `scores` alloca in `atlas_attention_f32` outside the per-batch loop (was accumulating across B iterations). Total stack now ~210 KB max. All 4 models pass long-prompt test (60+ tokens). Fixed in v1.4.0.
- **Bug 11 [10B tokenizer_offset int32 overflow]**: `int` (signed 32-bit) fur `tokenizer_offset` overflowt bei >2 GB Dateigroese. 10B Offset bei ~3.3 GB -> negativ als int32. Fix: `uint32_t` + `ptrdiff_t` cast. Einziger Bug in v1.2.0-pre (behoben in v1.2.0).

## Key Decisions
- **v5 format**: `[header:64] [dir:n*12] [name_block] [token_data...] [tokenizer_block]`. Tokenizer stored as separate raw JSON block (no merge — avoids tokenizers Rust parser corruption). Header bytes 29-32: tokenizer_size, 33-36: tokenizer_offset.
- **f32 bypass bleibt drin**: Eliminates engine quantization noise. Serves as numerical reference path.
- **Ternary-add kernel via vpsignb**: `_mm256_sign_epi8(act_i8, w_i8)` — pure sign operation, replaces `vpmaddubs_epi16(act_u8, w_i8)`. No 128*row_sum correction needed. Identical argmax.
- **Four matmul modes**: int8 (default, vpmaddubs), f32 (reference, fmadd with float activations), ternary (v1.3.0, vpsignb with quantized activations), TQ1-packed (v1.3.1, chunked decode + SIMD).
- **`atlas_forward` seq_now**: Must be actual sequence length, NOT layer count.
- **Shared gate+up quantization**: C++ fused path = single shared scale. Python per-layer = separate scales. 0.3% gap is EXPECTED.

## Next Steps
1. ~~v5 embedded tokenizer~~ ✅
2. ~~Linux compile~~ ✅
3. ~~All models packed + coherence verified~~ ✅
4. ~~v1.3.2 hybrid mode for all 4 models~~ ✅
5. **v2.0.0 — C++ Binary Tokenizer**: Remove Python tokenizer dependency entirely.

## Critical Context
- **v1.4.0 latest** (Stack overflow fix — heap-allocated attention workspace).
- **All 4 models on disk** (C:\models): 1B/3B/7B/10B + packed .atlas files in C:\atlas.
- **Hybrid default path**: FFN tensors decompressed to int8 (ttype=3), QKV/O stay packed (ttype=0). Per-tensor ttype dispatch in forward. For 1B (hidden<=2048), all tensors decompressed + f32 bypass enabled.
- **`.i8` cache**: Auto-generated on first load (full decompress path), mmap'd on subsequent loads.
- **Prefill is already batched**: `forward()` passes all prompt tokens as B=prompt_len in single `atlas_forward` call.
- **Three matmul modes**: hybrid (default), packed (use_packed_matmul=True), int8 (full cache).
- **Coherence**: ALL 4 models produce "Paris" at T=0 via `generate_c()`. 1B greedy degenerates to `,` (model-inherent distribution, not engine). 3B greedy repeats "Paris" sentence (model-inherent). 7B/10B give single "Paris". All pass at T=0.
- **f32 bypass dispatch**: `use_f32_matmul` is checked AFTER `t.ttype==0` in forward. For 1B hybrid, all tensors are decompressed so f32 bypass fires correctly. Pure packed + f32 bypass is broken (ttype==0 catches first) — not used because 1B hybrid auto-decompresses everything.
- **Sampling optimized (v1.5.0-pre)**: `gumbel_sample` collects survivors into `static std::vector<int> sidx` after top_k pruning. Top_p softmax + max-heap operate on survivors only (not full V=131072). Survivor collection integrated into top_k mask loop (no extra pass). Gumbel noise only on survivors. Eliminates O(V log V) sort bottleneck — top_k+p overhead ≈ top_k overhead.
- **Benchmarks (v1.5.0-pre, i5-1235U, 8 OMP threads, generate_c, warm)**:
  - **1B int8 cache**: 30-33 tok/s, ~1.2 GB, load 2.7s
  - **3B int8 cache**: 5-6 tok/s, ~4 GB, load 4.1s
  - **7B int8 (no cache)**: 19-20 tok/s, ~7 GB, load 26.3s
  - **10B int8 cache**: 10-13 tok/s, ~9 GB, load 25.0s
  - Sampling overhead (1B, top_k=40+p): ~3 tok/s (survivor-list makes top_p ≈ free after top_k)
- **128*row_sum correction**: REQUIRED for ALL uint8×int8 matmuls (activation quantization adds +128 bias, regardless of weight format).
- **v2.0.0 byte_encoder limitation**: Falcon3 overwrites IDs 0-255 with special tokens. 191/256 byte tokens missing from vocabulary. Only 65 bytes have Unicode tokens. C++ byte-level pre-encode fundamentally impossible — Python `tokenizers` for encode, C++ for decode is correct architecture.
- **`_apply_chat_template` format**: `<|role|>\n{content}\n` — NO `<|im_end|>` tokens. For assistant messages: `<|assistant|>\n{content}<|endoftext|>\n`. Generation prompt: `<|assistant|>\n`.
- **C++ decode**: Returns raw pool strings (e.g., "Ġis"). Python `_cpp_decode` applies `bytes_to_unicode()` inverse: maps each char to byte, interprets as UTF-8. Handles Ġ→space, Ċ→newline.

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
- `C:\dam\ATLAS-TQ1_0\atlas_packer.py`: v6 `build_tokenizer_binary()` with O(n) pool offset, bytes_to_unicode(), JSON-based merge reading.
- `C:\dam\ATLAS-TQ1_0\atlas_ffi.h`: v6 header bytes 37-44, API declarations for `atlas_has_binary_tokenizer`, `atlas_tokenizer_preencode`, `atlas_tokenizer_merge`, `atlas_tokenizer_decode`.
- `C:\dam\ATLAS-TQ1_0\atlas_api.cpp`: Binary tokenizer loading + hash table build (load time). `atlas_tokenizer_preencode` (raw byte LUT), `atlas_tokenizer_merge` (BPE loop), `atlas_tokenizer_decode` (pool lookup).
- `C:\dam\ATLAS-TQ1_0\atlas_infer.py`: Removed `AutoTokenizer`. `_cpp_encode` uses `self._tok.encode(text).ids`. `_cpp_decode` calls C++ pool lookup + Python ByteLevel post-processing. `_apply_chat_template` implemented (no Jinja2).
- `C:\dam\ATLAS-TQ1_0\add_v6_block.py`: Utility to append v6 binary block to existing v5 `.atlas` files. Reads tokenizer.json directly. Avoids full repack.
- `C:\dam\ATLAS-TQ1_0\v6-spec.md`: Binary tokenizer format spec (128-byte header, offsets/lengths/pool, merge arrays, byte_encoder, special tokens).
