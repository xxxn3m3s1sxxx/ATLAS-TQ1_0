# Bugfix Chronology

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

### Bug 13 [FIXED v2.6.3]: BitNet SubLN signal collapse (u8×i8 activation quant)

BitNet's SubLN (sub-layer normalization) architecture uses RMSNorm weights of magnitude ~0.01×. The uint8×int8 quantized matmul path (`vpmaddubbs` SIMD) adds +128 to activations, destroying these small signals in quantization rounding. Output was pure noise — correlation near zero, max token probability near uniform.

**Root cause**: `u8+128 activation_quant + vpmaddubbs` is lossy for activation magnitudes < ~0.05. SubLN-produced activations are ~10-50× smaller than standard LayerNorm outputs.

**Fix**: Force `f32_bypass` for `ARCH_BITNET` models — raw fp32 matmul preserves full signal. Added `atlas_ensure_layer_idx()` C API to set `model_arch` on C++ side for Python `forward()` path.

**Symptoms**: BitNet models produce empty/garbled output via default quantized path; T=0.7 yields near-uniform distribution. f32_bypass path produces correct output ("Paris is the capital of France.").

### Bug 14 [FIXED v2.6.3]: `atlas_ensure_layer_idx` missing for Python forward()

Python `forward()` path called `atlas_forward` without setting `model_arch` first. The C++ decoder checked `arch == ARCH_BITNET` to enable SubLN, but the arch field was uninitialized (default 0 = ARCH_FALCON). SubLN was skipped, activations exploded, forward() returned garbage.

**Fix**: New `atlas_ensure_layer_idx()` C API called during `AtlasModel.__init__()`. Sets `model_arch` from header byte 53, enabling proper SubLN routing for all generation paths.

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

All model families (Falcon3 1B/3B/7B/10B, Bonsai 1.7B/4B, BitNet-2B4T, TriLM 1.1B/1.5B) pass coherence: "The capital of France is Paris." at T=0 with appropriate matmul path. Small models (hidden ≤ 2048) require f32_bypass or T ≥ 0.7 — greedy decoding degenerates due to model-inherent distribution.

## Version History

| Version | Key Changes |
|---------|-------------|
| **v2.6.3** | BitNet ARCH_BITNET + Repo Cleanup: `atlas_ensure_layer_idx()` C API, f32_bypass forced for SubLN models. 91 scratch files deleted, dead packers removed, `.gitignore` hardened. macOS CI split into release-only (free tier runner bottleneck). |
| **v2.6.2** | Safe decompress_ttype5 dispatch (try/except guard) |
| **v2.6.1** | ttype=5 decompress + f32_bypass for all block-scaled models |
| **v2.6.0** | SSE Web-Server + Prompt-Caching + CI Pipeline |
| **v2.5.0** | Context Window Extension — ring buffer KV cache, NTK scaling |
| **v2.4.1** | Static analysis bughunt (5 C++ bugs), ttype=5 int8 decompress for Bonsai (10× speedup) |
| **v2.4.0** | Qwen3/Bonsai-4B TQ1.0 support — head_dim=128, QK-Norm, YaRN RoPE |
| **v2.3.1** | Windows packer hotfix, 7B v6 repair |
| **v2.3.0** | Int8 KV-Cache (fp16→int8), 10B@4K: 320→173 MB |
| **v2.2.2** | F16C in attention + weighted sum, 10B +47%, 3B +5.7% |
| **v2.2.1** | BPE-PQ priority queue (O(n²)→O(n log n)) |
| **v2.2.0** | TQ1-LUT in decompression, F16C for fp16→fp32 |
| **v2.1.1** | Repetition penalty in C-core |
| **v2.1.0** | Streaming + chat history |
| **v2.0.4** | Softmax sampling (replace Gumbel-max), default T=0.7 |
| **v2.0.3** | n_input ≥ max_seq_len guard, scores_buf OOM guard |
| **v2.0.2** | Memory leak, KV-cache overflow, stale .i8 cache, thread-local statics |
| **v2.0.1** | scores alloca → heap (stack fully sterile) |
| **v2.0.0** | C++ binary tokenizer (v6 format) |
| **v1.4.0** | Stack overflow fix, survivor-list sampling |
| **v1.3.2** | Hybrid mode (FFN int8 + QKV packed) |
| **v1.3.1** | Direct TQ1-packed matmul |
| **v1.3.0** | Ternary-add kernel (vpsignb) |
| **v1.2.0** | C++ sampling, atlas_generate |
| **v1.1.0** | AllocHdr-based valloc/vfree |
| **v1.0.0** | Initial TQ1.0 inference engine |
