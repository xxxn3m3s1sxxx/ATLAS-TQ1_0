// atlas_ffi.h — Pure C API contract for ATLAS TQ1.0 inference engine
// v1.3.0 — Single source of truth for FFI consumers (Mojo, Rust, Zig, Go, C)
//
// File format: ATLAS TQ1.0 (.atlas)
//   [0:5]   "ATLAS" magic
//   [5:2]   uint16 version
//   [7:2]   uint16 n_layers
//   [9:2]   uint16 hidden_dim
//   [11:2]  uint16 inter_dim
//   [13:1]  uint8  n_heads
//   [14:1]  uint8  n_kv_heads
//   [15:2]  uint16 head_dim
//   [17:4]  uint32 vocab_size (int32)
//   [21:8]  float64 rope_theta (version>=3), else 10000.0
//   [29:4]  int32  tokenizer_size (v5+), 0 if no embedded tokenizer
//   [33:4]  uint32 tokenizer_offset (v5+), absolute file offset
//   [37:4]  uint32 tokenizer_binary_size (v6+), 0 if no binary block
//   [41:4]  uint32 tokenizer_binary_offset (v6+), absolute file offset
//   [45:4]  uint32 eos_id (v5+)
//   [49:4]  uint32 pad_id (v5+)
//   [53:1]  uint8  model_flags (bit0=is_qwen3, bit1=tie_emb, bit2=thinking, bit3=relu2)
//   [54:2]  uint16 format_version (0=legacy, 2=v2.10.0 baseline)
//   [56:4]  int32  name_block_size (v4+)
//   [60:4]  int32  n_tensors
//   [64:]   tensor directory: n × [ttype:1][file_offset:4][row_dim:4][packed_cols:3]
//   then name block, then tensor data at each directory file_offset
//   (v5+: tokenizer data appended after tensor data, at tokenizer_offset)
//
// Int8 cache format (.i8 companion file)
//   [0:4]     int32 n_tensors (must match atlas n_tensors)
//   [4:]      n × [ttype:1][row_dim:4][pc:4][data_size:4][offset:8]
//   [4+n*21:] concatenated tensor data (only ttype==3 int8-decoded tensors)
//
// Tensor types (ttype):
//   0 = TQ1 packed (Base-3, 5 trits/byte), 2-byte scale prefix + packed bytes
//   1 = float16 vector (norm) or matrix (embed)— 2 bytes per element
//   2 = float16 matrix (lm_head) or vector (GQA scales)
//   3 = int8-decoded (after decompress or mmap cache load). [2:scale_fp16][rows×dim:i8][rows:row_sums_i32]
//   5 = TQ1 packed with per-block fp16 scales (Bonsai g128 format).
//       Layout: [block_size:1][n_blocks:2][scales:n_blocks*2 bytes fp16][packed_TQ1].
//       Block size is always 128 (g128). Scales apply per 128-column group.
//       Uses matmul_tq1_block_reorder (never decompressed to int8).
//  10 = TQ2 universal block-scaled ternary (TurboQuant 2.0).
//       Layout: [block_size:1][n_blocks:2][flags:1][scales:n_blocks*rows*2 fp16][packed_TQ1].
//       flags: bit0=has_sparsity_bitmap, bit1-7 reserved.
//       Block size default 128. Same TQ1 5-trit/byte Base-3 encoding as ttype=5.
//       Uses matmul_tq2 (on-the-fly decode, no decompress buffer).
//       Replaces ttype 0/3/5/7/8 — single universal weight format.
//
// All float16 values use IEEE 754 binary16.

#ifndef ATLAS_FFI_H
#define ATLAS_FFI_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
  #ifdef ATLAS_BUILD_DLL
    #define ATLAS_API __declspec(dllexport)
  #else
    #define ATLAS_API __declspec(dllimport)
  #endif
#else
  #define ATLAS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ─── Model config ─────────────────────────────────────────────────────
typedef struct {
    int n_layers;
    int hidden_dim;
    int inter_dim;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int vocab_size;
    float rope_theta;
} AtlasModelConfig;

// ─── Lifecycle ────────────────────────────────────────────────────────
// Load model from .atlas file. Returns opaque pointer or NULL on error.
// Memory: allocates all tensor data via VirtualAlloc (~2-3 GB for 10B).
ATLAS_API void* atlas_load(const char* path);

// Free all model resources. Must be called exactly once per atlas_load.
// Frees tensor data (unless mmap'd), unmaps cache, deletes model struct.
ATLAS_API void atlas_free(void* model);

// ─── Config query ─────────────────────────────────────────────────────
// Get all model parameters in one call. No allocation needed.
ATLAS_API AtlasModelConfig atlas_get_config(void* model);

// Legacy multi-param getter (deprecated — use atlas_get_config instead).
ATLAS_API void atlas_get_info(void* model, int* n_layers, int* hidden_dim,
                              int* inter_dim, int* n_heads, int* n_kv_heads,
                              int* head_dim, int* vocab_size);

// ─── Tensor name API (v4+, no safetensors dependency) ─────────────────
// Get number of tensors in the model. Returns 0 if names not loaded (v3).
ATLAS_API int atlas_get_tensor_count(void* model);

// Get tensor name at index. Returns chars written (excluding \0). 0 if error.
// buf_size should be at least 256.
ATLAS_API int atlas_get_tensor_name(void* model, int idx, char* buf, int buf_size);

// Find tensor index by exact name match. Returns -1 if not found.
ATLAS_API int atlas_get_tensor_index(void* model, const char* name);

// ─── Embedded tokenizer (v5+) ──────────────────────────────────────────
// Get embedded tokenizer data (JSON). Returns pointer to raw bytes or NULL.
// size: set to tokenizer length in bytes, or 0 if not available.
// Pointer is valid until atlas_free. Data is mmap'd from atlas file.
ATLAS_API const uint8_t* atlas_get_tokenizer(void* model, int* size);

// ─── v6 Tokenizer Binary Block ─────────────────────────────────────────
// Check if v6 binary tokenizer is available. Returns 1 if present, 0 otherwise.
ATLAS_API int atlas_has_binary_tokenizer(void* model);

// Pre-encode text to byte-level token IDs using raw byte_encoder lookup (no BPE merge).
// This is the "pre-tokenization" step — produces initial byte tokens that the
// BPE merge loop operates on. Use atlas_tokenizer_merge to complete encoding.
// text:      UTF-8 input bytes
// text_len:  length of input in bytes
// out_ids:   pre-allocated buffer for output byte token IDs
// max_ids:   capacity of out_ids
// Returns:   number of byte tokens written (>=0), or -1 on error
ATLAS_API int atlas_tokenizer_preencode(void* model,
    const char* text, int text_len,
    int* out_ids, int max_ids);

// Run BPE merge loop on pre-encoded byte token IDs. Modifies out_ids in-place.
// ids:       input byte token IDs (modified in-place to produce merged tokens)
// n_ids:     number of input byte tokens (modified to final token count)
// Returns:   0 on success, -1 on error
ATLAS_API int atlas_tokenizer_merge(void* model,
    int* ids, int* n_ids);

// Decode token IDs back to UTF-8 text using C++ tokenizer (v6 binary block).
// ids:       input token IDs
// n_ids:     number of input tokens
// out_text:  pre-allocated buffer for output UTF-8 text
// max_out:   capacity of out_text in bytes
// Returns:   number of bytes written (>=0), or -1 on error
ATLAS_API int atlas_tokenizer_decode(void* model,
    const int* ids, int n_ids,
    char* out_text, int max_out);

// ─── Decompression + cache ────────────────────────────────────────────
// Decompress all TQ1 tensors (ttype==0) to int8 (ttype==3) in-place.
// Frees packed TQ1 data after decompression. Call once after safetensors loaded.
ATLAS_API void atlas_decompress_all(void* model);

// Decompress block-scaled TQ1 (ttype==5) to uniform int8 (ttype==3) in-place.
// Converts per-block fp16 scales → single per-tensor int8 scale. v2.4.0.
ATLAS_API void atlas_decompress_ttype5(void* model);

// Decompress TurboQuant 2-bit (ttype==7) to uniform int8 (ttype==3) in-place.
// Converts on-the-fly decode → full int8 vpmaddubs speed. v2.7.0.
ATLAS_API void atlas_decompress_ttype7(void* model);

// Save decompressed int8 tensors to .i8 companion file. Safe to call
// multiple times (overwrites). Writes in 64KB chunks for Windows compat.
ATLAS_API void atlas_save_cache(void* model, const char* atlas_path);

// Load int8 cache file via mmap. Returns 1 on success, 0 if not found or corrupt.
// Replaces matching ttype==0 tensors with mmap'd ttype==3 pointers.
ATLAS_API int atlas_load_cache(void* model, const char* atlas_path);

// Prefetch all int8 data into physical RAM. Touches one byte per 4KB page.
// Call after decompress or cache load to prevent pagefault stalls.
ATLAS_API void atlas_prefetch_int8(void* model);

// Set full-precision matmul mode (no activation quantization).
// Enable for small models (1B) where u8 quantization degrades coherence.
ATLAS_API void atlas_set_use_f32_matmul(void* model, int val);

// v1.3.0: Set ternary matmul mode (vpsignb, no multiplication).
// Uses _mm256_sign_epi8 for pure sign-based ternary dot product.
// Works on existing .i8 cache weights. Eliminates 128*row_sum correction.
ATLAS_API void atlas_set_use_ternary_matmul(void* model, int val);

// v1.3.1: Set packed TQ1 matmul mode (direct on packed data, no decompression).
// Reads TQ1-packed weights directly — 5× less memory bandwidth than int8 path.
// Must be set BEFORE atlas_decompress_all (which would destroy the packed data).
ATLAS_API void atlas_set_use_packed_matmul(void* model, int val);

// v1.3.1: Set OpenMP thread count for this model (0 = use OMP default).
// Reduces CPU load on shared systems. 4 threads vs 8 typically loses ~10-20% tok/s.
ATLAS_API void atlas_set_num_threads(void* model, int n);

// v1.3.2: Set hybrid matmul mode (FFN int8 cache, QKV packed).
// Best speed/RAM balance: ~90% of full int8 speed, ~8 GB RAM.
ATLAS_API void atlas_set_use_hybrid_matmul(void* model, int val);

// v2.4.0: Set YaRN NTK RoPE scaling factor (1.0 = off, 4.0 = Bonsai-4B).
ATLAS_API void atlas_set_rope_interleaved(void* model, int enable);
ATLAS_API void atlas_set_rope_theta(void* model, float theta);
ATLAS_API void atlas_set_rope_scale(void* model, float scale);

// v2.5.0: Set base sequence length for NTK context extension.
// Default 4096. Set to model's trained context length (e.g., 2048 for Bonsai-1.7B).
// When max_seq_len > base_seq_len, NTK-aware frequency scaling is applied.
ATLAS_API void atlas_set_base_seq_len(void* model, int seq_len);

// v2.6.0: Reset KV cache — zeros all cache data without freeing allocation.
// Call between conversations to prevent context leakage across sessions.
ATLAS_API void atlas_reset_cache(void* model);

// v2.4.0: Set layer stride — tensors per transformer layer.
// 9 for Falcon3, 11 for Qwen3 (adds q_norm + k_norm).
ATLAS_API void atlas_set_layer_stride(void* model, int stride);
ATLAS_API void atlas_ensure_layer_idx(void* model);

// v1.3.2: Decompress only FFN tensors (gate/up/down) to int8, leave QKV packed.
ATLAS_API void atlas_decompress_ffn(void* model);

// v2.9.3: Convert all weight tensors to TQ2 (ttype=10) universal block-scaled format.
// Call after atlas_load(), before first inference. Converts ttype 0/3/5/7/8 → 10.
// Enables symmetric s8 activation quantization (no u8+128 offset, no f32 bypass).
ATLAS_API void atlas_convert_to_tq2(void* model);

// v2.10.0: i4 cache — save/load decompressed+quantized state for fast reload.
// Saves ttype=3 (int8) and ttype=8 (int4 packed) tensors to .i4 sidecar file.
// On next load, atlas_load_i4_cache restores state directly, skipping
// decompress_ffn + quantize_ffn_to_i4 entirely.
ATLAS_API void atlas_save_i4_cache(void* model, const char* atlas_path);
ATLAS_API int  atlas_load_i4_cache(void* model, const char* atlas_path);

// ─── Tensor access ────────────────────────────────────────────────────
// Get tensor metadata: type, row_dim, col_dim (=packed_cols*5 for TQ1, 0 otherwise).
ATLAS_API void atlas_tensor_info(void* model, int idx, int* ttype,
                                  int* row_dim, int* col_dim);

// Get raw tensor data pointer and size in bytes. View is valid until atlas_free.
ATLAS_API const uint8_t* atlas_tensor_data(void* model, int idx, int* size);

// Get int8-decoded tensor for direct access. Returns NULL if not ttype==3.
// scale: float32 dequant multiplier   row_sums: [rows] int32 Σ w_i per row
ATLAS_API const int8_t* atlas_get_int8(void* model, int idx, int* rows,
                                        int* input_dim, float* scale,
                                        const int32_t** row_sums);

// ─── Inference ────────────────────────────────────────────────────────
// Forward ALL transformer layers (RMSNorm + QKV + attention + FFN), fused.
// hidden_states: [B × hidden_dim] float32 — read from, overwritten with final layer output
// positions: [B] int32 — absolute position indices for RoPE
// layer_idx: [n_layers × 9] int32 — flat tensor indices per layer:
//   (ln1, q, k, v, o, ln2, gate, up, down) repeated for each layer
// k_cache, v_cache: [n_layers × n_kv_heads × max_seq_len × head_dim] uint16 (fp16)
// seq_now: current sequence length (positions will be < seq_now for decode)
//
// v2.3.0: KV cache is internal to the model (int8 quantized with per-position scale).
// k_cache and v_cache parameters removed — allocation/management handled inside.
//
// Final RMSNorm + LM head matmul should be applied in Python (numpy/MKL is faster).
// Memory: buffers allocated internally via valloc (mmap/VirtualAlloc).
//   ~4 × B × max(inter_dim, hidden_dim, n_heads*head_dim) × sizeof(float)
ATLAS_API void atlas_forward(void* model,
    float* hidden_states, int B,
    const int* positions,
    int max_seq_len, int seq_now,
    const int* layer_idx, int n_layers);

// ─── v1.2.0: Sampling + generation ────────────────────────────────────
// Seed the internal Xoshiro256** PRNG. Call once before generation.
ATLAS_API void atlas_set_seed(uint64_t seed);

// Sample one token from logits using softmax with top-k/top-p.
// logits: [vocab_size] float32 — modified in-place (used as scratch).
// output: [1] int32 — receives the sampled token ID.
ATLAS_API void atlas_sample(void* model, float* logits, int* output,
                             float temperature, int top_k, float top_p);

// End-to-end autoregressive generation. Single C call for the entire decode loop.
// Allocates internal scratch buffers (embedding, norm, logits). KV cache is
// managed internally (int8 quantized, allocated for max_seq_len).
//
// input_ids:  [n_input] int32 — tokenized prompt IDs
// output_ids: [max_new_tokens] int32 — receives generated token IDs
//
// Returns: number of tokens actually generated ( ≤ max_new_tokens ), or -1 on error.
// Stops when EOS token (id=0) is produced or max_new_tokens is reached.
ATLAS_API int atlas_generate(void* model,
    const int* input_ids, int n_input,
    int max_seq_len, int max_new_tokens,
    float temperature, int top_k, float top_p,
    float repetition_penalty,
    int min_new_tokens,
    int cache_offset,
    int* output_ids);

// ─── v2.3.0: Streaming generation ─────────────────────────────────────
// Callback type for streaming token output. Fired once per generated token.
// token_id: the sampled token ID
// user_data: opaque pointer passed to atlas_generate_stream (e.g., Python queue)
typedef void (*atlas_token_callback)(int token_id, void* user_data);

// Streaming variant of atlas_generate. Same parameters + callback + user_data.
// KV cache is managed internally. Fires callback for each token.
// Returns number of tokens generated, or -1 on error.
ATLAS_API int atlas_generate_stream(void* model,
    const int* input_ids, int n_input,
    int max_seq_len, int max_new_tokens,
    float temperature, int top_k, float top_p,
    float repetition_penalty,
    int min_new_tokens,
    int cache_offset,
    atlas_token_callback callback, void* user_data);

// ─── Int8 lm_head ─────────────────────────────────────────────────────
// Quantize lm_head from fp16 to per-row symmetric int8 (~403 MB vs 1.5 GB fp32).
// idx: tensor index of lm_head in the model's tensor list.
// keep_data: set to 1 for tie embeddings (embed_tokens also used for lookup).
// Frees the original fp16 data unless keep_data=1.
ATLAS_API void atlas_quantize_lmhead(void* model, int idx, int keep_data);

// GEMV: B tokens [B × hidden_dim] → logits [B × vocab_size] via int8 lm_head.
// AVX2 maddubs + offset trick + per-row dequant. OpenMP parallel over vocab.
ATLAS_API void atlas_lmhead_gemv(void* model, const float* act,
                                  float* output, int B);

// ─── Int8 matmul (AVX2 maddubs) ──────────────────────────────────────
// output[t][r] = sum_k act_u8[t][k] * w[r][k] - 128 * row_sums[r]
// act_u8: uint8 (= int8 activation + 128 offset)
// output: raw int32 result (needs dequant: out * max_abs[t] / (127 * scale))
ATLAS_API void atlas_matmul_i8_f32(int rows, int input_dim,
    const int8_t* weights, const uint8_t* act_u8,
    const int32_t* row_sums, float* output,
    int n_tokens);

// ─── Norms + positional encoding ─────────────────────────────────────
// RMSNorm: output[i] = x[i] * w[i] * rms(mean(x²) + eps)^{-1}
ATLAS_API void atlas_rmsnorm_f32(const float* x, const uint8_t* weight_f16,
                                  float* output, int n, float eps);

// RoPE: apply rotary embeddings to Q and K in-place for a single position
ATLAS_API void atlas_rope_f32(float* q, float* k, int n_heads, int n_kv_heads,
                               int head_dim, int position, float rope_theta);

// ─── Fused attention (RoPE + GQA + softmax + weighted sum) ───────────
// q, k, v: [B × n_heads*head_dim / n_kv_heads*head_dim] float32
//   q: RoPE applied in-place;  k, v: read only (cache used for attn)
// k_cache (int8), k_scale_cache (float): [n_kv_heads × max_seq × head_dim] int8 + per-position scaling
// v_cache (int8), v_scale_cache (float): same layout for V
// output: [B × n_heads × head_dim] float32
//
// v2.3.0: Int8 KV cache with per-kv_head, per-position scaling.
// Read: cache_row[h, s] = (float)k_cache[h, s, d] * k_scale_cache[h, s]
// v2.5.0: Ring buffer KV cache + NTK context extension (base_seq_len)
ATLAS_API void atlas_attention_f32(
    float* q, float* k, float* v, const int* positions,
    int8_t* k_cache, float* k_scale_cache,
    int8_t* v_cache, float* v_scale_cache,
    int max_seq_len, int seq_now, int B,
    int n_heads, int n_kv_heads, int head_dim,
    float rope_theta, float rope_scale, float* output,
    const uint8_t* q_norm_w, const uint8_t* k_norm_w,
    int base_seq_len);

#ifdef __cplusplus
}
#endif

#endif // ATLAS_FFI_H
