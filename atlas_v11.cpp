// Atlas v1.1 - Falcon3 Single-Layer Forward Pass
// sign_epi8 ternary kernel vs FP32 reference
// clang++ -O3 -mavx2 -lm atlas_v11.cpp -o atlas_v11.exe

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>

#define WEIGHTS "C:\\dam\\atlas\\weights\\"

// --- SIMD helpers ---
static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static inline int hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

static inline float hsum_epi32_f(__m256i v) { return (float)hsum_epi32(v); }

// --- Load binary weight files ---
// Format: [rows (int32)] [cols (int32)] [data (rows*cols * element_size)]
static int8_t* load_ternary(const char* name, int* rows, int* cols) {
    char path[256]; snprintf(path, sizeof(path), WEIGHT "%s_tern.bin", name);
    FILE* f = fopen(path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", path); return NULL; }
    fread(rows, 4, 1, f); fread(cols, 4, 1, f);
    int8_t* data = (int8_t*)_aligned_malloc((*rows) * (*cols), 32);
    fread(data, 1, (*rows) * (*cols), f);
    fclose(f);
    return data;
}

static float* load_fp32(const char* name, int* rows, int* cols) {
    char path[256]; snprintf(path, sizeof(path), WEIGHT "%s_fp32.bin", name);
    FILE* f = fopen(path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", path); return NULL; }
    fread(rows, 4, 1, f); fread(cols, 4, 1, f);
    float* data = (float*)_aligned_malloc((*rows) * (*cols) * sizeof(float), 32);
    fread(data, sizeof(float), (*rows) * (*cols), f);
    fclose(f);
    return data;
}

static float* load_vec(const char* name, int* n) {
    char path[256]; snprintf(path, sizeof(path), WEIGHT "%s.bin", name);
    FILE* f = fopen(path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", path); return NULL; }
    int r, c; fread(&r, 4, 1, f); fread(&c, 4, 1, f);
    *n = c;
    float* data = (float*)_aligned_malloc(c * sizeof(float), 32);
    fread(data, sizeof(float), c, f);
    fclose(f);
    return data;
}

// --- Kernels ---
// FP32 matrix-vector: y[rows] = w[rows*cols] @ x[cols]
static void mv_fp32(const float* w, const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        __m256 sum = _mm256_setzero_ps();
        for (int i = 0; i < cols; i += 8) {
            __m256 wv = _mm256_loadu_ps(&w[r * cols + i]);
            __m256 xv = _mm256_loadu_ps(&x[i]);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(wv, xv));
        }
        y[r] = hsum_ps(sum);
    }
}

// sign_epi8 ternary matrix-vector: y[rows] = w_tern[rows*cols] @ x_i8[cols]
// w_tern ∈ {-1, 0, +1}, x_i8 ∈ [-127, 127]
static void mv_ternary(const int8_t* w, const int8_t* a, int* y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        __m256i sum16 = _mm256_setzero_si256();
        for (int i = 0; i < cols; i += 32) {
            __m256i wv = _mm256_loadu_si256((__m256i*)&w[r * cols + i]);
            __m256i av = _mm256_loadu_si256((__m256i*)&a[i]);
            __m256i rv = _mm256_sign_epi8(av, wv);
            __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(rv));
            __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(rv, 1));
            sum16 = _mm256_add_epi16(sum16, _mm256_add_epi16(lo, hi));
        }
        __m256i sum32 = _mm256_madd_epi16(sum16, _mm256_set1_epi16(1));
        y[r] = hsum_epi32(sum32);
    }
}

// --- RMSNorm ---
// y = x / sqrt(mean(x^2) + eps) * weight
static void rms_norm(float* x, const float* weight, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) x[i] *= inv_rms * weight[i];
}

// --- SiLU ---
static inline float silu(float x) { return x / (1.0f + expf(-x)); }
static void silu_vec(float* x, int n) { for (int i = 0; i < n; i++) x[i] = silu(x[i]); }

// --- Single layer FP32 forward pass ---
// x: [3072] in/out. All weight arrays must be loaded.
static void layer_fp32(
    float* x,
    const float* w_q, const float* w_k, const float* w_v, const float* w_o,
    const float* w_gate, const float* w_up, const float* w_down,
    const float* ln1_w, const float* ln2_w,
    float* buf_q, float* buf_kv, float* buf_o,
    float* buf_gate, float* buf_up, float* buf_ffn,
    int d, int n_heads, int n_kv_heads, int head_dim, int intermediate
) {
    // Attention path
    memcpy(buf_o, x, d * sizeof(float));  // save residual
    rms_norm(x, ln1_w, d, 1e-6f);

    mv_fp32(w_q, x, buf_q, n_heads * head_dim, d);       // Q
    mv_fp32(w_k, x, buf_kv, n_kv_heads * head_dim, d);   // K
    mv_fp32(w_v, x, buf_kv + n_kv_heads * head_dim, n_kv_heads * head_dim, d); // V (append after K)

    // GQA: single token → attn_out = repeat_interleave(V, n_heads/n_kv_heads)
    float* v = buf_kv + n_kv_heads * head_dim;
    int g = n_heads / n_kv_heads;  // group size
    for (int h = 0; h < n_heads; h++) {
        int kv_idx = h / g;
        memcpy(&buf_q[h * head_dim], &v[kv_idx * head_dim], head_dim * sizeof(float));
    }

    // O_proj: attn_out → hidden
    mv_fp32(w_o, buf_q, buf_o + d, d, n_heads * head_dim);  // add to residual later
    
    // Residual: x = x + attn_out
    for (int i = 0; i < d; i++) x[i] = buf_o[i] + buf_o[d + i];

    // FFN path
    memcpy(buf_o, x, d * sizeof(float));  // save residual
    rms_norm(x, ln2_w, d, 1e-6f);

    mv_fp32(w_gate, x, buf_gate, intermediate, d);
    mv_fp32(w_up, x, buf_up, intermediate, d);
    silu_vec(buf_gate, intermediate);
    for (int i = 0; i < intermediate; i++) buf_ffn[i] = buf_gate[i] * buf_up[i];
    mv_fp32(w_down, buf_ffn, buf_q, d, intermediate);  // reuse buf_q for down output

    // Residual: x = x + ffn_out
    for (int i = 0; i < d; i++) x[i] = buf_o[i] + buf_q[i];
}

// --- Single layer TERNARY forward pass ---
static void layer_ternary(
    float* x,
    const int8_t* w_q, const int8_t* w_k, const int8_t* w_v, const int8_t* w_o,
    const int8_t* w_gate, const int8_t* w_up, const int8_t* w_down,
    float s_q, float s_k, float s_v, float s_o,
    float s_gate, float s_up, float s_down,
    const float* ln1_w, const float* ln2_w,
    int8_t* input_i8,
    int* buf_int, float* buf_fp,
    int d, int n_heads, int n_kv_heads, int head_dim, int intermediate
) {
    // Save residual
    memcpy(buf_fp, x, d * sizeof(float));
    rms_norm(x, ln1_w, d, 1e-6f);

    // Quantize input to int8
    for (int i = 0; i < d; i++) input_i8[i] = (int8_t)(x[i] * 127.0f);

    // Q, K, V
    mv_ternary(w_q, input_i8, buf_int, n_heads * head_dim, d);
    // Scale: output_fp32 ≈ scale * int32 / 127
    for (int i = 0; i < n_heads * head_dim; i++) x[i] = s_q * buf_int[i] / 127.0f;

    mv_ternary(w_k, input_i8, buf_int, n_kv_heads * head_dim, d);
    for (int i = 0; i < n_kv_heads * head_dim; i++) buf_fp[d + i] = s_k * buf_int[i] / 127.0f;

    mv_ternary(w_v, input_i8, buf_int, n_kv_heads * head_dim, d);
    float* v = buf_fp + d + n_kv_heads * head_dim;
    for (int i = 0; i < n_kv_heads * head_dim; i++) v[i] = s_v * buf_int[i] / 127.0f;

    // GQA: repeat_interleave V
    int g = n_heads / n_kv_heads;
    for (int h = 0; h < n_heads; h++) {
        int kv_idx = h / g;
        memcpy(&x[h * head_dim], &v[kv_idx * head_dim], head_dim * sizeof(float));
    }
    // Save attn output in buf_fp (d space starting after residual + kv space)
    float* attn_out = buf_fp + d + n_kv_heads * head_dim * 2;
    memcpy(attn_out, x, n_heads * head_dim * sizeof(float));

    // O_proj
    for (int i = 0; i < n_heads * head_dim; i++) {
        input_i8[i] = (int8_t)(attn_out[i] * 127.0f / s_o);
        // Scale so sign_epi8 output ≈ s_o * int32 / 127
        // We want: s_o * sum(w_tern[i] * a_i8[i]) / 127 ≈ s_o * sum(w_tern[i] * a_fp32[i])
        // Since a_i8 ≈ a_fp32 * 127: s_o * sum(w_tern * a_i8) / 127 ≈ s_o * sum(w_tern * a_fp32)
        // So input_i8 = a_fp32 * 127, and result = s_o * dot / 127
    }
    for (int i = 0; i < d; i++) input_i8[i] = (int8_t)(attn_out[i] * 127.0f);
    mv_ternary(w_o, input_i8, buf_int, d, n_heads * head_dim);
    // Residual: x = residual + attn_out (saved in buf_fp[0..d])
    for (int i = 0; i < d; i++) x[i] = buf_fp[i] + s_o * buf_int[i] / 127.0f;

    // FFN path
    memcpy(buf_fp, x, d * sizeof(float));
    rms_norm(x, ln2_w, d, 1e-6f);

    for (int i = 0; i < d; i++) input_i8[i] = (int8_t)(x[i] * 127.0f);

    mv_ternary(w_gate, input_i8, buf_int, intermediate, d);
    for (int i = 0; i < intermediate; i++) buf_fp[d + i] = s_gate * buf_int[i] / 127.0f;
    silu_vec(buf_fp + d, intermediate);

    mv_ternary(w_up, input_i8, buf_int, intermediate, d);
    for (int i = 0; i < intermediate; i++) {
        float up_val = s_up * buf_int[i] / 127.0f;
        buf_fp[d + i] *= up_val;  // gate_silu * up → ffn
    }

    mv_ternary(w_down, input_i8, buf_int, d, intermediate);
    // Re-quantize FFN output
    for (int i = 0; i < intermediate; i++) input_i8[i] = (int8_t)(buf_fp[d + i] * 127.0f);
    // Hmm, this only quantizes 'intermediate' values but w_down takes intermediate as input.
    // Let me redo this properly.
    
    // Actually, let me fix this approach:
}

// Let me rewrite layer_ternary more carefully... (see v12)
int main() {
    printf("Atlas v1.1 - Falcon3 Single Layer Forward Pass\n");
    printf("See atlas_v12.cpp for complete implementation\n");
    return 0;
}
