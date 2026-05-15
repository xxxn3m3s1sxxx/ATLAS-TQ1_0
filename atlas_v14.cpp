// Atlas v1.4 - Falcon3 KV Cache + GQA Attention
// Validates KV cache attention against single-token expansion
// clang++ -O3 -mavx2 -lm atlas_v14.cpp -o atlas_v14.exe

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>

#define WDIR "C:\\dam\\atlas\\weights\\"
#define HIDDEN 3072
#define N_HEADS 12
#define N_KV_HEADS 4
#define HEAD_DIM 256
#define INTERMEDIATE 23040
#define EPS 1e-6f
#define MAX_SEQ 2048

// ===== SIMD =====
static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
static inline int hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo); lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

// ===== I/O =====
static void* load_bin(const char* name, const char* suf, int esz, int* r, int* c) {
    char p[512]; snprintf(p, sizeof(p), "%s%s_%s", WDIR, name, suf);
    FILE* f = fopen(p, "rb"); if (!f) { printf("ERROR: %s\n", p); exit(1); }
    fread(r, 4, 1, f); fread(c, 4, 1, f);
    void* d = _aligned_malloc((*r) * (*c) * esz, 32);
    fread(d, esz, (*r) * (*c), f); fclose(f);
    return d;
}
#define load_tern(n, r, c) (int8_t*)load_bin(n, "tern.bin", 1, r, c)
#define load_fp32(n, r, c) (float*)load_bin(n, "fp32.bin", 4, r, c)

static float* load_vec(const char* name, int* n) {
    char p[512]; snprintf(p, sizeof(p), "%s%s", WDIR, name);
    FILE* f = fopen(p, "rb"); if (!f) { printf("ERROR: %s\n", p); exit(1); }
    int r; fread(&r, 4, 1, f); fread(n, 4, 1, f);
    float* d = (float*)_aligned_malloc((*n) * 4, 32);
    fread(d, 4, *n, f); fclose(f);
    return d;
}

// ===== Kernels =====
static void mv_fp32(const float* w, const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        __m256 sum = _mm256_setzero_ps();
        for (int i = 0; i < cols; i += 8)
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(&w[r*cols+i]), _mm256_loadu_ps(&x[i])));
        y[r] = hsum_ps(sum);
    }
}
static void mv_tern(const int8_t* w, const int8_t* a, int* y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        __m256i s16 = _mm256_setzero_si256();
        for (int i = 0; i < cols; i += 32) {
            __m256i wv = _mm256_loadu_si256((__m256i*)&w[r*cols+i]);
            __m256i av = _mm256_loadu_si256((__m256i*)&a[i]);
            __m256i rv = _mm256_sign_epi8(av, wv);
            __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(rv));
            __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(rv, 1));
            s16 = _mm256_add_epi16(s16, _mm256_add_epi16(lo, hi));
        }
        y[r] = hsum_epi32(_mm256_madd_epi16(s16, _mm256_set1_epi16(1)));
    }
}

static float quantize(const float* s, int8_t* d, int n) {
    float ma = 1e-10f; for (int i = 0; i < n; i++) { float a = fabsf(s[i]); if (a > ma) ma = a; }
    float q = ma / 127.0f; for (int i = 0; i < n; i++) d[i] = (int8_t)(s[i] / q);
    return q;
}
static void dequant(const int* s, float* d, float ws, float qs, int n) {
    float sc = ws * qs; for (int i = 0; i < n; i++) d[i] = sc * s[i];
}

static void rms_norm(float* x, const float* w, int n) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + EPS);
    for (int i = 0; i < n; i++) x[i] *= inv * w[i];
}
static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }
static void expand_gqa(float* out, const float* in, int nh, int nkv, int hd) {
    int g = nh / nkv;
    for (int h = 0; h < nh; h++) memcpy(&out[h*hd], &in[(h/g)*hd], hd * 4);
}

// ===== FP32 SIMD attention =====
// q: [n_heads, head_dim], k_cache/v_cache: [n_kv, max_seq, head_dim] (row-major per head)
// attn_out: [n_heads, head_dim]
// temp: scratch [max(seq_len) * 2] floats
static void attention_gqa(float* attn_out, const float* q,
                          const float* k_cache, const float* v_cache,
                          int n_heads, int n_kv, int hd, int seq_len, int max_s,
                          float* temp) {
    int g = n_heads / n_kv;
    for (int kv = 0; kv < n_kv; kv++) {
        // Scores: Q_group [g, hd] @ K_cache [seq_len, hd]^T → scores [g, seq_len]
        for (int qh = 0; qh < g; qh++) {
            const float* qv = &q[(kv * g + qh) * hd];
            float* sc = &temp[qh * seq_len];
            for (int s = 0; s < seq_len; s++) {
                __m256 sum = _mm256_setzero_ps();
                const float* kv_vec = &k_cache[(kv * max_s + s) * hd];
                for (int i = 0; i < hd; i += 8)
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(&qv[i]), _mm256_loadu_ps(&kv_vec[i])));
                sc[s] = hsum_ps(sum) / sqrtf(hd);
            }
        }
        // Softmax per Q head in group
        for (int qh = 0; qh < g; qh++) {
            float* sc = &temp[qh * seq_len];
            float mx = sc[0]; for (int s = 1; s < seq_len; s++) if (sc[s] > mx) mx = sc[s];
            float sum = 0; for (int s = 0; s < seq_len; s++) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
            float inv_sum = 1.0f / sum; for (int s = 0; s < seq_len; s++) sc[s] *= inv_sum;
        }
        // Weighted sum of V: attn_out[g * hd] = score @ V_cache
        for (int qh = 0; qh < g; qh++) {
            float* out = &attn_out[(kv * g + qh) * hd];
            const float* sc = &temp[qh * seq_len];
            for (int i = 0; i < hd; i++) {
                float sum = 0;
                int s = 0;
                for (; s + 8 <= seq_len; s += 8) {
                    __m256 sv = _mm256_loadu_ps(&sc[s]);
                    __m256 vv = _mm256_loadu_ps(&v_cache[(kv * max_s + s) * hd + i]);
                    sum += hsum_ps(_mm256_mul_ps(sv, vv));
                }
                for (; s < seq_len; s++)
                    sum += sc[s] * v_cache[(kv * max_s + s) * hd + i];
                out[i] = sum;
            }
        }
    }
}

// ===== Weights =====
struct LayerW {
    int8_t *tq, *tk, *tv, *to, *tg, *tu, *td;
    float *ln1, *ln2, sq, sk, sv, so, sg, su, sd;
};
static LayerW load_layer() {
    LayerW l; int r, c;
    printf("  Loading layer 0...\n");
    l.tq = load_tern("layer0_q_proj", &r, &c); l.sq=48.75f;
    l.tk = load_tern("layer0_k_proj", &r, &c); l.sk=36.75f;
    l.tv = load_tern("layer0_v_proj", &r, &c); l.sv=26.75f;
    l.to = load_tern("layer0_o_proj", &r, &c); l.so=27.875f;
    l.tg = load_tern("layer0_gate_proj", &r, &c); l.sg=11.4375f;
    l.tu = load_tern("layer0_up_proj", &r, &c); l.su=11.75f;
    l.td = load_tern("layer0_down_proj", &r, &c); l.sd=10.875f;
    l.ln1 = load_vec("layer0_input_layernorm.bin", &r);
    l.ln2 = load_vec("layer0_post_attention_layernorm.bin", &r);
    return l;
}

// ===== Single-token forward (no KV cache, expand_gqa) =====
// Used as reference for first-token validation
static void forward_single(const LayerW& l, float* x, float* scr,
                           int8_t* ai8, int* bi, float* gate, float* up) {
    int d=HIDDEN, hd=HEAD_DIM, in=INTERMEDIATE, nh=N_HEADS, nkv=N_KV_HEADS;
    float *res=scr, *q=scr+d, *kv=scr+d+nh*hd;

    memcpy(res, x, d*4); rms_norm(x, l.ln1, d);
    float qq = quantize(x, ai8, d);
    mv_tern(l.tq, ai8, bi, nh*hd, d); dequant(bi, q, l.sq, qq, nh*hd);
    mv_tern(l.tk, ai8, bi, nkv*hd, d); dequant(bi, kv, l.sk, qq, nkv*hd);
    mv_tern(l.tv, ai8, bi, nkv*hd, d); dequant(bi, kv+nkv*hd, l.sv, qq, nkv*hd);
    expand_gqa(q, kv+nkv*hd, nh, nkv, hd);  // V → Q buffer as attn_out
    float qo = quantize(q, ai8, nh*hd);
    mv_tern(l.to, ai8, bi, d, nh*hd); dequant(bi, kv, l.so, qo, d);
    for (int i = 0; i < d; i++) x[i] = res[i] + kv[i];

    memcpy(res, x, d*4); rms_norm(x, l.ln2, d);
    float qg = quantize(x, ai8, d);
    mv_tern(l.tg, ai8, bi, in, d); dequant(bi, gate, l.sg, qg, in);
    mv_tern(l.tu, ai8, bi, in, d); dequant(bi, up, l.su, qg, in);
    for (int i = 0; i < in; i++) gate[i] = silu_f(gate[i]) * up[i];
    float qd = quantize(gate, ai8, in);
    mv_tern(l.td, ai8, bi, d, in); dequant(bi, q, l.sd, qd, d);
    for (int i = 0; i < d; i++) x[i] = res[i] + q[i];
}

// ===== KV-cached forward =====
// k_cache/v_cache: [n_kv, max_seq, head_dim] already filled up to seq_len-1
// Appends current K,V at position seq_len, then computes attention over [0..seq_len]
static void forward_cached(const LayerW& l, float* x, float* scr,
                           int8_t* ai8, int* bi, float* gate, float* up,
                           float* k_cache, float* v_cache, int seq_len, int max_s) {
    int d=HIDDEN, hd=HEAD_DIM, in=INTERMEDIATE, nh=N_HEADS, nkv=N_KV_HEADS;
    float *res=scr, *q=scr+d, *kv=scr+d+nh*hd;

    memcpy(res, x, d*4); rms_norm(x, l.ln1, d);

    // Q, K, V
    float qq = quantize(x, ai8, d);
    mv_tern(l.tq, ai8, bi, nh*hd, d); dequant(bi, q, l.sq, qq, nh*hd);
    mv_tern(l.tk, ai8, bi, nkv*hd, d); dequant(bi, kv, l.sk, qq, nkv*hd);
    mv_tern(l.tv, ai8, bi, nkv*hd, d); dequant(bi, kv+nkv*hd, l.sv, qq, nkv*hd);

    // Append K, V to cache at position seq_len
    for (int kv_i = 0; kv_i < nkv; kv_i++) {
        memcpy(&k_cache[(kv_i * max_s + seq_len) * hd], &kv[kv_i * hd], hd * 4);
        memcpy(&v_cache[(kv_i * max_s + seq_len) * hd], &kv[nkv*hd + kv_i*hd], hd * 4);
    }

    // Attention over all cached tokens [0..seq_len]
    attention_gqa(q, q, k_cache, v_cache, nh, nkv, hd, seq_len + 1, max_s, scr + d + nh*hd + nkv*hd);

    // O_proj
    float qo = quantize(q, ai8, nh*hd);
    mv_tern(l.to, ai8, bi, d, nh*hd); dequant(bi, kv, l.so, qo, d);
    for (int i = 0; i < d; i++) x[i] = res[i] + kv[i];

    // FFN
    memcpy(res, x, d*4); rms_norm(x, l.ln2, d);
    float qg = quantize(x, ai8, d);
    mv_tern(l.tg, ai8, bi, in, d); dequant(bi, gate, l.sg, qg, in);
    mv_tern(l.tu, ai8, bi, in, d); dequant(bi, up, l.su, qg, in);
    for (int i = 0; i < in; i++) gate[i] = silu_f(gate[i]) * up[i];
    float qd = quantize(gate, ai8, in);
    mv_tern(l.td, ai8, bi, d, in); dequant(bi, q, l.sd, qd, d);
    for (int i = 0; i < d; i++) x[i] = res[i] + q[i];
}

// ===== Main =====
int main() {
    printf("Atlas v1.4 - KV Cache + GQA Attention Validation\n\n");
    LayerW lw = load_layer();

    // Buffers
    float* x    = (float*)_aligned_malloc(HIDDEN*4, 32);
    float* x_c  = (float*)_aligned_malloc(HIDDEN*4, 32);
    float* scr  = (float*)_aligned_malloc(131072*4, 32);
    int8_t* ai8 = (int8_t*)_aligned_malloc(INTERMEDIATE, 32);
    int* bi     = (int*)_aligned_malloc(INTERMEDIATE*4, 32);
    float* gate = (float*)_aligned_malloc(INTERMEDIATE*4, 32);
    float* up   = (float*)_aligned_malloc(INTERMEDIATE*4, 32);

    // KV cache: [n_kv_heads, max_seq, head_dim]
    float* k_cache = (float*)_aligned_malloc(N_KV_HEADS * MAX_SEQ * HEAD_DIM * 4, 32);
    float* v_cache = (float*)_aligned_malloc(N_KV_HEADS * MAX_SEQ * HEAD_DIM * 4, 32);

    // 3 random token embeddings
    srand(42);
    float emb[3][HIDDEN];
    for (int t = 0; t < 3; t++)
        for (int i = 0; i < HIDDEN; i++)
            emb[t][i] = (float)(rand()%200-100)/100.0f;

    printf("\n--- Test 1: Single-token path (independent tokens) ---\n");
    float out_single[3][HIDDEN];
    for (int t = 0; t < 3; t++) {
        memcpy(x, emb[t], HIDDEN*4);
        forward_single(lw, x, scr, ai8, bi, gate, up);
        memcpy(out_single[t], x, HIDDEN*4);
        printf("  Token %d: first 5 = %.2f %.2f %.2f %.2f %.2f\n",
               t, x[0], x[1], x[2], x[3], x[4]);
    }

    printf("\n--- Test 2: KV-cached path (sequential tokens) ---\n");
    memset(k_cache, 0, N_KV_HEADS * MAX_SEQ * HEAD_DIM * 4);
    memset(v_cache, 0, N_KV_HEADS * MAX_SEQ * HEAD_DIM * 4);
    float out_cached[3][HIDDEN];
    for (int t = 0; t < 3; t++) {
        memcpy(x_c, emb[t], HIDDEN*4);
        forward_cached(lw, x_c, scr, ai8, bi, gate, up, k_cache, v_cache, t, MAX_SEQ);
        memcpy(out_cached[t], x_c, HIDDEN*4);
        printf("  Token %d: first 5 = %.2f %.2f %.2f %.2f %.2f\n",
               t, x_c[0], x_c[1], x_c[2], x_c[3], x_c[4]);
    }

    printf("\n--- Validation: single vs cached ---\n");
    // Token 0 should match (single-token attention is trivially V)
    double dot=0, ff=0, tt=0;
    for (int i = 0; i < HIDDEN; i++) {
        dot += (double)out_single[0][i] * out_cached[0][i];
        ff  += (double)out_single[0][i] * out_single[0][i];
        tt  += (double)out_cached[0][i] * out_cached[0][i];
    }
    float cs0 = (float)(dot / sqrt(ff * tt));
    printf("  Token 0 cos_sim (should ≈ 1.0): %.6f\n", cs0);

    // Token 1 should differ (cached has context from token 0)
    dot=0; ff=0; tt=0;
    for (int i = 0; i < HIDDEN; i++) {
        dot += (double)out_single[1][i] * out_cached[1][i];
        ff  += (double)out_single[1][i] * out_single[1][i];
        tt  += (double)out_cached[1][i] * out_cached[1][i];
    }
    float cs1 = (float)(dot / sqrt(ff * tt));
    printf("  Token 1 cos_sim (should differ from 1.0): %.6f\n", cs1);

    // Token 2 should also differ
    dot=0; ff=0; tt=0;
    for (int i = 0; i < HIDDEN; i++) {
        dot += (double)out_single[2][i] * out_cached[2][i];
        ff  += (double)out_single[2][i] * out_single[2][i];
        tt  += (double)out_cached[2][i] * out_cached[2][i];
    }
    float cs2 = (float)(dot / sqrt(ff * tt));
    printf("  Token 2 cos_sim (should differ from 1.0): %.6f\n", cs2);

    _aligned_free(x); _aligned_free(x_c); _aligned_free(scr);
    _aligned_free(ai8); _aligned_free(bi); _aligned_free(gate); _aligned_free(up);
    _aligned_free(k_cache); _aligned_free(v_cache);
    return 0;
}
