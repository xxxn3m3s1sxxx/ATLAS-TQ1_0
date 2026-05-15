// Atlas v1.3 - Falcon3 Single-Layer Forward Pass (clean dynamic quant)
// clang++ -O3 -mavx2 -lm atlas_v13.cpp -o atlas_v13.exe
// Verifies: 1) correctness 2) speed of sign_epi8 vs FP32

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

// ========== SIMD ==========
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

// ========== I/O ==========
static void* load_bin(const char* name, const char* suf, int esz, int* r, int* c) {
    char p[512]; snprintf(p, sizeof(p), "%s%s_%s", WDIR, name, suf);
    FILE* f = fopen(p, "rb"); if (!f) { printf("ERROR: %s\n", p); exit(1); }
    fread(r, 4, 1, f); fread(c, 4, 1, f);
    void* d = _aligned_malloc((*r) * (*c) * esz, 32);
    fread(d, esz, (*r) * (*c), f); fclose(f);
    printf("  %-20s [%5d x %5d] %8.1f MB\n", name, *r, *c, (double)(*r)*(*c)*esz/1e6);
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

// ========== Kernels ==========
static void mv_fp32(const float* w, const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        __m256 sum = _mm256_setzero_ps();
        for (int i = 0; i < cols; i += 8) {
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(&w[r*cols+i]), _mm256_loadu_ps(&x[i])));
        }
        y[r] = hsum_ps(sum);
    }
}

// sign_epi8 ternary MV (needs int8 input already in [-128, 127])
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

// Dynamic quantize: find max_abs, scale to fit int8, return qscale
static float quantize(const float* src, int8_t* dst, int n) {
    float ma = 1e-10f;
    for (int i = 0; i < n; i++) { float a = fabsf(src[i]); if (a > ma) ma = a; }
    float qs = ma / 127.0f;
    for (int i = 0; i < n; i++) dst[i] = (int8_t)(src[i] / qs);
    return qs;
}

// Dequant: output[r] = w_scale * qscale * int32[r]
// (qscale already includes 1/127 factor since qs = max_abs/127)
static void dequant(const int* src, float* dst, float ws, float qs, int n) {
    float s = ws * qs;
    for (int i = 0; i < n; i++) dst[i] = s * src[i];
}

// RMSNorm
static void rms_norm(float* x, const float* w, int n) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + EPS);
    for (int i = 0; i < n; i++) x[i] *= inv * w[i];
}

static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }

// GQA expand: in[4][256] → out[12][256] (repeating each 3x)
static void expand_gqa(float* out, const float* in, int nh, int nkv, int hd) {
    int g = nh / nkv;
    for (int h = 0; h < nh; h++) memcpy(&out[h*hd], &in[(h/g)*hd], hd * sizeof(float));
}

// ========== Layer Weights ==========
struct LayerW {
    float *q_fp, *k_fp, *v_fp, *o_fp, *g_fp, *u_fp, *d_fp;
    int8_t *q_t, *k_t, *v_t, *o_t, *g_t, *u_t, *d_t;
    float *ln1, *ln2, sq, sk, sv, so, sg, su, sd;
};

static LayerW load_layer() {
    LayerW l; int r, c;
    printf("Loading layer 0 weights:\n");
    l.q_fp = load_fp32("layer0_q_proj", &r, &c); l.sq = 48.75f;
    l.k_fp = load_fp32("layer0_k_proj", &r, &c); l.sk = 36.75f;
    l.v_fp = load_fp32("layer0_v_proj", &r, &c); l.sv = 26.75f;
    l.o_fp = load_fp32("layer0_o_proj", &r, &c); l.so = 27.875f;
    l.g_fp = load_fp32("layer0_gate_proj", &r, &c); l.sg = 11.4375f;
    l.u_fp = load_fp32("layer0_up_proj", &r, &c); l.su = 11.75f;
    l.d_fp = load_fp32("layer0_down_proj", &r, &c); l.sd = 10.875f;
    l.ln1 = load_vec("layer0_input_layernorm.bin", &r);
    l.ln2 = load_vec("layer0_post_attention_layernorm.bin", &r);
    l.q_t = load_tern("layer0_q_proj", &r, &c);
    l.k_t = load_tern("layer0_k_proj", &r, &c);
    l.v_t = load_tern("layer0_v_proj", &r, &c);
    l.o_t = load_tern("layer0_o_proj", &r, &c);
    l.g_t = load_tern("layer0_gate_proj", &r, &c);
    l.u_t = load_tern("layer0_up_proj", &r, &c);
    l.d_t = load_tern("layer0_down_proj", &r, &c);
    return l;
}

// ========== FP32 Layer Forward ==========
// Uses scratch[] of size [81920] (enough for largest intermediate)
// scratch layout: [residual=3072] [normed=3072] [q=3072] [kv=2048] [attn=3072] [gate=23040] [up=23040]
static void forward_fp32(const LayerW& lw, float* x, float* scr) {
    int d=HIDDEN, hd=HEAD_DIM, in=INTERMEDIATE, nh=N_HEADS, nkv=N_KV_HEADS;
    float *res = scr, *nrm = scr+d, *q = scr+2*d, *kv = scr+2*d+nh*hd;
    float *attn = scr+2*d+nh*hd+nkv*hd, *gate = scr+2*d+nh*hd+nkv*hd+nh*hd;

    // Attention
    memcpy(res, x, d*4); rms_norm(x, lw.ln1, d);
    mv_fp32(lw.q_fp, x, q, nh*hd, d);        // Q
    mv_fp32(lw.k_fp, x, kv, nkv*hd, d);      // K
    mv_fp32(lw.v_fp, x, kv+nkv*hd, nkv*hd, d); // V
    expand_gqa(attn, kv+nkv*hd, nh, nkv, hd);  // V → attn
    mv_fp32(lw.o_fp, attn, q, d, nh*hd);     // O
    for (int i = 0; i < d; i++) x[i] = res[i] + q[i];  // + residual

    // FFN
    memcpy(res, x, d*4); rms_norm(x, lw.ln2, d);
    mv_fp32(lw.g_fp, x, gate, in, d);        // gate
    mv_fp32(lw.u_fp, x, gate+in, in, d);     // up
    for (int i = 0; i < in; i++) gate[i] = silu_f(gate[i]) * gate[in+i];  // silu(g)*up
    mv_fp32(lw.d_fp, gate, q, d, in);        // down
    for (int i = 0; i < d; i++) x[i] = res[i] + q[i];  // + residual
}

// ========== Ternary Layer Forward (sign_epi8) ==========
// Uses scr_fp[81920] floats + scr_i8[max(3072,23040)] int8 + buf_int[max(3072,23040)] int
static void forward_tern(const LayerW& lw, float* x, float* scr_fp, int8_t* scr_i8, int* buf_int) {
    int d=HIDDEN, hd=HEAD_DIM, in=INTERMEDIATE, nh=N_HEADS, nkv=N_KV_HEADS;
    float *res = scr_fp, *nrm = scr_fp+d, *q = scr_fp+2*d, *kv = scr_fp+2*d+nh*hd;
    float *attn = scr_fp+2*d+nh*hd+nkv*hd, *gate = scr_fp+2*d+nh*hd+nkv*hd+nh*hd;

    // --- Attention ---
    memcpy(res, x, d*4);
    rms_norm(x, lw.ln1, d);

    float qs_q = quantize(x, scr_i8, d);
    mv_tern(lw.q_t, scr_i8, buf_int, nh*hd, d);
    dequant(buf_int, q, lw.sq, qs_q, nh*hd);

    float qs_k = quantize(x, scr_i8, d);
    mv_tern(lw.k_t, scr_i8, buf_int, nkv*hd, d);
    dequant(buf_int, kv, lw.sk, qs_k, nkv*hd);

    float qs_v = quantize(x, scr_i8, d);
    mv_tern(lw.v_t, scr_i8, buf_int, nkv*hd, d);
    dequant(buf_int, kv+nkv*hd, lw.sv, qs_v, nkv*hd);

    expand_gqa(attn, kv+nkv*hd, nh, nkv, hd);

    float qs_o = quantize(attn, scr_i8, nh*hd);
    mv_tern(lw.o_t, scr_i8, buf_int, d, nh*hd);
    dequant(buf_int, q, lw.so, qs_o, d);
    for (int i = 0; i < d; i++) x[i] = res[i] + q[i];

    // --- FFN ---
    memcpy(res, x, d*4);
    rms_norm(x, lw.ln2, d);

    float qs_g = quantize(x, scr_i8, d);
    mv_tern(lw.g_t, scr_i8, buf_int, in, d);
    dequant(buf_int, gate, lw.sg, qs_g, in);

    float qs_u = quantize(x, scr_i8, d);
    mv_tern(lw.u_t, scr_i8, buf_int, in, d);
    dequant(buf_int, gate+in, lw.su, qs_u, in);

    for (int i = 0; i < in; i++) gate[i] = silu_f(gate[i]) * gate[in+i];

    float qs_d = quantize(gate, scr_i8, in);
    mv_tern(lw.d_t, scr_i8, buf_int, d, in);
    dequant(buf_int, q, lw.sd, qs_d, d);
    for (int i = 0; i < d; i++) x[i] = res[i] + q[i];
}

// ========== Main ==========
int main() {
    LayerW lw = load_layer();
    float *x_fp = (float*)_aligned_malloc(HIDDEN*4, 32);
    float *x_tn = (float*)_aligned_malloc(HIDDEN*4, 32);
    float *x_ref = (float*)_aligned_malloc(HIDDEN*4, 32);
    float *scr   = (float*)_aligned_malloc(81920*4, 32);
    int8_t* ai8  = (int8_t*)_aligned_malloc(INTERMEDIATE, 32);
    int* buf_int = (int*)_aligned_malloc(INTERMEDIATE*4, 32);

    srand(42);
    for (int i = 0; i < HIDDEN; i++) {
        float v = (float)(rand()%200-100)/100.0f;
        x_ref[i] = v; x_fp[i] = v; x_tn[i] = v;
    }

    int ITERS = 50;
    clock_t t;

    printf("\nRunning FP32 reference (%d iters)...\n", ITERS);
    t = clock();
    for (int it = 0; it < ITERS; it++) { memcpy(x_fp, x_ref, HIDDEN*4); forward_fp32(lw, x_fp, scr); }
    double tfp = (double)(clock()-t)/CLOCKS_PER_SEC;
    printf("  FP32: %.4f sec  |  %.4f ms/iter\n", tfp, tfp/ITERS*1000);

    printf("Running ternary sign_epi8 (%d iters)...\n", ITERS);
    t = clock();
    for (int it = 0; it < ITERS; it++) { memcpy(x_tn, x_ref, HIDDEN*4); forward_tern(lw, x_tn, scr, ai8, buf_int); }
    double ttn = (double)(clock()-t)/CLOCKS_PER_SEC;
    printf("  TERN: %.4f sec  |  %.4f ms/iter  |  %.2fx vs FP32\n", ttn, ttn/ITERS*1000, tfp/ttn);

    double dot_f=0, dot_ff=0, dot_tt=0;
    for (int i = 0; i < HIDDEN; i++) {
        dot_f  += (double)x_fp[i] * x_tn[i];
        dot_ff += (double)x_fp[i] * x_fp[i];
        dot_tt += (double)x_tn[i] * x_tn[i];
    }
    printf("\nCosine similarity: %.6f\n", (float)(dot_f/sqrt(dot_ff*dot_tt)));
    printf("First 8 outputs:\n");
    for (int i = 0; i < 8; i++)
        printf("  [%d] FP32=%.4f  TERN=%.4f  diff=%.4f\n", i, x_fp[i], x_tn[i], x_fp[i]-x_tn[i]);

    _aligned_free(x_fp); _aligned_free(x_tn); _aligned_free(x_ref);
    _aligned_free(scr); _aligned_free(ai8); _aligned_free(buf_int);
    return 0;
}
