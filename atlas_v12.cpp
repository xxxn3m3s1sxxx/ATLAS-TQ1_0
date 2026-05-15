// Atlas v1.2 - Falcon3 Single-Layer Forward Pass (clean)
// sign_epi8 ternary kernel vs FP32 reference
// clang++ -O3 -mavx2 -lm atlas_v12.cpp -o atlas_v12.exe

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>

#define WEIGHT_DIR "C:\\dam\\atlas\\weights\\"

// Config from config.json
#define HIDDEN 3072
#define N_HEADS 12
#define N_KV_HEADS 4
#define HEAD_DIM 256
#define INTERMEDIATE 23040
#define N_LAYERS 28
#define EPS 1e-6f

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

// --- Binary weight I/O ---
static void* load_bin(const char* name, const char* suffix, int elem_size, int* rows, int* cols) {
    char path[512]; snprintf(path, sizeof(path), "%s%s_%s", WEIGHT_DIR, name, suffix);
    FILE* f = fopen(path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", path); exit(1); }
    fread(rows, 4, 1, f); fread(cols, 4, 1, f);
    void* data = _aligned_malloc((*rows) * (*cols) * elem_size, 32);
    fread(data, elem_size, (*rows) * (*cols), f);
    fclose(f);
    printf("  %-20s [%5d x %5d] %7.1f MB\n", name, *rows, *cols,
           (double)(*rows) * (*cols) * elem_size / 1e6);
    return data;
}
#define load_tern(n, r, c) (int8_t*)load_bin(n, "tern.bin", 1, r, c)
#define load_fp32(n, r, c) (float*)load_bin(n, "fp32.bin", 4, r, c)

static float* load_vec_bin(const char* name, int* n) {
    char path[512]; snprintf(path, sizeof(path), "%s%s", WEIGHT_DIR, name);
    FILE* f = fopen(path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", path); exit(1); }
    int r, c; fread(&r, 4, 1, f); fread(&c, 4, 1, f);
    *n = c;
    float* data = (float*)_aligned_malloc(c * sizeof(float), 32);
    fread(data, sizeof(float), c, f);
    fclose(f);
    return data;
}

// --- FP32 MV ---
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

// --- sign_epi8 ternary MV ---
// Returns int32 dot products. To convert to FP32: y_fp32[r] = scale * buf[r] / 127
static void mv_tern(const int8_t* w, const int8_t* a, int* y, int rows, int cols) {
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
static void rms_norm(float* x, const float* weight, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) x[i] *= inv * weight[i];
}

// --- SiLU ---
static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }
static void silu_vec(float* x, int n) { for (int i = 0; i < n; i++) x[i] = silu_f(x[i]); }

// --- Quantize float vector to int8 with dynamic scaling ---
// Returns the scale factor used (so result_fp32 = result_i8 * scale / 127)
static float quantize_dyn(const float* src, int8_t* dst, int n) {
    float max_abs = 1e-10f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > max_abs) max_abs = a;
    }
    float qscale = max_abs / 127.0f;
    for (int i = 0; i < n; i++) dst[i] = (int8_t)(src[i] / qscale);
    return qscale;
}

// --- Dequantize int32 to float with two scale factors ---
// mv computes: int32 = sum(w_tern * a_i8), a_i8 = a_fp32 / qscale
// Desired: y_fp32 = w_scale * sum(w_tern * a_fp32)
// Actual: y_fp32 = w_scale * sum(w_tern * a_i8 * qscale) = w_scale * qscale * sum(w_tern * a_i8)
// So: y_fp32 = w_scale * qscale * int32_result
static void dequant_dyn(const int* src, float* dst, float w_scale, float qscale, int n) {
    float s = w_scale * qscale / 127.0f;
    for (int i = 0; i < n; i++) dst[i] = s * src[i];
}

// ============================================================
// Layer forward passes
// ============================================================

struct LayerWeights {
    // FP32 dequantized weights
    float *q, *k, *v, *o, *gate, *up, *down;
    float *ln1, *ln2;  // rms norm weights
    // Ternary weights
    int8_t *tq, *tk, *tv, *to, *tgate, *tup, *tdown;
    float sq, sk, sv, so, sgate, sup, sdown;
};

static void load_layer(struct LayerWeights* lw) {
    printf("Loading layer 0 weights:\n");
    int r, c;
    lw->q  = load_fp32("layer0_q_proj", &r, &c);
    lw->k  = load_fp32("layer0_k_proj", &r, &c);
    lw->v  = load_fp32("layer0_v_proj", &r, &c);
    lw->o  = load_fp32("layer0_o_proj", &r, &c);
    lw->gate = load_fp32("layer0_gate_proj", &r, &c);
    lw->up  = load_fp32("layer0_up_proj", &r, &c);
    lw->down = load_fp32("layer0_down_proj", &r, &c);
    lw->ln1 = load_vec_bin("layer0_input_layernorm.bin", &r);
    lw->ln2 = load_vec_bin("layer0_post_attention_layernorm.bin", &r);
    // Ternary
    lw->tq = load_tern("layer0_q_proj", &r, &c);
    lw->tk = load_tern("layer0_k_proj", &r, &c);
    lw->tv = load_tern("layer0_v_proj", &r, &c);
    lw->to = load_tern("layer0_o_proj", &r, &c);
    lw->tgate = load_tern("layer0_gate_proj", &r, &c);
    lw->tup = load_tern("layer0_up_proj", &r, &c);
    lw->tdown = load_tern("layer0_down_proj", &r, &c);

    // Extract scales from dequantized data: scale = max(abs(w_fp32))
    // Since w_tern = {-1,0,+1} and w_fp32 = w_tern * scale
    auto extract_scale = [](const float* w, int n) {
        float mx = 0;
        for (int i = 0; i < n; i++) if (fabsf(w[i]) > mx) mx = fabsf(w[i]);
        return mx;
    };
    lw->sq = extract_scale(lw->q, 3072*3072);
    lw->sk = extract_scale(lw->k, 1024*3072);
    lw->sv = extract_scale(lw->v, 1024*3072);
    lw->so = extract_scale(lw->o, 3072*3072);
    lw->sgate = extract_scale(lw->gate, 23040*3072);
    lw->sup = extract_scale(lw->up, 23040*3072);
    lw->sdown = extract_scale(lw->down, 3072*23040);
    printf("Scales: q=%.2f k=%.2f v=%.2f o=%.2f gate=%.2f up=%.2f down=%.2f\n",
           lw->sq, lw->sk, lw->sv, lw->so, lw->sgate, lw->sup, lw->sdown);
}

// --- FP32 reference layer ---
// Input x is modified in-place
static void forward_fp32(const struct LayerWeights* lw, float* x,
                         float* buf_qkv, float* buf_gate, float* buf_up, float* buf_ffn)
{
    int d = HIDDEN, hd = HEAD_DIM, inter = INTERMEDIATE;
    int nh = N_HEADS, nkv = N_KV_HEADS, g = nh / nkv;

    // Attention
    memcpy(buf_qkv, x, d * sizeof(float));  // residual
    rms_norm(x, lw->ln1, d, EPS);

    mv_fp32(lw->q, x, buf_qkv + d, nh * hd, d);          // Q
    mv_fp32(lw->k, x, buf_qkv + d + nh*hd, nkv * hd, d); // K
    mv_fp32(lw->v, x, buf_qkv + d + nh*hd + nkv*hd, nkv * hd, d); // V

    // GQA: expand V by repeating per group
    float* v = buf_qkv + d + nh * hd + nkv * hd;
    for (int h = 0; h < nh; h++)
        memcpy(&buf_qkv[h*hd], &v[(h/g)*hd], hd * sizeof(float));

    // O_proj
    mv_fp32(lw->o, buf_qkv, buf_qkv + d + nh*hd + nkv*hd, d, nh * hd);
    for (int i = 0; i < d; i++) x[i] += buf_qkv[d + nh*hd + nkv*hd + i];  // residual

    // FFN
    memcpy(buf_qkv, x, d * sizeof(float));  // residual
    rms_norm(x, lw->ln2, d, EPS);

    mv_fp32(lw->gate, x, buf_gate, inter, d);
    mv_fp32(lw->up, x, buf_up, inter, d);
    silu_vec(buf_gate, inter);
    for (int i = 0; i < inter; i++) buf_ffn[i] = buf_gate[i] * buf_up[i];

    mv_fp32(lw->down, buf_ffn, buf_ffn + inter, d, inter);
    for (int i = 0; i < d; i++) x[i] += buf_ffn[inter + i];  // residual
}

// --- Ternary forward pass (sign_epi8 kernel) ---
// Uses int8 temporary buffer for sign_epi8 input
static void forward_ternary(const struct LayerWeights* lw, float* x,
                            int8_t* ai8, int* buf_int, float* buf_fp,
                            float* buf_gate, float* buf_up)
{
    int d = HIDDEN, hd = HEAD_DIM, inter = INTERMEDIATE;
    int nh = N_HEADS, nkv = N_KV_HEADS, g = nh / nkv;

    // --- Attention ---
    memcpy(buf_fp, x, d * sizeof(float));  // save residual
    rms_norm(x, lw->ln1, d, EPS);

    quantize(x, ai8, d);
    mv_tern(lw->tq, ai8, buf_int, nh * hd, d);
    dequant(buf_int, buf_fp + d, lw->sq, nh * hd);  // Q in buf_fp+d

    mv_tern(lw->tk, ai8, buf_int, nkv * hd, d);
    dequant(buf_int, buf_fp + d + nh*hd, lw->sk, nkv * hd);  // K

    mv_tern(lw->tv, ai8, buf_int, nkv * hd, d);
    dequant(buf_int, buf_fp + d + nh*hd + nkv*hd, lw->sv, nkv * hd);  // V

    // GQA: expand V
    float* v = buf_fp + d + nh * hd + nkv * hd;
    for (int h = 0; h < nh; h++)
        memcpy(&buf_fp[h*hd], &v[(h/g)*hd], hd * sizeof(float));

    // O_proj: quantize attn output, MV, dequant, add residual
    quantize(buf_fp, ai8, nh * hd);
    mv_tern(lw->to, ai8, buf_int, d, nh * hd);
    dequant(buf_int, buf_fp + d + nh*hd + nkv*hd, lw->so, d);
    for (int i = 0; i < d; i++) x[i] = buf_fp[i] + buf_fp[d + nh*hd + nkv*hd + i];

    // --- FFN ---
    memcpy(buf_fp, x, d * sizeof(float));  // residual
    rms_norm(x, lw->ln2, d, EPS);

    quantize(x, ai8, d);
    mv_tern(lw->tgate, ai8, buf_int, inter, d);
    dequant(buf_int, buf_gate, lw->sgate, inter);
    silu_vec(buf_gate, inter);

    mv_tern(lw->tup, ai8, buf_int, inter, d);
    dequant(buf_int, buf_up, lw->sup, inter);
    for (int i = 0; i < inter; i++) buf_up[i] *= buf_gate[i];  // ffn = silu(gate) * up

    quantize(buf_up, ai8, inter);
    mv_tern(lw->tdown, ai8, buf_int, d, inter);
    dequant(buf_int, x, lw->sdown, d);  // reuse x for down output
    for (int i = 0; i < d; i++) x[i] += buf_fp[i];  // + residual
}

// ============================================================
int main() {
    struct LayerWeights lw;
    load_layer(&lw);

    // Allocate buffers
    float* x_fp32 = (float*)_aligned_malloc(HIDDEN * sizeof(float), 32);
    float* x_tern = (float*)_aligned_malloc(HIDDEN * sizeof(float), 32);

    // Random test input
    float* x_ref = (float*)_aligned_malloc(HIDDEN * sizeof(float), 32);
    srand(42);
    for (int i = 0; i < HIDDEN; i++) {
        float v = (float)(rand() % 200 - 100) / 100.0f;
        x_ref[i] = v;
        x_fp32[i] = v;
        x_tern[i] = v;
    }

    // FP32 buffers
    float* buf_qkv = (float*)_aligned_malloc((HIDDEN + N_HEADS*HEAD_DIM + N_KV_HEADS*HEAD_DIM*2) * sizeof(float), 32);
    float* buf_gate = (float*)_aligned_malloc(INTERMEDIATE * sizeof(float), 32);
    float* buf_up = (float*)_aligned_malloc(INTERMEDIATE * sizeof(float), 32);
    float* buf_ffn = (float*)_aligned_malloc((INTERMEDIATE + HIDDEN) * sizeof(float), 32);

    // Ternary buffers
    int8_t* ai8 = (int8_t*)_aligned_malloc((INTERMEDIATE > HIDDEN ? INTERMEDIATE : HIDDEN), 32);
    int* buf_int = (int*)_aligned_malloc((INTERMEDIATE > HIDDEN ? INTERMEDIATE : HIDDEN) * sizeof(int), 32);
    float* buf_fp = (float*)_aligned_malloc(
        (HIDDEN + N_HEADS*HEAD_DIM + N_KV_HEADS*HEAD_DIM*2) * sizeof(float), 32);
    float* tern_gate = (float*)_aligned_malloc(INTERMEDIATE * sizeof(float), 32);
    float* tern_up   = (float*)_aligned_malloc(INTERMEDIATE * sizeof(float), 32);

    printf("\nRunning FP32 reference layer...\n");
    clock_t t = clock();
    int ITERS = 100;
    for (int iter = 0; iter < ITERS; iter++) {
        memcpy(x_fp32, x_ref, HIDDEN * sizeof(float));
        forward_fp32(&lw, x_fp32, buf_qkv, buf_gate, buf_up, buf_ffn);
    }
    double t_fp32 = (double)(clock() - t) / CLOCKS_PER_SEC;
    printf("FP32:  %.4f sec  |  %.4f ms/iter\n", t_fp32, t_fp32/ITERS*1000);

    printf("\nRunning ternary (sign_epi8) layer...\n");
    t = clock();
    for (int iter = 0; iter < ITERS; iter++) {
        memcpy(x_tern, x_ref, HIDDEN * sizeof(float));
        forward_ternary(&lw, x_tern, ai8, buf_int, buf_fp, tern_gate, tern_up);
    }
    double t_tern = (double)(clock() - t) / CLOCKS_PER_SEC;
    printf("TERN: %.4f sec  |  %.4f ms/iter\n", t_tern, t_tern/ITERS*1000);
    printf("Speedup: %.2fx\n", t_fp32 / t_tern);

    // Compare last iteration outputs
    // (Need to re-run one more time to capture results — already done in loop)
    double dot_ft = 0, dot_ff = 0, dot_tt = 0;
    for (int i = 0; i < HIDDEN; i++) {
        dot_ft += (double)x_fp32[i] * x_tern[i];
        dot_ff += (double)x_fp32[i] * x_fp32[i];
        dot_tt += (double)x_tern[i] * x_tern[i];
    }
    float cos_sim = (float)(dot_ft / sqrt(dot_ff * dot_tt));
    printf("\nCosine similarity (FP32 vs ternary layer output): %.6f\n", cos_sim);

    printf("\nFirst 10 outputs:\n");
    for (int i = 0; i < 10; i++)
        printf("  [%3d] FP32=%.4f  TERN=%.4f  diff=%.4f\n",
               i, x_fp32[i], x_tern[i], x_fp32[i] - x_tern[i]);

    // Cleanup
    _aligned_free(lw.q); _aligned_free(lw.k); _aligned_free(lw.v); _aligned_free(lw.o);
    _aligned_free(lw.gate); _aligned_free(lw.up); _aligned_free(lw.down);
    _aligned_free(lw.ln1); _aligned_free(lw.ln2);
    _aligned_free(lw.tq); _aligned_free(lw.tk); _aligned_free(lw.tv); _aligned_free(lw.to);
    _aligned_free(lw.tgate); _aligned_free(lw.tup); _aligned_free(lw.tdown);
    _aligned_free(x_fp32); _aligned_free(x_tern);
    _aligned_free(buf_qkv); _aligned_free(buf_gate); _aligned_free(buf_up); _aligned_free(buf_ffn);
    _aligned_free(ai8); _aligned_free(buf_int); _aligned_free(buf_fp);
    _aligned_free(tern_gate); _aligned_free(tern_up);
    return 0;
}
