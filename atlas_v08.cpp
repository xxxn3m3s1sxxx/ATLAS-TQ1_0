// Atlas v0.8 - THE BREAKTHROUGH: _mm256_sign_epi8 + int8 ternary
// clang++ -O3 -mavx2 -mavx512f atlas_v08.cpp -o atlas_v08.exe 2>NUL || clang++ -O3 -mavx2 atlas_v08.cpp -o atlas_v08.exe

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>

static inline float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static inline int hsum_i(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

// ---- FP32 BASELINE (SIMD MUL+ADD) ----
float dot_fp32(const float* w, const float* x, int n) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256 wv = _mm256_loadu_ps(&w[i]);
        __m256 xv = _mm256_loadu_ps(&x[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(wv, xv));
    }
    return hsum(sum);
}

// ---- ATLAS SIGN_EPI8 KERNEL (int8 weights + int8 activations) ----
// Assumes activations are already int8 (pre-quantized)
int dot_sign8(const int8_t* w, const int8_t* a, int n) {
    __m256i sum16 = _mm256_setzero_si256();
    for (int i = 0; i < n; i += 32) {
        __m256i wv = _mm256_loadu_si256((__m256i*)&w[i]);  // 32 int8: {-1,0,+1}
        __m256i av = _mm256_loadu_si256((__m256i*)&a[i]);  // 32 int8 activations
        __m256i r = _mm256_sign_epi8(av, wv);  // THE KEY: 1 instruction, 1 cycle, no MUL!
        // r[i] = a[i] if w[i]=+1, -a[i] if w[i]=-1, 0 if w[i]=0

        // Expand to int16 and accumulate
        __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r));
        __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r, 1));
        sum16 = _mm256_add_epi16(sum16, lo);
        sum16 = _mm256_add_epi16(sum16, hi);
    }

    // Reduce 16 int16 to 8 int32
    __m256i sum32 = _mm256_madd_epi16(sum16, _mm256_set1_epi16(1));
    return hsum_i(sum32);
}

// ---- ATLAS SIGN_EPI8 with on-the-fly float->int8 quantization ----
float dot_sign8_quant(const int8_t* w, const float* x, int n) {
    int sum_i = 0;
    for (int i = 0; i < n; i += 32) {
        __m256i wv = _mm256_loadu_si256((__m256i*)&w[i]);

        // Quantize 32 floats to 32 int8
        __m256 x0 = _mm256_loadu_ps(&x[i]);
        __m256 x1 = _mm256_loadu_ps(&x[i+8]);
        __m256 x2 = _mm256_loadu_ps(&x[i+16]);
        __m256 x3 = _mm256_loadu_ps(&x[i+24]);

        __m256i i0 = _mm256_cvtps_epi32(x0);
        __m256i i1 = _mm256_cvtps_epi32(x1);
        __m256i i2 = _mm256_cvtps_epi32(x2);
        __m256i i3 = _mm256_cvtps_epi32(x3);

        // Pack 8 int32 -> 16 int16 -> 32 int8
        __m256i p16_0 = _mm256_packs_epi32(i0, i1);
        __m256i p16_1 = _mm256_packs_epi32(i2, i3);
        __m256i av = _mm256_packs_epi16(p16_0, p16_1);

        __m256i r = _mm256_sign_epi8(av, wv);

        __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r));
        __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r, 1));
        sum_i += hsum_i(_mm256_madd_epi16(_mm256_add_epi16(lo, hi), _mm256_set1_epi16(1)));
    }
    return (float)sum_i;
}

// ---- int8 MatMul (standard, no ternary) for comparison ----
int dot_int8_std(const int8_t* w, const int8_t* a, int n) {
    __m256i sum16 = _mm256_setzero_si256();
    for (int i = 0; i < n; i += 32) {
        __m256i wv = _mm256_loadu_si256((__m256i*)&w[i]);
        __m256i av = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i prod = _mm256_maddubs_epi16(av, wv);  // standard W8A8 multiply
        // maddubs: av is unsigned, wv is signed
        sum16 = _mm256_add_epi16(sum16, prod);
    }
    __m256i sum32 = _mm256_madd_epi16(sum16, _mm256_set1_epi16(1));
    return hsum_i(sum32);
}

int main() {
    srand(42);

    const int dim = 4096;
    const int rows = 4096;
    const int nlayers = 32;

    printf("=== Atlas v0.8 - _mm256_sign_epi8 BREAKTHROUGH ===\n");
    printf("CPU: i7-7700T, AVX2, 8 MB L3\n");
    printf("Dim: %d, Rows: %d, Layers: %d\n\n", dim, rows, nlayers);

    // FP32 data
    float* w_fp32 = (float*)_aligned_malloc(rows * dim * sizeof(float), 32);
    float* input_fp32 = (float*)_aligned_malloc(dim * sizeof(float), 32);
    memset(w_fp32, 0, rows * dim * sizeof(float));

    // int8 data: ternary weights + quantized activations
    int8_t* w_i8 = (int8_t*)_aligned_malloc(rows * dim, 32);
    int8_t* a_i8 = (int8_t*)_aligned_malloc(dim, 32);

    // int8 standard weights (full range, for comparison)
    int8_t* w_i8_std = (int8_t*)_aligned_malloc(rows * dim, 32);

    for (int i = 0; i < dim; i++) {
        input_fp32[i] = (float)(rand() % 200 - 100) / 100.0f;
        a_i8[i] = (int8_t)(input_fp32[i] * 127.0f);  // quantize
    }

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < dim; c++) {
            int8_t tw = (int8_t)(rand() % 3 - 1);
            w_fp32[r * dim + c] = (float)tw;
            w_i8[r * dim + c] = tw;
            w_i8_std[r * dim + c] = (int8_t)(rand() % 256 - 128);
        }
    }

    clock_t t;
    float* out = (float*)_aligned_malloc(rows * sizeof(float), 32);
    int* out_i = (int*)_aligned_malloc(rows * sizeof(int), 32);

    // ---- SINGLE LAYER BENCHMARKS ----
    printf("--- Single Layer (4096-dim MatMul) ---\n");

    t = clock();
    for (int r = 0; r < rows; r++) out[r] = dot_fp32(&w_fp32[r * dim], input_fp32, dim);
    double t_fp32 = (double)(clock() - t) / CLOCKS_PER_SEC;

    t = clock();
    for (int r = 0; r < rows; r++) out_i[r] = dot_sign8(&w_i8[r * dim], a_i8, dim);
    double t_i8 = (double)(clock() - t) / CLOCKS_PER_SEC;

    t = clock();
    for (int r = 0; r < rows; r++) out[r] = dot_sign8_quant(&w_i8[r * dim], input_fp32, dim);
    double t_quant = (double)(clock() - t) / CLOCKS_PER_SEC;

    t = clock();
    for (int r = 0; r < rows; r++) out_i[r] = dot_int8_std(&w_i8_std[r * dim], a_i8, dim);
    double t_i8_std = (double)(clock() - t) / CLOCKS_PER_SEC;

    // Verify correctness (sample)
    float ref_v0 = dot_fp32(w_fp32, input_fp32, dim);
    float ref_v1 = (float)dot_sign8(w_i8, a_i8, dim);
    float ref_v2 = dot_sign8_quant(w_i8, input_fp32, dim);
    double delta = fabs(ref_v0 - ref_v1);

    printf("FP32   (SIMD MUL+ADD):   %.4f sec  |  1.00x\n", t_fp32);
    printf("i8 TERN (sign_epi8):    %.4f sec  |  %.2fx  Delta=%.2f %s\n",
           t_i8, t_fp32/t_i8, delta, delta < 500 ? "OK" : "CHECK");
    printf("i8 TERN (on-the-fly Q): %.4f sec  |  %.2fx\n", t_quant, t_fp32/t_quant);
    printf("i8 STANDARD (maddubs):  %.4f sec  |  %.2fx\n\n", t_i8_std, t_fp32/t_i8_std);

    // ---- 32 LAYERS BENCHMARK ----
    printf("--- %d Layers ---\n", nlayers);

    float** l_fp32 = (float**)_aligned_malloc(nlayers * sizeof(float*), 32);
    int8_t** l_i8 = (int8_t**)_aligned_malloc(nlayers * sizeof(int8_t*), 32);
    int8_t** l_i8s = (int8_t**)_aligned_malloc(nlayers * sizeof(int8_t*), 32);

    for (int l = 0; l < nlayers; l++) {
        l_fp32[l] = (float*)_aligned_malloc(rows * dim * sizeof(float), 32);
        l_i8[l] = (int8_t*)_aligned_malloc(rows * dim, 32);
        l_i8s[l] = (int8_t*)_aligned_malloc(rows * dim, 32);
        memcpy(l_fp32[l], w_fp32, rows * dim * sizeof(float));
        memcpy(l_i8[l], w_i8, rows * dim);
        memcpy(l_i8s[l], w_i8_std, rows * dim);
    }

    int reps = 5;

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int r = 0; r < rows; r++) out[r] = dot_fp32(&l_fp32[l][r * dim], input_fp32, dim);
    double t_fp32_ml = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int r = 0; r < rows; r++) out_i[r] = dot_sign8(&l_i8[l][r * dim], a_i8, dim);
    double t_i8_ml = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int r = 0; r < rows; r++) out[r] = dot_sign8_quant(&l_i8[l][r * dim], input_fp32, dim);
    double t_quant_ml = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int r = 0; r < rows; r++) out_i[r] = dot_int8_std(&l_i8s[l][r * dim], a_i8, dim);
    double t_i8s_ml = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    printf("FP32   (SIMD MUL+ADD):  %.4f sec  (%.4f/l)  |  1.00x\n", t_fp32_ml, t_fp32_ml/nlayers);
    printf("i8 TERN (sign_epi8):   %.4f sec  (%.4f/l)  |  %.2fx\n", t_i8_ml, t_i8_ml/nlayers, t_fp32_ml/t_i8_ml);
    printf("i8 TERN (on-the-fly Q):%.4f sec  (%.4f/l)  |  %.2fx\n", t_quant_ml, t_quant_ml/nlayers, t_fp32_ml/t_quant_ml);
    printf("i8 STANDARD (maddubs): %.4f sec  (%.4f/l)  |  %.2fx\n", t_i8s_ml, t_i8s_ml/nlayers, t_fp32_ml/t_i8s_ml);

    // Memory
    double mem_fp32 = (double)rows * dim * 4 * nlayers / 1e9;
    double mem_i8 = (double)rows * dim * nlayers / 1e9;
    printf("\n--- Memory ---\n");
    printf("FP32:  %.2f GB  |  int8: %.2f GB (4x smaller)\n", mem_fp32, mem_i8);

    // Cleanup
    for (int l = 0; l < nlayers; l++) {
        _aligned_free(l_fp32[l]); _aligned_free(l_i8[l]); _aligned_free(l_i8s[l]);
    }
    _aligned_free(l_fp32); _aligned_free(l_i8); _aligned_free(l_i8s);
    _aligned_free(w_fp32); _aligned_free(w_i8); _aligned_free(w_i8_std);
    _aligned_free(input_fp32); _aligned_free(a_i8);
    _aligned_free(out); _aligned_free(out_i);
    return 0;
}
