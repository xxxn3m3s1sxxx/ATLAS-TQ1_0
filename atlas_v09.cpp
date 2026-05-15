// Atlas v0.9 - REAL Falcon3 Weight Benchmark
// clang++ -O3 -mavx2 atlas_v09.cpp -o atlas_v09.exe

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

// FP32 SIMD baseline
float dot_fp32(const float* w, const float* x, int n) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256 wv = _mm256_loadu_ps(&w[i]);
        __m256 xv = _mm256_loadu_ps(&x[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(wv, xv));
    }
    return hsum(sum);
}

// sign_epi8 kernel for ternary weights {-1, 0, +1}
int dot_sign8(const int8_t* w, const int8_t* a, int n) {
    __m256i sum16 = _mm256_setzero_si256();
    for (int i = 0; i < n; i += 32) {
        __m256i wv = _mm256_loadu_si256((__m256i*)&w[i]);
        __m256i av = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i r = _mm256_sign_epi8(av, wv);
        __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r));
        __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r, 1));
        sum16 = _mm256_add_epi16(sum16, _mm256_add_epi16(lo, hi));
    }
    __m256i sum32 = _mm256_madd_epi16(sum16, _mm256_set1_epi16(1));
    // Horizontal add
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum32), _mm256_extracti128_si256(sum32, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

// Safe int8 maddubs kernel: widen to int16 first to avoid maddubs_epi16 overflow
// (max product 128*255=32640, but maddubs sums pairs → 65280 > int16)
int dot_maddubs_i8(const int8_t* w, const uint8_t* a, int n) {
    __m256i sum32 = _mm256_setzero_si256();
    for (int i = 0; i < n; i += 16) {
        __m128i w8 = _mm_loadu_si128((__m128i*)&w[i]);
        __m128i a8 = _mm_loadu_si128((__m128i*)&a[i]);
        __m256i w16 = _mm256_cvtepi8_epi16(w8);
        __m256i a16 = _mm256_cvtepu8_epi16(a8);
        __m256i prod16 = _mm256_mullo_epi16(w16, a16);
        sum32 = _mm256_add_epi32(sum32, _mm256_madd_epi16(prod16, _mm256_set1_epi16(1)));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum32), _mm256_extracti128_si256(sum32, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

int main(int argc, char** argv) {
    // Load Falcon3 weights
    const char* bin_path = "C:\\dam\\atlas\\falcon3_qproj_u8.bin";
    const char* fp32_path = "C:\\dam\\atlas\\falcon3_qproj_fp32.bin";

    FILE* f = fopen(bin_path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", bin_path); return 1; }
    int rows, cols;
    fread(&rows, 4, 1, f);
    fread(&cols, 4, 1, f);
    printf("Loading Falcon3 QKV: %d x %d\n", rows, cols);

    int8_t* w_i8 = (int8_t*)_aligned_malloc(rows * cols, 32);
    fread(w_i8, 1, rows * cols, f);
    fclose(f);

    // Load FP32 weights
    f = fopen(fp32_path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", fp32_path); return 1; }
    int r2, c2;
    fread(&r2, 4, 1, f); fread(&c2, 4, 1, f);
    float* w_fp32 = (float*)_aligned_malloc(r2 * c2 * sizeof(float), 32);
    fread(w_fp32, sizeof(float), r2 * c2, f);
    fclose(f);

    // Create ternary version of weights: sign(w_i8) → {-1, 0, +1}
    int8_t* w_ternary = (int8_t*)_aligned_malloc(rows * cols, 32);
    for (int i = 0; i < rows * cols; i++) {
        if (w_i8[i] > 0) w_ternary[i] = 1;
        else if (w_i8[i] < 0) w_ternary[i] = -1;
        else w_ternary[i] = 0;
    }

    // Create input activations (random, realistic range)
    float* input_fp32 = (float*)_aligned_malloc(cols * sizeof(float), 32);
    int8_t* input_i8 = (int8_t*)_aligned_malloc(cols, 32);
    uint8_t* input_u8 = (uint8_t*)_aligned_malloc(cols, 32);
    srand(42);
    for (int i = 0; i < cols; i++) {
        input_fp32[i] = (float)(rand() % 200 - 100) / 100.0f;
        input_i8[i] = (int8_t)(input_fp32[i] * 127.0f);
        input_u8[i] = (uint8_t)(input_fp32[i] * 127.0f + 128.5f);  // round to nearest
    }

    float* out_fp32 = (float*)_aligned_malloc(rows * sizeof(float), 32);
    int* out_sign = (int*)_aligned_malloc(rows * sizeof(int), 32);
    int* out_madd = (int*)_aligned_malloc(rows * sizeof(int), 32);
    clock_t t;

    // Pre-compute sum of each weight row for maddubs bias correction
    // maddubs_raw = sum(w_i8 * (a*127 + 128)) = 127*sum(w_i8*a) + 128*sum(w_i8)
    int32_t* row_sum_w = (int32_t*)_aligned_malloc(rows * sizeof(int32_t), 32);
    for (int r = 0; r < rows; r++) {
        int32_t s = 0;
        for (int i = 0; i < cols; i++) s += (int32_t)w_i8[r * cols + i];
        row_sum_w[r] = s;
    }

    const int ITERS = 1000;

    // --- Verification: single-row dot product with all methods ---
    printf("\n--- Single-Row Verification ---\n");
    float ref0 = dot_fp32(&w_fp32[0], input_fp32, cols);  // w_i8 * a_fp32
    int madd_raw = dot_maddubs_i8(&w_i8[0], input_u8, cols);
    float madd_corr = (float)(madd_raw - 128 * row_sum_w[0]) / 127.0f;
    // Manual: compute sum(w_i8[i] * round(a_fp32[i]*127))/127 in FP32
    float manual = 0;
    for (int i = 0; i < cols; i++)
        manual += (float)w_i8[i] * (float)((int)(input_fp32[i] * 127.0f + 128.5f) - 128) / 127.0f;
    printf("  FP32 baseline (w_i8 * a_fp32):      %.4f\n", ref0);
    printf("  maddubs (bias-corrected):            %.4f\n", madd_corr);
    printf("  Manual int8->FP32 simulation:        %.4f\n", manual);
    printf("  Ratio FP32 / maddubs_corrected:      %.4f\n", ref0 / madd_corr);
    printf("  maddubs raw (before bias corr):      %d\n", madd_raw);
    printf("  sum(w_i8) for row 0:                 %d\n", row_sum_w[0]);

    // Debug: maddubs kernel on first 32 elements vs manual
    int sub_n = 32;
    int madd32 = dot_maddubs_i8(&w_i8[0], input_u8, sub_n);
    int sum_w32 = 0;
    for (int i = 0; i < sub_n; i++) sum_w32 += w_i8[i];
    float corr32 = (float)(madd32 - 128 * sum_w32) / 127.0f;
    float man32 = 0;
    for (int i = 0; i < sub_n; i++)
        man32 += (float)w_i8[i] * (float)((int)(input_fp32[i] * 127.0f + 128.5f) - 128) / 127.0f;
    printf("  First 32: maddubs_corrected=%.4f  manual=%.4f  sum_w32=%d\n", corr32, man32, sum_w32);
    // First 5 element details
    for (int i = 0; i < 5; i++) {
        float x = input_fp32[i] * 127.0f;
        int u8v = input_u8[i];
        printf("  [%d]: w_i8=%d  a=%.3f  x=%.2f  u8=%d  round(x)=%d\n",
               i, w_i8[i], input_fp32[i], x, u8v, (int)(x + 128.5f) - 128);
    }

    printf("\n--- REAL Falcon3 Layer Benchmark (%d iters each) ---\n", ITERS);

    // FP32 baseline
    t = clock();
    for (int iter = 0; iter < ITERS; iter++)
        for (int r = 0; r < rows; r++)
            out_fp32[r] = dot_fp32(&w_fp32[r * cols], input_fp32, cols);
    double t_fp32 = (double)(clock() - t) / CLOCKS_PER_SEC;
    printf("FP32 SIMD:        %.4f sec  |  %.6f ms/iter  |  1.00x\n", t_fp32, t_fp32/ITERS*1000);

    // sign_epi8 on TERNARY weights
    t = clock();
    for (int iter = 0; iter < ITERS; iter++)
        for (int r = 0; r < rows; r++)
            out_sign[r] = dot_sign8(&w_ternary[r * cols], input_i8, cols);
    double t_sign = (double)(clock() - t) / CLOCKS_PER_SEC;

    // Quality: FP32 vs sign_epi8
    double dot_ft = 0, dot_ff = 0, dot_tt = 0;
    for (int r = 0; r < rows; r++) {
        float s = (float)out_sign[r];
        float f = out_fp32[r];
        dot_ft += f * s;
        dot_ff += f * f;
        dot_tt += s * s;
    }
    float cos_sim_sign = (float)(dot_ft / sqrt(dot_ff * dot_tt));

    // maddubs on raw int8 weights WITH uint8 activations
    t = clock();
    for (int iter = 0; iter < ITERS; iter++)
        for (int r = 0; r < rows; r++)
            out_madd[r] = dot_maddubs_i8(&w_i8[r * cols], input_u8, cols);
    double t_madd = (double)(clock() - t) / CLOCKS_PER_SEC;

    // Quality: FP32 vs bias-corrected maddubs
    // maddubs_raw = sum(w_i8 * (a*127 + 128)) = 127*dot + 128*sum(w)
    // corrected = (maddubs_raw - 128*sum(w)) / 127 = dot = sum(w_i8 * a_fp32)
    // FP32 = scale * dot → perfectly correlated after bias correction
    dot_ft = 0; dot_ff = 0; dot_tt = 0;
    for (int r = 0; r < rows; r++) {
        float m = (float)(out_madd[r] - 128 * row_sum_w[r]) / 127.0f;
        float f = out_fp32[r];
        dot_ft += f * m;
        dot_ff += f * f;
        dot_tt += m * m;
    }
    float cos_sim_madd = (float)(dot_ft / sqrt(dot_ff * dot_tt));

    printf("sign_epi8 ternary: %.4f sec  |  %.6f ms/iter  |  %.2fx  |  cos_sim=%.4f\n",
           t_sign, t_sign/ITERS*1000, t_fp32/t_sign, cos_sim_sign);
    printf("maddubs int8:     %.4f sec  |  %.6f ms/iter  |  %.2fx  |  cos_sim=%.4f\n",
           t_madd, t_madd/ITERS*1000, t_fp32/t_madd, cos_sim_madd);

    // Print first 5 outputs
    printf("\nFirst 5 outputs:\n");
    for (int i = 0; i < 5; i++) {
        float m_corrected = (float)(out_madd[i] - 128 * row_sum_w[i]) / 127.0f;
        printf("  FP32=%.2f  sign8=%d  maddubs_corrected=%.2f\n",
               out_fp32[i], out_sign[i], m_corrected);
    }

    // Memory summary
    double mem_fp32 = (double)rows * cols * 4 / 1e6;
    double mem_i8 = (double)rows * cols / 1e6;
    printf("\nMemory per layer:\n");
    printf("  FP32:  %.1f MB\n", mem_fp32);
    printf("  int8:  %.1f MB (4x smaller)\n", mem_i8);

    _aligned_free(w_i8); _aligned_free(w_fp32); _aligned_free(w_ternary);
    _aligned_free(input_fp32); _aligned_free(input_i8); _aligned_free(input_u8);
    _aligned_free(out_fp32); _aligned_free(out_sign); _aligned_free(out_madd);
    _aligned_free(row_sum_w);
    return 0;
}
