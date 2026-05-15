// Atlas v1.0 - REAL Falcon3 Ternary Benchmark
// Properly unpacked ternary {-1,0,+1} weights + dequantized FP32 reference
// clang++ -O3 -mavx2 atlas_v10.cpp -o atlas_v10.exe

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

float dot_fp32(const float* w, const float* x, int n) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256 wv = _mm256_loadu_ps(&w[i]);
        __m256 xv = _mm256_loadu_ps(&x[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(wv, xv));
    }
    return hsum(sum);
}

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
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum32), _mm256_extracti128_si256(sum32, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

int main(int argc, char** argv) {
    const char* tern_path = "C:\\dam\\atlas\\falcon3_qproj_u8.bin";
    const char* fp32_path = "C:\\dam\\atlas\\falcon3_qproj_fp32.bin";

    // Load ternary weights (already {-1, 0, +1})
    FILE* f = fopen(tern_path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", tern_path); return 1; }
    int rows, cols;
    fread(&rows, 4, 1, f);
    fread(&cols, 4, 1, f);
    printf("Loading Falcon3 QKV: %d x %d\n", rows, cols);

    int8_t* w_tern = (int8_t*)_aligned_malloc(rows * cols, 32);
    fread(w_tern, 1, rows * cols, f);
    fclose(f);

    // Load FP32 dequantized reference (ternary * scale)
    f = fopen(fp32_path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", fp32_path); return 1; }
    int r2, c2;
    fread(&r2, 4, 1, f); fread(&c2, 4, 1, f);
    float* w_fp32 = (float*)_aligned_malloc(r2 * c2 * sizeof(float), 32);
    fread(w_fp32, sizeof(float), r2 * c2, f);
    fclose(f);

    // Verify ternary distribution
    int neg=0, zero=0, pos=0;
    for (int i = 0; i < rows * cols; i++) {
        if (w_tern[i] < 0) neg++;
        else if (w_tern[i] == 0) zero++;
        else pos++;
    }
    printf("Ternary distribution: neg=%d  zero=%d  pos=%d  (total=%d)\n", neg, zero, pos, rows*cols);

    // Create random input activations
    float* input_fp32 = (float*)_aligned_malloc(cols * sizeof(float), 32);
    int8_t* input_i8 = (int8_t*)_aligned_malloc(cols, 32);
    srand(42);
    for (int i = 0; i < cols; i++) {
        input_fp32[i] = (float)(rand() % 200 - 100) / 100.0f;
        input_i8[i] = (int8_t)(input_fp32[i] * 127.0f);
    }

    float* out_fp32 = (float*)_aligned_malloc(rows * sizeof(float), 32);
    int* out_sign = (int*)_aligned_malloc(rows * sizeof(int), 32);
    clock_t t;
    int ITERS = 1000;

    printf("\n--- Falcon3 TERNARY Benchmark (%d iters each) ---\n", ITERS);

    // FP32 baseline (dequantized ternary * scale)
    t = clock();
    for (int iter = 0; iter < ITERS; iter++)
        for (int r = 0; r < rows; r++)
            out_fp32[r] = dot_fp32(&w_fp32[r * cols], input_fp32, cols);
    double t_fp32 = (double)(clock() - t) / CLOCKS_PER_SEC;
    printf("FP32 dequantized:  %.4f sec  |  %.6f ms/iter  |  1.00x\n", t_fp32, t_fp32/ITERS*1000);

    // sign_epi8 on TRUE ternary {-1, 0, +1} weights
    t = clock();
    for (int iter = 0; iter < ITERS; iter++)
        for (int r = 0; r < rows; r++)
            out_sign[r] = dot_sign8(&w_tern[r * cols], input_i8, cols);
    double t_sign = (double)(clock() - t) / CLOCKS_PER_SEC;

    // Cos similarity: FP32 vs sign_epi8
    double dot_ft = 0, dot_ff = 0, dot_tt = 0;
    for (int r = 0; r < rows; r++) {
        float s = (float)out_sign[r];
        float f = out_fp32[r];
        dot_ft += f * s;
        dot_ff += f * f;
        dot_tt += s * s;
    }
    float cos_sim = (float)(dot_ft / sqrt(dot_ff * dot_tt));

    printf("sign_epi8 ternary: %.4f sec  |  %.6f ms/iter  |  %.2fx  |  cos_sim=%.4f\n",
           t_sign, t_sign/ITERS*1000, t_fp32/t_sign, cos_sim);

    // First 5 outputs
    printf("\nFirst 5 outputs:\n");
    for (int i = 0; i < 5; i++)
        printf("  FP32=%.2f  sign8=%d\n", out_fp32[i], out_sign[i]);

    double mem_fp32 = (double)rows * cols * 4 / 1e6;
    double mem_i8 = (double)rows * cols / 1e6;
    printf("\nMemory per layer:\n");
    printf("  FP32 dequant: %.1f MB\n", mem_fp32);
    printf("  int8 ternary: %.1f MB (4x smaller)\n", mem_i8);
    printf("  Packed U8:    %.1f MB (16x smaller)\n", mem_i8 / 4);

    _aligned_free(w_tern); _aligned_free(w_fp32);
    _aligned_free(input_fp32); _aligned_free(input_i8);
    _aligned_free(out_fp32); _aligned_free(out_sign);
    return 0;
}
