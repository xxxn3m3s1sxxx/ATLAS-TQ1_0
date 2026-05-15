// Atlas v0.4 - AVX2 SIMD Ternary Kernel
// Processes 32 values per iteration with sign-based ADD/SUB (zero MUL)
// clang++ -O3 -mavx2 atlas_v04.cpp -o atlas_v04.exe

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

// Atlas SIMD: 32 ternary values per iteration, zero MUL
// w_float is pre-unpacked float array {-1,0,+1} per row
float dot_atlas_simd(const float* w, const float* x, int n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 zero = _mm256_setzero_ps();

    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256 x0 = _mm256_loadu_ps(&x[i]);
        __m256 x1 = _mm256_loadu_ps(&x[i+8]);
        __m256 x2 = _mm256_loadu_ps(&x[i+16]);
        __m256 x3 = _mm256_loadu_ps(&x[i+24]);

        __m256 w0 = _mm256_loadu_ps(&w[i]);
        __m256 w1 = _mm256_loadu_ps(&w[i+8]);
        __m256 w2 = _mm256_loadu_ps(&w[i+16]);
        __m256 w3 = _mm256_loadu_ps(&w[i+24]);

        // Zero-multiplication: use sign-based ADD/SUB
        // mask_pos = (w > 0), mask_neg = (w < 0)
        __m256 pos0 = _mm256_cmp_ps(w0, zero, _CMP_GT_OS);
        __m256 pos1 = _mm256_cmp_ps(w1, zero, _CMP_GT_OS);
        __m256 pos2 = _mm256_cmp_ps(w2, zero, _CMP_GT_OS);
        __m256 pos3 = _mm256_cmp_ps(w3, zero, _CMP_GT_OS);

        __m256 neg0 = _mm256_cmp_ps(w0, zero, _CMP_LT_OS);
        __m256 neg1 = _mm256_cmp_ps(w1, zero, _CMP_LT_OS);
        __m256 neg2 = _mm256_cmp_ps(w2, zero, _CMP_LT_OS);
        __m256 neg3 = _mm256_cmp_ps(w3, zero, _CMP_LT_OS);

        // sum += AND(pos, x) - AND(neg, x)
        sum0 = _mm256_add_ps(sum0, _mm256_and_ps(pos0, x0));
        sum1 = _mm256_add_ps(sum1, _mm256_and_ps(pos1, x1));
        sum2 = _mm256_add_ps(sum2, _mm256_and_ps(pos2, x2));
        sum3 = _mm256_add_ps(sum3, _mm256_and_ps(pos3, x3));

        sum0 = _mm256_sub_ps(sum0, _mm256_and_ps(neg0, x0));
        sum1 = _mm256_sub_ps(sum1, _mm256_and_ps(neg1, x1));
        sum2 = _mm256_sub_ps(sum2, _mm256_and_ps(neg2, x2));
        sum3 = _mm256_sub_ps(sum3, _mm256_and_ps(neg3, x3));
    }

    float tail = 0;
    for (; i < n; i++) {
        if (w[i] > 0) tail += x[i];
        else if (w[i] < 0) tail -= x[i];
    }

    return hsum(sum0) + hsum(sum1) + hsum(sum2) + hsum(sum3) + tail;
}

// Standard FP32 dot product (MUL+ADD baseline)
float dot_standard(const float* w, const float* x, int n) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256 wv = _mm256_loadu_ps(&w[i]);
        __m256 xv = _mm256_loadu_ps(&x[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(wv, xv));
    }
    return hsum(sum);
}

int main() {
    srand(42);

    const int actual_dim = 4096;
    const int padded_dim = 4100;
    const int rows = 4096;
    const int nlayers = 32;

    printf("=== Atlas v0.4 - AVX2 SIMD Ternary Kernel ===\n");
    printf("Actual dim: %d, Padded dim: %d\n\n", actual_dim, padded_dim);

    // Pre-unpacked float weights for Atlas SIMD (one per row)
    float* w_std = (float*)_aligned_malloc(rows * padded_dim * sizeof(float), 32);
    float* input = (float*)_aligned_malloc(padded_dim * sizeof(float), 32);

    memset(w_std, 0, rows * padded_dim * sizeof(float));
    memset(input, 0, padded_dim * sizeof(float));
    for (int i = 0; i < actual_dim; i++) input[i] = 1.0f;

    for (int r = 0; r < rows; r++)
        for (int c = 0; c < actual_dim; c++)
            w_std[r * padded_dim + c] = (float)(rand() % 3 - 1);

    float* out_std = (float*)_aligned_malloc(rows * sizeof(float), 32);
    float* out_atl = (float*)_aligned_malloc(rows * sizeof(float), 32);

    clock_t t;

    // -------- Single layer --------
    t = clock();
    for (int r = 0; r < rows; r++)
        out_std[r] = dot_standard(&w_std[r * padded_dim], input, actual_dim);
    double t_std_1 = (double)(clock() - t) / CLOCKS_PER_SEC;

    t = clock();
    for (int r = 0; r < rows; r++)
        out_atl[r] = dot_atlas_simd(&w_std[r * padded_dim], input, actual_dim);
    double t_atl_1 = (double)(clock() - t) / CLOCKS_PER_SEC;

    double delta = 0;
    for (int i = 0; i < rows; i++) delta += fabs(out_std[i] - out_atl[i]);

    printf("--- Single Layer ---\n");
    printf("Delta: %.6f  %s\n", delta, (delta < 0.1) ? "CLEAN" : "ERROR");
    printf("Standard (MUL+ADD): %.4f sec\n", t_std_1);
    printf("Atlas (CMP+AND+ADD/SUB): %.4f sec\n", t_atl_1);
    printf("Speedup:  %.2fx\n\n", t_std_1 / t_atl_1);

    // -------- Multi-layer --------
    printf("--- %d Layers (weight reuse, pre-unpacked) ---\n", nlayers);

    float** std_layers = (float**)_aligned_malloc(nlayers * sizeof(float*), 32);
    for (int l = 0; l < nlayers; l++) {
        std_layers[l] = (float*)_aligned_malloc(rows * padded_dim * sizeof(float), 32);
        memcpy(std_layers[l], w_std, rows * padded_dim * sizeof(float));
    }

    int reps = 5;
    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int r = 0; r < rows; r++)
                out_std[r] = dot_standard(&std_layers[l][r * padded_dim], input, actual_dim);
    double t_std_n = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int r = 0; r < rows; r++)
                out_atl[r] = dot_atlas_simd(&std_layers[l][r * padded_dim], input, actual_dim);
    double t_atl_n = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    printf("Standard FP32:  %.4f sec (%.4f sec/layer)\n", t_std_n, t_std_n/nlayers);
    printf("Atlas SIMD:     %.4f sec (%.4f sec/layer)\n", t_atl_n, t_atl_n/nlayers);
    printf("Speedup:        %.2fx\n\n", t_std_n / t_atl_n);

    // Memory stats
    double std_total = (double)rows * padded_dim * 4 * nlayers / 1e9;
    printf("--- Estimated (%.2fB param model) ---\n",
           (double)actual_dim * actual_dim * nlayers / 1e9);
    printf("Standard:  %.1f t/s  (%.2f GB memory)\n", 1.0/t_std_n, std_total);
    printf("Atlas:     %.1f t/s\n", 1.0/t_atl_n);

    for (int l = 0; l < nlayers; l++)
        _aligned_free(std_layers[l]);
    _aligned_free(std_layers);
    _aligned_free(w_std);
    _aligned_free(input);
    _aligned_free(out_std);
    _aligned_free(out_atl);
    return 0;
}
