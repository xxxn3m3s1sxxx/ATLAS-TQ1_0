// Atlas v0.7 - Random Access Benchmark
// Zeigt: bei random access gewinnt Kompression (Prefetcher hilft nicht)
// clang++ -O3 -mavx2 atlas_v07.cpp -o atlas_v07.exe

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>

alignas(64) static float LUTF5[256][8];
void init() {
    for (int i = 0; i < 243; i++) {
        int v = i;
        for (int j = 0; j < 5; j++) {
            LUTF5[i][j] = (float)((v % 3) - 1);
            v /= 3;
        }
    }
}

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

float dot_tq1(const uint8_t* w, const float* x, int n) {
    float acc = 0;
    for (int i = 0; i < n / 5; i++) {
        float* v = LUTF5[w[i]];
        acc += x[i*5]*v[0] + x[i*5+1]*v[1] + x[i*5+2]*v[2] + x[i*5+3]*v[3] + x[i*5+4]*v[4];
    }
    return acc;
}

int main() {
    init();
    srand(42);

    const int dim = 4096;
    const int rows = 4096;
    const int nlayers = 8;

    // Padded dimensions for each format
    const int dim_pad = (dim + 4) / 5 * 5;  // 4100 for TQ1_0 alignment
    const int row_bytes_tq1 = dim_pad / 5;

    printf("=== Atlas v0.7 - Random Access Benchmark ===\n");
    printf("CPU: i7-7700T, L3: 8 MB\n");
    printf("Dim: %d, Rows: %d, Layers: %d\n\n", dim, rows, nlayers);

    // Weights
    float* w_fp32 = (float*)_aligned_malloc(rows * dim_pad * sizeof(float), 32);
    uint8_t* w_tq1 = (uint8_t*)_aligned_malloc(rows * row_bytes_tq1, 32);
    float* input = (float*)_aligned_malloc(dim_pad * sizeof(float), 32);

    memset(w_fp32, 0, rows * dim_pad * sizeof(float));
    for (int i = 0; i < dim; i++) input[i] = 1.0f;

    // Fill with ternary weights
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < dim; c++) {
            int8_t val = (int8_t)(rand() % 3 - 1);
            w_fp32[r * dim_pad + c] = (float)val;
        }
        // TQ1_0 packing
        for (int c = 0; c < dim_pad; c += 5) {
            int idx = 0, p3 = 1;
            for (int j = 0; j < 5; j++) {
                int8_t val = (c + j < dim) ? (int8_t)w_fp32[r * dim_pad + c + j] : 0;
                idx += (val + 1) * p3;
                p3 *= 3;
            }
            w_tq1[r * row_bytes_tq1 + c/5] = (uint8_t)idx;
        }
    }

    // Pre-generate random row indices
    int* rand_rows = (int*)_aligned_malloc(rows * sizeof(int), 32);
    for (int i = 0; i < rows; i++) rand_rows[i] = rand() % rows;

    float* out = (float*)_aligned_malloc(rows * sizeof(float), 32);
    clock_t t;

    // ---- SEQUENTIAL ACCESS (standard MatMul) ----
    printf("--- Sequential Access (cache-friendly MatMul) ---\n");

    t = clock();
    for (int rep = 0; rep < 5; rep++)
        for (int r = 0; r < rows; r++)
            out[r] = dot_fp32(&w_fp32[r * dim_pad], input, dim);
    double t_fp32_seq = (double)(clock() - t) / CLOCKS_PER_SEC / 5;

    t = clock();
    for (int rep = 0; rep < 5; rep++)
        for (int r = 0; r < rows; r++)
            out[r] = dot_tq1(&w_tq1[r * row_bytes_tq1], input, dim);
    double t_tq1_seq = (double)(clock() - t) / CLOCKS_PER_SEC / 5;

    printf("FP32 (SIMD):  %.4f sec  |  %.2fx\n", t_fp32_seq, 1.0);
    printf("TQ1_0 (pack): %.4f sec  |  %.2fx\n\n", t_tq1_seq, t_fp32_seq/t_tq1_seq);

    // ---- RANDOM ACCESS (Prefetcher kann nicht helfen) ----
    printf("--- Random Access (Prefetcher-defeated) ---\n");

    t = clock();
    for (int rep = 0; rep < 5; rep++)
        for (int i = 0; i < rows; i++) {
            int r = rand_rows[i];
            out[r] = dot_fp32(&w_fp32[r * dim_pad], input, dim);
        }
    double t_fp32_rnd = (double)(clock() - t) / CLOCKS_PER_SEC / 5;

    t = clock();
    for (int rep = 0; rep < 5; rep++)
        for (int i = 0; i < rows; i++) {
            int r = rand_rows[i];
            out[r] = dot_tq1(&w_tq1[r * row_bytes_tq1], input, dim);
        }
    double t_tq1_rnd = (double)(clock() - t) / CLOCKS_PER_SEC / 5;

    printf("FP32 (SIMD):  %.4f sec  |  %.2fx\n", t_fp32_rnd, 1.0);
    printf("TQ1_0 (pack): %.4f sec  |  %.2fx\n\n", t_tq1_rnd, t_fp32_rnd/t_tq1_rnd);

    printf("--- Speedup breakdown ---\n");
    printf("Sequential:  %.2fx (cache-friendly: FP32 ist schnell)\n", t_fp32_seq/t_tq1_seq);
    printf("Random:      %.2fx (cache-killing: Atlas gewinnt!)\n", t_fp32_rnd/t_tq1_rnd);

    // ---- Multi-layer random access ----
    printf("\n--- %d Layers Random Access ---\n", nlayers);

    float** layers_fp32 = (float**)_aligned_malloc(nlayers * sizeof(float*), 32);
    uint8_t** layers_tq1 = (uint8_t**)_aligned_malloc(nlayers * sizeof(uint8_t*), 32);

    for (int l = 0; l < nlayers; l++) {
        layers_fp32[l] = (float*)_aligned_malloc(rows * dim_pad * sizeof(float), 32);
        layers_tq1[l] = (uint8_t*)_aligned_malloc(rows * row_bytes_tq1, 32);
        memcpy(layers_fp32[l], w_fp32, rows * dim_pad * sizeof(float));
        memcpy(layers_tq1[l], w_tq1, rows * row_bytes_tq1);
    }

    t = clock();
    for (int rep = 0; rep < 3; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int i = 0; i < rows; i++) {
                int r = rand_rows[i];
                out[r] = dot_fp32(&layers_fp32[l][r * dim_pad], input, dim);
            }
    double t_fp32_ml = (double)(clock() - t) / CLOCKS_PER_SEC / 3;

    t = clock();
    for (int rep = 0; rep < 3; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int i = 0; i < rows; i++) {
                int r = rand_rows[i];
                out[r] = dot_tq1(&layers_tq1[l][r * row_bytes_tq1], input, dim);
            }
    double t_tq1_ml = (double)(clock() - t) / CLOCKS_PER_SEC / 3;

    printf("FP32 (SIMD):  %.4f sec  |  %.2fx\n", t_fp32_ml, 1.0);
    printf("TQ1_0 (pack): %.4f sec  |  %.2fx\n\n", t_tq1_ml, t_fp32_ml/t_tq1_ml);

    // Cleanup
    for (int l = 0; l < nlayers; l++) {
        _aligned_free(layers_fp32[l]);
        _aligned_free(layers_tq1[l]);
    }
    _aligned_free(layers_fp32);
    _aligned_free(layers_tq1);
    _aligned_free(w_fp32);
    _aligned_free(w_tq1);
    _aligned_free(input);
    _aligned_free(out);
    _aligned_free(rand_rows);
    return 0;
}
