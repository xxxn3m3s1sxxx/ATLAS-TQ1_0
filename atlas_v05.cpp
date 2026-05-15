// Atlas v0.5 - JIT Unpack + SIMD Compute
// Phase 1: Packed TQ1_0 -> Float (L1-cached)
// Phase 2: Full AVX2 MUL+ADD (kein alignment-Problem)
// clang++ -O3 -mavx2 atlas_v05.cpp -o atlas_v05.exe

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>

alignas(64) float LUTF[256][8];

void init_atlas() {
    for (int i = 0; i < 243; i++) {
        int v = i;
        for (int j = 0; j < 5; j++) {
            LUTF[i][j] = (float)((v % 3) - 1);
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

// Phase 1: JIT unpack row from TQ1_0 (820 bytes) to float (16 KB)
void unpack_row_f32(const uint8_t* packed, float* unpacked, int n) {
    int nb = n / 5;
    for (int i = 0; i < nb; i++) {
        float* v = LUTF[packed[i]];
        unpacked[i*5+0] = v[0];
        unpacked[i*5+1] = v[1];
        unpacked[i*5+2] = v[2];
        unpacked[i*5+3] = v[3];
        unpacked[i*5+4] = v[4];
    }
}

// Phase 2: SIMD dot product (standard MUL+ADD)
float dot_simd(const float* w, const float* x, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 wv = _mm256_loadu_ps(&w[i]);
        __m256 xv = _mm256_loadu_ps(&x[i]);
        __m256 prod = _mm256_mul_ps(wv, xv);
        sum = _mm256_add_ps(sum, prod);
    }
    float tail = 0;
    for (; i < n; i++) tail += w[i] * x[i];
    return hsum(sum) + tail;
}

// A single temp buffer for all rows (reused per layer)
// Benchmark v0.3 style (scalar, inline unpack) for comparison
float dot_v03(const uint8_t* w, const float* x, int n) {
    float acc = 0;
    for (int i = 0; i < n / 5; i++) {
        float* v = LUTF[w[i]];
        acc += x[i*5]*v[0] + x[i*5+1]*v[1] + x[i*5+2]*v[2] + x[i*5+3]*v[3] + x[i*5+4]*v[4];
    }
    return acc;
}

// Standard FP32 baseline
float dot_std(const float* w, const float* x, int n) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256 wv = _mm256_loadu_ps(&w[i]);
        __m256 xv = _mm256_loadu_ps(&x[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(wv, xv));
    }
    return hsum(sum);
}

int main() {
    init_atlas();
    srand(42);

    const int actual_dim = 4096;
    const int padded_dim = 4100;
    const int rows = 4096;
    const int nlayers = 32;
    const int row_bytes = padded_dim / 5;

    printf("=== Atlas v0.5 - JIT Unpack + SIMD ===\n");
    printf("Dim: %d (padded %d), row_bytes=%d, rows=%d, layers=%d\n\n",
           actual_dim, padded_dim, row_bytes, rows, nlayers);

    float* w_std = (float*)_aligned_malloc(rows * padded_dim * sizeof(float), 32);
    uint8_t* w_packed = (uint8_t*)_aligned_malloc(rows * row_bytes, 32);
    float* input = (float*)_aligned_malloc(padded_dim * sizeof(float), 32);
    float* temp = (float*)_aligned_malloc(padded_dim * sizeof(float), 32);

    memset(w_std, 0, rows * padded_dim * sizeof(float));
    memset(input, 0, padded_dim * sizeof(float));
    for (int i = 0; i < actual_dim; i++) input[i] = 1.0f;

    for (int r = 0; r < rows; r++)
        for (int c = 0; c < padded_dim; c += 5) {
            int idx = 0, p3 = 1;
            for (int j = 0; j < 5; j++) {
                int8_t val = (int8_t)((c + j < actual_dim) ? (rand() % 3 - 1) : 0);
                w_std[r * padded_dim + c + j] = (float)val;
                idx += (val + 1) * p3;
                p3 *= 3;
            }
            w_packed[(r * padded_dim + c) / 5] = (uint8_t)idx;
        }

    float* out1 = (float*)_aligned_malloc(rows * sizeof(float), 32);
    float* out2 = (float*)_aligned_malloc(rows * sizeof(float), 32);
    float* out3 = (float*)_aligned_malloc(rows * sizeof(float), 32);

    int reps = 5;
    clock_t t;

    // ---- BASELINE: Standard FP32 ----
    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int r = 0; r < rows; r++)
            out1[r] = dot_std(&w_std[r * padded_dim], input, actual_dim);
    double t_std = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    // ---- v0.3: Scalar inline unpack ----
    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int r = 0; r < rows; r++)
            out2[r] = dot_v03(&w_packed[r * row_bytes], input, actual_dim);
    double t_v03 = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    // ---- v0.5: JIT unpack + SIMD compute ----
    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int r = 0; r < rows; r++) {
            unpack_row_f32(&w_packed[r * row_bytes], temp, padded_dim);
            out3[r] = dot_simd(temp, input, actual_dim);
        }
    double t_v05 = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    // Verify correctness
    double d1 = 0, d2 = 0;
    for (int i = 0; i < rows; i++) {
        d1 += fabs(out1[i] - out2[i]);
        d2 += fabs(out1[i] - out3[i]);
    }
    printf("--- Results ---\n");
    printf("Standard FP32:      %.4f sec  |  1.00x 1.00x\n", t_std);
    printf("Atlas v0.3 (inline): %.4f sec  |  %.2fx %s\n", t_v03, t_std/t_v03,
           (d1 < 0.1) ? "CLEAN" : "ERROR");
    printf("Atlas v0.5 (JIT):   %.4f sec  |  %.2fx %s\n\n", t_v05, t_std/t_v05,
           (d2 < 0.1) ? "CLEAN" : "ERROR");

    // Multi-layer
    printf("--- %d Layers ---\n", nlayers);

    float** layers_std = (float**)_aligned_malloc(nlayers * sizeof(float*), 32);
    uint8_t** layers_packed = (uint8_t**)_aligned_malloc(nlayers * sizeof(uint8_t*), 32);

    for (int l = 0; l < nlayers; l++) {
        layers_std[l] = (float*)_aligned_malloc(rows * padded_dim * sizeof(float), 32);
        layers_packed[l] = (uint8_t*)_aligned_malloc(rows * row_bytes, 32);
        memcpy(layers_std[l], w_std, rows * padded_dim * sizeof(float));
        memcpy(layers_packed[l], w_packed, rows * row_bytes);
    }

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int r = 0; r < rows; r++)
                out1[r] = dot_std(&layers_std[l][r * padded_dim], input, actual_dim);
    double t_std_ml = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int r = 0; r < rows; r++)
                out2[r] = dot_v03(&layers_packed[l][r * row_bytes], input, actual_dim);
    double t_v03_ml = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int r = 0; r < rows; r++) {
                unpack_row_f32(&layers_packed[l][r * row_bytes], temp, padded_dim);
                out3[r] = dot_simd(temp, input, actual_dim);
            }
    double t_v05_ml = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    printf("Standard FP32:  %.4f sec  (%.4f/layer)\n", t_std_ml, t_std_ml/nlayers);
    printf("Atlas v0.3:     %.4f sec  (%.4f/layer)  %.2fx\n",
           t_v03_ml, t_v03_ml/nlayers, t_std_ml/t_v03_ml);
    printf("Atlas v0.5 JIT: %.4f sec  (%.4f/layer)  %.2fx\n\n",
           t_v05_ml, t_v05_ml/nlayers, t_std_ml/t_v05_ml);

    // Analysis
    double std_mem = (double)rows * padded_dim * 4 * nlayers / 1e9;
    double atl_mem = (double)rows * row_bytes * nlayers / 1e9;
    printf("--- Memory ---\n");
    printf("FP32:  %.2f GB  |  TQ1_0: %.2f GB  (%.1fx smaller)\n", std_mem, atl_mem, std_mem/atl_mem);
    printf("Total per layer:  FP32=%.1f MB  TQ1_0=%.1f MB\n\n",
           (double)rows*padded_dim*4/1e6, (double)rows*row_bytes/1e6);

    double params_b = (double)actual_dim * actual_dim * nlayers / 1e9;
    printf("--- Estimated (%.2fB param model) ---\n", params_b);
    printf("Standard:  %.1f t/s\n", 1.0/t_std_ml);
    printf("Atlas v03: %.1f t/s\n", 1.0/t_v03_ml);
    printf("Atlas v05: %.1f t/s\n", 1.0/t_v05_ml);

    for (int l = 0; l < nlayers; l++) {
        _aligned_free(layers_std[l]);
        _aligned_free(layers_packed[l]);
    }
    _aligned_free(layers_std);
    _aligned_free(layers_packed);
    _aligned_free(w_std);
    _aligned_free(w_packed);
    _aligned_free(input);
    _aligned_free(temp);
    _aligned_free(out1);
    _aligned_free(out2);
    _aligned_free(out3);
    return 0;
}
