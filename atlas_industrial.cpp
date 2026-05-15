// Atlas Industrial Kernel - 4096-dim Ternary MatMul
// Padded to 4100 (divisible by 5) for clean row alignment
// clang++ -O3 -mavx2 atlas_industrial.cpp -o atlas_industrial.exe

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

static int8_t LUT[243][5];

void init_lut() {
    for (int i = 0; i < 243; i++) {
        int val = i;
        for (int j = 0; j < 5; j++) {
            LUT[i][j] = (int8_t)((val % 3) - 1);
            val /= 3;
        }
    }
}

inline void unpack(uint8_t byte, int8_t* out) {
    int idx = (int)byte;
    if (idx > 242) idx = 242;
    memcpy(out, LUT[idx], 5);
}

static inline float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// Atlas: packed ternary dot product (scalar, zero multiplication)
// No MUL instruction - only ADD/SUB based on weight value
float dot_atlas(const uint8_t* w, const float* x, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        int8_t t[5];
        unpack(w[i/5], t);
        int8_t tw = t[i % 5];
        if (tw == 1) sum += x[i];
        else if (tw == -1) sum -= x[i];
        // tw == 0: skip (no ADD/SUB)
    }
    return sum;
}

// Standard FP32 dot product
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
    init_lut();
    srand(42);

    // Pad DIM to multiple of 5 for clean byte alignment
    const int DIM_RAW = 4096;
    const int DIM = ((DIM_RAW + 4) / 5) * 5;  // 4100
    const int NLAYERS = 32;
    printf("=== Atlas Industrial Kernel ===\n");
    printf("Raw dim: %d, Padded dim: %d (divisible by 5)\n\n", DIM_RAW, DIM);

    int total_weights = DIM * DIM;
    int row_bytes = DIM / 5;
    int packed_size = total_weights / 5;

    float* input = (float*)_aligned_malloc(DIM_RAW * sizeof(float), 32);
    // Pad input to DIM as well
    for (int i = 0; i < DIM_RAW; i++)
        input[i] = (float)(rand() % 100) / 50.0f - 1.0f;

    // FP32 weights: one layer
    float* w_std = (float*)_aligned_malloc(DIM * DIM * sizeof(float), 32);
    uint8_t* w_atlas = (uint8_t*)_aligned_malloc(packed_size, 32);

    memset(w_std, 0, DIM * DIM * sizeof(float));
    for (int row = 0; row < DIM; row++) {
        for (int col = 0; col < DIM; col += 5) {
            int v[5];
            int vmax = (col + 5 <= DIM_RAW) ? 5 : (DIM_RAW - col);
            if (vmax > 0) {
                for (int j = 0; j < vmax; j++)
                    v[j] = rand() % 3 - 1;
                for (int j = vmax; j < 5; j++)
                    v[j] = 0;
                int byte_idx = (row * DIM + col) / 5;
                int idx = (v[0]+1) + (v[1]+1)*3 + (v[2]+1)*9 + (v[3]+1)*27 + (v[4]+1)*81;
                w_atlas[byte_idx] = (uint8_t)idx;
                for (int j = 0; j < 5; j++)
                    w_std[row * DIM + col + j] = (float)v[j];
            }
        }
    }

    // Verify correctness
    double check_std = 0, check_atlas = 0;
    for (int row = 0; row < DIM_RAW; row++) {
        check_std += dot_standard(&w_std[row * DIM], input, DIM_RAW);
        check_atlas += dot_atlas(&w_atlas[row * row_bytes], input, DIM_RAW);
    }
    double delta = check_std - check_atlas;
    printf("Correctness: std=%.2f atlas=%.2f delta=%.2e  %s\n\n",
           check_std, check_atlas, delta,
           (delta * delta < 1.0) ? "OK" : "MISMATCH!");

    // ---- Benchmark 1: Single layer (cold cache) ----
    printf("--- Benchmark 1: Single layer ---\n");
    volatile float sink = 0;
    int REPS = 20;

    clock_t t0 = clock();
    for (int rep = 0; rep < REPS; rep++)
        for (int row = 0; row < DIM_RAW; row++)
            sink += dot_standard(&w_std[row * DIM], input, DIM_RAW);
    clock_t t1 = clock();
    double std_time = (double)(t1 - t0) / CLOCKS_PER_SEC / REPS;

    t0 = clock();
    for (int rep = 0; rep < REPS; rep++)
        for (int row = 0; row < DIM_RAW; row++)
            sink += dot_atlas(&w_atlas[row * row_bytes], input, DIM_RAW);
    t1 = clock();
    double atlas_time = (double)(t1 - t0) / CLOCKS_PER_SEC / REPS;

    (void)sink;
    printf("  Standard FP32:  %.4f sec\n", std_time);
    printf("  Atlas TQ1_0:    %.4f sec\n", atlas_time);
    printf("  Speedup:        %.2fx\n\n", std_time / atlas_time);

    // ---- Benchmark 2: N layers (simulate real inference) ----
    printf("--- Benchmark 2: %d layers ---\n", NLAYERS);

    float** std_layers = (float**)_aligned_malloc(NLAYERS * sizeof(float*), 32);
    uint8_t** atlas_layers = (uint8_t**)_aligned_malloc(NLAYERS * sizeof(uint8_t*), 32);

    for (int l = 0; l < NLAYERS; l++) {
        std_layers[l] = (float*)_aligned_malloc(DIM * DIM * sizeof(float), 32);
        atlas_layers[l] = (uint8_t*)_aligned_malloc(packed_size, 32);
        memcpy(std_layers[l], w_std, DIM * DIM * sizeof(float));
        memcpy(atlas_layers[l], w_atlas, packed_size);
    }

    int ML_REPS = 5;
    t0 = clock();
    double sum_std = 0;
    for (int rep = 0; rep < ML_REPS; rep++)
        for (int l = 0; l < NLAYERS; l++)
            for (int row = 0; row < DIM_RAW; row++)
                sum_std += dot_standard(std_layers[l] + row * DIM, input, DIM_RAW);
    t1 = clock();
    double std_ml_time = (double)(t1 - t0) / CLOCKS_PER_SEC / ML_REPS;

    t0 = clock();
    double sum_atlas = 0;
    for (int rep = 0; rep < ML_REPS; rep++)
        for (int l = 0; l < NLAYERS; l++)
            for (int row = 0; row < DIM_RAW; row++)
                sum_atlas += dot_atlas(atlas_layers[l] + row * row_bytes, input, DIM_RAW);
    t1 = clock();
    double atlas_ml_time = (double)(t1 - t0) / CLOCKS_PER_SEC / ML_REPS;

    printf("  Standard FP32:  %.4f sec (%.4f sec/layer)\n", std_ml_time, std_ml_time/NLAYERS);
    printf("  Atlas TQ1_0:    %.4f sec (%.4f sec/layer)\n", atlas_ml_time, atlas_ml_time/NLAYERS);
    printf("  Speedup:        %.2fx\n\n", std_ml_time / atlas_ml_time);

    printf("--- Memory ---\n");
    double std_total = (double)DIM * DIM * 4 * NLAYERS / 1e9;
    double atlas_total = (double)packed_size * NLAYERS / 1e9;
    printf("  FP32:  %.2f GB (%d layers)\n", std_total, NLAYERS);
    printf("  TQ1_0: %.2f GB (%d layers, %.1fx smaller)\n",
           atlas_total, NLAYERS, DIM*DIM*4.0/packed_size);

    double params_b = (double)DIM_RAW * DIM_RAW * NLAYERS / 1e9;
    printf("\n--- Estimated Throughput (%.2fB param model) ---\n", params_b);
    printf("  Standard: %.1f t/s  (%.2f GB memory traffic)\n", 1.0/std_ml_time, std_total);
    printf("  Atlas:    %.1f t/s  (%.2f GB memory traffic)\n", 1.0/atlas_ml_time, atlas_total);

    (void)sum_std; (void)sum_atlas;
    for (int l = 0; l < NLAYERS; l++) {
        _aligned_free(std_layers[l]);
        _aligned_free(atlas_layers[l]);
    }
    _aligned_free(std_layers);
    _aligned_free(atlas_layers);
    _aligned_free(w_std);
    _aligned_free(w_atlas);
    _aligned_free(input);
    return 0;
}
