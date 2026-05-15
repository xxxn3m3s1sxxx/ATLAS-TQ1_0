// Atlas v0.3 - Industrial Kernel (Padding Edition)
// 4096-dim, padded to 4100 for clean byte alignment
// clang++ -O3 -mavx2 atlas_v03.cpp -o atlas_v03.exe

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

alignas(32) int8_t LUT[256][8];

void init_atlas() {
    for (int i = 0; i < 243; i++) {
        int v = i;
        for (int j = 0; j < 5; j++) {
            LUT[i][j] = (int8_t)((v % 3) - 1);
            v /= 3;
        }
    }
}

int main() {
    init_atlas();
    srand(42);

    const int actual_dim = 4096;
    const int padded_dim = 4100;
    const int rows = 4096;
    const int nlayers = 32;

    printf("=== Atlas v0.3 - Industrial Ternary Kernel ===\n");
    printf("Actual dim: %d, Padded dim: %d (820 bytes/row)\n\n", actual_dim, padded_dim);

    float* w_std = (float*)_aligned_malloc(rows * padded_dim * sizeof(float), 32);
    uint8_t* w_atlas = (uint8_t*)_aligned_malloc(rows * (padded_dim / 5), 32);
    float* input = (float*)_aligned_malloc(padded_dim * sizeof(float), 32);
    float* out_std = (float*)_aligned_malloc(rows * sizeof(float), 32);
    float* out_atl = (float*)_aligned_malloc(rows * sizeof(float), 32);

    memset(w_std, 0, rows * padded_dim * sizeof(float));
    memset(input, 0, padded_dim * sizeof(float));

    for (int i = 0; i < actual_dim; i++) input[i] = 1.0f;

    // Fill weights with padding
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < padded_dim; c += 5) {
            int idx = 0, p3 = 1;
            for (int j = 0; j < 5; j++) {
                int8_t val = (int8_t)((c + j < actual_dim) ? (rand() % 3 - 1) : 0);
                w_std[r * padded_dim + c + j] = (float)val;
                idx += (val + 1) * p3;
                p3 *= 3;
            }
            w_atlas[(r * padded_dim + c) / 5] = (uint8_t)idx;
        }
    }

    // -------- BENCHMARK 1: Single layer correctness --------
    clock_t t;

    t = clock();
    for (int r = 0; r < rows; r++) {
        float acc = 0;
        float* row = &w_std[r * padded_dim];
        for (int c = 0; c < actual_dim; c++)
            acc += input[c] * row[c];
        out_std[r] = acc;
    }
    double t_std_1 = (double)(clock() - t) / CLOCKS_PER_SEC;

    t = clock();
    for (int r = 0; r < rows; r++) {
        float acc = 0;
        uint8_t* row_atl = &w_atlas[r * (padded_dim / 5)];
        for (int c = 0; c < padded_dim / 5; c++) {
            int8_t* v = LUT[row_atl[c]];
            acc += input[c*5+0]*v[0] + input[c*5+1]*v[1] + input[c*5+2]*v[2] + input[c*5+3]*v[3] + input[c*5+4]*v[4];
        }
        out_atl[r] = acc;
    }
    double t_atl_1 = (double)(clock() - t) / CLOCKS_PER_SEC;

    double delta = 0;
    for (int i = 0; i < rows; i++) delta += fabs(out_std[i] - out_atl[i]);
    printf("--- Single Layer ---\n");
    printf("Delta: %.6f  %s\n", delta, (delta < 0.1) ? "CLEAN" : "ERROR");
    printf("Standard FP32:  %.4f sec\n", t_std_1);
    printf("Atlas TQ1_0:    %.4f sec\n", t_atl_1);
    printf("Speedup:        %.2fx\n\n", t_std_1 / t_atl_1);

    // -------- BENCHMARK 2: Multi-layer (cache thrash) --------
    printf("--- %d Layers (cache pressure) ---\n", nlayers);

    float** std_layers = (float**)_aligned_malloc(nlayers * sizeof(float*), 32);
    uint8_t** atl_layers = (uint8_t**)_aligned_malloc(nlayers * sizeof(uint8_t*), 32);
    int row_bytes = padded_dim / 5;

    for (int l = 0; l < nlayers; l++) {
        std_layers[l] = (float*)_aligned_malloc(rows * padded_dim * sizeof(float), 32);
        atl_layers[l] = (uint8_t*)_aligned_malloc(rows * row_bytes, 32);
        memcpy(std_layers[l], w_std, rows * padded_dim * sizeof(float));
        memcpy(atl_layers[l], w_atlas, rows * row_bytes);
    }

    int reps = 5;

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int r = 0; r < rows; r++) {
                float acc = 0;
                float* row = &std_layers[l][r * padded_dim];
                for (int c = 0; c < actual_dim; c++)
                    acc += input[c] * row[c];
                out_std[r] = acc;
            }
    double t_std_n = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int r = 0; r < rows; r++) {
                float acc = 0;
                uint8_t* row_atl = &atl_layers[l][r * row_bytes];
                for (int c = 0; c < row_bytes; c++) {
                    int8_t* v = LUT[row_atl[c]];
                    acc += input[c*5+0]*v[0] + input[c*5+1]*v[1] + input[c*5+2]*v[2] + input[c*5+3]*v[3] + input[c*5+4]*v[4];
                }
                out_atl[r] = acc;
            }
    double t_atl_n = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    printf("Standard FP32:  %.4f sec (%.4f sec/layer)\n", t_std_n, t_std_n/nlayers);
    printf("Atlas TQ1_0:    %.4f sec (%.4f sec/layer)\n", t_atl_n, t_atl_n/nlayers);
    printf("Speedup:        %.2fx\n\n", t_std_n / t_atl_n);

    // -------- Memory stats --------
    double std_bytes = (double)rows * padded_dim * 4 * nlayers;
    double atl_bytes = (double)rows * row_bytes * nlayers;
    printf("--- Memory ---\n");
    printf("FP32:   %.2f GB (%d layers)\n", std_bytes / 1e9, nlayers);
    printf("TQ1_0:  %.2f GB (%d layers, %.1fx smaller)\n",
           atl_bytes / 1e9, nlayers, std_bytes / atl_bytes);

    double params_b = (double)actual_dim * actual_dim * nlayers / 1e9;
    printf("\n--- Estimated (%.2fB param model) ---\n", params_b);
    printf("Standard:  %.1f t/s\n", 1.0 / t_std_n);
    printf("Atlas:     %.1f t/s\n", 1.0 / t_atl_n);

    // Cleanup
    for (int l = 0; l < nlayers; l++) {
        _aligned_free(std_layers[l]);
        _aligned_free(atl_layers[l]);
    }
    _aligned_free(std_layers);
    _aligned_free(atl_layers);
    _aligned_free(w_std);
    _aligned_free(w_atlas);
    _aligned_free(input);
    _aligned_free(out_std);
    _aligned_free(out_atl);
    return 0;
}
