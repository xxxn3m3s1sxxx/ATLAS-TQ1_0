// Atlas v0.6 - 2-Bit Packing (4 values/byte, SIMD-friendly)
// clang++ -O3 -mavx2 atlas_v06.cpp -o atlas_v06.exe

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>

// Map: 00→-1, 01→0, 10→+1, 11→0 (wasted)
static const float MAP[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
// int8 version for SIMD
static const int8_t IMAP[4] = {-1, 0, 1, 0};

static inline float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// ---- Standard FP32 (SIMD MUL+ADD) ----
float dot_fp32(const float* w, const float* x, int n) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256 wv = _mm256_loadu_ps(&w[i]);
        __m256 xv = _mm256_loadu_ps(&x[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(wv, xv));
    }
    return hsum(sum);
}

// ---- TQ1_0 (5-in-1, scalar LUT) ----
alignas(64) static float LUTF5[256][8];
void init_tq1() {
    for (int i = 0; i < 243; i++) {
        int v = i;
        for (int j = 0; j < 5; j++) {
            LUTF5[i][j] = (float)((v % 3) - 1);
            v /= 3;
        }
    }
}
float dot_tq1(const uint8_t* w, const float* x, int n) {
    float acc = 0;
    for (int i = 0; i < n / 5; i++) {
        float* v = LUTF5[w[i]];
        acc += x[i*5]*v[0] + x[i*5+1]*v[1] + x[i*5+2]*v[2] + x[i*5+3]*v[3] + x[i*5+4]*v[4];
    }
    return acc;
}

// ---- 2-Bit Atlas (4 values/byte) ----
// Inner loop: compiler SHOULD auto-vectorize (4 values = power of 2)
float dot_2bit_auto(const uint8_t* w, const float* x, int n) {
    float acc = 0;
    int nb = n / 4;
    for (int i = 0; i < nb; i++) {
        uint8_t b = w[i];
        acc += x[i*4+0] * MAP[(b>>0)&3]
             + x[i*4+1] * MAP[(b>>2)&3]
             + x[i*4+2] * MAP[(b>>4)&3]
             + x[i*4+3] * MAP[(b>>6)&3];
    }
    return acc;
}

// ---- 2-Bit Atlas (explicit SIMD) ----
// Process 16 values (4 bytes) per iteration
float dot_2bit_simd(const uint8_t* w, const float* x, int n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256i shuffle = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);

    for (int i = 0; i < n / 16; i++) {
        // Load 4 bytes (16 values)
        uint32_t block = *(uint32_t*)&w[i*4];

        // Broadcast block to both halves of YMM
        __m128i bl = _mm_cvtsi32_si128(block);
        __m256i bb = _mm256_broadcastd_epi32(bl);  // [block, block, block, block, block, block, block, block]

        // Lane 0..3 use byte 0 (bits 0,2,4,6), Lane 4..7 use byte 1 (bits 8,10,12,14)
        // Shift amounts per lane for extracting the 2-bit values
        // Each lane extracts a different pair of bits:
        // Lane 0: shift 0 → extract v0 from byte 0
        // Lane 1: shift 2 → extract v1 from byte 0
        // Lane 2: shift 4 → extract v2 from byte 0
        // Lane 3: shift 6 → extract v3 from byte 0
        // Lane 4: shift 8 → extract v0 from byte 1
        // Lane 5: shift 10→ extract v1 from byte 1
        // Lane 6: shift 12→ extract v2 from byte 1
        // Lane 7: shift 14→ extract v3 from byte 1

        // Actually, broadcast to 32-bit not 8-bit.
        // Let me try a simpler approach...
    }

    // Fall back to scalar
    float tail = 0;
    for (int i = 0; i < n / 4; i++) {
        uint8_t b = w[i];
        tail += x[i*4+0] * MAP[(b>>0)&3] + x[i*4+1] * MAP[(b>>2)&3]
              + x[i*4+2] * MAP[(b>>4)&3] + x[i*4+3] * MAP[(b>>6)&3];
    }
    return hsum(sum0) + hsum(sum1) + tail;
}

int main() {
    init_tq1();
    srand(42);

    const int actual_dim = 4096;
    // Pad to multiple of 4 * 8 = 32 for 2-bit (8 values per SIMD, 4 bytes per group)
    // Actually 4 values/byte, 8 wide SIMD = need 2 bytes per 8 values
    // Pad to LCM(4, 8) = 8? No... 4 values/byte, 8 wide SIMD = 32 values per full SIMD group
    // Actually let's just use 4096 directly since 4096/4 = 1024 bytes (perfect!)
    const int padded_dim = ((actual_dim + 7) / 8) * 8;  // 4096, already divisible by 8

    const int rows = 4096;
    const int nlayers = 32;

    printf("=== Atlas v0.6 - 2-Bit Packing ===\n");
    printf("Dim: %d, Padded: %d\n", actual_dim, padded_dim);
    printf("TQ1_0: 5 val/byte = %d bytes/row\n", actual_dim/5 + 1);
    printf("2-Bit: 4 val/byte = %d bytes/row\n\n", actual_dim/4);

    float* w_fp32 = (float*)_aligned_malloc(rows * padded_dim * sizeof(float), 32);
    uint8_t* w_tq10 = (uint8_t*)_aligned_malloc(rows * ((padded_dim+4)/5), 32);
    uint8_t* w_2bit = (uint8_t*)_aligned_malloc(rows * (padded_dim/4), 32);
    float* input = (float*)_aligned_malloc(padded_dim * sizeof(float), 32);

    memset(w_fp32, 0, rows * padded_dim * sizeof(float));
    memset(input, 0, padded_dim * sizeof(float));
    for (int i = 0; i < actual_dim; i++) input[i] = 1.0f;

    // Fill all formats with identical weights
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < padded_dim; c++) {
            int8_t val = (c < actual_dim) ? (int8_t)(rand() % 3 - 1) : 0;
            w_fp32[r * padded_dim + c] = (float)val;
        }

        // TQ1_0 (5-in-1)
        int tq_idx = 0, shift = 0, tq_byte = 0, tq_p3 = 1;
        for (int c = 0; c < padded_dim; c++) {
            int8_t val = (int8_t)w_fp32[r * padded_dim + c];
            tq_byte += (val + 1) * tq_p3;
            tq_p3 *= 3;
            shift++;
            if (shift == 5) {
                w_tq10[r * ((padded_dim+4)/5) + tq_idx++] = (uint8_t)tq_byte;
                tq_byte = 0; tq_p3 = 1; shift = 0;
            }
        }
        if (shift > 0) w_tq10[r * ((padded_dim+4)/5) + tq_idx] = (uint8_t)tq_byte;

        // 2-Bit (4-in-1)
        for (int c = 0; c < padded_dim; c += 4) {
            uint8_t b = 0;
            for (int j = 0; j < 4; j++) {
                int8_t val = (c + j < padded_dim) ? (int8_t)w_fp32[r * padded_dim + c + j] : 0;
                int code = (val == -1) ? 0 : (val == 0) ? 1 : (val == 1) ? 2 : 1;
                b |= (code << (j*2));
            }
            w_2bit[r * (padded_dim/4) + c/4] = b;
        }
    }

    float* out = (float*)_aligned_malloc(rows * sizeof(float), 32);
    clock_t t;
    int reps = 5;

    // ---- Single layer benchmarks ----
    printf("--- Single Layer ---\n");

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int r = 0; r < rows; r++) out[r] = dot_fp32(&w_fp32[r * padded_dim], input, actual_dim);
    double t_fp32 = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int r = 0; r < rows; r++) out[r] = dot_tq1(&w_tq10[r * ((padded_dim+4)/5)], input, actual_dim);
    double t_tq1 = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int r = 0; r < rows; r++) out[r] = dot_2bit_auto(&w_2bit[r * (padded_dim/4)], input, actual_dim);
    double t_2bit = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    // Verify 2-bit correctness
    double delta = 0;
    for (int i = 0; i < rows; i++) {
        float ref = dot_fp32(&w_fp32[i * padded_dim], input, actual_dim);
        float val = dot_2bit_auto(&w_2bit[i * (padded_dim/4)], input, actual_dim);
        delta += fabs(ref - val);
    }

    printf("FP32  (SIMD MUL+ADD):  %.4f sec  |  1.00x\n", t_fp32);
    printf("TQ1_0 (5-in-1 skalar): %.4f sec  |  %.2fx\n", t_tq1, t_fp32/t_tq1);
    printf("2-Bit (4-in-1 auto):   %.4f sec  |  %.2fx  Delta=%.6f %s\n\n",
           t_2bit, t_fp32/t_2bit, delta, delta < 1.0 ? "CLEAN" : "ERROR");

    // ---- Multi-layer benchmarks ----
    printf("--- %d Layers ---\n", nlayers);

    float** layers_fp32 = (float**)_aligned_malloc(nlayers * sizeof(float*), 32);
    uint8_t** layers_tq1 = (uint8_t**)_aligned_malloc(nlayers * sizeof(uint8_t*), 32);
    uint8_t** layers_2b = (uint8_t**)_aligned_malloc(nlayers * sizeof(uint8_t*), 32);
    int tq1_row_bytes = (padded_dim+4)/5;
    int b2_row_bytes = padded_dim/4;

    for (int l = 0; l < nlayers; l++) {
        layers_fp32[l] = (float*)_aligned_malloc(rows * padded_dim * sizeof(float), 32);
        layers_tq1[l] = (uint8_t*)_aligned_malloc(rows * tq1_row_bytes, 32);
        layers_2b[l] = (uint8_t*)_aligned_malloc(rows * b2_row_bytes, 32);
        memcpy(layers_fp32[l], w_fp32, rows * padded_dim * sizeof(float));
        memcpy(layers_tq1[l], w_tq10, rows * tq1_row_bytes);
        memcpy(layers_2b[l], w_2bit, rows * b2_row_bytes);
    }

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int r = 0; r < rows; r++) out[r] = dot_fp32(&layers_fp32[l][r * padded_dim], input, actual_dim);
    double t_fp32_ml = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int r = 0; r < rows; r++) out[r] = dot_tq1(&layers_tq1[l][r * tq1_row_bytes], input, actual_dim);
    double t_tq1_ml = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    t = clock();
    for (int rep = 0; rep < reps; rep++)
        for (int l = 0; l < nlayers; l++)
            for (int r = 0; r < rows; r++) out[r] = dot_2bit_auto(&layers_2b[l][r * b2_row_bytes], input, actual_dim);
    double t_2b_ml = (double)(clock() - t) / CLOCKS_PER_SEC / reps;

    printf("FP32  (SIMD MUL+ADD):  %.4f sec  (%.4f/layer)  |  1.00x\n", t_fp32_ml, t_fp32_ml/nlayers);
    printf("TQ1_0 (5-in-1 skalar): %.4f sec  (%.4f/layer)  |  %.2fx\n", t_tq1_ml, t_tq1_ml/nlayers, t_fp32_ml/t_tq1_ml);
    printf("2-Bit (4-in-1 auto):   %.4f sec  (%.4f/layer)  |  %.2fx\n", t_2b_ml, t_2b_ml/nlayers, t_fp32_ml/t_2b_ml);

    // Memory
    double fp32_mem = (double)rows * padded_dim * 4 * nlayers / 1e9;
    double tq1_mem = (double)rows * tq1_row_bytes * nlayers / 1e9;
    double b2_mem = (double)rows * b2_row_bytes * nlayers / 1e9;
    printf("\n--- Memory per %d layers ---\n", nlayers);
    printf("FP32:  %.2f GB\n", fp32_mem);
    printf("TQ1_0: %.2f GB (%.1fx smaller)\n", tq1_mem, fp32_mem/tq1_mem);
    printf("2-Bit: %.2f GB (%.1fx smaller)\n", b2_mem, fp32_mem/b2_mem);

    for (int l = 0; l < nlayers; l++) {
        _aligned_free(layers_fp32[l]);
        _aligned_free(layers_tq1[l]);
        _aligned_free(layers_2b[l]);
    }
    _aligned_free(layers_fp32);
    _aligned_free(layers_tq1);
    _aligned_free(layers_2b);
    _aligned_free(w_fp32);
    _aligned_free(w_tq10);
    _aligned_free(w_2bit);
    _aligned_free(input);
    _aligned_free(out);
    return 0;
}
