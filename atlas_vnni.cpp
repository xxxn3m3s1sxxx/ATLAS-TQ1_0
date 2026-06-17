// atlas_vnni.cpp — AVX-512 VNNI / AVX10.2 kernels for int4 matmul
// Compiles on clang >= 19 (Linux) or clang >= 21 (Windows) only.
// Older compilers get stubs (atlas_vnni_available() == 0 → AVX2 fallback).

#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

extern "C" {

// VNNI available: clang >= 19, must NOT be Windows (target("avx10.2") crashes
// clang-22 backend on Windows as of 2026-05). Linux clang >= 19 works fine.
#if defined(__clang__) && __clang_major__ >= 19 && !defined(_WIN32)
int atlas_vnni_available(void) { return 1; }

// ─── Ternary block dot product via VNNI ──────────────────────────────────
// Used by the ternary matmul path (ttype=1) when AVX-512 VNNI is available.
// Both operands are signed int8. Uses _mm512_dpbssd_epi32.
__attribute__((target("avx10.2")))
int atlas_matmul_block_vnni(const int8_t* act, const int8_t* row, int blk_end, int blk_start) {
    __m512i acc_v = _mm512_setzero_si512();
    int j = blk_start;
    for (; j + 64 <= blk_end; j += 64) {
        __m512i av = _mm512_loadu_si512((const __m512i*)(act + j));
        __m512i wv = _mm512_loadu_si512((const __m512i*)(row + j));
        acc_v = _mm512_dpbssd_epi32(acc_v, av, wv);
    }
    int32_t dot;
    {   // Horizontal sum of 16 int32 values
        __m256i lo = _mm512_castsi512_si256(acc_v);
        __m256i hi = _mm512_extracti64x4_epi64(acc_v, 1);
        __m256i sum = _mm256_add_epi32(lo, hi);
        __m128i l = _mm256_castsi256_si128(sum);
        __m128i h = _mm256_extracti128_si256(sum, 1);
        l = _mm_add_epi32(l, h);
        l = _mm_hadd_epi32(l, l);
        l = _mm_hadd_epi32(l, l);
        dot = _mm_cvtsi128_si32(l);
    }
    for (; j < blk_end; j++) {
        dot += (int32_t)act[j] * (int32_t)row[j];
    }
    return dot;
}

// ─── i4 matmul via VNNI ─────────────────────────────────────────────────
// Processes 64 elements per iteration: nibble-unpack (256-bit) + VNNI (512-bit).
// Activations are uint8, weights are int4 packed 2/byte, sign-extended to int8.
// Uses _mm512_dpbusd_epi32 (uint8 × int8 → int32 dot product, VNNI).
__attribute__((target("avx10.2")))
void atlas_matmul_i4_vnni(int rows, int cols,
                          const uint8_t* __restrict__ packed_weights,
                          const uint8_t* __restrict__ act_u8,
                          const int32_t* __restrict__ row_sums,
                          float* __restrict__ output,
                          int n_tokens) {
    const __m256i c8 = _mm256_set1_epi8(8);
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
    #endif
    for (int r = 0; r < rows; r++) {
        const uint8_t* pw = packed_weights + (int64_t)r * cols / 2;
        int sum_w = row_sums[r];

        for (int t = 0; t < n_tokens; t++) {
            const uint8_t* a = act_u8 + (int64_t)t * cols;
            int c = 0;
            int dot = 0;
            __m512i acc = _mm512_setzero_si512();

            // Process 64 elements per iteration
            for (; c + 64 <= cols; c += 64) {
                int pc = c / 2; // 64 nibbles = 32 bytes
                // Nibble-unpack: 32 packed bytes → 2× sign-extended int8 vectors (256-bit)
                __m256i packed = _mm256_loadu_si256((const __m256i*)(pw + pc));
                __m256i low  = _mm256_and_si256(packed, mask_0f);
                __m256i high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_0f);
                // Sign-extend 4-bit to int8: (nibble ^ 8) - 8
                __m256i w_low_s  = _mm256_sub_epi8(_mm256_xor_si256(low,  c8), c8);
                __m256i w_high_s = _mm256_sub_epi8(_mm256_xor_si256(high, c8), c8);
                // Interleave low/high → contiguous [w0..w63]
                __m256i wt_lo = _mm256_unpacklo_epi8(w_low_s, w_high_s);
                __m256i wt_hi = _mm256_unpackhi_epi8(w_low_s, w_high_s);
                __m256i w_lo  = _mm256_permute2f128_si256(wt_lo, wt_hi, 0x20);
                __m256i w_hi  = _mm256_permute2f128_si256(wt_lo, wt_hi, 0x31);
                // Combine → 512-bit weights vector (signed int8)
                __m512i w512 = _mm512_inserti64x4(_mm512_castsi256_si512(w_lo), w_hi, 1);
                // Load 64 activation bytes (uint8)
                __m512i a512 = _mm512_loadu_si512((const __m512i*)(a + c));
                // VNNI: uint8(act) × int8(w) → int32 accumulate
                acc = _mm512_dpbusd_epi32(acc, a512, w512);
            }

            // Horizontal sum of 16 int32 values
            __m256i lo = _mm512_castsi512_si256(acc);
            __m256i hi = _mm512_extracti64x4_epi64(acc, 1);
            __m256i s256 = _mm256_add_epi32(lo, hi);
            __m128i l128 = _mm256_castsi256_si128(s256);
            __m128i h128 = _mm256_extracti128_si256(s256, 1);
            __m128i s128 = _mm_add_epi32(l128, h128);
            s128 = _mm_hadd_epi32(s128, s128);
            s128 = _mm_hadd_epi32(s128, s128);
            dot = _mm_cvtsi128_si32(s128);

            // Tail (< 64 elements): scalar nibble-unpack
            for (; c < cols; c++) {
                int packed_idx = ((int64_t)r * cols + c) / 2;
                int nibble = (c & 1)
                    ? (packed_weights[packed_idx] >> 4)
                    : (packed_weights[packed_idx] & 0x0F);
                int8_t w_val = (int8_t)((nibble ^ 8) - 8);
                dot += (int)a[c] * (int)w_val;
            }

            int result = dot - 128 * sum_w;
            output[(int64_t)t * rows + r] = (float)result;
        }
    }
}

#else
// Stub — no VNNI compile target available
int atlas_vnni_available(void) { return 0; }

int atlas_matmul_block_vnni(const int8_t* act, const int8_t* row, int blk_end, int blk_start) {
    (void)act; (void)row; (void)blk_end; (void)blk_start;
    return 0;
}

void atlas_matmul_i4_vnni(int rows, int cols,
                          const uint8_t* packed_weights,
                          const uint8_t* act_u8,
                          const int32_t* row_sums,
                          float* output,
                          int n_tokens) {
    (void)rows; (void)cols; (void)packed_weights;
    (void)act_u8; (void)row_sums; (void)output; (void)n_tokens;
}
#endif

} // extern "C"
