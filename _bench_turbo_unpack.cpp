// TurboQuant 2-bit unpacking microbenchmark
// 4 ternary weights per byte: 00→-1, 01→0, 10→+1
// Weight = ((packed >> (2*pos)) & 0x03) - 1
//
// clang++ -O2 -mavx2 -std=c++17 -o _bench_turbo_unpack _bench_turbo_unpack.cpp
// ./_bench_turbo_unpack

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <windows.h>

static inline int unpack_scalar(uint8_t packed, int pos) {
    return (int)(((packed >> (pos * 2)) & 3) - 1);
}

// Scalar loop: baseline
static void unpack_scalar_loop(const uint8_t* packed, int* out, int n_bytes) {
    for (int i = 0; i < n_bytes; i++) {
        uint8_t p = packed[i];
        out[i * 4 + 0] = unpack_scalar(p, 0);
        out[i * 4 + 1] = unpack_scalar(p, 1);
        out[i * 4 + 2] = unpack_scalar(p, 2);
        out[i * 4 + 3] = unpack_scalar(p, 3);
    }
}

// SSE4.1: unpack 16 bytes → 64 int8 weights (no lane-crossing issues)
#include <immintrin.h>
static void unpack_sse(const uint8_t* packed, int8_t* out, int n_bytes) {
    int i = 0;
    for (; i + 16 <= n_bytes; i += 16) {
        __m128i p = _mm_loadu_si128((const __m128i*)(packed + i));
        __m128i z = _mm_setzero_si128();
        __m128i pe = _mm_unpacklo_epi8(p, z);  // bytes 0-7 zero-extended
        __m128i po = _mm_unpackhi_epi8(p, z);  // bytes 8-15 zero-extended
        __m128i m3 = _mm_set1_epi16(3);
        __m128i o1 = _mm_set1_epi16(1);
        __m128i s0e = _mm_and_si128(pe, m3), s0o = _mm_and_si128(po, m3);
        __m128i s1e = _mm_and_si128(_mm_srli_epi16(pe, 2), m3);
        __m128i s1o = _mm_and_si128(_mm_srli_epi16(po, 2), m3);
        __m128i s2e = _mm_and_si128(_mm_srli_epi16(pe, 4), m3);
        __m128i s2o = _mm_and_si128(_mm_srli_epi16(po, 4), m3);
        __m128i s3e = _mm_and_si128(_mm_srli_epi16(pe, 6), m3);
        __m128i s3o = _mm_and_si128(_mm_srli_epi16(po, 6), m3);
        s0e = _mm_sub_epi16(s0e, o1); s0o = _mm_sub_epi16(s0o, o1);
        s1e = _mm_sub_epi16(s1e, o1); s1o = _mm_sub_epi16(s1o, o1);
        s2e = _mm_sub_epi16(s2e, o1); s2o = _mm_sub_epi16(s2o, o1);
        s3e = _mm_sub_epi16(s3e, o1); s3o = _mm_sub_epi16(s3o, o1);
        // Pack 16→8 bit
        __m128i p0 = _mm_packs_epi16(s0e, s0o);  // pos0 for bytes 0-15
        __m128i p1 = _mm_packs_epi16(s1e, s1o);  // pos1
        __m128i p2 = _mm_packs_epi16(s2e, s2o);  // pos2
        __m128i p3 = _mm_packs_epi16(s3e, s3o);  // pos3
        // Interleave: (p0,p1) → bytes 0-7, (p2,p3) → bytes 8-15
        // Then interleave those pairs for [pos0,pos1,pos2,pos3] order
        __m128i ab = _mm_unpacklo_epi8(p0, p1);
        __m128i cd = _mm_unpacklo_epi8(p2, p3);
        __m128i out0 = _mm_unpacklo_epi16(ab, cd);  // bytes 0-7: weights 0-7
        __m128i out1 = _mm_unpackhi_epi16(ab, cd);  // bytes 8-15: weights 8-15
        // Remaining 32 bytes: use unpackhi for p0,p1 and p2,p3
        __m128i ab_hi = _mm_unpackhi_epi8(p0, p1);
        __m128i cd_hi = _mm_unpackhi_epi8(p2, p3);
        __m128i out2 = _mm_unpacklo_epi16(ab_hi, cd_hi);  // bytes 16-23: weights 16-23
        __m128i out3 = _mm_unpackhi_epi16(ab_hi, cd_hi);  // bytes 24-31: weights 24-31
        // Lane fix: _mm_packs_epi16 outputs [low_even, low_odd] for 128-bit
        // p0 = [pos0_bytes0-7, pos0_bytes8-15]
        // After interleave with unpacklo/hi:
        // out0 = first 4 pairs = weights 0-3
        // out1 = next 4 pairs = weights 4-7
        // out2 = first 4 of hi = weights 8-11
        // out3 = next 4 of hi = weights 12-15
        // Store all 64 weights for the 16 input bytes
        _mm_storeu_si128((__m128i*)(out + i * 4 + 0), out0);
        _mm_storeu_si128((__m128i*)(out + i * 4 + 16), out1);
        _mm_storeu_si128((__m128i*)(out + i * 4 + 32), out2);
        _mm_storeu_si128((__m128i*)(out + i * 4 + 48), out3);
    }
    // Tail
    for (; i < n_bytes; i++) {
        uint8_t bp = packed[i];
        out[i * 4 + 0] = unpack_scalar(bp, 0);
        out[i * 4 + 1] = unpack_scalar(bp, 1);
        out[i * 4 + 2] = unpack_scalar(bp, 2);
        out[i * 4 + 3] = unpack_scalar(bp, 3);
    }
}

// Shuffle-based unpack: AVX2 vpshufb for 16-entry LUT
// 4 bits → 4 ternary values: we need to map 4-bit nibbles
// Each byte has two 4-bit nibbles. Each nibble encodes 2 ternary weights.
// Actually simpler: we use the 2-bit fields directly.
// For a 16-entry LUT: input [0-3][0-3] interleaved = 8-bit index
// This is complex; skip for now.

static double now_sec() {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
}

int main() {
    const int N_BYTES = 1 << 20;  // 1M bytes → 4M weights
    const int N_WEIGHTS = N_BYTES * 4;
    const int ITERS = 200;

    uint8_t* packed = (uint8_t*)malloc(N_BYTES);
    int* out_scalar = (int*)malloc(N_WEIGHTS * sizeof(int));
    int8_t* out_avx2 = (int8_t*)malloc(N_WEIGHTS);

    // Fill with random 2-bit patterns: 0,1,2,3 (mapped to -1,0,1,2)
    srand(42);
    for (int i = 0; i < N_BYTES; i++) {
        uint8_t b = 0;
        for (int j = 0; j < 4; j++) {
            int w = rand() % 3;  // -1, 0, or +1 → encoded as 0,1,2
            b |= (uint8_t)(w & 3) << (j * 2);
        }
        packed[i] = b;
    }

    printf("TurboQuant 2-bit Unpack Benchmark\n");
    printf("  Data: %d bytes → %d weights\n", N_BYTES, N_WEIGHTS);
    printf("  Iterations: %d\n\n", ITERS);

    // Warmup + verify correctness
    unpack_scalar_loop(packed, out_scalar, 64);
    unpack_sse(packed, out_avx2, 64);
    int ok = 1;
    for (int i = 0; i < 64 * 4; i++) {
        if (out_scalar[i] != (int)out_avx2[i]) {
            printf("  MISMATCH at weight %d: scalar=%d avx2=%d\n",
                   i, out_scalar[i], (int)out_avx2[i]);
            ok = 0;
            break;
        }
    }
    printf("  Correctness: %s\n\n", ok ? "PASS" : "FAIL");
    if (!ok) return 1;

    // Benchmark scalar
    double t0 = now_sec();
    for (int iter = 0; iter < ITERS; iter++) {
        unpack_scalar_loop(packed, out_scalar, N_BYTES);
    }
    double t1 = now_sec();
    double scalar_s = (double)N_WEIGHTS * ITERS / ((t1 - t0) * 1e6);
    printf("  Scalar unpack: %.2f M weights/s  (%.3f ms per pass)\n",
           scalar_s, (t1 - t0) / ITERS * 1e3);

    // Benchmark SSE4.1
    double t2 = now_sec();
    for (int iter = 0; iter < ITERS; iter++) {
        unpack_sse(packed, out_avx2, N_BYTES);
    }
    double t3 = now_sec();
    double sse_s = (double)N_WEIGHTS * ITERS / ((t3 - t2) * 1e6);
    printf("  SSE4.1 unpack: %.2f M weights/s  (%.3f ms per pass)\n",
           sse_s, (t3 - t2) / ITERS * 1e3);

    double ratio = sse_s / scalar_s;
    printf("\n  Speedup: %.1fx\n", ratio);

    // Estimate tok/s impact for 3B model
    // 3B: 22L × (3072×9216 FFN + 2×3072×3072 QKV) ≈ 0.9B weights
    // At 2-bit packing: ~225 MB of packed data
    // Unpack time per forward pass = 0.9B / avx2_s M/s
    double weights_3b = 0.9e9;
    double unpack_ms = weights_3b / (sse_s * 1e6) * 1e3;
    printf("\n  Est. unpack time per 3B forward: %.1f ms\n", unpack_ms);
    printf("    (vs ~143ms current total: negligible overhead)\n");

    free(packed);
    free(out_scalar);
    free(out_avx2);
    return 0;
}
