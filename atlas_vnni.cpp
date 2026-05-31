#include <stdint.h>
#include <string.h>
#include <immintrin.h>

extern "C" {

// VNNI available detection: returns 1 if real VNNI kernel compiled, 0 for stub
// Guard: clang >= 19 with working target("avx10.2").
//   Linux clang >= 19 works (ubuntu-24.04 CI).
//   Windows needs clang >= 21 (clang 20 ignores avx10.2 in target attribute).
#if defined(__clang__) && __clang_major__ >= 19 && (__clang_major__ >= 21 || !defined(_WIN32))
int atlas_vnni_available(void) { return 1; }

__attribute__((target("avx10.2")))
int atlas_matmul_block_vnni(const signed char* act, const signed char* row, int blk_end, int blk_start) {
    __m512i acc_v = _mm512_setzero_si512();
    int j = blk_start;
    for (; j + 64 <= blk_end; j += 64) {
        __m512i av = _mm512_loadu_si512((const __m512i*)(act + j));
        __m512i wv = _mm512_loadu_si512((const __m512i*)(row + j));
        acc_v = _mm512_dpbssd_epi32(acc_v, av, wv);
    }
    int32_t dot;
    {
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
#else
// Stub — VNNI not compiled (return 0 to signal noop)
int atlas_vnni_available(void) { return 0; }

int atlas_matmul_block_vnni(const signed char* act, const signed char* row, int blk_end, int blk_start) {
    (void)act; (void)row; (void)blk_end; (void)blk_start;
    return 0;
}
#endif

} // extern "C"
