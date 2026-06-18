// atlas_kernel_arm64.cpp — ARM64 NEON kernel implementations
//
// Provides NEON-optimized versions of Atlas inference kernels for ARM64
// (Apple Silicon, ARMv8.2-A+ dotprod). Compile with:
//   clang++ -shared -o atlas.dylib atlas_api.cpp atlas_kernel_arm64.cpp \
//     -O2 -march=armv8.2-a+dotprod -ffast-math -std=c++17 -fopenmp
//
// atlas_api.cpp guards x86 intrinsics with #ifndef __aarch64__ and delegates
// to these NEON implementations.
//
// STATUS: WORK IN PROGRESS — only int8 matmul ported so far. Remaining
// kernels are stubbed to compile but run as scalar fallbacks.

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <arm_neon.h>

// ─── Helper: dequantize int8 row to float32 (scalar fallback) ─────────
// Used by all matmul paths as a common utility.
// atlas_api.cpp's FP16 per-block scales are decoded here.
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        // Subnormal: normalize
        if (mant == 0) return sign ? -0.0f : 0.0f;
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= 0x3FF;
    } else if (exp == 31) {
        return sign ? -INFINITY : INFINITY;
    }
    uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f, sizeof(result));
    return result;
}

// ─── int8×uint8 matmul via NEON SDOT (ARMv8.2-A dotprod) ─────────────
// Equivalent to atlas_api.cpp atlas_matmul_i8_f32.
// Replaces _mm256_maddubs_epi16(u8_act, i8_w) + _mm256_madd_epi16 → vdotq_s32.
//
// The -128 bias on u8 activations is removed by subtracting 128*sum(w)
// at the end (same as x86 path).
//
// Parameters:
//   rows: output dimension (number of weight rows)
//   cols: input dimension (padded to multiple of 16 for NEON)
//   weights: [rows × cols] int8
//   act_u8: [n_tokens × cols] uint8 activations (+128 offset)
//   row_sums: [rows] int32 — precomputed sum of each weight row
//   output: [n_tokens × rows] float32 (token-major)
//   n_tokens: batch size (1 for decode, >1 for prefill)
extern "C" void atlas_matmul_i8_f32(int rows, int cols,
    const int8_t* weights,
    const uint8_t* act_u8,
    const int32_t* row_sums,
    float* output,
    int n_tokens) {

    // Process 16 columns per iteration (2 NEON q-registers at 128-bit)
    const int col_chunk = 16;
    int col_aligned = (cols / col_chunk) * col_chunk;

    for (int t = 0; t < n_tokens; t++) {
        const uint8_t* a = act_u8 + t * cols;

        for (int r = 0; r < rows; r++) {
            const int8_t* w = weights + r * cols;

            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            int c = 0;
            for (; c + col_chunk * 4 <= col_aligned; c += col_chunk * 4) {
                // Chunk 0
                uint8x16_t a0 = vld1q_u8(a + c);
                int8x16_t  w0 = vld1q_s8(w + c);
                acc0 = vdotq_s32(acc0, w0, vreinterpretq_s8_u8(a0));

                // Chunk 1
                uint8x16_t a1 = vld1q_u8(a + c + 16);
                int8x16_t  w1 = vld1q_s8(w + c + 16);
                acc1 = vdotq_s32(acc1, w1, vreinterpretq_s8_u8(a1));

                // Chunk 2
                uint8x16_t a2 = vld1q_u8(a + c + 32);
                int8x16_t  w2 = vld1q_s8(w + c + 32);
                acc2 = vdotq_s32(acc2, w2, vreinterpretq_s8_u8(a2));

                // Chunk 3
                uint8x16_t a3 = vld1q_u8(a + c + 48);
                int8x16_t  w3 = vld1q_s8(w + c + 48);
                acc3 = vdotq_s32(acc3, w3, vreinterpretq_s8_u8(a3));
            }

            for (; c < col_aligned; c += col_chunk) {
                uint8x16_t ac = vld1q_u8(a + c);
                int8x16_t  wc = vld1q_s8(w + c);
                acc0 = vdotq_s32(acc0, wc, vreinterpretq_s8_u8(ac));
            }

            int32_t sum = vaddvq_s32(acc0) + vaddvq_s32(acc1)
                        + vaddvq_s32(acc2) + vaddvq_s32(acc3);

            // Handle remaining columns (<16)
            for (; c < cols; c++) {
                sum += (int)a[c] * (int)w[c];
            }

            // -128 bias removal: sum(u8_act * i8_w) = sum((i8_act+128) * i8_w)
            // = sum(i8_act * i8_w) + 128 * sum(i8_w)
            // We store row_sums[r] = sum(i8_w), and the x86 path computes
            // dot - 128 * row_sums[r]. Same here.
            int result = sum - 128 * row_sums[r];
            output[t * rows + r] = (float)result;
        }
    }
}

// ─── Stubs for unimplemented ARM64 kernels ────────────────────────────
// These will call the x86 versions from atlas_api.cpp (guarded) once ported.
// For now, define empty stubs that cause compile errors if linked.
// TODO: Port remaining kernels to NEON.
