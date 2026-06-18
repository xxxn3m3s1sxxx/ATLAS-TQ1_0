// atlas_kernel_arm64.cpp — ARM64 NEON kernel implementations
//
// Provides NEON-optimized versions of Atlas inference kernels for ARM64
// (Apple Silicon, ARMv8.2-A+ dotprod). Compile with:
//   clang++ -shared -o atlas.dylib atlas_api.cpp atlas_kernel_arm64.cpp \
//     -O2 -march=armv8.2-a+dotprod -ffast-math -std=c++17 -fopenmp
//
// atlas_api.cpp guards x86 intrinsics with #ifndef __aarch64__ and calls
// these extern "C" functions when __aarch64__ is defined.
//
// NEON vs x86 key difference:
//   _mm256_maddubs_epi16 treats first operand as unsigned, second as signed.
//   vdotq_s32 treats BOTH as signed.
//   Symmetric int8 activations (range [-127,127]): vdotq_s32 gives correct
//     result directly. No XOR trick needed.
//   Unsigned uint8 activations (range [0,255]): XOR with 0x80 first to
//     convert [0,255] to [-128,127] = value - 128 in signed arithmetic.
//     Then vdotq_s32 gives the centered (signed) dot product directly.
//     No -128 * row_sums or -128 * blk_sums correction needed.

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <arm_neon.h>

// ─── Helper: fp16 → fp32 ──────────────────────────────────────────────
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
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

// ─── TQ1 decode LUT (5 ternary trits per byte) ────────────────────────
static int8_t tq1_decode[256][5];
static bool tq1_lut_initialized = false;

static void init_tq1_decode_lut() {
    if (tq1_lut_initialized) return;
    for (int b = 0; b < 256; b++) {
        int t = b;
        tq1_decode[b][0] = (int8_t)((t % 3) - 1); t /= 3;
        tq1_decode[b][1] = (int8_t)((t % 3) - 1); t /= 3;
        tq1_decode[b][2] = (int8_t)((t % 3) - 1); t /= 3;
        tq1_decode[b][3] = (int8_t)((t % 3) - 1); t /= 3;
        tq1_decode[b][4] = (int8_t)((t % 3) - 1);
    }
    tq1_lut_initialized = true;
}

// ═══════════════════════════════════════════════════════════════════════
// int8×uint8 matmul (row_sums correction eliminated via XOR-0x80 trick)
// ═══════════════════════════════════════════════════════════════════════
// Called from atlas_api.c before f32_bypass/quantized dispatch.
// act_u8: uint8 [0,255] with +128 offset; weights: int8.
// XOR act_u8 with 0x80 → converts to signed = val - 128.
// vdotq_s32(w, signed_act) = sum(w * (val-128)) = centered dot product.
// So NO row_sums correction needed (x86 path's -128*row_sums is built in).
extern "C" void atlas_matmul_i8_f32(int rows, int cols,
    const int8_t* weights,
    const uint8_t* act_u8,
    const int32_t* row_sums,
    float* output,
    int n_tokens) {

    (void)row_sums;
    const int col_chunk = 16;
    int col_aligned = (cols / col_chunk) * col_chunk;
    int8x16_t xor_mask = vdupq_n_s8(-128);

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
                uint8x16_t a0 = vld1q_u8(a + c);
                int8x16_t  w0 = vld1q_s8(w + c);
                acc0 = vdotq_s32(acc0, w0,
                    veorq_s8(vreinterpretq_s8_u8(a0), xor_mask));

                uint8x16_t a1 = vld1q_u8(a + c + 16);
                int8x16_t  w1 = vld1q_s8(w + c + 16);
                acc1 = vdotq_s32(acc1, w1,
                    veorq_s8(vreinterpretq_s8_u8(a1), xor_mask));

                uint8x16_t a2 = vld1q_u8(a + c + 32);
                int8x16_t  w2 = vld1q_s8(w + c + 32);
                acc2 = vdotq_s32(acc2, w2,
                    veorq_s8(vreinterpretq_s8_u8(a2), xor_mask));

                uint8x16_t a3 = vld1q_u8(a + c + 48);
                int8x16_t  w3 = vld1q_s8(w + c + 48);
                acc3 = vdotq_s32(acc3, w3,
                    veorq_s8(vreinterpretq_s8_u8(a3), xor_mask));
            }

            for (; c < col_aligned; c += col_chunk) {
                uint8x16_t ac = vld1q_u8(a + c);
                int8x16_t  wc = vld1q_s8(w + c);
                acc0 = vdotq_s32(acc0, wc,
                    veorq_s8(vreinterpretq_s8_u8(ac), xor_mask));
            }

            int32_t sum = vaddvq_s32(acc0) + vaddvq_s32(acc1)
                        + vaddvq_s32(acc2) + vaddvq_s32(acc3);

            for (; c < cols; c++) {
                sum += (int)(a[c] - 128) * (int)w[c];
            }

            output[t * rows + r] = (float)sum;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TQ1 block-scaled fused matmul (symmetric int8 activations)
// ═══════════════════════════════════════════════════════════════════════
// ARM64 equivalent of matmul_tq1_block_fused_s8 in atlas_api.cpp.
//
// act_s8: symmetric int8 [-127,127] (no +128 offset). Weights are {-1,0,+1}
// from TQ1 decode. vdotq_s32(w, act) gives sum(w * act) directly —
// NO XOR-0x80 trick needed (both operands already signed).
//
// Replaces _mm256_sign_epi8(act, w) → horizontal sum with vdotq_s32.
// vdotq_s32 computes sum(act[0..3] * w[0..3]) in one instruction vs
// x86's sign_epi8 + cvtepi8_epi16 + madd_epi16 + hadd_epi32 pipeline.
extern "C" void atlas_tq1_fused_s8_arm64(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const float* act_f32, float* output, int B) {

    init_tq1_decode_lut();
    int rows_packed = rows / 4;

    const uint8_t* scale_data = tensor_data + 3;
    const uint8_t* packed = tensor_data + 3 + rows * n_blocks * 2;

    // Pre-decode all per-row per-block fp16 scales to float (shared buffer)
    static float* s_block_scales = nullptr;
    static size_t s_block_scales_cap = 0;
    {
        size_t need_bs = (size_t)rows * n_blocks;
        if (need_bs > s_block_scales_cap) {
            free(s_block_scales);
            s_block_scales = (float*)malloc(need_bs * sizeof(float));
            s_block_scales_cap = need_bs;
        }
    }
    float* block_scales = s_block_scales;

    // Thread-local activation quantization buffer
    static thread_local int8_t* tl_act_s8 = nullptr;
    static thread_local size_t tl_act_s8_cap = 0;
    static thread_local float* tl_scale_x = nullptr;
    static thread_local size_t tl_scale_x_cap = 0;

    size_t need_as = (size_t)B * input_dim;
    if (need_as > tl_act_s8_cap) {
        free(tl_act_s8);
        tl_act_s8 = (int8_t*)malloc(need_as * sizeof(int8_t));
        tl_act_s8_cap = need_as;
    }
    int8_t* act_s8 = tl_act_s8;
    if ((size_t)B > tl_scale_x_cap) {
        free(tl_scale_x);
        tl_scale_x = (float*)malloc((size_t)B * sizeof(float));
        tl_scale_x_cap = B;
    }
    float* scale_x = tl_scale_x;

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        // Parallel fp16→fp32 scale decode
        #ifdef _OPENMP
        #pragma omp for
        #endif
        for (int i = 0; i < rows * n_blocks; i++) {
            uint16_t sr;
            memcpy(&sr, scale_data + i * 2, 2);
            block_scales[i] = fp16_to_fp32(sr);
        }

        // Thread-local decode buffer
        static thread_local int8_t* tl_decode_buf = nullptr;
        static thread_local size_t tl_decode_buf_cap = 0;
        size_t need_db = (size_t)4 * input_dim;
        if (need_db > tl_decode_buf_cap) {
            free(tl_decode_buf);
            tl_decode_buf = (int8_t*)malloc(need_db * sizeof(int8_t));
            tl_decode_buf_cap = need_db;
            memset(tl_decode_buf, 0, need_db * sizeof(int8_t));
        }
        int8_t* decode_buf = tl_decode_buf;

        // Act quant single-threaded (sequential across b)
        #ifdef _OPENMP
        #pragma omp single
        #endif
        {
            for (int b = 0; b < B; b++) {
                const float* act = act_f32 + b * input_dim;
                float max_val = 1e-5f;
                for (int i = 0; i < input_dim; i++) {
                    float av = fabsf(act[i]);
                    if (av > max_val) max_val = av;
                }
                scale_x[b] = max_val / 127.0f;
                float inv = 127.0f / max_val;
                int8_t* aq = act_s8 + b * input_dim;
                for (int i = 0; i < input_dim; i++) {
                    int q = (int)(act[i] * inv + 0.5f);
                    if (q < -127) q = -127;
                    if (q > 127) q = 127;
                    aq[i] = (int8_t)q;
                }
            }
        }

        #ifdef _OPENMP
        #pragma omp for
        #endif
        for (int ur = 0; ur < rows_packed; ur++) {
            // Decode TQ1 weights for 4 rows
            for (int sub = 0; sub < 4; sub++) {
                const uint8_t* w = packed + (ur * 4 + sub) * packed_cols;
                int8_t* row = decode_buf + sub * input_dim;
                int c = 0;
                for (; c < packed_cols - 1; c++) {
                    const int8_t* l = tq1_decode[w[c]];
                    int col = c * 5;
                    uint32_t v4 = (uint8_t)l[0] | ((uint32_t)(uint8_t)l[1] << 8) |
                                  ((uint32_t)(uint8_t)l[2] << 16) | ((uint32_t)(uint8_t)l[3] << 24);
                    memcpy(row + col, &v4, 4);
                    row[col + 4] = l[4];
                }
                {
                    const int8_t* l = tq1_decode[w[c]];
                    int col = c * 5;
                    if (col < input_dim) row[col] = l[0];
                    if (col + 1 < input_dim) row[col + 1] = l[1];
                    if (col + 2 < input_dim) row[col + 2] = l[2];
                    if (col + 3 < input_dim) row[col + 3] = l[3];
                    if (col + 4 < input_dim) row[col + 4] = l[4];
                }
            }

            for (int b = 0; b < B; b++) {
                const int8_t* act = act_s8 + b * input_dim;
                float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                for (int sub = 0; sub < 4; sub++) {
                    int shuffled_r = ur * 4 + sub;
                    const float* rscales = block_scales + shuffled_r * n_blocks;
                    const int8_t* row = decode_buf + sub * input_dim;

                    for (int blk = 0; blk < n_blocks; blk++) {
                        int blk_start = blk * block_size;
                        int blk_end = blk_start + block_size;
                        if (blk_end > input_dim) blk_end = input_dim;

                        int32x4_t acc0 = vdupq_n_s32(0);
                        int32x4_t acc1 = vdupq_n_s32(0);

                        int j = blk_start;
                        for (; j + 32 <= blk_end; j += 32) {
                            int8x16_t w0 = vld1q_s8(row + j);
                            int8x16_t w1 = vld1q_s8(row + j + 16);
                            int8x16_t a0 = vld1q_s8(act + j);
                            int8x16_t a1 = vld1q_s8(act + j + 16);

                            // Both operands signed: vdotq_s32 gives sum(w*act) directly.
                            // No XOR trick needed (activations are symmetric [-127,127]).
                            acc0 = vdotq_s32(acc0, w0, a0);
                            acc1 = vdotq_s32(acc1, w1, a1);
                        }

                        int32_t dot = vaddvq_s32(acc0) + vaddvq_s32(acc1);

                        for (; j < blk_end; j++) {
                            dot += (int32_t)act[j] * (int32_t)row[j];
                        }

                        out4[sub] += (float)dot * rscales[blk];
                    }
                }

                float deq = scale_x[b];
                float* dst = output + b * rows;
                dst[0 * rows_packed + ur] = out4[0] * deq;
                dst[1 * rows_packed + ur] = out4[1] * deq;
                dst[2 * rows_packed + ur] = out4[2] * deq;
                dst[3 * rows_packed + ur] = out4[3] * deq;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TQ1 block-scaled fused matmul (f32 bypass)
// ═══════════════════════════════════════════════════════════════════════
// ARM64 equivalent of matmul_tq1_block_fused_f32.
// Uses f32 activations directly (NO quantization).
// Replaces _mm256_fmadd_ps with vfmaq_f32.
extern "C" void atlas_tq1_fused_f32_arm64(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const float* act_f32, float* output, int B) {

    init_tq1_decode_lut();
    int rows_packed = rows / 4;

    const uint8_t* scale_data = tensor_data + 3;
    const uint8_t* packed = tensor_data + 3 + rows * n_blocks * 2;

    // Thread-local block scales
    static thread_local float* s_block_scales = nullptr;
    static thread_local size_t s_block_scales_cap = 0;
    {
        size_t need_bs = (size_t)rows * n_blocks;
        if (need_bs > s_block_scales_cap) {
            free(s_block_scales);
            s_block_scales = (float*)malloc(need_bs * sizeof(float));
            s_block_scales_cap = need_bs;
        }
    }
    float* block_scales = s_block_scales;

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        // Parallel fp16→fp32 scale decode
        #ifdef _OPENMP
        #pragma omp for
        #endif
        for (int i = 0; i < rows * n_blocks; i++) {
            uint16_t sr;
            memcpy(&sr, scale_data + i * 2, 2);
            block_scales[i] = fp16_to_fp32(sr);
        }

        static thread_local int8_t* tl_decode_buf = nullptr;
        static thread_local size_t tl_decode_buf_cap = 0;
        size_t need_db = (size_t)4 * input_dim;
        if (need_db > tl_decode_buf_cap) {
            free(tl_decode_buf);
            tl_decode_buf = (int8_t*)malloc(need_db * sizeof(int8_t));
            tl_decode_buf_cap = need_db;
            memset(tl_decode_buf, 0, need_db * sizeof(int8_t));
        }
        int8_t* decode_buf = tl_decode_buf;

        #ifdef _OPENMP
        #pragma omp for
        #endif
        for (int ur = 0; ur < rows_packed; ur++) {
            for (int sub = 0; sub < 4; sub++) {
                const uint8_t* w = packed + (ur * 4 + sub) * packed_cols;
                int8_t* row = decode_buf + sub * input_dim;
                int c = 0;
                for (; c < packed_cols - 1; c++) {
                    const int8_t* l = tq1_decode[w[c]];
                    int col = c * 5;
                    uint32_t v4 = (uint8_t)l[0] | ((uint32_t)(uint8_t)l[1] << 8) |
                                  ((uint32_t)(uint8_t)l[2] << 16) | ((uint32_t)(uint8_t)l[3] << 24);
                    memcpy(row + col, &v4, 4);
                    row[col + 4] = l[4];
                }
                {
                    const int8_t* l = tq1_decode[w[c]];
                    int col = c * 5;
                    if (col < input_dim) row[col] = l[0];
                    if (col + 1 < input_dim) row[col + 1] = l[1];
                    if (col + 2 < input_dim) row[col + 2] = l[2];
                    if (col + 3 < input_dim) row[col + 3] = l[3];
                    if (col + 4 < input_dim) row[col + 4] = l[4];
                }
            }

            for (int b = 0; b < B; b++) {
                const float* act = act_f32 + b * input_dim;
                float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                for (int sub = 0; sub < 4; sub++) {
                    int shuffled_r = ur * 4 + sub;
                    const float* rscales = block_scales + shuffled_r * n_blocks;
                    const int8_t* row = decode_buf + sub * input_dim;

                    for (int blk = 0; blk < n_blocks; blk++) {
                        int blk_start = blk * block_size;
                        int blk_end = blk_start + block_size;
                        if (blk_end > input_dim) blk_end = input_dim;

                        float32x4_t vacc0 = vdupq_n_f32(0);
                        float32x4_t vacc1 = vdupq_n_f32(0);

                        int j = blk_start;
                        for (; j + 8 <= blk_end; j += 8) {
                            // Load 8 int8 ternary weights, sign-extend to f32
                            int8x8_t w8 = vld1_s8(row + j);
                            int16x8_t w16 = vmovl_s8(w8);
                            float32x4_t wf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
                            float32x4_t wf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));

                            // Load f32 activations
                            float32x4_t af0 = vld1q_f32(act + j);
                            float32x4_t af1 = vld1q_f32(act + j + 4);

                            // FMA: acc += af * wf
                            vacc0 = vfmaq_f32(vacc0, af0, wf0);
                            vacc1 = vfmaq_f32(vacc1, af1, wf1);
                        }

                        float dot = vaddvq_f32(vacc0) + vaddvq_f32(vacc1);

                        for (; j < blk_end; j++) {
                            dot += act[j] * (float)row[j];
                        }

                        out4[sub] += dot * rscales[blk];
                    }
                }

                float* dst = output + b * rows;
                dst[0 * rows_packed + ur] = out4[0];
                dst[1 * rows_packed + ur] = out4[1];
                dst[2 * rows_packed + ur] = out4[2];
                dst[3 * rows_packed + ur] = out4[3];
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TQ2 wrapper (delegates to fused_s8 with flags byte skip)
// ═══════════════════════════════════════════════════════════════════════
extern "C" void atlas_tq2_arm64(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const float* act_f32, float* output, int B) {
    const uint8_t* w5 = tensor_data + 1;
    atlas_tq1_fused_s8_arm64(rows, input_dim, packed_cols,
        w5, block_size, n_blocks, act_f32, output, B);
}

// ═══════════════════════════════════════════════════════════════════════
// TQ2 f32 bypass wrapper (delegates to fused_f32 with flags byte skip)
// ═══════════════════════════════════════════════════════════════════════
extern "C" void atlas_tq2_f32_arm64(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const float* act_f32, float* output, int B) {
    const uint8_t* w5 = tensor_data + 1;
    atlas_tq1_fused_f32_arm64(rows, input_dim, packed_cols,
        w5, block_size, n_blocks, act_f32, output, B);
}

// ═══════════════════════════════════════════════════════════════════════
// Ternary-add matmul (no quant, pure sign) — NEON
// ═══════════════════════════════════════════════════════════════════════
// ARM64 equivalent of matmul_ternary_add_reorder in atlas_api.cpp.
//
// act_u8: uint8 [+128 offset] activations. weights: int8 {-1,0,+1}.
// XOR act_u8 with 0x80 → centered int8 = val-128.
// vdotq_s32(w, centered_act) = sum(w * (val-128)).
// No row_sum correction needed (built into the XOR trick).
//
// x86 pipeline: sub_epi8(-128) → sign_epi8 ×2 → cvtepi8_epi16 ×2 →
//   madd_epi16 ×2 → add_epi32 ×4 → hadd_epi32 ×3 = 11 instructions.
// NEON: veorq_s8 + vdotq_s32 ×2 + vaddvq_s32 = 4 instructions.
extern "C" void atlas_matmul_ternary_f32_arm64(int rows, int input_dim,
    const int8_t* weights, const uint8_t* act_u8,
    const float* max_abs, float scale, float* output, int B) {

    int rows_packed = rows / 4;
    int8x16_t xor_mask = vdupq_n_s8(-128);

    for (int ur = 0; ur < rows_packed; ur++) {
        for (int b = 0; b < B; b++) {
            const uint8_t* a = act_u8 + b * input_dim;
            float out4[4];

            for (int sub = 0; sub < 4; sub++) {
                const int8_t* w = weights + (ur * 4 + sub) * input_dim;

                int32x4_t acc0 = vdupq_n_s32(0);
                int32x4_t acc1 = vdupq_n_s32(0);

                int c = 0;
                for (; c + 32 <= input_dim; c += 32) {
                    uint8x16_t a0 = vld1q_u8(a + c);
                    uint8x16_t a1 = vld1q_u8(a + c + 16);
                    int8x16_t w0 = vld1q_s8(w + c);
                    int8x16_t w1 = vld1q_s8(w + c + 16);

                    int8x16_t s0 = veorq_s8(vreinterpretq_s8_u8(a0), xor_mask);
                    int8x16_t s1 = veorq_s8(vreinterpretq_s8_u8(a1), xor_mask);

                    acc0 = vdotq_s32(acc0, w0, s0);
                    acc1 = vdotq_s32(acc1, w1, s1);
                }

                int32_t dot = vaddvq_s32(acc0) + vaddvq_s32(acc1);

                for (; c < input_dim; c++) {
                    dot += ((int)a[c] - 128) * (int)w[c];
                }

                out4[sub] = (float)dot * max_abs[b] / (127.0f * scale);
            }

            float* dst = output + b * rows;
            dst[0 * rows_packed + ur] = out4[0];
            dst[1 * rows_packed + ur] = out4[1];
            dst[2 * rows_packed + ur] = out4[2];
            dst[3 * rows_packed + ur] = out4[3];
        }
    }
}

// ─── int4×uint8 matmul: nibble unpack + vdotq_s32 (ARM64 NEON) ──────────
// Replaces x86 _mm256_maddubs_epi16 + correction loop.
// XOR-0x80 trick: converts uint8 [+128 offset] activations to centered int8,
// so vdotq_s32 gives corrected dot product directly — no 128*row_sums needed.
extern "C" void atlas_matmul_i4_f32(int rows, int cols,
    const uint8_t* packed_weights, const uint8_t* act_u8,
    const int32_t* row_sums, float* output, int n_tokens) {
    (void)row_sums;
    int packed_cols = (cols + 1) / 2;
    uint8x16_t xor_80 = vdupq_n_u8(0x80);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int t = 0; t < n_tokens; t++) {
        const uint8_t* a = act_u8 + t * cols;
        for (int r = 0; r < rows; r++) {
            const uint8_t* pw = packed_weights + r * packed_cols;
            int32x4_t acc = vdupq_n_s32(0);
            int c = 0;

            for (; c + 32 <= cols; c += 32) {
                int pc = c / 2;
                uint8x16_t packed = vld1q_u8(pw + pc);
                uint8x16_t lo = vandq_u8(packed, vdupq_n_u8(0x0F));
                uint8x16_t hi = vshrq_n_u8(packed, 4);
                int8x16x2_t zipped = vzipq_s8(
                    vreinterpretq_s8_u8(lo),
                    vreinterpretq_s8_u8(hi));
                int8x16_t w0 = vsubq_s8(veorq_s8(zipped.val[0], vdupq_n_s8(8)), vdupq_n_s8(8));
                int8x16_t w1 = vsubq_s8(veorq_s8(zipped.val[1], vdupq_n_s8(8)), vdupq_n_s8(8));
                uint8x16_t au0 = vld1q_u8(a + c);
                uint8x16_t au1 = vld1q_u8(a + c + 16);
                int8x16_t act0 = vreinterpretq_s8_u8(veorq_u8(au0, xor_80));
                int8x16_t act1 = vreinterpretq_s8_u8(veorq_u8(au1, xor_80));
                acc = vdotq_s32(acc, act0, w0);
                acc = vdotq_s32(acc, act1, w1);
            }

            int32_t dot = vaddvq_s32(acc);
            for (; c < cols; c++) {
                int pc = c / 2;
                int nibble = (c & 1) ? (pw[pc] >> 4) : (pw[pc] & 0x0F);
                int8_t w_val = (int8_t)((nibble ^ 8) - 8);
                dot += ((int)a[c] ^ 0x80) * (int)w_val;
            }

            output[t * rows + r] = (float)dot;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Fused gate+up (f32 bypass): f32 activations × int8 weights, no quant
// ═══════════════════════════════════════════════════════════════════════
// ARM64 equivalent of the x86 inline loop in the f32_bypass else branch.
// Same 4-row grouped reorder output layout as x86.
extern "C" void atlas_fused_gate_up_f32_neon(int rows, int dim_w,
    const int8_t* gw, const int8_t* uw,
    const float* act_f32, int act_stride,
    float* buf_gate, float* buf_up,
    int B, float g_scale, float u_scale) {

    int rows_packed = rows / 4;

    #ifdef _OPENMP
    #pragma omp parallel for if(rows_packed > 4)
    #endif
    for (int ur = 0; ur < rows_packed; ur++) {
        const int8_t* gw4 = gw + ur * 4 * dim_w;
        const int8_t* uw4 = uw + ur * 4 * dim_w;

        for (int b = 0; b < B; b++) {
            const float* a = act_f32 + b * act_stride;
            float g_val[4], u_val[4];

            for (int sub = 0; sub < 4; sub++) {
                const int8_t* wg = gw4 + sub * dim_w;
                const int8_t* wu = uw4 + sub * dim_w;
                float32x4_t g_acc = vdupq_n_f32(0);
                float32x4_t u_acc = vdupq_n_f32(0);

                int c = 0;
                for (; c + 16 <= dim_w; c += 16) {
                    float32x4_t a0 = vld1q_f32(a + c);
                    float32x4_t a1 = vld1q_f32(a + c + 4);
                    float32x4_t a2 = vld1q_f32(a + c + 8);
                    float32x4_t a3 = vld1q_f32(a + c + 12);

                    int8x16_t wgv = vld1q_s8(wg + c);
                    int8x16_t wuv = vld1q_s8(wu + c);

                    int16x8_t wg16  = vmovl_s8(vget_low_s8(wgv));
                    int16x8_t wg16h = vmovl_s8(vget_high_s8(wgv));
                    int16x8_t wu16  = vmovl_s8(vget_low_s8(wuv));
                    int16x8_t wu16h = vmovl_s8(vget_high_s8(wuv));

                    float32x4_t g_wf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wg16)));
                    float32x4_t g_wf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wg16)));
                    float32x4_t g_wf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wg16h)));
                    float32x4_t g_wf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wg16h)));

                    float32x4_t u_wf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wu16)));
                    float32x4_t u_wf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wu16)));
                    float32x4_t u_wf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wu16h)));
                    float32x4_t u_wf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wu16h)));

                    g_acc = vfmaq_f32(g_acc, a0, g_wf0);
                    g_acc = vfmaq_f32(g_acc, a1, g_wf1);
                    g_acc = vfmaq_f32(g_acc, a2, g_wf2);
                    g_acc = vfmaq_f32(g_acc, a3, g_wf3);

                    u_acc = vfmaq_f32(u_acc, a0, u_wf0);
                    u_acc = vfmaq_f32(u_acc, a1, u_wf1);
                    u_acc = vfmaq_f32(u_acc, a2, u_wf2);
                    u_acc = vfmaq_f32(u_acc, a3, u_wf3);
                }

                float gs = vaddvq_f32(g_acc);
                float us = vaddvq_f32(u_acc);

                for (; c < dim_w; c++) {
                    gs += a[c] * wg[c];
                    us += a[c] * wu[c];
                }

                g_val[sub] = gs / g_scale;
                u_val[sub] = us / u_scale;
            }

            float* g_out = buf_gate + b * rows;
            float* u_out = buf_up + b * rows;
            g_out[0 * rows_packed + ur] = g_val[0];
            g_out[1 * rows_packed + ur] = g_val[1];
            g_out[2 * rows_packed + ur] = g_val[2];
            g_out[3 * rows_packed + ur] = g_val[3];
            u_out[0 * rows_packed + ur] = u_val[0];
            u_out[1 * rows_packed + ur] = u_val[1];
            u_out[2 * rows_packed + ur] = u_val[2];
            u_out[3 * rows_packed + ur] = u_val[3];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Fused gate+up (default quantized): uint8 activations × int8 weights
// ═══════════════════════════════════════════════════════════════════════
// ARM64 equivalent of the x86 default inline loop (vpmaddubsw + row_sums
// correction). XOR-0x80 trick eliminates the 128*row_sums subtraction.
extern "C" void atlas_fused_gate_up_default_neon(int rows, int dim_w,
    const int8_t* gw, const int8_t* uw,
    const uint8_t* act_u8,
    const float* max_abs, int B,
    float* buf_gate, float* buf_up,
    float g_scale, float u_scale) {

    int rows_packed = rows / 4;
    int8x16_t xor_mask = vdupq_n_s8(-128);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 32)
    #endif
    for (int ur = 0; ur < rows_packed; ur++) {
        const int8_t* gw4 = gw + ur * 4 * dim_w;
        const int8_t* uw4 = uw + ur * 4 * dim_w;

        for (int b = 0; b < B; b++) {
            const uint8_t* a = act_u8 + b * dim_w;
            float deq = max_abs[b] / 127.0f;

            float g_val[4], u_val[4];
            for (int sub = 0; sub < 4; sub++) {
                const int8_t* wg = gw4 + sub * dim_w;
                const int8_t* wu = uw4 + sub * dim_w;

                int32x4_t g_acc = vdupq_n_s32(0);
                int32x4_t u_acc = vdupq_n_s32(0);

                int c = 0;
                for (; c + 16 <= dim_w; c += 16) {
                    uint8x16_t av = vld1q_u8(a + c);
                    int8x16_t gv = vld1q_s8(wg + c);
                    int8x16_t uv = vld1q_s8(wu + c);
                    int8x16_t av_s = veorq_s8(vreinterpretq_s8_u8(av), xor_mask);
                    g_acc = vdotq_s32(g_acc, gv, av_s);
                    u_acc = vdotq_s32(u_acc, uv, av_s);
                }

                int g_dot = vaddvq_s32(g_acc);
                int u_dot = vaddvq_s32(u_acc);

                for (; c < dim_w; c++) {
                    int8_t av = (int8_t)(a[c] ^ 0x80);
                    g_dot += (int)av * (int)wg[c];
                    u_dot += (int)av * (int)wu[c];
                }

                g_val[sub] = (float)g_dot * deq / g_scale;
                u_val[sub] = (float)u_dot * deq / u_scale;
            }

            float* g_out = buf_gate + b * rows;
            float* u_out = buf_up + b * rows;
            g_out[0 * rows_packed + ur] = g_val[0];
            g_out[1 * rows_packed + ur] = g_val[1];
            g_out[2 * rows_packed + ur] = g_val[2];
            g_out[3 * rows_packed + ur] = g_val[3];
            u_out[0 * rows_packed + ur] = u_val[0];
            u_out[1 * rows_packed + ur] = u_val[1];
            u_out[2 * rows_packed + ur] = u_val[2];
            u_out[3 * rows_packed + ur] = u_val[3];
        }
    }
}
