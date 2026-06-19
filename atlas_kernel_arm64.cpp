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

// ─── Helper: fp16 → fp32 (NEON single-instruction FCVT) ────────────────
static inline float fp16_to_fp32(uint16_t h) {
    uint16x4_t uv = vld1_u16(&h);
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(uv)), 0);
}

// ─── TQ1 decode LUT (5 ternary trits per byte) ────────────────────────
// Two layouts for different access patterns:
//   tq1_decode[b][t] — byte-major (scalar per-byte access)
//   tq1_lut[t][b]    — trit-major  (NEON TBL batch decode)
alignas(128) static int8_t tq1_decode[256][5];
alignas(128) static int8_t tq1_lut[5][256];
static bool tq1_lut_initialized = false;

static void init_tq1_decode_lut() {
    if (tq1_lut_initialized) return;
    for (int b = 0; b < 256; b++) {
        int t = b;
        int8_t v0 = (int8_t)((t % 3) - 1); t /= 3;
        int8_t v1 = (int8_t)((t % 3) - 1); t /= 3;
        int8_t v2 = (int8_t)((t % 3) - 1); t /= 3;
        int8_t v3 = (int8_t)((t % 3) - 1); t /= 3;
        int8_t v4 = (int8_t)((t % 3) - 1);
        tq1_decode[b][0] = v0; tq1_decode[b][1] = v1;
        tq1_decode[b][2] = v2; tq1_decode[b][3] = v3; tq1_decode[b][4] = v4;
        tq1_lut[0][b] = v0; tq1_lut[1][b] = v1;
        tq1_lut[2][b] = v2; tq1_lut[3][b] = v3; tq1_lut[4][b] = v4;
    }
    tq1_lut_initialized = true;
}

// Batch decode shared buffers (removed — fused kernel decodes inline)
// The tq1_decode_row function was replaced by decode_4packed + inline decode
// in the fused kernel (v2.17.0). The function is kept for reference:
// static inline void tq1_decode_row(...) — see git history for definition.






// ═══════════════════════════════════════════════════════════════════════
// int8×uint8 matmul (row_sums correction eliminated via XOR-0x80 trick)
// ═══════════════════════════════════════════════════════════════════════
// Called from atlas_api.c before f32_bypass/quantized dispatch.
// act_u8: uint8 [0,255] with +128 offset; weights: int8.
// XOR act_u8 with 0x80 → converts to signed = val - 128.
// vdotq_s32(w, signed_act) = sum(w * (val-128)) = centered dot product.
// So NO row_sums correction needed (x86 path's -128*row_sums is built in).
extern "C" void atlas_matmul_i8_f32(int rows, int cols,
    const int8_t* __restrict__ weights,
    const uint8_t* __restrict__ act_u8,
    const int32_t* __restrict__ row_sums,
    float* __restrict__ output,
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
// Fused decode + matmul: decode 4 packed TQ1 bytes (=20 ternary values)
// into a local buffer, then vdotq_s32 / vfmaq_f32 immediately.
// Eliminates the 4×input_dim decode_buf write+read L1 roundtrip.
//
// Per-block structure for g128 (128-value blocks):
//   128 values / 20-per-chunk = 6 chunks + 8 scalar tail
//   Each packed byte spans ≤2 blocks → max 2× decode per byte
// ═══════════════════════════════════════════════════════════════════════
// v2.17.0: fused decode+matmul for symmetric int8 activations
static inline void decode_4packed(const uint8_t* src, int8_t* dst) {
    uint32_t p32;
    memcpy(&p32, src, 4);
    uint8_t b0 = (uint8_t)(p32);
    uint8_t b1 = (uint8_t)(p32 >> 8);
    uint8_t b2 = (uint8_t)(p32 >> 16);
    uint8_t b3 = (uint8_t)(p32 >> 24);
    uint32_t v0; memcpy(&v0, tq1_decode[b0], 4);
    uint32_t v1; memcpy(&v1, tq1_decode[b1], 4);
    uint32_t v2; memcpy(&v2, tq1_decode[b2], 4);
    uint32_t v3; memcpy(&v3, tq1_decode[b3], 4);
    memcpy(dst,     &v0, 4); dst[4]  = tq1_decode[b0][4];
    memcpy(dst + 5, &v1, 4); dst[9]  = tq1_decode[b1][4];
    memcpy(dst + 10,&v2, 4); dst[14] = tq1_decode[b2][4];
    memcpy(dst + 15,&v3, 4); dst[19] = tq1_decode[b3][4];
}

extern "C" void atlas_tq1_fused_s8_arm64(int rows, int input_dim, int packed_cols,
    const uint8_t* __restrict__ tensor_data, int block_size, int n_blocks,
    const float* __restrict__ act_f32, float* __restrict__ output, int B) {

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

        // FUSED KERNEL: decode-4-packed + vdotq_s32 per block
        // No decode_buf — each 4-byte chunk decoded inline, consumed immediately
        #ifdef _OPENMP
        #pragma omp for
        #endif
        for (int ur = 0; ur < rows_packed; ur++) {
            for (int b = 0; b < B; b++) {
                const int8_t* act = act_s8 + b * input_dim;
                float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                for (int sub = 0; sub < 4; sub++) {
                    const uint8_t* row_packed = packed + (ur * 4 + sub) * packed_cols;
                    int shuffled_r = ur * 4 + sub;
                    const float* rscales = block_scales + shuffled_r * n_blocks;

                    for (int blk = 0; blk < n_blocks; blk++) {
                        int blk_start = blk * block_size;
                        int blk_end = blk_start + block_size;
                        if (blk_end > input_dim) blk_end = input_dim;

                        int32x4_t acc0 = vdupq_n_s32(0);
                        int32x4_t acc1 = vdupq_n_s32(0);
                        int32_t tail_acc = 0;
                        int j = blk_start;

                        // Fused main loop: 4 packed bytes → 20 values
                        // Process 40 values per outer iteration (2× decode+matmul)
                        // for dual-accumulator ILP
                        for (; j + 40 <= blk_end; j += 40) {
                            int8_t w20a[20];
                            decode_4packed(row_packed + j / 5, w20a);
                            int8x16_t w16a = vld1q_s8(w20a);
                            int8x16_t a16a = vld1q_s8(act + j);
                            acc0 = vdotq_s32(acc0, w16a, a16a);
                            tail_acc += (int32_t)act[j+16] * (int32_t)w20a[16];
                            tail_acc += (int32_t)act[j+17] * (int32_t)w20a[17];
                            tail_acc += (int32_t)act[j+18] * (int32_t)w20a[18];
                            tail_acc += (int32_t)act[j+19] * (int32_t)w20a[19];

                            int8_t w20b[20];
                            decode_4packed(row_packed + (j + 20) / 5, w20b);
                            int8x16_t w16b = vld1q_s8(w20b);
                            int8x16_t a16b = vld1q_s8(act + j + 20);
                            acc1 = vdotq_s32(acc1, w16b, a16b);
                            tail_acc += (int32_t)act[j+36] * (int32_t)w20b[16];
                            tail_acc += (int32_t)act[j+37] * (int32_t)w20b[17];
                            tail_acc += (int32_t)act[j+38] * (int32_t)w20b[18];
                            tail_acc += (int32_t)act[j+39] * (int32_t)w20b[19];
                        }

                        int32_t dot = vaddvq_s32(acc0) + vaddvq_s32(acc1) + tail_acc;

                        for (; j + 20 <= blk_end; j += 20) {
                            int8_t w20[20];
                            decode_4packed(row_packed + j / 5, w20);
                            int8x16_t w16 = vld1q_s8(w20);
                            int8x16_t a16 = vld1q_s8(act + j);
                            acc0 = vdotq_s32(vdupq_n_s32(0), w16, a16);
                            dot += vaddvq_s32(acc0);
                            dot += (int32_t)act[j+16] * (int32_t)w20[16];
                            dot += (int32_t)act[j+17] * (int32_t)w20[17];
                            dot += (int32_t)act[j+18] * (int32_t)w20[18];
                            dot += (int32_t)act[j+19] * (int32_t)w20[19];
                        }

                        for (; j < blk_end; j++) {
                            int pc = j / 5;
                            int off = j % 5;
                            uint8_t bv = row_packed[pc];
                            dot += (int32_t)act[j] * (int32_t)tq1_decode[bv][off];
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
// TQ1 block-scaled fused matmul with f32 bypass
// ═══════════════════════════════════════════════════════════════════════
// Same fused decode+matmul structure as s8 path but uses f32 activations
// directly — vfmaq_f32 instead of vdotq_s32.
// v2.17.0: fused decode+matmul, no decode_buf
extern "C" void atlas_tq1_fused_f32_arm64(int rows, int input_dim, int packed_cols,
    const uint8_t* __restrict__ tensor_data, int block_size, int n_blocks,
    const float* __restrict__ act_f32, float* __restrict__ output, int B) {

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

        // FUSED KERNEL: decode-4-packed + vfmaq_f32 per block
        #ifdef _OPENMP
        #pragma omp for
        #endif
        for (int ur = 0; ur < rows_packed; ur++) {
            for (int b = 0; b < B; b++) {
                const float* act = act_f32 + b * input_dim;
                float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                for (int sub = 0; sub < 4; sub++) {
                    const uint8_t* row_packed = packed + (ur * 4 + sub) * packed_cols;
                    int shuffled_r = ur * 4 + sub;
                    const float* rscales = block_scales + shuffled_r * n_blocks;

                    for (int blk = 0; blk < n_blocks; blk++) {
                        int blk_start = blk * block_size;
                        int blk_end = blk_start + block_size;
                        if (blk_end > input_dim) blk_end = input_dim;

                        float32x4_t vacc0 = vdupq_n_f32(0);
                        float32x4_t vacc1 = vdupq_n_f32(0);
                        float tail_acc = 0.0f;
                        int j = blk_start;

                        // Fused main loop: 4 packed bytes → 20 f32 weights
                        // Process 40 values per outer iteration with dual-acc ILP
                        for (; j + 40 <= blk_end; j += 40) {
                            int8_t w20a[20];
                            decode_4packed(row_packed + j / 5, w20a);
                            int8x16_t w16a = vld1q_s8(w20a);
                            int16x8_t w16a_lo = vmovl_s8(vget_low_s8(w16a));
                            int16x8_t w16a_hi = vmovl_s8(vget_high_s8(w16a));
                            float32x4_t wf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16a_lo)));
                            float32x4_t wf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16a_lo)));
                            float32x4_t wf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16a_hi)));
                            float32x4_t wf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16a_hi)));
                            float32x4_t af0 = vld1q_f32(act + j);
                            float32x4_t af1 = vld1q_f32(act + j + 4);
                            float32x4_t af2 = vld1q_f32(act + j + 8);
                            float32x4_t af3 = vld1q_f32(act + j + 12);
                            vacc0 = vfmaq_f32(vacc0, af0, wf0);
                            vacc1 = vfmaq_f32(vacc1, af1, wf1);
                            vacc0 = vfmaq_f32(vacc0, af2, wf2);
                            vacc1 = vfmaq_f32(vacc1, af3, wf3);
                            tail_acc += act[j+16] * (float)w20a[16];
                            tail_acc += act[j+17] * (float)w20a[17];
                            tail_acc += act[j+18] * (float)w20a[18];
                            tail_acc += act[j+19] * (float)w20a[19];

                            int8_t w20b[20];
                            decode_4packed(row_packed + (j + 20) / 5, w20b);
                            int8x16_t w16b = vld1q_s8(w20b);
                            int16x8_t w16b_lo = vmovl_s8(vget_low_s8(w16b));
                            int16x8_t w16b_hi = vmovl_s8(vget_high_s8(w16b));
                            float32x4_t wf4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16b_lo)));
                            float32x4_t wf5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16b_lo)));
                            float32x4_t wf6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16b_hi)));
                            float32x4_t wf7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16b_hi)));
                            float32x4_t af4 = vld1q_f32(act + j + 20);
                            float32x4_t af5 = vld1q_f32(act + j + 24);
                            float32x4_t af6 = vld1q_f32(act + j + 28);
                            float32x4_t af7 = vld1q_f32(act + j + 32);
                            vacc0 = vfmaq_f32(vacc0, af4, wf4);
                            vacc1 = vfmaq_f32(vacc1, af5, wf5);
                            vacc0 = vfmaq_f32(vacc0, af6, wf6);
                            vacc1 = vfmaq_f32(vacc1, af7, wf7);
                            tail_acc += act[j+36] * (float)w20b[16];
                            tail_acc += act[j+37] * (float)w20b[17];
                            tail_acc += act[j+38] * (float)w20b[18];
                            tail_acc += act[j+39] * (float)w20b[19];
                        }

                        float dot = vaddvq_f32(vacc0) + vaddvq_f32(vacc1) + tail_acc;

                        for (; j + 20 <= blk_end; j += 20) {
                            int8_t w20[20];
                            decode_4packed(row_packed + j / 5, w20);
                            int8x16_t w16 = vld1q_s8(w20);
                            int16x8_t w16_lo = vmovl_s8(vget_low_s8(w16));
                            int16x8_t w16_hi = vmovl_s8(vget_high_s8(w16));
                            float32x4_t wf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_lo)));
                            float32x4_t wf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_lo)));
                            float32x4_t wf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_hi)));
                            float32x4_t wf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_hi)));
                            float32x4_t af0 = vld1q_f32(act + j);
                            float32x4_t af1 = vld1q_f32(act + j + 4);
                            float32x4_t af2 = vld1q_f32(act + j + 8);
                            float32x4_t af3 = vld1q_f32(act + j + 12);
                            float d0 = vaddvq_f32(vfmaq_f32(vdupq_n_f32(0), af0, wf0));
                            float d1 = vaddvq_f32(vfmaq_f32(vdupq_n_f32(0), af1, wf1));
                            float d2 = vaddvq_f32(vfmaq_f32(vdupq_n_f32(0), af2, wf2));
                            float d3 = vaddvq_f32(vfmaq_f32(vdupq_n_f32(0), af3, wf3));
                            dot += d0 + d1 + d2 + d3;
                            dot += act[j+16] * (float)w20[16];
                            dot += act[j+17] * (float)w20[17];
                            dot += act[j+18] * (float)w20[18];
                            dot += act[j+19] * (float)w20[19];
                        }

                        for (; j < blk_end; j++) {
                            int pc = j / 5;
                            int off = j % 5;
                            uint8_t bv = row_packed[pc];
                            dot += act[j] * (float)tq1_decode[bv][off];
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
    const uint8_t* __restrict__ tensor_data, int block_size, int n_blocks,
    const float* __restrict__ act_f32, float* __restrict__ output, int B) {
    const uint8_t* w5 = tensor_data + 1;
    atlas_tq1_fused_s8_arm64(rows, input_dim, packed_cols,
        w5, block_size, n_blocks, act_f32, output, B);
}

// ═══════════════════════════════════════════════════════════════════════
// TQ2 f32 bypass wrapper (delegates to fused_f32 with flags byte skip)
// ═══════════════════════════════════════════════════════════════════════
extern "C" void atlas_tq2_f32_arm64(int rows, int input_dim, int packed_cols,
    const uint8_t* __restrict__ tensor_data, int block_size, int n_blocks,
    const float* __restrict__ act_f32, float* __restrict__ output, int B) {
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
    const int8_t* __restrict__ weights, const uint8_t* __restrict__ act_u8,
    const float* __restrict__ max_abs, float scale, float* __restrict__ output, int B) {

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
    const uint8_t* __restrict__ packed_weights, const uint8_t* __restrict__ act_u8,
    const int32_t* __restrict__ row_sums, float* __restrict__ output, int n_tokens) {
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
    const int8_t* __restrict__ gw, const int8_t* __restrict__ uw,
    const float* __restrict__ act_f32, int act_stride,
    float* __restrict__ buf_gate, float* __restrict__ buf_up,
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
    const int8_t* __restrict__ gw, const int8_t* __restrict__ uw,
    const uint8_t* __restrict__ act_u8,
    const float* __restrict__ max_abs, int B,
    float* __restrict__ buf_gate, float* __restrict__ buf_up,
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
