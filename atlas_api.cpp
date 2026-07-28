// atlas_api.cpp — C-exported DLL for TQ1.0 inference acceleration
// atlas_ffi.h is the pure C API contract for FFI consumers (standalone reference)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>
#include <string>
#include <mutex>
#include <cstring>
#include <cmath>
#include <cstdint>

static inline int align_up4(int n) { return (n + 3) & ~3; }
// After a 2-byte header, ensure 4-byte alignment of int32_t array
static inline int align_after_u16(int n) { return n + ((6 - n % 4) % 4); }
static inline float safe_int_from_float(float x) {
    return (std::isnan(x) || std::isinf(x)) ? 0.0f : x;
}
static inline int32_t read_i32_unaligned(const void* p) {
    int32_t v; std::memcpy(&v, p, 4); return v;
}
static inline uint16_t read_u16_unaligned(const void* p) {
    uint16_t v; std::memcpy(&v, p, 2); return v;
}

// Normalize ARM64 detection across compilers. GCC defines __aarch64__,
// Apple Clang defines __arm64__, MSVC defines _M_ARM64. Unify to __aarch64__
// so the dozens of #if[n]def __aarch64__ guards throughout the file work.
#if (defined(__arm64__) || defined(_M_ARM64) || defined(__ARM_ARCH_ISA_A64)) && !defined(__aarch64__)
#define __aarch64__ 1
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#else
#include <immintrin.h>
extern "C" void atlas_matmul_ternary_f32_arm64(int rows, int input_dim,
    const int8_t* weights, const uint8_t* act_u8,
    const float* max_abs, float scale, float* output, int B);
#endif

// ARM64 NEON kernel declarations (atlas_kernel_arm64.cpp)
// Called when __aarch64__ is defined (Apple Silicon, ARMv8.2-A+ dotprod).
#ifdef __aarch64__
extern "C" void atlas_matmul_i8_f32(int rows, int cols,
    const int8_t* weights, const uint8_t* act_u8,
    const int32_t* row_sums, float* output, int n_tokens);
extern "C" void atlas_tq1_fused_s8_arm64(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const float* act_f32, float* output, int B);
extern "C" void atlas_tq1_fused_f32_arm64(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const float* act_f32, float* output, int B);
extern "C" void atlas_tq2_arm64(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const float* act_f32, float* output, int B);
extern "C" void atlas_tq2_f32_arm64(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const float* act_f32, float* output, int B);
extern "C" void atlas_matmul_i4_f32(int rows, int cols,
    const uint8_t* packed_weights, const uint8_t* act_u8,
    const int32_t* row_sums, float* output, int n_tokens);
extern "C" void atlas_fused_gate_up_f32_neon(int rows, int dim_w,
    const int8_t* gw, const int8_t* uw,
    const float* act_f32, int act_stride,
    float* buf_gate, float* buf_up,
    int B, float g_scale, float u_scale);
extern "C" void atlas_fused_gate_up_default_neon(int rows, int dim_w,
    const int8_t* gw, const int8_t* uw,
    const uint8_t* act_u8,
    const float* max_abs, int B,
    float* buf_gate, float* buf_up,
    float g_scale, float u_scale);
extern "C" void atlas_matmul_ternary_f32_arm64(int rows, int input_dim,
    const int8_t* weights, const uint8_t* act_u8,
    const float* max_abs, float scale, float* output, int B);

#endif  // __aarch64__

// C++ linkage — print ARM64 FFN micro-probes (no-op on non-arm64)
#ifdef __aarch64__
void profile_print_arm64();
extern "C" void profile_print_tq1();    // defined in scalar fallback PROFILE_MODE block (or as stub)
#else
static void profile_print_arm64() {}
extern "C" void profile_print_tq1() {}
#endif


#include <omp.h>

// Debug logging: compile with -DATLAS_DEBUG_MODE for runtime probes
#ifdef ATLAS_DEBUG_MODE
  #define ATLAS_LOG(fmt, ...) do { printf("[ATLAS_DBG] " fmt, ##__VA_ARGS__); fflush(stdout); } while (0)
#else
  #define ATLAS_LOG(fmt, ...) do {} while (0)
#endif

// VNNI kernel lives in atlas_vnni.cpp (compiled with target("avx10.2"))
// ARM64: not needed (NEON SDOT replaces AVX-VNNI paths).
#ifndef __aarch64__
extern "C" int atlas_vnni_available(void);
extern "C" int atlas_matmul_block_vnni(const int8_t* act, const int8_t* row, int blk_end, int blk_start);
extern "C" void atlas_matmul_i4_vnni(int rows, int cols,
                                     const uint8_t* packed_weights,
                                     const uint8_t* act_u8,
                                     const int32_t* row_sums,
                                     float* output,
                                     int n_tokens);
#endif

#ifdef _WIN32
  #define ATLAS_API __declspec(dllexport)
  #include <malloc.h>
  #include <io.h>
  #include <windows.h>
  // VirtualAlloc returns memory to OS on free — no fragmentation
  struct AllocHdr { void* base; size_t total; };
  static uint8_t* atlas_valloc(size_t size) {
      size_t hdr = sizeof(AllocHdr), align = 32;
      size_t total = hdr + size + (align - 1);
      uint8_t* base = (uint8_t*)VirtualAlloc(NULL, total,
          MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
      if (!base) return nullptr;
      uint8_t* data = (uint8_t*)(((uintptr_t)(base + hdr + align - 1)) & ~(align - 1));
      AllocHdr* h = (AllocHdr*)(data - hdr);
      h->base = base; h->total = total;
      return data;
  }
  static void atlas_vfree(uint8_t* ptr) {
      if (!ptr) return;
      AllocHdr* h = (AllocHdr*)(ptr - sizeof(AllocHdr));
      VirtualFree(h->base, 0, MEM_RELEASE);
  }
  #define FSEEK _fseeki64
  #define FTELL _ftelli64
  #define STRICMP _stricmp
#else
  #define ATLAS_API __attribute__((visibility("default")))
  #include <cstdlib>
  #include <unistd.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  // mmap-based allocator — MAP_POPULATE hints pages into RAM
  struct AllocHdr { void* base; size_t total; };
  static uint8_t* atlas_valloc(size_t size) {
      size_t hdr = sizeof(AllocHdr), align = 32;
      size_t total = hdr + size + (align - 1);
      size_t pages = (total + 4095) & ~4095;
      void* p = mmap(NULL, pages, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS
#ifdef MAP_POPULATE
                     | MAP_POPULATE
#endif
                     , -1, 0);
      if (p == MAP_FAILED) return nullptr;
      uint8_t* base = (uint8_t*)p;
      uint8_t* data = (uint8_t*)(((uintptr_t)(base + hdr + align - 1)) & ~(align - 1));
      AllocHdr* h = (AllocHdr*)(data - hdr);
      h->base = base; h->total = pages;
      return data;
  }
  static void atlas_vfree(uint8_t* ptr) {
      if (!ptr) return;
      AllocHdr* h = (AllocHdr*)(ptr - sizeof(AllocHdr));
      munmap(h->base, h->total);
  }
  #define FSEEK fseeko
  #define FTELL ftello
  #define STRICMP strcasecmp
#endif

// ─── AVX2 CPUID check ─────────────────────────────────────────────────
// Returns 1 if AVX2 is supported, 0 otherwise. Prints error to stderr.
// Prevents SIGILL on pre-Haswell CPUs (before 2013).
#ifndef __aarch64__
#ifdef _WIN32
#include <intrin.h>
static int check_avx2(void) {
    int info[4] = {0};
    __cpuidex(info, 7, 0);
    return (info[1] >> 5) & 1; // EBX bit 5 = AVX2
}
#else
#include <cpuid.h>
#include <x86intrin.h>
static int check_avx2(void) {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        return (ebx >> 5) & 1;
    return 0;
}
#endif

#ifdef _WIN32
static int check_avx512_vnni(void) {
    int info[4] = {0};
    __cpuidex(info, 7, 0);
    int has_avx512f = (info[1] >> 16) & 1;
    int has_avx512bw = (info[1] >> 30) & 1;
    int has_avx512vnni = (info[2] >> 11) & 1;
    return has_avx512f && has_avx512bw && has_avx512vnni && atlas_vnni_available();
}
#else
static int check_avx512_vnni(void) {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        int has_avx512f = (ebx >> 16) & 1;
        int has_avx512bw = (ebx >> 30) & 1;
        int has_avx512vnni = (ecx >> 11) & 1;
        return has_avx512f && has_avx512bw && has_avx512vnni && atlas_vnni_available();
    }
    return 0;
}
#endif
#endif // !__aarch64__

// ─── Runtime CPU Feature Dispatch ───────────────────────────────────────
// Populated once at atlas_load(). Kernels dispatch via g_cpu pointers.
#ifndef __aarch64__
typedef int  (*fn_dot_vnni)(const int8_t*, const int8_t*, int, int);
typedef void (*fn_matmul_i4)(int, int, const uint8_t*, const uint8_t*,
                             const int32_t*, float*, int);

struct CpuFeatures {
    int has_avx2;
    int has_avx512_vnni;
    int has_avx512_vbmi;
    fn_dot_vnni   dot_tq1_vnni;   // atlas_matmul_block_vnni or NULL
    fn_matmul_i4  matmul_i4_vnni; // atlas_matmul_i4_vnni or NULL
};

static CpuFeatures g_cpu = {};

static void atlas_init_cpu_features(void) {
    memset(&g_cpu, 0, sizeof(g_cpu));
#ifndef __aarch64__
    g_cpu.has_avx2          = check_avx2();
    g_cpu.has_avx512_vnni   = check_avx512_vnni();
    // VBMI: CPUID leaf 7, ECX bit 5
#ifdef _WIN32
    {   int info[4] = {0};
        __cpuidex(info, 7, 0);
        g_cpu.has_avx512_vbmi = (info[2] >> 5) & 1;
    }
#else
    {   unsigned int eax=0, ebx=0, ecx=0, edx=0;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
            g_cpu.has_avx512_vbmi = (ecx >> 5) & 1;
    }
#endif
    g_cpu.dot_tq1_vnni   = g_cpu.has_avx512_vnni ? atlas_matmul_block_vnni : NULL;
    g_cpu.matmul_i4_vnni = g_cpu.has_avx512_vnni ? atlas_matmul_i4_vnni  : NULL;
#endif
    if (g_cpu.has_avx512_vnni)
        printf("[ATLAS] CPU: AVX2 + AVX512_VNNI%s\n",
               g_cpu.has_avx512_vbmi ? " + VBMI" : "");
    else
        printf("[ATLAS] CPU: AVX2 only\n");
}
#endif // !__aarch64__

// --- Int4 KV cache block size ------------------------------------------------
#define KV_BLOCK_SIZE 32
#define KV_BYTES_PER_BLOCK (2 + KV_BLOCK_SIZE / 2)
static inline int kv_pos_bytes(int hd) {
    int n_blk = (hd + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    return n_blk * KV_BYTES_PER_BLOCK;
}

// ─── Xoshiro256** PRNG (thread-safe via thread_local, 64-bit) ──────────
static thread_local uint64_t xoshiro_state[4] = {0};

static void xoshiro_seed(uint64_t seed) {
    auto sm64 = [](uint64_t& s) {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    };
    xoshiro_state[0] = sm64(seed);
    xoshiro_state[1] = sm64(seed);
    xoshiro_state[2] = sm64(seed);
    xoshiro_state[3] = sm64(seed);
}

static uint64_t xoshiro_next() {
    uint64_t t = xoshiro_state[1] << 17;
    xoshiro_state[2] ^= xoshiro_state[0];
    xoshiro_state[3] ^= xoshiro_state[1];
    xoshiro_state[1] ^= xoshiro_state[2];
    xoshiro_state[0] ^= xoshiro_state[3];
    xoshiro_state[2] ^= t;
    t = xoshiro_state[3];
    t = (t ^ (t >> 16)) * 0x45D9F3BEB5AADB97ull;
    t = (t ^ (t >> 16)) * 0x45D9F3BEB5AADB97ull;
    return t ^ (t >> 16);
}

static float xoshiro_float() {
    return (float)((xoshiro_next() >> 11) * 0x1.0p-53);
}

// ─── Gumbel-max sample internal (used by both atlas_sample and atlas_generate) ──
// Modifies logits in-place as scratch. Returns sampled token ID.
// When temperature <= 0: deterministic argmax (greedy).
static int gumbel_sample(float* logits, int V,
                          float temperature, int top_k, float top_p,
                          const int* prev_ids, int n_prev,
                          float repetition_penalty) {
    if (temperature <= 0.0f) {
        int best = 0;
        float best_val = logits[0];
        for (int i = 1; i < V; i++)
            if (logits[i] > best_val) { best_val = logits[i]; best = i; }
        return best;
    }

    if (temperature != 1.0f) {
        float invT = 1.0f / temperature;
        for (int i = 0; i < V; i++) logits[i] *= invT;
    }

    // Repetition penalty: penalize logits of previously-seen tokens
    if (repetition_penalty > 1.0f && prev_ids && n_prev > 0) {
        for (int i = 0; i < n_prev; i++) {
            int tid = prev_ids[i];
            if (tid < 0 || tid >= V) continue;
            if (logits[tid] >= 0.0f)
                logits[tid] /= repetition_penalty;
            else
                logits[tid] *= repetition_penalty;
        }
    }

    static thread_local std::vector<int> sidx;

    if (top_k > 0 && top_k < V) {
        static thread_local std::vector<float> scopy;
        if ((int)scopy.size() < V) scopy.resize(V);
        memcpy(scopy.data(), logits, V * sizeof(float));
        std::nth_element(scopy.begin(), scopy.begin() + top_k - 1, scopy.end(),
                         [](float a, float b) { return a > b; });
        float threshold = scopy[top_k - 1];
        sidx.clear();
        for (int i = 0; i < V; i++) {
            if (logits[i] < threshold) {
                logits[i] = -FLT_MAX / 4;
            } else {
                sidx.push_back(i);
            }
        }
    } else {
        // Collect survivors the normal way
        sidx.clear();
        for (int i = 0; i < V; i++)
            if (logits[i] > -FLT_MAX / 8) sidx.push_back(i);
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        int S = (int)sidx.size();
        if (S > 1) {
            // Softmax on survivors only
            float max_val = logits[sidx[0]];
            for (int i = 1; i < S; i++)
                if (logits[sidx[i]] > max_val) max_val = logits[sidx[i]];
            float sum_exp = 0.0f;
            for (int i = 0; i < S; i++) sum_exp += expf(logits[sidx[i]] - max_val);
            float inv_sum = 1.0f / (sum_exp + 1e-38f);
            for (int i = 0; i < S; i++)
                logits[sidx[i]] = expf(logits[sidx[i]] - max_val) * inv_sum;

            // Max-heap on survivors only: pop largest until cum > top_p.
            std::make_heap(sidx.begin(), sidx.end(),
                           [&](int a, int b) { return logits[a] < logits[b]; });
            float cum = 0.0f;
            int n = S;
            for (int i = 0; i < S && cum < top_p; i++) {
                std::pop_heap(sidx.begin(), sidx.begin() + n,
                              [&](int a, int b) { return logits[a] < logits[b]; });
                n--;
                cum += logits[sidx[n]];
            }
            for (int j = 0; j < n; j++) logits[sidx[j]] = -FLT_MAX / 4;
        }
    }

    // Multinomial sampling on survivors (replaces Gumbel-max for stability)
    {
        // Collect valid survivors (skip -inf pruned entries)
        // Use a small array on the stack — at most top_k → 40 entries
        int valid[128];
        float val_buf[128];
        int n_valid = 0;
        for (int i : sidx) {
            if (logits[i] <= -FLT_MAX / 8) continue;
            valid[n_valid] = i;
            val_buf[n_valid] = logits[i];
            n_valid++;
            if (n_valid >= 128) break;
        }

        if (n_valid <= 1) {
            return (n_valid == 1) ? valid[0] : (sidx.empty() ? 0 : sidx[0]);
        }

        bool has_probs = (top_p > 0.0f && top_p < 1.0f);

        float target;
        if (has_probs) {
            // Survivors already have probabilities (from top_p step): sample directly
            float sum_p = 0.0f;
            for (int i = 0; i < n_valid; i++) sum_p += val_buf[i];
            float r = xoshiro_float();
            if (r <= 0.0f) r = 1e-38f;
            target = r * sum_p;
            float cum = 0.0f;
            for (int i = 0; i < n_valid; i++) {
                cum += val_buf[i];
                if (target <= cum) return valid[i];
            }
        } else {
            // Survivors have temperature-scaled logits: compute softmax
            float max_l = val_buf[0];
            for (int i = 1; i < n_valid; i++)
                if (val_buf[i] > max_l) max_l = val_buf[i];
            float sum_exp = 0.0f;
            for (int i = 0; i < n_valid; i++) {
                val_buf[i] = expf(val_buf[i] - max_l);
                sum_exp += val_buf[i];
            }
            float r = xoshiro_float();
            if (r <= 0.0f) r = 1e-38f;
            target = r * sum_exp;
            float cum = 0.0f;
            for (int i = 0; i < n_valid; i++) {
                cum += val_buf[i];
                if (target <= cum) return valid[i];
            }
        }
        return valid[n_valid - 1];
    }
}

extern "C" {

// Callback for streaming generation (v2.3.0)
typedef void (*atlas_token_callback)(int token_id, void* user_data);

// ─── v2.18.0: Grammar-constrained sampling callbacks ──────────────────
typedef void (*atlas_logit_processor_cb)(float* logits, int vocab_size, void* user_data);
typedef void (*atlas_token_notify_cb)(int token_id, void* user_data);

// Timer for debug profiling
#if defined(ATLAS_DEBUG_MODE) || defined(PROFILE_MODE)
#include "atlas_timer.h"

// ─── v2.12.0: Per-operation profiling accumulators ───
// Tracks cumulative wall time across all layers + lm_head.
// Reset at start of atlas_generate, dumped at end.
struct ProfileAccum {
    double t_rmsnorm = 0;
    double t_qkv = 0;
    double t_attention = 0;
    // v2.12.0b: attention sub-section wall times
    double t_attn_prep = 0;    // QK-Norm + RoPE + KV cache store
    double t_attn_scores = 0;  // score computation (K dequant + Q·K)
    double t_attn_softmax = 0; // causal mask + softmax
    double t_attn_weighted = 0;// weighted sum (V dequant + accumulate)
    // v2.12.0b: RDTSC micro-probes for dequant breakdown (per-thread)
    uint64_t t_attn_block_cycles = 0;  // fp16 scale load per block
    uint64_t t_attn_nibble_cycles = 0; // nibble unpack + sign extend
    uint64_t t_attn_scale_cycles = 0;  // scale multiply 
    uint64_t t_attn_fma_cycles = 0;    // FMA with query/value
    double t_o_proj = 0;  // O projection matmul + residual
    double t_ffn_gate_up = 0;
    double t_silu_down = 0;  // SiLU + down projection matmul + residual
    double t_lmhead = 0;
    double t_other = 0;
    int n_layers = 0;
    int n_tokens = 0;
};
static thread_local ProfileAccum g_prof;

#define P_START(name) double _prof_##name = atlas_now()
#define P_ACCUM(name) g_prof.t_##name += atlas_now() - _prof_##name
#define P_RESTART(name) _prof_##name = atlas_now()
#define PROF_ADD_LAYERS(n) g_prof.n_layers += (n)
#define PROF_ADD_TOKENS(n) g_prof.n_tokens += (n)

// RDTSC micro-probes for int4 dequant breakdown
#define TSC_START() uint64_t _tsc = __rdtsc()
#define TSC_ACCUM(name) g_prof.t_attn_##name += (__rdtsc() - _tsc)
#define TSC_RESTART(name) do { \
    uint64_t _tsc_end = __rdtsc(); \
    g_prof.t_attn_##name += (_tsc_end - _tsc); \
    _tsc = _tsc_end; \
} while(0)

static void profile_reset() { g_prof = ProfileAccum(); }

static void profile_print() {
    double total = g_prof.t_rmsnorm + g_prof.t_qkv + g_prof.t_attention
                 + g_prof.t_o_proj + g_prof.t_ffn_gate_up + g_prof.t_silu_down
                 + g_prof.t_lmhead + g_prof.t_other;
    if (total < 0.001) return;
    fprintf(stderr, "\n═══════ PROFILE (%d layers × %d tokens, %.3fs total) ═══════\n",
            g_prof.n_layers, g_prof.n_tokens, total);
    double tok_s = g_prof.n_tokens / total;
    auto pct = [total](double t) { return 100.0 * t / total; };
    fprintf(stderr, "  RMSNorm       %7.3fs (%5.1f%%) — avg %.3fms/layer\n",
            g_prof.t_rmsnorm, pct(g_prof.t_rmsnorm),
            1000.0 * g_prof.t_rmsnorm / (g_prof.n_layers * g_prof.n_tokens));
    fprintf(stderr, "  QKV matmul    %7.3fs (%5.1f%%)\n",
            g_prof.t_qkv, pct(g_prof.t_qkv));
    double t_attn_detail = g_prof.t_attn_prep + g_prof.t_attn_scores + g_prof.t_attn_softmax + g_prof.t_attn_weighted;
    if (t_attn_detail > 0.001) {
        fprintf(stderr, "  Attention     %7.3fs (%5.1f%%) [prep %.3f scores %.3f smax %.3f weighted %.3f]\n",
                g_prof.t_attention, pct(g_prof.t_attention),
                g_prof.t_attn_prep, g_prof.t_attn_scores,
                g_prof.t_attn_softmax, g_prof.t_attn_weighted);
        uint64_t tot_cyc = g_prof.t_attn_block_cycles + g_prof.t_attn_nibble_cycles
                         + g_prof.t_attn_scale_cycles + g_prof.t_attn_fma_cycles;
        if (tot_cyc > 0) {
            auto pct_cyc = [tot_cyc](uint64_t c) { return 100.0 * (double)c / (double)tot_cyc; };
            fprintf(stderr, "    ├ dequant micro (main thread cycles): %.0fM total\n",
                    (double)tot_cyc / 1e6);
            fprintf(stderr, "    ├ block scale load: %llu (%5.1f%%)\n",
                    g_prof.t_attn_block_cycles, pct_cyc(g_prof.t_attn_block_cycles));
            fprintf(stderr, "    ├ nibble unpack:   %llu (%5.1f%%)\n",
                    g_prof.t_attn_nibble_cycles, pct_cyc(g_prof.t_attn_nibble_cycles));
            fprintf(stderr, "    ├ scale multiply:  %llu (%5.1f%%)\n",
                    g_prof.t_attn_scale_cycles, pct_cyc(g_prof.t_attn_scale_cycles));
            fprintf(stderr, "    └ FMA:             %llu (%5.1f%%)\n",
                    g_prof.t_attn_fma_cycles, pct_cyc(g_prof.t_attn_fma_cycles));
        }
    } else {
        fprintf(stderr, "  Attention     %7.3fs (%5.1f%%) — avg %.3fms/layer\n",
                g_prof.t_attention, pct(g_prof.t_attention),
                1000.0 * g_prof.t_attention / (g_prof.n_layers * g_prof.n_tokens));
    }
    fprintf(stderr, "  O proj        %7.3fs (%5.1f%%)\n",
            g_prof.t_o_proj, pct(g_prof.t_o_proj));
    fprintf(stderr, "  FFN gate+up   %7.3fs (%5.1f%%)\n",
            g_prof.t_ffn_gate_up, pct(g_prof.t_ffn_gate_up));
    fprintf(stderr, "  SiLU+down     %7.3fs (%5.1f%%)\n",
            g_prof.t_silu_down, pct(g_prof.t_silu_down));
    fprintf(stderr, "  lm_head       %7.3fs (%5.1f%%) — avg %.3fms/token\n",
            g_prof.t_lmhead, pct(g_prof.t_lmhead), 1000.0 * g_prof.t_lmhead / g_prof.n_tokens);
    fprintf(stderr, "  other         %7.3fs (%5.1f%%)\n",
            g_prof.t_other, pct(g_prof.t_other));
    fprintf(stderr, "═══════════════════════════════════════════════════\n\n");
}
#else
#define P_START(name)
#define P_ACCUM(name)
#define P_RESTART(name)
#define TSC_START()
#define TSC_ACCUM(name)
#define TSC_RESTART(name)
#define PROF_ADD_LAYERS(n)
#define PROF_ADD_TOKENS(n)
static void profile_reset() {}
static void profile_print() {}
#endif

static inline float fp16_to_fp32(uint16_t h) {
#ifndef __aarch64__
    float r;
    __m128i h4 = _mm_cvtsi32_si128((int)(unsigned)h);
    __m128 f4 = _mm_cvtph_ps(h4);
    _mm_store_ss(&r, f4);
    return r;
#else
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
    exp = exp + 127 - 15;
    mant <<= 13;
    uint32_t result = (sign << 31) | (exp << 23) | mant;
    float r; memcpy(&r, &result, 4);
    return r;
#endif
}
static inline uint16_t fp32_to_fp16(float v) {
#ifndef __aarch64__
    __m128 f4 = _mm_set_ss(v);
    return (uint16_t)(unsigned)_mm_extract_epi16(_mm_cvtps_ph(f4, 0), 0);
#else
    uint32_t u; memcpy(&u, &v, 4);
    uint32_t sign = (u >> 31) & 1;
    int32_t exp = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;
    if (exp == 0) {
        return sign ? 0x8000 : 0;
    } else if (exp == 255) {
        return sign ? 0xFC00 : 0x7C00;
    }
    exp = exp - 127 + 15;
    if (exp >= 31) {
        return sign ? 0xFC00 : 0x7C00;
    } else if (exp <= 0) {
        if (exp < -10) return sign ? 0x8000 : 0;
        mant = (mant | 0x800000) >> (1 - exp);
        exp = 0;
    }
    uint16_t result = (sign << 15) | ((uint16_t)exp << 10) | (mant >> 13);
    return result;
#endif
}

// ─── Tensor info ────────────────────────────────────────────────────────
struct TensorInfo {
    int ttype;          // 0=TQ1, 1=norm/embed, 2=other, 5=TQ1+per-block scales, 10=TQ2 (universal)
    int row_dim;        // output rows (weight) or flat size (norm/embed)
    int packed_cols;    // packed bytes per row (0 for non-TQ1)
    uint32_t file_offset;
    uint8_t* data;      // loaded data (scale prefix + packed for TQ1, raw fp16 for others)
    int data_size;
    int block_size;     // per-block scale group size (0=ttype 0/3, 128=g128)
    int n_blocks;       // number of per-block scale groups (0=unused)
};

enum { ARCH_LLAMA = 0, ARCH_QWEN3 = 1, ARCH_BITNET = 2 };

struct AtlasModel {
    int n_layers;
    int hidden_dim;
    int inter_dim;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int vocab_size;
    float rope_theta;
    float rope_scale = 1.0f; // YaRN NTK scaling (1.0 = off, 4.0 = Bonsai)
    int base_seq_len = 4096; // v2.5.0: trained context length for NTK extension
    int layer_stride = 9;    // tensors per layer (9 Falcon3, 11 Qwen3/BitNet)
    int model_arch = 0;      // 0=LLaMA, 1=Qwen3, 2=BitNet
    int eos_id = 11;   // default Falcon3 endoftext
    int eos_id2 = -1;  // second EOS (CANN: </s> ID 2)
    int pad_id = 0;    // default padding token
    bool use_f32_matmul = false; // skip activation quantization (1B model needs full precision)
    bool use_ternary_matmul = false; // v1.3.0: vpsignb-based ternary-add kernel (no multiplication)
    bool use_packed_matmul = false; // v1.3.1: operate on 2-bit packed ternary weights (4× less memory)
    bool use_hybrid_matmul = false; // v1.3.2: FFN int8 cache, QKV packed
    bool use_relu2 = false;         // v2.8.0: ReLU² activation (BitNet b1.58)
    float scale_depth_factor = 1.0f; // MiniCPM: scale_depth/sqrt(n_layers), default 1.0 = no-op
    float scale_emb = 1.0f;          // MiniCPM: embedding multiplier (default 1.0 = no-op)
    bool rope_interleaved = true;   // true=interleaved (Falcon3/Qwen3), false=half-split (Llama/BitNet)
    int rope_interleaved_set = 0;   // was explicitly set from config (v6+ meta-block)
    int has_meta = 0;               // v8: meta-block was parsed
    std::vector<TensorInfo> tensors;
    // Tensor names (loaded from v4+ atlas files, eliminates safetensors dependency)
    std::vector<std::string> tensor_names;

    // Cached layer index array for atlas_generate (v1.2.0)
    std::vector<int> layer_idx_cache;
    bool has_layer_idx = false;

    // Pre-allocated scratch buffers for forward_layer (lazy init)
    float* buf_gate = nullptr;      // [max_batch * inter_dim]
    float* buf_up = nullptr;        // [max_batch * inter_dim]
    float* buf_hidden = nullptr;    // [max_batch * inter_dim]
    float* buf_act = nullptr;       // [max_batch * max_dim] quantized f32→i8 scratch
    float* buf_ffn_f32 = nullptr;   // [max_batch * inter_dim] FP32 SwiGLU safety buffer
    uint8_t* buf_i8 = nullptr;      // [max_batch * max_dim] uint8 quantized activations
    float* buf_out = nullptr;       // [max_batch * hidden_dim] layer output ping-pong for atlas_forward
    // Attention workspace (heap-allocated, avoids stack overflow with large B)
    float* attn_ws = nullptr;       // [max_batch * n_heads * head_dim * 4]
    int max_batch = 0;
    // mmap cache handles (for int8 data loaded from .i8 file)
    void* mmap_base = nullptr;      // MapViewOfFile base (.i8 cache)
    void* mmap_handle = nullptr;    // CreateFileMapping handle (Win) / file size (Lin)
    void* mmap_file = nullptr;      // CreateFile handle (Win) / fd (Lin)

    // v2.12.0: Int4 KV cache (per-block fp16 scales, block_size=32)
    uint8_t* k_cache = nullptr;
    uint8_t* v_cache = nullptr;
    int cache_max_seq_len = 0;

    // Ensure KV cache is allocated for at least max_seq_len
    // Returns false on allocation failure (old cache preserved).
    bool ensure_cache(int max_seq_len) {
        if (max_seq_len <= cache_max_seq_len) return true;
        if (is_mla) {
            // MLA: compressed KV latent cache instead of per-head int4
            if (!ensure_compressed_kv_cache(max_seq_len)) return false;
            cache_max_seq_len = max_seq_len;
            return true;
        }
        size_t cache_sz = (size_t)n_layers * n_kv_heads * max_seq_len * kv_pos_bytes(head_dim);
        uint8_t* new_k = (uint8_t*)atlas_valloc(cache_sz);
        uint8_t* new_v = (uint8_t*)atlas_valloc(cache_sz);
        if (!new_k || !new_v) {
            if (new_k) atlas_vfree(new_k);
            if (new_v) atlas_vfree(new_v);
            return false;
        }
        if (k_cache) atlas_vfree(k_cache);
        if (v_cache) atlas_vfree(v_cache);
        k_cache = new_k;
        v_cache = new_v;
        cache_max_seq_len = max_seq_len;
        return true;
    }
    size_t mmap_size = 0;           // actual file size for range checks
    // mmap atlas file handles (for fp16 tensor data — demand-paged by OS)
    void* atlas_mmap_base = nullptr;
    void* atlas_mmap_handle = nullptr;  // CreateFileMapping handle (Win) / file size (Lin)
    void* atlas_mmap_file = nullptr;    // duplicated fd (Win: HANDLE, Lin: fd)
    size_t atlas_mmap_size = 0;        // actual file size for range checks
    // Embedded tokenizer data (v5+)
    int tokenizer_size = 0;
    uint32_t tokenizer_offset = 0;
    // v6 binary tokenizer
    int tokenizer_binary_size = 0;
    uint32_t tokenizer_binary_offset = 0;
    const uint8_t* binary_tok_base = nullptr;  // pointer into mmap'd atlas file
    struct AddedTokenSpec {
        const char* str;
        uint16_t str_len;
        uint32_t token_id;
    };
    struct {
        uint32_t magic;
        uint32_t version;
        uint32_t vocab_size;
        uint32_t max_token_length;
        uint32_t special_count;
        uint32_t flags;
        const uint32_t* offsets;       // [vocab_size]
        const uint16_t* lengths;       // [vocab_size]
        const char* pool;              // [pool_size]
        uint64_t pool_size;
        const uint32_t* merge_left;    // [vocab_size]
        const uint32_t* merge_right;   // [vocab_size]
        const uint32_t* merge_rank;    // [vocab_size]
        const uint16_t* byte_encoder;  // [256]
        const uint16_t* byte_decoder;  // [256]
        const uint32_t* special;       // [7] eos/bos/pad/unk/mask/sep/cls
        int* merge_lookup;             // hash table: (left<<12)|right → merged_id, size = vocab_size*2
        uint32_t num_added_tokens;     // out-of-vocab added tokens (e.g. Llama3 128000+)
        AddedTokenSpec* added_specs;   // allocated array, sorted by str_len descending
    } tok = {};
    // Int8 quantized lm_head (per-row symmetric, ~403 MB instead of 1.5 GB fp32)
    int8_t* lm_head_i8 = nullptr;
    int32_t* lm_head_offsets = nullptr;  // precomputed 128 * sum(w) per row
    float* lm_head_scales = nullptr;
    bool lm_head_quantized = false;

    // ─── MoE (Mixture of Experts) ────────────────────────────────────────
    int n_experts = 0;               // 0 = dense model, >0 = MoE
    int n_experts_active = 0;        // top-k experts per token
    int moe_layer_stride = 0;        // tensors per MoE layer (varies by arch)
    // Per-layer expert offset table: [n_layers][n_experts+1]
    // expert_offsets[L][e] = byte offset into repacked buffer for expert e
    std::vector<int64_t> expert_offsets;   // flat: [n_layers * (n_experts+1)]
    uint8_t* repacked_expert_data = nullptr; // contiguous buffer for all expert weights
    int64_t repacked_expert_size = 0;        // total bytes allocated
    // Per-(layer,expert,proj) → tensor index mapping
    // moe_expert_tidx[layer * n_experts * 3 + expert * 3 + proj] = tensor index
    // proj: 0=gate, 1=up, 2=down
    std::vector<int> moe_expert_tidx;

    // ─── MLA (Multi-head Latent Attention) — DeepSeek-V2/V3 ──────────────
    bool is_mla = false;             // true = MLA architecture (DeepSeek-V2)
    int kv_lora_rank = 0;            // compressed KV latent dimension (512 for DS-V2)
    int q_lora_rank = 0;             // query compression rank (0=no compression=V2-Lite, >0=V3)
    int qk_nope_head_dim = 0;        // non-RoPE portion of Q/K head (128)
    int qk_rope_head_dim = 0;        // RoPE portion of Q/K head (64)
    int v_head_dim = 0;              // V head dimension (128)
    bool has_gated_o_proj = false;   // V3: gated output projection (o_proj.0/1/2)
    int n_shared_experts = 0;        // always-active shared experts (2 for DS-V2)
    int first_k_dense_replace = 0;   // layers 0..N-1 are dense, N+ are MoE
    // Compressed KV cache: [n_layers × max_seq × compressed_kv_stride] fp16
    // Stride = kv_lora_rank + qk_rope_head_dim (c_kv + k_pe per position)
    uint8_t* compressed_kv_cache = nullptr;
    int compressed_kv_max_seq = 0;
    int compressed_kv_stride = 0;  // kv_lora_rank + qk_rope_head_dim
    // Scratch buffer for compressed latent during forward: [max_batch × kv_lora_rank]
    float* buf_c_kv = nullptr;
    // Scratch for MoE router logits: [max_batch × n_experts]
    float* buf_router = nullptr;

    bool ensure_compressed_kv_cache(int max_seq) {
        if (max_seq <= compressed_kv_max_seq) return true;
        if (compressed_kv_stride <= 0) compressed_kv_stride = kv_lora_rank + qk_rope_head_dim;
        size_t sz = (size_t)n_layers * max_seq * compressed_kv_stride * sizeof(uint16_t);
        uint8_t* new_cache = (uint8_t*)atlas_valloc(sz);
        if (!new_cache) return false;
        if (compressed_kv_cache) atlas_vfree(compressed_kv_cache);
        compressed_kv_cache = new_cache;
        compressed_kv_max_seq = max_seq;
        return true;
    }

    ~AtlasModel() {
        if (buf_gate) atlas_vfree((uint8_t*)buf_gate);
        if (buf_up) atlas_vfree((uint8_t*)buf_up);
        if (buf_hidden) atlas_vfree((uint8_t*)buf_hidden);
        if (buf_act) atlas_vfree((uint8_t*)buf_act);
        if (buf_ffn_f32) atlas_vfree((uint8_t*)buf_ffn_f32);
        if (buf_i8) atlas_vfree((uint8_t*)buf_i8);
        if (buf_out) atlas_vfree((uint8_t*)buf_out);
        if (attn_ws) atlas_vfree((uint8_t*)attn_ws);
        if (lm_head_i8) atlas_vfree((uint8_t*)lm_head_i8);
        if (lm_head_offsets) atlas_vfree((uint8_t*)lm_head_offsets);
        if (lm_head_scales) atlas_vfree((uint8_t*)lm_head_scales);
        if (repacked_expert_data) atlas_vfree(repacked_expert_data);
        if (compressed_kv_cache) atlas_vfree(compressed_kv_cache);
        if (buf_c_kv) atlas_vfree((uint8_t*)buf_c_kv);
        if (buf_router) atlas_vfree((uint8_t*)buf_router);
        delete[] tok.merge_lookup;
        delete[] tok.added_specs;
    }

    bool is_mapped(const uint8_t* ptr) const {
        if (mmap_base && mmap_size > 0) {
            if (ptr >= (const uint8_t*)mmap_base &&
                ptr < (const uint8_t*)mmap_base + mmap_size) return true;
        }
        if (atlas_mmap_base && atlas_mmap_size > 0) {
            return ptr >= (const uint8_t*)atlas_mmap_base &&
                   ptr < (const uint8_t*)atlas_mmap_base + atlas_mmap_size;
        }
        return false;
    }

    // Allocate FFN + attention scratch buffers for batch size B.
    // Returns false on allocation failure (old buffers preserved).
        bool ensure_buffers(int B) {
        if (B <= max_batch) return true;
        int max_dim = inter_dim > hidden_dim ? inter_dim : hidden_dim;
        if (n_heads * head_dim > max_dim) max_dim = n_heads * head_dim;
        int max_aligned = ((max_dim + 7) + 31) & ~31;
        size_t ws;
        if (is_mla) {
            size_t qd_sz = (size_t)B * n_heads * (qk_nope_head_dim + qk_rope_head_dim);
            ws = qd_sz * 2
                + (size_t)B * qk_rope_head_dim
                + (size_t)n_heads * (qk_nope_head_dim + v_head_dim)
                + (size_t)kv_lora_rank
                + (size_t)qk_rope_head_dim;
        } else {
            ws = (size_t)B * n_heads * head_dim * 4;
        }
        float* new_gate = (float*)atlas_valloc((size_t)B * max_dim * sizeof(float));
        float* new_up = (float*)atlas_valloc((size_t)B * max_dim * sizeof(float));
        float* new_hidden = (float*)atlas_valloc((size_t)B * max_dim * sizeof(float));
        float* new_act = (float*)atlas_valloc((size_t)B * max_aligned * sizeof(float));
        uint8_t* new_i8 = (uint8_t*)atlas_valloc((size_t)B * max_aligned * sizeof(uint8_t));
        float* new_out = (float*)atlas_valloc((size_t)B * hidden_dim * sizeof(float));
        float* new_ffn = (float*)atlas_valloc((size_t)B * max_dim * sizeof(float));
        float* new_ws = (float*)atlas_valloc((size_t)ws * sizeof(float));
        if (!new_gate || !new_up || !new_hidden || !new_act || !new_i8 || !new_out || !new_ffn || !new_ws) {
            if (new_gate) atlas_vfree((uint8_t*)new_gate);
            if (new_up) atlas_vfree((uint8_t*)new_up);
            if (new_hidden) atlas_vfree((uint8_t*)new_hidden);
            if (new_act) atlas_vfree((uint8_t*)new_act);
            if (new_i8) atlas_vfree((uint8_t*)new_i8);
            if (new_out) atlas_vfree((uint8_t*)new_out);
            if (new_ffn) atlas_vfree((uint8_t*)new_ffn);
            if (new_ws) atlas_vfree((uint8_t*)new_ws);
            return false;
        }
        if (buf_gate) atlas_vfree((uint8_t*)buf_gate);
        if (buf_up) atlas_vfree((uint8_t*)buf_up);
        if (buf_hidden) atlas_vfree((uint8_t*)buf_hidden);
        if (buf_act) atlas_vfree((uint8_t*)buf_act);
        if (buf_ffn_f32) atlas_vfree((uint8_t*)buf_ffn_f32);
        if (buf_i8) atlas_vfree((uint8_t*)buf_i8);
        if (buf_out) atlas_vfree((uint8_t*)buf_out);
        if (attn_ws) atlas_vfree((uint8_t*)attn_ws);
        buf_gate = new_gate;
        buf_up = new_up;
        buf_hidden = new_hidden;
        buf_act = new_act;
        buf_ffn_f32 = new_ffn;
        buf_i8 = new_i8;
        buf_out = new_out;
        attn_ws = new_ws;
        max_batch = B;
        // MLA scratch buffers (reallocate when batch grows)
        if (is_mla) {
            if (buf_c_kv) atlas_vfree((uint8_t*)buf_c_kv);
            buf_c_kv = (float*)atlas_valloc((size_t)B * kv_lora_rank * sizeof(float));
            if (n_experts > 0) {
                if (buf_router) atlas_vfree((uint8_t*)buf_router);
                buf_router = (float*)atlas_valloc((size_t)B * n_experts * sizeof(float));
            }
        }
        return true;
    }
};

// ─── TQ1 byte → int8 decode LUT (v1.3.1 chunked decode) ──────────────
// Decode LUT: 5 int8 trits pro Byte (1280 bytes, L1-resident)
static std::once_flag tq1_decode_init_flag;
alignas(128) static int8_t tq1_decode[256][5];

static void init_tq1_decode_lut() {
    std::call_once(tq1_decode_init_flag, [&]() {
    for (int b = 0; b < 256; b++) {
        int t = b;
        tq1_decode[b][0] = (int8_t)((t % 3) - 1); t /= 3;
        tq1_decode[b][1] = (int8_t)((t % 3) - 1); t /= 3;
        tq1_decode[b][2] = (int8_t)((t % 3) - 1); t /= 3;
        tq1_decode[b][3] = (int8_t)((t % 3) - 1); t /= 3;
        tq1_decode[b][4] = (int8_t)((t % 3) - 1);
    }
    }); // std::call_once
}

// ─── v8 meta-block parser ──────────────────────────────────────────────
// Find value position for a JSON key. Returns pointer to value or nullptr.
static const char* find_after(const char* json, const char* key) {
    const char* p = json;
    while (1) {
        p = strstr(p, key);
        if (!p) return nullptr;
        if (p > json) {
            char prev = p[-1];
            if (prev != '{' && prev != ',' && prev != ' ') {
                p++;
                continue;
            }
        }
        p += strlen(key);
        while (*p == ' ' || *p == '\t' || *p == ':') p++;
        if (*p == '"') p++;
        return p;
    }
}

static void parse_meta_block(AtlasModel* m, const char* json) {
    char arch[32] = {0};
    const char* v = find_after(json, "\"arch\"");
    if (v) {
        const char* end = strchr(v, '"');
        if (end) {
            int len = (int)(end - v);
            if (len > 31) len = 31;
            memcpy(arch, v, len); arch[len] = 0;
        }
    }
    if (arch[0] == 0) return;

    m->has_meta = 1;
    if (strcmp(arch, "falcon3") == 0) {
        m->model_arch = ARCH_LLAMA;
        m->layer_stride = 9;
    } else if (strcmp(arch, "qwen3") == 0) {
        m->model_arch = ARCH_QWEN3;
        m->layer_stride = 11;
    } else if (strcmp(arch, "bitnet") == 0 || strcmp(arch, "trilm") == 0) {
        m->model_arch = ARCH_BITNET;
        m->layer_stride = 11;
    } else if (strcmp(arch, "deepseek_v2") == 0) {
        m->model_arch = ARCH_LLAMA;  // MLA detected by kv_lora_rank below
    }

    v = find_after(json, "\"rope_interleaved\"");
    if (v) { m->rope_interleaved = (*v == 't' || *v == '1'); m->rope_interleaved_set = 1; }

    v = find_after(json, "\"use_f32_bypass\"");
    if (v) m->use_f32_matmul = (*v == 't' || *v == '1');

    v = find_after(json, "\"rope_theta\"");
    if (v) m->rope_theta = (float)strtod(v, nullptr);

    v = find_after(json, "\"rope_scale\"");
    if (v) m->rope_scale = (float)strtod(v, nullptr);

    v = find_after(json, "\"base_seq_len\"");
    if (v) m->base_seq_len = (int)strtol(v, nullptr, 10);

    v = find_after(json, "\"scale_emb\"");
    if (v) {
        m->scale_emb = (float)strtod(v, nullptr);
        printf("[ATLAS] scale_emb=%.2f\n", m->scale_emb);
    }

    v = find_after(json, "\"scale_depth\"");
    if (v) {
        float sd = (float)strtod(v, nullptr);
        int layers = m->n_layers > 0 ? m->n_layers : 1;
        m->scale_depth_factor = sd / sqrtf((float)layers);
        printf("[ATLAS] scale_depth=%.2f sqrtL=%.2f factor=%.4f\n", sd, sqrtf((float)layers), m->scale_depth_factor);
    }


    v = find_after(json, "\"head_dim\"");
    if (v) m->head_dim = (int)strtol(v, nullptr, 10);

    v = find_after(json, "\"hidden_act\"");
    if (v) {
        const char* end = strchr(v, '"');
        if (end) {
            char buf[16];
            int len = (int)(end - v);
            if (len > 15) len = 15;
            memcpy(buf, v, len); buf[len] = 0;
            m->use_relu2 = (strcmp(buf, "relu2") == 0);
        }
    }

    // ─── MLA (Multi-head Latent Attention) fields ───
    v = find_after(json, "\"kv_lora_rank\"");
    if (v) { m->kv_lora_rank = (int)strtol(v, nullptr, 10); m->is_mla = true; }
    v = find_after(json, "\"q_lora_rank\"");
    if (v) {
        if (*v == 'n' || *v == 'N') m->q_lora_rank = 0;  // "null"
        else m->q_lora_rank = (int)strtol(v, nullptr, 10);
    }
    v = find_after(json, "\"qk_nope_head_dim\"");
    if (v) m->qk_nope_head_dim = (int)strtol(v, nullptr, 10);
    v = find_after(json, "\"qk_rope_head_dim\"");
    if (v) m->qk_rope_head_dim = (int)strtol(v, nullptr, 10);
    v = find_after(json, "\"v_head_dim\"");
    if (v) m->v_head_dim = (int)strtol(v, nullptr, 10);
    v = find_after(json, "\"n_shared_experts\"");
    if (v) m->n_shared_experts = (int)strtol(v, nullptr, 10);
    v = find_after(json, "\"first_k_dense_replace\"");
    if (v) m->first_k_dense_replace = (int)strtol(v, nullptr, 10);
    v = find_after(json, "\"n_routed_experts\"");
    if (v) m->n_experts = (int)strtol(v, nullptr, 10);
    v = find_after(json, "\"n_activated_experts\"");
    if (v) m->n_experts_active = (int)strtol(v, nullptr, 10);
    v = find_after(json, "\"moe_intermediate_size\"");
    if (v) m->inter_dim = (int)strtol(v, nullptr, 10);
    if (m->is_mla) {
        m->compressed_kv_stride = m->kv_lora_rank + m->qk_rope_head_dim;
        // MLA layer_stride: 6 fixed + 3*n_shared + kv_a_layernorm + router
        // V3 adds: q_b_proj + q_a_layernorm + o_proj.1 + o_proj.2 = +4
        int ns = m->n_shared_experts > 0 ? m->n_shared_experts : 1;
        int extra_v3 = (m->q_lora_rank > 0) ? 4 : 0;  // q_b, q_a_ln, o.1, o.2
        m->layer_stride = 6 + 3 * ns + 2 + extra_v3;
        printf("[ATLAS] MLA: kv_lora=%d q_lora=%d nope=%d rope=%d v_head=%d dense_up_to=%d routed=%d act=%d shared=%d intermediate=%d stride=%d\n",
               m->kv_lora_rank, m->q_lora_rank, m->qk_nope_head_dim, m->qk_rope_head_dim, m->v_head_dim,
               m->first_k_dense_replace, m->n_experts, m->n_experts_active, m->n_shared_experts, m->inter_dim,
               m->layer_stride);
    }
}

// Forward declaration
ATLAS_API int atlas_repack_experts(AtlasModel* m);

// ─── Load model ─────────────────────────────────────────────────────────
ATLAS_API AtlasModel* atlas_load(const char* path) {
#ifndef __aarch64__
    if (!check_avx2()) {
        fprintf(stderr, "[ATLAS] Error: AVX2 instruction set required.\n");
        fprintf(stderr, "  ATLAS needs AVX2 (Haswell, ~2013+) for fast int8 matmul.\n");
        fprintf(stderr, "  Your CPU does not report AVX2 support.\n");
        return nullptr;
    }
#endif
#ifndef __aarch64__
    atlas_init_cpu_features();
#endif
    init_tq1_decode_lut();
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ATLAS] Cannot open %s\n", path); return nullptr; }

    uint8_t hdr[64];
    if (fread(hdr, 1, 64, f) != 64) { fclose(f); return nullptr; }
    if (memcmp(hdr, "ATLAS", 5) != 0) { fclose(f); return nullptr; }

    // Bytes 54-55: format_version (v2.10.0+). 0 = legacy, 2 = v2.10.0 baseline.
    uint16_t fmt_ver;
    memcpy(&fmt_ver, hdr+54, 2);
    if (fmt_ver > 2) {
        fprintf(stderr, "[ATLAS] Error: Model format version %u requires Atlas Engine "
                        "v2.11+.\n"
                        "  Download: https://github.com/xxxn3m3s1sxxx/ATLAS-TQ1_0/releases\n",
                (unsigned)fmt_ver);
        fclose(f);
        return nullptr;
    }

    AtlasModel* m = new AtlasModel();
    uint16_t tmp; memcpy(&tmp, hdr+7, 2); m->n_layers = tmp;
    memcpy(&tmp, hdr+9, 2); m->hidden_dim = tmp;
    memcpy(&tmp, hdr+11, 2); m->inter_dim = tmp;
    m->n_heads = hdr[13]; m->n_kv_heads = hdr[14];
    memcpy(&tmp, hdr+15, 2); m->head_dim = tmp;
    uint32_t tmp32; memcpy(&tmp32, hdr+17, 4); m->vocab_size = (int)tmp32;
    int n_tensors; memcpy(&n_tensors, hdr+60, 4);
    uint16_t version; memcpy(&version, hdr+5, 2);
    if (version >= 3) {
        double rt; memcpy(&rt, hdr+21, 8); m->rope_theta = (float)rt;
    } else {
        m->rope_theta = 10000.0f;  // default for old files
    }

    // v5+ tokenizer header fields
    memcpy(&tmp32, hdr+29, 4); m->tokenizer_size = (int)tmp32;
    uint32_t tok_off_u32; memcpy(&tok_off_u32, hdr+33, 4); m->tokenizer_offset = tok_off_u32;
    // v6 binary tokenizer header fields
    memcpy(&tmp32, hdr+37, 4); m->tokenizer_binary_size = (int)tmp32;
    uint32_t bin_off_u32; memcpy(&bin_off_u32, hdr+41, 4); m->tokenizer_binary_offset = bin_off_u32;

    // EOS/PAD IDs from header (bytes 45-52), fallback to defaults
    uint32_t eos_val; memcpy(&eos_val, hdr+45, 4);
    uint32_t pad_val; memcpy(&pad_val, hdr+49, 4);
    if (eos_val != 0 && eos_val != 0xFFFFFFFF) m->eos_id = (int)eos_val;
    if (pad_val != 0 && pad_val != 0xFFFFFFFF) m->pad_id = (int)pad_val;
    if (m->vocab_size == 73448) m->eos_id2 = 2; // CANN: also stop on </s>

    // Byte 53: model_flags — use_relu2 (bit 3) for ReLU² (BitNet b1.58)
    m->use_relu2 = (hdr[53] >> 3) & 1;

    printf("[ATLAS] v%d model: %dL %dH %dI %d/%d heads %d vocab %.0f theta | %d tensors %s\n",
           version,
           m->n_layers, m->hidden_dim, m->inter_dim, m->n_heads, m->n_kv_heads,
           m->vocab_size, m->rope_theta, n_tensors,
           m->tokenizer_size > 0 ? "(embedded tokenizer)" : "");

    // v8: Read meta-block after header (meta_size includes the 4-byte size field)
    uint32_t meta_size = 0;
    if (version >= 8) {
        fread(&meta_size, 1, 4, f);
        if (meta_size > 4 && meta_size <= 4096) {
            char* json_buf = (char*)malloc(meta_size - 3);
            if (json_buf && fread(json_buf, 1, meta_size - 4, f) == (size_t)(meta_size - 4)) {
                json_buf[meta_size - 4] = '\0';
                parse_meta_block(m, json_buf);
            }
            free(json_buf);
        }
    }

    // Read directory (v10+: 16-byte entries with 64-bit offset; v9-: 12-byte with 32-bit)
    int dir_stride = (version >= 10) ? 16 : 12;
    m->tensors.resize(n_tensors);
    std::vector<uint32_t> file_offsets(n_tensors);
    FSEEK(f, 64 + meta_size, SEEK_SET);
    for (int i = 0; i < n_tensors; i++) {
        uint8_t e[16]; if (fread(e, 1, dir_stride, f) != (size_t)dir_stride) { fclose(f); delete m; fprintf(stderr, "[ATLAS] Error: truncated file (corrupt tensor directory)\n"); return nullptr; }
        m->tensors[i].ttype = e[0];
        if (version >= 10) {
            uint64_t off64; memcpy(&off64, e+1, 8);
            file_offsets[i] = (uint32_t)off64; m->tensors[i].file_offset = file_offsets[i];
        } else {
            memcpy(&file_offsets[i], e+1, 4); m->tensors[i].file_offset = file_offsets[i];
        }
        int row_off = (version >= 10) ? 9 : 5;
        memcpy(&m->tensors[i].row_dim, e+row_off, 4);
        int ppr_off = (version >= 10) ? 13 : 9;
        {
            uint32_t ppr_val = e[ppr_off] | (e[ppr_off+1]<<8) | (e[ppr_off+2]<<16);
            if (m->tensors[i].ttype == 5 || m->tensors[i].ttype == 7) {
                m->tensors[i].packed_cols = ppr_val & 0x1FFFFF;
            } else {
                m->tensors[i].packed_cols = ppr_val;
            }
        }
    }

    // Load tensor names (v4+)
    m->tensor_names.clear();
    if (version >= 4) {
        int nb_size; memcpy(&nb_size, hdr+56, 4);
        if (nb_size > 0) {
            // Names stored right after directory: [name_block_size:4] [name_0\0]... 
            FSEEK(f, 64 + meta_size + n_tensors * dir_stride, SEEK_SET);
            uint8_t* nb = new uint8_t[nb_size];
            if (fread(nb, 1, nb_size, f) != (size_t)nb_size) { delete[] nb; fclose(f); delete m; fprintf(stderr, "[ATLAS] Error: truncated file (tensor names)\n"); return nullptr; }
            int pos = 4;  // skip size field
            for (int i = 0; i < n_tensors && pos < nb_size; i++) {
                const char* s = (const char*)(nb + pos);
                int len = (int)strnlen(s, nb_size - pos);
                m->tensor_names.push_back(std::string(s, len));
                pos += len + 1;
            }
            delete[] nb;
        }
    }

    // Compute file size for last-tensor calculation
    FSEEK(f, 0, SEEK_END);
    int64_t file_size = FTELL(f);
    FSEEK(f, 64, SEEK_SET);

    // Load all tensor data from mmap'd atlas file
    // mmap the entire atlas file for zero-copy access to tensor data
#ifdef _WIN32
    HANDLE hFileW = (HANDLE)_get_osfhandle(_fileno(f));
    HANDLE hDup = INVALID_HANDLE_VALUE;
    DuplicateHandle(GetCurrentProcess(), hFileW, GetCurrentProcess(),
                    &hDup, FILE_READ_DATA, FALSE, 0);
    HANDLE hMapW = CreateFileMappingW(hDup, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMapW) {
        uint8_t* map_base = (uint8_t*)MapViewOfFile(hMapW, FILE_MAP_READ, 0, 0, 0);
        if (map_base) {
            m->atlas_mmap_base = map_base;
            m->atlas_mmap_handle = hMapW;
            m->atlas_mmap_file = (void*)hDup;
            m->atlas_mmap_size = (size_t)file_size;
        } else {
            CloseHandle(hMapW); CloseHandle(hDup);
        }
    } else {
        CloseHandle(hDup);
    }
#else
    int fd = fileno(f);
    // dup fd so mmap outlives fclose
    int map_fd = dup(fd);
    if (file_size > 0 && map_fd >= 0) {
        uint8_t* map_base = (uint8_t*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, map_fd, 0);
        if (map_base != MAP_FAILED) {
            m->atlas_mmap_base = map_base;
            m->atlas_mmap_handle = (void*)(intptr_t)file_size;
            m->atlas_mmap_file = (void*)(intptr_t)map_fd;
            m->atlas_mmap_size = (size_t)file_size;
        } else {
            close(map_fd);
        }
    }
#endif

    for (int i = 0; i < n_tensors; i++) {
        auto& t = m->tensors[i];

        if (t.ttype == 5 || t.ttype == 7) {
            // Block-scaled format: [block_size:1][n_blocks:2][scales:n_blocks*2][packed_data]
            // ttype=5: TQ1 Base-3 (5 trits/byte), ttype=7: TurboQuant 2-bit (4 weights/byte)
            // Over-allocate generously; we'll fix data_size after loading.
            t.data_size = 3 + 512 * 2 + t.row_dim * t.packed_cols;
            t.block_size = 0;
            t.n_blocks = 0;
        }
        if (t.ttype == 0) {  // TQ1: 2-byte scale + packed data
            t.data_size = 2 + t.row_dim * t.packed_cols;
        } else if (t.ttype == 1) {  // raw float16 — norm, embed, or fp16 weight
            if (t.packed_cols > 1) {
                t.data_size = (int64_t)t.row_dim * t.packed_cols * 2;
            } else if (t.row_dim == m->vocab_size && i < (int)m->tensor_names.size() &&
                       (m->tensor_names[i].find("embed_tokens") != std::string::npos ||
                        m->tensor_names[i].find("token_embd") != std::string::npos)) {
                t.data_size = (int64_t)t.row_dim * m->hidden_dim * 2;
            } else {
                t.data_size = (int64_t)t.row_dim * 2;
            }
            t.packed_cols = 0;
        } else if (t.ttype == 3 || t.ttype == 8 || t.ttype == 11) {
            // ttype=3: [fp16_scale:2][i8:rows*cols][row_sums:rows*4]
            // ttype=8: [fp16_scale:2][packed_i4:rows*packed_cols][row_sums:rows*4]
            // ttype=11: [fp16_row_scales:rows*2][i8:rows*cols][row_sums:rows*4]
            int actual = (i + 1 < n_tensors)
                ? (int)(file_offsets[i + 1] - file_offsets[i])
                : (int)(file_size - (int64_t)file_offsets[i]);
            if (actual % 32 != 0) actual += 32 - (actual % 32);
            t.data_size = actual;
        } else if (t.ttype != 5 && t.ttype != 7) {  // lm_head / scales (not block-scaled TQ1)
            int actual_bytes = (i + 1 < n_tensors)
                ? (int)(file_offsets[i + 1] - file_offsets[i]) - ((int)(file_offsets[i + 1] - file_offsets[i]) % 32)
                : (int)(file_size - (int64_t)file_offsets[i]);
            int64_t expected = (int64_t)t.row_dim * m->hidden_dim * 2;
            if (actual_bytes < expected && actual_bytes > 0) {
                t.data_size = actual_bytes;
            } else {
                t.data_size = expected;
            }
            t.packed_cols = 0;
        }

        // Point into mmap'd atlas file instead of fread
        if (m->atlas_mmap_base) {
            t.data = (uint8_t*)m->atlas_mmap_base + file_offsets[i];
        } else {
            // Fallback: fread into valloc'd buffer (no mmap available)
            t.data = atlas_valloc(t.data_size);
            if (!t.data) {
                fprintf(stderr, "[ATLAS] OOM loading tensor %d (size=%lld)\n",
                        i, (long long)t.data_size);
                continue;
            }
            FSEEK(f, (int64_t)file_offsets[i], SEEK_SET);
            fread(t.data, 1, t.data_size, f);
        }

        // Post-process block-scaled tensors (ttype=5/7): parse block_size and n_blocks
        if (t.ttype == 5 || t.ttype == 7) {
            t.block_size = t.data[0];
            uint16_t nb; memcpy(&nb, t.data + 1, 2); t.n_blocks = nb;
            t.data_size = 3 + t.row_dim * t.n_blocks * 2 + t.row_dim * t.packed_cols;
        }
    }

    // Repack MoE expert weights into contiguous buffers (if MoE model)
    atlas_repack_experts(m);

    // Parse v6 binary tokenizer block if present
    if (m->tokenizer_binary_size > 0 && m->atlas_mmap_base && m->tokenizer_binary_offset > 0) {
        const uint8_t* base = (const uint8_t*)m->atlas_mmap_base + (ptrdiff_t)m->tokenizer_binary_offset;
        m->binary_tok_base = base;

        // Read header fields
        const uint32_t* h = (const uint32_t*)base;
        if (h[0] == 0x544F4B42) {  // "TOKB" magic
            m->tok.magic = h[0];
            m->tok.version = h[1];
            m->tok.vocab_size = h[2];
            m->tok.max_token_length = h[3];
            m->tok.special_count = h[4];
            m->tok.flags = h[5];
            // Offsets are stored as uint64 starting at byte 24
            const uint64_t* offs = (const uint64_t*)(base + 24);
            m->tok.offsets = (const uint32_t*)(base + offs[0]);   // offset_offsets
            m->tok.lengths = (const uint16_t*)(base + offs[1]);   // offset_lengths
            m->tok.pool = (const char*)(base + offs[2]);          // offset_pool
            m->tok.pool_size = offs[3];                            // pool_size
            m->tok.merge_left = (const uint32_t*)(base + offs[4]); // offset_merge_left
            m->tok.merge_right = (const uint32_t*)(base + offs[5]);// offset_merge_right
            m->tok.merge_rank = (const uint32_t*)(base + offs[6]); // offset_merge_rank
            m->tok.byte_encoder = (const uint16_t*)(base + offs[7]); // offset_byte_enc
            m->tok.byte_decoder = (const uint16_t*)(base + offs[8]); // offset_byte_dec
            m->tok.special = (const uint32_t*)(base + offs[9]);     // offset_special

            // Parse out-of-vocab added tokens (offs[10], byte 104)
            m->tok.num_added_tokens = h[4];  // stored at byte 16
            uint64_t added_off = (m->tok.num_added_tokens > 0) ? offs[10] : 0;
            if (added_off > 0 && (ptrdiff_t)added_off < m->tokenizer_binary_size - 8) {
                const uint8_t* ap = base + (ptrdiff_t)added_off;
                uint32_t count; memcpy(&count, ap, 4); ap += 4;
                if (count > 0 && count <= 4096) {
                    m->tok.num_added_tokens = count;
                    m->tok.added_specs = new AtlasModel::AddedTokenSpec[count];
                    for (uint32_t i = 0; i < count; i++) {
                        uint32_t slen; memcpy(&slen, ap, 4); ap += 4;
                        m->tok.added_specs[i].str = (const char*)ap;
                        m->tok.added_specs[i].str_len = (uint16_t)(slen < 65536 ? slen : 0);
                        ap += slen;
                        uint32_t pad = (4 - slen % 4) % 4;
                        ap += pad;
                        memcpy(&m->tok.added_specs[i].token_id, ap, 4); ap += 4;
                    }
                }
            }

            // Build merge lookup hash table for O(1) pair → merged_id
            // Open-addressing, table size = 2 * vocab_size (power of 2)
            int V = (int)m->tok.vocab_size;
            int ht_size = 1;
            while (ht_size < V * 2) ht_size <<= 1;
            m->tok.merge_lookup = new int[ht_size];
            // Initialize to -1 (empty)
            for (int i = 0; i < ht_size; i++) m->tok.merge_lookup[i] = -1;

            for (int merged_id = 0; merged_id < V; merged_id++) {
                uint32_t rank = m->tok.merge_rank[merged_id];
                if (rank == 0) continue;  // base token, not a merge
                uint32_t left = m->tok.merge_left[merged_id];
                uint32_t right = m->tok.merge_right[merged_id];
                if (left == 0xFFFFFFFF) continue;
                // Hash: (left << 12) | right  (12 bits for right, up to 4096 — sufficient for BPE)
                // Better: use full 32-bit hash
                uint32_t h = (left * 2654435761u) ^ (right * 2246822519u);
                int idx = (int)(h & (ht_size - 1));
                while (m->tok.merge_lookup[idx] >= 0) {
                    idx = (idx + 1) & (ht_size - 1);
                }
                // Store merged_id as value (always >= base_vocab_size)
                m->tok.merge_lookup[idx] = merged_id;
            }
        }
    }

    // Read eos_id from binary tokenizer special tokens (reliable, not header-guess)
    if (m->tok.special) {
        uint32_t tok_eos = m->tok.special[0];
        if (tok_eos != 0xFFFFFFFF) {
            m->eos_id = (int)tok_eos;
        }
    }

    fclose(f);
    return m;

fail:
    fclose(f);
    if (m) {
        // m->~AtlasModel() via delete cleans up partial allocations
        delete m;
    }
    fprintf(stderr, "[ATLAS] Error loading model (corrupt or truncated file)\n");
    return nullptr;
}

// ─── Free model ─────────────────────────────────────────────────────────
ATLAS_API void atlas_free(AtlasModel* m) {
    if (!m) return;
    // Free valloc'd tensors (not mmap'd ones, not repacked sub-pointers)
    // Note: repacked_expert_data + compressed_kv_cache freed in ~AtlasModel()
    for (auto& t : m->tensors) {
        if (!t.data) continue;
        if (m->is_mapped(t.data)) continue;
        // Skip tensors pointing into the MoE repacked buffer
        if (m->repacked_expert_data && t.data >= m->repacked_expert_data
            && t.data < m->repacked_expert_data + m->repacked_expert_size) continue;
        atlas_vfree(t.data);
    }
    // Unmap cache if loaded
    if (m->mmap_base) {
#ifdef _WIN32
        UnmapViewOfFile(m->mmap_base);
        CloseHandle(m->mmap_handle);
        CloseHandle((HANDLE)m->mmap_file);
#else
        munmap(m->mmap_base, m->mmap_size);
        close((int)(intptr_t)m->mmap_file);
#endif
    }
    // Unmap atlas file if loaded
    if (m->atlas_mmap_base) {
#ifdef _WIN32
        UnmapViewOfFile(m->atlas_mmap_base);
        CloseHandle(m->atlas_mmap_handle);
        CloseHandle((HANDLE)m->atlas_mmap_file);
#else
        munmap(m->atlas_mmap_base, m->atlas_mmap_size);
        close((int)(intptr_t)m->atlas_mmap_file);
#endif
    }
    // Free KV cache (v2.9.2: fp16 internal)
    if (m->k_cache) atlas_vfree((uint8_t*)m->k_cache);
    if (m->v_cache) atlas_vfree((uint8_t*)m->v_cache);
    delete m;
}

// ─── MoE Expert Repacking ──────────────────────────────────────────────
// Scans tensor_names for expert weights (model.layers.{L}.block_sparse_moe.experts.{E}.*)
// and copies them into a single contiguous buffer per layer for cache-friendly access.
// Returns the number of MoE layers found (0 = dense model, no-op).
//
// Layout of repacked buffer per layer:
//   [E0_gate|E0_up|E0_down|E1_gate|E1_up|E1_down|...|EN_gate|EN_up|EN_down]
// expert_offsets[layer * (n_experts+1) + e] = byte offset for expert e
// expert_offsets[layer * (n_experts+1) + n_experts] = total bytes (sentinel)
ATLAS_API int atlas_repack_experts(AtlasModel* m) {
    if (!m || m->tensor_names.empty()) return 0;
    int L = m->n_layers;
    int V = (int)m->tensor_names.size();

    // Pass 1: detect MoE — count unique expert indices in tensor names
    // We look for pattern: model.layers.{L}.block_sparse_moe.experts.{E}.
    // Also support: model.layers.{L}.mlp.experts.{E}. (Mixtral-style)
    int max_expert_idx = -1;
    int moe_layer_count = 0;

    // Per-layer, per-expert, per-projection tracking
    // key = "L:E:proj" → tensor index
    struct MoETensorKey { int layer, expert; int proj; /* 0=gate,1=up,2=down */ int tidx; };
    std::vector<MoETensorKey> moe_tensors;

    for (int i = 0; i < V; i++) {
        const std::string& name = m->tensor_names[i];
        // Match "model.layers.{L}.block_sparse_moe.experts.{E}." or "mlp.experts.{E}."
        const char* p = strstr(name.c_str(), "block_sparse_moe.experts.");
        if (!p) p = strstr(name.c_str(), "mlp.experts.");
        if (!p) continue;

        // Parse layer index: walk back to find "model.layers.{N}"
        int layer_idx = -1;
        {
            const char* lp = strstr(name.c_str(), "model.layers.");
            if (lp) layer_idx = atoi(lp + 13);
        }
        if (layer_idx < 0 || layer_idx >= L) continue;

        // Parse expert index after "experts."
        const char* ep = p + strlen("block_sparse_moe.experts.");
        if (p == strstr(name.c_str(), "mlp.experts.")) ep = p + strlen("mlp.experts.");
        int expert_idx = atoi(ep);
        if (expert_idx < 0) continue;
        if (expert_idx > max_expert_idx) max_expert_idx = expert_idx;

        // Detect projection type
        int proj = -1;
        if (strstr(ep, "gate_proj.weight")) proj = 0;
        else if (strstr(ep, "up_proj.weight")) proj = 1;
        else if (strstr(ep, "down_proj.weight")) proj = 2;
        if (proj >= 0) {
            moe_tensors.push_back({layer_idx, expert_idx, proj, i});
            moe_layer_count++;
        }
    }

    if (max_expert_idx < 0) return 0; // no MoE tensors found

    int n_experts = max_expert_idx + 1;
    m->n_experts = n_experts;

    printf("[ATLAS] MoE detected: %d experts, %d MoE layers\n",
           n_experts, moe_layer_count / n_experts);

    // Pass 2: calculate sizes and offsets
    // For each layer, compute per-expert total byte size
    // Layout per expert: [gate_data|up_data|down_data]
    m->expert_offsets.resize((size_t)L * (n_experts + 1), 0);
    int64_t total_bytes = 0;

    for (int li = 0; li < L; li++) {
        int64_t layer_offset = 0;
        for (int e = 0; e < n_experts; e++) {
            m->expert_offsets[(size_t)li * (n_experts + 1) + e] = layer_offset;
            // Sum gate + up + down sizes for this expert
            for (auto& mt : moe_tensors) {
                if (mt.layer == li && mt.expert == e) {
                    auto& t = m->tensors[mt.tidx];
                    layer_offset += t.data_size;
                }
            }
        }
        m->expert_offsets[(size_t)li * (n_experts + 1) + n_experts] = layer_offset;
        if (layer_offset > total_bytes) total_bytes = layer_offset; // per-layer peak
    }

    // Allocate contiguous buffer (use the maximum per-layer size × 2 for safety)
    // In practice, all layers should have the same expert sizes
    int64_t buf_size = 0;
    for (int li = 0; li < L; li++) {
        int64_t layer_total = m->expert_offsets[(size_t)li * (n_experts + 1) + n_experts];
        if (layer_total > buf_size) buf_size = layer_total;
    }
    // Multiply by number of layers that share the buffer (we reuse it per-layer in forward)
    // For now, allocate for 1 layer worth (forward pass repacks per-layer)
    // Actually: allocate for ALL layers to avoid re-packing
    int64_t alloc_size = 0;
    for (int li = 0; li < L; li++) {
        alloc_size += m->expert_offsets[(size_t)li * (n_experts + 1) + n_experts];
    }

    // Recompute offsets as absolute (not per-layer relative)
    {
        int64_t running = 0;
        for (int li = 0; li < L; li++) {
            int64_t layer_size = m->expert_offsets[(size_t)li * (n_experts + 1) + n_experts];
            // Shift all expert offsets by running total
            for (int e = 0; e <= n_experts; e++) {
                m->expert_offsets[(size_t)li * (n_experts + 1) + e] += running;
            }
            running += layer_size;
        }
        alloc_size = running;
    }

    m->repacked_expert_data = (uint8_t*)atlas_valloc(alloc_size);
    if (!m->repacked_expert_data) {
        fprintf(stderr, "[ATLAS] OOM allocating MoE repacked buffer (%lld bytes)\n",
                (long long)alloc_size);
        return 0;
    }
    m->repacked_expert_size = alloc_size;

    // Build (layer,expert,proj) → tensor index mapping
    m->moe_expert_tidx.resize((size_t)L * n_experts * 3, -1);
    for (auto& mt : moe_tensors) {
        m->moe_expert_tidx[(size_t)mt.layer * n_experts * 3 + mt.expert * 3 + mt.proj] = mt.tidx;
    }

    // Pass 3: copy weights into contiguous buffer
    for (auto& mt : moe_tensors) {
        auto& t = m->tensors[mt.tidx];
        if (!t.data) continue;
        // Compute destination: layer base + expert offset + projection offset
        int64_t layer_base = m->expert_offsets[(size_t)mt.layer * (n_experts + 1)];
        int64_t expert_base = m->expert_offsets[(size_t)mt.layer * (n_experts + 1) + mt.expert];
        // Within the expert, projections are laid out gate|up|down
        int64_t proj_offset = 0;
        if (mt.proj == 1) {
            // Find gate_proj size for this layer+expert to compute up offset
            for (auto& mt2 : moe_tensors) {
                if (mt2.layer == mt.layer && mt2.expert == mt.expert && mt2.proj == 0) {
                    proj_offset = m->tensors[mt2.tidx].data_size;
                    break;
                }
            }
        } else if (mt.proj == 2) {
            // Find gate + up sizes
            for (auto& mt2 : moe_tensors) {
                if (mt2.layer == mt.layer && mt2.expert == mt.expert && (mt2.proj == 0 || mt2.proj == 1)) {
                    proj_offset += m->tensors[mt2.tidx].data_size;
                }
            }
        }
        int64_t dst_offset = layer_base + (expert_base - m->expert_offsets[(size_t)mt.layer * (n_experts + 1)]) + proj_offset;
        memcpy(m->repacked_expert_data + dst_offset, t.data, t.data_size);
        // Update tensor data pointer to point into repacked buffer
        t.data = m->repacked_expert_data + dst_offset;
    }

    printf("[ATLAS] MoE repacked: %lld bytes contiguous expert buffer\n",
           (long long)alloc_size);
    return moe_layer_count / n_experts;
}

// ─── Model config struct ───────────────────────────────────────────────
typedef struct {
    int n_layers;
    int hidden_dim;
    int inter_dim;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int vocab_size;
    float rope_theta;
} AtlasModelConfig;

// ─── Get model info ────────────────────────────────────────────────────
ATLAS_API AtlasModelConfig atlas_get_config(AtlasModel* m) {
    AtlasModelConfig cfg;
    cfg.n_layers = m->n_layers;
    cfg.hidden_dim = m->hidden_dim;
    cfg.inter_dim = m->inter_dim;
    cfg.n_heads = m->n_heads;
    cfg.n_kv_heads = m->n_kv_heads;
    cfg.head_dim = m->head_dim;
    cfg.vocab_size = m->vocab_size;
    cfg.rope_theta = m->rope_theta;
    return cfg;
}

ATLAS_API float atlas_get_scale_emb(AtlasModel* m) {
    return m ? m->scale_emb : 1.0f;
}

ATLAS_API void atlas_get_info(AtlasModel* m, int* n_layers, int* hidden_dim,
                               int* inter_dim, int* n_heads, int* n_kv_heads,
                               int* head_dim, int* vocab_size) {
    if (n_layers) *n_layers = m->n_layers;
    if (hidden_dim) *hidden_dim = m->hidden_dim;
    if (inter_dim) *inter_dim = m->inter_dim;
    if (n_heads) *n_heads = m->n_heads;
    if (n_kv_heads) *n_kv_heads = m->n_kv_heads;
    if (head_dim) *head_dim = m->head_dim;
    if (vocab_size) *vocab_size = m->vocab_size;
}

// ─── Tensor name API (v4+, no safetensors dependency) ─────────────────
ATLAS_API int atlas_get_tensor_count(AtlasModel* m) {
    return m ? (int)m->tensor_names.size() : 0;
}

ATLAS_API int atlas_get_tensor_name(AtlasModel* m, int idx, char* buf, int buf_size) {
    if (!m || idx < 0 || idx >= (int)m->tensor_names.size() || !buf || buf_size <= 0)
        return 0;
    const std::string& s = m->tensor_names[idx];
    int len = (int)s.size();
    int copy = len < buf_size - 1 ? len : buf_size - 1;
    memcpy(buf, s.data(), copy);
    buf[copy] = '\0';
    return copy;
}

// ─── Embedded tokenizer API (v5+) ────────────────────────────────────
ATLAS_API const uint8_t* atlas_get_tokenizer(AtlasModel* m, int* size) {
    if (!m || !m->tokenizer_size || !m->atlas_mmap_base) {
        if (size) *size = 0; return nullptr;
    }
    if (size) *size = m->tokenizer_size;
    return (const uint8_t*)m->atlas_mmap_base + (ptrdiff_t)m->tokenizer_offset;
}

// ─── v6 Binary Tokenizer API ─────────────────────────────────────────
ATLAS_API int atlas_has_binary_tokenizer(AtlasModel* m) {
    if (!m) return 0;
    return (m->tok.magic == 0x544F4B42) ? 1 : 0;
}

// Pre-encode UTF-8 text → token IDs. First scans for out-of-vocab added tokens
// (e.g. Llama3 special tokens 128000-128255), then falls back to byte_encoder + BPE.
// Returns number of tokens, or -1 on error.
ATLAS_API int atlas_tokenizer_preencode(AtlasModel* m,
    const char* text, int text_len,
    int* out_ids, int max_ids) {
    if (!m || !text || !out_ids || m->tok.magic != 0x544F4B42)
        return -1;
    if (text_len <= 0 || max_ids <= 0) return 0;

    const uint16_t* byte_enc = m->tok.byte_encoder;
    uint32_t unk = m->tok.special ? m->tok.special[3] : 0;
    const uint32_t num_added = m->tok.num_added_tokens;
    const AtlasModel::AddedTokenSpec* specs = m->tok.added_specs;
    int n = 0;
    int i = 0;
    while (i < text_len && n < max_ids) {
        // Scan for out-of-vocab added tokens first (longest match, sorted desc)
        int matched = 0;
        for (uint32_t j = 0; j < num_added; j++) {
            uint16_t slen = specs[j].str_len;
            if (i + slen <= text_len) {
                if (memcmp(text + i, specs[j].str, slen) == 0) {
                    int id = (int)specs[j].token_id;
                    out_ids[n++] = id;
                    i += slen;
                    matched = 1;
                    break;
                }
            }
        }
        if (!matched) {
            uint8_t b = (uint8_t)text[i];
            uint16_t tid = byte_enc[b];
            out_ids[n++] = (tid != 0xFFFF) ? (int)tid : (int)unk;
            i++;
        }
    }
    return n;
}

// Run BPE merge loop on pre-encoded byte token IDs. Modifies ids in-place.
// Uses priority queue for O(n log n) instead of O(n²) scan.
// n_ids is updated to the final token count. Returns 0 on success, -1 on error.
ATLAS_API int atlas_tokenizer_merge(AtlasModel* m,
    int* ids, int* n_ids) {
    if (!m || !ids || !n_ids || m->tok.magic != 0x544F4B42)
        return -1;

    const int V = (int)m->tok.vocab_size;
    const int* merge_lookup = m->tok.merge_lookup;
    const uint32_t* merge_rank = m->tok.merge_rank;
    auto find_merge = [&](int left, int right) -> int {
        if (left < 0 || left >= V || right < 0 || right >= V) return -1;
        uint32_t h = ((uint32_t)left * 2654435761u) ^ ((uint32_t)right * 2246822519u);
        int ht_size = 1;
        while (ht_size < V * 2) ht_size <<= 1;
        int idx = (int)(h & (ht_size - 1));
        while (true) {
            int mid = merge_lookup[idx];
            if (mid < 0) break;
            if ((int)m->tok.merge_left[mid] == left &&
                (int)m->tok.merge_right[mid] == right)
                return mid;
            idx = (idx + 1) & (ht_size - 1);
        }
        return -1;
    };

    int n = *n_ids;
    if (n <= 1) return 0;

    // Min-heap of merge candidates: (rank, position, merged_id)
    // Use std::vector + std::make_heap for cache-friendly flat heap
    struct PQEntry {
        uint32_t rank;
        int pos;
        int merged_id;
    };
    auto heap_cmp = [](const PQEntry& a, const PQEntry& b) {
        return a.rank > b.rank;  // higher rank = lower priority (min-heap)
    };
    std::vector<PQEntry> heap;
    heap.reserve((size_t)n * 2);

    for (int i = 0; i < n - 1; i++) {
        int mid = find_merge(ids[i], ids[i + 1]);
        if (mid >= 0) {
            uint32_t r = merge_rank[mid];
            if (r > 0) heap.push_back({r, i, mid});
        }
    }
    std::make_heap(heap.begin(), heap.end(), heap_cmp);

    while (!heap.empty() && n > 1) {
        std::pop_heap(heap.begin(), heap.end(), heap_cmp);
        auto [rank, pos, merged_id] = heap.back();
        heap.pop_back();

        // Validate: pair still valid?
        if (pos < 0 || pos + 1 >= n) continue;
        if (find_merge(ids[pos], ids[pos + 1]) != merged_id) continue;

        // Merge
        ids[pos] = merged_id;
        // Shift down
        for (int j = pos + 1; j < n - 1; j++)
            ids[j] = ids[j + 1];
        n--;

        // Push new pairs adjacent to merge point
        if (pos > 0) {
            int mid = find_merge(ids[pos - 1], ids[pos]);
            if (mid >= 0) {
                uint32_t r = merge_rank[mid];
                if (r > 0) {
                    heap.push_back({r, pos - 1, mid});
                    std::push_heap(heap.begin(), heap.end(), heap_cmp);
                }
            }
        }
        if (pos < n - 1) {
            int mid = find_merge(ids[pos], ids[pos + 1]);
            if (mid >= 0) {
                uint32_t r = merge_rank[mid];
                if (r > 0) {
                    heap.push_back({r, pos, mid});
                    std::push_heap(heap.begin(), heap.end(), heap_cmp);
                }
            }
        }
    }

    *n_ids = n;
    return 0;
}

// Decode token IDs → UTF-8 text via pool lookup.
// Returns number of bytes written, or -1 on error.
ATLAS_API int atlas_tokenizer_decode(AtlasModel* m,
    const int* ids, int n_ids,
    char* out_text, int max_out) {
    if (!m || !ids || !out_text || m->tok.magic != 0x544F4B42)
        return -1;
    if (n_ids <= 0 || max_out <= 0) return 0;

    const uint32_t* offsets = m->tok.offsets;
    const uint16_t* lengths = m->tok.lengths;
    const char* pool = m->tok.pool;
    // special tokens array available at m->tok.special

    int pos = 0;
    for (int i = 0; i < n_ids; i++) {
        int tid = ids[i];
        const char* str = nullptr;
        uint32_t slen = 0;
        if (tid >= 0 && tid < (int)m->tok.vocab_size) {
            str = pool + offsets[tid];
            slen = lengths[tid];
        } else if (m->tok.num_added_tokens > 0 && m->tok.added_specs) {
            // Look up out-of-vocab added token by ID
            for (uint32_t j = 0; j < m->tok.num_added_tokens; j++) {
                if ((int)m->tok.added_specs[j].token_id == tid) {
                    str = m->tok.added_specs[j].str;
                    slen = m->tok.added_specs[j].str_len;
                    break;
                }
            }
        }
        if (!str) continue;
        if (pos + (int)slen > max_out - 1) {
            int copy = max_out - 1 - pos;
            if (copy > 0) {
                memcpy(out_text + pos, str, copy);
                pos += copy;
            }
            break;
        }
        memcpy(out_text + pos, str, slen);
        pos += slen;
    }
    out_text[pos] = '\0';
    return pos;
}

ATLAS_API int atlas_get_tensor_index(AtlasModel* m, const char* name) {
    if (!m || !name) return -1;
    for (int i = 0; i < (int)m->tensor_names.size(); i++) {
        if (m->tensor_names[i] == name) return i;
    }
    return -1;
}

// ─── Cache file ──────────────────────────────────────────────────────
// Save decompressed int8 tensors to a .i8 cache file for instant reload.
// Format: [n_tensors:4] then per-tensor [ttype:1][row_dim:4][pc:4][ds:4][off:8]
//         then all tensor data concatenated.

static void cache_path(const char* atlas_path, char* out, int out_size) {
    snprintf(out, out_size, "%s", atlas_path);
    int len = (int)strlen(out);
    // Replace .atlas suffix with .i8 (or append .i8 if no .atlas)
    const char* dot = strrchr(out, '.');
    if (dot && STRICMP(dot, ".atlas") == 0) {
        int prefix_len = (int)(dot - out);
        out[prefix_len] = '.';
        out[prefix_len+1] = 'i';
        out[prefix_len+2] = '8';
        out[prefix_len+3] = '\0';
    } else {
        strncat(out, ".i8", out_size - len - 1);
    }
}

ATLAS_API void atlas_save_cache(AtlasModel* m, const char* atlas_path) {
    char path[1024]; cache_path(atlas_path, path, sizeof(path));

    int n = (int)m->tensors.size();

    // Get atlas file size to prevent stale cache loading
    int64_t atlas_file_size = 0;
#ifdef _WIN32
    HANDLE hA = CreateFileA(atlas_path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hA != INVALID_HANDLE_VALUE) {
        LARGE_INTEGER fs; if (GetFileSizeEx(hA, &fs)) atlas_file_size = fs.QuadPart;
        CloseHandle(hA);
    }
#else
    struct stat st;
    if (stat(atlas_path, &st) == 0) atlas_file_size = (int64_t)st.st_size;
#endif

    // Compute total data size needed
    int64_t total_data = 0;
    for (int i = 0; i < n; i++) {
        auto& t = m->tensors[i];
        if (t.ttype == 3 && t.data_size > 0 && t.data)
            total_data += t.data_size;
    }

    int64_t header_size = 12 + (int64_t)n * 21;
    int64_t cache_size = header_size + total_data;

    // Check available disk space on Windows
#ifdef _WIN32
    char abs_path[MAX_PATH];
    char root[4] = { 0 };
    DWORD abs_len = GetFullPathNameA(path, MAX_PATH, abs_path, NULL);
    if (abs_len >= 3 && abs_len < MAX_PATH) {
        root[0] = abs_path[0]; root[1] = ':'; root[2] = '\\'; root[3] = 0;
    }
    ULARGE_INTEGER free_bytes;
    if (root[0] && GetDiskFreeSpaceExA(root, &free_bytes, NULL, NULL)) {
        if (cache_size > (int64_t)free_bytes.QuadPart) {
            printf("[CACHE] Skip: need %.1f GB, only %.1f GB free on %s\n",
                   cache_size / 1e9, (double)free_bytes.QuadPart / 1e9, root);
            return;
        }
    }
#endif

    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[CACHE] Cannot write %s\n", path); return; }
#ifdef _WIN32
    setvbuf(f, NULL, _IONBF, 0);
#endif

    fwrite(&n, 4, 1, f);
    fwrite(&atlas_file_size, 8, 1, f);

    // Only cache int8-decoded tensors (ttype==3). Non-int8 tensors (norms, embed, GQA scales)
    // have correct data from atlas_load and don't need caching.
    std::vector<int64_t> offsets(n, -1);
    int64_t cur = 0;
    for (int i = 0; i < n; i++) {
        auto& t = m->tensors[i];
        if (t.ttype == 3 && t.data_size > 0 && t.data) {
            offsets[i] = cur;
            cur += t.data_size;
        }
    }

    // Write header with correct offsets
    for (int i = 0; i < n; i++) {
        uint8_t ttype = (uint8_t)m->tensors[i].ttype;
        int row_dim = m->tensors[i].row_dim;
        int pc = m->tensors[i].packed_cols;
        int ds = m->tensors[i].data_size;
        int64_t off = offsets[i] >= 0 ? offsets[i] : 0;
        fwrite(&ttype, 1, 1, f);
        fwrite(&row_dim, 4, 1, f);
        fwrite(&pc, 4, 1, f);
        fwrite(&ds, 4, 1, f);
        fwrite(&off, 8, 1, f);
    }

    // Write data, retrying on short writes
    bool ok = true;
    for (int i = 0; i < n && ok; i++) {
        if (offsets[i] < 0) continue;
        const uint8_t* ptr = m->tensors[i].data;
        int remaining = m->tensors[i].data_size;
        while (remaining > 0) {
            size_t written = fwrite(ptr, 1, remaining, f);
            if ((int)written <= 0) {
                fprintf(stderr, "[CACHE] ERROR: tensor %d write failed (%d remaining)\n", i, remaining);
                ok = false; break;
            }
            ptr += written;
            remaining -= (int)written;
        }
    }

    fclose(f);

    if (!ok) {
        fprintf(stderr, "[CACHE] Write failed, deleting partial cache\n");
        remove(path);
    } else {
        int n_saved = 0;
        for (int i = 0; i < n; i++) if (offsets[i] >= 0) n_saved++;
        printf("[CACHE] Saved %d/%d int8 tensors (%.1f MB)\n", n_saved, n, cache_size / 1e6);
    }
}

ATLAS_API int atlas_load_cache(AtlasModel* m, const char* atlas_path) {
    char path[1024]; cache_path(atlas_path, path, sizeof(path));
    uint8_t* base = nullptr;
    void* hFile = nullptr;
    void* hMap = nullptr;
    size_t file_size = 0;

#ifdef _WIN32
    HANDLE hFileW = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFileW == INVALID_HANDLE_VALUE) return 0;
    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(hFileW, &fsize) || fsize.QuadPart < 4) {
        CloseHandle(hFileW); return 0;
    }
    file_size = (size_t)fsize.QuadPart;
    HANDLE hMapW = CreateFileMappingA(hFileW, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapW) { CloseHandle(hFileW); return 0; }
    base = (uint8_t*)MapViewOfFile(hMapW, FILE_MAP_READ, 0, 0, 0);
    if (!base) { CloseHandle(hMapW); CloseHandle(hFileW); return 0; }
    hFile = (void*)hFileW;
    hMap = (void*)hMapW;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    off_t fsize = lseek(fd, 0, SEEK_END);
    if (fsize <= 4) { close(fd); return 0; }
    file_size = (size_t)fsize;
    base = (uint8_t*)mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { close(fd); return 0; }
    hFile = (void*)(intptr_t)fd;
    hMap = (void*)(intptr_t)fsize;
#endif

    int n = *(int*)base;
    if (file_size < 12) {
#ifdef _WIN32
        UnmapViewOfFile(base); CloseHandle((HANDLE)hMap); CloseHandle((HANDLE)hFile);
#else
        munmap(base, (size_t)(intptr_t)hMap); close((int)(intptr_t)hFile);
#endif
        return 0;
    }
    int64_t min_size = 12 + (int64_t)n * 21;
    if (n != (int)m->tensors.size() || file_size < (size_t)min_size) {
#ifdef _WIN32
        UnmapViewOfFile(base); CloseHandle((HANDLE)hMap); CloseHandle((HANDLE)hFile);
#else
        munmap(base, (size_t)(intptr_t)hMap); close((int)(intptr_t)hFile);
#endif
        return 0;
    }

    // Validate atlas file size to prevent loading stale cache
    int64_t cached_atlas_size;
    memcpy(&cached_atlas_size, base + 4, 8);
    int64_t current_atlas_size = 0;
#ifdef _WIN32
    HANDLE hA = CreateFileA(atlas_path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hA != INVALID_HANDLE_VALUE) {
        LARGE_INTEGER fs; if (GetFileSizeEx(hA, &fs)) current_atlas_size = fs.QuadPart;
        CloseHandle(hA);
    }
#else
    struct stat st;
    if (stat(atlas_path, &st) == 0) current_atlas_size = (int64_t)st.st_size;
#endif
    if (cached_atlas_size != current_atlas_size || current_atlas_size == 0) {
        printf("[CACHE] Atlas file size mismatch (%lld vs %lld), ignoring\n",
               (long long)cached_atlas_size, (long long)current_atlas_size);
#ifdef _WIN32
        UnmapViewOfFile(base); CloseHandle((HANDLE)hMap); CloseHandle((HANDLE)hFile);
#else
        munmap(base, (size_t)(intptr_t)hMap); close((int)(intptr_t)hFile);
#endif
        return 0;
    }

    uint8_t* hdr = base + 12;
    int64_t data_start = min_size;

    int replaced = 0;
    for (int i = 0; i < n; i++) {
        uint8_t* e = hdr + i * 21;
        int cttype = (int)e[0];
        int row_dim; memcpy(&row_dim, e+1, 4);
        int cpc; memcpy(&cpc, e+5, 4);
        int ds; memcpy(&ds, e+9, 4);
        int64_t off; memcpy(&off, e+13, 8);

        // Validate offset + size fits within file
        if (cttype == 3 && ds > 0 && off >= 0) {
            if ((size_t)(data_start + off + ds) > file_size) {
                // Truncated/partial cache — unsafe to use
                printf("[CACHE] Truncated (tensor %d exceeds file), ignoring cache\n", i);
#ifdef _WIN32
                UnmapViewOfFile(base); CloseHandle((HANDLE)hMap); CloseHandle((HANDLE)hFile);
#else
                munmap(base, (size_t)(intptr_t)hMap); close((int)(intptr_t)hFile);
#endif
                return 0;
            }
        }

        auto& t = m->tensors[i];
        if ((t.ttype == 0 || t.ttype == 5 || t.ttype == 7) && cttype == 3 && ds > 0 && off >= 0) {
            // Validate tensor shapes match model
            if (t.row_dim != row_dim || t.packed_cols != cpc) {
                printf("[CACHE] Tensor %d shape mismatch (model: %dx%d, cache: %dx%d), ignoring\n",
                       i, t.row_dim, t.packed_cols, row_dim, cpc);
#ifdef _WIN32
                UnmapViewOfFile(base); CloseHandle((HANDLE)hMap); CloseHandle((HANDLE)hFile);
#else
                munmap(base, (size_t)(intptr_t)hMap); close((int)(intptr_t)hFile);
#endif
                return 0;
            }
            if (t.data && !m->is_mapped(t.data)) atlas_vfree(t.data);
            t.ttype = 3;
            t.data_size = ds;
            t.data = (uint8_t*)(base + data_start + off);
            replaced++;
        }
    }

    printf("[CACHE] Loaded %d/%d tensors\n", replaced, n);

    // Only keep mmap mapping if tensors were replaced (data pointers reference it).
    // If nothing was replaced, unmap immediately — avoids blocking atlas_save_cache
    // and keeps mmap_base/handle clean for the destructor.
    if (replaced > 0) {
        m->mmap_base = base;
        m->mmap_handle = hMap;
        m->mmap_file = hFile;
        m->mmap_size = file_size;
    } else {
#ifdef _WIN32
        UnmapViewOfFile(base); CloseHandle((HANDLE)hMap); CloseHandle((HANDLE)hFile);
#else
        munmap(base, (size_t)(intptr_t)hMap); close((int)(intptr_t)hFile);
#endif
    }
    return replaced > 0 ? 1 : 0;
}

static void i4_cache_path(const char* atlas_path, char* out, int out_size) {
    snprintf(out, out_size, "%s", atlas_path);
    int len = (int)strlen(out);
    const char* dot = strrchr(out, '.');
    if (dot && STRICMP(dot, ".atlas") == 0) {
        int prefix_len = (int)(dot - out);
        out[prefix_len] = '.';
        out[prefix_len+1] = 'i';
        out[prefix_len+2] = '4';
        out[prefix_len+3] = '\0';
    } else {
        strncat(out, ".i4", out_size - len - 1);
    }
}

// ─── Save int4 (ttype=8) + int8 (ttype=3) tensors to .i4 cache ───────────
// Saves the state after decompress_ffn + quantize_ffn_to_i4.
// On next load, atlas_load_i4_cache restores this state directly —
// skipping decompress + quantize entirely.
ATLAS_API void atlas_save_i4_cache(AtlasModel* m, const char* atlas_path) {
    char path[1024]; i4_cache_path(atlas_path, path, sizeof(path));

    int n = (int)m->tensors.size();

    // Get atlas file size to prevent stale cache loading
    int64_t atlas_file_size = 0;
#ifdef _WIN32
    HANDLE hA = CreateFileA(atlas_path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hA != INVALID_HANDLE_VALUE) {
        LARGE_INTEGER fs; if (GetFileSizeEx(hA, &fs)) atlas_file_size = fs.QuadPart;
        CloseHandle(hA);
    }
#else
    struct stat st;
    if (stat(atlas_path, &st) == 0) atlas_file_size = (int64_t)st.st_size;
#endif

    // Compute total data size (ttype=3 or ttype=8 only)
    int64_t total_data = 0;
    for (int i = 0; i < n; i++) {
        auto& t = m->tensors[i];
        if ((t.ttype == 3 || t.ttype == 8) && t.data_size > 0 && t.data)
            total_data += t.data_size;
    }

    int64_t header_size = 12 + (int64_t)n * 21;
    int64_t cache_size = header_size + total_data;

#ifdef _WIN32
    char abs_path[MAX_PATH];
    char root[4] = { 0 };
    DWORD abs_len = GetFullPathNameA(path, MAX_PATH, abs_path, NULL);
    if (abs_len >= 3 && abs_len < MAX_PATH) {
        root[0] = abs_path[0]; root[1] = ':'; root[2] = '\\'; root[3] = 0;
    }
    ULARGE_INTEGER free_bytes;
    if (root[0] && GetDiskFreeSpaceExA(root, &free_bytes, NULL, NULL)) {
        if (cache_size > (int64_t)free_bytes.QuadPart) {
            printf("[CACHE] i4: skip — need %.1f GB, only %.1f GB free on %s\n",
                   cache_size / 1e9, (double)free_bytes.QuadPart / 1e9, root);
            return;
        }
    }
#endif

    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[CACHE] i4: cannot write %s\n", path); return; }
#ifdef _WIN32
    setvbuf(f, NULL, _IONBF, 0);
#endif

    fwrite(&n, 4, 1, f);
    fwrite(&atlas_file_size, 8, 1, f);

    std::vector<int64_t> offsets(n, -1);
    int64_t cur = 0;
    for (int i = 0; i < n; i++) {
        auto& t = m->tensors[i];
        if ((t.ttype == 3 || t.ttype == 8) && t.data_size > 0 && t.data) {
            offsets[i] = cur;
            cur += t.data_size;
        }
    }

    for (int i = 0; i < n; i++) {
        uint8_t ttype = (uint8_t)m->tensors[i].ttype;
        int row_dim = m->tensors[i].row_dim;
        int pc = m->tensors[i].packed_cols;
        int ds = m->tensors[i].data_size;
        int64_t off = offsets[i] >= 0 ? offsets[i] : 0;
        fwrite(&ttype, 1, 1, f);
        fwrite(&row_dim, 4, 1, f);
        fwrite(&pc, 4, 1, f);
        fwrite(&ds, 4, 1, f);
        fwrite(&off, 8, 1, f);
    }

    bool ok = true;
    for (int i = 0; i < n && ok; i++) {
        if (offsets[i] < 0) continue;
        const uint8_t* ptr = m->tensors[i].data;
        int remaining = m->tensors[i].data_size;
        while (remaining > 0) {
            size_t written = fwrite(ptr, 1, remaining, f);
            if ((int)written <= 0) {
                fprintf(stderr, "[CACHE] i4: tensor %d write failed\n", i);
                ok = false; break;
            }
            ptr += written;
            remaining -= (int)written;
        }
    }

    fclose(f);

    if (!ok) {
        fprintf(stderr, "[CACHE] i4: write failed, deleting partial cache\n");
        remove(path);
    } else {
        int n_saved = 0;
        for (int i = 0; i < n; i++) if (offsets[i] >= 0) n_saved++;
        printf("[CACHE] i4: saved %d/%d tensors (%.1f MB)\n", n_saved, n, cache_size / 1e6);
    }
}

// ─── Load .i4 cache — restore int4 (ttype=8) + int8 (ttype=3) state ─────
// Returns 1 if cache was loaded, 0 if not found or invalid.
ATLAS_API int atlas_load_i4_cache(AtlasModel* m, const char* atlas_path) {
    char path[1024]; i4_cache_path(atlas_path, path, sizeof(path));
    uint8_t* base = nullptr;
    void* hFile = nullptr;
    void* hMap = nullptr;
    size_t file_size = 0;

#ifdef _WIN32
    HANDLE hFileW = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFileW == INVALID_HANDLE_VALUE) return 0;
    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(hFileW, &fsize) || fsize.QuadPart < 4) {
        CloseHandle(hFileW); return 0;
    }
    file_size = (size_t)fsize.QuadPart;
    HANDLE hMapW = CreateFileMappingA(hFileW, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapW) { CloseHandle(hFileW); return 0; }
    base = (uint8_t*)MapViewOfFile(hMapW, FILE_MAP_READ, 0, 0, 0);
    if (!base) { CloseHandle(hMapW); CloseHandle(hFileW); return 0; }
    hFile = (void*)hFileW;
    hMap = (void*)hMapW;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    off_t fsize = lseek(fd, 0, SEEK_END);
    if (fsize <= 4) { close(fd); return 0; }
    file_size = (size_t)fsize;
    base = (uint8_t*)mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { close(fd); return 0; }
    hFile = (void*)(intptr_t)fd;
    hMap = (void*)(intptr_t)fsize;
#endif

    int n = *(int*)base;
    int64_t current_atlas_size = 0;
    {
        if (file_size < 12) { goto fail; }
        int64_t min_size = 12 + (int64_t)n * 21;
        if (n != (int)m->tensors.size() || file_size < (size_t)min_size) { goto fail; }

        int64_t cached_atlas_size;
        memcpy(&cached_atlas_size, base + 4, 8);
#ifdef _WIN32
        HANDLE hA = CreateFileA(atlas_path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hA != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER fs; if (GetFileSizeEx(hA, &fs)) current_atlas_size = fs.QuadPart;
            CloseHandle(hA);
        }
#else
        struct stat st;
        if (stat(atlas_path, &st) == 0) current_atlas_size = (int64_t)st.st_size;
#endif
        if (cached_atlas_size != current_atlas_size || current_atlas_size == 0) {
            printf("[CACHE] i4: atlas file size mismatch (%lld vs %lld), ignoring\n",
                   (long long)cached_atlas_size, (long long)current_atlas_size);
            goto fail;
        }

        uint8_t* hdr = base + 12;
        int64_t data_start = min_size;
        int replaced = 0;

        for (int i = 0; i < n; i++) {
            uint8_t* e = hdr + i * 21;
            int cttype = (int)e[0];
            int row_dim; memcpy(&row_dim, e+1, 4);
            int cpc; memcpy(&cpc, e+5, 4);
            int ds; memcpy(&ds, e+9, 4);
            int64_t off; memcpy(&off, e+13, 8);

            if ((cttype == 3 || cttype == 8) && ds > 0 && off >= 0) {
                if ((size_t)(data_start + off + ds) > file_size) {
                    printf("[CACHE] i4: truncated (tensor %d exceeds file), ignoring\n", i);
                    goto fail;
                }
            }

            auto& t = m->tensors[i];
            int model_ttype = t.ttype;
            bool can_replace = (model_ttype == 0 && (cttype == 3 || cttype == 8))
                            || (model_ttype == 5 && cttype == 3)
                            || (model_ttype == 7 && cttype == 3);
            if (can_replace && ds > 0 && off >= 0) {
                if (t.row_dim != row_dim || t.packed_cols != cpc) {
                    printf("[CACHE] i4: tensor %d shape mismatch (model: %dx%d, cache: %dx%d), ignoring\n",
                           i, t.row_dim, t.packed_cols, row_dim, cpc);
                    goto fail;
                }
                if (t.data && !m->is_mapped(t.data)) atlas_vfree(t.data);
                t.ttype = cttype;
                t.data_size = ds;
                t.data = (uint8_t*)(base + data_start + off);
                replaced++;
            }
        }

        m->mmap_base = base;
        m->mmap_handle = hMap;
        m->mmap_file = hFile;
        m->mmap_size = file_size;

        printf("[CACHE] i4: loaded %d/%d tensors\n", replaced, n);
        return replaced > 0 ? 1 : 0;
    }

fail:
#ifdef _WIN32
    if (base) UnmapViewOfFile(base);
    if (hMap) CloseHandle((HANDLE)hMap);
    if (hFile) CloseHandle((HANDLE)hFile);
#else
    if (base) munmap(base, file_size);
    if (hFile) close((int)(intptr_t)hFile);
#endif
    return 0;
}

// ─── Decompress all TQ1 tensors to int8 in-place ──────────────────────
// Handles ttype=0 (raw ternary) only.
// ttype=5 (block-scaled) stays packed — fused kernel preserves per-block scales.
// Call after Python closes safetensors.
ATLAS_API void atlas_decompress_all(AtlasModel* m) {
    int total = 0;
    init_tq1_decode_lut();
    for (auto& t : m->tensors) {
        if (t.ttype != 0) continue;
        total++;

        int input_dim = t.packed_cols * 5;
        int n_vals = t.row_dim * input_dim;
        int n_vals_padded = align_after_u16(n_vals);

        // ─── ttype=0: raw ternary {-1,0,1} decompression ───
        uint8_t* new_data = atlas_valloc(2 + n_vals_padded + t.row_dim * 4);
        if (!new_data) {
            fprintf(stderr, "[ATLAS] OOM decompressing tensor (row_dim=%d)\n", t.row_dim);
            total--;
            continue;
        }
        new_data[0] = t.data[0];
        new_data[1] = t.data[1];

        int8_t* i8 = (int8_t*)(new_data + 2);
        int32_t* rs = (int32_t*)(i8 + n_vals_padded);
        const uint8_t* packed = t.data + 2;

        for (int r = 0; r < t.row_dim; r++) {
            int sum = 0;
            int pos = 0;
            for (int c = 0; c < t.packed_cols; c++) {
                const int8_t* l = tq1_decode[packed[r * t.packed_cols + c]];
                i8[r * input_dim + pos++] = l[0]; sum += l[0];
                i8[r * input_dim + pos++] = l[1]; sum += l[1];
                i8[r * input_dim + pos++] = l[2]; sum += l[2];
                i8[r * input_dim + pos++] = l[3]; sum += l[3];
                i8[r * input_dim + pos++] = l[4]; sum += l[4];
            }
            rs[r] = sum;
        }

        if (t.data && !m->is_mapped(t.data)) atlas_vfree(t.data);
        t.data = new_data;
        t.data_size = 2 + n_vals + t.row_dim * 4;
        t.ttype = 3;
    }
    printf("[ATLAS] Decompressed %d TQ1 tensors to int8\n", total);
}

// ─── Decompress ttype=5 (block-scaled) tensors to int8 ──────────────
// Converts per-block fp16 scales → uniform int8 for f32_bypass path.
// Only called for small models (hidden <= 2048) where f32_bypass avoids
// the uint8+128 signal collapse amplification.
ATLAS_API void atlas_decompress_ttype5(AtlasModel* m) {
    int total = 0;
    init_tq1_decode_lut();
    for (auto& t : m->tensors) {
        if (t.ttype != 5) continue;
        total++;
        int input_dim = t.packed_cols * 5;
        int n_vals = t.row_dim * input_dim;
        int n_vals_padded = align_after_u16(n_vals);
        int bs = t.block_size;
        int nbk = t.n_blocks;

        const uint8_t* raw_scales = t.data + 3;
        const uint8_t* packed = t.data + 3 + t.row_dim * nbk * 2;

        float* decoded_scales = (float*)malloc(t.row_dim * nbk * sizeof(float));
        for (int i = 0; i < t.row_dim * nbk; i++) {
            uint16_t sr; memcpy(&sr, raw_scales + i * 2, 2);
            decoded_scales[i] = fp16_to_fp32(sr);
        }

        float global_max = 1e-10f;
        float* f32_row = (float*)malloc(input_dim * sizeof(float));
        for (int r = 0; r < t.row_dim; r++) {
            for (int c = 0; c < t.packed_cols; c++) {
                if (packed[r * t.packed_cols + c] == 121) {
                    int col_start = c * 5;
                    for (int t2 = 0; t2 < 5 && col_start + t2 < input_dim; t2++) {
                        f32_row[col_start + t2] = 0.0f;
                    }
                    continue;
                }
                const int8_t* l = tq1_decode[packed[r * t.packed_cols + c]];
                for (int t2 = 0; t2 < 5; t2++) {
                    int col = c * 5 + t2;
                    if (col >= input_dim) break;
                    int blk = col / bs;
                    float scale = (blk < nbk) ? decoded_scales[r * nbk + blk] : 0.0f;
                    f32_row[col] = (float)l[t2] * scale;
                    float av = fabsf(f32_row[col]);
                    if (av > global_max) global_max = av;
                }
            }
        }

        float quant_scale = global_max / 127.0f;
        float stored_scale = 127.0f / global_max;

        uint8_t* new_data = atlas_valloc(2 + n_vals_padded + t.row_dim * 4);
        if (!new_data) {
            free(decoded_scales);
            free(f32_row);
            fprintf(stderr, "[ATLAS] OOM decompressing ttype=5 tensor\n");
            total--;
            continue;
        }
        uint16_t scale_u16 = fp32_to_fp16(stored_scale);
        memcpy(new_data, &scale_u16, 2);
        int8_t* i8 = (int8_t*)(new_data + 2);
        int32_t* rs = (int32_t*)(i8 + n_vals_padded);

        for (int r = 0; r < t.row_dim; r++) {
            int pos = 0;
            int sum = 0;
            for (int c = 0; c < t.packed_cols; c++) {
                if (packed[r * t.packed_cols + c] == 121) {
                    for (int t2 = 0; t2 < 5 && pos < input_dim; t2++) {
                        i8[r * input_dim + pos] = 0;
                        pos++;
                    }
                    continue;
                }
                const int8_t* l = tq1_decode[packed[r * t.packed_cols + c]];
                for (int t2 = 0; t2 < 5; t2++) {
                    int col = c * 5 + t2;
                    if (col >= input_dim) break;
                    int blk = col / bs;
                    float scale2 = (blk < nbk) ? decoded_scales[r * nbk + blk] : 0.0f;
                    float val = (float)l[t2] * scale2;
                    int q = (int)safe_int_from_float(val / quant_scale + 0.5f);
                    if (q < -127) q = -127;
                    if (q > 127) q = 127;
                    i8[r * input_dim + pos] = (int8_t)q;
                    sum += q;
                    pos++;
                }
            }
            rs[r] = sum;
        }

        free(decoded_scales);
        free(f32_row);
        // Guard: don't vfree a repacked sub-pointer (MoE contiguous buffer)
        bool is_repacked = m->repacked_expert_data &&
            t.data >= m->repacked_expert_data &&
            t.data < m->repacked_expert_data + m->repacked_expert_size;
        if (t.data && !m->is_mapped(t.data) && !is_repacked) atlas_vfree(t.data);
        t.data = new_data;
        t.data_size = 2 + n_vals + t.row_dim * 4;
        t.ttype = 3;
    }
    if (total > 0) printf("[ATLAS] Decompressed %d ttype=5 tensors to int8\n", total);
}

// ─── v2.7.0: Decompress ttype=7 (TurboQuant 2-bit) to int8 in memory ──
// Decodes 2-bit packed ternary values → dequantizes with per-block fp16
// scales → requantizes to uniform per-tensor int8. No .i8 disk cache needed.
ATLAS_API void atlas_decompress_ttype7(AtlasModel* m) {
    int total = 0;
    for (size_t i = 0; i < m->tensors.size(); i++) {
        auto& t = m->tensors[i];
        if (t.ttype != 7) continue;
        total++;

        int packed_cols = t.packed_cols;
        int input_dim = packed_cols * 4;
        int n_vals = t.row_dim * input_dim;
        int n_vals_padded = align_after_u16(n_vals);
        int block_size = t.block_size;
        int n_blocks = t.n_blocks;

        const uint8_t* raw_scales = t.data + 3;
        const uint8_t* packed = t.data + 3 + t.row_dim * n_blocks * 2;

        // Decode all rows to f32 to find global max
        float* f32_all = (float*)atlas_valloc(n_vals * sizeof(float));
        if (!f32_all) {
            fprintf(stderr, "[ATLAS] OOM decoding ttype=7 (f32_all)\n");
            total--;
            continue;
        }
        float global_max = 1e-10f;
        for (int r = 0; r < t.row_dim; r++) {
            const uint8_t* row_packed = packed + r * packed_cols;
            float* row_f32 = f32_all + r * input_dim;
            for (int c = 0; c < packed_cols; c++) {
                uint8_t byte = row_packed[c];
                for (int bit = 0; bit < 4; bit++) {
                    int col = c * 4 + bit;
                    int8_t val = (int8_t)(((byte >> (bit * 2)) & 3) - 1);
                    if (val == -1 || val == 0 || val == 1) {
                        int blk = col / block_size;
                        uint16_t ws16; memcpy(&ws16, raw_scales + r * n_blocks * 2 + blk * 2, 2);
                        row_f32[col] = (float)val * fp16_to_fp32(ws16);
                    } else {
                        row_f32[col] = 0.0f;
                    }
                    float av = fabsf(row_f32[col]);
                    if (av > global_max) global_max = av;
                }
            }
        }

        // Quantize to int8
        float quant_scale = global_max / 127.0f;
        float stored_scale = 127.0f / global_max;

        uint8_t* new_data = atlas_valloc(2 + n_vals_padded + t.row_dim * 4);
        if (!new_data) {
            atlas_vfree((uint8_t*)f32_all);
            fprintf(stderr, "[ATLAS] OOM decompressing ttype=7 (new_data)\n");
            total--;
            continue;
        }
        uint16_t scale_u16 = fp32_to_fp16(stored_scale);
        memcpy(new_data, &scale_u16, 2);
        int8_t* i8 = (int8_t*)(new_data + 2);
        int32_t* rs = (int32_t*)(i8 + n_vals_padded);

        for (int64_t i = 0; i < n_vals; i++) {
            float vq = f32_all[i] / quant_scale;
            int q = (vq >= 0) ? (int)safe_int_from_float(vq + 0.5f) : (int)safe_int_from_float(vq - 0.5f);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            i8[i] = (int8_t)q;
        }

        // Row sums for 128*row_sum correction
        for (int r = 0; r < t.row_dim; r++) {
            int sum = 0;
            for (int c = 0; c < input_dim; c++) {
                sum += i8[r * input_dim + c];
            }
            rs[r] = 128 * sum;
        }

        // Pre-shuffle rows to compensate for C++ matmul strided output layout.
        // matmul_f32_reorder writes output[(r%4)*rows_packed + r/4] = W[r]·act.
        // We store W such that this produces linear output: output[i] = W_nat[i]·act.
        // target = (r % rows_packed) * 4 + r / rows_packed
        // NOTE: skipped by packer for ttype=7 (pre_shuffle_rows only for ttype!=7).
        {
            int rows_packed = t.row_dim / 4;
            int8_t* tmp_i8 = (int8_t*)atlas_valloc(n_vals);
            int32_t* tmp_rs = (int32_t*)atlas_valloc(t.row_dim * sizeof(int32_t));
            if (!tmp_i8 || !tmp_rs) {
                if (tmp_i8) atlas_vfree((uint8_t*)tmp_i8);
                if (tmp_rs) atlas_vfree((uint8_t*)tmp_rs);
                atlas_vfree((uint8_t*)f32_all);
                atlas_vfree((uint8_t*)new_data);
                fprintf(stderr, "[ATLAS] OOM shuffling ttype=7\n");
                total--;
                continue;
            }
            for (int r = 0; r < t.row_dim; r++) {
                int target = (r % rows_packed) * 4 + r / rows_packed;
                memcpy(tmp_i8 + target * input_dim, i8 + r * input_dim, input_dim);
                tmp_rs[target] = rs[r];
            }
            memcpy(i8, tmp_i8, n_vals);
            memcpy(rs, tmp_rs, t.row_dim * sizeof(int32_t));
            atlas_vfree((uint8_t*)tmp_i8);
            atlas_vfree((uint8_t*)tmp_rs);
        }

        atlas_vfree((uint8_t*)f32_all);
        if (t.data && !m->is_mapped(t.data)) atlas_vfree(t.data);
        t.data = new_data;
        t.data_size = 2 + n_vals + t.row_dim * 4;
        t.ttype = 3;
    }
    if (total > 0) printf("[ATLAS] Decompressed %d ttype=7 tensors to int8\n", total);
}

// ─── v2.8.0: Convert FFN int8 tensors to int4 (ttype=8) in-place ───
// Halves memory bandwidth for FFN matmuls. Clip int8→int4, pack 2/byte.
ATLAS_API void atlas_quantize_ffn_to_i4(AtlasModel* m) {
    if (m->use_f32_matmul) return; // int4 FFN is a net loss for f32_bypass (nibble-unpack overhead > bandwidth savings at B=1)
    int total = 0;
    for (size_t i = 0; i < m->tensors.size(); i++) {
        auto& t = m->tensors[i];
        if (t.ttype != 3) continue;
        if (m->tensor_names.size() > i) {
            const std::string& name = m->tensor_names[i];
            if (name.find("gate") == std::string::npos &&
                name.find("down") == std::string::npos &&
                name.find("up") == std::string::npos) continue;
        } else {
            continue;  // can't identify without names
        }

        // ttype=3 layout: [fp16_scale:2] [i8_weights:rows*cols] [row_sums:rows*4]
        uint16_t s16; memcpy(&s16, t.data, 2);
        float scale = fp16_to_fp32(s16);
        const int8_t* i8 = (const int8_t*)(t.data + 2);
        int rows = t.row_dim;
        
        int64_t ds = t.data_size;
        int cols = (int)((ds - 2 - (int64_t)rows * 4) / rows);
        if (cols <= 0 || cols > 1000000) continue;
        
        // Rescale int8 FFN weights to use full int4 range [-7,7].
        // w_i4 = round(w_i8 * 7/max_abs), stored_scale = original_scale * max_abs/7
        // so dequant: w_i4 * stored_scale = w_i8 * original_scale
        int n_vals = rows * cols;
        int i4_max = 7;
        int max_abs = 0;
        for (int j = 0; j < n_vals; j++) {
            int v = (int)i8[j]; if (v < 0) v = -v;
            if (v > max_abs) max_abs = v;
        }
        float pack_rescale = (max_abs > i4_max) ? (float)i4_max / (float)max_abs : 1.0f;
        float scale_rescale = (max_abs > i4_max) ? (float)max_abs / (float)i4_max : 1.0f;
        uint16_t new_s16 = (max_abs > i4_max)
            ? fp32_to_fp16(scale * scale_rescale)
            : s16;
        
        int packed_cols = (cols + 1) / 2;
        int packed_bytes = align_up4(rows * packed_cols);
        int new_size = 2 + packed_bytes + rows * 4;
        
        uint8_t* new_data = atlas_valloc(new_size);
        if (!new_data) {
            fprintf(stderr, "[ATLAS] OOM quantizing FFN to int4\n");
            continue;
        }
        memcpy(new_data, &new_s16, 2);
        
        uint8_t* packed = new_data + 2;
        int32_t* new_rs = (int32_t*)(packed + packed_bytes);
        
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
        for (int r = 0; r < rows; r++) {
            int sum = 0;
            for (int c = 0; c < cols; c += 2) {
                float f0 = (float)i8[r * cols + c] * pack_rescale;
                float f1 = (c + 1 < cols) ? (float)i8[r * cols + c + 1] * pack_rescale : 0.0f;
                int v0 = (int)(f0 >= 0.0f ? f0 + 0.5f : f0 - 0.5f);
                int v1 = (int)(f1 >= 0.0f ? f1 + 0.5f : f1 - 0.5f);
                
                if (v0 < -8) v0 = -8;
                if (v0 > 7) v0 = 7;
                if (v1 < -8) v1 = -8;
                if (v1 > 7) v1 = 7;
                
                int u0 = v0 & 0x0F;
                int u1 = v1 & 0x0F;
                
                packed[r * packed_cols + c / 2] = (uint8_t)(u0 | (u1 << 4));
                
                sum += v0;
                if (c + 1 < cols) sum += v1;
            }
            new_rs[r] = sum;
        }
        
        if (t.data && !m->is_mapped(t.data)) atlas_vfree(t.data);
        t.data = new_data;
        t.data_size = new_size;
        t.packed_cols = packed_cols;
        t.ttype = 8;
        total++;
    }
    if (total > 0) printf("[ATLAS] Quantized %d FFN tensors to int4 (ttype=8)\n", total);
}

// ─── v1.3.2: Decompress only FFN tensors to int8 (gate/up/down) ────
// v2.4.0: ttype=5 handled by atlas_decompress_ttype5 / fused kernel — skip here
ATLAS_API void atlas_decompress_ffn(AtlasModel* m) {
    int total = 0;
    init_tq1_decode_lut();
    for (size_t i = 0; i < m->tensors.size(); i++) {
        auto& t = m->tensors[i];
        if (t.ttype != 0) continue;
        // Check name: only decompress MLP tensors
        // ttype=5 (block-scaled) handled by f32_bypass path — skip
        if (m->tensor_names.size() > i) {
            const std::string& name = m->tensor_names[i];
            if (name.find("mlp") == std::string::npos &&
                name.find("gate") == std::string::npos &&
                name.find("down") == std::string::npos) continue;
        }

        int input_dim = t.packed_cols * 5;
        int n_vals = t.row_dim * input_dim;
        total++;

        // ─── ttype=0: raw ternary {-1,0,1} decompression ───
        uint8_t* new_data = atlas_valloc(2 + n_vals + t.row_dim * 4);
        if (!new_data) {
            fprintf(stderr, "[ATLAS] OOM decompressing FFN tensor\n");
            total--;
            continue;
        }
        new_data[0] = t.data[0];
        new_data[1] = t.data[1];

        int8_t* i8 = (int8_t*)(new_data + 2);
        int32_t* rs = (int32_t*)(i8 + n_vals);
        const uint8_t* packed = t.data + 2;

        for (int r = 0; r < t.row_dim; r++) {
            int sum = 0;
            int pos = 0;
            for (int c = 0; c < t.packed_cols; c++) {
                const int8_t* l = tq1_decode[packed[r * t.packed_cols + c]];
                i8[r * input_dim + pos++] = l[0]; sum += l[0];
                i8[r * input_dim + pos++] = l[1]; sum += l[1];
                i8[r * input_dim + pos++] = l[2]; sum += l[2];
                i8[r * input_dim + pos++] = l[3]; sum += l[3];
                i8[r * input_dim + pos++] = l[4]; sum += l[4];
            }
            rs[r] = sum;
        }

        if (t.data && !m->is_mapped(t.data)) atlas_vfree(t.data);
        t.data = new_data;
        t.data_size = 2 + n_vals + t.row_dim * 4;
        t.ttype = 3;
    }
    printf("[ATLAS] Decompressed %d FFN tensors to int8\n", total);
}

// ─── Prefetch int8 data into physical RAM ─────────────────────────────
// Touch one byte per 4KB page to force page-in / prevent working-set trim lag.
ATLAS_API void atlas_prefetch_int8(AtlasModel* m) {
    int64_t total = 0;
    int64_t step = 4096 / sizeof(int8_t);
    int n = (int)m->tensors.size();
    #pragma omp parallel for reduction(+:total) schedule(dynamic, 4)
    for (int ti = 0; ti < n; ti++) {
        auto& t = m->tensors[ti];
        if (t.ttype == 3) {
            uintptr_t d8 = (uintptr_t)t.data;
            if (!d8) continue;
            int n_vals = (int)(t.data_size - 2 - t.row_dim * 4);
            if (n_vals <= 0) continue;
            int8_t* data = (int8_t*)(d8 + 2);
            for (int64_t i = 0; i < n_vals; i += step) {
                #ifdef __x86_64__
                _mm_prefetch((const char*)&data[i], _MM_HINT_NTA);
                #else
                __builtin_prefetch(&data[i], 0, 0);
                #endif
            }
            total += n_vals;
        } else if (t.ttype == 11) {
            uintptr_t d8 = (uintptr_t)t.data;
            if (!d8) continue;
            int n_vals = (int)(t.data_size - t.row_dim * 6);
            if (n_vals <= 0) continue;
            int8_t* data = (int8_t*)(d8 + t.row_dim * 2);
            for (int64_t i = 0; i < n_vals; i += step) {
                #ifdef __x86_64__
                _mm_prefetch((const char*)&data[i], _MM_HINT_NTA);
                #else
                __builtin_prefetch(&data[i], 0, 0);
                #endif
            }
            total += n_vals;
        } else if (t.ttype == 10) {
            // Prefetch TQ2 packed data + fp16 scales
            const uint8_t* d = t.data;
            if (!d) continue;
            int64_t sz = t.data_size;
            for (int64_t i = 0; i < sz; i += step) {
                #ifdef __x86_64__
                _mm_prefetch((const char*)&d[i], _MM_HINT_NTA);
                #else
                __builtin_prefetch(&d[i], 0, 0);
                #endif
            }
            total += sz;
        }
    }
    printf("[ATLAS] Prefetched %lld int8/TQ2 values\n", (long long)total);
}

// ─── Get int8-decoded tensor from C++ side ─────────────────────────────
// Returns i8 data pointer or nullptr if not decoded
ATLAS_API const int8_t* atlas_get_int8(AtlasModel* m, int idx, int* rows,
                                        int* input_dim, float* scale,
                                        const int32_t** row_sums) {
    if (idx < 0 || idx >= (int)m->tensors.size()) return nullptr;
    auto& t = m->tensors[idx];
    if (t.ttype != 3) return nullptr;

    uint16_t scale_raw;
    memcpy(&scale_raw, t.data, 2);
    if (scale) *scale = fp16_to_fp32(scale_raw);
    if (rows) *rows = t.row_dim;
    if (input_dim) {
        if (t.ttype == 3)
            *input_dim = (int)((t.data_size - 2 - (int64_t)t.row_dim * 4) / t.row_dim);
        else
            *input_dim = t.packed_cols * 5;
    }
    if (row_sums) {
        int n_vals;
        if (t.ttype == 3)
            n_vals = t.row_dim * (int)((t.data_size - 2 - (int64_t)t.row_dim * 4) / t.row_dim);
        else
            n_vals = t.row_dim * t.packed_cols * 5;
        *row_sums = (const int32_t*)(t.data + 2 + n_vals);
    }
    return (const int8_t*)(t.data + 2);
}

// ─── Get tensor info ────────────────────────────────────────────────────
ATLAS_API void atlas_tensor_info(AtlasModel* m, int idx, int* ttype,
                                  int* row_dim, int* col_dim) {
    if (idx < 0 || idx >= (int)m->tensors.size()) return;
    auto& t = m->tensors[idx];
    if (ttype) *ttype = t.ttype;
    if (row_dim) *row_dim = t.row_dim;
    if (col_dim) *col_dim = (t.ttype == 0 || t.ttype == 5) ? t.packed_cols * 5 : (t.ttype == 7 ? t.packed_cols * 4 : 0);
}

// ─── Access tensor data ─────────────────────────────────────────────────
ATLAS_API const uint8_t* atlas_tensor_data(AtlasModel* m, int idx, int* size) {
    if (idx < 0 || idx >= (int)m->tensors.size()) return nullptr;
    if (size) *size = m->tensors[idx].data_size;
    return m->tensors[idx].data;
}

// ─── Matmul: int8 weights × uint8 activations → float32 output ──────────
// ARM64 (AArch64): NEON vdotq_s32 version in atlas_kernel_arm64.cpp.
#ifndef __aarch64__

// ─── Matmul: int8 weights × int8 activations → float32 output ──────────
// Uses _mm256_maddubs_epi16 with offset trick:
//   act_int8 ∈ [-127, 127],  act_u8 = act_int8 + 128 ∈ [1, 255]
//   w_int8 ∈ {-1, 0, 1}
//   sum(act_i * w_i) = sum((act_u8 - 128) * w_i)
//                    = sum(act_u8 * w_i) - 128 * row_sum
//   where _mm256_maddubs_epi16(act_u8, w_i8) computes sum(act_u8 * w_i)
//
// activations: [n_tokens × input_dim] int8 (IN THE RANGE [-127, 127])
// output:      [n_tokens × rows] float32
// row_sums:    [rows] int32 — precomputed Σ w_i per output row
ATLAS_API void atlas_matmul_i8_f32(int rows, int input_dim,
                                    const int8_t* __restrict__ weights,
                                    const uint8_t* __restrict__ act_u8,
                                    const int32_t* __restrict__ row_sums,
                                    float* __restrict__ output,
                                    int n_tokens) {
    (void)row_sums;
    const int TILE_B = 8;
    const __m256i c80 = _mm256_set1_epi8((char)0x80);
    for (int t0 = 0; t0 < n_tokens; t0 += TILE_B) {
        int t_end = t0 + TILE_B;
        if (t_end > n_tokens) t_end = n_tokens;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64)
        #endif
        for (int r = 0; r < rows; r++) {
            const int8_t* w = weights + r * input_dim;

            for (int t = t0; t < t_end; t++) {
                const uint8_t* a = act_u8 + t * input_dim;

                int c = 0;
                int dot = 0;

                __m256i acc = _mm256_setzero_si256();

                for (; c + 32 <= input_dim; c += 32) {
                    __m256i a8 = _mm256_loadu_si256((const __m256i*)(a + c));
                    __m256i w8 = _mm256_loadu_si256((const __m256i*)(w + c));
                    __m256i ac = _mm256_xor_si256(a8, c80);
                    __m256i a16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(ac));
                    __m256i a16h = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(ac, 1));
                    __m256i w16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w8));
                    __m256i w16h = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w8, 1));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16, w16));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16h, w16h));
                }

                __m128i lo = _mm256_castsi256_si128(acc);
                __m128i hi = _mm256_extracti128_si256(acc, 1);
                __m128i sum128 = _mm_add_epi32(lo, hi);
                sum128 = _mm_hadd_epi32(sum128, sum128);
                sum128 = _mm_hadd_epi32(sum128, sum128);
                dot = _mm_cvtsi128_si32(sum128);

                for (; c < input_dim; c++) {
                    dot += ((int)a[c] - 128) * (int)w[c];
                }

                output[t * rows + r] = (float)dot;
            }
        }
    }
}

#endif // !__aarch64__

// ─── int4×uint8 matmul: nibble unpack + vpmaddubs + sign-extend ──────
// ARM64: provided by atlas_kernel_arm64.cpp (NEON vqtbl1q_u8 + vdotq_s32).
#ifndef __aarch64__
// weights: packed int4 (2 per byte), cols = input_dim (padded to even)
// act_u8: [n_tokens × cols] uint8 (+128 offset) quantized activations
// row_sums: [rows] int32 sum of each sign-extended int4 weight row
// output: [n_tokens × rows] float32 (reorder: token-major)
ATLAS_API void atlas_matmul_i4_f32(int rows, int cols,
                                    const uint8_t* __restrict__ packed_weights,
                                    const uint8_t* __restrict__ act_u8,
                                    const int32_t* __restrict__ row_sums,
                                    float* __restrict__ output,
                                    int n_tokens) {
    (void)row_sums;
    if (g_cpu.matmul_i4_vnni) {
        g_cpu.matmul_i4_vnni(rows, cols, packed_weights, act_u8, row_sums, output, n_tokens);
        return;
    }

    const __m256i c8 = _mm256_set1_epi8(8);
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i c80 = _mm256_set1_epi8((char)0x80);

    const int TILE_B = 8;
    for (int t0 = 0; t0 < n_tokens; t0 += TILE_B) {
        int t_end = t0 + TILE_B;
        if (t_end > n_tokens) t_end = n_tokens;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64)
        #endif
        for (int r = 0; r < rows; r++) {
            const uint8_t* pw = packed_weights + r * cols / 2;

            for (int t = t0; t < t_end; t++) {
                const uint8_t* a = act_u8 + t * cols;

                int c = 0;
                int dot = 0;
                __m256i acc = _mm256_setzero_si256();

                for (; c + 64 <= cols; c += 64) {
                    int pc = c / 2;
                    __m256i packed = _mm256_loadu_si256((const __m256i*)(pw + pc));

                    __m256i low = _mm256_and_si256(packed, mask_0f);
                    __m256i high = _mm256_and_si256(
                        _mm256_srli_epi16(packed, 4), mask_0f);

                    __m256i w_low_s = _mm256_sub_epi8(
                        _mm256_xor_si256(low, c8), c8);
                    __m256i w_high_s = _mm256_sub_epi8(
                        _mm256_xor_si256(high, c8), c8);

                    __m256i w_tmp_lo = _mm256_unpacklo_epi8(w_low_s, w_high_s);
                    __m256i w_tmp_hi = _mm256_unpackhi_epi8(w_low_s, w_high_s);
                    __m256i w_lo = _mm256_permute2f128_si256(w_tmp_lo, w_tmp_hi, 0x20);
                    __m256i w_hi = _mm256_permute2f128_si256(w_tmp_lo, w_tmp_hi, 0x31);

                    // XOR act_u8 with 0x80 centered: (a[c] - 128) as int8 → sign-extend to int16
                    __m256i a0 = _mm256_loadu_si256((const __m256i*)(a + c));
                    __m256i w_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_lo));
                    __m256i w_lo16h = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_lo, 1));
                    __m256i a0c = _mm256_xor_si256(a0, c80);
                    __m256i a0c16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a0c));
                    __m256i a0c16h = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a0c, 1));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a0c16, w_lo16));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a0c16h, w_lo16h));

                    __m256i a1 = _mm256_loadu_si256((const __m256i*)(a + c + 32));
                    __m256i w_hi16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_hi));
                    __m256i w_hi16h = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_hi, 1));
                    __m256i a1c = _mm256_xor_si256(a1, c80);
                    __m256i a1c16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a1c));
                    __m256i a1c16h = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a1c, 1));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a1c16, w_hi16));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a1c16h, w_hi16h));
                }

                __m128i lo = _mm256_castsi256_si128(acc);
                __m128i hi = _mm256_extracti128_si256(acc, 1);
                __m128i sum128 = _mm_add_epi32(lo, hi);
                sum128 = _mm_hadd_epi32(sum128, sum128);
                sum128 = _mm_hadd_epi32(sum128, sum128);
                dot = _mm_cvtsi128_si32(sum128);

                for (; c < cols; c++) {
                    int packed_idx = (r * cols + c) / 2;
                    int nibble = (c & 1)
                        ? (packed_weights[packed_idx] >> 4)
                        : (packed_weights[packed_idx] & 0x0F);
                    int8_t w_val = (int8_t)((nibble ^ 8) - 8);
                    dot += ((int)a[c] - 128) * (int)w_val;
                }

                output[t * rows + r] = (float)dot;
            }
        }
    }
}

#endif // !__aarch64__ (int4 matmul)

// ─── Norm: float16 tensor → RMSNorm ────────────────────────────────────
// Performs: output[i] = x[i] * weight[i] * rms(mean(x^2) + eps)
// Where weight is loaded from atlas tensor (float16)
ATLAS_API void atlas_rmsnorm_f32(const float* x, const uint8_t* weight_f16,
                                  float* output, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) {
        uint16_t w;
        memcpy(&w, weight_f16 + i * 2, 2);
        output[i] = x[i] * rms * fp16_to_fp32(w);
    }
}

// ─── RoPE: apply rotary embeddings ──────────────────────────────────────
// Modifies q and k in-place for a single position
// interleaved=true: pairs (2i, 2i+1) — Falcon3/Qwen3 format
// interleaved=false: pairs (i, i+hd/2) — Llama format
ATLAS_API void atlas_rope_f32(float* q, float* k, int n_heads, int n_kv_heads,
                               int head_dim, int position, float rope_theta,
                               bool interleaved) {
    float theta_base = rope_theta;
    for (int h = 0; h < n_heads; h++) {
        float* qh = q + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta_base, 2.0f * i / head_dim);
            float cos_v = cosf(position * freq);
            float sin_v = sinf(position * freq);
            if (interleaved) {
                float a = qh[2*i], b = qh[2*i+1];
                qh[2*i]   = a * cos_v - b * sin_v;
                qh[2*i+1] = a * sin_v + b * cos_v;
            } else {
                int j = i + head_dim / 2;
                float a = qh[i], b = qh[j];
                qh[i] = a * cos_v - b * sin_v;
                qh[j] = a * sin_v + b * cos_v;
            }
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float* kh = k + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta_base, 2.0f * i / head_dim);
            float cos_v = cosf(position * freq);
            float sin_v = sinf(position * freq);
            if (interleaved) {
                float a = kh[2*i], b = kh[2*i+1];
                kh[2*i]   = a * cos_v - b * sin_v;
                kh[2*i+1] = a * sin_v + b * cos_v;
            } else {
                int j = i + head_dim / 2;
                float a = kh[i], b = kh[j];
                kh[i] = a * cos_v - b * sin_v;
                kh[j] = a * sin_v + b * cos_v;
            }
        }
    }
}

// ─── Fused attention: QK-Norm + RoPE + GQA + softmax + weighted sum ───
// v2.9.2: FP16 KV cache (full precision, no quantization noise)
// q: [B, n_heads * head_dim] float32 — RoPE applied in-place, modified
// k: [B, n_kv_heads * head_dim] float32 — RoPE applied in-place, modified
// v: [B, n_kv_heads * head_dim] float32
// positions: [B] int32
// k_cache: [n_kv_heads, max_seq] uint8_t — int4 K cache (per-block fp16 scale)
// v_cache: [n_kv_heads, max_seq] uint8_t — int4 V cache (per-block fp16 scale)
// output: [B, n_heads * head_dim] float32
// q_norm_w, k_norm_w: [head_dim] uint8 fp16 RMSNorm weights (QK-Norm, Qwen3), NULL = skip
#ifndef __aarch64__
ATLAS_API void atlas_attention_f32(
    float* q, float* k, float* v, const int* positions,
    uint8_t* k_cache, uint8_t* v_cache,
    int max_seq_len, int seq_now, int B,
    int n_heads, int n_kv_heads, int head_dim,
    float rope_theta, float rope_scale, float* output,
    const uint8_t* q_norm_w, const uint8_t* k_norm_w,
    int base_seq_len, bool interleaved_rope) {

    int n_rep = n_heads / n_kv_heads;
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    // v2.5.0: NTK context extension — compound rope_scale with ctx_scale
    float ctx_scale = base_seq_len > 0 ? (float)max_seq_len / (float)base_seq_len : 1.0f;
    if (ctx_scale < 1.0f) ctx_scale = 1.0f;
    float total_scale = rope_scale;
    if (ctx_scale > 1.001f) total_scale *= ctx_scale;
    float eff_theta = rope_theta;
    if (total_scale > 1.001f) {
        eff_theta *= powf(total_scale, (float)head_dim / (float)(head_dim - 2));
    }

    // v2.5.0: Ring buffer — first valid position in cache
    int ring_start = seq_now > max_seq_len ? seq_now - max_seq_len : 0;
    int ring_len = seq_now > max_seq_len ? max_seq_len : seq_now;

    // Attention scores — heap-allocated, reusable buffer (avoids ~192KB stack alloca)
    int max_seq = ring_len;
    static thread_local float* scores_buf = nullptr;
    static thread_local size_t scores_cap = 0;
    size_t needed = (size_t)n_heads * max_seq;
    if (needed > scores_cap) {
        free(scores_buf);
        scores_cap = needed;
        scores_buf = (float*)malloc(scores_cap * sizeof(float));
        if (!scores_buf) { scores_cap = 0; return; }
    }
    float* scores = scores_buf;

    for (int b = 0; b < B; b++) {
        P_START(attn_prep);
        int pos = positions[b];
        float* qb = q + b * n_heads * head_dim;
        float* kb = k + b * n_kv_heads * head_dim;
        float* vb = v + b * n_kv_heads * head_dim;

        // QK-Norm on Q (before RoPE, matching Qwen3 reference order)
        if (q_norm_w) {
            for (int h = 0; h < n_heads; h++) {
                float* qh = qb + h * head_dim;
                float ss = 0.0f;
                for (int d = 0; d < head_dim; d++) ss += qh[d] * qh[d];
                float rms = 1.0f / sqrtf(ss / head_dim + 1e-6f);
                for (int d = 0; d < head_dim; d++) {
                    uint16_t w16; memcpy(&w16, q_norm_w + d * 2, 2);
                    qh[d] *= rms * fp16_to_fp32(w16);
                }
            }
        }
        // QK-Norm on K
        if (k_norm_w) {
            for (int h = 0; h < n_kv_heads; h++) {
                float* kh = kb + h * head_dim;
                float ss = 0.0f;
                for (int d = 0; d < head_dim; d++) ss += kh[d] * kh[d];
                float rms = 1.0f / sqrtf(ss / head_dim + 1e-6f);
                for (int d = 0; d < head_dim; d++) {
                    uint16_t w16; memcpy(&w16, k_norm_w + d * 2, 2);
                    kh[d] *= rms * fp16_to_fp32(w16);
                }
            }
        }

        // RoPE on Q (after QK-Norm, YaRN NTK-aware)
        for (int h = 0; h < n_heads; h++) {
            float* qh = qb + h * head_dim;
            for (int i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(eff_theta, 2.0f * i / head_dim);
                float c = cosf(pos * freq), s = sinf(pos * freq);
                if (interleaved_rope) {
                    float a = qh[2*i], b0 = qh[2*i+1];
                    qh[2*i]   = a * c - b0 * s;
                    qh[2*i+1] = a * s + b0 * c;
                } else {
                    int j = i + head_dim / 2;
                    float a = qh[i], b0 = qh[j];
                    qh[i] = a * c - b0 * s;
                    qh[j] = a * s + b0 * c;
                }
            }
        }
        // RoPE on K (after QK-Norm)
        for (int h = 0; h < n_kv_heads; h++) {
            float* kh = kb + h * head_dim;
            for (int i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(eff_theta, 2.0f * i / head_dim);
                float c = cosf(pos * freq), s = sinf(pos * freq);
                if (interleaved_rope) {
                    float a = kh[2*i], b0 = kh[2*i+1];
                    kh[2*i]   = a * c - b0 * s;
                    kh[2*i+1] = a * s + b0 * c;
                } else {
                    int j = i + head_dim / 2;
                    float a = kh[i], b0 = kh[j];
                    kh[i] = a * c - b0 * s;
                    kh[j] = a * s + b0 * c;
                }
            }
        }

        // Store K, V into cache (fp32 → int4, per-block fp16 scale)
        int n_blk = (head_dim + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
        int pos_bytes = kv_pos_bytes(head_dim);
        for (int h = 0; h < n_kv_heads; h++) {
            float* k_row = kb + h * head_dim;
            float* v_row = vb + h * head_dim;
            int cache_pos = pos % max_seq_len;
            uint8_t* kc = k_cache + (size_t)h * max_seq_len * pos_bytes + (size_t)cache_pos * pos_bytes;
            uint8_t* vc = v_cache + (size_t)h * max_seq_len * pos_bytes + (size_t)cache_pos * pos_bytes;
            for (int blk = 0; blk < n_blk; blk++) {
                int blk_start = blk * KV_BLOCK_SIZE;
                int blk_end = blk_start + KV_BLOCK_SIZE;
                if (blk_end > head_dim) blk_end = head_dim;
                float max_abs_k = 0.0f, max_abs_v = 0.0f;
                for (int d = blk_start; d < blk_end; d++) {
                    float ak = fabsf(k_row[d]); if (ak > max_abs_k) max_abs_k = ak;
                    float av = fabsf(v_row[d]); if (av > max_abs_v) max_abs_v = av;
                }
                float scale_k = (max_abs_k > 1e-10f) ? max_abs_k / 7.0f : 1.0f;
                float scale_v = (max_abs_v > 1e-10f) ? max_abs_v / 7.0f : 1.0f;
                uint16_t sk = fp32_to_fp16(scale_k), sv = fp32_to_fp16(scale_v);
                memcpy(kc + blk * KV_BYTES_PER_BLOCK, &sk, 2);
                memcpy(vc + blk * KV_BYTES_PER_BLOCK, &sv, 2);
                float inv_sk = 1.0f / scale_k, inv_sv = 1.0f / scale_v;
                int ne = blk_end - blk_start;
                for (int i = 0; i < ne; i += 2) {
                    int d = blk_start + i;
                    int vk0 = (int)safe_int_from_float(k_row[d] * inv_sk); if (vk0 < -8) vk0 = -8; if (vk0 > 7) vk0 = 7;
                    int vk1 = (int)safe_int_from_float(k_row[d+1] * inv_sk); if (vk1 < -8) vk1 = -8; if (vk1 > 7) vk1 = 7;
                    kc[blk * KV_BYTES_PER_BLOCK + 2 + i/2] = (vk0 & 0x0F) | ((vk1 & 0x0F) << 4);
                    int vv0 = (int)safe_int_from_float(v_row[d] * inv_sv); if (vv0 < -8) vv0 = -8; if (vv0 > 7) vv0 = 7;
                    int vv1 = (int)safe_int_from_float(v_row[d+1] * inv_sv); if (vv1 < -8) vv1 = -8; if (vv1 > 7) vv1 = 7;
                    vc[blk * KV_BYTES_PER_BLOCK + 2 + i/2] = (vv0 & 0x0F) | ((vv1 & 0x0F) << 4);
                }
            }
        }
        P_ACCUM(attn_prep);

        P_START(attn_scores);

        const int TILE_S = 32;

        {

            __m256 neg_inf = _mm256_set1_ps(-1e9f);
            int total = n_heads * max_seq;
            int i = 0;
            for (; i + 8 <= total; i += 8)
                _mm256_storeu_ps(scores + i, neg_inf);
            for (; i < total; i++)
                scores[i] = -1e9f;
        }

        for (int s0 = 0; s0 < ring_len; s0 += TILE_S) {
            int s1 = s0 + TILE_S;
            if (s1 > ring_len) s1 = ring_len;
            int key_pos_start = ring_start + s0;
            int key_pos_end = ring_start + s1 - 1;

            if (key_pos_start > pos) break;

            bool all_past = (key_pos_end <= pos);

            #pragma omp parallel for
            for (int h = 0; h < n_heads; h++) {
                int kh = h / n_rep;
                const float* qh = qb + h * head_dim;
                float* sh = scores + h * max_seq;

                if (all_past) {
                    for (int s = s0; s < s1; s++) {
                        int cache_idx = (ring_start + s) % max_seq_len;
                        const uint8_t* k_pos = k_cache + (size_t)kh * max_seq_len * pos_bytes + (size_t)cache_idx * pos_bytes;
                        __m256 sum_v = _mm256_setzero_ps();
                        for (int blk = 0; blk < n_blk; blk++) {
                            int blk_start = blk * KV_BLOCK_SIZE;
                            int blk_end = blk_start + KV_BLOCK_SIZE;
                            if (blk_end > head_dim) blk_end = head_dim;
                            TSC_START();
                            uint16_t sr; memcpy(&sr, k_pos + blk * KV_BYTES_PER_BLOCK, 2);
                            float scale = fp16_to_fp32(sr);
                            __m256 scale_v = _mm256_set1_ps(scale);
                            TSC_RESTART(block_cycles);
                            const uint8_t* nb = k_pos + blk * KV_BYTES_PER_BLOCK + 2;
                            int j = blk_start;
                            for (; j + 8 <= blk_end; j += 8) {
                                int nib_off = (j - blk_start) / 2;
                                TSC_RESTART(nibble_cycles);
                                int32_t pw; memcpy(&pw, nb + nib_off, 4);
                                __m128i pack = _mm_cvtsi32_si128(pw);
                                __m128i lo = _mm_and_si128(pack, _mm_set1_epi8(0x0F));
                                __m128i hi = _mm_and_si128(_mm_srli_epi16(pack, 4), _mm_set1_epi8(0x0F));
                                __m128i inter = _mm_unpacklo_epi8(lo, hi);
                                inter = _mm_sub_epi8(_mm_xor_si128(inter, _mm_set1_epi8(8)), _mm_set1_epi8(8));
                                TSC_RESTART(scale_cycles);
                                __m256 wf = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(inter)), scale_v);
                                __m256 qv = _mm256_loadu_ps(qh + j);
                                TSC_RESTART(fma_cycles);
                                sum_v = _mm256_fmadd_ps(qv, wf, sum_v);
                            }
                            TSC_ACCUM(fma_cycles);
                            for (; j < blk_end; j++) {
                                int nib_off = (j - blk_start) / 2;
                                int shft = ((j - blk_start) & 1) ? 4 : 0;
                                int nib = (nb[nib_off] >> shft) & 0x0F;
                                sum_v = _mm256_add_ps(sum_v, _mm256_set1_ps(qh[j] * (float)((nib ^ 8) - 8) * scale));
                            }
                        }
                        float sum = sum_v[0] + sum_v[1] + sum_v[2] + sum_v[3]
                                  + sum_v[4] + sum_v[5] + sum_v[6] + sum_v[7];
                        sh[s] = sum * inv_sqrt_d;
                    }
                } else {
                    for (int s = s0; s < s1; s++) {
                        int attn_pos = ring_start + s;
                        if (attn_pos <= pos) {
                            int cache_idx = (ring_start + s) % max_seq_len;
                            const uint8_t* k_pos = k_cache + (size_t)kh * max_seq_len * pos_bytes + (size_t)cache_idx * pos_bytes;
                            const float* qh = qb + h * head_dim;
                            __m256 sum_v = _mm256_setzero_ps();
                            for (int blk = 0; blk < n_blk; blk++) {
                                int blk_start = blk * KV_BLOCK_SIZE;
                                int blk_end = blk_start + KV_BLOCK_SIZE;
                                if (blk_end > head_dim) blk_end = head_dim;
                                TSC_START();
                                uint16_t sr; memcpy(&sr, k_pos + blk * KV_BYTES_PER_BLOCK, 2);
                                float scale = fp16_to_fp32(sr);
                                __m256 scale_v = _mm256_set1_ps(scale);
                                TSC_RESTART(block_cycles);
                                const uint8_t* nb = k_pos + blk * KV_BYTES_PER_BLOCK + 2;
                                int j = blk_start;
                                for (; j + 8 <= blk_end; j += 8) {
                                    int nib_off = (j - blk_start) / 2;
                                    TSC_RESTART(nibble_cycles);
                                    int32_t pw; memcpy(&pw, nb + nib_off, 4);
                                    __m128i pack = _mm_cvtsi32_si128(pw);
                                    __m128i lo = _mm_and_si128(pack, _mm_set1_epi8(0x0F));
                                    __m128i hi = _mm_and_si128(_mm_srli_epi16(pack, 4), _mm_set1_epi8(0x0F));
                                    __m128i inter = _mm_unpacklo_epi8(lo, hi);
                                    inter = _mm_sub_epi8(_mm_xor_si128(inter, _mm_set1_epi8(8)), _mm_set1_epi8(8));
                                    TSC_RESTART(scale_cycles);
                                    __m256 wf = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(inter)), scale_v);
                                    __m256 qv = _mm256_loadu_ps(qh + j);
                                    TSC_RESTART(fma_cycles);
                                    sum_v = _mm256_fmadd_ps(qv, wf, sum_v);
                                }
                                TSC_ACCUM(fma_cycles);
                                for (; j < blk_end; j++) {
                                    int nib_off = (j - blk_start) / 2;
                                    int shft = ((j - blk_start) & 1) ? 4 : 0;
                                    int nib = (nb[nib_off] >> shft) & 0x0F;
                                    sum_v = _mm256_add_ps(sum_v, _mm256_set1_ps(qh[j] * (float)((nib ^ 8) - 8) * scale));
                                }
                            }
                            float sum = sum_v[0] + sum_v[1] + sum_v[2] + sum_v[3]
                                      + sum_v[4] + sum_v[5] + sum_v[6] + sum_v[7];
                            sh[s] = sum * inv_sqrt_d;
                        }
                    }
                }
            }
        }
        P_ACCUM(attn_scores);

        P_START(attn_softmax);
        #pragma omp parallel for
        for (int h = 0; h < n_heads; h++) {
            float* sh = scores + h * max_seq;
            float max_val = -1e9f;
            for (int s = 0; s < max_seq; s++) {
                float val = sh[s];
                if (val > max_val) max_val = val;
            }
            float sum = 0.0f;
            for (int s = 0; s < max_seq; s++) {
                float e = expf(sh[s] - max_val);
                sh[s] = e;
                sum += e;
            }
            float inv_sum = 1.0f / fmaxf(sum, 1e-10f);
            for (int s = 0; s < max_seq; s++) sh[s] *= inv_sum;
        }
        P_ACCUM(attn_softmax);
        P_START(attn_weighted);
        {
            const int TILE_S = 32;
            for (int s0 = 0; s0 < ring_len; s0 += TILE_S) {
                int s1 = s0 + TILE_S;
                if (s1 > ring_len) s1 = ring_len;
                int key_pos_start = ring_start + s0;
                int key_pos_end = ring_start + s1 - 1;
                if (key_pos_start > pos) break;
                bool all_past = (key_pos_end <= pos);
                #pragma omp parallel for
                for (int h = 0; h < n_heads; h++) {
                    int kh = h / n_rep;
                    float* sh = scores + h * max_seq;
                    float* out_h = output + b * n_heads * head_dim + h * head_dim;
                    if (s0 == 0) {
                        for (int d = 0; d < head_dim; d++) out_h[d] = 0.0f;
                    }
                    if (all_past) {
                        for (int s = s0; s < s1; s++) {
                            float score = sh[s];
                            int cache_idx = (ring_start + s) % max_seq_len;
                            const uint8_t* v_pos = v_cache + (size_t)kh * max_seq_len * pos_bytes + (size_t)cache_idx * pos_bytes;
                            __m256 sv = _mm256_set1_ps(score);
                            for (int blk = 0; blk < n_blk; blk++) {
                                int blk_start = blk * KV_BLOCK_SIZE;
                                int blk_end = blk_start + KV_BLOCK_SIZE;
                                if (blk_end > head_dim) blk_end = head_dim;
                                uint16_t sr; memcpy(&sr, v_pos + blk * KV_BYTES_PER_BLOCK, 2);
                                float scale = fp16_to_fp32(sr);
                                __m256 scale_v = _mm256_set1_ps(scale);
                                const uint8_t* nb = v_pos + blk * KV_BYTES_PER_BLOCK + 2;
                                for (int j = blk_start; j + 8 <= blk_end; j += 8) {
                                    int nib_off = (j - blk_start) / 2;
                                    int32_t pw; memcpy(&pw, nb + nib_off, 4);
                                    __m128i pack = _mm_cvtsi32_si128(pw);
                                    __m128i lo = _mm_and_si128(pack, _mm_set1_epi8(0x0F));
                                    __m128i hi = _mm_and_si128(_mm_srli_epi16(pack, 4), _mm_set1_epi8(0x0F));
                                    __m128i inter = _mm_unpacklo_epi8(lo, hi);
                                    inter = _mm_sub_epi8(_mm_xor_si128(inter, _mm_set1_epi8(8)), _mm_set1_epi8(8));
                                    __m256 wf = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(inter)), scale_v);
                                    __m256 out_v = _mm256_loadu_ps(out_h + j);
                                    out_v = _mm256_fmadd_ps(sv, wf, out_v);
                                    _mm256_storeu_ps(out_h + j, out_v);
                                }
                                for (int j = blk_start + ((blk_end - blk_start) / 8) * 8; j < blk_end; j++) {
                                    int nib_off = (j - blk_start) / 2;
                                    int shft = ((j - blk_start) & 1) ? 4 : 0;
                                    int nib = (nb[nib_off] >> shft) & 0x0F;
                                    float vv = (float)((nib ^ 8) - 8) * scale;
                                    out_h[j] += score * vv;
                                }
                            }
                        }
                    } else {
                        for (int s = s0; s < s1; s++) {
                            int attn_pos = ring_start + s;
                            if (attn_pos > pos) continue;
                            float score = sh[s];
                            int cache_idx = (ring_start + s) % max_seq_len;
                            const uint8_t* v_pos = v_cache + (size_t)kh * max_seq_len * pos_bytes + (size_t)cache_idx * pos_bytes;
                            __m256 sv = _mm256_set1_ps(score);
                            for (int blk = 0; blk < n_blk; blk++) {
                                int blk_start = blk * KV_BLOCK_SIZE;
                                int blk_end = blk_start + KV_BLOCK_SIZE;
                                if (blk_end > head_dim) blk_end = head_dim;
                                uint16_t sr; memcpy(&sr, v_pos + blk * KV_BYTES_PER_BLOCK, 2);
                                float scale = fp16_to_fp32(sr);
                                __m256 scale_v = _mm256_set1_ps(scale);
                                const uint8_t* nb = v_pos + blk * KV_BYTES_PER_BLOCK + 2;
                                for (int j = blk_start; j + 8 <= blk_end; j += 8) {
                                    int nib_off = (j - blk_start) / 2;
                                    int32_t pw; memcpy(&pw, nb + nib_off, 4);
                                    __m128i pack = _mm_cvtsi32_si128(pw);
                                    __m128i lo = _mm_and_si128(pack, _mm_set1_epi8(0x0F));
                                    __m128i hi = _mm_and_si128(_mm_srli_epi16(pack, 4), _mm_set1_epi8(0x0F));
                                    __m128i inter = _mm_unpacklo_epi8(lo, hi);
                                    inter = _mm_sub_epi8(_mm_xor_si128(inter, _mm_set1_epi8(8)), _mm_set1_epi8(8));
                                    __m256 wf = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(inter)), scale_v);
                                    __m256 out_v = _mm256_loadu_ps(out_h + j);
                                    out_v = _mm256_fmadd_ps(sv, wf, out_v);
                                    _mm256_storeu_ps(out_h + j, out_v);
                                }
                                for (int j = blk_start + ((blk_end - blk_start) / 8) * 8; j < blk_end; j++) {
                                    int nib_off = (j - blk_start) / 2;
                                    int shft = ((j - blk_start) & 1) ? 4 : 0;
                                    int nib = (nb[nib_off] >> shft) & 0x0F;
                                    float vv = (float)((nib ^ 8) - 8) * scale;
                                    out_h[j] += score * vv;
                                }
                            }
                        }
                    }
                }
            }
        }
        P_ACCUM(attn_weighted);
    }
}
#else
// ARM64 NEON version of atlas_attention_f32
// Uses NEON intrinsics for int4 V-cache decode + f32 FMA
ATLAS_API void atlas_attention_f32(
    float* q, float* k, float* v, const int* positions,
    uint8_t* k_cache, uint8_t* v_cache,
    int max_seq_len, int seq_now, int B,
    int n_heads, int n_kv_heads, int head_dim,
    float rope_theta, float rope_scale, float* output,
    const uint8_t* q_norm_w, const uint8_t* k_norm_w,
    int base_seq_len, bool interleaved_rope) {

    int n_rep = n_heads / n_kv_heads;
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    float ctx_scale = base_seq_len > 0 ? (float)max_seq_len / (float)base_seq_len : 1.0f;
    if (ctx_scale < 1.0f) ctx_scale = 1.0f;
    float total_scale = rope_scale;
    if (ctx_scale > 1.001f) total_scale *= ctx_scale;
    float eff_theta = rope_theta;
    if (total_scale > 1.001f) {
        eff_theta *= powf(total_scale, (float)head_dim / (float)(head_dim - 2));
    }

    int ring_start = seq_now > max_seq_len ? seq_now - max_seq_len : 0;
    int ring_len = seq_now > max_seq_len ? max_seq_len : seq_now;

    int max_seq = ring_len;
    static thread_local float* scores_buf = nullptr;
    static thread_local size_t scores_cap = 0;
    size_t needed = (size_t)n_heads * max_seq;
    if (needed > scores_cap) {
        free(scores_buf);
        scores_cap = needed;
        scores_buf = (float*)malloc(scores_cap * sizeof(float));
        if (!scores_buf) { scores_cap = 0; return; }
    }
    float* scores = scores_buf;

    for (int b = 0; b < B; b++) {
        P_START(attn_prep);
        int pos = positions[b];
        float* qb = q + b * n_heads * head_dim;
        float* kb = k + b * n_kv_heads * head_dim;
        float* vb = v + b * n_kv_heads * head_dim;

        if (q_norm_w) {
            for (int h = 0; h < n_heads; h++) {
                float* qh = qb + h * head_dim;
                float ss = 0.0f;
                for (int d = 0; d < head_dim; d++) ss += qh[d] * qh[d];
                float rms = 1.0f / sqrtf(ss / head_dim + 1e-6f);
                for (int d = 0; d < head_dim; d++) {
                    uint16_t w16; memcpy(&w16, q_norm_w + d * 2, 2);
                    qh[d] *= rms * fp16_to_fp32(w16);
                }
            }
        }
        if (k_norm_w) {
            for (int h = 0; h < n_kv_heads; h++) {
                float* kh = kb + h * head_dim;
                float ss = 0.0f;
                for (int d = 0; d < head_dim; d++) ss += kh[d] * kh[d];
                float rms = 1.0f / sqrtf(ss / head_dim + 1e-6f);
                for (int d = 0; d < head_dim; d++) {
                    uint16_t w16; memcpy(&w16, k_norm_w + d * 2, 2);
                    kh[d] *= rms * fp16_to_fp32(w16);
                }
            }
        }

        // RoPE on Q
        for (int h = 0; h < n_heads; h++) {
            float* qh = qb + h * head_dim;
            for (int i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(eff_theta, 2.0f * i / head_dim);
                float c = cosf(pos * freq), s = sinf(pos * freq);
                if (interleaved_rope) {
                    float a = qh[2*i], b0 = qh[2*i+1];
                    qh[2*i]   = a * c - b0 * s;
                    qh[2*i+1] = a * s + b0 * c;
                } else {
                    int jj = i + head_dim / 2;
                    float a = qh[i], b0 = qh[jj];
                    qh[i] = a * c - b0 * s;
                    qh[jj] = a * s + b0 * c;
                }
            }
        }
        // RoPE on K
        for (int h = 0; h < n_kv_heads; h++) {
            float* kh = kb + h * head_dim;
            for (int i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(eff_theta, 2.0f * i / head_dim);
                float c = cosf(pos * freq), s = sinf(pos * freq);
                if (interleaved_rope) {
                    float a = kh[2*i], b0 = kh[2*i+1];
                    kh[2*i]   = a * c - b0 * s;
                    kh[2*i+1] = a * s + b0 * c;
                } else {
                    int jj = i + head_dim / 2;
                    float a = kh[i], b0 = kh[jj];
                    kh[i] = a * c - b0 * s;
                    kh[jj] = a * s + b0 * c;
                }
            }
        }

        // Store K, V into int4 cache
        int n_blk = (head_dim + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
        int pos_bytes = kv_pos_bytes(head_dim);
        for (int h = 0; h < n_kv_heads; h++) {
            float* k_row = kb + h * head_dim;
            float* v_row = vb + h * head_dim;
            int cache_pos = pos % max_seq_len;
            uint8_t* kc = k_cache + (size_t)h * max_seq_len * pos_bytes + (size_t)cache_pos * pos_bytes;
            uint8_t* vc = v_cache + (size_t)h * max_seq_len * pos_bytes + (size_t)cache_pos * pos_bytes;
            for (int blk = 0; blk < n_blk; blk++) {
                int blk_start = blk * KV_BLOCK_SIZE;
                int blk_end = blk_start + KV_BLOCK_SIZE;
                if (blk_end > head_dim) blk_end = head_dim;
                float max_abs_k = 0.0f, max_abs_v = 0.0f;
                for (int d = blk_start; d < blk_end; d++) {
                    float ak = fabsf(k_row[d]); if (ak > max_abs_k) max_abs_k = ak;
                    float av = fabsf(v_row[d]); if (av > max_abs_v) max_abs_v = av;
                }
                float scale_k = (max_abs_k > 1e-10f) ? max_abs_k / 7.0f : 1.0f;
                float scale_v = (max_abs_v > 1e-10f) ? max_abs_v / 7.0f : 1.0f;
                uint16_t sk = fp32_to_fp16(scale_k), sv = fp32_to_fp16(scale_v);
                memcpy(kc + blk * KV_BYTES_PER_BLOCK, &sk, 2);
                memcpy(vc + blk * KV_BYTES_PER_BLOCK, &sv, 2);
                float inv_sk = 1.0f / scale_k, inv_sv = 1.0f / scale_v;
                int ne = blk_end - blk_start;
                for (int i = 0; i < ne; i += 2) {
                    int d = blk_start + i;
                    int vk0 = (int)(k_row[d] * inv_sk); if (vk0 < -8) vk0 = -8; if (vk0 > 7) vk0 = 7;
                    int vk1 = (int)(k_row[d+1] * inv_sk); if (vk1 < -8) vk1 = -8; if (vk1 > 7) vk1 = 7;
                    kc[blk * KV_BYTES_PER_BLOCK + 2 + i/2] = (vk0 & 0x0F) | ((vk1 & 0x0F) << 4);
                    int vv0 = (int)(v_row[d] * inv_sv); if (vv0 < -8) vv0 = -8; if (vv0 > 7) vv0 = 7;
                    int vv1 = (int)(v_row[d+1] * inv_sv); if (vv1 < -8) vv1 = -8; if (vv1 > 7) vv1 = 7;
                    vc[blk * KV_BYTES_PER_BLOCK + 2 + i/2] = (vv0 & 0x0F) | ((vv1 & 0x0F) << 4);
                }
            }
        }
        P_ACCUM(attn_prep);

        P_START(attn_scores);
        {
            for (int i = 0; i < n_heads * max_seq; i++)
                scores[i] = -1e9f;
        }

        const int TILE_S = 32;
        float32x4_t zero = vdupq_n_f32(0.0f);

        for (int s0 = 0; s0 < ring_len; s0 += TILE_S) {
            int s1 = s0 + TILE_S;
            if (s1 > ring_len) s1 = ring_len;
            int key_pos_start = ring_start + s0;
            int key_pos_end = ring_start + s1 - 1;

            if (key_pos_start > pos) break;

            bool all_past = (key_pos_end <= pos);

            #pragma omp parallel for
            for (int h = 0; h < n_heads; h++) {
                int kh = h / n_rep;
                const float* qh = qb + h * head_dim;
                float* sh = scores + h * max_seq;

                if (all_past) {
                    for (int s = s0; s < s1; s++) {
                        int cache_idx = (ring_start + s) % max_seq_len;
                        const uint8_t* k_pos = k_cache + (size_t)kh * max_seq_len * pos_bytes + (size_t)cache_idx * pos_bytes;
                        float32x4_t sum_lo = zero, sum_hi = zero;
                        for (int blk = 0; blk < n_blk; blk++) {
                            int blk_start = blk * KV_BLOCK_SIZE;
                            int blk_end = blk_start + KV_BLOCK_SIZE;
                            if (blk_end > head_dim) blk_end = head_dim;
                            uint16_t sr; memcpy(&sr, k_pos + blk * KV_BYTES_PER_BLOCK, 2);
                            float scale = fp16_to_fp32(sr);
                            float32x4_t scale_v = vdupq_n_f32(scale);
                            const uint8_t* nb = k_pos + blk * KV_BYTES_PER_BLOCK + 2;
                            int j = blk_start;
                            for (; j + 8 <= blk_end; j += 8) {
                                int nib_off = (j - blk_start) / 2;
                                int32_t pw; memcpy(&pw, nb + nib_off, 4);
                                uint32x4_t pw_v = vdupq_n_u32(0);
                                pw_v = vsetq_lane_u32((uint32_t)pw, pw_v, 0);
                                uint8x16_t v16 = vreinterpretq_u8_u32(pw_v);
                                uint8x8_t v8 = vget_low_u8(v16);
                                uint8x8_t lo = vand_u8(v8, vdup_n_u8(0x0F));
                                uint8x8_t hi = vshr_n_u8(v8, 4);
                                int8x8_t inter = vzip_s8(vreinterpret_s8_u8(lo), vreinterpret_s8_u8(hi)).val[0];
                                inter = vsub_s8(veor_s8(inter, vdup_n_s8(8)), vdup_n_s8(8));
                                int16x8_t i16 = vmovl_s8(inter);
                                float32x4_t wf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16)));
                                float32x4_t wf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16)));
                                wf_lo = vmulq_f32(wf_lo, scale_v);
                                wf_hi = vmulq_f32(wf_hi, scale_v);
                                float32x4_t q_lo = vld1q_f32(qh + j);
                                float32x4_t q_hi = vld1q_f32(qh + j + 4);
                                sum_lo = vfmaq_f32(sum_lo, q_lo, wf_lo);
                                sum_hi = vfmaq_f32(sum_hi, q_hi, wf_hi);
                            }
                            for (; j < blk_end; j++) {
                                int nib_off = (j - blk_start) / 2;
                                int shft = ((j - blk_start) & 1) ? 4 : 0;
                                int nib = (nb[nib_off] >> shft) & 0x0F;
                                float32x4_t tv = vdupq_n_f32(qh[j] * (float)((nib ^ 8) - 8) * scale);
                                sum_lo = vaddq_f32(sum_lo, tv);
                                sum_hi = vaddq_f32(sum_hi, tv);
                            }
                        }
                        float sum = vaddvq_f32(sum_lo) + vaddvq_f32(sum_hi);
                        sh[s] = sum * inv_sqrt_d;
                    }
                } else {
                    for (int s = s0; s < s1; s++) {
                        int attn_pos = ring_start + s;
                        if (attn_pos <= pos) {
                            int cache_idx = (ring_start + s) % max_seq_len;
                            const uint8_t* k_pos = k_cache + (size_t)kh * max_seq_len * pos_bytes + (size_t)cache_idx * pos_bytes;
                            float32x4_t sum_lo = zero, sum_hi = zero;
                            for (int blk = 0; blk < n_blk; blk++) {
                                int blk_start = blk * KV_BLOCK_SIZE;
                                int blk_end = blk_start + KV_BLOCK_SIZE;
                                if (blk_end > head_dim) blk_end = head_dim;
                                uint16_t sr; memcpy(&sr, k_pos + blk * KV_BYTES_PER_BLOCK, 2);
                                float scale = fp16_to_fp32(sr);
                                float32x4_t scale_v = vdupq_n_f32(scale);
                                const uint8_t* nb = k_pos + blk * KV_BYTES_PER_BLOCK + 2;
                                int j = blk_start;
                                for (; j + 8 <= blk_end; j += 8) {
                                    int nib_off = (j - blk_start) / 2;
                                    int32_t pw; memcpy(&pw, nb + nib_off, 4);
                                    uint32x4_t pw_v = vdupq_n_u32(0);
                                    pw_v = vsetq_lane_u32((uint32_t)pw, pw_v, 0);
                                    uint8x16_t v16 = vreinterpretq_u8_u32(pw_v);
                                    uint8x8_t v8 = vget_low_u8(v16);
                                    uint8x8_t lo = vand_u8(v8, vdup_n_u8(0x0F));
                                    uint8x8_t hi = vshr_n_u8(v8, 4);
                                    int8x8_t inter = vzip_s8(vreinterpret_s8_u8(lo), vreinterpret_s8_u8(hi)).val[0];
                                    inter = vsub_s8(veor_s8(inter, vdup_n_s8(8)), vdup_n_s8(8));
                                    int16x8_t i16 = vmovl_s8(inter);
                                    float32x4_t wf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16)));
                                    float32x4_t wf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16)));
                                    wf_lo = vmulq_f32(wf_lo, scale_v);
                                    wf_hi = vmulq_f32(wf_hi, scale_v);
                                    float32x4_t q_lo = vld1q_f32(qh + j);
                                    float32x4_t q_hi = vld1q_f32(qh + j + 4);
                                    sum_lo = vfmaq_f32(sum_lo, q_lo, wf_lo);
                                    sum_hi = vfmaq_f32(sum_hi, q_hi, wf_hi);
                                }
                                for (; j < blk_end; j++) {
                                    int nib_off = (j - blk_start) / 2;
                                    int shft = ((j - blk_start) & 1) ? 4 : 0;
                                    int nib = (nb[nib_off] >> shft) & 0x0F;
                                    float32x4_t tv = vdupq_n_f32(qh[j] * (float)((nib ^ 8) - 8) * scale);
                                    sum_lo = vaddq_f32(sum_lo, tv);
                                    sum_hi = vaddq_f32(sum_hi, tv);
                                }
                            }
                            float sum = vaddvq_f32(sum_lo) + vaddvq_f32(sum_hi);
                            sh[s] = sum * inv_sqrt_d;
                        }
                    }
                }
            }
        }
        P_ACCUM(attn_scores);

        P_START(attn_softmax);
        #pragma omp parallel for
        for (int h = 0; h < n_heads; h++) {
            float* sh = scores + h * max_seq;
            float max_val = -1e9f;
            for (int s = 0; s < max_seq; s++) {
                float val = sh[s];
                if (val > max_val) max_val = val;
            }
            float sum = 0.0f;
            for (int s = 0; s < max_seq; s++) {
                float e = expf(sh[s] - max_val);
                sh[s] = e;
                sum += e;
            }
            float inv_sum = 1.0f / fmaxf(sum, 1e-10f);
            for (int s = 0; s < max_seq; s++) sh[s] *= inv_sum;
        }
        P_ACCUM(attn_softmax);
        P_START(attn_weighted);
        {
            const int TILE_S = 32;
            for (int s0 = 0; s0 < ring_len; s0 += TILE_S) {
                int s1 = s0 + TILE_S;
                if (s1 > ring_len) s1 = ring_len;
                int key_pos_start = ring_start + s0;
                int key_pos_end = ring_start + s1 - 1;
                if (key_pos_start > pos) break;
                bool all_past = (key_pos_end <= pos);
                #pragma omp parallel for
                for (int h = 0; h < n_heads; h++) {
                    int kh = h / n_rep;
                    float* sh = scores + h * max_seq;
                    float* out_h = output + b * n_heads * head_dim + h * head_dim;
                    if (s0 == 0) {
                        for (int d = 0; d < head_dim; d++) out_h[d] = 0.0f;
                    }
                    if (all_past) {
                        for (int s = s0; s < s1; s++) {
                            float score = sh[s];
                            float32x4_t sv = vdupq_n_f32(score);
                            int cache_idx = (ring_start + s) % max_seq_len;
                            const uint8_t* v_pos = v_cache + (size_t)kh * max_seq_len * pos_bytes + (size_t)cache_idx * pos_bytes;
                            for (int blk = 0; blk < n_blk; blk++) {
                                int blk_start = blk * KV_BLOCK_SIZE;
                                int blk_end = blk_start + KV_BLOCK_SIZE;
                                if (blk_end > head_dim) blk_end = head_dim;
                                uint16_t sr; memcpy(&sr, v_pos + blk * KV_BYTES_PER_BLOCK, 2);
                                float scale = fp16_to_fp32(sr);
                                float32x4_t scale_v = vdupq_n_f32(scale);
                                const uint8_t* nb = v_pos + blk * KV_BYTES_PER_BLOCK + 2;
                                for (int j = blk_start; j + 8 <= blk_end; j += 8) {
                                    int nib_off = (j - blk_start) / 2;
                                    int32_t pw; memcpy(&pw, nb + nib_off, 4);
                                    uint32x4_t pw_v = vdupq_n_u32(0);
                                    pw_v = vsetq_lane_u32((uint32_t)pw, pw_v, 0);
                                    uint8x16_t v16 = vreinterpretq_u8_u32(pw_v);
                                    uint8x8_t v8 = vget_low_u8(v16);
                                    uint8x8_t lo = vand_u8(v8, vdup_n_u8(0x0F));
                                    uint8x8_t hi = vshr_n_u8(v8, 4);
                                    int8x8_t inter = vzip_s8(vreinterpret_s8_u8(lo), vreinterpret_s8_u8(hi)).val[0];
                                    inter = vsub_s8(veor_s8(inter, vdup_n_s8(8)), vdup_n_s8(8));
                                    int16x8_t i16 = vmovl_s8(inter);
                                    float32x4_t wf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16)));
                                    float32x4_t wf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16)));
                                    wf_lo = vmulq_f32(wf_lo, scale_v);
                                    wf_hi = vmulq_f32(wf_hi, scale_v);
                                    float32x4_t out_lo = vld1q_f32(out_h + j);
                                    float32x4_t out_hi = vld1q_f32(out_h + j + 4);
                                    out_lo = vfmaq_f32(out_lo, sv, wf_lo);
                                    out_hi = vfmaq_f32(out_hi, sv, wf_hi);
                                    vst1q_f32(out_h + j, out_lo);
                                    vst1q_f32(out_h + j + 4, out_hi);
                                }
                                for (int j = blk_start + ((blk_end - blk_start) / 8) * 8; j < blk_end; j++) {
                                    int nib_off = (j - blk_start) / 2;
                                    int shft = ((j - blk_start) & 1) ? 4 : 0;
                                    int nib = (nb[nib_off] >> shft) & 0x0F;
                                    float vv = (float)((nib ^ 8) - 8) * scale;
                                    out_h[j] += score * vv;
                                }
                            }
                        }
                    } else {
                        for (int s = s0; s < s1; s++) {
                            int attn_pos = ring_start + s;
                            if (attn_pos > pos) continue;
                            float score = sh[s];
                            float32x4_t sv = vdupq_n_f32(score);
                            int cache_idx = (ring_start + s) % max_seq_len;
                            const uint8_t* v_pos = v_cache + (size_t)kh * max_seq_len * pos_bytes + (size_t)cache_idx * pos_bytes;
                            for (int blk = 0; blk < n_blk; blk++) {
                                int blk_start = blk * KV_BLOCK_SIZE;
                                int blk_end = blk_start + KV_BLOCK_SIZE;
                                if (blk_end > head_dim) blk_end = head_dim;
                                uint16_t sr; memcpy(&sr, v_pos + blk * KV_BYTES_PER_BLOCK, 2);
                                float scale = fp16_to_fp32(sr);
                                float32x4_t scale_v = vdupq_n_f32(scale);
                                const uint8_t* nb = v_pos + blk * KV_BYTES_PER_BLOCK + 2;
                                for (int j = blk_start; j + 8 <= blk_end; j += 8) {
                                    int nib_off = (j - blk_start) / 2;
                                    int32_t pw; memcpy(&pw, nb + nib_off, 4);
                                    uint32x4_t pw_v = vdupq_n_u32(0);
                                    pw_v = vsetq_lane_u32((uint32_t)pw, pw_v, 0);
                                    uint8x16_t v16 = vreinterpretq_u8_u32(pw_v);
                                    uint8x8_t v8 = vget_low_u8(v16);
                                    uint8x8_t lo = vand_u8(v8, vdup_n_u8(0x0F));
                                    uint8x8_t hi = vshr_n_u8(v8, 4);
                                    int8x8_t inter = vzip_s8(vreinterpret_s8_u8(lo), vreinterpret_s8_u8(hi)).val[0];
                                    inter = vsub_s8(veor_s8(inter, vdup_n_s8(8)), vdup_n_s8(8));
                                    int16x8_t i16 = vmovl_s8(inter);
                                    float32x4_t wf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16)));
                                    float32x4_t wf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16)));
                                    wf_lo = vmulq_f32(wf_lo, scale_v);
                                    wf_hi = vmulq_f32(wf_hi, scale_v);
                                    float32x4_t out_lo = vld1q_f32(out_h + j);
                                    float32x4_t out_hi = vld1q_f32(out_h + j + 4);
                                    out_lo = vfmaq_f32(out_lo, sv, wf_lo);
                                    out_hi = vfmaq_f32(out_hi, sv, wf_hi);
                                    vst1q_f32(out_h + j, out_lo);
                                    vst1q_f32(out_h + j + 4, out_hi);
                                }
                                for (int j = blk_start + ((blk_end - blk_start) / 8) * 8; j < blk_end; j++) {
                                    int nib_off = (j - blk_start) / 2;
                                    int shft = ((j - blk_start) & 1) ? 4 : 0;
                                    int nib = (nb[nib_off] >> shft) & 0x0F;
                                    float vv = (float)((nib ^ 8) - 8) * scale;
                                    out_h[j] += score * vv;
                                }
                            }
                        }
                    }
                }
            }
        }
        P_ACCUM(attn_weighted);
    }
}
#endif
// act: [B, D] float32 → act_u8: [B, D] uint8, max_abs: [B] float32
static void quantize_f32_to_u8(const float* act, int B, int D,
                                float* max_abs_out, uint8_t* act_u8_out) {
    for (int t = 0; t < B; t++) {
        float max_val = 1e-5f;
        for (int i = 0; i < D; i++) {
            float v = fabsf(act[t * D + i]);
            if (v > max_val) max_val = v;
        }
        max_abs_out[t] = max_val;
        float inv = 127.0f / max_val;
        for (int i = 0; i < D; i++) {
            int q = (int)safe_int_from_float(act[t * D + i] * inv + 128.5f);
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            act_u8_out[t * D + i] = (uint8_t)q;
        }
    }
}

// ─── Set f32 matmul mode (no activation quantization) ────────────────
ATLAS_API void atlas_set_use_f32_matmul(AtlasModel* m, int val) {
    if (m) m->use_f32_matmul = val ? true : false;
}

// ─── v1.3.0: Set ternary matmul mode (vpsignb, no multiplication) ────
// Uses _mm256_sign_epi8 for pure sign-based ternary dot product.
// Eliminates 128*row_sum correction. Requires weights ∈ {-1, 0, +1}.
ATLAS_API void atlas_set_use_ternary_matmul(AtlasModel* m, int val) {
    if (m) m->use_ternary_matmul = val ? true : false;
}

// ─── v1.3.1: Set packed matmul mode (2-bit packed ternary weights) ────
// Operates directly on 2-bit compressed weights (4 values/byte).
// Requires TQ1→2-bit conversion at load time. 4× less memory bandwidth.
ATLAS_API void atlas_set_use_packed_matmul(AtlasModel* m, int val) {
    if (m) m->use_packed_matmul = val ? true : false;
}

// ─── v1.3.2: Set hybrid matmul mode (FFN int8, QKV packed) ──────────
ATLAS_API void atlas_set_use_hybrid_matmul(AtlasModel* m, int val) {
    if (m) m->use_hybrid_matmul = val ? true : false;
}

// ─── v1.3.1: Set OpenMP thread count ──────────────────────────────────
// n=0 resets to default (all available threads).
ATLAS_API void atlas_set_num_threads(AtlasModel* m, int n) {
    (void)m;
    #ifdef _OPENMP
    omp_set_num_threads(n > 0 ? n : omp_get_max_threads());
    #endif
}

// ─── v2.4.0: Set YaRN NTK RoPE scaling factor ────────────────────────
ATLAS_API void atlas_set_rope_scale(AtlasModel* m, float scale) {
    if (m) m->rope_scale = scale;
}

// ─── v2.9.2: Set RoPE format ──────────────────────────────────────────
// interleaved: 1=interleaved pairs (2i,2i+1), 0=half-split pairs (i,i+hd/2)
// Auto-detected from model architecture during load, but user can override.
ATLAS_API void atlas_set_rope_interleaved(AtlasModel* m, int enable) {
    if (m) m->rope_interleaved = (enable != 0);
}

ATLAS_API void atlas_set_rope_theta(AtlasModel* m, float theta) {
    if (m) m->rope_theta = theta;
}

// ─── v2.4.0: Set layer stride (9 Falcon3, 11 Qwen3 with QK-Norm) ────
ATLAS_API void atlas_set_layer_stride(AtlasModel* m, int stride) {
    if (m) m->layer_stride = stride;
}

// Forward declaration (defined after atlas_generate)
static void ensure_layer_idx(AtlasModel* m);

// ─── v2.7.6: Ensure layer index cache + model arch detection ───
ATLAS_API void atlas_ensure_layer_idx(AtlasModel* m) {
    if (m) ensure_layer_idx(m);
}

// ─── v2.5.0: Set base sequence length (trained context for NTK extension) ──
ATLAS_API void atlas_set_base_seq_len(AtlasModel* m, int seq_len) {
    if (m && seq_len > 0) m->base_seq_len = seq_len;
}

// v2.11.0: Set MiniCPM scale_depth factor (scale_depth/sqrt(n_layers)) for residual scaling.
// Default 1.0f = standard transformer (no-op).
ATLAS_API void atlas_set_scale_depth_factor(void* model, float factor) {
    AtlasModel* m = (AtlasModel*)model;
    if (m && factor > 0.0f) m->scale_depth_factor = factor;
}

// v2.12.0: Reset int4 KV cache — zeros all cache data without freeing allocation.
ATLAS_API void atlas_reset_cache(void* model) {
    AtlasModel* m = (AtlasModel*)model;
    if (!m || !m->k_cache) return;
    size_t cache_bytes = (size_t)m->n_layers * m->n_kv_heads * m->cache_max_seq_len * kv_pos_bytes(m->head_dim);
    memset(m->k_cache, 0, cache_bytes);
    memset(m->v_cache, 0, cache_bytes);
}

// ─── Helper: horizontal sum of __m256 float ──────────────────────────
#ifndef __aarch64__
static inline float hsum_ps(__m256 v) {
    __m128 l = _mm256_castps256_ps128(v);
    __m128 h = _mm256_extractf128_ps(v, 1);
    l = _mm_add_ps(l, h);
    l = _mm_hadd_ps(l, l);
    l = _mm_hadd_ps(l, l);
    return _mm_cvtss_f32(l);
}
#endif

// ─── v2.7.0: TurboQuant fused matmul kernel (2-bit packed, K_tile=32) ───
// Fused 2-bit unpack (SSE4.1) + f32×f32 FMA (AVX2) with g128 block-scaling.
// No intermediate int8 buffer — decode on-the-fly in registers.
// ttype=7 tensor_data layout: [block_size:1][n_blocks:2][scales:N_blocks*2][packed:rows*packed_cols]
#ifndef __aarch64__
static void matmul_turboquant_fused(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const float* activations, float* output, int B) {
    const uint8_t* packed = tensor_data + 3 + rows * n_blocks * 2;
    const __m128i m3 = _mm_set1_epi16(3);
    const __m128i o1 = _mm_set1_epi16(1);
    const __m128i z = _mm_setzero_si128();

    for (int r = 0; r < rows; r++) {
        const uint8_t* row_scales = tensor_data + 3 + r * n_blocks * 2;
        const uint8_t* row_packed = packed + r * packed_cols;
        for (int b = 0; b < B; b++) {
            const float* act = activations + b * input_dim;
            __m256 row_acc = _mm256_setzero_ps();
            for (int blk = 0; blk < n_blocks; blk++) {
                __m256 vacc = _mm256_setzero_ps();
                int blk_start = blk * block_size;
                int blk_bytes = blk_start / 4;
                for (int off = 0; off < block_size; off += 64) {
                    const uint8_t* bp = row_packed + blk_bytes + off / 4;
                    const float* ap = act + blk_start + off;
                    __m128i p = _mm_loadu_si128((const __m128i*)bp);
                    __m128i pel = _mm_unpacklo_epi8(p, z);
                    __m128i peh = _mm_unpackhi_epi8(p, z);
                    __m128i s0el = _mm_and_si128(pel, m3);
                    __m128i s0eh = _mm_and_si128(peh, m3);
                    __m128i s1el = _mm_and_si128(_mm_srli_epi16(pel, 2), m3);
                    __m128i s1eh = _mm_and_si128(_mm_srli_epi16(peh, 2), m3);
                    __m128i s2el = _mm_and_si128(_mm_srli_epi16(pel, 4), m3);
                    __m128i s2eh = _mm_and_si128(_mm_srli_epi16(peh, 4), m3);
                    __m128i s3el = _mm_and_si128(_mm_srli_epi16(pel, 6), m3);
                    __m128i s3eh = _mm_and_si128(_mm_srli_epi16(peh, 6), m3);
                    s0el = _mm_sub_epi16(s0el, o1); s0eh = _mm_sub_epi16(s0eh, o1);
                    s1el = _mm_sub_epi16(s1el, o1); s1eh = _mm_sub_epi16(s1eh, o1);
                    s2el = _mm_sub_epi16(s2el, o1); s2eh = _mm_sub_epi16(s2eh, o1);
                    s3el = _mm_sub_epi16(s3el, o1); s3eh = _mm_sub_epi16(s3eh, o1);
                    __m128i p0 = _mm_packs_epi16(s0el, s0eh);
                    __m128i p1 = _mm_packs_epi16(s1el, s1eh);
                    __m128i p2 = _mm_packs_epi16(s2el, s2eh);
                    __m128i p3 = _mm_packs_epi16(s3el, s3eh);
                    __m128i abl = _mm_unpacklo_epi8(p0, p1);
                    __m128i cdl = _mm_unpacklo_epi8(p2, p3);
                    __m128i t0 = _mm_unpacklo_epi16(abl, cdl);
                    __m128i t1 = _mm_unpackhi_epi16(abl, cdl);
                    __m128i abh = _mm_unpackhi_epi8(p0, p1);
                    __m128i cdh = _mm_unpackhi_epi8(p2, p3);
                    __m128i t2 = _mm_unpacklo_epi16(abh, cdh);
                    __m128i t3 = _mm_unpackhi_epi16(abh, cdh);
                    #define TQ7_FMA(t_, ap_, base_, acc_) do { \
                        __m128i t_hi = _mm_unpackhi_epi64(t_, t_); \
                        __m256i i32_l = _mm256_cvtepi8_epi32(t_); \
                        __m256i i32_h = _mm256_cvtepi8_epi32(t_hi); \
                        __m256 f_l = _mm256_cvtepi32_ps(i32_l); \
                        __m256 f_h = _mm256_cvtepi32_ps(i32_h); \
                        __m256 a_l = _mm256_loadu_ps(ap_ + (base_)); \
                        __m256 a_h = _mm256_loadu_ps(ap_ + (base_) + 8); \
                        acc_ = _mm256_fmadd_ps(f_l, a_l, acc_); \
                        acc_ = _mm256_fmadd_ps(f_h, a_h, acc_); \
                    } while(0)
                    TQ7_FMA(t0, ap, 0, vacc);
                    TQ7_FMA(t1, ap, 16, vacc);
                    TQ7_FMA(t2, ap, 32, vacc);
                    TQ7_FMA(t3, ap, 48, vacc);
                    #undef TQ7_FMA
                }
                uint16_t ws16; memcpy(&ws16, row_scales + blk * 2, 2);
                __m256 v_scale = _mm256_set1_ps(fp16_to_fp32(ws16));
                row_acc = _mm256_fmadd_ps(vacc, v_scale, row_acc);
            }
            output[b * rows + r] = hsum_ps(row_acc);
        }
    }
}
#else
static void matmul_turboquant_fused(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const float* activations, float* output, int B) {
    (void)packed_cols;
    const uint8_t* packed = tensor_data + 3 + rows * n_blocks * 2;
    for (int r = 0; r < rows; r++) {
        const uint8_t* row_scales = tensor_data + 3 + r * n_blocks * 2;
        const uint8_t* row_packed = packed + r * packed_cols;
        for (int b = 0; b < B; b++) {
            const float* act = activations + b * input_dim;
            float row_acc = 0.0f;
            for (int blk = 0; blk < n_blocks; blk++) {
                float vacc = 0.0f;
                int blk_start = blk * block_size;
                int blk_bytes = blk_start / 4;
                for (int off = 0; off < block_size; off += 4) {
                    const uint8_t* bp = row_packed + blk_bytes + off / 4;
                    uint8_t byte = bp[0];
                    for (int i = 0; i < 4; i++) {
                        int val = (int)((byte >> (2 * i)) & 3) - 1;
                        vacc += act[blk_start + off + i] * (float)val;
                    }
                }
                uint16_t ws16; memcpy(&ws16, row_scales + blk * 2, 2);
                row_acc += vacc * fp16_to_fp32(ws16);
            }
            output[b * rows + r] = row_acc;
        }
    }
}
#endif

// ─── v1.3.0: Ternary-add matmul + reorder (vpsignb, no multiplication) ─
// act_u8: [B, input_dim] uint8 quantized activations [1, 255]
// weights: [rows, input_dim] int8 ternary weights {-1, 0, +1}
// max_abs: [B] per-token max abs (for dequant)
// scale: per-tensor dequant scale
// output: [B, rows] reordered float
// Uses _mm256_sign_epi8 → pure sign/flip/zero, no i8×i8 multiply.
static void matmul_ternary_add_reorder(int rows, int input_dim,
    const int8_t* weights, const uint8_t* act_u8,
    const float* max_abs, float scale, float* output, int B) {
    #ifdef __aarch64__
    atlas_matmul_ternary_f32_arm64(rows, input_dim, weights,
        act_u8, max_abs, scale, output, B);
    #else
    int rows_packed = rows / 4;
    #ifdef _OPENMP
    #pragma omp parallel for if(rows_packed > 4)
    #endif
    for (int ur = 0; ur < rows_packed; ur++) {
        for (int b = 0; b < B; b++) {
            const uint8_t* a = act_u8 + b * input_dim;
            float out4[4];
            for (int sub = 0; sub < 4; sub++) {
                const int8_t* w = weights + (ur * 4 + sub) * input_dim;
                __m256i acc = _mm256_setzero_si256();
                int c = 0;
                for (; c + 32 <= input_dim; c += 32) {
                    __m256i au = _mm256_loadu_si256((const __m256i*)(a + c));
                    __m256i wv = _mm256_loadu_si256((const __m256i*)(w + c));
                    __m256i ai = _mm256_sub_epi8(au, _mm256_set1_epi8(-128));
                    __m256i prod = _mm256_sign_epi8(ai, wv);
                    __m128i lo = _mm256_castsi256_si128(prod);
                    __m128i hi = _mm256_extracti128_si256(prod, 1);
                    __m256i s16_lo = _mm256_cvtepi8_epi16(lo);
                    __m256i s16_hi = _mm256_cvtepi8_epi16(hi);
                    __m256i s32_lo = _mm256_madd_epi16(s16_lo, _mm256_set1_epi16(1));
                    __m256i s32_hi = _mm256_madd_epi16(s16_hi, _mm256_set1_epi16(1));
                    acc = _mm256_add_epi32(acc, s32_lo);
                    acc = _mm256_add_epi32(acc, s32_hi);
                }
                __m128i lo2 = _mm256_castsi256_si128(acc);
                __m128i hi2 = _mm256_extracti128_si256(acc, 1);
                __m128i s = _mm_add_epi32(lo2, hi2);
                s = _mm_hadd_epi32(s, s);
                s = _mm_hadd_epi32(s, s);
                int dot = _mm_cvtsi128_si32(s);
                for (; c < input_dim; c++) {
                    dot += ((int)a[c] - 128) * (int)w[c];
                }
                out4[sub] = (float)dot * max_abs[b] / (127.0f * scale);
            }
            float* dst = output + b * rows;
            dst[ur * 4 + 0] = out4[0];
            dst[ur * 4 + 1] = out4[1];
            dst[ur * 4 + 2] = out4[2];
            dst[ur * 4 + 3] = out4[3];
        }
    }
    #endif
}

// ─── Fused TQ1 decode + int32 dot product for packed weights ──────────
// Decodes TQ1 bytes to int8 trits on-the-fly and computes the activation dot
// product in one pass. No intermediate decode buffer needed.
// Uses the +128 offset trick:
//   sum(act_u8[i] * w[i]) → dot = sum(act_u8[i] * lut[t]) - 128 * sum_w
//   result = dot / (127.0 * scale)  -- dequant scale applied by caller
#ifndef __aarch64__
static void matmul_tq1_packed_reorder(int rows, int input_dim,
    const uint8_t* packed, int packed_cols,
    const uint8_t* act_u8, const float* max_abs,
    float scale, float* output, int B) {
    init_tq1_decode_lut();
    int rows_packed = rows / 4;
    #ifdef ATLAS_DEBUG_MODE
    static int tq1_call_count = 0;
    #endif

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        int8_t* decode_buf = (int8_t*)malloc(4 * input_dim * sizeof(int8_t));
        int32_t row_sums[4];

        #ifdef _OPENMP
        #pragma omp for
        #endif
        for (int ur = 0; ur < rows_packed; ur++) {
            for (int sub = 0; sub < 4; sub++) {
                const uint8_t* w = packed + (ur * 4 + sub) * packed_cols;
                int8_t* row = decode_buf + sub * input_dim;
                int32_t sum_w = 0;
                for (int c = 0; c < packed_cols; c++) {
                    const int8_t* l = tq1_decode[w[c]];
                    row[c * 5 + 0] = l[0]; sum_w += l[0];
                    row[c * 5 + 1] = l[1]; sum_w += l[1];
                    row[c * 5 + 2] = l[2]; sum_w += l[2];
                    row[c * 5 + 3] = l[3]; sum_w += l[3];
                    row[c * 5 + 4] = l[4]; sum_w += l[4];
                }
                row_sums[sub] = sum_w;
            }

            for (int b = 0; b < B; b++) {
                const uint8_t* act = act_u8 + b * input_dim;
                float deq = max_abs[b] / (127.0f * scale);
                float out4[4];

                for (int sub = 0; sub < 4; sub++) {
                    const int8_t* row = decode_buf + sub * input_dim;
                    __m256i sum = _mm256_setzero_si256();
                    int j = 0;
                    for (; j + 32 <= input_dim; j += 32) {
                        __m256i wv = _mm256_loadu_si256((const __m256i*)(row + j));
                        __m256i av = _mm256_loadu_si256((const __m256i*)(act + j));
                        __m256i m = _mm256_maddubs_epi16(av, wv);
                        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(m, _mm256_set1_epi16(1)));
                    }
                    int32_t dot = 0;
                    __m128i l = _mm256_castsi256_si128(sum);
                    __m128i h = _mm256_extracti128_si256(sum, 1);
                    l = _mm_add_epi32(l, h);
                    l = _mm_hadd_epi32(l, l);
                    l = _mm_hadd_epi32(l, l);
                    dot = _mm_cvtsi128_si32(l);
                    for (; j < input_dim; j++) dot += (int32_t)act[j] * (int32_t)row[j];
                    out4[sub] = (float)(dot - 128 * row_sums[sub]) * deq;
                }

                #ifdef ATLAS_DEBUG_MODE
                if (tq1_call_count <= 12 && ur == 0 && b == 0) {
                    printf("[TQ1_DBG#%d] rows=%d dim=%d pcols=%d scale=%f max_abs=%f deq=%g\n",
                           tq1_call_count, rows, input_dim, packed_cols, scale, max_abs[b], deq);
                    printf("[TQ1_DBG#%d]   row_sums=[%d %d %d %d] out4=[%.4f %.4f %.4f %.4f]\n",
                           tq1_call_count,
                           row_sums[0], row_sums[1], row_sums[2], row_sums[3],
                           out4[0], out4[1], out4[2], out4[3]);
                }
                #endif

                float* dst = output + b * rows;
                dst[0 * rows_packed + ur] = out4[0];
                dst[1 * rows_packed + ur] = out4[1];
                dst[2 * rows_packed + ur] = out4[2];
                dst[3 * rows_packed + ur] = out4[3];
            }
        }
        free(decode_buf);
    }
}
#else
#ifdef PROFILE_MODE
#include "atlas_timer.h"
static struct {
    uint64_t tq1_decode;
    uint64_t tq1_dot;
} g_prof_tq1;
#define ARM64_TQ1_START() do { _tq1_tsc[0] = atlas_cycles(); } while(0)
#define ARM64_TQ1_ACCUM_DECODE() __sync_fetch_and_add(&g_prof_tq1.tq1_decode, atlas_cycles() - _tq1_tsc[0])
#define ARM64_TQ1_ACCUM_DOT()    __sync_fetch_and_add(&g_prof_tq1.tq1_dot,    atlas_cycles() - _tq1_tsc[0])
static uint64_t _tq1_tsc[1];

void profile_print_tq1() {
    uint64_t tot = g_prof_tq1.tq1_decode + g_prof_tq1.tq1_dot;
    fprintf(stderr, "[PROFILE] profile_print_tq1 called, tot=%llu\n", (unsigned long long)tot);
    if (tot == 0) return;
    auto pc = [tot](uint64_t v) { return 100.0 * (double)v / (double)tot; };
    fprintf(stderr, "\n── ARM64 TQ1 micro (%lluK cycles) ──\n", (unsigned long long)(tot / 1000));
    fprintf(stderr, "  tq1 decode    %9llu (%5.1f%%)\n", (unsigned long long)g_prof_tq1.tq1_decode, pc(g_prof_tq1.tq1_decode));
    fprintf(stderr, "  tq1 dot+sumw  %9llu (%5.1f%%)\n", (unsigned long long)g_prof_tq1.tq1_dot, pc(g_prof_tq1.tq1_dot));
    fprintf(stderr, "────────────────────────────────\n");
}
#else
#define ARM64_TQ1_START()
#define ARM64_TQ1_ACCUM_DECODE()
#define ARM64_TQ1_ACCUM_DOT()
void profile_print_tq1() {}
#endif

static void matmul_tq1_packed_reorder(int rows, int input_dim,
    const uint8_t* packed, int packed_cols,
    const uint8_t* act_u8, const float* max_abs,
    float scale, float* output, int B) {
    (void)packed_cols;
    init_tq1_decode_lut();
    for (int b = 0; b < B; b++) {
        const uint8_t* a = act_u8 + b * input_dim;
        float deq = max_abs[b] / 127.0f;
        for (int r = 0; r < rows; r++) {
            const uint8_t* wp = packed + r * packed_cols;
            int dot = 0;
            int sum_w = 0;
            for (int c = 0; c < input_dim; c += 5) {
                ARM64_TQ1_START();
                uint8_t byte = wp[c / 5];
                int w_vals[5];
                for (int i = 0; i < 5 && c + i < input_dim; i++) {
                    w_vals[i] = (int)tq1_decode[byte][i];
                }
                ARM64_TQ1_ACCUM_DECODE();
                ARM64_TQ1_START();
                for (int i = 0; i < 5 && c + i < input_dim; i++) {
                    dot += (int)a[c + i] * w_vals[i];
                    sum_w += w_vals[i];
                }
                ARM64_TQ1_ACCUM_DOT();
            }
            dot -= 128 * sum_w;
            output[b * rows + r] = (float)dot * deq / (127.0f * scale);
        }
    }
}
#endif

// ─── Gate activation: SiLU (default) vs ReLU² (BitNet) ────────────
static inline float gate_activation(float g, bool use_relu2) {
    if (use_relu2) {
        float r = g > 0.0f ? g : 0.0f;
        return r * r;
    }
    return g / (1.0f + expf(-g));
}

static inline void apply_sub_norm(float* buf, int n, const uint8_t* w_data) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += buf[i] * buf[i];
    float rms = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) {
        uint16_t w16; memcpy(&w16, w_data + i * 2, 2);
        buf[i] *= rms * fp16_to_fp32(w16);
    }
}

// ─── LUT Prefill Threshold ─────────────────────────────────────────
// When B >= LUT_THRESHOLD, use LUT-based prefill instead of
// decompress+SIMD matmul for ttype=5 tensors.
#ifndef LUT_THRESHOLD
#define LUT_THRESHOLD 16
#endif

// ─── Scalar LUT Builder: 243 entries from 5 int8 activations ──────
// table: 243 int16 entries, one per TQ1 weight pattern (base-3 encoding).
//   table[idx] = sum(act[k] * w_k) for w_k ∈ {-1,0,1}
//   idx = (w_0+1)*1 + (w_1+1)*3 + (w_2+1)*9 + (w_3+1)*27 + (w_4+1)*81
// Cost: 1215 integer multiply-adds per call (243 entries × 5 terms).
static void make_tq1_lut_scalar(int16_t* table, const int8_t act5[5]) {
    for (int idx = 0; idx < 243; idx++) {
        int v = idx;
        int16_t dot = 0;
        dot += (int16_t)act5[0] * (int16_t)((v % 3) - 1); v /= 3;
        dot += (int16_t)act5[1] * (int16_t)((v % 3) - 1); v /= 3;
        dot += (int16_t)act5[2] * (int16_t)((v % 3) - 1); v /= 3;
        dot += (int16_t)act5[3] * (int16_t)((v % 3) - 1); v /= 3;
        dot += (int16_t)act5[4] * (int16_t)((v % 3) - 1);
        table[idx] = dot;
    }
}

// ─── TQ1 LUT Prefill Kernel ───────────────────────────────────────
// LUT-based matmul for ttype=5 tensors with B >= LUT_THRESHOLD.
// Replaces matmul_tq1_block_fused_s8 for prefill: for each (token, quint),
// builds a 243-entry LUT from 5 int8 activation values, then looks up
// each weight byte instead of decompressing and doing sign_epi8.
//
// tensor_data: [block_size:1][n_blocks:2][scales:rows*n_blocks*2][packed:rows*packed_cols]
// act_f32: [B, input_dim] float activations
// output: [B, rows] float output
static void tq1_lut_prefill_kernel(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const float* act_f32, float* output, int B) {

    #ifdef ATLAS_DEBUG_MODE
    printf("[LUT] tq1_lut_prefill_kernel rows=%d dim=%d pc=%d bs=%d nb=%d B=%d\n",
           rows, input_dim, packed_cols, block_size, n_blocks, B);
    #endif

    const uint8_t* scale_data = tensor_data + 3;
    const uint8_t* packed = tensor_data + 3 + (size_t)rows * n_blocks * 2;

    // Pre-decode fp16→fp32 per-row per-block scales (shared across tokens)
    float* block_scales = (float*)malloc((size_t)rows * n_blocks * sizeof(float));
    for (int i = 0; i < rows * n_blocks; i++) {
        uint16_t sr; memcpy(&sr, scale_data + i * 2, 2);
        block_scales[i] = fp16_to_fp32(sr);
    }

    int nquints_total = (input_dim + 4) / 5;

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int b = 0; b < B; b++) {
        // Quantize activations to symmetric int8 (no +128 offset)
        const float* act = act_f32 + (size_t)b * input_dim;
        float max_val = 1e-5f;
        for (int i = 0; i < input_dim; i++) {
            float av = fabsf(act[i]);
            if (av > max_val) max_val = av;
        }
        float scale_x = max_val / 127.0f;
        float inv = (max_val > 1e-10f) ? 127.0f / max_val : 0.0f;

        int8_t* aq = (int8_t*)alloca(input_dim * sizeof(int8_t));
        for (int i = 0; i < input_dim; i++) {
            int q = (int)safe_int_from_float(act[i] * inv + 0.5f);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            aq[i] = (int8_t)q;
        }

        int16_t* lut = (int16_t*)alloca(243 * sizeof(int16_t));
        float* out = output + (size_t)b * rows;
        memset(out, 0, (size_t)rows * sizeof(float));

        // Process each quint group: build LUT from 5 activations,
        // look up weight byte across all rows, accumulate with block scale.
        // For boundary quints (spanning 2 blocks), use first block's scale
        // — slight approximation, affects ≤2 quints per block boundary.
        for (int q = 0; q < nquints_total; q++) {
            int d0 = q * 5;
            int nvals = input_dim - d0;
            if (nvals > 5) nvals = 5;

            int8_t act5[5] = {0};
            for (int k = 0; k < nvals; k++) {
                act5[k] = aq[d0 + k];
            }
            make_tq1_lut_scalar(lut, act5);

            int blk = d0 / block_size;

            for (int r = 0; r < rows; r++) {
                uint8_t byte = packed[(size_t)r * packed_cols + q];
                float blk_scale = block_scales[(size_t)r * n_blocks + blk];
                out[r] += (float)lut[byte] * blk_scale;
            }
        }

        // Apply per-token dequant
        for (int r = 0; r < rows; r++) {
            out[r] *= scale_x;
        }
    }

    free(block_scales);
}

// ─── Block-scaled TQ1 matmul (ttype=5, Bonsai g128 per-row format) ──
// Per-row per-block fp16 scales decoded from tensor_data header.
// Each output row has its own set of n_blocks fp16 scales.
// tensor_data layout: [block_size:1][n_blocks:2][scales: rows*n_blocks*2][packed_TQ1]
#ifndef __aarch64__
static void matmul_tq1_block_reorder(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const uint8_t* act_u8, const float* max_abs,
    float* output, int B,
    const float* act_f32_debug = nullptr) {
    init_tq1_decode_lut();
    int rows_packed = rows / 4;

    const uint8_t* scale_data = tensor_data + 3;
    const uint8_t* packed = tensor_data + 3 + rows * n_blocks * 2;

    // Pre-decode all per-row per-block scales to float
    float* block_scales = (float*)malloc(rows * n_blocks * sizeof(float));
    for (int i = 0; i < rows * n_blocks; i++) {
        uint16_t sr; memcpy(&sr, scale_data + i * 2, 2);
        block_scales[i] = fp16_to_fp32(sr);
    }

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        int8_t* decode_buf = (int8_t*)calloc(4 * input_dim, sizeof(int8_t));
        int32_t* blk_sums = (int32_t*)calloc(n_blocks * 4, sizeof(int32_t));

        #ifdef _OPENMP
        #pragma omp for
        #endif
        for (int ur = 0; ur < rows_packed; ur++) {
            memset(blk_sums, 0, n_blocks * 4 * sizeof(int32_t));
            for (int sub = 0; sub < 4; sub++) {
                const uint8_t* w = packed + (ur * 4 + sub) * packed_cols;
                int8_t* row = decode_buf + sub * input_dim;
                for (int c = 0; c < packed_cols; c++) {
                    if (w[c] == 121) {
                        int col_base = c * 5;
                        for (int t = 0; t < 5; t++) {
                            int col = col_base + t;
                            if (col >= input_dim) break;
                            row[col] = 0;
                        }
                        continue;
                    }
                    const int8_t* l = tq1_decode[w[c]];
                    for (int t = 0; t < 5; t++) {
                        int col = c * 5 + t;
                        if (col >= input_dim) break;
                        int8_t val = l[t];
                        int blk = col / block_size;
                        row[col] = val;
                        blk_sums[blk * 4 + sub] += val;
                    }
                }
            }

            for (int b = 0; b < B; b++) {
                const uint8_t* act = act_u8 + b * input_dim;
                float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                for (int sub = 0; sub < 4; sub++) {
                    int shuffled_r = ur * 4 + sub;  // packed row index (shuffled order)
                    const float* row_scales = block_scales + shuffled_r * n_blocks;
                    const int8_t* row = decode_buf + sub * input_dim;

                    for (int blk = 0; blk < n_blocks; blk++) {
                        int blk_start = blk * block_size;
                        int blk_end = blk_start + block_size;
                        if (blk_end > input_dim) blk_end = input_dim;

                        __m256i sum = _mm256_setzero_si256();
                        int j = blk_start;
                        for (; j + 32 <= blk_end; j += 32) {
                            __m256i wv = _mm256_loadu_si256((const __m256i*)(row + j));
                            __m256i av = _mm256_loadu_si256((const __m256i*)(act + j));
                            __m256i m = _mm256_maddubs_epi16(av, wv);
                            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(m, _mm256_set1_epi16(1)));
                        }
                        int32_t dot = 0;
                        __m128i l = _mm256_castsi256_si128(sum);
                        __m128i h = _mm256_extracti128_si256(sum, 1);
                        l = _mm_add_epi32(l, h);
                        l = _mm_hadd_epi32(l, l);
                        l = _mm_hadd_epi32(l, l);
                        dot = _mm_cvtsi128_si32(l);
                        for (; j < blk_end; j++) {
                            dot += (int32_t)act[j] * (int32_t)row[j];
                        }
                        out4[sub] += (float)(dot - 128 * blk_sums[blk * 4 + sub])
                                    * row_scales[blk];
                    }
                }

                float deq = max_abs[b] / 127.0f;
                float* dst = output + b * rows;
                dst[0 * rows_packed + ur] = out4[0] * deq;
                dst[1 * rows_packed + ur] = out4[1] * deq;
                dst[2 * rows_packed + ur] = out4[2] * deq;
                dst[3 * rows_packed + ur] = out4[3] * deq;
            }
        }
        free(decode_buf);
        free(blk_sums);
    }

#ifdef ATLAS_DEBUG_TTYPE5
    if (act_f32_debug != nullptr) {
        float* ref = (float*)malloc(rows * sizeof(float));
        double total_sq_err = 0, max_rel = 0;
        int count = 0;
        for (int b = 0; b < B; b++) {
            for (int r = 0; r < rows; r++) {
                const uint8_t* w = packed + r * packed_cols;
                const float* act = act_f32_debug + b * input_dim;
                const float* rs = block_scales + r * n_blocks;
                double dot = 0.0;
                for (int c = 0; c < packed_cols; c++) {
                    const int8_t* l = tq1_decode[w[c]];
                    for (int t = 0; t < 5; t++) {
                        int col = c * 5 + t;
                        if (col >= input_dim) break;
                        dot += (double)act[col] * (double)l[t] * (double)rs[col / block_size];
                    }
                }
                ref[r] = (float)dot;
            }
            const float* out_b = output + b * rows;
            for (int r = 0; r < rows; r++) {
                int ur = r / 4, sub = r % 4;
                float actual = out_b[sub * rows_packed + ur];
                float diff = actual - ref[r];
                double rel = fabsf(diff) / (fabsf(ref[r]) + 1e-10f);
                total_sq_err += (double)diff * diff;
                count++;
                if (rel > max_rel) max_rel = rel;
            }
        }
        if (count > 0) {
            printf("[TQ5-DEBUG] mse=%.8f max_rel_err=%.6f rows=%d B=%d dim=%d\n",
                   (float)(total_sq_err / count), (float)max_rel, rows, B, input_dim);
        }
        free(ref);
    }
#endif

    free(block_scales);
}
#endif


// VNNI kernel lives in atlas_vnni.cpp (compiled with target("avx10.2"))

// ─── v2.10.4: F32 bypass for TQ1 block-scaled ternary matmul ──────────────
// Like matmul_tq1_block_fused_s8 but NO activation quantization.
// Uses f32 activations directly with decoded ternary {-1,0,+1} weights.
static void matmul_tq1_block_fused_f32(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const float* act_f32,
    float* output, int B) {
    #ifdef __aarch64__
    atlas_tq1_fused_f32_arm64(rows, input_dim, packed_cols,
        tensor_data, block_size, n_blocks, act_f32, output, B);
    #else
    #ifdef ATLAS_DEBUG_MODE
    double t0 = atlas_now();
    #endif
    init_tq1_decode_lut();
    int rows_packed = rows / 4;

    const uint8_t* scale_data = tensor_data + 3;
    const uint8_t* packed = tensor_data + 3 + rows * n_blocks * 2;

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
        #ifdef _OPENMP
        #pragma omp for
        #endif
        for (int i = 0; i < rows * n_blocks; i++) {
            uint16_t sr; memcpy(&sr, scale_data + i * 2, 2);
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
            int null_byte_val = 121;
            for (int sub = 0; sub < 4; sub++) {
                const uint8_t* w = packed + (ur * 4 + sub) * packed_cols;
                int8_t* row = decode_buf + sub * input_dim;
                int c = 0;
                for (; c < packed_cols - 1; c++) {
                    if (w[c] == null_byte_val) {
                        int col = c * 5;
                        row[col] = 0; row[col+1] = 0; row[col+2] = 0; row[col+3] = 0; row[col+4] = 0;
                        continue;
                    }
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

                        __m256 acc = _mm256_setzero_ps();
                        int j = blk_start;

                        for (; j + 8 <= blk_end; j += 8) {
                            __m128i w8 = _mm_loadl_epi64((const __m128i*)(row + j));
                            __m128i w4_0 = w8;
                            __m128i w4_1 = _mm_srli_si128(w8, 4);
                            __m256 wf_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w4_0));
                            __m256 wf_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w4_1));
                            __m256 af_0 = _mm256_loadu_ps(act + j);
                            __m256 af_1 = _mm256_loadu_ps(act + j + 4);
                            acc = _mm256_fmadd_ps(af_0, wf_0, acc);
                            acc = _mm256_fmadd_ps(af_1, wf_1, acc);
                        }

                        float dot = hsum_ps(acc);
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
    #endif
    #ifdef ATLAS_DEBUG_MODE
    double elapsed = atlas_now() - t0;
    if (elapsed > 0.1) {
        printf("[TIMER] matmul_tq1_fused_f32 rows=%d dim=%d B=%d: %.3fs\n", rows, input_dim, B, elapsed);
    }
    #endif
}

// ─── v2.7.0: Fused symmetric int8 quant + ternary matmul + per-block dequant ───
// act_f32: [B, input_dim] raw float activations (no pre-quantization needed)
// tensor_data: ttype=5 TQ1-packed weights + per-row per-block fp16 scales
// output: [B, rows] reordered float output
// Fuses: per-token max_abs → symmetric int8 → _mm256_sign_epi8 ternary matmul
//        → per-block fp16 scale → per-token dequant (scale_x)
// No uint8+128 offset, no row_sum correction. Eliminates the 14912× signal collapse.
static void matmul_tq1_block_fused_s8(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const float* act_f32,
    float* output, int B) {
    #ifdef __aarch64__
    atlas_tq1_fused_s8_arm64(rows, input_dim, packed_cols,
        tensor_data, block_size, n_blocks, act_f32, output, B);
    #else
    #ifdef ATLAS_DEBUG_MODE
    double t0 = atlas_now();
    #endif
    init_tq1_decode_lut();
    int rows_packed = rows / 4;

    const uint8_t* scale_data = tensor_data + 3;
    const uint8_t* packed = tensor_data + 3 + rows * n_blocks * 2;

    // Shared (non-TLS) buffer for per-row per-block scales
    static float* s_block_scales = nullptr;
    static size_t s_block_scales_cap = 0;
    static std::mutex s_bs_mutex;
    {
        size_t need_bs = (size_t)rows * n_blocks;
        if (need_bs > s_block_scales_cap) {
            std::lock_guard<std::mutex> lock(s_bs_mutex);
            if (need_bs > s_block_scales_cap) {
                free(s_block_scales);
                s_block_scales = (float*)malloc(need_bs * sizeof(float));
                s_block_scales_cap = need_bs;
            }
        }
    }
    float* block_scales = s_block_scales;
    static thread_local int8_t* tl_act_s8 = nullptr;
    static thread_local size_t tl_act_s8_cap = 0;
    static thread_local float* tl_scale_x = nullptr;
    static thread_local size_t tl_scale_x_cap = 0;

    // Step 1: Quantize activations to symmetric int8 (per-token, no +128 offset)
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

    // C++11 function-local static const = thread-safe one-time init
    // (Now using g_cpu dispatch table initialized in atlas_load)

    // Step 2: TQ1 decode + ternary matmul via _mm256_sign_epi8
    // Scale decode (fp16→fp32) integrated into parallel region
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        // Parallel fp16→fp32 scale decode
        #ifdef _OPENMP
        #pragma omp for
        #endif
        for (int i = 0; i < rows * n_blocks; i++) {
            uint16_t sr; memcpy(&sr, scale_data + i * 2, 2);
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

        // Act quantization: parallel over B (each token is independent)
        #ifdef _OPENMP
        #pragma omp for schedule(static)
        #endif
        for (int b = 0; b < B; b++) {
            const float* act = act_f32 + b * input_dim;
            float max_val = 1e-5f;
            for (int i = 0; i < input_dim; i++) {
                float av = fabsf(act[i]);
                if (av > max_val) max_val = av;
            }
            scale_x[b] = max_val / 127.0f;
            float inv = (max_val > 1e-10f) ? 127.0f / max_val : 0.0f;
            int8_t* aq = act_s8 + b * input_dim;
            for (int i = 0; i < input_dim; i++) {
                int q = (int)safe_int_from_float(act[i] * inv + 0.5f);
                if (q < -127) q = -127;
                if (q > 127) q = 127;
                aq[i] = (int8_t)q;
            }
        }

        #ifdef _OPENMP
        #pragma omp for
        #endif
        for (int ur = 0; ur < rows_packed; ur++) {
            // Decode TQ1 weights for 4 rows (same block_size/n_blocks as old kernel)
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

                        int32_t dot = 0;
                        int j = blk_start;
                        if (g_cpu.has_avx512_vnni) {
                            for (; j + 64 <= blk_end; j += 64) {
                                dot += atlas_matmul_block_vnni(act, row, j + 64, j);
                            }
                        } else {
                            __m256i acc_v = _mm256_setzero_si256();
                            for (; j + 64 <= blk_end; j += 64) {
                                __m256i av0 = _mm256_loadu_si256((const __m256i*)(act + j));
                                __m256i wv0 = _mm256_loadu_si256((const __m256i*)(row + j));
                                __m256i av1 = _mm256_loadu_si256((const __m256i*)(act + j + 32));
                                __m256i wv1 = _mm256_loadu_si256((const __m256i*)(row + j + 32));
                                __m256i prod0 = _mm256_sign_epi8(av0, wv0);
                                __m256i prod1 = _mm256_sign_epi8(av1, wv1);
                                __m128i lo0 = _mm256_castsi256_si128(prod0);
                                __m128i hi0 = _mm256_extracti128_si256(prod0, 1);
                                __m128i lo1 = _mm256_castsi256_si128(prod1);
                                __m128i hi1 = _mm256_extracti128_si256(prod1, 1);
                                __m256i lo16_0 = _mm256_cvtepi8_epi16(lo0);
                                __m256i hi16_0 = _mm256_cvtepi8_epi16(hi0);
                                __m256i lo16_1 = _mm256_cvtepi8_epi16(lo1);
                                __m256i hi16_1 = _mm256_cvtepi8_epi16(hi1);
                                __m256i sum32_0 = _mm256_add_epi32(
                                    _mm256_madd_epi16(lo16_0, _mm256_set1_epi16(1)),
                                    _mm256_madd_epi16(hi16_0, _mm256_set1_epi16(1)));
                                __m256i sum32_1 = _mm256_add_epi32(
                                    _mm256_madd_epi16(lo16_1, _mm256_set1_epi16(1)),
                                    _mm256_madd_epi16(hi16_1, _mm256_set1_epi16(1)));
                                acc_v = _mm256_add_epi32(acc_v, _mm256_add_epi32(sum32_0, sum32_1));
                            }
                            for (; j + 32 <= blk_end; j += 32) {
                                __m256i av = _mm256_loadu_si256((const __m256i*)(act + j));
                                __m256i wv = _mm256_loadu_si256((const __m256i*)(row + j));
                                __m256i prod = _mm256_sign_epi8(av, wv);
                                __m128i lo = _mm256_castsi256_si128(prod);
                                __m128i hi = _mm256_extracti128_si256(prod, 1);
                                __m256i lo16 = _mm256_cvtepi8_epi16(lo);
                                __m256i hi16 = _mm256_cvtepi8_epi16(hi);
                                __m256i sum32 = _mm256_add_epi32(
                                    _mm256_madd_epi16(lo16, _mm256_set1_epi16(1)),
                                    _mm256_madd_epi16(hi16, _mm256_set1_epi16(1)));
                                acc_v = _mm256_add_epi32(acc_v, sum32);
                            }
                            {
                                __m128i l = _mm256_castsi256_si128(acc_v);
                                __m128i h = _mm256_extracti128_si256(acc_v, 1);
                                l = _mm_add_epi32(l, h);
                                l = _mm_hadd_epi32(l, l);
                                l = _mm_hadd_epi32(l, l);
                                dot = _mm_cvtsi128_si32(l);
                            }
                        }
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
    #endif
    #ifdef ATLAS_DEBUG_MODE
    double elapsed = atlas_now() - t0;
    if (elapsed > 0.1) {
        printf("[TIMER] matmul_tq1_fused rows=%d dim=%d B=%d: %.3fs\n", rows, input_dim, B, elapsed);
    }
    #endif
}

// ─── TQ2: Universal block-scaled ternary matmul ─────────────────────
// TQ2 header: [block_size:1][n_blocks:2][flags:1][scales:rows*n_blocks*2][packed_TQ1]
// P1 delegate to matmul_tq1_block_fused_s8 (skip flags byte, matching ttype=5 layout).
static void matmul_tq2(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const float* act_f32, float* output, int B) {
    #ifdef __aarch64__
    atlas_tq2_arm64(rows, input_dim, packed_cols,
        tensor_data, block_size, n_blocks, act_f32, output, B);
    #else
    // skip flags byte so ttype=5 layout aligns (scales at +3, packed at +3+rows*n_blocks*2)
    const uint8_t* w5 = tensor_data + 1;
    matmul_tq1_block_fused_s8(rows, input_dim, packed_cols,
        w5, block_size, n_blocks, act_f32, output, B);
    #endif
}

// ─── v2.10.4: F32 bypass wrapper for ttype=10 ──────────────────────
static void matmul_tq2_f32(int rows, int input_dim, int packed_cols,
    const uint8_t* tensor_data, int block_size, int n_blocks,
    const float* act_f32, float* output, int B) {
    #ifdef __aarch64__
    atlas_tq2_f32_arm64(rows, input_dim, packed_cols,
        tensor_data, block_size, n_blocks, act_f32, output, B);
    #else
    const uint8_t* w5 = tensor_data + 1;
    matmul_tq1_block_fused_f32(rows, input_dim, packed_cols,
        w5, block_size, n_blocks, act_f32, output, B);
    #endif
}

// ─── TQ1 encode: 5 ternary values → 1 byte (Base-3) ────────────────
// Ternary values in {-1,0,1}, stored as int8.
// Encoding: -1→0, 0→1, 1→2, then byte = t0 + t1*3 + t2*9 + t3*27 + t4*81
static int tq1_encode_5(const int8_t v[5]) {
    int b = 0, mul = 1;
    for (int i = 0; i < 5; i++) {
        b += (v[i] + 1) * mul;
        mul *= 3;
    }
    if (b > 242) b = 242;  // 3^5 - 1 = 242, should never exceed
    return b;
}

// ─── TQ2 Converter: int8 weights → block-scaled TQ2 format ────────
// Takes int8 weight matrix [rows × input_dim] with per-tensor fp16 scale,
// produces TQ2 packed buffer (caller must atlas_vfree).
// Returns 0 on success, -1 on error.
static int quantize_weights_to_tq2(
    const int8_t* weights, float weight_scale,
    int rows, int input_dim,
    uint8_t** out_buf, int* out_size,
    int block_size = 128) {

    int n_blocks = (input_dim + block_size - 1) / block_size;
    int packed_per_row = (input_dim + 4) / 5;  // ceil(input_dim / 5)
    int hdr = 4;  // block_size + n_blocks + flags
    int scales_size = rows * n_blocks * 2;
    int packed_size = rows * packed_per_row;
    int total = hdr + scales_size + packed_size;

    uint8_t* buf = atlas_valloc(total);
    if (!buf) return -1;

    // Write header
    buf[0] = (uint8_t)block_size;
    buf[1] = (uint8_t)(n_blocks & 0xFF);
    buf[2] = (uint8_t)((n_blocks >> 8) & 0xFF);
    buf[3] = 0;  // flags: no bitmap

    uint8_t* scales = buf + hdr;
    uint8_t* packed = buf + hdr + scales_size;

    for (int r = 0; r < rows; r++) {
        const int8_t* row = weights + r * input_dim;

        // Per-block scale: max(abs(weight)) for each block
        int8_t ternary[128];  // block_size=128
        int dc = 0;  // ternary index for this row

        for (int blk = 0; blk < n_blocks; blk++) {
            int blk_start = blk * block_size;
            int blk_end = blk_start + block_size;
            if (blk_end > input_dim) blk_end = input_dim;

            float max_abs = 1e-10f;
            int8_t blk_tern[128];
            for (int c = blk_start; c < blk_end; c++) {
                float v = (float)row[c] / weight_scale;
                int t = (v >= 0) ? (int)safe_int_from_float(v + 0.5f) : (int)safe_int_from_float(v - 0.5f);
                if (t < -1) t = -1;
                if (t > 1) t = 1;
                blk_tern[c - blk_start] = (int8_t)t;
                if (t != 0) {
                    float av = fabsf((float)t);
                    if (av > max_abs) max_abs = av;
                }
            }

            // Store fp16 scale (recomputed from actual ternary max)
            int16_t scale_f16 = fp32_to_fp16(max_abs);
            memcpy(scales + (r * n_blocks + blk) * 2, &scale_f16, 2);

            // Copy block ternary values to row buffer
            for (int c = 0; c < blk_end - blk_start; c++) {
                ternary[dc++] = blk_tern[c];
            }
        }

        // Pad to full blocks of 5
        while (dc % 5 != 0) ternary[dc++] = 0;

        // Pack 5 trits/byte
        int n5 = dc / 5;
        for (int g = 0; g < n5; g++) {
            packed[r * packed_per_row + g] = (uint8_t)tq1_encode_5(ternary + g * 5);
        }
    }

    *out_buf = buf;
    *out_size = total;
    return 0;
}

// ─── Load-time TQ2 conversion: scan tensors, convert old formats to ttype=10 ──
// Converts ttype=0,3,5,7,8 to ttype=10 in-place. Leaves ttype=1,2 (norms/embeds).
// Call after atlas_load(), before first inference.
ATLAS_API void atlas_convert_to_tq2(void* model_ptr) {
    AtlasModel* m = (AtlasModel*)model_ptr;
    if (!m) return;
    int total = 0;

    for (size_t i = 0; i < m->tensors.size(); i++) {
        auto& t = m->tensors[i];
        if (t.ttype == 1 || t.ttype == 2) continue;  // norms/embeds — keep as fp16

        if (t.ttype == 10) continue;  // already TQ2

        if (t.ttype == 5 || t.ttype == 7) {
            // Already block-scaled — just change ttype and add flags byte
            // ttype=5: [block_size:1][n_blocks:2][scales:...][packed:...]
            // ttype=7: [block_size:1][n_blocks:2][scales:...][packed:...]
            // ttype=10: [block_size:1][n_blocks:2][flags:1][scales:...][packed:...]
            // Insert flags byte at offset 3, shift data by 1
            int new_size = t.data_size + 1;
            uint8_t* new_data = atlas_valloc(new_size);
            if (!new_data) continue;
            memcpy(new_data, t.data, 3);              // block_size + n_blocks
            new_data[3] = 0;                          // flags = 0
            memcpy(new_data + 4, t.data + 3, t.data_size - 3);  // scales + packed
            if (t.data && !m->is_mapped(t.data)) atlas_vfree(t.data);
            t.data = new_data;
            t.data_size = new_size;
            t.ttype = 10;
            total++;
            continue;
        }

        // ttype=0: [fp16_scale:2][packed_TQ1:rows*packed_per_row]
        // ttype=3: [fp16_scale:2][i8_weights:rows*cols][row_sums:rows*4]
        // ttype=8: [fp16_scale:2][packed_i4:rows*packed_cols][row_sums:rows*4]
        // Need to decompress → per-block scales → TQ1 repack

        int rows = t.row_dim;
        int cols = 0;
        float scale = 0.0f;
        const int8_t* i8_weights = nullptr;

        if (t.ttype == 0) {
            // TQ1 packed: decode to int8 first
            init_tq1_decode_lut();
            uint16_t s16; memcpy(&s16, t.data, 2);
            scale = fp16_to_fp32(s16);
            cols = t.packed_cols * 5;
            int n_vals = rows * cols;
            int8_t* i8 = (int8_t*)malloc(n_vals);
            const uint8_t* packed = t.data + 2;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < t.packed_cols; c++) {
                    const int8_t* l = tq1_decode[packed[r * t.packed_cols + c]];
                    for (int t5 = 0; t5 < 5; t5++) {
                        int col = c * 5 + t5;
                        if (col < cols) i8[r * cols + col] = l[t5];
                    }
                }
            }
            i8_weights = i8;
        } else if (t.ttype == 3) {
            uint16_t s16; memcpy(&s16, t.data, 2);
            scale = fp16_to_fp32(s16);
            int rs_size = rows * 4;
            cols = (t.data_size - 2 - rs_size) / rows;
            i8_weights = (const int8_t*)(t.data + 2);
        } else if (t.ttype == 8) {
            // Int4 packed: decompress to int8
            uint16_t s16; memcpy(&s16, t.data, 2);
            scale = fp16_to_fp32(s16);
            cols = t.packed_cols * 2;
            int rs_size = rows * 4;
            int n_vals = rows * cols;
            int8_t* i8 = (int8_t*)malloc(n_vals);
            const uint8_t* packed_i4 = t.data + 2;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < t.packed_cols; c++) {
                    int byte = packed_i4[r * t.packed_cols + c];
                    int c2 = c * 2;
                    if (c2 < cols) {
                        int v0 = (int8_t)((byte & 0x0F) ^ 8) - 8;
                        i8[r * cols + c2] = (int8_t)v0;
                    }
                    if (c2 + 1 < cols) {
                        int v1 = (int8_t)(((byte >> 4) & 0x0F) ^ 8) - 8;
                        i8[r * cols + c2 + 1] = (int8_t)v1;
                    }
                }
            }
            i8_weights = i8;
        } else {
            continue;  // unknown ttype, skip
        }

        if (!i8_weights || cols <= 0) continue;

        uint8_t* tq2_buf = nullptr;
        int tq2_size = 0;
        t.block_size = 128;
        t.n_blocks = (cols + 127) / 128;
        // ttype=0 int8 values are already ternary {-1,0,+1}. Don't divide:
        // pass scale=1.0, then multiply block scales by original weight_scale.
        float tq2_scale = (t.ttype == 0) ? 1.0f : scale;
        int ret = quantize_weights_to_tq2(i8_weights, tq2_scale, rows, cols,
                                          &tq2_buf, &tq2_size, 128);
        if (ret == 0 && tq2_buf) {
            // For already-ternary weights, restore the original weight_scale
            // into per-block fp16 scales.
            if (t.ttype == 0 && scale > 0.0f) {
                int nb = t.n_blocks;
                uint16_t* sptr = (uint16_t*)(tq2_buf + 4);
                for (int j = 0; j < rows * nb; j++) {
                    float sf = fp16_to_fp32(sptr[j]) * scale;
                    if (sf < 1e-10f) sf = 1e-10f;
                    sptr[j] = fp32_to_fp16(sf);
                }
            }
            if (t.data && !m->is_mapped(t.data)) {
                if (t.ttype == 0 || t.ttype == 8) {
                    free((void*)i8_weights);  // we allocated this
                }
                atlas_vfree(t.data);
            }
            t.data = tq2_buf;
            t.data_size = tq2_size;
            t.packed_cols = (cols + 4) / 5;
            t.ttype = 10;
            total++;
        } else {
            if (t.ttype == 0 || t.ttype == 8) free((void*)i8_weights);
        }
    }

    if (total > 0) {
        printf("[ATLAS] Converted %d tensors to TQ2\n", total);
        // TQ2 uses symmetric s8 activation quant — no f32 bypass needed
        m->use_f32_matmul = 0;
    }
}

// ─── Helper: f32×i8 matmul + reorder (no activation quantization) ───
// act_f32: [B, input_dim] float activations (not quantized)
// weights: [rows, input_dim] int8 weights
// scale: per-tensor dequant scale
// output: [B, rows] reordered float output
#ifndef __aarch64__
static void matmul_f32_reorder(int rows, int input_dim,
    const int8_t* __restrict__ weights, const float* __restrict__ act_f32,
    float scale, float* __restrict__ output, int B) {
    int rows_packed = rows / 4;
    #ifdef _OPENMP
    #pragma omp parallel for if(rows_packed > 4)
    #endif
    for (int ur = 0; ur < rows_packed; ur++) {
        for (int b = 0; b < B; b++) {
            const float* a = act_f32 + b * input_dim;
            float out4[4];
            for (int sub = 0; sub < 4; sub++) {
                const int8_t* w = weights + (ur * 4 + sub) * input_dim;
                __m256 sum = _mm256_setzero_ps();
                int c = 0;
                for (; c + 8 <= input_dim; c += 8) {
                    __m256 af = _mm256_loadu_ps(a + c);
                    __m128i w8 = _mm_loadl_epi64((const __m128i*)(w + c));
                    __m256i w32 = _mm256_cvtepi8_epi32(w8);
                    __m256 wf = _mm256_cvtepi32_ps(w32);
                    sum = _mm256_fmadd_ps(af, wf, sum);
                }
                float s = hsum_ps(sum);
                for (; c < input_dim; c++) s += a[c] * w[c];
                out4[sub] = s / scale;
            }
            float* dst = output + b * rows;
            dst[0 * rows_packed + ur] = out4[0];
            dst[1 * rows_packed + ur] = out4[1];
            dst[2 * rows_packed + ur] = out4[2];
            dst[3 * rows_packed + ur] = out4[3];
        }
    }
}

// ─── Per-row int8 matmul (ttype=11) — reordered output ───────────────
// Like matmul_f32_reorder but each weight row has its own fp16 scale.
// row_scales_fp16[r] = 127 / max_abs(row_r). Reordered output (4-row chunks).
static void matmul_f32_reorder_per_row(int rows, int dim,
    const int8_t* __restrict__ weights, const float* __restrict__ act_f32,
    const uint16_t* __restrict__ row_scales_fp16,
    float* __restrict__ output, int B) {
    int rows_packed = rows / 4;
    #ifdef _OPENMP
    #pragma omp parallel for if(rows_packed > 4)
    #endif
    for (int ur = 0; ur < rows_packed; ur++) {
        float rsc[4];
        for (int sub = 0; sub < 4; sub++)
            rsc[sub] = fp16_to_fp32(row_scales_fp16[ur * 4 + sub]);
        for (int b = 0; b < B; b++) {
            const float* a = act_f32 + b * dim;
            float out4[4];
            for (int sub = 0; sub < 4; sub++) {
                const int8_t* w = weights + (ur * 4 + sub) * dim;
                __m256 sum = _mm256_setzero_ps();
                int c = 0;
                for (; c + 8 <= dim; c += 8) {
                    __m256 af = _mm256_loadu_ps(a + c);
                    __m128i w8 = _mm_loadl_epi64((const __m128i*)(w + c));
                    __m256i w32 = _mm256_cvtepi8_epi32(w8);
                    __m256 wf = _mm256_cvtepi32_ps(w32);
                    sum = _mm256_fmadd_ps(af, wf, sum);
                }
                float s = hsum_ps(sum);
                for (; c < dim; c++) s += a[c] * w[c];
                out4[sub] = s / rsc[sub];
            }
            float* dst = output + b * rows;
            dst[0 * rows_packed + ur] = out4[0];
            dst[1 * rows_packed + ur] = out4[1];
            dst[2 * rows_packed + ur] = out4[2];
            dst[3 * rows_packed + ur] = out4[3];
        }
    }
    // Remaining rows (if rows % 4 != 0)
    int rem_start = rows_packed * 4;
    for (int r = 0; r < rows - rem_start; r++) {
        int ri = rem_start + r;
        float rs = fp16_to_fp32(row_scales_fp16[ri]);
        for (int b = 0; b < B; b++) {
            const int8_t* w = weights + ri * dim;
            const float* a = act_f32 + b * dim;
            __m256 sum = _mm256_setzero_ps();
            int c = 0;
            for (; c + 8 <= dim; c += 8) {
                __m256 af = _mm256_loadu_ps(a + c);
                __m128i w8 = _mm_loadl_epi64((const __m128i*)(w + c));
                __m256i w32 = _mm256_cvtepi8_epi32(w8);
                __m256 wf = _mm256_cvtepi32_ps(w32);
                sum = _mm256_fmadd_ps(af, wf, sum);
            }
            float s = hsum_ps(sum);
            for (; c < dim; c++) s += a[c] * w[c];
            output[b * rows + ri] = s / rs;
        }
    }
}

// ─── Per-row int8 matmul (ttype=11) — non-reordered, stride-aware ───
static void matmul_f32_per_row(int rows, int input_dim,
    const int8_t* __restrict__ weights, const float* __restrict__ act_f32,
    int act_stride, const uint16_t* __restrict__ row_scales_fp16,
    float* __restrict__ output, int B) {
    int rows_packed = rows / 4;
    #ifdef _OPENMP
    #pragma omp parallel for if(rows_packed > 4)
    #endif
    for (int ur = 0; ur < rows_packed; ur++) {
        float rsc[4];
        for (int sub = 0; sub < 4; sub++)
            rsc[sub] = fp16_to_fp32(row_scales_fp16[ur * 4 + sub]);
        for (int b = 0; b < B; b++) {
            const float* a = act_f32 + b * act_stride;
            float out4[4];
            for (int sub = 0; sub < 4; sub++) {
                const int8_t* w = weights + (ur * 4 + sub) * input_dim;
                __m256 sum = _mm256_setzero_ps();
                int c = 0;
                for (; c + 8 <= input_dim; c += 8) {
                    __m256 af = _mm256_loadu_ps(a + c);
                    __m128i w8 = _mm_loadl_epi64((const __m128i*)(w + c));
                    __m256i w32 = _mm256_cvtepi8_epi32(w8);
                    __m256 wf = _mm256_cvtepi32_ps(w32);
                    sum = _mm256_fmadd_ps(af, wf, sum);
                }
                float s = hsum_ps(sum);
                for (; c < input_dim; c++) s += a[c] * w[c];
                out4[sub] = s / rsc[sub];
            }
            float* dst = output + b * rows;
            dst[ur * 4 + 0] = out4[0];
            dst[ur * 4 + 1] = out4[1];
            dst[ur * 4 + 2] = out4[2];
            dst[ur * 4 + 3] = out4[3];
        }
    }
}
#else
static void matmul_f32_reorder(int rows, int input_dim,
    const int8_t* __restrict__ weights, const float* __restrict__ act_f32,
    float scale, float* __restrict__ output, int B) {
    int rows_packed = rows / 4;
    #ifdef _OPENMP
    #pragma omp parallel for if(rows_packed > 4)
    #endif
    for (int ur = 0; ur < rows_packed; ur++) {
        for (int b = 0; b < B; b++) {
            const float* a = act_f32 + b * input_dim;
            float out4[4];
            for (int sub = 0; sub < 4; sub++) {
                const int8_t* w = weights + (ur * 4 + sub) * input_dim;
                float s = 0.0f;
                for (int i = 0; i < input_dim; i++) s += a[i] * (float)w[i];
                out4[sub] = s / scale;
            }
            float* dst = output + b * rows;
            dst[0 * rows_packed + ur] = out4[0];
            dst[1 * rows_packed + ur] = out4[1];
            dst[2 * rows_packed + ur] = out4[2];
            dst[3 * rows_packed + ur] = out4[3];
        }
    }
}
static void matmul_f32_reorder_per_row(int rows, int dim,
    const int8_t* __restrict__ weights, const float* __restrict__ act_f32,
    const uint16_t* __restrict__ row_scales_fp16,
    float* __restrict__ output, int B) {
    int rows_packed = rows / 4;
    #ifdef _OPENMP
    #pragma omp parallel for if(rows_packed > 4)
    #endif
    for (int ur = 0; ur < rows_packed; ur++) {
        float rsc[4];
        for (int sub = 0; sub < 4; sub++)
            rsc[sub] = fp16_to_fp32(row_scales_fp16[ur * 4 + sub]);
        for (int b = 0; b < B; b++) {
            const float* a = act_f32 + b * dim;
            float out4[4];
            for (int sub = 0; sub < 4; sub++) {
                const int8_t* w = weights + (ur * 4 + sub) * dim;
                float s = 0.0f;
                for (int i = 0; i < dim; i++) s += a[i] * (float)w[i];
                out4[sub] = s / rsc[sub];
            }
            float* dst = output + b * rows;
            dst[0 * rows_packed + ur] = out4[0];
            dst[1 * rows_packed + ur] = out4[1];
            dst[2 * rows_packed + ur] = out4[2];
            dst[3 * rows_packed + ur] = out4[3];
        }
    }
    int rem_start = rows_packed * 4;
    for (int r = 0; r < rows - rem_start; r++) {
        int ri = rem_start + r;
        float rs = fp16_to_fp32(row_scales_fp16[ri]);
        for (int b = 0; b < B; b++) {
            const int8_t* w = weights + ri * dim;
            const float* a = act_f32 + b * dim;
            float s = 0.0f;
            for (int i = 0; i < dim; i++) s += a[i] * (float)w[i];
            output[b * rows + ri] = s / rs;
        }
    }
}
static void matmul_f32_per_row(int rows, int input_dim,
    const int8_t* __restrict__ weights, const float* __restrict__ act_f32,
    int act_stride, const uint16_t* __restrict__ row_scales_fp16,
    float* __restrict__ output, int B) {
    #ifdef _OPENMP
    #pragma omp parallel for if(rows > 4)
    #endif
    for (int r = 0; r < rows; r++) {
        float rs = fp16_to_fp32(row_scales_fp16[r]);
        for (int b = 0; b < B; b++) {
            const float* a = act_f32 + b * act_stride;
            const int8_t* w = weights + r * input_dim;
            float s = 0.0f;
            for (int i = 0; i < input_dim; i++) s += a[i] * (float)w[i];
            output[b * rows + r] = s / rs;
        }
    }
}
#endif

// ─── Helper: int8 matmul + reorder + dequant ──────────────────────────
// act_u8: [B, input_dim] quantized activations
// max_abs: [B] per-token max
// scratch: [B, rows] raw matmul scratch
// output: [B, rows] dequantized + reordered
static void matmul_reorder_deq(int rows, int input_dim,
    const int8_t* weights, const int32_t* row_sums,
    const uint8_t* act_u8, const float* max_abs,
    float scale, float* scratch, float* output, int B) {
    atlas_matmul_i8_f32(rows, input_dim, weights, act_u8, row_sums, scratch, B);
    int rows_packed = rows / 4;
    float deq_scale = 1.0f / (127.0f * scale);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(B > 4)
    #endif
    for (int t = 0; t < B; t++) {
        float mabs = max_abs[t];
        float* dst = output + t * rows;
        float* src = scratch + t * rows;
        for (int ur = 0; ur < rows_packed; ur++) {
            int a0 = ur * 4 + 0, a1 = ur * 4 + 1, a2 = ur * 4 + 2, a3 = ur * 4 + 3;
            int h0 = 0 * rows_packed + ur;
            int h1 = 1 * rows_packed + ur;
            int h2 = 2 * rows_packed + ur;
            int h3 = 3 * rows_packed + ur;
            dst[h0] = src[a0] * mabs * deq_scale;
            dst[h1] = src[a1] * mabs * deq_scale;
            dst[h2] = src[a2] * mabs * deq_scale;
            dst[h3] = src[a3] * mabs * deq_scale;
        }
    }
}

// ─── int4 matmul wrapper: matmul + dequant + reorder (modeled on matmul_reorder_deq) ───
static void matmul_i4_reorder_deq(int rows, int cols,
    const uint8_t* packed_weights, const int32_t* row_sums,
    const uint8_t* act_u8, const float* max_abs,
    float scale, float* scratch, float* output, int B) {
    atlas_matmul_i4_f32(rows, cols, packed_weights, act_u8, row_sums, scratch, B);
    int rows_packed = rows / 4;
    float deq_scale = 1.0f / (127.0f * scale);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(B > 4)
    #endif
    for (int t = 0; t < B; t++) {
        float mabs = max_abs[t];
        float* dst = output + t * rows;
        float* src = scratch + t * rows;
        for (int ur = 0; ur < rows_packed; ur++) {
            int a0 = ur * 4 + 0, a1 = ur * 4 + 1, a2 = ur * 4 + 2, a3 = ur * 4 + 3;
            int h0 = 0 * rows_packed + ur;
            int h1 = 1 * rows_packed + ur;
            int h2 = 2 * rows_packed + ur;
            int h3 = 3 * rows_packed + ur;
            dst[h0] = src[a0] * mabs * deq_scale;
            dst[h1] = src[a1] * mabs * deq_scale;
            dst[h2] = src[a2] * mabs * deq_scale;
            dst[h3] = src[a3] * mabs * deq_scale;
        }
    }
}


// ============================================================================
// MLA (Multi-head Latent Attention) + MoE Kernels for DeepSeek-V2
// ============================================================================

// ─── Row-major matmul: output[rows] = weights[rows,dim] @ input[dim] ───
// Unlike matmul_f32_reorder (blocked output), this writes standard row-major.
// Used for KV_b decompression: c_kv [lora_rank] → k_nope + v [out_dim]
#ifndef __aarch64__
static void matmul_f32_row_major(int rows, int dim,
    const int8_t* __restrict__ weights, const float* __restrict__ act,
    float scale, float* __restrict__ output) {
    #ifdef _OPENMP
    #pragma omp parallel for if(rows > 16)
    #endif
    for (int r = 0; r < rows; r++) {
        const int8_t* w = weights + (int64_t)r * dim;
        __m256 sum = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= dim; c += 8) {
            __m256 af = _mm256_loadu_ps(act + c);
            __m128i w8 = _mm_loadl_epi64((const __m128i*)(w + c));
            __m256i w32 = _mm256_cvtepi8_epi32(w8);
            __m256 wf = _mm256_cvtepi32_ps(w32);
            sum = _mm256_fmadd_ps(af, wf, sum);
        }
        float s = hsum_ps(sum);
        for (; c < dim; c++) s += act[c] * (float)w[c];
        output[r] = s / scale;
    }
}

// Per-row variant (ttype=11)
static void matmul_f32_row_major_per_row(int rows, int dim,
    const int8_t* __restrict__ weights, const float* __restrict__ act,
    const uint16_t* __restrict__ row_scales_fp16, float* __restrict__ output) {
    #ifdef _OPENMP
    #pragma omp parallel for if(rows > 16)
    #endif
    for (int r = 0; r < rows; r++) {
        float rs = fp16_to_fp32(row_scales_fp16[r]);
        const int8_t* w = weights + (int64_t)r * dim;
        __m256 sum = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= dim; c += 8) {
            __m256 af = _mm256_loadu_ps(act + c);
            __m128i w8 = _mm_loadl_epi64((const __m128i*)(w + c));
            __m256i w32 = _mm256_cvtepi8_epi32(w8);
            __m256 wf = _mm256_cvtepi32_ps(w32);
            sum = _mm256_fmadd_ps(af, wf, sum);
        }
        float s = hsum_ps(sum);
        for (; c < dim; c++) s += act[c] * (float)w[c];
        output[r] = s / rs;
    }
}
#else
static void matmul_f32_row_major(int rows, int dim,
    const int8_t* __restrict__ weights, const float* __restrict__ act,
    float scale, float* __restrict__ output) {
    #ifdef _OPENMP
    #pragma omp parallel for if(rows > 16)
    #endif
    for (int r = 0; r < rows; r++) {
        const int8_t* w = weights + (int64_t)r * dim;
        float s = 0.0f;
        for (int c = 0; c < dim; c++) s += act[c] * (float)w[c];
        output[r] = s / scale;
    }
}
static void matmul_f32_row_major_per_row(int rows, int dim,
    const int8_t* __restrict__ weights, const float* __restrict__ act,
    const uint16_t* __restrict__ row_scales_fp16, float* __restrict__ output) {
    #ifdef _OPENMP
    #pragma omp parallel for if(rows > 16)
    #endif
    for (int r = 0; r < rows; r++) {
        float rs = fp16_to_fp32(row_scales_fp16[r]);
        const int8_t* w = weights + (int64_t)r * dim;
        float s = 0.0f;
        for (int c = 0; c < dim; c++) s += act[c] * (float)w[c];
        output[r] = s / rs;
    }
}
#endif

// Batched row-major matmul wrappers: write directly to output in row-major layout.
// Used for MLA projections (Q, kv_a, O) to avoid blocked layout + un-reorder.
static void matmul_f32_row_major_batched(int rows, int dim,
    const int8_t* __restrict__ weights, const float* __restrict__ act,
    int act_stride, float scale, float* __restrict__ output, int B) {
    for (int b = 0; b < B; b++)
        matmul_f32_row_major(rows, dim, weights, act + b * act_stride, scale, output + b * rows);
}

static void matmul_f32_row_major_per_row_batched(int rows, int dim,
    const int8_t* __restrict__ weights, const float* __restrict__ act,
    int act_stride, const uint16_t* __restrict__ row_scales_fp16,
    float* __restrict__ output, int B) {
    for (int b = 0; b < B; b++)
        matmul_f32_row_major_per_row(rows, dim, weights, act + b * act_stride, row_scales_fp16, output + b * rows);
}

// ─── Tensor dimension helpers (standalone, used by MLA + MoE paths) ───
static int t3_dim(const TensorInfo& t) {
    if (t.ttype == 10 || t.ttype == 5) return t.packed_cols * 5;
    if (t.ttype == 3) return (int)((t.data_size - 2 - (int64_t)t.row_dim * 4) / t.row_dim);
    if (t.ttype == 11) return (int)((t.data_size - (int64_t)t.row_dim * 6) / t.row_dim);
    if (t.ttype == 7) return t.packed_cols * 4;
    if (t.ttype == 8) return t.packed_cols * 2;
    return t.packed_cols * 5;
}

static void get_i8(const TensorInfo& t, int8_t*& w, int32_t*& rs,
                   int& rows, int& dim, float& scale) {
    if (t.ttype != 3) { rows = 0; dim = 0; w = nullptr; rs = nullptr; return; }
    uint16_t sr; memcpy(&sr, t.data, 2);
    scale = fp16_to_fp32(sr);
    rows = t.row_dim;
    dim = t3_dim(t);
    int nv = rows * dim;
    w = (int8_t*)(t.data + 2);
    const uint8_t* raw_rs = (const uint8_t*)(w + nv);
    thread_local int32_t aligned_rs_buf[8192];
    if (rows <= 8192) {
        for (int r = 0; r < rows; r++)
            aligned_rs_buf[r] = read_i32_unaligned(raw_rs + r * 4);
        rs = aligned_rs_buf;
    } else {
        rs = (int32_t*)(w + nv);
    }
}

// ─── MLA Attention: On-the-fly KV decompression + standard dot-product attention ───
// Compressed cache layout per layer, per position:
//   [ c_kv (kv_lora_rank fp16) | k_pe (qk_rope_head_dim fp16) ]
// Reconstruction: K_nope, V = W_kv_b @ c_kv;  K_pe from cache (shared across heads)
static void atlas_attention_mla(
    AtlasModel* m,
    float* q_full,             // [B, n_heads * head_dim] — Q after projection, RoPE applied in-place to q_pe
    float* output,             // [B, n_heads * head_dim] — attention output
    const float* c_kv_new,     // [B, kv_lora_rank] — new compressed KV latent (after layernorm)
    const float* k_pe_new,     // [B, qk_rope_head_dim] — new RoPE key component
    int layer, const int* positions,
    int max_seq_len, int seq_now, int B) {

    int nH = m->n_heads;
    int nope = m->qk_nope_head_dim;
    int rope = m->qk_rope_head_dim;
    int hd = nope + rope;
    int lora = m->kv_lora_rank;
    int qd = nH * hd;
    float theta = m->rope_theta;
    float inv_sqrt_d = 1.0f / sqrtf((float)hd);

    // NTK context extension (same logic as atlas_attention_f32)
    float ctx_scale = m->base_seq_len > 0 ? (float)max_seq_len / (float)m->base_seq_len : 1.0f;
    if (ctx_scale < 1.0f) ctx_scale = 1.0f;
    float total_scale = m->rope_scale;
    if (ctx_scale > 1.001f) total_scale *= ctx_scale;
    float eff_theta = theta;
    if (total_scale > 1.001f)
        eff_theta *= powf(total_scale, (float)rope / (float)(rope - 2));

    int ring_start = seq_now > max_seq_len ? seq_now - max_seq_len : 0;
    int ring_len = seq_now > max_seq_len ? max_seq_len : seq_now;

    // Store new c_kv + k_pe into compressed cache
    int stride = m->compressed_kv_stride;
    if (stride <= 0) stride = lora + rope;
    for (int b = 0; b < B; b++) {
        int pos = positions[b];
        int cache_pos = pos % max_seq_len;
        uint16_t* dst = (uint16_t*)m->compressed_kv_cache +
            ((size_t)layer * max_seq_len + cache_pos) * stride;
        // Store c_kv (fp32 → fp16)
        for (int i = 0; i < lora; i++)
            dst[i] = fp32_to_fp16(c_kv_new[b * lora + i]);
        // Store k_pe (fp32 → fp16)
        for (int i = 0; i < rope; i++)
            dst[lora + i] = fp32_to_fp16(k_pe_new[b * rope + i]);
    }

    // Get W_kv_b weight info
    const int* idx = m->layer_idx_cache.data() + layer * m->layer_stride;
    auto& t_kv_b = m->tensors[idx[3]];
    int kv_b_rows = nH * (nope + m->v_head_dim);  // total output dim of kv_b
    int8_t* w_kv_b = nullptr;
    int32_t* rs_kv_b = nullptr;
    int kv_b_dim = 0;
    float kv_b_scale = 1.0f;
    const uint16_t* kv_b_row_scales = nullptr;
    if (t_kv_b.ttype == 3) {
        uint16_t sr; memcpy(&sr, t_kv_b.data, 2);
        kv_b_scale = fp16_to_fp32(sr);
        w_kv_b = (int8_t*)(t_kv_b.data + 2);
        rs_kv_b = (int32_t*)(w_kv_b + (int64_t)kv_b_rows * kv_b_dim);
        kv_b_dim = t3_dim(t_kv_b);
        rs_kv_b = (int32_t*)(w_kv_b + (int64_t)kv_b_rows * kv_b_dim);
    } else if (t_kv_b.ttype == 11) {
        kv_b_row_scales = (const uint16_t*)(t_kv_b.data);
        w_kv_b = (int8_t*)(t_kv_b.data + kv_b_rows * 2);
        kv_b_dim = t3_dim(t_kv_b);
    }

    // Scratch for per-position kv_b output + RoPE scratch (from attn_ws heap)
    size_t qd_sz = (size_t)B * nH * hd;
    float* kv_out   = m->attn_ws + 2 * qd_sz + (size_t)B * rope;
    float* c_kv_f32 = kv_out + kv_b_rows;
    float* k_pe_f32 = c_kv_f32 + lora;

    // Attention scores buffer (heap-allocated for large seq)
    static thread_local float* scores_buf = nullptr;
    static thread_local size_t scores_cap = 0;
    size_t needed = (size_t)nH * ring_len;
    if (needed > scores_cap) {
        free(scores_buf);
        scores_cap = needed;
        scores_buf = (float*)malloc(scores_cap * sizeof(float));
        if (!scores_buf) { scores_cap = 0; return; }
    }
    float* scores = scores_buf;

    for (int b = 0; b < B; b++) {
        int pos = positions[b];
        float* qb = q_full + b * qd;
        float* out_b = output + b * qd;
        memset(out_b, 0, qd * sizeof(float));

        // ── Phase 1: Compute all attention scores ──
        for (int i = 0; i < nH * ring_len; i++) scores[i] = -1e9f;

        for (int s = 0; s < ring_len; s++) {
            int key_pos = ring_start + s;
            if (key_pos > pos) break;
            int cache_pos = key_pos % max_seq_len;

            // Load compressed c_kv from cache (fp16 → f32)
            const uint16_t* c_kv_fp16 = (const uint16_t*)m->compressed_kv_cache +
                ((size_t)layer * max_seq_len + cache_pos) * stride;

            // c_kv_f32 already hoisted before ring loop
            for (int i = 0; i < lora; i++)
                c_kv_f32[i] = fp16_to_fp32(c_kv_fp16[i]);

            // Matmul: kv_out = W_kv_b @ c_kv_f32
            if (w_kv_b && kv_b_dim > 0) {
                if (kv_b_row_scales)
                    matmul_f32_row_major_per_row(kv_b_rows, kv_b_dim, w_kv_b, c_kv_f32, kv_b_row_scales, kv_out);
                else
                    matmul_f32_row_major(kv_b_rows, kv_b_dim, w_kv_b, c_kv_f32, kv_b_scale, kv_out);
            }

            // k_pe_f32 already hoisted before ring loop
            for (int i = 0; i < rope; i++)
                k_pe_f32[i] = fp16_to_fp32(c_kv_fp16[lora + i]);

            // Apply RoPE to k_pe (half-split, matching DeepSeek-V2 convention)
            for (int i = 0; i < rope / 2; i++) {
                float freq = 1.0f / powf(eff_theta, 2.0f * i / rope);
                float cv = cosf(key_pos * freq), sv = sinf(key_pos * freq);
                int j = i + rope / 2;
                float a = k_pe_f32[i], b0 = k_pe_f32[j];
                k_pe_f32[i] = a * cv - b0 * sv;
                k_pe_f32[j] = a * sv + b0 * cv;
            }

            // Split kv_out: k_nope[nH, nope] then v[nH, v_head_dim]
            int v_offset = nH * nope;

            // Compute Q·K scores for all heads
            #pragma omp parallel for
            for (int h = 0; h < nH; h++) {
                float* qh = qb + h * hd;
                const float* k_nope_h = kv_out + h * nope;
                float score = 0.0f;

                // nope dot product: Q_nope[h] · K_nope[h]
                int d = 0;
#ifndef __aarch64__
                __m256 sv_sum = _mm256_setzero_ps();
                for (; d + 8 <= nope; d += 8) {
                    __m256 qv = _mm256_loadu_ps(qh + d);
                    __m256 kv = _mm256_loadu_ps(k_nope_h + d);
                    sv_sum = _mm256_fmadd_ps(qv, kv, sv_sum);
                }
                score = hsum_ps(sv_sum);
#endif
                for (; d < nope; d++) score += qh[d] * k_nope_h[d];

                // rope dot product: Q_pe[h] · K_pe (shared across heads)
                const float* q_pe = qh + nope;
#ifndef __aarch64__
                __m256 rv_sum = _mm256_setzero_ps();
                d = 0;
                for (; d + 8 <= rope; d += 8) {
                    __m256 qv = _mm256_loadu_ps(q_pe + d);
                    __m256 kv = _mm256_loadu_ps(k_pe_f32 + d);
                    rv_sum = _mm256_fmadd_ps(qv, kv, rv_sum);
                }
                score += hsum_ps(rv_sum);
#endif
                for (; d < rope; d++) score += q_pe[d] * k_pe_f32[d];

                scores[h * ring_len + s] = score * inv_sqrt_d;
            }
        }

        // ── Phase 2: Softmax ──
        #pragma omp parallel for
        for (int h = 0; h < nH; h++) {
            float* sh = scores + h * ring_len;
            float max_val = -1e9f;
            for (int s = 0; s < ring_len; s++)
                if (sh[s] > max_val) max_val = sh[s];
            float sum = 0.0f;
            for (int s = 0; s < ring_len; s++) {
                float e = expf(sh[s] - max_val);
                sh[s] = e;
                sum += e;
            }
            float inv = 1.0f / fmaxf(sum, 1e-10f);
            for (int s = 0; s < ring_len; s++) sh[s] *= inv;
        }

        // ── Phase 3: Weighted sum over V (recompute V from cache) ──
        for (int s = 0; s < ring_len; s++) {
            int key_pos = ring_start + s;
            if (key_pos > pos) break;
            int cache_pos = key_pos % max_seq_len;

            // Reload c_kv and recompute kv_b output for V
            const uint16_t* c_kv_fp16 = (const uint16_t*)m->compressed_kv_cache +
                ((size_t)layer * max_seq_len + cache_pos) * stride;
            // c_kv_f32 already hoisted before ring loop
            for (int i = 0; i < lora; i++)
                c_kv_f32[i] = fp16_to_fp32(c_kv_fp16[i]);

            if (w_kv_b && kv_b_dim > 0) {
                if (kv_b_row_scales)
                    matmul_f32_row_major_per_row(kv_b_rows, kv_b_dim, w_kv_b, c_kv_f32, kv_b_row_scales, kv_out);
                else
                    matmul_f32_row_major(kv_b_rows, kv_b_dim, w_kv_b, c_kv_f32, kv_b_scale, kv_out);
            }

            int v_offset = nH * nope;

            // Accumulate: output[h] += scores[h][s] * V[h]
            #pragma omp parallel for
            for (int h = 0; h < nH; h++) {
                float sc = scores[h * ring_len + s];
                if (sc == 0.0f) continue;
                const float* v_h = kv_out + v_offset + h * m->v_head_dim;
                float* out_h = out_b + h * hd;
#ifndef __aarch64__
                __m256 sv = _mm256_set1_ps(sc);
                int d = 0;
                for (; d + 8 <= m->v_head_dim; d += 8) {
                    __m256 ov = _mm256_loadu_ps(out_h + d);
                    __m256 vv = _mm256_loadu_ps(v_h + d);
                    ov = _mm256_fmadd_ps(sv, vv, ov);
                    _mm256_storeu_ps(out_h + d, ov);
                }
                for (; d < m->v_head_dim; d++)
                    out_h[d] += sc * v_h[d];
#else
                for (int d = 0; d < m->v_head_dim; d++)
                    out_h[d] += sc * v_h[d];
#endif
            }
        }
    }
}

// ─── MoE Forward: Router + Top-K Expert Dispatch + Shared Experts ───
// hidden_states: [B, hidden_dim], output: [B, hidden_dim]
// router_logits written to m->buf_router
static void atlas_moe_forward(
    AtlasModel* m,
    const float* hidden_states, float* output, int B,
    const float* x_norm,   // [B, H] normalized input (for FFN)
    int layer) {

    int H = m->hidden_dim;
    int n_experts = m->n_experts;
    int n_active = m->n_experts_active;
    int n_shared = m->n_shared_experts;
    int inter = m->inter_dim;

    const int* idx = m->layer_idx_cache.data() + layer * m->layer_stride;
    float* router_logits = m->buf_router;

    // ── 1. Router logits: router_logits = x_norm @ W_gate^T ──
    // W_gate is at idx[6 + 3*n_shared + 1] for MoE layers (fp16, ttype=2)
    int router_idx = 6 + 3 * n_shared + 1;
    auto& t_router = m->tensors[idx[router_idx]];
    if (t_router.ttype == 2 && t_router.data) {
        // fp16 weights: scalar matmul (router is small: 64 × H)
        const uint16_t* w16 = (const uint16_t*)t_router.data;
        for (int b = 0; b < B; b++) {
            for (int e = 0; e < n_experts; e++) {
                float sum = 0.0f;
                for (int d = 0; d < H; d++)
                    sum += x_norm[b * H + d] * fp16_to_fp32(w16[e * H + d]);
                router_logits[b * n_experts + e] = sum;
            }
        }
    } else if (t_router.ttype == 3 && t_router.data) {
        // int8 weights with shared scale
        uint16_t sr; memcpy(&sr, t_router.data, 2);
        float rscale = fp16_to_fp32(sr);
        const int8_t* w8 = (const int8_t*)(t_router.data + 2);
        int w_dim = t3_dim(t_router);
        for (int b = 0; b < B; b++) {
            for (int e = 0; e < n_experts; e++) {
                const int8_t* wrow = w8 + (int64_t)e * w_dim;
                float sum = 0.0f;
                int d = 0;
#ifndef __aarch64__
                __m256 vs = _mm256_setzero_ps();
                for (; d + 8 <= w_dim; d += 8) {
                    __m256 hv = _mm256_loadu_ps(x_norm + b * H + d);
                    __m128i w8v = _mm_loadl_epi64((const __m128i*)(wrow + d));
                    __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8v));
                    vs = _mm256_fmadd_ps(hv, wf, vs);
                }
                sum = hsum_ps(vs);
#endif
                for (; d < w_dim; d++)
                    sum += x_norm[b * H + d] * (float)wrow[d];
                router_logits[b * n_experts + e] = sum / rscale;
            }
        }
    }

    // ── 2. Softmax + Top-K selection per batch ──
    // Store selected expert indices and weights
    // Allocate from attn_ws: x_safe [B·H] is dead post-expert-dispatch
    int moe_ws_off = B * H;
    int* selected = (int*)(m->attn_ws + moe_ws_off);
    float* expert_weights = m->attn_ws + moe_ws_off + B * 64;
    for (int b = 0; b < B; b++) {
        float* logits = router_logits + b * n_experts;
        // Softmax
        float max_logit = -1e30f;
        for (int e = 0; e < n_experts; e++)
            if (logits[e] > max_logit) max_logit = logits[e];
        float sum_exp = 0.0f;
        for (int e = 0; e < n_experts; e++) {
            logits[e] = expf(logits[e] - max_logit);
            sum_exp += logits[e];
        }
        float inv_sum = 1.0f / fmaxf(sum_exp, 1e-10f);
        for (int e = 0; e < n_experts; e++) logits[e] *= inv_sum;

        // Simple selection sort for top-K (n_active <= 8, so this is fast)
        int cnt = 0;
        // Copy logits for partial sort
        float tmp[256]; // max 256 experts
        int tmp_idx[256];
        for (int e = 0; e < n_experts; e++) { tmp[e] = logits[e]; tmp_idx[e] = e; }
        for (int k = 0; k < n_active && k < n_experts; k++) {
            int best = k;
            for (int e = k + 1; e < n_experts; e++)
                if (tmp[e] > tmp[best]) best = e;
            if (best != k) {
                float t = tmp[k]; tmp[k] = tmp[best]; tmp[best] = t;
                int ti = tmp_idx[k]; tmp_idx[k] = tmp_idx[best]; tmp_idx[best] = ti;
            }
            selected[b * 64 + cnt] = tmp_idx[k];
            expert_weights[b * 64 + cnt] = tmp[k];
            cnt++;
        }
    }

    // ── 3. Expert dispatch: Sparse FFN for active experts ──
    // ACCUMULATE into buf_hidden (NOT output) to avoid aliasing:
    // output == buf_gate (from caller), gate projection writes buf_gate[0..inter],
    // so accumulating into output would be overwritten by next expert's gate proj.
    memset(m->buf_hidden, 0, (size_t)B * H * sizeof(float));

    // Copy x_norm to scratch to avoid aliasing: x_norm == buf_act == ab.
    // For B>1, SiLU for b=0 overwrites buf_act[0..inter], corrupting x_norm for b=1+.
    // Reuse attn_ws — q_full is dead post-attention, safe for B*H scratch.
    float* x_safe = m->attn_ws;
    memcpy(x_safe, x_norm, (size_t)B * H * sizeof(float));

    // Process each (batch, selected-expert) pair
    for (int k = 0; k < n_active; k++) {
        for (int b = 0; b < B; b++) {
            int e = selected[b * 64 + k];
            float w = expert_weights[b * 64 + k];
            if (w == 0.0f) continue;

            // Look up gate/up/down tensors via repacked mapping
            int base = layer * n_experts * 3 + e * 3;
            int tidx_g = m->moe_expert_tidx[base + 0];
            int tidx_u = m->moe_expert_tidx[base + 1];
            int tidx_d = m->moe_expert_tidx[base + 2];
            if (tidx_g < 0 || tidx_u < 0 || tidx_d < 0) continue;
            auto& t_g = m->tensors[tidx_g];
            auto& t_u = m->tensors[tidx_u];
            auto& t_d = m->tensors[tidx_d];

            const float* xb = x_safe + b * H;
            float* gb = m->buf_gate;   // [inter] — reused per (b,k)
            float* ub = m->buf_up;     // [inter]
            float* ab = m->buf_ffn_f32;    // [inter] — SiLU(gate)*up result (FP32 safety buffer)

            // ── gate projection: xb[H] → gb[inter] ──
            if (t_g.ttype == 3) {
                int8_t* gw; int32_t* grs; int g_rows, g_dim; float g_scale;
                get_i8(t_g, gw, grs, g_rows, g_dim, g_scale);
                memcpy(ab, xb, H * sizeof(float));
                memset(ab + H, 0, (g_dim - H) * sizeof(float));
                matmul_f32_row_major(g_rows, g_dim, gw, ab, g_scale, gb);
            } else if (t_g.ttype == 11) {
                const uint16_t* rs = (const uint16_t*)(t_g.data);
                const int8_t* gw = (int8_t*)(t_g.data + t_g.row_dim * 2);
                int dim11 = t3_dim(t_g);
                memcpy(ab, xb, H * sizeof(float));
                memset(ab + H, 0, (dim11 - H) * sizeof(float));
                matmul_f32_row_major_per_row(t_g.row_dim, dim11, gw, ab, rs, gb);
            } else if (t_g.ttype == 8) {
                uint16_t s16; memcpy(&s16, t_g.data, 2); float sc = fp16_to_fp32(s16);
                int rows = t_g.row_dim, pw = t_g.packed_cols;
                int dim_w = pw * 2;
                memcpy(ab, xb, H * sizeof(float));
                memset(ab + H, 0, (dim_w - H) * sizeof(float));
                const uint8_t* gw = t_g.data + 2;
                int rows_packed = rows / 4;
                for (int ur = 0; ur < rows_packed; ur++) {
                    const uint8_t* gw4 = gw + ur * 4 * pw;
                    for (int sub = 0; sub < 4; sub++) {
                        const uint8_t* wg = gw4 + sub * pw;
                        float s = 0.0f;
                        int c = 0;
#ifndef __aarch64__
                        {
                        const __m128i mask_low = _mm_set1_epi8(0x0F);
                        const __m128i xor8 = _mm_set1_epi8(8);
                        __m256 gs = _mm256_setzero_ps();
                        for (; c + 8 <= dim_w; c += 8) {
                            __m256 af = _mm256_loadu_ps(ab + c);
                            uint32_t p4; memcpy(&p4, wg + c / 2, 4);
                            __m128i nib = _mm_cvtsi32_si128((int)p4);
                            __m128i lo = _mm_and_si128(nib, mask_low);
                            __m128i hi = _mm_and_si128(_mm_srli_epi16(nib, 4), mask_low);
                            __m128i w8 = _mm_unpacklo_epi8(lo, hi);
                            __m128i ws = _mm_sub_epi8(_mm_xor_si128(w8, xor8), xor8);
                            __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ws));
                            gs = _mm256_fmadd_ps(af, wf, gs);
                        }
                        s = hsum_ps(gs);
                        }
#endif
                        for (; c < dim_w; c++) {
                            int8_t wv = (c & 1) ? (int8_t)((((wg[c / 2] >> 4) & 0x0F) ^ 8) - 8)
                                                : (int8_t)(((wg[c / 2] & 0x0F) ^ 8) - 8);
                            s += ab[c] * (float)wv;
                        }
                        gb[ur * 4 + sub] = s / sc;
                    }
                }
            }

            // ── up projection: xb[H] → ub[inter] ──
            if (t_u.ttype == 3) {
                int8_t* uw; int32_t* urs; int u_rows, u_dim; float u_scale;
                get_i8(t_u, uw, urs, u_rows, u_dim, u_scale);
                memcpy(ab, xb, H * sizeof(float));
                memset(ab + H, 0, (u_dim - H) * sizeof(float));
                matmul_f32_row_major(u_rows, u_dim, uw, ab, u_scale, ub);
            } else if (t_u.ttype == 11) {
                const uint16_t* rs = (const uint16_t*)(t_u.data);
                const int8_t* uw = (int8_t*)(t_u.data + t_u.row_dim * 2);
                int dim11 = t3_dim(t_u);
                memcpy(ab, xb, H * sizeof(float));
                memset(ab + H, 0, (dim11 - H) * sizeof(float));
                matmul_f32_row_major_per_row(t_u.row_dim, dim11, uw, ab, rs, ub);
            } else if (t_u.ttype == 8) {
                uint16_t s16; memcpy(&s16, t_u.data, 2); float sc = fp16_to_fp32(s16);
                int rows = t_u.row_dim, pw = t_u.packed_cols;
                int dim_w = pw * 2;
                memcpy(ab, xb, H * sizeof(float));
                memset(ab + H, 0, (dim_w - H) * sizeof(float));
                const uint8_t* uw = t_u.data + 2;
                int rows_packed = rows / 4;
                for (int ur = 0; ur < rows_packed; ur++) {
                    const uint8_t* uw4 = uw + ur * 4 * pw;
                    for (int sub = 0; sub < 4; sub++) {
                        const uint8_t* w = uw4 + sub * pw;
                        float s = 0.0f;
                        int c = 0;
#ifndef __aarch64__
                        {
                        const __m128i mask_low = _mm_set1_epi8(0x0F);
                        const __m128i xor8 = _mm_set1_epi8(8);
                        __m256 us = _mm256_setzero_ps();
                        for (; c + 8 <= dim_w; c += 8) {
                            __m256 af = _mm256_loadu_ps(ab + c);
                            uint32_t p4; memcpy(&p4, w + c / 2, 4);
                            __m128i nib = _mm_cvtsi32_si128((int)p4);
                            __m128i lo = _mm_and_si128(nib, mask_low);
                            __m128i hi = _mm_and_si128(_mm_srli_epi16(nib, 4), mask_low);
                            __m128i w8 = _mm_unpacklo_epi8(lo, hi);
                            __m128i ws = _mm_sub_epi8(_mm_xor_si128(w8, xor8), xor8);
                            __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ws));
                            us = _mm256_fmadd_ps(af, wf, us);
                        }
                        s = hsum_ps(us);
                        }
#endif
                        for (; c < dim_w; c++) {
                            int8_t wv = (c & 1) ? (int8_t)((((w[c / 2] >> 4) & 0x0F) ^ 8) - 8)
                                                : (int8_t)(((w[c / 2] & 0x0F) ^ 8) - 8);
                            s += ab[c] * (float)wv;
                        }
                        ub[ur * 4 + sub] = s / sc;
                    }
                }
            }

            // ── SiLU(gate) * up → ab ──
            int inter_e = t3_dim(t_d);
            for (int i = 0; i < inter; i++)
                ab[i] = gate_activation(gb[i], false) * ub[i];
            for (int i = inter; i < inter_e; i++) ab[i] = 0.0f;

            // ── down projection: ab[inter] → accumulate into buf_hidden[b*H] ──
            float* ob = m->buf_hidden + b * H;
            float* tmp_down = m->buf_up + b * H;  // buf_up freed after SiLU
            if (t_d.ttype == 3) {
                int8_t* dw; int32_t* drs; int d_rows, d_dim; float d_scale;
                get_i8(t_d, dw, drs, d_rows, d_dim, d_scale);
                matmul_f32_row_major(d_rows, d_dim, dw, ab, d_scale, tmp_down);
                for (int i = 0; i < H; i++) ob[i] += w * tmp_down[i];
            } else if (t_d.ttype == 11) {
                const uint16_t* rs = (const uint16_t*)(t_d.data);
                const int8_t* dw = (int8_t*)(t_d.data + t_d.row_dim * 2);
                int dim11 = t3_dim(t_d);
                matmul_f32_row_major_per_row(t_d.row_dim, dim11, dw, ab, rs, tmp_down);
                for (int i = 0; i < H; i++) ob[i] += w * tmp_down[i];
            } else if (t_d.ttype == 8) {
                uint16_t s16; memcpy(&s16, t_d.data, 2); float sc = fp16_to_fp32(s16);
                int rows = t_d.row_dim, pw = t_d.packed_cols;
                int dim_w = pw * 2;
                const uint8_t* dw = t_d.data + 2;
                int rows_packed = rows / 4;
                for (int ur = 0; ur < rows_packed; ur++) {
                    const uint8_t* dw4 = dw + ur * 4 * pw;
                    for (int sub = 0; sub < 4; sub++) {
                        const uint8_t* wd = dw4 + sub * pw;
                        float s = 0.0f;
                        int c = 0;
#ifndef __aarch64__
                        {
                        const __m128i mask_low = _mm_set1_epi8(0x0F);
                        const __m128i xor8 = _mm_set1_epi8(8);
                        __m256 ds = _mm256_setzero_ps();
                        for (; c + 8 <= dim_w; c += 8) {
                            __m256 af = _mm256_loadu_ps(ab + c);
                            uint32_t p4; memcpy(&p4, wd + c / 2, 4);
                            __m128i nib = _mm_cvtsi32_si128((int)p4);
                            __m128i lo = _mm_and_si128(nib, mask_low);
                            __m128i hi = _mm_and_si128(_mm_srli_epi16(nib, 4), mask_low);
                            __m128i w8 = _mm_unpacklo_epi8(lo, hi);
                            __m128i ws = _mm_sub_epi8(_mm_xor_si128(w8, xor8), xor8);
                            __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ws));
                            ds = _mm256_fmadd_ps(af, wf, ds);
                        }
                        s = hsum_ps(ds);
                        }
#endif
                        for (; c < dim_w; c++) {
                            int8_t wv = (c & 1) ? (int8_t)((((wd[c / 2] >> 4) & 0x0F) ^ 8) - 8)
                                                : (int8_t)(((wd[c / 2] & 0x0F) ^ 8) - 8);
                            s += ab[c] * (float)wv;
                        }
                        tmp_down[ur * 4 + sub] = s / sc;
                    }
                }
                for (int i = 0; i < H; i++) ob[i] += w * tmp_down[i];
            }
        }
    }

    // ── 4. Shared experts (always active): gate + up → SiLU → down ──
    // Shared expert tensors at idx[6 + s_idx*3 .. 8 + s_idx*3] per expert
    for (int s_idx = 0; s_idx < n_shared; s_idx++) {
        int base = 6 + s_idx * 3;
        auto& t_sg = m->tensors[idx[base + 0]];
        auto& t_su = m->tensors[idx[base + 1]];
        auto& t_sd = m->tensors[idx[base + 2]];

        int sg_rows = 0, su_rows = 0, sd_rows = 0;
        int sg_dim = 0, su_dim = 0, sd_dim = 0;
        int8_t* sg_w = nullptr; int32_t* sg_rs = nullptr; float sg_scale = 1.0f;
        int8_t* su_w = nullptr; int32_t* su_rs = nullptr; float su_scale = 1.0f;
        int8_t* sd_w = nullptr; int32_t* sd_rs = nullptr; float sd_scale = 1.0f;

        // Use row-major matmul for shared experts
        if (t_sg.ttype == 11) {
            const uint16_t* rs_fp16 = (const uint16_t*)(t_sg.data);
            const int8_t* w = (int8_t*)(t_sg.data + t_sg.row_dim * 2);
            int dim11 = t3_dim(t_sg);
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * dim11, x_norm + b * H, H * sizeof(float));
                memset(m->buf_act + b * dim11 + H, 0, (dim11 - H) * sizeof(float));
            }
            matmul_f32_row_major_per_row_batched(t_sg.row_dim, dim11, w, m->buf_act, dim11, rs_fp16, m->buf_gate, B);
        } else if (t_sg.ttype == 3) {
            get_i8(t_sg, sg_w, sg_rs, sg_rows, sg_dim, sg_scale);
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * sg_dim, x_norm + b * H, H * sizeof(float));
                memset(m->buf_act + b * sg_dim + H, 0, (sg_dim - H) * sizeof(float));
            }
            matmul_f32_row_major_batched(sg_rows, sg_dim, sg_w, m->buf_act, sg_dim, sg_scale, m->buf_gate, B);
        } else if (t_sg.ttype == 8) {
            for (int b = 0; b < B; b++) {
                int dim = t3_dim(t_sg);
                memcpy(m->buf_act + b * dim, x_norm + b * H, H * sizeof(float));
                memset(m->buf_act + b * dim + H, 0, (dim - H) * sizeof(float));
            }
            uint16_t s16; memcpy(&s16, t_sg.data, 2); float sc = fp16_to_fp32(s16);
            int rows = t_sg.row_dim, pw = t_sg.packed_cols;
            int dim_w = pw * 2;
            const uint8_t* gw = t_sg.data + 2;
            int rows_packed = rows / 4;
            for (int ur = 0; ur < rows_packed; ur++) {
                const uint8_t* gw4 = gw + ur * 4 * pw;
                for (int b = 0; b < B; b++) {
                    const float* a = m->buf_ffn_f32 + b * dim_w;
                    for (int sub = 0; sub < 4; sub++) {
                        const uint8_t* wg = gw4 + sub * pw;
                        float s = 0.0f;
                        int c = 0;
#ifndef __aarch64__
                        {
                        const __m128i mask_low = _mm_set1_epi8(0x0F);
                        const __m128i xor8 = _mm_set1_epi8(8);
                        __m256 gs = _mm256_setzero_ps();
                        for (; c + 8 <= dim_w; c += 8) {
                            __m256 af = _mm256_loadu_ps(a + c);
                            uint32_t p4; memcpy(&p4, wg + c / 2, 4);
                            __m128i nib = _mm_cvtsi32_si128((int)p4);
                            __m128i lo = _mm_and_si128(nib, mask_low);
                            __m128i hi = _mm_and_si128(_mm_srli_epi16(nib, 4), mask_low);
                            __m128i w8 = _mm_unpacklo_epi8(lo, hi);
                            __m128i ws = _mm_sub_epi8(_mm_xor_si128(w8, xor8), xor8);
                            __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ws));
                            gs = _mm256_fmadd_ps(af, wf, gs);
                        }
                        s = hsum_ps(gs);
                        }
#endif
                        for (; c < dim_w; c++) {
                            int8_t wv = (c & 1) ? (int8_t)((((wg[c / 2] >> 4) & 0x0F) ^ 8) - 8)
                                                : (int8_t)(((wg[c / 2] & 0x0F) ^ 8) - 8);
                            s += a[c] * (float)wv;
                        }
                        m->buf_gate[b * rows + ur * 4 + sub] = s / sc;
                    }
                }
            }
        }

        // up projection
        if (t_su.ttype == 11) {
            const uint16_t* rs_fp16 = (const uint16_t*)(t_su.data);
            const int8_t* w = (int8_t*)(t_su.data + t_su.row_dim * 2);
            int dim11 = t3_dim(t_su);
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * dim11, x_norm + b * H, H * sizeof(float));
                memset(m->buf_act + b * dim11 + H, 0, (dim11 - H) * sizeof(float));
            }
            matmul_f32_row_major_per_row_batched(t_su.row_dim, dim11, w, m->buf_act, dim11, rs_fp16, m->buf_up, B);
        } else if (t_su.ttype == 3) {
            get_i8(t_su, su_w, su_rs, su_rows, su_dim, su_scale);
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * su_dim, x_norm + b * H, H * sizeof(float));
                memset(m->buf_act + b * su_dim + H, 0, (su_dim - H) * sizeof(float));
            }
            matmul_f32_row_major_batched(su_rows, su_dim, su_w, m->buf_act, su_dim, su_scale, m->buf_up, B);
        } else if (t_su.ttype == 8) {
            for (int b = 0; b < B; b++) {
                int dim = t3_dim(t_su);
                memcpy(m->buf_act + b * dim, x_norm + b * H, H * sizeof(float));
                memset(m->buf_act + b * dim + H, 0, (dim - H) * sizeof(float));
            }
            uint16_t s16; memcpy(&s16, t_su.data, 2); float sc = fp16_to_fp32(s16);
            int rows = t_su.row_dim, pw = t_su.packed_cols;
            int dim_w = pw * 2;
            const uint8_t* uw = t_su.data + 2;
            int rows_packed = rows / 4;
            for (int ur = 0; ur < rows_packed; ur++) {
                const uint8_t* uw4 = uw + ur * 4 * pw;
                for (int b = 0; b < B; b++) {
                    const float* a = m->buf_act + b * dim_w;
                    for (int sub = 0; sub < 4; sub++) {
                        const uint8_t* w = uw4 + sub * pw;
                        float s = 0.0f;
                        int c = 0;
#ifndef __aarch64__
                        {
                        const __m128i mask_low = _mm_set1_epi8(0x0F);
                        const __m128i xor8 = _mm_set1_epi8(8);
                        __m256 us = _mm256_setzero_ps();
                        for (; c + 8 <= dim_w; c += 8) {
                            __m256 af = _mm256_loadu_ps(a + c);
                            uint32_t p4; memcpy(&p4, w + c / 2, 4);
                            __m128i nib = _mm_cvtsi32_si128((int)p4);
                            __m128i lo = _mm_and_si128(nib, mask_low);
                            __m128i hi = _mm_and_si128(_mm_srli_epi16(nib, 4), mask_low);
                            __m128i w8 = _mm_unpacklo_epi8(lo, hi);
                            __m128i ws = _mm_sub_epi8(_mm_xor_si128(w8, xor8), xor8);
                            __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ws));
                            us = _mm256_fmadd_ps(af, wf, us);
                        }
                        s = hsum_ps(us);
                        }
#endif
                        for (; c < dim_w; c++) {
                            int8_t wv = (c & 1) ? (int8_t)((((w[c / 2] >> 4) & 0x0F) ^ 8) - 8)
                                                : (int8_t)(((w[c / 2] & 0x0F) ^ 8) - 8);
                            s += a[c] * (float)wv;
                        }
                        m->buf_up[b * rows + ur * 4 + sub] = s / sc;
                    }
                }
            }
        }

        // SiLU(gate) * up → tmp in buf_ffn_f32
        int inter_sd = t3_dim(t_sd);
        for (int b = 0; b < B; b++) {
            float* tmp = m->buf_ffn_f32 + b * inter_sd;
            for (int i = 0; i < inter; i++)
                tmp[i] = gate_activation(m->buf_gate[b * inter + i], false) * m->buf_up[b * inter + i];
            for (int i = inter; i < inter_sd; i++) tmp[i] = 0.0f;
        }

        // down projection: tmp → accumulate into buf_hidden (routed expert accumulator)
        float* tmp_shared = m->buf_gate;  // buf_gate free after SiLU, use as scratch
        if (t_sd.ttype == 11) {
            const uint16_t* rs_fp16 = (const uint16_t*)(t_sd.data);
            const int8_t* w = (int8_t*)(t_sd.data + t_sd.row_dim * 2);
            int dim11 = t3_dim(t_sd);
            matmul_f32_row_major_per_row_batched(t_sd.row_dim, dim11, w, m->buf_ffn_f32, inter_sd, rs_fp16, tmp_shared, B);
            for (int i = 0; i < B * H; i++) m->buf_hidden[i] += tmp_shared[i];
        } else if (t_sd.ttype == 3) {
            get_i8(t_sd, sd_w, sd_rs, sd_rows, sd_dim, sd_scale);
            matmul_f32_row_major_batched(sd_rows, sd_dim, sd_w, m->buf_ffn_f32, inter_sd, sd_scale, tmp_shared, B);
            for (int i = 0; i < B * H; i++) m->buf_hidden[i] += tmp_shared[i];
        } else if (t_sd.ttype == 8) {
            uint16_t s16; memcpy(&s16, t_sd.data, 2); float sc = fp16_to_fp32(s16);
            int rows = t_sd.row_dim, pw = t_sd.packed_cols;
            int dim_w = pw * 2;
            const uint8_t* dw = t_sd.data + 2;
            int rows_packed = rows / 4;
            for (int ur = 0; ur < rows_packed; ur++) {
                const uint8_t* dw4 = dw + ur * 4 * pw;
                for (int b = 0; b < B; b++) {
                    const float* a = m->buf_ffn_f32 + b * dim_w;
                    for (int sub = 0; sub < 4; sub++) {
                        const uint8_t* w = dw4 + sub * pw;
                        float s = 0.0f;
                        int c = 0;
#ifndef __aarch64__
                        {
                        const __m128i mask_low = _mm_set1_epi8(0x0F);
                        const __m128i xor8 = _mm_set1_epi8(8);
                        __m256 ds = _mm256_setzero_ps();
                        for (; c + 8 <= dim_w; c += 8) {
                            __m256 af = _mm256_loadu_ps(a + c);
                            uint32_t p4; memcpy(&p4, w + c / 2, 4);
                            __m128i nib = _mm_cvtsi32_si128((int)p4);
                            __m128i lo = _mm_and_si128(nib, mask_low);
                            __m128i hi = _mm_and_si128(_mm_srli_epi16(nib, 4), mask_low);
                            __m128i w8 = _mm_unpacklo_epi8(lo, hi);
                            __m128i ws = _mm_sub_epi8(_mm_xor_si128(w8, xor8), xor8);
                            __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ws));
                            ds = _mm256_fmadd_ps(af, wf, ds);
                        }
                        s = hsum_ps(ds);
                        }
#endif
                        for (; c < dim_w; c++) {
                            int8_t wv = (c & 1) ? (int8_t)((((w[c / 2] >> 4) & 0x0F) ^ 8) - 8)
                                                : (int8_t)(((w[c / 2] & 0x0F) ^ 8) - 8);
                            s += a[c] * (float)wv;
                        }
                        m->buf_gate[b * rows + ur * 4 + sub] = s / sc;
                    }
                }
            }
            for (int i = 0; i < B * H; i++) m->buf_hidden[i] += m->buf_gate[i];
        }
    }

    // ── 5. Copy accumulated result from buf_hidden to output ──
    memcpy(output, m->buf_hidden, (size_t)B * H * sizeof(float));
}

// ─── MLA Layer Wrapper: Complete transformer layer for DeepSeek-V2 ───
static void forward_layer_internal_mla(
    AtlasModel* m,
    const float* input, float* output, int B,
    const int* positions,
    int max_seq_len, int seq_now,
    int layer) {

    int H = m->hidden_dim;
    int nH = m->n_heads;
    int nope = m->qk_nope_head_dim;
    int rope = m->qk_rope_head_dim;
    int hd = nope + rope;
    int lora = m->kv_lora_rank;
    int qd = nH * hd;
    int inter = m->inter_dim;
    float theta = m->rope_theta;

    const int* idx = m->layer_idx_cache.data() + layer * m->layer_stride;

    #ifdef ATLAS_DEBUG_MODE
    fprintf(stderr, "[MLA_STEP] L=%d idx=[%d %d %d %d %d %d %d %d %d] attn_ws=%p buf_gate=%p buf_c_kv=%p\n",
            layer, idx[0], idx[1], idx[2], idx[3], idx[4], idx[5],
            idx[6], idx[7], idx[8],
            (void*)m->attn_ws, (void*)m->buf_gate, (void*)m->buf_c_kv);
    #endif

    #ifdef ATLAS_DEBUG_MODE
    fprintf(stderr, "[MLA_STEP] L=%d step1_rmsnorm t_ln1.data=%p\n", layer, (void*)m->tensors[idx[0]].data);
    #endif

    // ─── 1. Pre-attention RMSNorm ───
    auto& t_ln1 = m->tensors[idx[0]];
    float* x_norm = output;  // write normalized into output buffer
    {
        const uint8_t* w = t_ln1.data;
        for (int b = 0; b < B; b++) {
            const float* xb = input + b * H;
            float* nb = x_norm + b * H;
            float ss = 0.0f;
            for (int i = 0; i < H; i++) ss += xb[i] * xb[i];
            float rms = 1.0f / sqrtf(ss / H + 1e-6f);
            for (int i = 0; i < H; i++) {
                uint16_t w16; memcpy(&w16, w + i * 2, 2);
                nb[i] = xb[i] * rms * fp16_to_fp32(w16);
            }
        }
    }

    #ifdef ATLAS_DEBUG_MODE
    fprintf(stderr, "[MLA_STEP] L=%d step2_post_rmsnorm\n", layer);
    #endif

    // ─── 2. Q projection: V3 (q_a → layernorm → q_b) or V2-Lite (single q_proj) ───
    float* q_full = m->attn_ws;  // [B * qd]
    int ns_fwd = m->n_shared_experts > 0 ? m->n_shared_experts : 1;
    int kv_ln_fwd = 6 + 3 * ns_fwd;
    if (m->q_lora_rank > 0) {
        // V3 path: x_norm → q_a_proj → rmsnorm → q_b_proj → q_full
        float* q_comp = m->buf_hidden;  // [B, q_lora_rank] scratch
        // Step 2a: q_a projection: x_norm[H] → q_comp[q_lora_rank]
        {
            auto& t_qa = m->tensors[idx[1]];  // q_a_proj
            if (t_qa.ttype == 3 && m->use_f32_matmul) {
                int8_t* w; int32_t* rs; int rows, dim; float scale;
                get_i8(t_qa, w, rs, rows, dim, scale);
                for (int b = 0; b < B; b++) {
                    memcpy(m->buf_act + b * dim, x_norm + b * H, H * sizeof(float));
                    memset(m->buf_act + b * dim + H, 0, (dim - H) * sizeof(float));
                }
                matmul_f32_row_major_batched(rows, dim, w, m->buf_act, dim, scale, q_comp, B);
            } else if (t_qa.ttype == 11 && m->use_f32_matmul) {
                const uint16_t* rs_fp16 = (const uint16_t*)(t_qa.data);
                const int8_t* w = (int8_t*)(t_qa.data + t_qa.row_dim * 2);
                int rows = t_qa.row_dim;
                int dim = t3_dim(t_qa);
                for (int b = 0; b < B; b++) {
                    memcpy(m->buf_act + b * dim, x_norm + b * H, H * sizeof(float));
                    memset(m->buf_act + b * dim + H, 0, (dim - H) * sizeof(float));
                }
                matmul_f32_row_major_per_row_batched(rows, dim, w, m->buf_act, dim, rs_fp16, q_comp, B);
            }
        }
        // Step 2b: q_a_layernorm: q_comp[B, q_lora_rank] → q_comp_norm
        {
            auto& t_qa_ln = m->tensors[idx[kv_ln_fwd + 3]];  // q_a_layernorm
            if (t_qa_ln.data) {
                for (int b = 0; b < B; b++) {
                    float ss = 0.0f;
                    for (int i = 0; i < m->q_lora_rank; i++) ss += q_comp[b * m->q_lora_rank + i] * q_comp[b * m->q_lora_rank + i];
                    float rms = 1.0f / sqrtf(ss / m->q_lora_rank + 1e-6f);
                    for (int i = 0; i < m->q_lora_rank; i++) {
                        uint16_t w16; memcpy(&w16, t_qa_ln.data + i * 2, 2);
                        q_comp[b * m->q_lora_rank + i] *= rms * fp16_to_fp32(w16);
                    }
                }
            }
        }
        // Step 2c: q_b projection: q_comp[q_lora_rank] → q_full[nH * qk_head_dim]
        {
            auto& t_qb = m->tensors[idx[kv_ln_fwd + 2]];  // q_b_proj
            if (t_qb.ttype == 3 && m->use_f32_matmul) {
                int8_t* w; int32_t* rs; int rows, dim; float scale;
                get_i8(t_qb, w, rs, rows, dim, scale);
                for (int b = 0; b < B; b++) {
                    memcpy(m->buf_act + b * dim, q_comp + b * m->q_lora_rank, m->q_lora_rank * sizeof(float));
                    memset(m->buf_act + b * dim + m->q_lora_rank, 0, (dim - m->q_lora_rank) * sizeof(float));
                }
                matmul_f32_row_major_batched(rows, dim, w, m->buf_act, dim, scale, q_full, B);
            } else if (t_qb.ttype == 11 && m->use_f32_matmul) {
                const uint16_t* rs_fp16 = (const uint16_t*)(t_qb.data);
                const int8_t* w = (int8_t*)(t_qb.data + t_qb.row_dim * 2);
                int rows = t_qb.row_dim;
                int dim = t3_dim(t_qb);
                for (int b = 0; b < B; b++) {
                    memcpy(m->buf_act + b * dim, q_comp + b * m->q_lora_rank, m->q_lora_rank * sizeof(float));
                    memset(m->buf_act + b * dim + m->q_lora_rank, 0, (dim - m->q_lora_rank) * sizeof(float));
                }
                matmul_f32_row_major_per_row_batched(rows, dim, w, m->buf_act, dim, rs_fp16, q_full, B);
            }
        }
    } else {
        // V2-Lite path: single q_proj
        auto& tq = m->tensors[idx[1]];
        if (tq.ttype == 3 && m->use_f32_matmul) {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(tq, w, rs, rows, dim, scale);
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * dim, x_norm + b * H, H * sizeof(float));
                memset(m->buf_act + b * dim + H, 0, (dim - H) * sizeof(float));
            }
            matmul_f32_row_major_batched(rows, dim, w, m->buf_act, dim, scale, q_full, B);
        } else if (tq.ttype == 11 && m->use_f32_matmul) {
            const uint16_t* rs_fp16 = (const uint16_t*)(tq.data);
            const int8_t* w = (int8_t*)(tq.data + tq.row_dim * 2);
            int rows = tq.row_dim;
            int dim = t3_dim(tq);
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * dim, x_norm + b * H, H * sizeof(float));
                memset(m->buf_act + b * dim + H, 0, (dim - H) * sizeof(float));
            }
            matmul_f32_row_major_per_row_batched(rows, dim, w, m->buf_act, dim, rs_fp16, q_full, B);
        }
    }

    #ifdef ATLAS_DEBUG_MODE
    fprintf(stderr, "[MLA_STEP] L=%d step3_post_q\n", layer);
    #endif

    // ─── 3. KV compression: x_norm → W_kv_a → [B, lora + rope] (row-major) ───
    float* kv_comp = m->buf_hidden;  // [B, lora + rope]
    {
        auto& t_kv_a = m->tensors[idx[2]];
        if (t_kv_a.ttype == 3 && m->use_f32_matmul) {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(t_kv_a, w, rs, rows, dim, scale);
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * dim, x_norm + b * H, H * sizeof(float));
                memset(m->buf_act + b * dim + H, 0, (dim - H) * sizeof(float));
            }
            matmul_f32_row_major_batched(rows, dim, w, m->buf_act, dim, scale, kv_comp, B);
        } else if (t_kv_a.ttype == 11 && m->use_f32_matmul) {
            const uint16_t* rs_fp16 = (const uint16_t*)(t_kv_a.data);
            const int8_t* w = (int8_t*)(t_kv_a.data + t_kv_a.row_dim * 2);
            int rows = t_kv_a.row_dim;
            int dim = t3_dim(t_kv_a);
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * dim, x_norm + b * H, H * sizeof(float));
                memset(m->buf_act + b * dim + H, 0, (dim - H) * sizeof(float));
            }
            matmul_f32_row_major_per_row_batched(rows, dim, w, m->buf_act, dim, rs_fp16, kv_comp, B);
        }
    }

    // ─── 3b. Extract k_pe from kv_comp BEFORE layernorm (layernorm only touches c_kv portion) ───
    size_t qd_sz_mla = (size_t)B * nH * (m->qk_nope_head_dim + m->qk_rope_head_dim);
    float* k_pe_all = m->attn_ws + 2 * qd_sz_mla;
    for (int b = 0; b < B; b++) {
        const float* src = kv_comp + b * (lora + rope) + lora;
        float* dst = k_pe_all + b * rope;
        for (int i = 0; i < rope; i++) dst[i] = src[i];
    }

    // ─── 3c. Split kv_comp: apply kv_a_layernorm to c_kv portion → c_kv_all ───
    float* c_kv_all = m->buf_c_kv;   // [B, lora]
    {
        int ns = m->n_shared_experts > 0 ? m->n_shared_experts : 1;
        auto& t_kvn = m->tensors[idx[6 + 3 * ns]];
        const uint8_t* lnw = t_kvn.data;
        for (int b = 0; b < B; b++) {
            float ss = 0.0f;
            for (int i = 0; i < lora; i++) {
                float v = kv_comp[b * (lora + rope) + i];
                ss += v * v;
            }
            float rms = 1.0f / sqrtf(ss / lora + 1e-6f);
            for (int i = 0; i < lora; i++) {
                uint16_t w16; memcpy(&w16, lnw + i * 2, 2);
                c_kv_all[b * lora + i] = kv_comp[b * (lora + rope) + i] * rms * fp16_to_fp32(w16);
            }
        }
    }

    // ─── 4. Apply RoPE to Q (q_pe portion only) ───
    // Use same NTK scaling formula as K RoPE in atlas_attention_mla
    float ctx_scale = m->base_seq_len > 0 ? (float)max_seq_len / (float)m->base_seq_len : 1.0f;
    if (ctx_scale < 1.0f) ctx_scale = 1.0f;
    float total_scale = m->rope_scale;
    if (ctx_scale > 1.001f) total_scale *= ctx_scale;
    float eff_theta = theta;
    if (total_scale > 1.001f)
        eff_theta *= powf(total_scale, (float)rope / (float)(rope - 2));
    for (int b = 0; b < B; b++) {
        int pos = positions[b];
        float* qb = q_full + b * qd;
        for (int h = 0; h < nH; h++) {
            float* qh = qb + h * hd;
            float* q_pe = qh + nope;
            for (int i = 0; i < rope / 2; i++) {
                float freq = 1.0f / powf(eff_theta, 2.0f * i / rope);
                float cv = cosf(pos * freq), sv = sinf(pos * freq);
                int j = i + rope / 2;
                float a = q_pe[i], b0 = q_pe[j];
                q_pe[i] = a * cv - b0 * sv;
                q_pe[j] = a * sv + b0 * cv;
            }
        }
    }

    // ─── 5. MLA Attention ───
    #ifdef ATLAS_DEBUG_MODE
    fprintf(stderr, "[MLA_STEP] L=%d step5_pre_attn\n", layer);
    #endif
    float* attn_out = m->attn_ws + B * qd;  // use second half of attn_ws
    atlas_attention_mla(m, q_full, attn_out,
        c_kv_all, k_pe_all,
        layer, positions, max_seq_len, seq_now, B);
    #ifdef ATLAS_DEBUG_MODE
    fprintf(stderr, "[MLA_STEP] L=%d step5_post_attn\n", layer);
    #endif

    // ─── 6. O projection: V3 gated (compress → bias → expand) or V2-Lite (single) ───
    if (m->has_gated_o_proj) {
        // V3 path: attn_out → W_gate → bias → W_up → output
        // Step 6a: Extract V from attn_out (same as V2-Lite: first v_head_dim of each head)
        int v_out_dim = nH * m->v_head_dim;
        for (int b = 0; b < B; b++) {
            const float* src = attn_out + b * qd;
            float* dst = m->buf_act + b * v_out_dim;
            for (int h = 0; h < nH; h++)
                memcpy(dst + h * m->v_head_dim, src + h * hd, m->v_head_dim * sizeof(float));
        }
        // Step 6b: Gate projection: v_out[v_out_dim] → latent[kv_lora+qk_rope]
        int latent_dim = m->kv_lora_rank + m->qk_rope_head_dim;
        {
            auto& t_og = m->tensors[idx[4]];  // o_proj.0 (gate)
            if (t_og.ttype == 3 && m->use_f32_matmul) {
                int8_t* w; int32_t* rs; int rows, dim; float scale;
                get_i8(t_og, w, rs, rows, dim, scale);
                for (int b = 0; b < B; b++) {
                    int copy_dim = v_out_dim < dim ? v_out_dim : dim;
                    memcpy(m->buf_up + b * dim, m->buf_act + b * v_out_dim, copy_dim * sizeof(float));
                    if (dim > copy_dim) memset(m->buf_up + b * dim + copy_dim, 0, (dim - copy_dim) * sizeof(float));
                }
                matmul_f32_row_major_batched(rows, dim, w, m->buf_up, dim, scale, m->buf_gate, B);
            } else if (t_og.ttype == 11 && m->use_f32_matmul) {
                const uint16_t* rs_fp16 = (const uint16_t*)(t_og.data);
                const int8_t* w = (int8_t*)(t_og.data + t_og.row_dim * 2);
                int rows = t_og.row_dim;
                int dim = t3_dim(t_og);
                for (int b = 0; b < B; b++) {
                    int copy_dim = v_out_dim < dim ? v_out_dim : dim;
                    memcpy(m->buf_up + b * dim, m->buf_act + b * v_out_dim, copy_dim * sizeof(float));
                    if (dim > copy_dim) memset(m->buf_up + b * dim + copy_dim, 0, (dim - copy_dim) * sizeof(float));
                }
                matmul_f32_row_major_per_row_batched(rows, dim, w, m->buf_up, dim, rs_fp16, m->buf_gate, B);
            }
        }
        // Step 6c: Add bias (o_proj.1)
        {
            auto& t_ob = m->tensors[idx[kv_ln_fwd + 4]];  // o_proj.1 (bias)
            if (t_ob.data) {
                const uint16_t* bias16 = (const uint16_t*)t_ob.data;
                for (int b = 0; b < B; b++)
                    for (int i = 0; i < latent_dim; i++)
                        m->buf_gate[b * latent_dim + i] += fp16_to_fp32(bias16[i]);
            }
        }
        // Step 6d: Expand: latent[latent_dim] → output[H]
        {
            auto& t_ou = m->tensors[idx[kv_ln_fwd + 5]];  // o_proj.2 (up)
            if (t_ou.ttype == 3 && m->use_f32_matmul) {
                int8_t* w; int32_t* rs; int rows, dim; float scale;
                get_i8(t_ou, w, rs, rows, dim, scale);
                for (int b = 0; b < B; b++) {
                    memcpy(m->buf_up + b * dim, m->buf_gate + b * latent_dim, latent_dim * sizeof(float));
                    if (dim > latent_dim) memset(m->buf_up + b * dim + latent_dim, 0, (dim - latent_dim) * sizeof(float));
                }
                matmul_f32_row_major_batched(rows, dim, w, m->buf_up, dim, scale, m->buf_gate, B);
            } else if (t_ou.ttype == 11 && m->use_f32_matmul) {
                const uint16_t* rs_fp16 = (const uint16_t*)(t_ou.data);
                const int8_t* w = (int8_t*)(t_ou.data + t_ou.row_dim * 2);
                int rows = t_ou.row_dim;
                int dim = t3_dim(t_ou);
                for (int b = 0; b < B; b++) {
                    memcpy(m->buf_up + b * dim, m->buf_gate + b * latent_dim, latent_dim * sizeof(float));
                    if (dim > latent_dim) memset(m->buf_up + b * dim + latent_dim, 0, (dim - latent_dim) * sizeof(float));
                }
                matmul_f32_row_major_per_row_batched(rows, dim, w, m->buf_up, dim, rs_fp16, m->buf_gate, B);
            }
        }
    } else {
        // V2-Lite path: single o_proj
        auto& to = m->tensors[idx[4]];
        if (to.ttype == 11 && m->use_f32_matmul) {
            const uint16_t* o_rs_fp16 = (const uint16_t*)(to.data);
            const int8_t* ow = (int8_t*)(to.data + to.row_dim * 2);
            int o_rows = to.row_dim;
            int o_dim = t3_dim(to);
            for (int b = 0; b < B; b++) {
                float* dst = m->buf_act + b * o_dim;
                const float* src = attn_out + b * qd;
                for (int h = 0; h < nH; h++)
                    memcpy(dst + h * m->v_head_dim, src + h * hd, m->v_head_dim * sizeof(float));
            }
            matmul_f32_row_major_per_row_batched(o_rows, o_dim, ow, m->buf_act, o_dim, o_rs_fp16, m->buf_gate, B);
        } else {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(to, w, rs, rows, dim, scale);
            for (int b = 0; b < B; b++) {
                float* dst = m->buf_act + b * dim;
                const float* src = attn_out + b * qd;
                for (int h = 0; h < nH; h++)
                    memcpy(dst + h * m->v_head_dim, src + h * hd, m->v_head_dim * sizeof(float));
            }
            matmul_f32_row_major_batched(rows, dim, w, m->buf_act, dim, scale, m->buf_gate, B);
        }
    }

    #ifdef ATLAS_DEBUG_MODE
    fprintf(stderr, "[MLA_STEP] L=%d step6_post_oproj B=%d\n", layer, B);
    #endif

    // ─── 7. Residual: output = input + O_proj ───
    for (int i = 0; i < B * H; i++)
        output[i] = input[i] + m->buf_gate[i] * m->scale_depth_factor;

    #ifdef ATLAS_DEBUG_MODE
    fprintf(stderr, "[MLA_STEP] L=%d step7_post_residual\n", layer);
    #endif

    // ─── 8. Post-attention RMSNorm ───
    {
        auto& t_ln2 = m->tensors[idx[5]];
        const uint8_t* w = t_ln2.data;
        for (int b = 0; b < B; b++) {
            const float* xb = output + b * H;
            float* nb = m->buf_act + b * H;
            float ss = 0.0f;
            for (int i = 0; i < H; i++) ss += xb[i] * xb[i];
            float rms = 1.0f / sqrtf(ss / H + 1e-6f);
            for (int i = 0; i < H; i++) {
                uint16_t w16; memcpy(&w16, w + i * 2, 2);
                nb[i] = xb[i] * rms * fp16_to_fp32(w16);
            }
        }
    }

    #ifdef ATLAS_DEBUG_MODE
    fprintf(stderr, "[MLA_STEP] L=%d step8_post_rmsnorm2\n", layer);
    #endif

    // ─── 9. FFN: Dense (L < first_k_dense_replace) or MoE (L >= first_k_dense_replace) ───
    if (layer < m->first_k_dense_replace) {
        // Dense FFN: gate + up → SiLU → down (reuse existing logic)
        // For now: simplified path using buf_act as x_norm2
        float* x_norm2 = m->buf_act;
        auto& tg = m->tensors[idx[6]];
        auto& tu = m->tensors[idx[7]];
        auto& td = m->tensors[idx[8]];

        // gate + up projections
        int g_dim = t3_dim(tg);
        int u_dim = t3_dim(tu);
        int ffn_dim = g_dim > u_dim ? g_dim : u_dim;

        for (int b = 0; b < B; b++) {
            memmove(m->buf_act + b * ffn_dim, x_norm2 + b * H, H * sizeof(float));
            memset(m->buf_act + b * ffn_dim + H, 0, (ffn_dim - H) * sizeof(float));
        }

        if (m->use_f32_matmul && tg.ttype == 11) {
            const uint16_t* g_rs = (const uint16_t*)(tg.data);
            const int8_t* gw = (int8_t*)(tg.data + tg.row_dim * 2);
            const uint16_t* u_rs = (const uint16_t*)(tu.data);
            const uint8_t* uw_raw = tu.data + tu.row_dim * 2;
            int8_t* uw = (int8_t*)uw_raw;
            int dim11 = t3_dim(tg);
            #ifdef ATLAS_DEBUG_MODE
            fprintf(stderr, "[MLA_STEP] L=%d step9a_gate_t11 g_rows=%d dim11=%d\n", layer, tg.row_dim, dim11);
            #endif
            matmul_f32_per_row(tg.row_dim, dim11, gw, m->buf_act, ffn_dim, g_rs, m->buf_gate, B);
            #ifdef ATLAS_DEBUG_MODE
            fprintf(stderr, "[MLA_STEP] L=%d step9b_up_t11\n", layer);
            #endif
            matmul_f32_per_row(tu.row_dim, dim11, uw, m->buf_act, ffn_dim, u_rs, m->buf_up, B);
        } else if (m->use_f32_matmul) {
            int8_t* gw; int32_t* grs; int g_rows, g_dim_v; float g_scale;
            int8_t* uw; int32_t* urs; int u_rows, u_dim_v; float u_scale;
            get_i8(tg, gw, grs, g_rows, g_dim_v, g_scale);
            get_i8(tu, uw, urs, u_rows, u_dim_v, u_scale);
            #ifdef ATLAS_DEBUG_MODE
            fprintf(stderr, "[MLA_STEP] L=%d step9a_gate g_rows=%d g_dim=%d B=%d\n", layer, g_rows, g_dim_v, B);
            #endif
            matmul_f32_reorder(g_rows, g_dim_v, gw, m->buf_act, g_scale, m->buf_gate, B);
            #ifdef ATLAS_DEBUG_MODE
            fprintf(stderr, "[MLA_STEP] L=%d step9b_up u_rows=%d\n", layer, u_rows);
            #endif
            matmul_f32_reorder(u_rows, u_dim_v, uw, m->buf_act, u_scale, m->buf_up, B);
        }

        // SiLU(gate) * up → down
        #ifdef ATLAS_DEBUG_MODE
        fprintf(stderr, "[MLA_STEP] L=%d step9c_silu inter=%d\n", layer, inter);
        #endif
        auto& t_down = m->tensors[idx[8]];
        int d_dim = t3_dim(t_down);
        for (int b = 0; b < B; b++) {
            float* tmp = m->buf_ffn_f32 + b * d_dim;
            for (int i = 0; i < inter; i++)
                tmp[i] = gate_activation(m->buf_gate[b * inter + i], m->model_arch == ARCH_BITNET || m->use_relu2)
                         * m->buf_up[b * inter + i];
            for (int i = inter; i < d_dim; i++) tmp[i] = 0.0f;
        }
        #ifdef ATLAS_DEBUG_MODE
        fprintf(stderr, "[MLA_STEP] L=%d step9d_down_pre\n", layer);
        #endif
        if (m->use_f32_matmul && t_down.ttype == 11) {
            const uint16_t* d_rs = (const uint16_t*)(t_down.data);
            const int8_t* dw = (int8_t*)(t_down.data + t_down.row_dim * 2);
            int dim11 = t3_dim(t_down);
            matmul_f32_per_row(t_down.row_dim, dim11, dw, m->buf_ffn_f32, d_dim, d_rs, m->buf_gate, B);
        } else {
            int8_t* dw; int32_t* drs; int d_rows, d_dim_v; float d_scale;
            get_i8(t_down, dw, drs, d_rows, d_dim_v, d_scale);
            #ifdef ATLAS_DEBUG_MODE
            fprintf(stderr, "[MLA_STEP] L=%d step9d_down d_rows=%d d_dim=%d\n", layer, d_rows, d_dim_v);
            #endif
            matmul_f32_reorder(d_rows, d_dim_v, dw, m->buf_ffn_f32, d_scale, m->buf_gate, B);
        }
        #ifdef ATLAS_DEBUG_MODE
        fprintf(stderr, "[MLA_STEP] L=%d step9e_down_post\n", layer);
        #endif
    } else {
        // MoE FFN: Router + Sparse Experts + Shared Experts
        atlas_moe_forward(m, output, m->buf_gate, B, m->buf_act, layer);
    }

    // ─── 10. Residual: output += FFN_down (* scale_depth_factor) ───
    for (int i = 0; i < B * H; i++)
        output[i] += m->buf_gate[i] * m->scale_depth_factor;
}


// ─── Internal: forward one transformer layer ──────────────────────────
// input: [B, H] float32 (read-only, preserved for residual)
// output: [B, H] float32 (must not alias input)
// K/V cache is accessed from model struct (int8 + per-position scaling)
static void forward_layer_internal(
    AtlasModel* m,
    const float* input, float* output, int B,
    const int* positions,
    uint8_t* k_cache_layer, uint8_t* v_cache_layer,
    int max_seq_len, int seq_now,
    int idx_ln1, int idx_q, int idx_k, int idx_v, int idx_o,
    int idx_ln2, int idx_gate, int idx_up, int idx_down,
    int idx_q_norm = -1, int idx_k_norm = -1,
    int idx_attn_sub_norm = -1, int idx_ffn_sub_norm = -1) {

    int H = m->hidden_dim;
    int nH = m->n_heads, nKV = m->n_kv_heads, hd = m->head_dim;
    int qd = nH * hd, kvd = nKV * hd;
    int inter = m->inter_dim;
    float theta = m->rope_theta;

    #ifdef ATLAS_DEBUG_MODE
    static int dbg_layer_count = 0;
    int dbg_cur_layer = dbg_layer_count++;
    char dbg_path[256];
    if (B <= 6 && H == 2048) {
        fprintf(stderr, "DBG_INPUT_b0: ");
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", input[i]);
        fprintf(stderr, "\n");
        snprintf(dbg_path, sizeof(dbg_path), "dbg_L%d_INPUT.bin", dbg_cur_layer);
        FILE* f = fopen(dbg_path, "wb");
        if (f) { fwrite(input, sizeof(float), B * H, f); fclose(f); }
    }
    #endif

    P_START(rmsnorm);
    // ─── 1. Pre-attention RMSNorm ───
    {
        auto& t = m->tensors[idx_ln1];
        const uint8_t* w = t.data;
        for (int b = 0; b < B; b++) {
            const float* xb = input + b * H;
            float* nb = output + b * H;
            float ss = 0.0f;
            for (int i = 0; i < H; i++) ss += xb[i] * xb[i];
            float rms = 1.0f / sqrtf(ss / H + 1e-6f);
            for (int i = 0; i < H; i++) {
                uint16_t w16; memcpy(&w16, w + i * 2, 2);
                nb[i] = xb[i] * rms * fp16_to_fp32(w16);
            }
        }
        #ifdef ATLAS_DEBUG_MODE
        if (B <= 6 && H == 2048) {
            fprintf(stderr, "DEBUG_LN1_b0: ");
            for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", output[i]);
            fprintf(stderr, "\n");
            fprintf(stderr, "DEBUG_INPUT_b0: ");
            for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", input[i]);
            fprintf(stderr, "\n");
            snprintf(dbg_path, sizeof(dbg_path), "dbg_L%d_LN1.bin", dbg_cur_layer);
            FILE* f = fopen(dbg_path, "wb");
            if (f) { fwrite(output, sizeof(float), B * H, f); fclose(f); }
        }
        #endif
    }
    P_ACCUM(rmsnorm);
    float* x_norm = output;

    P_START(qkv);
    // ─── 2. QKV projections (int8) ───
    auto t3_dim = [](const TensorInfo& t) -> int {
        if (t.ttype == 10 || t.ttype == 5) return t.packed_cols * 5;
        if (t.ttype == 3) return (int)((t.data_size - 2 - (int64_t)t.row_dim * 4) / t.row_dim);
        if (t.ttype == 11) return (int)((t.data_size - (int64_t)t.row_dim * 6) / t.row_dim);
        if (t.ttype == 7) return t.packed_cols * 4;
        if (t.ttype == 8) return t.packed_cols * 2;
        return t.packed_cols * 5;
    };
    auto get_i8 = [&](const TensorInfo& t, int8_t*& w, int32_t*& rs,
                     int& rows, int& dim, float& scale) {
        if (t.ttype != 3) { rows = 0; dim = 0; w = nullptr; rs = nullptr; return; }
        uint16_t sr; memcpy(&sr, t.data, 2);
        scale = fp16_to_fp32(sr);
        rows = t.row_dim;
        dim = t3_dim(t);
        int nv = rows * dim;
        w = (int8_t*)(t.data + 2);
        rs = (int32_t*)(w + nv);
    };
    auto get_tq1_packed = [](const TensorInfo& t, const uint8_t*& w, int& rows,
                      int& dim, int& pc, float& scale) {
        if (t.ttype != 0) { rows = 0; dim = 0; pc = 0; w = nullptr; return; }
        uint16_t sr; memcpy(&sr, t.data, 2);
        scale = fp16_to_fp32(sr);
        rows = t.row_dim;
        pc = t.packed_cols;
        dim = pc * 5;
        w = t.data + 2;
    };

    auto& tq = m->tensors[idx_q];
    auto& tk = m->tensors[idx_k];
    int i8_q_dim = t3_dim(tq);
    int i8_k_dim = t3_dim(tk);
    int max_qkv_dim = i8_q_dim > i8_k_dim ? i8_q_dim : i8_k_dim;

    for (int b = 0; b < B; b++) {
        memcpy(m->buf_act + b * max_qkv_dim, x_norm + b * H, H * sizeof(float));
        memset(m->buf_act + b * max_qkv_dim + H, 0,
               (max_qkv_dim - H) * sizeof(float));
    }

    float* max_abs = (float*)alloca(B * sizeof(float));

    auto& tv = m->tensors[idx_v];
    if (tq.ttype == 10) {
        if (m->use_f32_matmul) {
            auto tq2_fn = matmul_tq2_f32;
            { auto& t = m->tensors[idx_q]; tq2_fn(t.row_dim, max_qkv_dim, t.packed_cols, t.data, t.block_size, t.n_blocks, m->buf_act, m->buf_gate, B); }
            { auto& t = m->tensors[idx_k]; tq2_fn(t.row_dim, max_qkv_dim, t.packed_cols, t.data, t.block_size, t.n_blocks, m->buf_act, m->buf_hidden, B); }
            { auto& t = m->tensors[idx_v]; tq2_fn(t.row_dim, max_qkv_dim, t.packed_cols, t.data, t.block_size, t.n_blocks, m->buf_act, m->buf_up, B); }
        } else if (B >= LUT_THRESHOLD) {
            // TQ2 LUT prefill: x86 delegates to TQ1 kernel, skip 1-byte flags
            { auto& t = m->tensors[idx_q]; tq1_lut_prefill_kernel(t.row_dim, max_qkv_dim, t.packed_cols, t.data + 1, t.block_size, t.n_blocks, m->buf_act, m->buf_gate, B); }
            { auto& t = m->tensors[idx_k]; tq1_lut_prefill_kernel(t.row_dim, max_qkv_dim, t.packed_cols, t.data + 1, t.block_size, t.n_blocks, m->buf_act, m->buf_hidden, B); }
            { auto& t = m->tensors[idx_v]; tq1_lut_prefill_kernel(t.row_dim, max_qkv_dim, t.packed_cols, t.data + 1, t.block_size, t.n_blocks, m->buf_act, m->buf_up, B); }
        } else {
            auto tq2_fn = matmul_tq2;
            { auto& t = m->tensors[idx_q]; tq2_fn(t.row_dim, max_qkv_dim, t.packed_cols, t.data, t.block_size, t.n_blocks, m->buf_act, m->buf_gate, B); }
            { auto& t = m->tensors[idx_k]; tq2_fn(t.row_dim, max_qkv_dim, t.packed_cols, t.data, t.block_size, t.n_blocks, m->buf_act, m->buf_hidden, B); }
            { auto& t = m->tensors[idx_v]; tq2_fn(t.row_dim, max_qkv_dim, t.packed_cols, t.data, t.block_size, t.n_blocks, m->buf_act, m->buf_up, B); }
        }
    } else if (tq.ttype == 5) {
        if (m->use_f32_matmul) {
            {
                auto& t = m->tensors[idx_q];
                matmul_tq1_block_fused_f32(t.row_dim, max_qkv_dim, t.packed_cols,
                    t.data, t.block_size, t.n_blocks,
                    m->buf_act, m->buf_gate, B);
            }
            {
                auto& t = m->tensors[idx_k];
                matmul_tq1_block_fused_f32(t.row_dim, max_qkv_dim, t.packed_cols,
                    t.data, t.block_size, t.n_blocks,
                    m->buf_act, m->buf_hidden, B);
            }
            {
                auto& t = m->tensors[idx_v];
                matmul_tq1_block_fused_f32(t.row_dim, max_qkv_dim, t.packed_cols,
                    t.data, t.block_size, t.n_blocks,
                    m->buf_act, m->buf_up, B);
            }
        } else if (B >= LUT_THRESHOLD) {
            {
                auto& t = m->tensors[idx_q];
                tq1_lut_prefill_kernel(t.row_dim, max_qkv_dim, t.packed_cols,
                    t.data, t.block_size, t.n_blocks,
                    m->buf_act, m->buf_gate, B);
            }
            {
                auto& t = m->tensors[idx_k];
                tq1_lut_prefill_kernel(t.row_dim, max_qkv_dim, t.packed_cols,
                    t.data, t.block_size, t.n_blocks,
                    m->buf_act, m->buf_hidden, B);
            }
            {
                auto& t = m->tensors[idx_v];
                tq1_lut_prefill_kernel(t.row_dim, max_qkv_dim, t.packed_cols,
                    t.data, t.block_size, t.n_blocks,
                    m->buf_act, m->buf_up, B);
            }
        } else {
            {
                auto& t = m->tensors[idx_q];
                matmul_tq1_block_fused_s8(t.row_dim, max_qkv_dim, t.packed_cols,
                    t.data, t.block_size, t.n_blocks,
                    m->buf_act, m->buf_gate, B);
            }
            {
                auto& t = m->tensors[idx_k];
                matmul_tq1_block_fused_s8(t.row_dim, max_qkv_dim, t.packed_cols,
                    t.data, t.block_size, t.n_blocks,
                    m->buf_act, m->buf_hidden, B);
            }
            {
                auto& t = m->tensors[idx_v];
                matmul_tq1_block_fused_s8(t.row_dim, max_qkv_dim, t.packed_cols,
                    t.data, t.block_size, t.n_blocks,
                    m->buf_act, m->buf_up, B);
            }
        }
    } else if (tq.ttype == 7) {
        // v2.7.0: TurboQuant 2-bit fused matmul (on-the-fly decode in registers)
        {
            auto& t = m->tensors[idx_q];
            matmul_turboquant_fused(t.row_dim, max_qkv_dim, t.packed_cols,
                t.data, t.block_size, t.n_blocks,
                m->buf_act, m->buf_gate, B);
        }
        {
            auto& t = m->tensors[idx_k];
            matmul_turboquant_fused(t.row_dim, max_qkv_dim, t.packed_cols,
                t.data, t.block_size, t.n_blocks,
                m->buf_act, m->buf_hidden, B);
        }
        {
            auto& t = m->tensors[idx_v];
            matmul_turboquant_fused(t.row_dim, max_qkv_dim, t.packed_cols,
                t.data, t.block_size, t.n_blocks,
                m->buf_act, m->buf_up, B);
        }
    } else if (tq.ttype == 0) {
        // v1.3.1+1.3.2: Direct TQ1-packed matmul (QKV stay packed in hybrid mode)
        quantize_f32_to_u8(m->buf_act, B, max_qkv_dim, max_abs, m->buf_i8);
        {
            const uint8_t* w; int rows, dim, pc; float scale;
            get_tq1_packed(tq, w, rows, dim, pc, scale);
            matmul_tq1_packed_reorder(rows, dim, w, pc, m->buf_i8, max_abs, scale, m->buf_gate, B);
        }
        {
            const uint8_t* w; int rows, dim, pc; float scale;
            get_tq1_packed(tk, w, rows, dim, pc, scale);
            matmul_tq1_packed_reorder(rows, dim, w, pc, m->buf_i8, max_abs, scale, m->buf_hidden, B);
        }
        {
            const uint8_t* w; int rows, dim, pc; float scale;
            get_tq1_packed(tv, w, rows, dim, pc, scale);
            matmul_tq1_packed_reorder(rows, dim, w, pc, m->buf_i8, max_abs, scale, m->buf_up, B);
        }
    } else if (m->use_ternary_matmul) {

        // Ternary-add path: vpsignb, no multiplication, no row_sums
        quantize_f32_to_u8(m->buf_act, B, max_qkv_dim, max_abs, m->buf_i8);
        {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(tq, w, rs, rows, dim, scale);
            matmul_ternary_add_reorder(rows, dim, w, m->buf_i8, max_abs, scale, m->buf_gate, B);
        }
        {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(tk, w, rs, rows, dim, scale);
            matmul_ternary_add_reorder(rows, dim, w, m->buf_i8, max_abs, scale, m->buf_hidden, B);
        }
        {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(tv, w, rs, rows, dim, scale);
            matmul_ternary_add_reorder(rows, dim, w, m->buf_i8, max_abs, scale, m->buf_up, B);
        }
    } else if (m->use_f32_matmul) {
        // Full-precision path: f32×i8, no activation quantization
        {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            if (tq.ttype == 11) {
                const uint16_t* rs_fp16 = (const uint16_t*)(tq.data);
                w = (int8_t*)(tq.data + tq.row_dim * 2);
                rows = tq.row_dim;
                dim = t3_dim(tq);
                matmul_f32_reorder_per_row(rows, dim, w, m->buf_act, rs_fp16, m->buf_gate, B);
            } else {
                get_i8(tq, w, rs, rows, dim, scale);
                #ifdef ATLAS_DEBUG_MODE
                if (B <= 6 && H == 2048) fprintf(stderr, "DEBUG_Q: rows=%d dim=%d scale=%.6f\n", rows, dim, scale);
                #endif
                matmul_f32_reorder(rows, dim, w, m->buf_act, scale, m->buf_gate, B);
            }
            #ifdef ATLAS_DEBUG_MODE
            if (B <= 6 && H == 2048) {
                fprintf(stderr, "DEBUG_Q_b0: ");
                for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", m->buf_gate[i]);
                fprintf(stderr, "\n");
            }
            #endif
        }
        {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            if (tk.ttype == 11) {
                const uint16_t* rs_fp16 = (const uint16_t*)(tk.data);
                w = (int8_t*)(tk.data + tk.row_dim * 2);
                rows = tk.row_dim;
                dim = t3_dim(tk);
                matmul_f32_reorder_per_row(rows, dim, w, m->buf_act, rs_fp16, m->buf_hidden, B);
            } else {
                get_i8(tk, w, rs, rows, dim, scale);
                #ifdef ATLAS_DEBUG_MODE
                if (B <= 6 && H == 2048) fprintf(stderr, "DEBUG_K: rows=%d dim=%d scale=%.6f\n", rows, dim, scale);
                #endif
                matmul_f32_reorder(rows, dim, w, m->buf_act, scale, m->buf_hidden, B);
            }
            #ifdef ATLAS_DEBUG_MODE
            if (B <= 6 && H == 2048) {
                fprintf(stderr, "DEBUG_K_b0: ");
                for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", m->buf_hidden[i]);
                fprintf(stderr, "\n");
            }
            #endif
        }
        {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            if (tv.ttype == 11) {
                const uint16_t* rs_fp16 = (const uint16_t*)(tv.data);
                w = (int8_t*)(tv.data + tv.row_dim * 2);
                rows = tv.row_dim;
                dim = t3_dim(tv);
                matmul_f32_reorder_per_row(rows, dim, w, m->buf_act, rs_fp16, m->buf_up, B);
            } else {
                get_i8(tv, w, rs, rows, dim, scale);
                #ifdef ATLAS_DEBUG_MODE
                if (B <= 6 && H == 2048) fprintf(stderr, "DEBUG_V: rows=%d dim=%d scale=%.6f\n", rows, dim, scale);
                #endif
                matmul_f32_reorder(rows, dim, w, m->buf_act, scale, m->buf_up, B);
            }
            #ifdef ATLAS_DEBUG_MODE
            if (B <= 6 && H == 2048) {
                fprintf(stderr, "DEBUG_V_b0: ");
                for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", m->buf_up[i]);
                fprintf(stderr, "\n");
            }
            #endif
        }
    } else {
        quantize_f32_to_u8(m->buf_act, B, max_qkv_dim, max_abs, m->buf_i8);

        // Q projection: scratch=buf_act, output=buf_gate
        {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(tq, w, rs, rows, dim, scale);
            matmul_reorder_deq(rows, dim, w, rs, m->buf_i8, max_abs,
                              scale, m->buf_act, m->buf_gate, B);
        }
        int i8_v_dim = t3_dim(tv);

        // K projection: scratch=buf_act (reused), output=buf_hidden
        {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(tk, w, rs, rows, dim, scale);
            if (i8_k_dim != max_qkv_dim) {
                for (int b = 0; b < B; b++) {
                    memcpy(m->buf_act + b * i8_k_dim, x_norm + b * H, H * sizeof(float));
                    memset(m->buf_act + b * i8_k_dim + H, 0,
                           (i8_k_dim - H) * sizeof(float));
                }
                quantize_f32_to_u8(m->buf_act, B, i8_k_dim, max_abs, m->buf_i8);
            }
            matmul_reorder_deq(rows, dim, w, rs, m->buf_i8, max_abs,
                              scale, m->buf_act, m->buf_hidden, B);
        }
        // V projection: scratch=buf_act (reused again, free after K over), output=buf_up
        {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(tv, w, rs, rows, dim, scale);
            if (i8_v_dim != i8_k_dim) {
                for (int b = 0; b < B; b++) {
                    memcpy(m->buf_act + b * i8_v_dim, x_norm + b * H, H * sizeof(float));
                    memset(m->buf_act + b * i8_v_dim + H, 0,
                           (i8_v_dim - H) * sizeof(float));
                }
                quantize_f32_to_u8(m->buf_act, B, i8_v_dim, max_abs, m->buf_i8);
            }
            matmul_reorder_deq(rows, dim, w, rs, m->buf_i8, max_abs,
                              scale, m->buf_act, m->buf_up, B);
        }
    }
    P_ACCUM(qkv);

    P_START(attention);
    // ─── 3. Fused attention (reuse atlas_attention_f32) ───
    int ws = B * nH * hd;
    float* attn_out = m->attn_ws;
    float* q_f32 = m->attn_ws + ws;
    float* k_f32 = m->attn_ws + ws * 2;
    float* v_f32 = m->attn_ws + ws * 3;
    for (int b = 0; b < B; b++) {
        memcpy(q_f32 + b * qd, m->buf_gate + b * tq.row_dim, qd * sizeof(float));
        memcpy(k_f32 + b * kvd, m->buf_hidden + b * tk.row_dim, kvd * sizeof(float));
        memcpy(v_f32 + b * kvd, m->buf_up + b * tv.row_dim, kvd * sizeof(float));
    }
    const uint8_t* qn_w = (idx_q_norm >= 0) ? m->tensors[idx_q_norm].data : nullptr;
    const uint8_t* kn_w = (idx_k_norm >= 0) ? m->tensors[idx_k_norm].data : nullptr;
    atlas_attention_f32(q_f32, k_f32, v_f32, positions,
        k_cache_layer, v_cache_layer,
        max_seq_len, seq_now, B,
        nH, nKV, hd, theta, m->rope_scale, attn_out, qn_w, kn_w,
        m->base_seq_len, m->rope_interleaved);

    // ─── 3.5. attn_sub_norm: RMSNorm(attn_out) per BitNet reference ───
    if (idx_attn_sub_norm >= 0) {
        for (int b = 0; b < B; b++) {
            float* ap = attn_out + b * qd;
            apply_sub_norm(ap, qd, m->tensors[idx_attn_sub_norm].data);
        }
    }

    #ifdef ATLAS_DEBUG_MODE
    if (B <= 6 && H == 2048) {
        fprintf(stderr, "DEBUG_ATTN_b0: ");
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", attn_out[i]);
        fprintf(stderr, "\n");
    }
    #endif
    P_ACCUM(attention);
    P_START(o_proj);

    // ─── 4. O projection (int8) ───
    {
        auto& to = m->tensors[idx_o];
        if (to.ttype == 10) {
            if (m->use_f32_matmul) {
                matmul_tq2_f32(to.row_dim, qd, to.packed_cols, to.data, to.block_size, to.n_blocks, attn_out, m->buf_gate, B);
            } else if (B >= LUT_THRESHOLD) {
                tq1_lut_prefill_kernel(to.row_dim, qd, to.packed_cols, to.data + 1, to.block_size, to.n_blocks, attn_out, m->buf_gate, B);
            } else {
                matmul_tq2(to.row_dim, qd, to.packed_cols, to.data, to.block_size, to.n_blocks, attn_out, m->buf_gate, B);
            }
    } else if (to.ttype == 5) {
            int o_dim = to.packed_cols * 5;
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * o_dim, attn_out + b * qd, qd * sizeof(float));
                memset(m->buf_act + b * o_dim + qd, 0,
                       (o_dim - qd) * sizeof(float));
            }
            if (m->use_f32_matmul) {
                matmul_tq1_block_fused_f32(to.row_dim, o_dim, to.packed_cols,
                    to.data, to.block_size, to.n_blocks,
                    m->buf_act, m->buf_gate, B);
            } else if (B >= LUT_THRESHOLD && to.block_size > 0 && to.n_blocks > 0) {
                tq1_lut_prefill_kernel(to.row_dim, o_dim, to.packed_cols,
                    to.data, to.block_size, to.n_blocks,
                    m->buf_act, m->buf_gate, B);
            } else {
                matmul_tq1_block_fused_s8(to.row_dim, o_dim, to.packed_cols,
                    to.data, to.block_size, to.n_blocks,
                    m->buf_act, m->buf_gate, B);
            }
        } else if (to.ttype == 7) {
            int o_dim = to.packed_cols * 4;
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * o_dim, attn_out + b * qd, qd * sizeof(float));
                memset(m->buf_act + b * o_dim + qd, 0, (o_dim - qd) * sizeof(float));
            }
            matmul_turboquant_fused(to.row_dim, o_dim, to.packed_cols,
                to.data, to.block_size, to.n_blocks,
                m->buf_act, m->buf_gate, B);
        } else if (to.ttype == 0) {
            const uint8_t* wp; int rows, dim, pc; float scale;
            get_tq1_packed(to, wp, rows, dim, pc, scale);
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * dim, attn_out + b * qd, qd * sizeof(float));
                memset(m->buf_act + b * dim + qd, 0, (dim - qd) * sizeof(float));
            }
            quantize_f32_to_u8(m->buf_act, B, dim, max_abs, m->buf_i8);
            matmul_tq1_packed_reorder(rows, dim, wp, pc, m->buf_i8, max_abs, scale, m->buf_gate, B);
        } else if (to.ttype == 11) {
            // Per-row int8 O projection
            const uint16_t* o_rs_fp16 = (const uint16_t*)(to.data);
            const int8_t* ow = (const int8_t*)(to.data + to.row_dim * 2);
            int o11_rows = to.row_dim;
            int o11_dim = t3_dim(to);
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * o11_dim, attn_out + b * qd, qd * sizeof(float));
                memset(m->buf_act + b * o11_dim + qd, 0, (o11_dim - qd) * sizeof(float));
            }
            matmul_f32_reorder_per_row(o11_rows, o11_dim, ow, m->buf_act, o_rs_fp16, m->buf_gate, B);
        } else {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(to, w, rs, rows, dim, scale);
            if (dim <= 0) { for (int b = 0; b < B; b++) memset(m->buf_act + b * qd, 0, qd * sizeof(float)); } else
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * dim, attn_out + b * qd, qd * sizeof(float));
                memset(m->buf_act + b * dim + qd, 0, (dim - qd) * sizeof(float));
            }
            if (m->use_ternary_matmul) {
                quantize_f32_to_u8(m->buf_act, B, dim, max_abs, m->buf_i8);
                matmul_ternary_add_reorder(rows, dim, w, m->buf_i8, max_abs,
                                          scale, m->buf_gate, B);
            } else if (m->use_f32_matmul) {
                matmul_f32_reorder(rows, dim, w, m->buf_act, scale, m->buf_gate, B);
            } else {
                quantize_f32_to_u8(m->buf_act, B, dim, max_abs, m->buf_i8);
                matmul_reorder_deq(rows, dim, w, rs, m->buf_i8, max_abs,
                                  scale, m->buf_hidden, m->buf_gate, B);
            }
        }
    }

    #ifdef ATLAS_DEBUG_MODE
    if (B <= 6 && H == 2048) {
        fprintf(stderr, "DEBUG_O_b0: ");
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", m->buf_gate[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "DEBUG_SDF: %.6f\n", m->scale_depth_factor);
        snprintf(dbg_path, sizeof(dbg_path), "dbg_L%d_O.bin", dbg_cur_layer);
        FILE* f = fopen(dbg_path, "wb");
        if (f) { fwrite(m->buf_gate, sizeof(float), B * H, f); fclose(f); }
    }
    #endif
    P_ACCUM(o_proj);
    // ─── 5. Residual: output = input + attn_out_proj (* scale_depth_factor) ───
    for (int i = 0; i < B * H; i++) {
        output[i] = input[i] + m->buf_gate[i] * m->scale_depth_factor;
    }
    #ifdef ATLAS_DEBUG_MODE
    if (B <= 6 && H == 2048) {
        fprintf(stderr, "DBG_RES1_b0: ");
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", output[i]);
        fprintf(stderr, "\n");
        snprintf(dbg_path, sizeof(dbg_path), "dbg_L%d_RES1.bin", dbg_cur_layer);
        FILE* f = fopen(dbg_path, "wb");
        if (f) { fwrite(output, sizeof(float), B * H, f); fclose(f); }
    }
    #endif
    P_RESTART(rmsnorm);
    // ─── 6. Post-attention RMSNorm ───
    {
        auto& t = m->tensors[idx_ln2];
        const uint8_t* w = t.data;
        for (int b = 0; b < B; b++) {
            const float* xb = output + b * H;
            float* nb = m->buf_act + b * H;
            float ss = 0.0f;
            for (int i = 0; i < H; i++) ss += xb[i] * xb[i];
            float rms = 1.0f / sqrtf(ss / H + 1e-6f);
            for (int i = 0; i < H; i++) {
                uint16_t w16; memcpy(&w16, w + i * 2, 2);
                nb[i] = xb[i] * rms * fp16_to_fp32(w16);
            }
        }
    }
    #ifdef ATLAS_DEBUG_MODE
    if (B <= 6 && H == 2048) {
        fprintf(stderr, "DBG_LN2_b0: ");
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", m->buf_act[i]);
        fprintf(stderr, "\n");
        snprintf(dbg_path, sizeof(dbg_path), "dbg_L%d_LN2.bin", dbg_cur_layer);
        FILE* f = fopen(dbg_path, "wb");
        if (f) { fwrite(m->buf_act, sizeof(float), B * H, f); fclose(f); }
    }
    #endif
    float* x_norm2 = m->buf_act;
    P_ACCUM(rmsnorm);
    P_START(ffn_gate_up);

    // ─── 7. FFN: fused gate + up projections (one OMP region) ───
    // Fused kernel processes both in one pass: shared activation, no scratch buffer,
    // reorder+dequant inline.
    auto& tg = m->tensors[idx_gate];
    auto& tu = m->tensors[idx_up];
    int g_dim = t3_dim(tg);
    int u_dim = t3_dim(tu);
    int ffn_dim = g_dim > u_dim ? g_dim : u_dim;

    for (int b = 0; b < B; b++) {
        memmove(m->buf_act + b * ffn_dim, x_norm2 + b * H, H * sizeof(float));
        memset(m->buf_act + b * ffn_dim + H, 0, (ffn_dim - H) * sizeof(float));
    }

    if (tg.ttype == 10) {
        if (m->use_f32_matmul) {
            { auto& t = m->tensors[idx_gate]; matmul_tq2_f32(t.row_dim, ffn_dim, t.packed_cols, t.data, t.block_size, t.n_blocks, m->buf_act, m->buf_gate, B); }
            { auto& t = m->tensors[idx_up];   matmul_tq2_f32(t.row_dim, ffn_dim, t.packed_cols, t.data, t.block_size, t.n_blocks, m->buf_act, m->buf_up, B); }
        } else if (B >= LUT_THRESHOLD) {
            { auto& t = m->tensors[idx_gate]; tq1_lut_prefill_kernel(t.row_dim, ffn_dim, t.packed_cols, t.data + 1, t.block_size, t.n_blocks, m->buf_act, m->buf_gate, B); }
            { auto& t = m->tensors[idx_up];   tq1_lut_prefill_kernel(t.row_dim, ffn_dim, t.packed_cols, t.data + 1, t.block_size, t.n_blocks, m->buf_act, m->buf_up, B); }
        } else {
            { auto& t = m->tensors[idx_gate]; matmul_tq2(t.row_dim, ffn_dim, t.packed_cols, t.data, t.block_size, t.n_blocks, m->buf_act, m->buf_gate, B); }
            { auto& t = m->tensors[idx_up];   matmul_tq2(t.row_dim, ffn_dim, t.packed_cols, t.data, t.block_size, t.n_blocks, m->buf_act, m->buf_up, B); }
        }
    } else if (tg.ttype == 5) {
        if (m->use_f32_matmul) {
            {
                auto& t = m->tensors[idx_gate];
                matmul_tq1_block_fused_f32(t.row_dim, ffn_dim, t.packed_cols,
                    t.data, t.block_size, t.n_blocks,
                    m->buf_act, m->buf_gate, B);
            }
            {
                auto& t = m->tensors[idx_up];
                matmul_tq1_block_fused_f32(t.row_dim, ffn_dim, t.packed_cols,
                    t.data, t.block_size, t.n_blocks,
                    m->buf_act, m->buf_up, B);
            }
        } else if (B >= LUT_THRESHOLD) {
            auto& t = m->tensors[idx_gate];
            tq1_lut_prefill_kernel(t.row_dim, ffn_dim, t.packed_cols,
                t.data, t.block_size, t.n_blocks,
                m->buf_act, m->buf_gate, B);
            auto& tu = m->tensors[idx_up];
            tq1_lut_prefill_kernel(tu.row_dim, ffn_dim, tu.packed_cols,
                tu.data, tu.block_size, tu.n_blocks,
                m->buf_act, m->buf_up, B);
        } else {
            {
                auto& t = m->tensors[idx_gate];
                matmul_tq1_block_fused_s8(t.row_dim, ffn_dim, t.packed_cols,
                    t.data, t.block_size, t.n_blocks,
                    m->buf_act, m->buf_gate, B);
            }
            {
                auto& t = m->tensors[idx_up];
                matmul_tq1_block_fused_s8(t.row_dim, ffn_dim, t.packed_cols,
                    t.data, t.block_size, t.n_blocks,
                    m->buf_act, m->buf_up, B);
            }
        }
    } else if (tg.ttype == 7) {
        // v2.7.0: TurboQuant 2-bit fused matmul (on-the-fly decode in registers)
        {
            auto& t = m->tensors[idx_gate];
            matmul_turboquant_fused(t.row_dim, ffn_dim, t.packed_cols,
                t.data, t.block_size, t.n_blocks,
                m->buf_act, m->buf_gate, B);
        }
        {
            auto& t = m->tensors[idx_up];
            matmul_turboquant_fused(t.row_dim, ffn_dim, t.packed_cols,
                t.data, t.block_size, t.n_blocks,
                m->buf_act, m->buf_up, B);
        }
    } else if (tg.ttype == 0) {
        // v1.3.1+1.3.2: Direct TQ1-packed gate+up (FFN stay packed in full-packed mode)
        quantize_f32_to_u8(m->buf_act, B, ffn_dim, max_abs, m->buf_i8);
        {
            const uint8_t* wp; int rows, dim, pc; float scale;
            get_tq1_packed(tg, wp, rows, dim, pc, scale);
            matmul_tq1_packed_reorder(rows, dim, wp, pc, m->buf_i8, max_abs, scale, m->buf_gate, B);
        }
        {
            const uint8_t* wp; int rows, dim, pc; float scale;
            get_tq1_packed(tu, wp, rows, dim, pc, scale);
            matmul_tq1_packed_reorder(rows, dim, wp, pc, m->buf_i8, max_abs, scale, m->buf_up, B);
        }
    } else if (tg.ttype == 8 && !m->use_f32_matmul) {
        uint16_t gs16; memcpy(&gs16, tg.data, 2); float g_scale = fp16_to_fp32(gs16);
        uint16_t us16; memcpy(&us16, tu.data, 2); float u_scale = fp16_to_fp32(us16);
        int g_cols = tg.packed_cols * 2;
        int u_cols = tu.packed_cols * 2;
        const uint8_t* g_pw = tg.data + 2;
        const uint8_t* u_pw = tu.data + 2;
        const int32_t* g_rs = (const int32_t*)(tg.data + 2 + tg.row_dim * tg.packed_cols);
        const int32_t* u_rs = (const int32_t*)(tu.data + 2 + tu.row_dim * tu.packed_cols);
        quantize_f32_to_u8(m->buf_act, B, ffn_dim, max_abs, m->buf_i8);
        matmul_i4_reorder_deq(tg.row_dim, g_cols, g_pw, g_rs, m->buf_i8, max_abs, g_scale, m->buf_act, m->buf_gate, B);
        matmul_i4_reorder_deq(tu.row_dim, u_cols, u_pw, u_rs, m->buf_i8, max_abs, u_scale, m->buf_act, m->buf_up, B);
    } else if (m->use_ternary_matmul) {

        // Ternary-add FFN: vpsignb, no multiplication, no row_sums correction
        quantize_f32_to_u8(m->buf_act, B, ffn_dim, max_abs, m->buf_i8);

        int8_t* gw; int32_t* grs; int g_rows, g_dim_v; float g_scale;
        int8_t* uw; int32_t* urs; int u_rows, u_dim_v; float u_scale;
        get_i8(tg, gw, grs, g_rows, g_dim_v, g_scale);
        get_i8(tu, uw, urs, u_rows, u_dim_v, u_scale);
        int rows = g_rows, dim_w = g_dim_v;
        int rows_packed = rows / 4;

#ifndef __aarch64__
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int ur = 0; ur < rows_packed; ur++) {
            const int8_t* gw4 = gw + ur * 4 * dim_w;
            const int8_t* uw4 = uw + ur * 4 * dim_w;

            for (int b = 0; b < B; b++) {
                const uint8_t* a = m->buf_i8 + b * ffn_dim;
                float deq = max_abs[b] / 127.0f;

                float g_val[4], u_val[4];
                for (int sub = 0; sub < 4; sub++) {
                    const int8_t* wg = gw4 + sub * dim_w;
                    const int8_t* wu = uw4 + sub * dim_w;
                    __m256i g_acc = _mm256_setzero_si256();
                    __m256i u_acc = _mm256_setzero_si256();
                    int c = 0;
                    for (; c + 32 <= dim_w; c += 32) {
                        __m256i au = _mm256_loadu_si256((const __m256i*)(a + c));
                        __m256i ai = _mm256_sub_epi8(au, _mm256_set1_epi8(-128));
                        __m256i g_wv = _mm256_loadu_si256((const __m256i*)(wg + c));
                        __m256i u_wv = _mm256_loadu_si256((const __m256i*)(wu + c));
                        __m256i g_prod = _mm256_sign_epi8(ai, g_wv);
                        __m256i u_prod = _mm256_sign_epi8(ai, u_wv);

                        __m128i g_lo = _mm256_castsi256_si128(g_prod);
                        __m128i g_hi = _mm256_extracti128_si256(g_prod, 1);
                        __m256i g16_lo = _mm256_cvtepi8_epi16(g_lo);
                        __m256i g16_hi = _mm256_cvtepi8_epi16(g_hi);
                        __m256i g32_lo = _mm256_madd_epi16(g16_lo, _mm256_set1_epi16(1));
                        __m256i g32_hi = _mm256_madd_epi16(g16_hi, _mm256_set1_epi16(1));
                        g_acc = _mm256_add_epi32(g_acc, g32_lo);
                        g_acc = _mm256_add_epi32(g_acc, g32_hi);

                        __m128i u_lo = _mm256_castsi256_si128(u_prod);
                        __m128i u_hi = _mm256_extracti128_si256(u_prod, 1);
                        __m256i u16_lo = _mm256_cvtepi8_epi16(u_lo);
                        __m256i u16_hi = _mm256_cvtepi8_epi16(u_hi);
                        __m256i u32_lo = _mm256_madd_epi16(u16_lo, _mm256_set1_epi16(1));
                        __m256i u32_hi = _mm256_madd_epi16(u16_hi, _mm256_set1_epi16(1));
                        u_acc = _mm256_add_epi32(u_acc, u32_lo);
                        u_acc = _mm256_add_epi32(u_acc, u32_hi);
                    }

                    auto hsum_i32 = [](__m256i acc) -> int {
                        __m128i lo = _mm256_castsi256_si128(acc);
                        __m128i hi = _mm256_extracti128_si256(acc, 1);
                        __m128i s = _mm_add_epi32(lo, hi);
                        s = _mm_hadd_epi32(s, s);
                        s = _mm_hadd_epi32(s, s);
                        return _mm_cvtsi128_si32(s);
                    };
                    int g_dot = hsum_i32(g_acc);
                    int u_dot = hsum_i32(u_acc);

                    for (; c < dim_w; c++) {
                        int ai8 = (int)a[c] - 128;
                        g_dot += ai8 * (int)wg[c];
                        u_dot += ai8 * (int)wu[c];
                    }

                    // No 128*row_sum correction needed — vpsignb already gives Σ(ai8 * w_i)
                    g_val[sub] = (float)g_dot * deq / g_scale;
                    u_val[sub] = (float)u_dot * deq / u_scale;
                }

                float* g_out = m->buf_gate + b * rows;
                float* u_out = m->buf_up + b * rows;
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
#else
        atlas_matmul_ternary_f32_arm64(rows, dim_w, gw, m->buf_i8, max_abs, g_scale, m->buf_gate, B);
        atlas_matmul_ternary_f32_arm64(rows, dim_w, uw, m->buf_i8, max_abs, u_scale, m->buf_up, B);
#endif
    } else if (m->use_f32_matmul) {
        // Full-precision FFN: f32 activations × int8 weights, no activation quantization
        if (tg.ttype == 11 && tu.ttype == 11) {
#ifndef __aarch64__
            // Per-row int8 (ttype=11): gate and up separately
            const uint16_t* g_rs_fp16 = (const uint16_t*)(tg.data);
            const int8_t* gw = (const int8_t*)(tg.data + tg.row_dim * 2);
            const uint16_t* u_rs_fp16 = (const uint16_t*)(tu.data);
            const int8_t* uw = (const int8_t*)(tu.data + tu.row_dim * 2);
            int dim11 = t3_dim(tg);
            for (int b = 0; b < B; b++) {
                memmove(m->buf_act + b * ffn_dim, x_norm2 + b * H, H * sizeof(float));
                memset(m->buf_act + b * ffn_dim + H, 0, (ffn_dim - H) * sizeof(float));
            }
            matmul_f32_per_row(tg.row_dim, dim11, gw, m->buf_act, ffn_dim, g_rs_fp16, m->buf_gate, B);
            matmul_f32_per_row(tu.row_dim, dim11, uw, m->buf_act, ffn_dim, u_rs_fp16, m->buf_up, B);
#endif
        } else if (tg.ttype == 8 && tu.ttype == 8) {
#ifndef __aarch64__
            uint16_t g_s16; memcpy(&g_s16, tg.data, 2); float g_sc = fp16_to_fp32(g_s16);
            uint16_t u_s16; memcpy(&u_s16, tu.data, 2); float u_sc = fp16_to_fp32(u_s16);
            int rows = tg.row_dim, dim_w = tg.packed_cols * 2;
            int rows_packed = rows / 4;
            int packed_cols = tg.packed_cols;
            const uint8_t* gw = tg.data + 2;
            const uint8_t* uw = tu.data + 2;
            const __m128i mask_low = _mm_set1_epi8(0x0F);
            const __m128i xor8 = _mm_set1_epi8(8);
            const __m128i sub8 = _mm_set1_epi8(8);
            #ifdef _OPENMP
            #pragma omp parallel for if(rows_packed > 4)
            #endif
            for (int ur = 0; ur < rows_packed; ur++) {
                const uint8_t* gw4 = gw + ur * 4 * packed_cols;
                const uint8_t* uw4 = uw + ur * 4 * packed_cols;
                for (int b = 0; b < B; b++) {
                    const float* a = m->buf_act + b * ffn_dim;
                    float g_val[4], u_val[4];
                    for (int sub = 0; sub < 4; sub++) {
                        const uint8_t* wg = gw4 + sub * packed_cols;
                        const uint8_t* wu = uw4 + sub * packed_cols;
                        __m256 gs = _mm256_setzero_ps();
                        __m256 us = _mm256_setzero_ps();
                        int c = 0;
                        for (; c + 8 <= dim_w; c += 8) {
                            __m256 af = _mm256_loadu_ps(a + c);
                            uint32_t g_p4, u_p4;
                            memcpy(&g_p4, wg + c / 2, 4);
                            memcpy(&u_p4, wu + c / 2, 4);
                            __m128i gnib = _mm_cvtsi32_si128((int)g_p4);
                            __m128i unib = _mm_cvtsi32_si128((int)u_p4);
                            __m128i glo = _mm_and_si128(gnib, mask_low);
                            __m128i ghi = _mm_and_si128(_mm_srli_epi16(gnib, 4), mask_low);
                            __m128i ulo = _mm_and_si128(unib, mask_low);
                            __m128i uhi = _mm_and_si128(_mm_srli_epi16(unib, 4), mask_low);
                            __m128i gw8 = _mm_unpacklo_epi8(glo, ghi);
                            __m128i uw8 = _mm_unpacklo_epi8(ulo, uhi);
                            __m128i gws = _mm_sub_epi8(_mm_xor_si128(gw8, xor8), sub8);
                            __m128i uws = _mm_sub_epi8(_mm_xor_si128(uw8, xor8), sub8);
                            __m256 gwf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(gws));
                            __m256 uwf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(uws));
                            gs = _mm256_fmadd_ps(af, gwf, gs);
                            us = _mm256_fmadd_ps(af, uwf, us);
                        }
                        float gs_f = hsum_ps(gs);
                        float us_f = hsum_ps(us);
                        for (; c < dim_w; c++) {
                            int8_t g_wv = (c & 1) ? (int8_t)((((wg[c / 2] >> 4) & 0x0F) ^ 8) - 8)
                                                  : (int8_t)(((wg[c / 2] & 0x0F) ^ 8) - 8);
                            int8_t u_wv = (c & 1) ? (int8_t)((((wu[c / 2] >> 4) & 0x0F) ^ 8) - 8)
                                                  : (int8_t)(((wu[c / 2] & 0x0F) ^ 8) - 8);
                            gs_f += a[c] * (float)g_wv;
                            us_f += a[c] * (float)u_wv;
                        }
                        g_val[sub] = gs_f / g_sc;
                        u_val[sub] = us_f / u_sc;
                    }
                    float* g_out = m->buf_gate + b * rows;
                    float* u_out = m->buf_up + b * rows;
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
#endif
        } else {
            int8_t* gw; int32_t* grs; int g_rows, g_dim_v; float g_scale;
            int8_t* uw; int32_t* urs; int u_rows, u_dim_v; float u_scale;
            get_i8(tg, gw, grs, g_rows, g_dim_v, g_scale);
            get_i8(tu, uw, urs, u_rows, u_dim_v, u_scale);
            int rows = g_rows, dim_w = g_dim_v;
#ifndef __aarch64__
            int rows_packed = rows / 4;

            #ifdef _OPENMP
            #pragma omp parallel for if(rows_packed > 4)
            #endif
            for (int ur = 0; ur < rows_packed; ur++) {
                const int8_t* gw4 = gw + ur * 4 * dim_w;
                const int8_t* uw4 = uw + ur * 4 * dim_w;
                for (int b = 0; b < B; b++) {
                    const float* a = m->buf_act + b * ffn_dim;
                    float g_val[4], u_val[4];
                    for (int sub = 0; sub < 4; sub++) {
                        const int8_t* wg = gw4 + sub * dim_w;
                        const int8_t* wu = uw4 + sub * dim_w;
                        __m256 gs = _mm256_setzero_ps();
                        __m256 us = _mm256_setzero_ps();
                        int c = 0;
                        for (; c + 8 <= dim_w; c += 8) {
                            __m256 af = _mm256_loadu_ps(a + c);
                            __m128i wg8 = _mm_loadl_epi64((const __m128i*)(wg + c));
                            __m128i wu8 = _mm_loadl_epi64((const __m128i*)(wu + c));
                            __m256 wg_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wg8));
                            __m256 wu_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wu8));
                            gs = _mm256_fmadd_ps(af, wg_f, gs);
                            us = _mm256_fmadd_ps(af, wu_f, us);
                        }
                        float gs_f = hsum_ps(gs);
                        float us_f = hsum_ps(us);
                        for (; c < dim_w; c++) {
                            gs_f += a[c] * wg[c];
                            us_f += a[c] * wu[c];
                        }
                        g_val[sub] = gs_f / g_scale;
                        u_val[sub] = us_f / u_scale;
                    }
                    float* g_out = m->buf_gate + b * rows;
                    float* u_out = m->buf_up + b * rows;
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
#else
            atlas_fused_gate_up_f32_neon(rows, dim_w, gw, uw, m->buf_act, ffn_dim,
                m->buf_gate, m->buf_up, B, g_scale, u_scale);
#endif
        }
        #ifdef ATLAS_DEBUG_MODE
        if (B <= 6 && H == 2048) {
            fprintf(stderr, "DBG_GATE_f32_b0: ");
            for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", m->buf_gate[i]);
            fprintf(stderr, "\n");
            fprintf(stderr, "DBG_UP_f32_b0: ");
            for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", m->buf_up[i]);
            fprintf(stderr, "\n");
            snprintf(dbg_path, sizeof(dbg_path), "dbg_L%d_GATE.bin", dbg_cur_layer);
            FILE* f = fopen(dbg_path, "wb");
            if (f) { fwrite(m->buf_gate, sizeof(float), B * m->inter_dim, f); fclose(f); }
            snprintf(dbg_path, sizeof(dbg_path), "dbg_L%d_UP.bin", dbg_cur_layer);
            f = fopen(dbg_path, "wb");
            if (f) { fwrite(m->buf_up, sizeof(float), B * m->inter_dim, f); fclose(f); }
        }
        #endif
    } else {
        quantize_f32_to_u8(m->buf_act, B, ffn_dim, max_abs, m->buf_i8);

        int8_t* gw; int32_t* grs; int g_rows, g_dim_v; float g_scale;
        int8_t* uw; int32_t* urs; int u_rows, u_dim_v; float u_scale;
        get_i8(tg, gw, grs, g_rows, g_dim_v, g_scale);
        get_i8(tu, uw, urs, u_rows, u_dim_v, u_scale);
        int rows = g_rows, dim_w = g_dim_v;
#ifndef __aarch64__
        int rows_packed = rows / 4;
        const int GATE_TILE_B = 8;

        for (int t0 = 0; t0 < B; t0 += GATE_TILE_B) {
            int t_end = t0 + GATE_TILE_B;
            if (t_end > B) t_end = B;
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 32)
            #endif
            for (int ur = 0; ur < rows_packed; ur++) {
                const int8_t* gw4 = gw + ur * 4 * dim_w;
                const int8_t* uw4 = uw + ur * 4 * dim_w;
                int32_t g_off[4] = {128 * grs[ur*4+0], 128 * grs[ur*4+1],
                                    128 * grs[ur*4+2], 128 * grs[ur*4+3]};
                int32_t u_off[4] = {128 * urs[ur*4+0], 128 * urs[ur*4+1],
                                    128 * urs[ur*4+2], 128 * urs[ur*4+3]};

                for (int b = t0; b < t_end; b++) {
                    const uint8_t* a = m->buf_i8 + b * ffn_dim;
                    float deq = max_abs[b] / 127.0f;

                float g_val[4], u_val[4];
                for (int sub = 0; sub < 4; sub++) {
                    const int8_t* wg = gw4 + sub * dim_w;
                    const int8_t* wu = uw4 + sub * dim_w;
                    int c = 0, g_dot = 0, u_dot = 0;

                    __m256i g_acc = _mm256_setzero_si256();
                    __m256i u_acc = _mm256_setzero_si256();
                    for (; c + 32 <= dim_w; c += 32) {
                        __m256i au = _mm256_loadu_si256((const __m256i*)(a + c));
                        __m256i g_wv = _mm256_loadu_si256((const __m256i*)(wg + c));
                        __m256i u_wv = _mm256_loadu_si256((const __m256i*)(wu + c));
                        g_acc = _mm256_add_epi32(g_acc,
                            _mm256_madd_epi16(_mm256_maddubs_epi16(au, g_wv), _mm256_set1_epi16(1)));
                        u_acc = _mm256_add_epi32(u_acc,
                            _mm256_madd_epi16(_mm256_maddubs_epi16(au, u_wv), _mm256_set1_epi16(1)));
                    }

                    {
                        __m128i lo = _mm256_castsi256_si128(g_acc);
                        __m128i hi = _mm256_extracti128_si256(g_acc, 1);
                        __m128i s = _mm_add_epi32(lo, hi);
                        s = _mm_hadd_epi32(s, s);
                        s = _mm_hadd_epi32(s, s);
                        g_dot = _mm_cvtsi128_si32(s);
                    }
                    {
                        __m128i lo = _mm256_castsi256_si128(u_acc);
                        __m128i hi = _mm256_extracti128_si256(u_acc, 1);
                        __m128i s = _mm_add_epi32(lo, hi);
                        s = _mm_hadd_epi32(s, s);
                        s = _mm_hadd_epi32(s, s);
                        u_dot = _mm_cvtsi128_si32(s);
                    }

                    for (; c < dim_w; c++) {
                        g_dot += (int)a[c] * (int)wg[c];
                        u_dot += (int)a[c] * (int)wu[c];
                    }

                    g_dot -= g_off[sub]; u_dot -= u_off[sub];
                    g_val[sub] = (float)g_dot * deq / g_scale;
                    u_val[sub] = (float)u_dot * deq / u_scale;
                }

                float* g_out = m->buf_gate + b * rows;
                float* u_out = m->buf_up + b * rows;
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
#else
        atlas_fused_gate_up_default_neon(rows, dim_w, gw, uw, m->buf_i8, max_abs, B,
            m->buf_gate, m->buf_up, g_scale, u_scale);
#endif
    }
    P_ACCUM(ffn_gate_up);
    P_START(silu_down);

    // ─── 8. Fused SiLU(gate)*up → down matmul with optional ffn_sub_norm ───
    {
        auto& td = m->tensors[idx_down];
        if (td.ttype == 10) {
            int down_dim = td.packed_cols * 5;
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_ffn_f32 + b * down_dim;
                for (int i = 0; i < inter; i++) {
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                }
                for (int i = inter; i < down_dim; i++) tmp[i] = 0.0f;
            }
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_ffn_f32 + b * down_dim;
                    float ss = 0.0f;
                    for (int i = 0; i < inter; i++) ss += ob[i] * ob[i];
                    float rms = 1.0f / sqrtf(ss / inter + 1e-6f);
                    for (int i = 0; i < inter; i++) {
                        uint16_t w16; memcpy(&w16, snw + i * 2, 2);
                        ob[i] *= rms * fp16_to_fp32(w16);
                    }
                }
            }
            if (m->use_f32_matmul) {
                matmul_tq2_f32(td.row_dim, down_dim, td.packed_cols, td.data, td.block_size, td.n_blocks, m->buf_ffn_f32, m->buf_gate, B);
            } else if (B >= LUT_THRESHOLD) {
                tq1_lut_prefill_kernel(td.row_dim, down_dim, td.packed_cols, td.data + 1, td.block_size, td.n_blocks, m->buf_ffn_f32, m->buf_gate, B);
            } else {
                matmul_tq2(td.row_dim, down_dim, td.packed_cols, td.data, td.block_size, td.n_blocks, m->buf_ffn_f32, m->buf_gate, B);
            }
        } else if (td.ttype == 5) {
            int down_dim = td.packed_cols * 5;
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_ffn_f32 + b * down_dim;
                for (int i = 0; i < inter; i++) {
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                }
                for (int i = inter; i < down_dim; i++) tmp[i] = 0.0f;
            }
            // ffn_sub_norm: normalize intermediate BEFORE down_proj
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_ffn_f32 + b * down_dim;
                    float ss = 0.0f;
                    for (int i = 0; i < inter; i++) ss += ob[i] * ob[i];
                    float rms = 1.0f / sqrtf(ss / inter + 1e-6f);
                    for (int i = 0; i < inter; i++) {
                        uint16_t w16; memcpy(&w16, snw + i * 2, 2);
                        ob[i] *= rms * fp16_to_fp32(w16);
                    }
                }
            }
            if (m->use_f32_matmul) {
                matmul_tq1_block_fused_f32(td.row_dim, down_dim, td.packed_cols,
                    td.data, td.block_size, td.n_blocks,
                    m->buf_ffn_f32, m->buf_gate, B);
            } else if (B >= LUT_THRESHOLD && td.block_size > 0 && td.n_blocks > 0) {
                tq1_lut_prefill_kernel(td.row_dim, down_dim, td.packed_cols,
                    td.data, td.block_size, td.n_blocks,
                    m->buf_ffn_f32, m->buf_gate, B);
            } else {
                matmul_tq1_block_fused_s8(td.row_dim, down_dim, td.packed_cols,
                    td.data, td.block_size, td.n_blocks,
                    m->buf_ffn_f32, m->buf_gate, B);
            }
        } else if (td.ttype == 7) {
            int down_dim = td.packed_cols * 4;
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_ffn_f32 + b * down_dim;
                for (int i = 0; i < inter; i++) {
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                }
                for (int i = inter; i < down_dim; i++) tmp[i] = 0.0f;
            }
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_ffn_f32 + b * down_dim;
                    float ss = 0.0f;
                    for (int i = 0; i < inter; i++) ss += ob[i] * ob[i];
                    float rms = 1.0f / sqrtf(ss / inter + 1e-6f);
                    for (int i = 0; i < inter; i++) {
                        uint16_t w16; memcpy(&w16, snw + i * 2, 2);
                        ob[i] *= rms * fp16_to_fp32(w16);
                    }
                }
            }
            matmul_turboquant_fused(td.row_dim, down_dim, td.packed_cols,
                td.data, td.block_size, td.n_blocks,
                m->buf_ffn_f32, m->buf_gate, B);
        } else if (td.ttype == 0) {
            const uint8_t* wp; int rows, dim, pc; float scale;
            get_tq1_packed(td, wp, rows, dim, pc, scale);
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_ffn_f32 + b * dim;
                for (int i = 0; i < inter; i++) {
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                }
                for (int i = inter; i < dim; i++) tmp[i] = 0.0f;
            }
            // ffn_sub_norm before down_proj
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_ffn_f32 + b * dim;
                    float ss = 0.0f;
                    for (int i = 0; i < inter; i++) ss += ob[i] * ob[i];
                    float rms = 1.0f / sqrtf(ss / inter + 1e-6f);
                    for (int i = 0; i < inter; i++) {
                        uint16_t w16; memcpy(&w16, snw + i * 2, 2);
                        ob[i] *= rms * fp16_to_fp32(w16);
                    }
                }
            }
            float mb = 1e-5f;
            for (int b = 0; b < B; b++) {
                const float* tmp = m->buf_ffn_f32 + b * dim;
                float bmax = 1e-5f;
                for (int i = 0; i < inter; i++) {
                    float av = fabsf(tmp[i]);
                    if (av > bmax) bmax = av;
                }
                float inv = 127.0f / bmax;
                max_abs[b] = bmax;
                for (int i = 0; i < dim; i++) {
                    int q = (int)(tmp[i] * inv + 128.5f);
                    if (q < 0) q = 0; if (q > 255) q = 255;
                    m->buf_i8[b * dim + i] = (uint8_t)q;
                }
            }
            matmul_tq1_packed_reorder(rows, dim, wp, pc, m->buf_i8, max_abs, scale, m->buf_gate, B);
        } else if (td.ttype == 8 && !m->use_f32_matmul) {
            uint16_t d16; memcpy(&d16, td.data, 2); float d_scale = fp16_to_fp32(d16);
            int d_cols = td.packed_cols * 2;
            const uint8_t* d_pw = td.data + 2;
            const int32_t* d_rs = (const int32_t*)(td.data + 2 + td.row_dim * td.packed_cols);
            int down_dim = d_cols;
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_ffn_f32 + b * down_dim;
                for (int i = 0; i < inter; i++) {
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                }
                for (int i = inter; i < down_dim; i++) tmp[i] = 0.0f;
            }
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_ffn_f32 + b * down_dim;
                    float ss = 0.0f;
                    for (int i = 0; i < inter; i++) ss += ob[i] * ob[i];
                    float rms = 1.0f / sqrtf(ss / inter + 1e-6f);
                    for (int i = 0; i < inter; i++) {
                        uint16_t w16; memcpy(&w16, snw + i * 2, 2);
                        ob[i] *= rms * fp16_to_fp32(w16);
                    }
                }
            }
            quantize_f32_to_u8(m->buf_ffn_f32, B, down_dim, max_abs, m->buf_i8);
            matmul_i4_reorder_deq(td.row_dim, d_cols, d_pw, d_rs, m->buf_i8, max_abs, d_scale, m->buf_act, m->buf_gate, B);
        } else if (td.ttype == 8 && m->use_f32_matmul) {
#ifndef __aarch64__
            uint16_t d_s16; memcpy(&d_s16, td.data, 2); float d_sc = fp16_to_fp32(d_s16);
            int d_cols = td.packed_cols * 2;
            const uint8_t* d_pw = td.data + 2;
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_ffn_f32 + b * d_cols;
                for (int i = 0; i < inter; i++)
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                for (int i = inter; i < d_cols; i++) tmp[i] = 0.0f;
            }
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_ffn_f32 + b * d_cols;
                    float ss = 0.0f;
                    for (int i = 0; i < inter; i++) ss += ob[i] * ob[i];
                    float rms = 1.0f / sqrtf(ss / inter + 1e-6f);
                    for (int i = 0; i < inter; i++) {
                        uint16_t w16; memcpy(&w16, snw + i * 2, 2);
                        ob[i] *= rms * fp16_to_fp32(w16);
                    }
                }
            }
            int d_rows = td.row_dim;
            int d_cols_i = d_cols;
            int d_rows_packed = d_rows / 4;
            int d_packed_cols = td.packed_cols;
            const uint8_t* dw = td.data + 2;
            const __m128i mask_low_d = _mm_set1_epi8(0x0F);
            const __m128i xor8_d = _mm_set1_epi8(8);
            const __m128i sub8_d = _mm_set1_epi8(8);
            #ifdef _OPENMP
            #pragma omp parallel for if(d_rows_packed > 4)
            #endif
            for (int ur = 0; ur < d_rows_packed; ur++) {
                const uint8_t* dw4 = dw + ur * 4 * d_packed_cols;
                for (int b = 0; b < B; b++) {
                    const float* a = m->buf_ffn_f32 + b * d_cols_i;
                    float out4[4];
                    for (int sub = 0; sub < 4; sub++) {
                        const uint8_t* w = dw4 + sub * d_packed_cols;
                        __m256 acc = _mm256_setzero_ps();
                        int c = 0;
                        for (; c + 8 <= d_cols_i; c += 8) {
                            __m256 af = _mm256_loadu_ps(a + c);
                            uint32_t p4;
                            memcpy(&p4, w + c / 2, 4);
                            __m128i nib = _mm_cvtsi32_si128((int)p4);
                            __m128i lo = _mm_and_si128(nib, mask_low_d);
                            __m128i hi = _mm_and_si128(_mm_srli_epi16(nib, 4), mask_low_d);
                            __m128i w8 = _mm_unpacklo_epi8(lo, hi);
                            __m128i ws = _mm_sub_epi8(_mm_xor_si128(w8, xor8_d), sub8_d);
                            __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ws));
                            acc = _mm256_fmadd_ps(af, wf, acc);
                        }
                        float s = hsum_ps(acc);
                        for (; c < d_cols_i; c++) {
                            int8_t wv = (c & 1) ? (int8_t)((((w[c / 2] >> 4) & 0x0F) ^ 8) - 8)
                                                : (int8_t)(((w[c / 2] & 0x0F) ^ 8) - 8);
                            s += a[c] * (float)wv;
                        }
                        out4[sub] = s / d_sc;
                    }
                    float* dst = m->buf_gate + b * d_rows;
                    dst[0 * d_rows_packed + ur] = out4[0];
                    dst[1 * d_rows_packed + ur] = out4[1];
                    dst[2 * d_rows_packed + ur] = out4[2];
                    dst[3 * d_rows_packed + ur] = out4[3];
                }
            }
#endif
        } else if (td.ttype == 11) {
#ifndef __aarch64__
            // Per-row int8 down_proj (ttype=11)
            const uint16_t* d_rs_fp16 = (const uint16_t*)(td.data);
            const int8_t* dw = (const int8_t*)(td.data + td.row_dim * 2);
            int d11_dim = t3_dim(td);
            int d11_rows = td.row_dim;
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_ffn_f32 + b * d11_dim;
                for (int i = 0; i < inter; i++)
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                for (int i = inter; i < d11_dim; i++) tmp[i] = 0.0f;
            }
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_ffn_f32 + b * d11_dim;
                    float ss = 0.0f;
                    for (int i = 0; i < inter; i++) ss += ob[i] * ob[i];
                    float rms = 1.0f / sqrtf(ss / inter + 1e-6f);
                    for (int i = 0; i < inter; i++) {
                        uint16_t w16; memcpy(&w16, snw + i * 2, 2);
                        ob[i] *= rms * fp16_to_fp32(w16);
                    }
                }
            }
            #ifdef ATLAS_DEBUG_MODE
            if (B <= 6 && H == 2048) {
                fprintf(stderr, "DBG_SILUUP_f32_b0: ");
                for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", m->buf_ffn_f32[i]);
                fprintf(stderr, "\n");
                snprintf(dbg_path, sizeof(dbg_path), "dbg_L%d_SILUUP.bin", dbg_cur_layer);
                FILE* f = fopen(dbg_path, "wb");
                if (f) { fwrite(m->buf_ffn_f32, sizeof(float), B * m->inter_dim, f); fclose(f); }
            }
            #endif
            matmul_f32_per_row(d11_rows, d11_dim, dw, m->buf_ffn_f32, d11_dim, d_rs_fp16, m->buf_gate, B);
            #ifdef ATLAS_DEBUG_MODE
            if (B <= 6 && H == 2048) {
                snprintf(dbg_path, sizeof(dbg_path), "dbg_L%d_DOWN.bin", dbg_cur_layer);
                FILE* f = fopen(dbg_path, "wb");
                if (f) { fwrite(m->buf_gate, sizeof(float), B * d11_rows, f); fclose(f); }
            fprintf(stderr, "DBG_DOWN_b0: ");
            for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", m->buf_gate[i]);
            fprintf(stderr, "\n");
            }
            #endif
#endif
        } else {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(td, w, rs, rows, dim, scale);
            if (m->use_ternary_matmul) {
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_ffn_f32 + b * dim;
                for (int i = 0; i < inter; i++) {
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                }
                for (int i = inter; i < dim; i++) tmp[i] = 0.0f;
            }
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_ffn_f32 + b * dim;
                    float ss = 0.0f;
                    for (int i = 0; i < inter; i++) ss += ob[i] * ob[i];
                    float rms = 1.0f / sqrtf(ss / inter + 1e-6f);
                    for (int i = 0; i < inter; i++) {
                        uint16_t w16; memcpy(&w16, snw + i * 2, 2);
                        ob[i] *= rms * fp16_to_fp32(w16);
                    }
                }
            }
            float mb = 1e-5f;
            for (int b = 0; b < B; b++) {
                const float* tmp = m->buf_ffn_f32 + b * dim;
                float bmax = 1e-5f;
                for (int i = 0; i < inter; i++) {
                    float av = fabsf(tmp[i]);
                    if (av > bmax) bmax = av;
                }
                float inv = 127.0f / bmax;
                max_abs[b] = bmax;
                for (int i = 0; i < dim; i++) {
                    int q = (int)(tmp[i] * inv + 128.5f);
                    if (q < 0) q = 0; if (q > 255) q = 255;
                    m->buf_i8[b * dim + i] = (uint8_t)q;
                }
            }
            matmul_ternary_add_reorder(rows, dim, w, m->buf_i8, max_abs,
                                      scale, m->buf_gate, B);
        } else if (m->use_f32_matmul) {
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_ffn_f32 + b * dim;
                for (int i = 0; i < inter; i++)
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                // removed: dead code (ARCH_BITNET never has idx_k_norm >= 0)
                for (int i = inter; i < dim; i++) tmp[i] = 0.0f;
            }
            #ifdef ATLAS_DEBUG_MODE
            if (B <= 6 && H == 2048) {
                fprintf(stderr, "DBG_SILUUP_f32_b0: ");
                for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", m->buf_ffn_f32[i]);
                fprintf(stderr, "\n");
                snprintf(dbg_path, sizeof(dbg_path), "dbg_L%d_SILUUP.bin", dbg_cur_layer);
                FILE* f = fopen(dbg_path, "wb");
                if (f) { fwrite(m->buf_ffn_f32, sizeof(float), B * m->inter_dim, f); fclose(f); }
            }
            #endif
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_ffn_f32 + b * dim;
                    float ss = 0.0f;
                    for (int i = 0; i < inter; i++) ss += ob[i] * ob[i];
                    float rms = 1.0f / sqrtf(ss / inter + 1e-6f);
                    for (int i = 0; i < inter; i++) {
                        uint16_t w16; memcpy(&w16, snw + i * 2, 2);
                        ob[i] *= rms * fp16_to_fp32(w16);
                    }
                }
            }
            matmul_f32_reorder(rows, dim, w, m->buf_ffn_f32, scale, m->buf_gate, B);
            #ifdef ATLAS_DEBUG_MODE
            if (B <= 6 && H == 2048) {
                fprintf(stderr, "DBG_DOWN_f32_b0: ");
                for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", m->buf_gate[i]);
                fprintf(stderr, "\n");
                snprintf(dbg_path, sizeof(dbg_path), "dbg_L%d_DOWN.bin", dbg_cur_layer);
                FILE* f = fopen(dbg_path, "wb");
                if (f) { fwrite(m->buf_gate, sizeof(float), B * H, f); fclose(f); }
            }
            #endif
        } else {
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_ffn_f32 + b * dim;
                for (int i = 0; i < inter; i++) {
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                }
                for (int i = inter; i < dim; i++) tmp[i] = 0.0f;
            }
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_ffn_f32 + b * dim;
                    float ss = 0.0f;
                    for (int i = 0; i < inter; i++) ss += ob[i] * ob[i];
                    float rms = 1.0f / sqrtf(ss / inter + 1e-6f);
                    for (int i = 0; i < inter; i++) {
                        uint16_t w16; memcpy(&w16, snw + i * 2, 2);
                        ob[i] *= rms * fp16_to_fp32(w16);
                    }
                }
            }
            float mb = 1e-5f;
            for (int b = 0; b < B; b++) {
                const float* tmp = m->buf_ffn_f32 + b * dim;
                float bmax = 1e-5f;
                for (int i = 0; i < inter; i++) {
                    float av = fabsf(tmp[i]);
                    if (av > bmax) bmax = av;
                }
                float inv = 127.0f / bmax;
                max_abs[b] = bmax;
                for (int i = 0; i < dim; i++) {
                    int q = (int)(tmp[i] * inv + 128.5f);
                    if (q < 0) q = 0; if (q > 255) q = 255;
                    m->buf_i8[b * dim + i] = (uint8_t)q;
                }
            }
            matmul_reorder_deq(rows, dim, w, rs, m->buf_i8, max_abs,
                              scale, m->buf_hidden, m->buf_gate, B);
        }
        }
    }

    // ─── 10. Residual: output += down_proj (* scale_depth_factor) ───
    for (int i = 0; i < B * H; i++) {
        output[i] += m->buf_gate[i] * m->scale_depth_factor;
    }
    #ifdef ATLAS_DEBUG_MODE
    if (B <= 6 && H == 2048) {
        fprintf(stderr, "DBG_FINAL_b0: ");
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", output[i]);
        fprintf(stderr, "\n");
        snprintf(dbg_path, sizeof(dbg_path), "dbg_L%d_FINAL.bin", dbg_cur_layer);
        FILE* f = fopen(dbg_path, "wb");
        if (f) { fwrite(output, sizeof(float), B * H, f); fclose(f); }
    }
    #endif
    PROF_ADD_LAYERS(1);
    P_ACCUM(silu_down);
}


// ─── Forward ALL transformer layers in one C call ────────────────────
// (single-layer atlas_forward_layer removed — fusion is always used)
// hidden_states: [B, H] float32 — overwritten with final layer output
// positions: [B] int32 position indices
// layer_idx: [n_layers * 9] int32 — flat array of tensor indices per layer
//           (ln1, q, k, v, o, ln2, gate, up, down) repeated for each layer
// K/V cache is internal to the model (int8 + per-position scales)
// Returns 0 on success, -1 on allocation failure.
ATLAS_API int atlas_forward(
    AtlasModel* m,
    float* hidden_states, int B,
    const int* positions,
    int max_seq_len, int seq_now,
    const int* layer_idx, int n_layers) {

    if (!m->ensure_buffers(B)) return -1;
    if (!m->ensure_cache(max_seq_len)) return -1;
    int H = m->hidden_dim;
    ATLAS_LOG("atlas_forward: B=%d max_seq=%d seq_now=%d H=%d nKV=%d hd=%d n_layers=%d stride=%d\n",
              B, max_seq_len, seq_now, H, m->n_kv_heads, m->head_dim, n_layers, m->layer_stride);
    int nKV = m->n_kv_heads, hd = m->head_dim;

    // Ping-pong: layer N output goes to separate buf_out (not buf_hidden, which is scratch)
    float* buf_a = hidden_states;
    float* buf_b = m->buf_out;

    for (int L = 0; L < n_layers; L++) {
        const int* idx = layer_idx + L * m->layer_stride;
        uint8_t* kc = m->k_cache + (size_t)L * nKV * max_seq_len * kv_pos_bytes(hd);
        uint8_t* vc = m->v_cache + (size_t)L * nKV * max_seq_len * kv_pos_bytes(hd);
        int qn_i = -1, kn_i = -1, asn_i = -1, fsn_i = -1;
        if (m->layer_stride >= 11) {
            if (m->model_arch == ARCH_BITNET) {
                asn_i = idx[9];
                fsn_i = idx[10];
            } else {
                qn_i = idx[9];
                kn_i = idx[10];
            }
        }
        ATLAS_LOG("forward L=%d: B=%d seq_now=%d max_seq=%d | idx[0..8]=%d %d %d %d %d %d %d %d %d\n",
                  L, B, seq_now, max_seq_len,
                  idx[0], idx[1], idx[2], idx[3], idx[4],
                  idx[5], idx[6], idx[7], idx[8]);
        #ifdef ATLAS_DEBUG_MODE
        double tL = atlas_now();
        #endif
        if (m->is_mla) {
            forward_layer_internal_mla(m, buf_a, buf_b, B, positions,
                max_seq_len, seq_now, L);
        } else {
            forward_layer_internal(m, buf_a, buf_b, B, positions,
                kc, vc, max_seq_len, seq_now,
                idx[0], idx[1], idx[2], idx[3], idx[4],
                idx[5], idx[6], idx[7], idx[8],
                qn_i, kn_i, asn_i, fsn_i);
        }
        #ifdef ATLAS_DEBUG_MODE
        double tLend = atlas_now() - tL;
        if (tLend > 0.1) ATLAS_LOG("[TIMER] L=%d: %.3fs\n", L, tLend);
        #endif
        ATLAS_LOG("forward L=%d done\n", L);
        float* tmp = buf_a; buf_a = buf_b; buf_b = tmp;
    }

    // If odd number of layers, final output is in buf_a after the last swap,
    // not in hidden_states. Copy it back.
    if (n_layers % 2 == 1) {
        memcpy(hidden_states, buf_a, (size_t)B * H * sizeof(float));
    }
    return 0;
}

// ─── Quantize lm_head from fp16 to per-row symmetric int8 ─────────────
// Reads the fp16 tensor at idx, quantizes each row to int8, stores in
// AtlasModel fields. Frees the fp16 data (saves 768 MB).
// Call after Python has created its fp32 copy (never before step 4 above).
ATLAS_API void atlas_quantize_lmhead(AtlasModel* m, int idx, int keep_data) {
    if (!m || idx < 0 || idx >= (int)m->tensors.size()) return;
    auto& t = m->tensors[idx];
    if ((t.ttype != 2 && t.ttype != 1) || t.data == nullptr) return;

    int V = t.row_dim;
    int H = m->hidden_dim;
    int64_t n_vals = (int64_t)V * H;

    int8_t* i8 = (int8_t*)atlas_valloc((size_t)n_vals);
    int32_t* offs = (int32_t*)atlas_valloc((size_t)V * sizeof(int32_t));
    float* scales = (float*)atlas_valloc((size_t)V * sizeof(float));
    if (!i8 || !offs || !scales) {
        if (i8) atlas_vfree((uint8_t*)i8);
        if (offs) atlas_vfree((uint8_t*)offs);
        if (scales) atlas_vfree((uint8_t*)scales);
        fprintf(stderr, "[ATLAS] OOM quantizing lm_head (%d x %d)\n", V, H);
        return;
    }

    uint16_t* fp16 = (uint16_t*)t.data;

    for (int r = 0; r < V; r++) {
        float max_abs = 1e-5f;
        for (int c = 0; c < H; c++) {
            float v = fp16_to_fp32(fp16[r * H + c]);
            float av = fabsf(v);
            if (av > max_abs) max_abs = av;
        }
        float inv = max_abs / 127.0f;
        scales[r] = inv;
        int32_t row_sum = 0;
        for (int c = 0; c < H; c++) {
            float v = fp16_to_fp32(fp16[r * H + c]);
            int q = (int)safe_int_from_float(v / inv + 0.5f);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            i8[r * H + c] = (int8_t)q;
            row_sum += q;
        }
        offs[r] = 128 * row_sum;
    }

    if (!keep_data) {
        if (!m->is_mapped(t.data)) atlas_vfree(t.data);
        t.data = nullptr;
        t.data_size = 0;
    }

    m->lm_head_i8 = i8;
    m->lm_head_offsets = offs;
    m->lm_head_scales = scales;
    m->lm_head_quantized = true;

    float mb = (float)(n_vals + (int64_t)V * 6) / (1024.0f * 1024.0f);
    printf("[ATLAS] Quantized lm_head: %d × %d = %.1f MB int8\n", V, H, mb);
}

// ─── GEMV: int8 lm_head × u8 quantized activations ────────────────────
// Quantizes B hidden-state vectors [B, H] to u8, then computes full vocab
// dot products with per-row dequant. AVX2 maddubs + offset trick.
//
//   out[b][v] = (Σ_h act_u8[b][h] * W_i8[v][h] - offset[v])
//               * (max_abs_act[b] / 127) * (weight_scale[v])
//
// act: [B, H] float32    output: [B, V] float32
ATLAS_API void atlas_lmhead_gemv(AtlasModel* m, const float* act,
                                  float* output, int B) {
    if (!m || !m->lm_head_quantized) return;
    int V = m->vocab_size;
    int H = m->hidden_dim;

    // Quantize activations to u8
    uint8_t* act_u8 = (uint8_t*)atlas_valloc((size_t)B * H);
    float* max_abs = (float*)atlas_valloc((size_t)B * sizeof(float));
    if (!act_u8 || !max_abs) {
        if (act_u8) atlas_vfree((uint8_t*)act_u8);
        if (max_abs) atlas_vfree((uint8_t*)max_abs);
        fprintf(stderr, "[ATLAS] OOM quantizing activations in lm_head gemv\n");
        return;
    }

    for (int b = 0; b < B; b++) {
        float ma = 1e-5f;
        for (int i = 0; i < H; i++) {
            float v = fabsf(act[b * H + i]);
            if (v > ma) ma = v;
        }
        max_abs[b] = ma;
        float inv = (ma > 1e-10f) ? 127.0f / ma : 0.0f;
        for (int i = 0; i < H; i++) {
            int q = (int)safe_int_from_float(act[b * H + i] * inv + 128.5f);
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            act_u8[b * H + i] = (uint8_t)q;
        }
    }

    const int8_t* w = m->lm_head_i8;
    const int32_t* offs = m->lm_head_offsets;
    const float* scales = m->lm_head_scales;

    #ifndef __aarch64__
    // x86 path: _mm256_maddubs_epi16 + offset correction
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int r = 0; r < V; r++) {
        const int8_t* wr = w + r * H;
        int32_t off = offs[r];
        float s = scales[r];

        for (int b = 0; b < B; b++) {
            const uint8_t* a = act_u8 + b * H;
            int c = 0;
            int dot = 0;
            __m256i acc = _mm256_setzero_si256();

            for (; c + 32 <= H; c += 32) {
                __m256i au = _mm256_loadu_si256((const __m256i*)(a + c));
                __m256i wv = _mm256_loadu_si256((const __m256i*)(wr + c));
                __m256i prod16 = _mm256_maddubs_epi16(au, wv);
                __m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));
                acc = _mm256_add_epi32(acc, prod32);
            }

            __m128i lo = _mm256_castsi256_si128(acc);
            __m128i hi = _mm256_extracti128_si256(acc, 1);
            __m128i sum128 = _mm_add_epi32(lo, hi);
            sum128 = _mm_hadd_epi32(sum128, sum128);
            sum128 = _mm_hadd_epi32(sum128, sum128);
            dot = _mm_cvtsi128_si32(sum128);

            for (; c < H; c++) {
                dot += (int)a[c] * (int)wr[c];
            }

            dot -= off;
            output[b * V + r] = (float)dot * (max_abs[b] / 127.0f) * s;
        }
    }
    #else
    // ARM64 NEON path: XOR-0x80 + vdotq_s32 (eliminates offset subtraction)
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int r = 0; r < V; r++) {
        const int8_t* wr = w + r * H;
        float s = scales[r];

        for (int b = 0; b < B; b++) {
            const uint8_t* a = act_u8 + b * H;
            int c = 0;
            int32x4_t acc = vdupq_n_s32(0);
            uint8x16_t xor_80 = vdupq_n_u8(0x80);

            for (; c + 32 <= H; c += 32) {
                uint8x16_t au0 = vld1q_u8(a + c);
                uint8x16_t au1 = vld1q_u8(a + c + 16);
                int8x16_t wv0 = vld1q_s8(wr + c);
                int8x16_t wv1 = vld1q_s8(wr + c + 16);

                int8x16_t a0 = vreinterpretq_s8_u8(veorq_u8(au0, xor_80));
                int8x16_t a1 = vreinterpretq_s8_u8(veorq_u8(au1, xor_80));

                acc = vdotq_s32(acc, a0, wv0);
                acc = vdotq_s32(acc, a1, wv1);
            }

            int32_t dot = vaddvq_s32(acc);

            for (; c < H; c++) {
                dot += ((int)a[c] ^ 0x80) * (int)wr[c];
            }

            output[b * V + r] = (float)dot * (max_abs[b] / 127.0f) * s;
        }
    }
    #endif

    atlas_vfree((uint8_t*)act_u8);
    atlas_vfree((uint8_t*)max_abs);
}

// ─── v1.2.0: Seed PRNG ─────────────────────────────────────────────────
ATLAS_API void atlas_set_seed(uint64_t seed) {
    xoshiro_seed(seed);
}

// ─── v1.2.0: Sample one token via Gumbel-max ──────────────────────────
ATLAS_API void atlas_sample(AtlasModel* m, float* logits, int* output,
                             float temperature, int top_k, float top_p) {
    (void)m;
    if (!output || !logits) return;
    *output = gumbel_sample(logits, m ? m->vocab_size : (151669),
                             temperature, top_k, top_p,
                             nullptr, 0, 1.0f);
}

// ─── v1.2.0: End-to-end generation (single C call) ───────────────────
// Builds layer index array from tensor names (cached after first call).
static void ensure_layer_idx(AtlasModel* m) {
    if (m->has_layer_idx) return;
    auto& names = m->tensor_names;
    auto find = [&](const std::string& n) -> int {
        for (int i = 0; i < (int)names.size(); i++)
            if (names[i] == n) return i;
        return -1;
    };

    int stride = 9;
    int model_arch = ARCH_LLAMA;
    {
        char test_name[128];
        snprintf(test_name, sizeof(test_name), "model.layers.0.self_attn.attn_sub_norm.weight");
        if (find(test_name) >= 0) { stride = 11; model_arch = ARCH_BITNET; }
        else {
            snprintf(test_name, sizeof(test_name), "model.layers.0.self_attn.q_norm.weight");
            if (find(test_name) >= 0) { stride = 11; model_arch = ARCH_QWEN3; }
        }
    }
    // If meta set a stride, prefer it (known arch was recognized)
    if (m->has_meta && m->layer_stride > 9) {
        stride = m->layer_stride;
        model_arch = m->model_arch;
    }
    m->layer_stride = stride;
    m->model_arch = model_arch;
    // RoPE format: interleaved for Qwen3 and Falcon3 (head_dim>=256), half-split for Llama/BitNet
    // Respect config.json override (v6+ meta-block): if rope_interleaved was explicitly set,
    // the heuristic cannot override it.
    if (!m->rope_interleaved_set) {
        m->rope_interleaved = (m->head_dim >= 256) && (model_arch != ARCH_QWEN3);
    }
    if (model_arch == ARCH_BITNET) m->use_f32_matmul = 1;
    m->layer_idx_cache.clear();
    m->layer_idx_cache.reserve(m->n_layers * stride);
    for (int L = 0; L < m->n_layers; L++) {
        char buf[128];
        auto push = [&](const char* suffix) {
            snprintf(buf, sizeof(buf), "model.layers.%d.%s", L, suffix);
            m->layer_idx_cache.push_back(find(buf));
        };
        if (model_arch == ARCH_BITNET) {
            push("input_layernorm.weight");
            push("self_attn.q_proj.weight");
            push("self_attn.k_proj.weight");
            push("self_attn.v_proj.weight");
            push("self_attn.o_proj.weight");
            push("post_attention_layernorm.weight");
            push("mlp.gate_proj.weight");
            push("mlp.up_proj.weight");
            push("mlp.down_proj.weight");
            push("self_attn.attn_sub_norm.weight");
            push("mlp.ffn_sub_norm.weight");
        } else if (stride > 9 && m->is_mla) {
            // ─── MLA layer index: dynamic stride ───
            // V2-Lite: stride = 6 + 3*n_shared + 2
            // V3 (+q_lora_rank): stride += 4 (q_b, q_a_layernorm, o_proj.1, o_proj.2)
            // [0] ln1, [1] q (q_a_proj or q_proj), [2] kv_a, [3] kv_b, [4] o (o_proj.0 or o_proj), [5] ln2
            // [6..6+3*n_shared-1] shared experts (gate/up/down per expert)
            // [6+3*n_shared] kv_a_layernorm, [6+3*n_shared+1] router
            // [6+3*n_shared+2] q_b (V3 only)
            // [6+3*n_shared+3] q_a_layernorm (V3 only)
            // [6+3*n_shared+4] o_proj.1 bias (V3 only)
            // [6+3*n_shared+5] o_proj.2 up (V3 only)
            int ns = m->n_shared_experts > 0 ? m->n_shared_experts : 1;
            // Detect gated o_proj from tensor names
            if (m->q_lora_rank > 0 && !m->has_gated_o_proj) {
                for (int tn = 0; tn < (int)m->tensor_names.size(); tn++) {
                    if (m->tensor_names[tn].find("self_attn.o_proj.0.weight") != std::string::npos) {
                        m->has_gated_o_proj = true;
                        break;
                    }
                }
                // Adjust stride if gated o_proj detected (may differ from parse_meta_block estimate)
                if (m->has_gated_o_proj) {
                    int expected = 6 + 3 * ns + 2 + 4;
                    if (m->layer_stride < expected) {
                        printf("[ATLAS] V3 detected: extending MLA stride %d → %d (gated o_proj)\n",
                               m->layer_stride, expected);
                        m->layer_stride = expected;
                    }
                }
            }
            push("input_layernorm.weight");          // idx[0]
            if (m->q_lora_rank > 0) {
                push("self_attn.q_a_proj.weight");    // idx[1] V3: q_a
            } else {
                push("self_attn.q_proj.weight");      // idx[1] V2-Lite: q
            }
            push("self_attn.kv_a_proj_with_mqa.weight"); // idx[2]
            push("self_attn.kv_b_proj.weight");       // idx[3]
            if (m->has_gated_o_proj) {
                push("self_attn.o_proj.0.weight");    // idx[4] V3: gated o_proj gate
            } else {
                push("self_attn.o_proj.weight");      // idx[4] V2-Lite: simple o_proj
            }
            push("post_attention_layernorm.weight");  // idx[5]
            if (L < m->first_k_dense_replace) {
                // Dense layer: standard FFN (only 1 set, rest are dummy)
                push("mlp.gate_proj.weight");         // idx[6]
                push("mlp.up_proj.weight");           // idx[7]
                push("mlp.down_proj.weight");         // idx[8]
                for (int si = 1; si < ns; si++) {
                    m->layer_idx_cache.push_back(-1); // dummy for extra shared experts
                    m->layer_idx_cache.push_back(-1);
                    m->layer_idx_cache.push_back(-1);
                }
            } else {
                // MoE layer: push all shared expert tensors
                for (int si = 0; si < ns; si++) {
                    if (si == 0) {
                        push("mlp.shared_experts.gate_proj.weight");
                        push("mlp.shared_experts.up_proj.weight");
                        push("mlp.shared_experts.down_proj.weight");
                    } else {
                        // Additional shared experts: naming convention varies
                        push("mlp.shared_experts.gate_proj.weight");
                        push("mlp.shared_experts.up_proj.weight");
                        push("mlp.shared_experts.down_proj.weight");
                    }
                }
            }
            int kv_ln_idx = 6 + 3 * ns;
            push("self_attn.kv_a_layernorm.weight"); // idx[kv_ln_idx]
            if (L >= m->first_k_dense_replace) {
                push("mlp.gate.weight");              // idx[kv_ln_idx+1] router
            } else {
                m->layer_idx_cache.push_back(-1);     // dummy
            }
            // ─── V3 extra slots (after router) ───
            if (m->q_lora_rank > 0) {
                push("self_attn.q_b_proj.weight");       // idx[kv_ln_idx+2]
                push("self_attn.q_a_layernorm.weight");  // idx[kv_ln_idx+3]
            }
            if (m->has_gated_o_proj) {
                push("self_attn.o_proj.1.weight");       // idx[kv_ln_idx+4] bias
                push("self_attn.o_proj.2.weight");       // idx[kv_ln_idx+5] up
            }
        } else {
            push("input_layernorm.weight");
            push("self_attn.q_proj.weight");
            push("self_attn.k_proj.weight");
            push("self_attn.v_proj.weight");
            push("self_attn.o_proj.weight");
            push("post_attention_layernorm.weight");
            push("mlp.gate_proj.weight");
            push("mlp.up_proj.weight");
            push("mlp.down_proj.weight");
            if (stride >= 11) {
                push("self_attn.q_norm.weight");
                push("self_attn.k_norm.weight");
            }
        }
    }
    m->has_layer_idx = true;
}

ATLAS_API int atlas_generate(AtlasModel* m,
    const int* input_ids, int n_input,
    int max_seq_len, int max_new_tokens,
    float temperature, int top_k, float top_p,
    float repetition_penalty,
    int min_new_tokens,
    int cache_offset,
    int* output_ids,
    atlas_logit_processor_cb logit_cb, void* logit_cb_data,
    atlas_token_notify_cb token_notify_cb, void* token_notify_data)
{
    profile_reset();
    if (!m || !input_ids || !output_ids || n_input < 1 || max_new_tokens < 1)
        return -1;
    if (n_input >= max_seq_len) {
        fprintf(stderr, "[ATLAS] atlas_generate: prompt (%d) >= max_seq_len (%d), use shorter prompt\n",
                n_input, max_seq_len);
        return -1;
    }

    // Find required tensors by name
    int idx_norm = -1, idx_embed = -1;
    for (int i = 0; i < (int)m->tensor_names.size(); i++) {
        if (m->tensor_names[i] == "model.norm.weight") idx_norm = i;
        if (m->tensor_names[i] == "model.embed_tokens.weight") idx_embed = i;
    }
    if (idx_norm < 0 || idx_embed < 0) {
        fprintf(stderr, "[ATLAS] atlas_generate: missing norm/embed tensors\n");
        return -1;
    }

    int H = m->hidden_dim;
    int V = m->vocab_size;
    uint16_t* embed_w = (uint16_t*)m->tensors[idx_embed].data;
    uint8_t* norm_w = m->tensors[idx_norm].data;

    ATLAS_LOG("atlas_generate: L=%d H=%d I=%d nH=%d nKV=%d hd=%d V=%d stride=%d\n",
              m->n_layers, H, m->inter_dim, m->n_heads, m->n_kv_heads, m->head_dim, V,
              m->layer_stride);
    ATLAS_LOG("embed_w=%p norm_w=%p n_input=%d max_seq=%d max_new=%d\n",
              (void*)embed_w, (void*)norm_w, n_input, max_seq_len, max_new_tokens);
    for (int i = 0; i < (int)m->tensor_names.size(); i++) {
        ATLAS_LOG("tensor[%d]: %s ttype=%d data=%p sz=%d\n",
                  i, m->tensor_names[i].c_str(), m->tensors[i].ttype,
                  (void*)m->tensors[i].data, m->tensors[i].data_size);
    }

    if (!m->lm_head_quantized) {
        fprintf(stderr, "[ATLAS] atlas_generate: lm_head not quantized\n");
        return -1;
    }

    ensure_layer_idx(m);
    const int* layer_idx = m->layer_idx_cache.data();

    // Clamp cache_offset to valid range
    if (cache_offset < 0) cache_offset = 0;
    if (cache_offset >= n_input) cache_offset = n_input - 1;
    int n_new = n_input - cache_offset;

    // Allocate scratch: embed_buf for NEW tokens only
    float* embed_buf = (float*)atlas_valloc((size_t)n_new * H * sizeof(float));
    float* h_norm = (float*)atlas_valloc((size_t)H * sizeof(float));
    float* logits = (float*)atlas_valloc((size_t)V * sizeof(float));
    int* context = (int*)atlas_valloc((size_t)(n_input + max_new_tokens) * sizeof(int));
    if (!embed_buf || !h_norm || !logits || !context) {
        if (embed_buf) atlas_vfree((uint8_t*)embed_buf);
        if (h_norm) atlas_vfree((uint8_t*)h_norm);
        if (logits) atlas_vfree((uint8_t*)logits);
        if (context) atlas_vfree((uint8_t*)context);
        return -1;
    }
    memcpy(context, input_ids, (size_t)n_input * sizeof(int));

    // ─── Prefill: embed NEW tokens only (cache_offset..n_input-1) ───
    int n_gen = 0;
    for (int i = 0; i < n_new; i++) {
        int idx = cache_offset + i;
        int tid = input_ids[idx];
        if (tid < 0 || tid >= V) tid = 0;
        for (int j = 0; j < H; j++)
            embed_buf[i * H + j] = fp16_to_fp32(embed_w[tid * H + j]) * m->scale_emb;
    }

    // Build position array starting from cache_offset
    int* positions = (int*)atlas_valloc((size_t)n_new * sizeof(int));
    if (!positions) {
        atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
        atlas_vfree((uint8_t*)logits); atlas_vfree((uint8_t*)context); return -1;
    }
    for (int i = 0; i < n_new; i++) positions[i] = cache_offset + i;

    // Fused forward for new tokens only; seq_now = total cached tokens
    ATLAS_LOG("prefill forward: B=%d n_new=%d\n", n_new, n_new);
    if (atlas_forward(m, embed_buf, n_new, positions,
                      max_seq_len, n_input,
                      layer_idx, m->n_layers) != 0) {
        atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
        atlas_vfree((uint8_t*)logits); atlas_vfree((uint8_t*)positions);
        atlas_vfree((uint8_t*)context);
        return -1;
    }
    atlas_vfree((uint8_t*)positions);
    ATLAS_LOG("prefill forward done\n");

    // Final RMSNorm + LM head — only the last new token's logits are needed
    {
        const float* x = embed_buf + (int64_t)(n_new - 1) * H;
        P_START(rmsnorm);
        atlas_rmsnorm_f32(x, norm_w, h_norm, H, 1e-6f);
        P_ACCUM(rmsnorm);
        P_START(lmhead);
        atlas_lmhead_gemv(m, h_norm, logits, 1);
        P_ACCUM(lmhead);
    }

    // Sample first token from prefill logits
    const int eos_id = m->eos_id;
    const int eos_id2 = m->eos_id2;
    if (n_gen < min_new_tokens) {
        if (eos_id >= 0 && eos_id < V) logits[eos_id] = -1e9f;
        if (eos_id2 >= 0 && eos_id2 < V) logits[eos_id2] = -1e9f;
    }
    if (logit_cb) logit_cb(logits, V, logit_cb_data);
    int next_token = gumbel_sample(logits, V, temperature, top_k, top_p,
                                    context, n_input, repetition_penalty);
    if (token_notify_cb) token_notify_cb(next_token, token_notify_data);
    context[n_input] = next_token;
    output_ids[n_gen++] = next_token;

    if (next_token == eos_id || next_token == eos_id2) {
        atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
        atlas_vfree((uint8_t*)logits); atlas_vfree((uint8_t*)context);
        atlas_vfree((uint8_t*)positions);
        return n_gen;
    }

    // ─── Decode loop (v2.5.0: ring buffer — no veto, wraps at max_seq_len) ───
    for (int step = 1; step < max_new_tokens; step++) {
        // Embed last generated token
        int tid = next_token;
        if (tid < 0 || tid >= V) tid = 0;
        float* h = embed_buf;  // reuse embed_buf as single-token buffer
        for (int j = 0; j < H; j++)
            h[j] = fp16_to_fp32(embed_w[tid * H + j]) * m->scale_emb;

        int seq_now = n_input + step;
        int pos = seq_now - 1;
        ATLAS_LOG("decode step=%d seq_now=%d pos=%d\n", step, seq_now, pos);
        if (atlas_forward(m, h, 1, &pos,
                          max_seq_len, seq_now,
                          layer_idx, m->n_layers) != 0) {
            atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
            atlas_vfree((uint8_t*)logits); atlas_vfree((uint8_t*)context);
            return -1;
        }
        ATLAS_LOG("decode forward done\n");

        P_START(rmsnorm);
        atlas_rmsnorm_f32(h, norm_w, h_norm, H, 1e-6f);
        P_ACCUM(rmsnorm);
        P_START(lmhead);
        atlas_lmhead_gemv(m, h_norm, logits, 1);
        P_ACCUM(lmhead);
        PROF_ADD_TOKENS(1);

        if (n_gen < min_new_tokens) {
            if (eos_id >= 0 && eos_id < V) logits[eos_id] = -1e9f;
            if (eos_id2 >= 0 && eos_id2 < V) logits[eos_id2] = -1e9f;
        }
        if (logit_cb) logit_cb(logits, V, logit_cb_data);
        next_token = gumbel_sample(logits, V, temperature, top_k, top_p,
                                    context, n_input + n_gen, repetition_penalty);
        if (token_notify_cb) token_notify_cb(next_token, token_notify_data);
        context[n_input + n_gen] = next_token;
        output_ids[n_gen++] = next_token;

        if (next_token == eos_id || next_token == eos_id2) break;  // EOS
    }

    atlas_vfree((uint8_t*)embed_buf);
    atlas_vfree((uint8_t*)h_norm);
    atlas_vfree((uint8_t*)logits);
    atlas_vfree((uint8_t*)context);
    PROF_ADD_TOKENS(1);  // account for prefill token

    profile_print();
    profile_print_arm64();
    profile_print_tq1();
    return n_gen;
}

// ─── v2.3.0: Streaming generation (callback per token) ───────────────────
ATLAS_API int atlas_generate_stream(AtlasModel* m,
    const int* input_ids, int n_input,
    int max_seq_len, int max_new_tokens,
    float temperature, int top_k, float top_p,
    float repetition_penalty,
    int min_new_tokens,
    int cache_offset,
    atlas_token_callback callback, void* user_data,
    int prefill_only,
    atlas_logit_processor_cb logit_cb, void* logit_cb_data,
    atlas_token_notify_cb token_notify_cb, void* token_notify_data)
{
    profile_reset();
    if (!m || !input_ids || n_input < 1)
        return -1;
    if (!prefill_only && !callback) return -1;
    if (n_input >= max_seq_len) {
        fprintf(stderr, "[ATLAS] atlas_generate_stream: prompt (%d) >= max_seq_len (%d)\n",
                n_input, max_seq_len);
        return -1;
    }

    int idx_norm = -1, idx_embed = -1;
    for (int i = 0; i < (int)m->tensor_names.size(); i++) {
        if (m->tensor_names[i] == "model.norm.weight") idx_norm = i;
        if (m->tensor_names[i] == "model.embed_tokens.weight") idx_embed = i;
    }
    if (idx_norm < 0 || idx_embed < 0) return -1;

    int H = m->hidden_dim;
    int V = m->vocab_size;
    uint16_t* embed_w = (uint16_t*)m->tensors[idx_embed].data;

    if (!m->lm_head_quantized) return -1;

    ensure_layer_idx(m);
    const int* layer_idx = m->layer_idx_cache.data();

    if (cache_offset < 0) cache_offset = 0;
    if (cache_offset >= n_input) cache_offset = n_input - 1;
    int n_new = n_input - cache_offset;

    // ─── Embed new tokens ───
    float* embed_buf = (float*)atlas_valloc((size_t)n_new * H * sizeof(float));
    if (!embed_buf) return -1;
    for (int i = 0; i < n_new; i++) {
        int idx = cache_offset + i;
        int tid = input_ids[idx];
        if (tid < 0 || tid >= V) tid = 0;
        for (int j = 0; j < H; j++)
            embed_buf[i * H + j] = fp16_to_fp32(embed_w[tid * H + j]) * m->scale_emb;
    }

    int* positions = (int*)atlas_valloc((size_t)n_new * sizeof(int));
    if (!positions) { atlas_vfree((uint8_t*)embed_buf); return -1; }
    for (int i = 0; i < n_new; i++) positions[i] = cache_offset + i;

    ATLAS_LOG("prefill%s forward: B=%d cache_offset=%d\n",
              prefill_only ? "_only" : "", n_new, cache_offset);
    if (atlas_forward(m, embed_buf, n_new, positions,
                      max_seq_len, n_input,
                      layer_idx, m->n_layers) != 0) {
        atlas_vfree((uint8_t*)embed_buf);
        atlas_vfree((uint8_t*)positions);
        return -1;
    }
    atlas_vfree((uint8_t*)positions);

    // ─── Prefill-only mode: skip sampling, return 0 ───
    if (prefill_only) {
        atlas_vfree((uint8_t*)embed_buf);
        return 0;
    }

    // ─── Full generation mode: RMSNorm + lm_head + sample ───
    float* h_norm = (float*)atlas_valloc((size_t)H * sizeof(float));
    float* logits = (float*)atlas_valloc((size_t)V * sizeof(float));
    int* context = (int*)atlas_valloc((size_t)(n_input + max_new_tokens) * sizeof(int));
    if (!h_norm || !logits || !context) {
        if (h_norm) atlas_vfree((uint8_t*)h_norm);
        if (logits) atlas_vfree((uint8_t*)logits);
        if (context) atlas_vfree((uint8_t*)context);
        atlas_vfree((uint8_t*)embed_buf);
        return -1;
    }
    memcpy(context, input_ids, (size_t)n_input * sizeof(int));
    uint8_t* norm_w = m->tensors[idx_norm].data;

    int n_gen = 0;

    {
        const float* x = embed_buf + (int64_t)(n_new - 1) * H;
        P_START(rmsnorm);
        atlas_rmsnorm_f32(x, norm_w, h_norm, H, 1e-6f);
        P_ACCUM(rmsnorm);
        P_START(lmhead);
        atlas_lmhead_gemv(m, h_norm, logits, 1);
        P_ACCUM(lmhead);
    }

    const int eos_id = m->eos_id;
    const int eos_id2 = m->eos_id2;
    if (n_gen < min_new_tokens) {
        if (eos_id >= 0 && eos_id < V) logits[eos_id] = -1e9f;
        if (eos_id2 >= 0 && eos_id2 < V) logits[eos_id2] = -1e9f;
    }
    if (logit_cb) logit_cb(logits, V, logit_cb_data);
    int next_token = gumbel_sample(logits, V, temperature, top_k, top_p,
                                    context, n_input, repetition_penalty);
    if (token_notify_cb) token_notify_cb(next_token, token_notify_data);
    context[n_input] = next_token;
    callback(next_token, user_data);
    n_gen++;

    if (next_token == eos_id || next_token == eos_id2) {
        atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
        atlas_vfree((uint8_t*)logits);
        atlas_vfree((uint8_t*)context);
        return n_gen;
    }

    // ─── Decode loop (v2.5.0: ring buffer — no veto, wraps at max_seq_len) ───
    for (int step = 1; step < max_new_tokens; step++) {
        int tid = next_token;
        if (tid < 0 || tid >= V) tid = 0;
        float* h = embed_buf;
        for (int j = 0; j < H; j++)
            h[j] = fp16_to_fp32(embed_w[tid * H + j]) * m->scale_emb;

        int seq_now = n_input + step;
        int pos = seq_now - 1;
        if (atlas_forward(m, h, 1, &pos,
                          max_seq_len, seq_now,
                          layer_idx, m->n_layers) != 0) {
            atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
            atlas_vfree((uint8_t*)logits); atlas_vfree((uint8_t*)context);
            return n_gen > 0 ? n_gen : -1;
        }

        P_START(rmsnorm);
        atlas_rmsnorm_f32(h, norm_w, h_norm, H, 1e-6f);
        P_ACCUM(rmsnorm);
        P_START(lmhead);
        atlas_lmhead_gemv(m, h_norm, logits, 1);
        P_ACCUM(lmhead);
        PROF_ADD_TOKENS(1);

        if (n_gen < min_new_tokens) {
            if (eos_id >= 0 && eos_id < V) logits[eos_id] = -1e9f;
            if (eos_id2 >= 0 && eos_id2 < V) logits[eos_id2] = -1e9f;
        }
        if (logit_cb) logit_cb(logits, V, logit_cb_data);
        next_token = gumbel_sample(logits, V, temperature, top_k, top_p,
                                    context, n_input + n_gen, repetition_penalty);
        if (token_notify_cb) token_notify_cb(next_token, token_notify_data);
        context[n_input + n_gen] = next_token;
        callback(next_token, user_data);
        n_gen++;

        if (next_token == eos_id || next_token == eos_id2) break;
    }

    atlas_vfree((uint8_t*)embed_buf);
    atlas_vfree((uint8_t*)h_norm);
    atlas_vfree((uint8_t*)logits);
    atlas_vfree((uint8_t*)context);
    PROF_ADD_TOKENS(1);

    profile_print();
    profile_print_arm64();
    profile_print_tq1();
    return n_gen;
}

}  // extern "C"
