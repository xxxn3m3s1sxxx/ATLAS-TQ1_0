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
#include <immintrin.h>
#include <omp.h>

// Debug logging: compile with -DATLAS_DEBUG_MODE for runtime probes
#ifdef ATLAS_DEBUG_MODE
  #define ATLAS_LOG(fmt, ...) do { printf("[ATLAS_DBG] " fmt, ##__VA_ARGS__); fflush(stdout); } while (0)
#else
  #define ATLAS_LOG(fmt, ...) do {} while (0)
#endif

// VNNI kernel lives in atlas_vnni.cpp (compiled with target("avx10.2"))
extern "C" int atlas_matmul_block_vnni(const int8_t* act, const int8_t* row, int blk_end, int blk_start);
extern "C" int atlas_vnni_available(void);

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
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
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
#ifdef _WIN32
#include <intrin.h>
static int check_avx2(void) {
    int info[4] = {0};
    __cpuidex(info, 7, 0);
    return (info[1] >> 5) & 1; // EBX bit 5 = AVX2
}
#else
#include <cpuid.h>
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

// ─── Xoshiro256** PRNG (thread-safe, 64-bit) ──────────────────────────
static uint64_t xoshiro_state[4] = {0};

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

// Timer for debug profiling
#ifdef ATLAS_DEBUG_MODE
static inline double atlas_now() {
    LARGE_INTEGER c, f;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#endif

static inline float fp16_to_fp32(uint16_t h) {
    float r;
    __m128i h4 = _mm_cvtsi32_si128((int)(unsigned)h);  // zero-extend to 32-bit
    __m128 f4 = _mm_cvtph_ps(h4);                       // F16C: fp16→fp32
    _mm_store_ss(&r, f4);
    return r;
}
static inline uint16_t fp32_to_fp16(float v) {
    __m128 f4 = _mm_set_ss(v);
    return (uint16_t)(unsigned)_mm_extract_epi16(_mm_cvtps_ph(f4, 0), 0);
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
    int pad_id = 0;    // default padding token
    bool use_f32_matmul = false; // skip activation quantization (1B model needs full precision)
    bool use_ternary_matmul = false; // v1.3.0: vpsignb-based ternary-add kernel (no multiplication)
    bool use_packed_matmul = false; // v1.3.1: operate on 2-bit packed ternary weights (4× less memory)
    bool use_hybrid_matmul = false; // v1.3.2: FFN int8 cache, QKV packed
    bool use_relu2 = false;         // v2.8.0: ReLU² activation (BitNet b1.58)
    bool rope_interleaved = true;   // true=interleaved (Falcon3/Qwen3), false=half-split (Llama/BitNet)
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
    uint8_t* buf_i8 = nullptr;      // [max_batch * max_dim] uint8 quantized activations
    float* buf_out = nullptr;       // [max_batch * hidden_dim] layer output ping-pong for atlas_forward
    // Attention workspace (heap-allocated, avoids stack overflow with large B)
    float* attn_ws = nullptr;       // [max_batch * n_heads * head_dim * 4]
    int max_batch = 0;
    // mmap cache handles (for int8 data loaded from .i8 file)
    void* mmap_base = nullptr;      // MapViewOfFile base (.i8 cache)
    void* mmap_handle = nullptr;    // CreateFileMapping handle (Win) / file size (Lin)
    void* mmap_file = nullptr;      // CreateFile handle (Win) / fd (Lin)

    // v2.9.2: FP16 KV cache (full precision, no quantization noise)
    uint16_t* k_cache = nullptr;
    uint16_t* v_cache = nullptr;
    int cache_max_seq_len = 0;

    // Ensure KV cache is allocated for at least max_seq_len
    void ensure_cache(int max_seq_len) {
        if (max_seq_len <= cache_max_seq_len) return;
        size_t cache_sz = (size_t)n_kv_heads * max_seq_len * head_dim * n_layers;
        if (k_cache) atlas_vfree((uint8_t*)k_cache);
        if (v_cache) atlas_vfree((uint8_t*)v_cache);
        k_cache = (uint16_t*)atlas_valloc(cache_sz * sizeof(uint16_t));
        v_cache = (uint16_t*)atlas_valloc(cache_sz * sizeof(uint16_t));
        cache_max_seq_len = max_seq_len;
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
    } tok = {};
    // Int8 quantized lm_head (per-row symmetric, ~403 MB instead of 1.5 GB fp32)
    int8_t* lm_head_i8 = nullptr;
    int32_t* lm_head_offsets = nullptr;  // precomputed 128 * sum(w) per row
    float* lm_head_scales = nullptr;
    bool lm_head_quantized = false;
    ~AtlasModel() {
        if (buf_gate) atlas_vfree((uint8_t*)buf_gate);
        if (buf_up) atlas_vfree((uint8_t*)buf_up);
        if (buf_hidden) atlas_vfree((uint8_t*)buf_hidden);
        if (buf_act) atlas_vfree((uint8_t*)buf_act);
        if (buf_i8) atlas_vfree((uint8_t*)buf_i8);
        if (buf_out) atlas_vfree((uint8_t*)buf_out);
        if (attn_ws) atlas_vfree((uint8_t*)attn_ws);
        if (lm_head_i8) atlas_vfree((uint8_t*)lm_head_i8);
        if (lm_head_offsets) atlas_vfree((uint8_t*)lm_head_offsets);
        if (lm_head_scales) atlas_vfree((uint8_t*)lm_head_scales);
        delete[] tok.merge_lookup;
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

    void ensure_buffers(int B) {
        if (B <= max_batch) return;
        if (buf_gate) atlas_vfree((uint8_t*)buf_gate);
        if (buf_up) atlas_vfree((uint8_t*)buf_up);
        if (buf_hidden) atlas_vfree((uint8_t*)buf_hidden);
        if (buf_act) atlas_vfree((uint8_t*)buf_act);
        if (buf_i8) atlas_vfree((uint8_t*)buf_i8);
        if (buf_out) atlas_vfree((uint8_t*)buf_out);
        if (attn_ws) atlas_vfree((uint8_t*)attn_ws);

        int max_dim = inter_dim > hidden_dim ? inter_dim : hidden_dim;
        if (n_heads * head_dim > max_dim) max_dim = n_heads * head_dim;
        int max_aligned = ((max_dim + 7) + 31) & ~31;  // +7 for TQ1 padding (packed_cols*5 up to dim+4)

        buf_gate = (float*)atlas_valloc((size_t)B * max_dim * sizeof(float));
        buf_up = (float*)atlas_valloc((size_t)B * max_dim * sizeof(float));
        buf_hidden = (float*)atlas_valloc((size_t)B * max_dim * sizeof(float));
        buf_act = (float*)atlas_valloc((size_t)B * max_aligned * sizeof(float));
        buf_i8 = (uint8_t*)atlas_valloc((size_t)B * max_aligned * sizeof(uint8_t));
        buf_out = (float*)atlas_valloc((size_t)B * hidden_dim * sizeof(float));
        int ws = (int)((size_t)B * n_heads * head_dim * 4);
        attn_ws = (float*)atlas_valloc((size_t)ws * sizeof(float));
        max_batch = B;
    }
};

// ─── TQ1 byte → int8 decode LUT (v1.3.1 chunked decode) ──────────────
// Decode LUT: 5 int8 trits pro Byte (1280 bytes, L1-resident)
static std::once_flag tq1_decode_init_flag;
alignas(32) static int8_t tq1_decode[256][5];

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
    }

    v = find_after(json, "\"rope_interleaved\"");
    if (v) m->rope_interleaved = (*v == 't' || *v == '1');

    v = find_after(json, "\"use_f32_bypass\"");
    if (v) m->use_f32_matmul = (*v == 't' || *v == '1');

    v = find_after(json, "\"rope_theta\"");
    if (v) m->rope_theta = (float)strtod(v, nullptr);

    v = find_after(json, "\"rope_scale\"");
    if (v) m->rope_scale = (float)strtod(v, nullptr);

    v = find_after(json, "\"base_seq_len\"");
    if (v) m->base_seq_len = (int)strtol(v, nullptr, 10);

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
}

// ─── Load model ─────────────────────────────────────────────────────────
ATLAS_API AtlasModel* atlas_load(const char* path) {
    if (!check_avx2()) {
        fprintf(stderr, "[ATLAS] Error: AVX2 instruction set required.\n");
        fprintf(stderr, "  ATLAS needs AVX2 (Haswell, ~2013+) for fast int8 matmul.\n");
        fprintf(stderr, "  Your CPU does not report AVX2 support.\n");
        return nullptr;
    }
    init_tq1_decode_lut();
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ATLAS] Cannot open %s\n", path); return nullptr; }

    uint8_t hdr[64];
    if (fread(hdr, 1, 64, f) != 64) { fclose(f); return nullptr; }
    if (memcmp(hdr, "ATLAS", 5) != 0) { fclose(f); return nullptr; }

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
    if (eos_val != 0) m->eos_id = (int)eos_val;
    if (pad_val != 0) m->pad_id = (int)pad_val;

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

    // Read directory
    m->tensors.resize(n_tensors);
    std::vector<uint32_t> file_offsets(n_tensors);
    FSEEK(f, 64 + meta_size, SEEK_SET);
    for (int i = 0; i < n_tensors; i++) {
        uint8_t e[12]; fread(e, 1, 12, f);
        m->tensors[i].ttype = e[0];
        memcpy(&file_offsets[i], e+1, 4); m->tensors[i].file_offset = file_offsets[i];
        memcpy(&m->tensors[i].row_dim, e+5, 4);
        m->tensors[i].packed_cols = e[9] | (e[10]<<8) | (e[11]<<16);
    }

    // Load tensor names (v4+)
    m->tensor_names.clear();
    if (version >= 4) {
        int nb_size; memcpy(&nb_size, hdr+56, 4);
        if (nb_size > 0) {
            // Names stored right after directory: [name_block_size:4] [name_0\0]... 
            FSEEK(f, 64 + meta_size + n_tensors * 12, SEEK_SET);
            uint8_t* nb = new uint8_t[nb_size];
            fread(nb, 1, nb_size, f);
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
        } else if (t.ttype != 5 && t.ttype != 7) {  // lm_head / scales (not block-scaled TQ1)
            int actual_bytes = (i + 1 < n_tensors)
                ? (int)(file_offsets[i + 1] - file_offsets[i]) - ((int)(file_offsets[i + 1] - file_offsets[i]) % 32)
                : (int)(file_size - (int64_t)file_offsets[i]);
            int expected = t.row_dim * m->hidden_dim * 2;
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
    auto& last = m->tensors.back();
    int64_t total = last.file_offset + last.data_size;
    printf("[ATLAS] Loaded %d tensors (%.2f MB)%s\n", n_tensors, total / 1048576.0,
           m->tok.magic == 0x544F4B42 ? " (v6 binary tokenizer)" : "");
    return m;
}

// ─── Free model ─────────────────────────────────────────────────────────
ATLAS_API void atlas_free(AtlasModel* m) {
    if (!m) return;
    // Free valloc'd tensors (not mmap'd ones)
    for (auto& t : m->tensors) {
        if (t.data && !m->is_mapped(t.data)) atlas_vfree(t.data);
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

// Pre-encode UTF-8 text → byte-level token IDs (raw byte_encoder lookup, no BPE).
// Returns number of tokens, or -1 on error.
ATLAS_API int atlas_tokenizer_preencode(AtlasModel* m,
    const char* text, int text_len,
    int* out_ids, int max_ids) {
    if (!m || !text || !out_ids || m->tok.magic != 0x544F4B42)
        return -1;
    if (text_len <= 0 || max_ids <= 0) return 0;

    const uint16_t* byte_enc = m->tok.byte_encoder;
    int n = 0;
    for (int i = 0; i < text_len && n < max_ids; i++) {
        uint8_t b = (uint8_t)text[i];
        uint16_t tid = byte_enc[b];
        if (tid == 0xFFFF) {
            uint32_t unk = m->tok.special ? m->tok.special[3] : 0;
            out_ids[n++] = (int)unk;
        } else {
            out_ids[n++] = (int)tid;
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
        if (tid < 0 || tid >= (int)m->tok.vocab_size) continue;
        uint32_t off = offsets[tid];
        uint16_t len = lengths[tid];
        if (off + len > m->tok.pool_size) continue;
        if (pos + (int)len > max_out - 1) {
            // Truncate, but don't overflow
            int copy = max_out - 1 - pos;
            if (copy > 0) {
                memcpy(out_text + pos, pool + off, copy);
                pos += copy;
            }
            break;
        }
        memcpy(out_text + pos, pool + off, len);
        pos += len;
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

    // Store mmap handles + size for cleanup in atlas_free
    m->mmap_base = base;
    m->mmap_handle = hMap;
    m->mmap_file = hFile;
    m->mmap_size = file_size;

    printf("[CACHE] Loaded %d/%d tensors\n", replaced, n);
    return replaced > 0 ? 1 : 0;
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

        // ─── ttype=0: raw ternary {-1,0,1} decompression ───
        uint8_t* new_data = atlas_valloc(2 + n_vals + t.row_dim * 4);
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
        float* f32_row = (float*)alloca(input_dim * sizeof(float));
        for (int r = 0; r < t.row_dim; r++) {
            for (int c = 0; c < t.packed_cols; c++) {
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

        uint8_t* new_data = atlas_valloc(2 + n_vals + t.row_dim * 4);
        uint16_t scale_u16 = fp32_to_fp16(stored_scale);
        memcpy(new_data, &scale_u16, 2);
        int8_t* i8 = (int8_t*)(new_data + 2);
        int32_t* rs = (int32_t*)(i8 + n_vals);

        for (int r = 0; r < t.row_dim; r++) {
            int pos = 0;
            int sum = 0;
            for (int c = 0; c < t.packed_cols; c++) {
                const int8_t* l = tq1_decode[packed[r * t.packed_cols + c]];
                for (int t2 = 0; t2 < 5; t2++) {
                    int col = c * 5 + t2;
                    if (col >= input_dim) break;
                    int blk = col / bs;
                    float scale2 = (blk < nbk) ? decoded_scales[r * nbk + blk] : 0.0f;
                    float val = (float)l[t2] * scale2;
                    int q = (int)(val / quant_scale + 0.5f);
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
        if (t.data && !m->is_mapped(t.data)) atlas_vfree(t.data);
        t.data = new_data;
        t.data_size = 2 + n_vals + t.row_dim * 4;
        t.packed_cols = input_dim;
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
        int block_size = t.block_size;
        int n_blocks = t.n_blocks;

        const uint8_t* raw_scales = t.data + 3;
        const uint8_t* packed = t.data + 3 + t.row_dim * n_blocks * 2;

        // Decode all rows to f32 to find global max
        float* f32_all = (float*)atlas_valloc(n_vals * sizeof(float));
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

        uint8_t* new_data = atlas_valloc(2 + n_vals + t.row_dim * 4);
        uint16_t scale_u16 = fp32_to_fp16(stored_scale);
        memcpy(new_data, &scale_u16, 2);
        int8_t* i8 = (int8_t*)(new_data + 2);
        int32_t* rs = (int32_t*)(i8 + n_vals);

        for (int64_t i = 0; i < n_vals; i++) {
            float vq = f32_all[i] / quant_scale;
            int q = (vq >= 0) ? (int)(vq + 0.5f) : (int)(vq - 0.5f);
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
    if (m->use_f32_matmul) return;  // f32_bypass needs full precision, no activation quant
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
        int input_dim; memcpy(&input_dim, t.data + 2, 4); // HACK: need actual dim
        // Actually, for ttype=3, data_size = 2 + rows*cols + rows*4
        // So cols = (data_size - 2 - rows*4) / rows
        
        // Hmm, I need the cols dimension. Let me derive it from data_size.
        // data_size = 2 + rows * cols + rows * 4
        // cols = (data_size - 2 - rows * 4) / rows
        int64_t ds = t.data_size;
        int cols = (int)((ds - 2 - (int64_t)rows * 4) / rows);
        if (cols <= 0 || cols > 1000000) continue;  // sanity check
        
        int old_size = (int)ds;
        int packed_cols = (cols + 1) / 2;  // round up
        int new_size = 2 + rows * packed_cols + rows * 4;
        
        uint8_t* new_data = atlas_valloc(new_size);
        // Copy fp16 scale
        memcpy(new_data, t.data, 2);
        
        uint8_t* packed = new_data + 2;
        int32_t* new_rs = (int32_t*)(packed + rows * packed_cols);
        
        // OMP parallel over rows (each row is independent)
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int r = 0; r < rows; r++) {
            int sum = 0;
            for (int c = 0; c < cols; c += 2) {
                int v0 = (int)i8[r * cols + c];
                int v1 = (c + 1 < cols) ? (int)i8[r * cols + c + 1] : 0;
                
                // Clip to int4 range [-8, 7]
                if (v0 < -8) v0 = -8;
                if (v0 > 7) v0 = 7;
                if (v1 < -8) v1 = -8;
                if (v1 > 7) v1 = 7;
                
                // Convert to 4-bit unsigned
                int u0 = v0 & 0x0F;  // -8→8, -1→15, 0→0, 7→7
                int u1 = v1 & 0x0F;
                
                // Pack: low nibble = v0, high nibble = v1
                packed[r * packed_cols + c / 2] = (uint8_t)(u0 | (u1 << 4));
                
                // Recalculate row_sum from clipped values
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
// v2.4.0: Also handles ttype=5 (per-row block-scaled TQ1) — dequantizes
// per-block scales, then re-quantizes to uniform int8 with a single global scale.
ATLAS_API void atlas_decompress_ffn(AtlasModel* m) {
    int total = 0;
    init_tq1_decode_lut();
    for (size_t i = 0; i < m->tensors.size(); i++) {
        auto& t = m->tensors[i];
        if (t.ttype != 0) continue;
        // Check name: only decompress MLP tensors
        // ttype=5 (block-scaled) handled directly by fused kernel — skip
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
            volatile uintptr_t d8 = (uintptr_t)t.data;
            if (!d8) continue;
            int n_vals = (int)(t.data_size - 2 - t.row_dim * 4);
            if (n_vals <= 0) continue;
            int8_t* data = (int8_t*)(d8 + 2);
            for (int64_t i = 0; i < n_vals; i += step) {
                volatile int sink = data[i]; (void)sink;
            }
            total += n_vals;
        } else if (t.ttype == 10) {
            // Prefetch TQ2 packed data + fp16 scales
            volatile const uint8_t* d = t.data;
            if (!d) continue;
            int64_t sz = t.data_size;
            for (int64_t i = 0; i < sz; i += step) {
                volatile int sink = d[i]; (void)sink;
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
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int r = 0; r < rows; r++) {
        const int8_t* w = weights + r * input_dim;
        int sum_w = row_sums[r];

        // Cross-row prefetch: hide DRAM latency for next 2 weight rows
        if (r + 2 < rows) {
            _mm_prefetch((const char*)(weights + (r + 2) * input_dim), _MM_HINT_T1);
        }

        for (int t = 0; t < n_tokens; t++) {
            const uint8_t* a = act_u8 + t * input_dim;

            int c = 0;
            int dot = 0;

            // AVX2: 32 bytes per iteration → 16 pairs via maddubs → 8 int32 via madd
            __m256i acc = _mm256_setzero_si256();

            for (; c + 32 <= input_dim; c += 32) {
                __m256i au = _mm256_loadu_si256((const __m256i*)(a + c));
                __m256i wi = _mm256_loadu_si256((const __m256i*)(w + c));
                __m256i prod16 = _mm256_maddubs_epi16(au, wi);
                __m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));
                acc = _mm256_add_epi32(acc, prod32);
            }

            // Horizontal sum of acc
            __m128i lo = _mm256_castsi256_si128(acc);
            __m128i hi = _mm256_extracti128_si256(acc, 1);
            __m128i sum128 = _mm_add_epi32(lo, hi);
            sum128 = _mm_hadd_epi32(sum128, sum128);
            sum128 = _mm_hadd_epi32(sum128, sum128);
            dot = _mm_cvtsi128_si32(sum128);

            // Tail (less than 32 elements)
            for (; c < input_dim; c++) {
                dot += (int)a[c] * (int)w[c];
            }

            // Undo 128 offset: dot' = dot - 128 * Σ w_i
            int result = dot - 128 * sum_w;
            output[t * rows + r] = (float)result;
        }
    }

}

// ─── int4×uint8 matmul: nibble unpack + vpmaddubs + sign-extend ──────
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
    // Sign-extension for 4-bit nibble: freq = (nibble ^ 8) - 8
    // Maps 0..15 → -8..+7 with signed overflow for val >= 8
    // __m256i version: _mm256_sub_epi8(_mm256_xor_si256(v, c8), c8)
    const __m256i c8 = _mm256_set1_epi8(8);
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int r = 0; r < rows; r++) {
        const uint8_t* pw = packed_weights + r * cols / 2;
        int sum_w = row_sums[r];

        // Cross-row prefetch: next 2 weight rows in L2
        if (r + 2 < rows) {
            _mm_prefetch((const char*)(packed_weights + (r + 2) * cols / 2), _MM_HINT_T1);
        }

        for (int t = 0; t < n_tokens; t++) {
            const uint8_t* a = act_u8 + t * cols;

            int c = 0;
            int dot = 0;
            __m256i acc = _mm256_setzero_si256();

            // Process 64 elements per iteration: 32 packed bytes → 2×32 int8 weights
            for (; c + 64 <= cols; c += 64) {
                int pc = c / 2;
                __m256i packed = _mm256_loadu_si256((const __m256i*)(pw + pc));

                // Low nibbles: mask lower 4 bits of each byte
                __m256i low = _mm256_and_si256(packed, mask_0f);
                // High nibbles: shift right 4 bits per 16-bit word, then mask
                __m256i high = _mm256_and_si256(
                    _mm256_srli_epi16(packed, 4), mask_0f);

                // Sign-extend 4-bit to int8: (nibble ^ 8) - 8
                __m256i w_low_s = _mm256_sub_epi8(
                    _mm256_xor_si256(low, c8), c8);
                __m256i w_high_s = _mm256_sub_epi8(
                    _mm256_xor_si256(high, c8), c8);

                // Interleave low/high back to contiguous: [v0,v1,v2,...,v63]
                // _mm256_unpack* operates per 128-bit lane, so lanes need fixing:
                // unpacklo → [v0..v15](lane0) + [v32..v47](lane1)
                // unpackhi → [v16..v31](lane0) + [v48..v63](lane1)
                __m256i w_tmp_lo = _mm256_unpacklo_epi8(w_low_s, w_high_s);
                __m256i w_tmp_hi = _mm256_unpackhi_epi8(w_low_s, w_high_s);
                // permute2f128: 0x20 = lane0(v0..v15) + lane0(v16..v31) = [v0..v31]
                //              0x31 = lane1(v32..v47) + lane1(v48..v63) = [v32..v63]
                __m256i w_lo = _mm256_permute2f128_si256(w_tmp_lo, w_tmp_hi, 0x20);
                __m256i w_hi = _mm256_permute2f128_si256(w_tmp_lo, w_tmp_hi, 0x31);

                // activations[c..c+31] × weights[0..31]
                __m256i a0 = _mm256_loadu_si256((const __m256i*)(a + c));
                __m256i p16_0 = _mm256_maddubs_epi16(a0, w_lo);
                __m256i p32_0 = _mm256_madd_epi16(p16_0, ones16);
                acc = _mm256_add_epi32(acc, p32_0);

                // activations[c+32..c+63] × weights[32..63]
                __m256i a1 = _mm256_loadu_si256((const __m256i*)(a + c + 32));
                __m256i p16_1 = _mm256_maddubs_epi16(a1, w_hi);
                __m256i p32_1 = _mm256_madd_epi16(p16_1, ones16);
                acc = _mm256_add_epi32(acc, p32_1);
            }

            // Horizontal sum of acc
            __m128i lo = _mm256_castsi256_si128(acc);
            __m128i hi = _mm256_extracti128_si256(acc, 1);
            __m128i sum128 = _mm_add_epi32(lo, hi);
            sum128 = _mm_hadd_epi32(sum128, sum128);
            sum128 = _mm_hadd_epi32(sum128, sum128);
            dot = _mm_cvtsi128_si32(sum128);

            // Tail (< 64 elements): nibble unpack + sign-extend scalar
            for (; c < cols; c++) {
                int packed_idx = (r * cols + c) / 2;
                int nibble = (c & 1)
                    ? (packed_weights[packed_idx] >> 4)
                    : (packed_weights[packed_idx] & 0x0F);
                int8_t w_val = (int8_t)((nibble ^ 8) - 8);
                dot += (int)a[c] * (int)w_val;
            }

            int result = dot - 128 * sum_w;
            output[t * rows + r] = (float)result;
        }
    }
}

// ─── Norm: float16 tensor → RMSNorm ────────────────────────────────────
// Performs: output[i] = x[i] * weight[i] * rms(mean(x^2) + eps)
// Where weight is loaded from atlas tensor (float16)
ATLAS_API void atlas_rmsnorm_f32(const float* x, const uint8_t* weight_f16,
                                  float* output, int n, float eps) {
    float ss = 0.0f;
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 x2 = _mm256_mul_ps(xv, xv);
        ss += x2[0] + x2[1] + x2[2] + x2[3] + x2[4] + x2[5] + x2[6] + x2[7];
    }
    for (; i < n; i++) ss += x[i] * x[i];
    float rms = 1.0f / sqrtf(ss / n + eps);

    __m256 rv = _mm256_set1_ps(rms);
    for (i = 0; i + 8 <= n; i += 8) {
        __m128i w8 = _mm_loadu_si128((const __m128i*)(weight_f16 + i * 2));
        __m256 w32 = _mm256_cvtph_ps(w8);
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 r = _mm256_mul_ps(_mm256_mul_ps(xv, rv), w32);
        _mm256_storeu_ps(output + i, r);
    }
    for (; i < n; i++) {
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
// k_cache: [n_kv_heads, max_seq, head_dim] uint16_t — fp16 K cache
// v_cache: [n_kv_heads, max_seq, head_dim] uint16_t — fp16 V cache
// output: [B, n_heads * head_dim] float32
// q_norm_w, k_norm_w: [head_dim] uint8 fp16 RMSNorm weights (QK-Norm, Qwen3), NULL = skip
ATLAS_API void atlas_attention_f32(
    float* q, float* k, float* v, const int* positions,
    uint16_t* k_cache, uint16_t* v_cache,
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

        // Store K, V into cache (fp32 -> fp16, full precision)
        for (int h = 0; h < n_kv_heads; h++) {
            float* k_row = kb + h * head_dim;
            float* v_row = vb + h * head_dim;
            int cache_pos = pos % max_seq_len;
            uint16_t* kc = k_cache + (size_t)h * max_seq_len * head_dim + (size_t)cache_pos * head_dim;
            uint16_t* vc = v_cache + (size_t)h * max_seq_len * head_dim + (size_t)cache_pos * head_dim;
            for (int d = 0; d < head_dim; d += 8) {
                __m128i kh = _mm256_cvtps_ph(_mm256_loadu_ps(k_row + d), _MM_FROUND_TO_NEAREST_INT);
                __m128i vh = _mm256_cvtps_ph(_mm256_loadu_ps(v_row + d), _MM_FROUND_TO_NEAREST_INT);
                _mm_storeu_si128((__m128i*)(kc + d), kh);
                _mm_storeu_si128((__m128i*)(vc + d), vh);
            }
        }

        for (int h = 0; h < n_heads; h++) {
            int kh = h / n_rep;
            for (int s = 0; s < ring_len; s++) {
                int cache_idx = (ring_start + s) % max_seq_len;
                const uint16_t* k_row = k_cache + (size_t)kh * max_seq_len * head_dim + (size_t)cache_idx * head_dim;
                const float* qh = qb + h * head_dim;
                int d = 0;
                __m256 sum_v = _mm256_setzero_ps();
                for (; d + 8 <= head_dim; d += 8) {
                    __m128i k16 = _mm_loadu_si128((const __m128i*)(k_row + d));
                    __m256 k32 = _mm256_cvtph_ps(k16);
                    __m256 qv8 = _mm256_loadu_ps(qh + d);
                    sum_v = _mm256_fmadd_ps(qv8, k32, sum_v);
                }
                float sum = sum_v[0] + sum_v[1] + sum_v[2] + sum_v[3]
                          + sum_v[4] + sum_v[5] + sum_v[6] + sum_v[7];
                for (; d < head_dim; d++) {
                    __m128i k_fp16 = _mm_set1_epi16((short)k_row[d]);
                    float k_val = _mm_cvtss_f32(_mm_cvtph_ps(k_fp16));
                    sum += qh[d] * k_val;
                }
                scores[h * max_seq + s] = sum * inv_sqrt_d;
            }
        }

        // Causal mask + softmax per head (ring_start + s = actual position)
        for (int h = 0; h < n_heads; h++) {
            float* sh = scores + h * max_seq;
            float max_val = -1e9f;
            for (int s = 0; s < max_seq; s++) {
                int attn_pos = ring_start + s;
                float val = (attn_pos > pos) ? -1e9f : sh[s];
                sh[s] = val;
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

        // Weighted sum: output[h, d] = sum_s scores[h, s] * v_cache[kh, s, d]
        for (int h = 0; h < n_heads; h++) {
            int kh = h / n_rep;
            float* sh = scores + h * max_seq;
            float* out_h = output + b * n_heads * head_dim + h * head_dim;
            for (int d = 0; d < head_dim; d++) out_h[d] = 0.0f;
            for (int s = 0; s < max_seq; s++) {
                int cache_idx = (ring_start + s) % max_seq_len;
                const uint16_t* v_row = v_cache
                    + (size_t)kh * max_seq_len * head_dim + (size_t)cache_idx * head_dim;
                float score = sh[s];
                __m256 sv = _mm256_set1_ps(score);
                int d = 0;
                for (; d + 8 <= head_dim; d += 8) {
                    __m128i v16 = _mm_loadu_si128((const __m128i*)(v_row + d));
                    __m256 v32 = _mm256_cvtph_ps(v16);
                    __m256 out_v = _mm256_loadu_ps(out_h + d);
                    out_v = _mm256_fmadd_ps(sv, v32, out_v);
                    _mm256_storeu_ps(out_h + d, out_v);
                }
                for (; d < head_dim; d++) {
                    __m128i vh = _mm_set1_epi16((short)v_row[d]);
                    float v_val = _mm_cvtss_f32(_mm_cvtph_ps(vh));
                    out_h[d] += score * v_val;
                }
            }
        }
    }
}

// ─── Helper: quantize float32 activations → uint8 (+128 offset) ────────
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
            int q = (int)(act[t * D + i] * inv + 128.5f);
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

// v2.6.0: Reset KV cache — zeros all cache data without freeing allocation.
ATLAS_API void atlas_reset_cache(void* model) {
    AtlasModel* m = (AtlasModel*)model;
    if (!m || !m->k_cache) return;
    size_t cache_bytes = (size_t)m->n_kv_heads * m->cache_max_seq_len * m->head_dim * m->n_layers * sizeof(uint16_t);
    memset(m->k_cache, 0, cache_bytes);
    memset(m->v_cache, 0, cache_bytes);
}

// ─── Helper: horizontal sum of __m256 float ──────────────────────────
static inline float hsum_ps(__m256 v) {
    __m128 l = _mm256_castps256_ps128(v);
    __m128 h = _mm256_extractf128_ps(v, 1);
    l = _mm_add_ps(l, h);
    l = _mm_hadd_ps(l, l);
    l = _mm_hadd_ps(l, l);
    return _mm_cvtss_f32(l);
}

// ─── v2.7.0: TurboQuant fused matmul kernel (2-bit packed, K_tile=32) ───
// Fused 2-bit unpack (SSE4.1) + f32×f32 FMA (AVX2) with g128 block-scaling.
// No intermediate int8 buffer — decode on-the-fly in registers.
// ttype=7 tensor_data layout: [block_size:1][n_blocks:2][scales:N_blocks*2][packed:rows*packed_cols]
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
            dst[0 * rows_packed + ur] = out4[0];
            dst[1 * rows_packed + ur] = out4[1];
            dst[2 * rows_packed + ur] = out4[2];
            dst[3 * rows_packed + ur] = out4[3];
        }
    }
}

// ─── Fused TQ1 decode + int32 dot product for packed weights ──────────
// Decodes TQ1 bytes to int8 trits on-the-fly and computes the activation dot
// product in one pass. No intermediate decode buffer needed.
// Uses the +128 offset trick:
//   sum(act_u8[i] * w[i]) → dot = sum(act_u8[i] * lut[t]) - 128 * sum_w
//   result = dot / (127.0 * scale)  -- dequant scale applied by caller
static void matmul_tq1_packed_reorder(int rows, int input_dim,
    const uint8_t* packed, int packed_cols,
    const uint8_t* act_u8, const float* max_abs,
    float scale, float* output, int B) {
    init_tq1_decode_lut();
    int rows_packed = rows / 4;

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

// ─── Block-scaled TQ1 matmul (ttype=5, Bonsai g128 per-row format) ──
// Per-row per-block fp16 scales decoded from tensor_data header.
// Each output row has its own set of n_blocks fp16 scales.
// tensor_data layout: [block_size:1][n_blocks:2][scales: rows*n_blocks*2][packed_TQ1]
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

// VNNI kernel lives in atlas_vnni.cpp (compiled with target("avx10.2"))
static int g_has_avx512_vnni = -1;

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

    // Initialize VNNI flag before OMP region (single-threaded)
    if (g_has_avx512_vnni < 0) {
        g_has_avx512_vnni = check_avx512_vnni() ? 1 : 0;
    }

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
                        if (g_has_avx512_vnni) {
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
    // skip flags byte so ttype=5 layout aligns (scales at +3, packed at +3+rows*n_blocks*2)
    const uint8_t* w5 = tensor_data + 1;
    matmul_tq1_block_fused_s8(rows, input_dim, packed_cols,
        w5, block_size, n_blocks, act_f32, output, B);
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

    uint8_t* buf = (uint8_t*)malloc(total);
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
                int t = (v >= 0) ? (int)(v + 0.5f) : (int)(v - 0.5f);
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
            uint8_t* new_data = (uint8_t*)malloc(new_size);
            if (!new_data) continue;
            memcpy(new_data, t.data, 3);              // block_size + n_blocks
            new_data[3] = 0;                          // flags = 0
            memcpy(new_data + 4, t.data + 3, t.data_size - 3);  // scales + packed
            if (t.data && !m->is_mapped(t.data)) free(t.data);
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
        int ret = quantize_weights_to_tq2(i8_weights, scale, rows, cols,
                                          &tq2_buf, &tq2_size, 128);
        if (ret == 0 && tq2_buf) {
            if (t.data && !m->is_mapped(t.data)) {
                if (t.ttype == 0 || t.ttype == 8) {
                    free((void*)i8_weights);  // we allocated this
                }
                free(t.data);
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


// ─── Internal: forward one transformer layer ──────────────────────────
// input: [B, H] float32 (read-only, preserved for residual)
// output: [B, H] float32 (must not alias input)
// K/V cache is accessed from model struct (int8 + per-position scaling)
static void forward_layer_internal(
    AtlasModel* m,
    const float* input, float* output, int B,
    const int* positions,
    uint16_t* k_cache_layer, uint16_t* v_cache_layer,
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
    }
    float* x_norm = output;

    // ─── 2. QKV projections (int8) ───
    auto t3_dim = [](const TensorInfo& t) -> int {
        if (t.ttype == 10 || t.ttype == 5) return t.packed_cols * 5;
        if (t.ttype == 3) return (int)((t.data_size - 2 - (int64_t)t.row_dim * 4) / t.row_dim);
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
        // TQ2 universal block-scaled ternary matmul
        {
            auto& t = m->tensors[idx_q];
            matmul_tq2(t.row_dim, max_qkv_dim, t.packed_cols,
                t.data, t.block_size, t.n_blocks,
                m->buf_act, m->buf_gate, B);
        }
        {
            auto& t = m->tensors[idx_k];
            matmul_tq2(t.row_dim, max_qkv_dim, t.packed_cols,
                t.data, t.block_size, t.n_blocks,
                m->buf_act, m->buf_hidden, B);
        }
        {
            auto& t = m->tensors[idx_v];
            matmul_tq2(t.row_dim, max_qkv_dim, t.packed_cols,
                t.data, t.block_size, t.n_blocks,
                m->buf_act, m->buf_up, B);
        }
    } else if (tq.ttype == 5) {
        // v2.7.0: Fused symmetric int8 quant + ternary matmul + per-block dequant
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
            get_i8(tq, w, rs, rows, dim, scale);
            matmul_f32_reorder(rows, dim, w, m->buf_act, scale, m->buf_gate, B);
        }
        {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(tk, w, rs, rows, dim, scale);
            matmul_f32_reorder(rows, dim, w, m->buf_act, scale, m->buf_hidden, B);
        }
        {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(tv, w, rs, rows, dim, scale);
            matmul_f32_reorder(rows, dim, w, m->buf_act, scale, m->buf_up, B);
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
    const uint8_t* qn_w = (idx_q_norm >= 0 && m->model_arch == ARCH_QWEN3) ? m->tensors[idx_q_norm].data : nullptr;
    const uint8_t* kn_w = (idx_k_norm >= 0 && m->model_arch == ARCH_QWEN3) ? m->tensors[idx_k_norm].data : nullptr;
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

    // ─── 4. O projection (int8) ───
    {
        auto& to = m->tensors[idx_o];
        if (to.ttype == 10) {
        auto& t = m->tensors[idx_o];
        matmul_tq2(t.row_dim, qd, t.packed_cols,
            t.data, t.block_size, t.n_blocks,
            attn_out, m->buf_gate, B);
    } else if (to.ttype == 5) {
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * to.packed_cols * 5, attn_out + b * qd, qd * sizeof(float));
                memset(m->buf_act + b * to.packed_cols * 5 + qd, 0,
                       (to.packed_cols * 5 - qd) * sizeof(float));
            }
            matmul_tq1_block_fused_s8(to.row_dim, to.packed_cols * 5, to.packed_cols,
                to.data, to.block_size, to.n_blocks,
                m->buf_act, m->buf_gate, B);
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

    // ─── 5. Residual: output = input + attn_out_proj ───
    for (int i = 0; i < B * H; i++) {
        output[i] = input[i] + m->buf_gate[i];
    }
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
    float* x_norm2 = m->buf_act;

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
        {
            auto& t = m->tensors[idx_gate];
            matmul_tq2(t.row_dim, ffn_dim, t.packed_cols,
                t.data, t.block_size, t.n_blocks,
                m->buf_act, m->buf_gate, B);
        }
        {
            auto& t = m->tensors[idx_up];
            matmul_tq2(t.row_dim, ffn_dim, t.packed_cols,
                t.data, t.block_size, t.n_blocks,
                m->buf_act, m->buf_up, B);
        }
    } else if (tg.ttype == 5) {
        // v2.7.0: Fused symmetric int8 quant + ternary matmul + per-block dequant
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
    } else if (m->use_f32_matmul) {
        // Full-precision FFN: f32 activations × int8 weights, no quantization
        int8_t* gw; int32_t* grs; int g_rows, g_dim_v; float g_scale;
        int8_t* uw; int32_t* urs; int u_rows, u_dim_v; float u_scale;
        get_i8(tg, gw, grs, g_rows, g_dim_v, g_scale);
        get_i8(tu, uw, urs, u_rows, u_dim_v, u_scale);
        int rows = g_rows, dim_w = g_dim_v;
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
    } else {
        quantize_f32_to_u8(m->buf_act, B, ffn_dim, max_abs, m->buf_i8);

        int8_t* gw; int32_t* grs; int g_rows, g_dim_v; float g_scale;
        int8_t* uw; int32_t* urs; int u_rows, u_dim_v; float u_scale;
        get_i8(tg, gw, grs, g_rows, g_dim_v, g_scale);
        get_i8(tu, uw, urs, u_rows, u_dim_v, u_scale);
        int rows = g_rows, dim_w = g_dim_v;
        int rows_packed = rows / 4;

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int ur = 0; ur < rows_packed; ur++) {
            const int8_t* gw4 = gw + ur * 4 * dim_w;
            const int8_t* uw4 = uw + ur * 4 * dim_w;
            int32_t g_off[4] = {128 * grs[ur*4+0], 128 * grs[ur*4+1],
                                128 * grs[ur*4+2], 128 * grs[ur*4+3]};
            int32_t u_off[4] = {128 * urs[ur*4+0], 128 * urs[ur*4+1],
                                128 * urs[ur*4+2], 128 * urs[ur*4+3]};

            for (int b = 0; b < B; b++) {
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

    // ─── 8. Fused SiLU(gate)*up → down matmul with optional ffn_sub_norm ───
    {
        auto& td = m->tensors[idx_down];
        if (td.ttype == 10) {
            int down_dim = td.packed_cols * 5;
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_act + b * down_dim;
                for (int i = 0; i < inter; i++) {
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                }
                for (int i = inter; i < down_dim; i++) tmp[i] = 0.0f;
            }
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_act + b * down_dim;
                    float ss = 0.0f;
                    for (int i = 0; i < inter; i++) ss += ob[i] * ob[i];
                    float rms = 1.0f / sqrtf(ss / inter + 1e-6f);
                    for (int i = 0; i < inter; i++) {
                        uint16_t w16; memcpy(&w16, snw + i * 2, 2);
                        ob[i] *= rms * fp16_to_fp32(w16);
                    }
                }
            }
            matmul_tq2(td.row_dim, down_dim, td.packed_cols,
                td.data, td.block_size, td.n_blocks,
                m->buf_act, m->buf_gate, B);
        } else if (td.ttype == 5) {
            int down_dim = td.packed_cols * 5;
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_act + b * down_dim;
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
                    float* ob = m->buf_act + b * down_dim;
                    float ss = 0.0f;
                    for (int i = 0; i < inter; i++) ss += ob[i] * ob[i];
                    float rms = 1.0f / sqrtf(ss / inter + 1e-6f);
                    for (int i = 0; i < inter; i++) {
                        uint16_t w16; memcpy(&w16, snw + i * 2, 2);
                        ob[i] *= rms * fp16_to_fp32(w16);
                    }
                }
            }
            matmul_tq1_block_fused_s8(td.row_dim, down_dim, td.packed_cols,
                td.data, td.block_size, td.n_blocks,
                m->buf_act, m->buf_gate, B);
        } else if (td.ttype == 7) {
            int down_dim = td.packed_cols * 4;
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_act + b * down_dim;
                for (int i = 0; i < inter; i++) {
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                }
                for (int i = inter; i < down_dim; i++) tmp[i] = 0.0f;
            }
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_act + b * down_dim;
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
                m->buf_act, m->buf_gate, B);
        } else if (td.ttype == 0) {
            const uint8_t* wp; int rows, dim, pc; float scale;
            get_tq1_packed(td, wp, rows, dim, pc, scale);
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_act + b * dim;
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
                    float* ob = m->buf_act + b * dim;
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
                const float* tmp = m->buf_act + b * dim;
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
                float* tmp = m->buf_act + b * down_dim;
                for (int i = 0; i < inter; i++) {
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                }
                for (int i = inter; i < down_dim; i++) tmp[i] = 0.0f;
            }
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_act + b * down_dim;
                    float ss = 0.0f;
                    for (int i = 0; i < inter; i++) ss += ob[i] * ob[i];
                    float rms = 1.0f / sqrtf(ss / inter + 1e-6f);
                    for (int i = 0; i < inter; i++) {
                        uint16_t w16; memcpy(&w16, snw + i * 2, 2);
                        ob[i] *= rms * fp16_to_fp32(w16);
                    }
                }
            }
            quantize_f32_to_u8(m->buf_act, B, down_dim, max_abs, m->buf_i8);
            matmul_i4_reorder_deq(td.row_dim, d_cols, d_pw, d_rs, m->buf_i8, max_abs, d_scale, m->buf_act, m->buf_gate, B);
        } else {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(td, w, rs, rows, dim, scale);
            if (m->use_ternary_matmul) {
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_act + b * dim;
                for (int i = 0; i < inter; i++) {
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                }
                for (int i = inter; i < dim; i++) tmp[i] = 0.0f;
            }
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_act + b * dim;
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
                const float* tmp = m->buf_act + b * dim;
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
                float* tmp = m->buf_act + b * dim;
                for (int i = 0; i < inter; i++)
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                if (m->model_arch == ARCH_BITNET && idx_k_norm >= 0) {
                    apply_sub_norm(tmp, inter, m->tensors[idx_k_norm].data);
                }
                for (int i = inter; i < dim; i++) tmp[i] = 0.0f;
            }
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_act + b * dim;
                    float ss = 0.0f;
                    for (int i = 0; i < inter; i++) ss += ob[i] * ob[i];
                    float rms = 1.0f / sqrtf(ss / inter + 1e-6f);
                    for (int i = 0; i < inter; i++) {
                        uint16_t w16; memcpy(&w16, snw + i * 2, 2);
                        ob[i] *= rms * fp16_to_fp32(w16);
                    }
                }
            }
            matmul_f32_reorder(rows, dim, w, m->buf_act, scale, m->buf_gate, B);
        } else {
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_act + b * dim;
                for (int i = 0; i < inter; i++) {
                    tmp[i] = gate_activation(g[i], m->model_arch == ARCH_BITNET || m->use_relu2) * u[i];
                }
                for (int i = inter; i < dim; i++) tmp[i] = 0.0f;
            }
            if (idx_ffn_sub_norm >= 0) {
                auto& sn = m->tensors[idx_ffn_sub_norm];
                const uint8_t* snw = sn.data;
                for (int b = 0; b < B; b++) {
                    float* ob = m->buf_act + b * dim;
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
                const float* tmp = m->buf_act + b * dim;
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

    // ─── 10. Residual: output += down_proj ───
    for (int i = 0; i < B * H; i++) {
        output[i] += m->buf_gate[i];
    }
}


// ─── Forward ALL transformer layers in one C call ────────────────────
// (single-layer atlas_forward_layer removed — fusion is always used)
// hidden_states: [B, H] float32 — overwritten with final layer output
// positions: [B] int32 position indices
// layer_idx: [n_layers * 9] int32 — flat array of tensor indices per layer
//           (ln1, q, k, v, o, ln2, gate, up, down) repeated for each layer
// K/V cache is internal to the model (int8 + per-position scales)
ATLAS_API void atlas_forward(
    AtlasModel* m,
    float* hidden_states, int B,
    const int* positions,
    int max_seq_len, int seq_now,
    const int* layer_idx, int n_layers) {

    m->ensure_buffers(B);
    m->ensure_cache(max_seq_len);
    int H = m->hidden_dim;
    ATLAS_LOG("atlas_forward: B=%d max_seq=%d seq_now=%d H=%d nKV=%d hd=%d n_layers=%d stride=%d\n",
              B, max_seq_len, seq_now, H, m->n_kv_heads, m->head_dim, n_layers, m->layer_stride);
    int nKV = m->n_kv_heads, hd = m->head_dim;

    // Ping-pong: layer N output goes to separate buf_out (not buf_hidden, which is scratch)
    float* buf_a = hidden_states;
    float* buf_b = m->buf_out;

    for (int L = 0; L < n_layers; L++) {
        const int* idx = layer_idx + L * m->layer_stride;
        uint16_t* kc = m->k_cache + (size_t)L * nKV * max_seq_len * hd;
        uint16_t* vc = m->v_cache + (size_t)L * nKV * max_seq_len * hd;
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
        forward_layer_internal(m, buf_a, buf_b, B, positions,
            kc, vc, max_seq_len, seq_now,
            idx[0], idx[1], idx[2], idx[3], idx[4],
            idx[5], idx[6], idx[7], idx[8],
            qn_i, kn_i, asn_i, fsn_i);
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
            int q = (int)(v / inv + 0.5f);
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

    for (int b = 0; b < B; b++) {
        float ma = 1e-5f;
        for (int i = 0; i < H; i++) {
            float v = fabsf(act[b * H + i]);
            if (v > ma) ma = v;
        }
        max_abs[b] = ma;
        float inv = 127.0f / ma;
        for (int i = 0; i < H; i++) {
            int q = (int)(act[b * H + i] * inv + 128.5f);
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            act_u8[b * H + i] = (uint8_t)q;
        }
    }

    const int8_t* w = m->lm_head_i8;
    const int32_t* offs = m->lm_head_offsets;
    const float* scales = m->lm_head_scales;

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
    *output = gumbel_sample(logits, m ? m->vocab_size : 131072,
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
    // Detect architecture from tensor names (fallback when no v8 meta-block)
    int stride = m->layer_stride;
    int model_arch = m->model_arch;
    if (!m->has_meta) {
        stride = 9;
        model_arch = ARCH_LLAMA;
        char test_name[128];
        snprintf(test_name, sizeof(test_name), "model.layers.0.self_attn.attn_sub_norm.weight");
        if (find(test_name) >= 0) { stride = 11; model_arch = ARCH_BITNET; }
        else {
            snprintf(test_name, sizeof(test_name), "model.layers.0.self_attn.q_norm.weight");
            if (find(test_name) >= 0) { stride = 11; model_arch = ARCH_QWEN3; }
        }
        m->layer_stride = stride;
        m->model_arch = model_arch;
        // RoPE format: interleaved for Qwen3 and Falcon3 (head_dim>=256), half-split for Llama/BitNet
        m->rope_interleaved = (model_arch == ARCH_QWEN3) || (m->head_dim >= 256);
        if (model_arch == ARCH_BITNET) m->use_f32_matmul = 1;
    }
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
    int* output_ids)
{
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
            embed_buf[i * H + j] = fp16_to_fp32(embed_w[tid * H + j]);
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
    atlas_forward(m, embed_buf, n_new, positions,
                  max_seq_len, n_input,
                  layer_idx, m->n_layers);
    ATLAS_LOG("prefill forward done\n");

    // Final RMSNorm + LM head — only the last new token's logits are needed
    {
        const float* x = embed_buf + (int64_t)(n_new - 1) * H;
        ATLAS_LOG("prefill rmsnorm: x=%p norm_w=%p h_norm=%p H=%d\n", (void*)x, (void*)norm_w, (void*)h_norm, H);
        atlas_rmsnorm_f32(x, norm_w, h_norm, H, 1e-6f);
        ATLAS_LOG("prefill rmsnorm done, calling lmhead_gemv\n");
        atlas_lmhead_gemv(m, h_norm, logits, 1);
        ATLAS_LOG("prefill lmhead done\n");
    }

    // DEBUG: print top-5 logits from prefill (enable for diagnostics)
    // { int idx[5]={0}; float vl[5]={-1e10f}; int sc=0; for (int i=0;i<V;i++) { float v=logits[i]; int p=4; while(p>=0&&v>vl[p]) p--; p++; if(p<5) { memmove(idx+p+1,idx+p,(4-p)*sizeof(int)); memmove(vl+p+1,vl+p,(4-p)*sizeof(float)); idx[p]=i; vl[p]=v; if(sc<5) sc++; } } fprintf(stderr,"[DEBUG] Prefill top-5:%d(%.1f)%d(%.1f)%d(%.1f)%d(%.1f)%d(%.1f)\n",idx[0],vl[0],idx[1],vl[1],idx[2],vl[2],idx[3],vl[3],idx[4],vl[4]); }

    // Sample first token from prefill logits
    const int eos_id = m->eos_id;
    if (n_gen < min_new_tokens && eos_id >= 0 && eos_id < V) logits[eos_id] = -1e9f;
    int next_token = gumbel_sample(logits, V, temperature, top_k, top_p,
                                    context, n_input, repetition_penalty);
    context[n_input] = next_token;
    output_ids[n_gen++] = next_token;

    if (next_token == eos_id) {
        atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
        atlas_vfree((uint8_t*)logits); atlas_vfree((uint8_t*)positions);
        atlas_vfree((uint8_t*)context);
        return n_gen;
    }

    // ─── Decode loop (v2.5.0: ring buffer — no veto, wraps at max_seq_len) ───
    atlas_vfree((uint8_t*)positions);
    for (int step = 1; step < max_new_tokens; step++) {
        // Embed last generated token
        int tid = next_token;
        if (tid < 0 || tid >= V) tid = 0;
        float* h = embed_buf;  // reuse embed_buf as single-token buffer
        for (int j = 0; j < H; j++)
            h[j] = fp16_to_fp32(embed_w[tid * H + j]);

        int seq_now = n_input + step;
        int pos = seq_now - 1;
        ATLAS_LOG("decode step=%d seq_now=%d pos=%d\n", step, seq_now, pos);
        atlas_forward(m, h, 1, &pos,
                      max_seq_len, seq_now,
                      layer_idx, m->n_layers);
        ATLAS_LOG("decode forward done\n");

        atlas_rmsnorm_f32(h, norm_w, h_norm, H, 1e-6f);
        atlas_lmhead_gemv(m, h_norm, logits, 1);
        ATLAS_LOG("decode lmhead done\n");

        if (n_gen < min_new_tokens && eos_id >= 0 && eos_id < V) logits[eos_id] = -1e9f;
        next_token = gumbel_sample(logits, V, temperature, top_k, top_p,
                                    context, n_input + n_gen, repetition_penalty);
        context[n_input + n_gen] = next_token;
        output_ids[n_gen++] = next_token;

        if (next_token == eos_id) break;  // EOS
    }

    atlas_vfree((uint8_t*)embed_buf);
    atlas_vfree((uint8_t*)h_norm);
    atlas_vfree((uint8_t*)logits);
    atlas_vfree((uint8_t*)context);

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
    atlas_token_callback callback, void* user_data)
{
    if (!m || !input_ids || !callback || n_input < 1 || max_new_tokens < 1)
        return -1;
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
    uint8_t* norm_w = m->tensors[idx_norm].data;

    if (!m->lm_head_quantized) return -1;

    ensure_layer_idx(m);
    const int* layer_idx = m->layer_idx_cache.data();

    if (cache_offset < 0) cache_offset = 0;
    if (cache_offset >= n_input) cache_offset = n_input - 1;
    int n_new = n_input - cache_offset;

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

    // ─── Prefill: NEW tokens only ───
    int n_gen = 0;
    for (int i = 0; i < n_new; i++) {
        int idx = cache_offset + i;
        int tid = input_ids[idx];
        if (tid < 0 || tid >= V) tid = 0;
        for (int j = 0; j < H; j++)
            embed_buf[i * H + j] = fp16_to_fp32(embed_w[tid * H + j]);
    }

    int* positions = (int*)atlas_valloc((size_t)n_new * sizeof(int));
    if (!positions) {
        atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
        atlas_vfree((uint8_t*)logits); atlas_vfree((uint8_t*)context); return -1;
    }
    for (int i = 0; i < n_new; i++) positions[i] = cache_offset + i;

    atlas_forward(m, embed_buf, n_new, positions,
                  max_seq_len, n_input,
                  layer_idx, m->n_layers);

    {
        const float* x = embed_buf + (int64_t)(n_new - 1) * H;
        atlas_rmsnorm_f32(x, norm_w, h_norm, H, 1e-6f);
        atlas_lmhead_gemv(m, h_norm, logits, 1);
    }

    const int eos_id = m->eos_id;
    if (n_gen < min_new_tokens && eos_id >= 0 && eos_id < V) logits[eos_id] = -1e9f;
    int next_token = gumbel_sample(logits, V, temperature, top_k, top_p,
                                    context, n_input, repetition_penalty);
    context[n_input] = next_token;
    callback(next_token, user_data);
    n_gen++;

    if (next_token == eos_id) {
        atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
        atlas_vfree((uint8_t*)logits); atlas_vfree((uint8_t*)positions);
        atlas_vfree((uint8_t*)context);
        return n_gen;
    }

    // ─── Decode loop (v2.5.0: ring buffer — no veto, wraps at max_seq_len) ───
    atlas_vfree((uint8_t*)positions);
    for (int step = 1; step < max_new_tokens; step++) {
        int tid = next_token;
        if (tid < 0 || tid >= V) tid = 0;
        float* h = embed_buf;
        for (int j = 0; j < H; j++)
            h[j] = fp16_to_fp32(embed_w[tid * H + j]);

        int seq_now = n_input + step;
        int pos = seq_now - 1;
        atlas_forward(m, h, 1, &pos,
                      max_seq_len, seq_now,
                      layer_idx, m->n_layers);

        atlas_rmsnorm_f32(h, norm_w, h_norm, H, 1e-6f);
        atlas_lmhead_gemv(m, h_norm, logits, 1);

        if (n_gen < min_new_tokens && eos_id >= 0 && eos_id < V) logits[eos_id] = -1e9f;
        next_token = gumbel_sample(logits, V, temperature, top_k, top_p,
                                    context, n_input + n_gen, repetition_penalty);
        context[n_input + n_gen] = next_token;
        callback(next_token, user_data);
        n_gen++;

        if (next_token == eos_id) break;
    }

    atlas_vfree((uint8_t*)embed_buf);
    atlas_vfree((uint8_t*)h_norm);
    atlas_vfree((uint8_t*)logits);
    atlas_vfree((uint8_t*)context);

    return n_gen;
}

}  // extern "C"
