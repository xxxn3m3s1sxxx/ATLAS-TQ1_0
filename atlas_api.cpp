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

    static std::vector<int> sidx;

    if (top_k > 0 && top_k < V) {
        static std::vector<float> scopy;
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

static inline float fp16_to_fp32(uint16_t h) {
    float r;
    __m128i h4 = _mm_cvtsi32_si128((int)(unsigned)h);  // zero-extend to 32-bit
    __m128 f4 = _mm_cvtph_ps(h4);                       // F16C: fp16→fp32
    _mm_store_ss(&r, f4);
    return r;
}

// ─── Tensor info ────────────────────────────────────────────────────────
struct TensorInfo {
    int ttype;          // 0=TQ1, 1=norm/embed, 2=other
    int row_dim;        // output rows (weight) or flat size (norm/embed)
    int packed_cols;    // packed bytes per row (0 for non-TQ1)
    uint32_t file_offset;
    uint8_t* data;      // loaded data (scale prefix + packed for TQ1, raw fp16 for others)
    int data_size;
};

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
    int layer_stride = 9;    // tensors per layer (9 Falcon3, 11 Qwen3 with QK-Norm)
    int eos_id = 11;   // default Falcon3 endoftext
    int pad_id = 0;    // default padding token
    bool use_f32_matmul = false; // skip activation quantization (1B model needs full precision)
    bool use_ternary_matmul = false; // v1.3.0: vpsignb-based ternary-add kernel (no multiplication)
    bool use_packed_matmul = false; // v1.3.1: operate on 2-bit packed ternary weights (4× less memory)
    bool use_hybrid_matmul = false; // v1.3.2: FFN int8 cache, QKV packed
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

    // v2.3.0: Int8 KV cache (internal, auto-allocated)
    int8_t* k_cache = nullptr;
    int8_t* v_cache = nullptr;
    float* k_scale_cache = nullptr;
    float* v_scale_cache = nullptr;
    int cache_max_seq_len = 0;

    // Ensure KV cache is allocated for at least max_seq_len
    void ensure_cache(int max_seq_len) {
        if (max_seq_len <= cache_max_seq_len) return;
        size_t layer_sz = (size_t)n_kv_heads * max_seq_len * head_dim;
        size_t scale_sz = (size_t)n_kv_heads * max_seq_len;
        if (k_cache) atlas_vfree((uint8_t*)k_cache);
        if (k_scale_cache) atlas_vfree((uint8_t*)k_scale_cache);
        if (v_cache) atlas_vfree((uint8_t*)v_cache);
        if (v_scale_cache) atlas_vfree((uint8_t*)v_scale_cache);
        k_cache = (int8_t*)atlas_valloc(layer_sz * n_layers);
        k_scale_cache = (float*)atlas_valloc(scale_sz * n_layers * sizeof(float));
        v_cache = (int8_t*)atlas_valloc(layer_sz * n_layers);
        v_scale_cache = (float*)atlas_valloc(scale_sz * n_layers * sizeof(float));
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

        buf_gate = (float*)atlas_valloc((size_t)B * inter_dim * sizeof(float));
        buf_up = (float*)atlas_valloc((size_t)B * inter_dim * sizeof(float));
        buf_hidden = (float*)atlas_valloc((size_t)B * inter_dim * sizeof(float));
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

// ─── Load model ─────────────────────────────────────────────────────────
ATLAS_API AtlasModel* atlas_load(const char* path) {
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
    if (eos_val != 0 || hdr[45]|hdr[46]|hdr[47]|hdr[48]) m->eos_id = (int)eos_val;
    if (pad_val != 0 || hdr[49]|hdr[50]|hdr[51]|hdr[52]) m->pad_id = (int)pad_val;

    printf("[ATLAS] v%d model: %dL %dH %dI %d/%d heads %d vocab %.0f theta | %d tensors %s\n",
           version,
           m->n_layers, m->hidden_dim, m->inter_dim, m->n_heads, m->n_kv_heads,
           m->vocab_size, m->rope_theta, n_tensors,
           m->tokenizer_size > 0 ? "(embedded tokenizer)" : "");

    // Read directory
    m->tensors.resize(n_tensors);
    std::vector<uint32_t> file_offsets(n_tensors);
    FSEEK(f, 64, SEEK_SET);
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
            FSEEK(f, 64 + n_tensors * 12, SEEK_SET);
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

        if (t.ttype == 0) {  // TQ1: 2-byte scale + packed data
            t.data_size = 2 + t.row_dim * t.packed_cols;
        } else if (t.ttype == 1) {  // norm/embed: raw float16
            if (t.row_dim == m->vocab_size) {
                t.data_size = t.row_dim * m->hidden_dim * 2;
            } else {
                t.data_size = t.row_dim * 2;
            }
            t.packed_cols = 0;
        } else {  // lm_head / scales
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
    // Free KV cache (v2.3.0: int8 internal)
    if (m->k_cache) atlas_vfree((uint8_t*)m->k_cache);
    if (m->k_scale_cache) atlas_vfree((uint8_t*)m->k_scale_cache);
    if (m->v_cache) atlas_vfree((uint8_t*)m->v_cache);
    if (m->v_scale_cache) atlas_vfree((uint8_t*)m->v_scale_cache);
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
        printf("[CACHE] Saved %d int8 tensors (%.1f MB)\n", n, cache_size / 1e6);
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
        if (t.ttype == 0 && cttype == 3 && ds > 0 && off >= 0) {
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
// Frees packed data after decompression. Call after Python closes safetensors.
ATLAS_API void atlas_decompress_all(AtlasModel* m) {
    int total = 0;
    for (auto& t : m->tensors) {
        if (t.ttype != 0) continue;
        total++;

        int input_dim = t.packed_cols * 5;
        int n_vals = t.row_dim * input_dim;

        // Allocate: [scale_fp16(2)] [i8_data(rows × input_dim)] [row_sums(rows × 4)]
        uint8_t* new_data = atlas_valloc(2 + n_vals + t.row_dim * 4);
        new_data[0] = t.data[0];
        new_data[1] = t.data[1];

        init_tq1_decode_lut();
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

        // Free packed data, replace with int8
        if (t.data && !m->is_mapped(t.data)) atlas_vfree(t.data);
        t.data = new_data;
        t.data_size = 2 + n_vals + t.row_dim * 4;
        t.ttype = 3;  // int8-decoded
    }
    printf("[ATLAS] Decompressed %d TQ1 tensors to int8\n", total);
}

// ─── v1.3.2: Decompress only FFN tensors to int8 (gate/up/down) ────
ATLAS_API void atlas_decompress_ffn(AtlasModel* m) {
    int total = 0;
    for (size_t i = 0; i < m->tensors.size(); i++) {
        auto& t = m->tensors[i];
        if (t.ttype != 0) continue;
        // Check name: only decompress MLP tensors
        if (m->tensor_names.size() > i) {
            const std::string& name = m->tensor_names[i];
            if (name.find("mlp") == std::string::npos &&
                name.find("gate") == std::string::npos &&
                name.find("down") == std::string::npos) continue;
        }
        total++;

        int input_dim = t.packed_cols * 5;
        int n_vals = t.row_dim * input_dim;

        uint8_t* new_data = atlas_valloc(2 + n_vals + t.row_dim * 4);
        new_data[0] = t.data[0];
        new_data[1] = t.data[1];

        init_tq1_decode_lut();
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
        if (t.ttype != 3) continue;
        int n_vals = t.row_dim * t.packed_cols * 5;
        int8_t* data = (int8_t*)(t.data + 2);
        for (int64_t i = 0; i < n_vals; i += step) {
            volatile int sink = data[i]; (void)sink;
        }
        total += n_vals;
    }
    printf("[ATLAS] Prefetched %lld int8 values\n", (long long)total);
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
    if (input_dim) *input_dim = t.packed_cols * 5;
    if (row_sums) {
        int n_vals = t.row_dim * t.packed_cols * 5;
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
    if (col_dim) *col_dim = t.ttype == 0 ? t.packed_cols * 5 : 0;
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
                                    const int8_t* weights, const uint8_t* act_u8,
                                    const int32_t* row_sums, float* output,
                                    int n_tokens) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int r = 0; r < rows; r++) {
        const int8_t* w = weights + r * input_dim;
        int sum_w = row_sums[r];

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
// q/k: [n_heads × head_dim] float32 (interleaved format)
ATLAS_API void atlas_rope_f32(float* q, float* k, int n_heads, int n_kv_heads,
                               int head_dim, int position, float rope_theta) {
    float theta_base = rope_theta;
    for (int h = 0; h < n_heads; h++) {
        float* qh = q + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta_base, 2.0f * i / head_dim);
            float cos_v = cosf(position * freq);
            float sin_v = sinf(position * freq);
            float a = qh[2*i], b = qh[2*i+1];
            qh[2*i]   = a * cos_v - b * sin_v;
            qh[2*i+1] = a * sin_v + b * cos_v;
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float* kh = k + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta_base, 2.0f * i / head_dim);
            float cos_v = cosf(position * freq);
            float sin_v = sinf(position * freq);
            float a = kh[2*i], b = kh[2*i+1];
            kh[2*i]   = a * cos_v - b * sin_v;
            kh[2*i+1] = a * sin_v + b * cos_v;
        }
    }
}

// ─── Fused attention: RoPE + GQA + softmax + weighted sum ────────────
// v2.3.0: Int8 KV cache with per-position scaling (half memory bandwidth)
// q: [B, n_heads * head_dim] float32 — RoPE applied in-place, modified
// k: [B, n_kv_heads * head_dim] float32 — RoPE applied in-place, modified
// v: [B, n_kv_heads * head_dim] float32
// positions: [B] int32
// k_cache: [n_kv_heads, max_seq, head_dim] int8 — quantized K cache
// k_scale_cache: [n_kv_heads, max_seq] float — per-(kv_head,pos) scale for K
// v_cache: [n_kv_heads, max_seq, head_dim] int8 — quantized V cache
// v_scale_cache: [n_kv_heads, max_seq] float — per-(kv_head,pos) scale for V
// output: [B, n_heads * head_dim] float32
// q_norm_w, k_norm_w: [head_dim] uint8 fp16 RMSNorm weights (QK-Norm, Qwen3), NULL = skip
ATLAS_API void atlas_attention_f32(
    float* q, float* k, float* v, const int* positions,
    int8_t* k_cache, float* k_scale_cache,
    int8_t* v_cache, float* v_scale_cache,
    int max_seq_len, int seq_now, int B,
    int n_heads, int n_kv_heads, int head_dim,
    float rope_theta, float rope_scale, float* output,
    const uint8_t* q_norm_w, const uint8_t* k_norm_w) {

    int n_rep = n_heads / n_kv_heads;
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    // YaRN NTK-aware effective theta (rope_scale=4.0 for Bonsai-4B)
    float eff_theta = rope_theta;
    if (rope_scale > 1.001f) {
        eff_theta *= powf(rope_scale, (float)head_dim / (float)(head_dim - 2));
    }

    // Attention scores — heap-allocated, reusable buffer (avoids ~192KB stack alloca)
    int max_seq = seq_now > max_seq_len ? max_seq_len : seq_now;
    static float* scores_buf = nullptr;
    static size_t scores_cap = 0;
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

        // RoPE on Q (YaRN NTK-aware: use eff_theta instead of raw rope_theta)
        for (int h = 0; h < n_heads; h++) {
            float* qh = qb + h * head_dim;
            for (int i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(eff_theta, 2.0f * i / head_dim);
                float c = cosf(pos * freq), s = sinf(pos * freq);
                float a = qh[2*i], b0 = qh[2*i+1];
                qh[2*i]   = a * c - b0 * s;
                qh[2*i+1] = a * s + b0 * c;
            }
        }
        // RoPE on K
        for (int h = 0; h < n_kv_heads; h++) {
            float* kh = kb + h * head_dim;
            for (int i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(eff_theta, 2.0f * i / head_dim);
                float c = cosf(pos * freq), s = sinf(pos * freq);
                float a = kh[2*i], b0 = kh[2*i+1];
                kh[2*i]   = a * c - b0 * s;
                kh[2*i+1] = a * s + b0 * c;
            }
        }

        // QK-Norm (Qwen3: RMSNorm on Q/K after RoPE, before cache + score)
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

        // Store K, V into cache (fp32 -> int8 quantized, per-kv_head scaling)
        for (int h = 0; h < n_kv_heads; h++) {
            float* k_row = kb + h * head_dim;
            float* v_row = vb + h * head_dim;
            float k_max = 1e-10f, v_max = 1e-10f;
            for (int d = 0; d < head_dim; d++) {
                float ka = fabsf(k_row[d]), va = fabsf(v_row[d]);
                if (ka > k_max) k_max = ka;
                if (va > v_max) v_max = va;
            }
            float k_scale = k_max / 127.0f;
            float v_scale = v_max / 127.0f;
            float k_inv = 127.0f / k_max;
            float v_inv = 127.0f / v_max;
            int8_t* kc = k_cache + (size_t)h * max_seq_len * head_dim + (size_t)pos * head_dim;
            int8_t* vc = v_cache + (size_t)h * max_seq_len * head_dim + (size_t)pos * head_dim;
            for (int d = 0; d < head_dim; d++) {
                int kq = (int)(k_row[d] * k_inv);
                int vq = (int)(v_row[d] * v_inv);
                if (kq < -128) kq = -128; if (kq > 127) kq = 127;
                if (vq < -128) vq = -128; if (vq > 127) vq = 127;
                kc[d] = (int8_t)kq;
                vc[d] = (int8_t)vq;
            }
            k_scale_cache[(size_t)h * max_seq_len + pos] = k_scale;
            v_scale_cache[(size_t)h * max_seq_len + pos] = v_scale;
        }

        for (int h = 0; h < n_heads; h++) {
            int kh = h / n_rep;
            for (int s = 0; s < max_seq; s++) {
                const int8_t* k_row = k_cache + (size_t)kh * max_seq_len * head_dim + (size_t)s * head_dim;
                float k_scale = k_scale_cache[(size_t)kh * max_seq_len + s];
                const float* qh = qb + h * head_dim;
                int d = 0;
                __m256 sum_v = _mm256_setzero_ps();
                for (; d + 8 <= head_dim; d += 8) {
                    // Load 8 int8 values, sign-extend to int16, then to int32, then to float
                    __m128i k8 = _mm_loadl_epi64((const __m128i*)(k_row + d));
                    __m256i k16 = _mm256_cvtepi8_epi16(k8);
                    __m256i k32_i = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(k16));
                    __m256 k32 = _mm256_cvtepi32_ps(k32_i);
                    __m256 qv8 = _mm256_loadu_ps(qh + d);
                    sum_v = _mm256_fmadd_ps(qv8, k32, sum_v);
                }
                float sum = sum_v[0] + sum_v[1] + sum_v[2] + sum_v[3]
                          + sum_v[4] + sum_v[5] + sum_v[6] + sum_v[7];
                for (; d < head_dim; d++)
                    sum += qh[d] * (float)k_row[d];
                scores[h * max_seq + s] = sum * k_scale * inv_sqrt_d;
            }
        }

        // Causal mask + softmax per head
        for (int h = 0; h < n_heads; h++) {
            float* sh = scores + h * max_seq;
            float max_val = -1e9f;
            for (int s = 0; s < max_seq; s++) {
                float val = (s > pos) ? -1e9f : sh[s];
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

        // Weighted sum: output[h, d] = sum_s scores[h, s] * v_cache[kh, s, d] * v_scale
        for (int h = 0; h < n_heads; h++) {
            int kh = h / n_rep;
            float* sh = scores + h * max_seq;
            float* out_h = output + b * n_heads * head_dim + h * head_dim;
            for (int d = 0; d < head_dim; d++) out_h[d] = 0.0f;
            for (int s = 0; s < max_seq; s++) {
                const int8_t* v_row = v_cache
                    + (size_t)kh * max_seq_len * head_dim + (size_t)s * head_dim;
                float v_scale = v_scale_cache[(size_t)kh * max_seq_len + s];
                float score = sh[s];
                __m256 sv = _mm256_set1_ps(score * v_scale);
                int d = 0;
                for (; d + 8 <= head_dim; d += 8) {
                    __m128i v8 = _mm_loadl_epi64((const __m128i*)(v_row + d));
                    __m256i v16 = _mm256_cvtepi8_epi16(v8);
                    __m256i v32_i = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v16));
                    __m256 v32 = _mm256_cvtepi32_ps(v32_i);
                    __m256 out_v = _mm256_loadu_ps(out_h + d);
                    out_v = _mm256_fmadd_ps(sv, v32, out_v);
                    _mm256_storeu_ps(out_h + d, out_v);
                }
                for (; d < head_dim; d++)
                    out_h[d] += score * (float)v_row[d] * v_scale;
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

// ─── v2.4.0: Set layer stride (9 Falcon3, 11 Qwen3 with QK-Norm) ────
ATLAS_API void atlas_set_layer_stride(AtlasModel* m, int stride) {
    if (m) m->layer_stride = stride;
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

// ─── Helper: f32×i8 matmul + reorder (no activation quantization) ───
// act_f32: [B, input_dim] float activations (not quantized)
// weights: [rows, input_dim] int8 weights
// scale: per-tensor dequant scale
// output: [B, rows] reordered float output
static void matmul_f32_reorder(int rows, int input_dim,
    const int8_t* weights, const float* act_f32,
    float scale, float* output, int B) {
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


// ─── Internal: forward one transformer layer ──────────────────────────
// input: [B, H] float32 (read-only, preserved for residual)
// output: [B, H] float32 (must not alias input)
// K/V cache is accessed from model struct (int8 + per-position scaling)
static void forward_layer_internal(
    AtlasModel* m,
    const float* input, float* output, int B,
    const int* positions,
    int8_t* k_cache_layer, float* k_scale_layer,
    int8_t* v_cache_layer, float* v_scale_layer,
    int max_seq_len, int seq_now,
    int idx_ln1, int idx_q, int idx_k, int idx_v, int idx_o,
    int idx_ln2, int idx_gate, int idx_up, int idx_down,
    int idx_q_norm = -1, int idx_k_norm = -1) {

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
    auto get_i8 = [](const TensorInfo& t, int8_t*& w, int32_t*& rs,
                     int& rows, int& dim, float& scale) {
        if (t.ttype != 3) { rows = 0; dim = 0; w = nullptr; rs = nullptr; return; }
        uint16_t sr; memcpy(&sr, t.data, 2);
        scale = fp16_to_fp32(sr);
        rows = t.row_dim;
        dim = t.packed_cols * 5;
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
    int i8_q_dim = tq.packed_cols * 5;
    int i8_k_dim = tk.packed_cols * 5;
    int max_qkv_dim = i8_q_dim > i8_k_dim ? i8_q_dim : i8_k_dim;

    for (int b = 0; b < B; b++) {
        memcpy(m->buf_act + b * max_qkv_dim, x_norm + b * H, H * sizeof(float));
        memset(m->buf_act + b * max_qkv_dim + H, 0,
               (max_qkv_dim - H) * sizeof(float));
    }

    float* max_abs = (float*)alloca(B * sizeof(float));

    auto& tv = m->tensors[idx_v];
    if (tq.ttype == 0) {
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
        int i8_v_dim = tv.packed_cols * 5;

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
    const uint8_t* qn_w = (idx_q_norm >= 0) ? m->tensors[idx_q_norm].data : nullptr;
    const uint8_t* kn_w = (idx_k_norm >= 0) ? m->tensors[idx_k_norm].data : nullptr;
    atlas_attention_f32(q_f32, k_f32, v_f32, positions,
        k_cache_layer, k_scale_layer, v_cache_layer, v_scale_layer,
        max_seq_len, seq_now, B,
        nH, nKV, hd, theta, m->rope_scale, attn_out, qn_w, kn_w);

    // ─── 4. O projection (int8) ───
    {
        auto& to = m->tensors[idx_o];
        if (to.ttype == 0) {
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
    int g_dim = tg.packed_cols * 5;
    int u_dim = tu.packed_cols * 5;
    int ffn_dim = g_dim > u_dim ? g_dim : u_dim;

    for (int b = 0; b < B; b++) {
        memcpy(m->buf_act + b * ffn_dim, x_norm2 + b * H, H * sizeof(float));
        memset(m->buf_act + b * ffn_dim + H, 0, (ffn_dim - H) * sizeof(float));
    }

    if (tg.ttype == 0) {
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

    // ─── 8. Fused SiLU(gate)*up → down matmul ───
    {
        auto& td = m->tensors[idx_down];
        if (td.ttype == 0) {
            const uint8_t* wp; int rows, dim, pc; float scale;
            get_tq1_packed(td, wp, rows, dim, pc, scale);
            // SiLU(gate)*up → quantize to u8 → TQ1-packed matmul
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_act + b * dim;
                float mb = 1e-5f;
                for (int i = 0; i < inter; i++) {
                    float v = (g[i] / (1.0f + expf(-g[i]))) * u[i];
                    tmp[i] = v;
                    float av = fabsf(v);
                    if (av > mb) mb = av;
                }
                for (int i = inter; i < dim; i++) tmp[i] = 0.0f;
                float inv = 127.0f / mb;
                max_abs[b] = mb;
                for (int i = 0; i < dim; i++) {
                    int q = (int)(tmp[i] * inv + 128.5f);
                    if (q < 0) q = 0; if (q > 255) q = 255;
                    m->buf_i8[b * dim + i] = (uint8_t)q;
                }
            }
            matmul_tq1_packed_reorder(rows, dim, wp, pc, m->buf_i8, max_abs, scale, m->buf_gate, B);
        } else {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(td, w, rs, rows, dim, scale);
            if (m->use_ternary_matmul) {
            // Ternary-add: SiLU(gate)*up → quantize to u8 → vpsignb ternary-add → dequant
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_act + b * dim;
                float mb = 1e-5f;
                for (int i = 0; i < inter; i++) {
                    float v = (g[i] / (1.0f + expf(-g[i]))) * u[i];
                    tmp[i] = v;
                    float av = fabsf(v);
                    if (av > mb) mb = av;
                }
                for (int i = inter; i < dim; i++) tmp[i] = 0.0f;
                float inv = 127.0f / mb;
                max_abs[b] = mb;
                for (int i = 0; i < dim; i++) {
                    int q = (int)(tmp[i] * inv + 128.5f);
                    if (q < 0) q = 0; if (q > 255) q = 255;
                    m->buf_i8[b * dim + i] = (uint8_t)q;
                }
            }
            matmul_ternary_add_reorder(rows, dim, w, m->buf_i8, max_abs,
                                      scale, m->buf_gate, B);
        } else if (m->use_f32_matmul) {
            // Full-precision: compute SiLU(gate)*up directly into buf_act, then f32×i8 matmul
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_act + b * dim;
                for (int i = 0; i < inter; i++)
                    tmp[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
                for (int i = inter; i < dim; i++) tmp[i] = 0.0f;
            }
            matmul_f32_reorder(rows, dim, w, m->buf_act, scale, m->buf_gate, B);
        } else {
            // Quantized path: SiLU(gate)*up → quantize to u8 → maddubs × i8 → dequant
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_act + b * dim;
                float mb = 1e-5f;
                for (int i = 0; i < inter; i++) {
                    float v = (g[i] / (1.0f + expf(-g[i]))) * u[i];
                    tmp[i] = v;
                    float av = fabsf(v);
                    if (av > mb) mb = av;
                }
                for (int i = inter; i < dim; i++) tmp[i] = 0.0f;
                float inv = 127.0f / mb;
                max_abs[b] = mb;
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
    int nKV = m->n_kv_heads, hd = m->head_dim;

    // Ping-pong: layer N output goes to separate buf_out (not buf_hidden, which is scratch)
    float* buf_a = hidden_states;
    float* buf_b = m->buf_out;

    for (int L = 0; L < n_layers; L++) {
        const int* idx = layer_idx + L * m->layer_stride;
        int8_t* kc = m->k_cache + (size_t)L * nKV * max_seq_len * hd;
        float* ksc = m->k_scale_cache + (size_t)L * nKV * max_seq_len;
        int8_t* vc = m->v_cache + (size_t)L * nKV * max_seq_len * hd;
        float* vsc = m->v_scale_cache + (size_t)L * nKV * max_seq_len;
        int qn_i = (m->layer_stride >= 11) ? idx[9] : -1;
        int kn_i = (m->layer_stride >= 11) ? idx[10] : -1;
        forward_layer_internal(m, buf_a, buf_b, B, positions,
            kc, ksc, vc, vsc, max_seq_len, seq_now,
            idx[0], idx[1], idx[2], idx[3], idx[4],
            idx[5], idx[6], idx[7], idx[8],
            qn_i, kn_i);
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
    // Detect QK-Norm (Qwen3): stride = 11 if q_norm exists
    int stride = 9;
    char test_name[128];
    snprintf(test_name, sizeof(test_name), "model.layers.0.self_attn.q_norm.weight");
    if (find(test_name) >= 0) stride = 11;
    m->layer_stride = stride;
    m->layer_idx_cache.clear();
    m->layer_idx_cache.reserve(m->n_layers * stride);
    for (int L = 0; L < m->n_layers; L++) {
        char buf[128];
        auto push = [&](const char* suffix) {
            snprintf(buf, sizeof(buf), "model.layers.%d.%s", L, suffix);
            m->layer_idx_cache.push_back(find(buf));
        };
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
    m->has_layer_idx = true;
}

ATLAS_API int atlas_generate(AtlasModel* m,
    const int* input_ids, int n_input,
    int max_seq_len, int max_new_tokens,
    float temperature, int top_k, float top_p,
    float repetition_penalty,
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

    if (!m->lm_head_quantized) {
        fprintf(stderr, "[ATLAS] atlas_generate: lm_head not quantized\n");
        return -1;
    }

    ensure_layer_idx(m);
    const int* layer_idx = m->layer_idx_cache.data();

    // Allocate scratch: embedding buffer, norm scratch, logits
    float* embed_buf = (float*)atlas_valloc((size_t)n_input * H * sizeof(float));
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

    // ─── Prefill: embed all input tokens ───
    int n_gen = 0;
    for (int i = 0; i < n_input; i++) {
        int tid = input_ids[i];
        if (tid < 0 || tid >= V) tid = 0;
        for (int j = 0; j < H; j++)
            embed_buf[i * H + j] = fp16_to_fp32(embed_w[tid * H + j]);
    }

    // Build position array for prefill
    int* positions = (int*)atlas_valloc((size_t)n_input * sizeof(int));
    if (!positions) {
        atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
        atlas_vfree((uint8_t*)logits); atlas_vfree((uint8_t*)context); return -1;
    }
    for (int i = 0; i < n_input; i++) positions[i] = i;

    // Fused forward for all prompt tokens at once
    atlas_forward(m, embed_buf, n_input, positions,
                  max_seq_len, n_input,
                  layer_idx, m->n_layers);

    // Final RMSNorm + LM head — only the last prompt token's logits are needed
    {
        const float* x = embed_buf + (int64_t)(n_input - 1) * H;
        atlas_rmsnorm_f32(x, norm_w, h_norm, H, 1e-6f);
        atlas_lmhead_gemv(m, h_norm, logits, 1);
    }

    // Sample first token from prefill logits
    int next_token = gumbel_sample(logits, V, temperature, top_k, top_p,
                                    context, n_input, repetition_penalty);
    context[n_input] = next_token;
    output_ids[n_gen++] = next_token;

    const int eos_id = m->eos_id;
    if (next_token == eos_id) {
        atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
        atlas_vfree((uint8_t*)logits); atlas_vfree((uint8_t*)positions);
        atlas_vfree((uint8_t*)context);
        return n_gen;
    }

    // ─── Decode loop ───
    atlas_vfree((uint8_t*)positions);
    // Veto: clamp max_new_tokens to prevent KV cache overflow
    int max_possible = max_seq_len - n_input;
    if (max_possible <= 0) {
        atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
        atlas_vfree((uint8_t*)logits); atlas_vfree((uint8_t*)context); return 0;
    }
    if (max_new_tokens > max_possible) max_new_tokens = max_possible;
    for (int step = 1; step < max_new_tokens; step++) {
        // Embed last generated token
        int tid = next_token;
        if (tid < 0 || tid >= V) tid = 0;
        float* h = embed_buf;  // reuse embed_buf as single-token buffer
        for (int j = 0; j < H; j++)
            h[j] = fp16_to_fp32(embed_w[tid * H + j]);

        int seq_now = n_input + step;
        int pos = seq_now - 1;
        atlas_forward(m, h, 1, &pos,
                      max_seq_len, seq_now,
                      layer_idx, m->n_layers);

        atlas_rmsnorm_f32(h, norm_w, h_norm, H, 1e-6f);
        atlas_lmhead_gemv(m, h_norm, logits, 1);

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

    float* embed_buf = (float*)atlas_valloc((size_t)n_input * H * sizeof(float));
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

    // ─── Prefill ───
    int n_gen = 0;
    for (int i = 0; i < n_input; i++) {
        int tid = input_ids[i];
        if (tid < 0 || tid >= V) tid = 0;
        for (int j = 0; j < H; j++)
            embed_buf[i * H + j] = fp16_to_fp32(embed_w[tid * H + j]);
    }

    int* positions = (int*)atlas_valloc((size_t)n_input * sizeof(int));
    if (!positions) {
        atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
        atlas_vfree((uint8_t*)logits); atlas_vfree((uint8_t*)context); return -1;
    }
    for (int i = 0; i < n_input; i++) positions[i] = i;

    atlas_forward(m, embed_buf, n_input, positions,
                  max_seq_len, n_input,
                  layer_idx, m->n_layers);

    {
        const float* x = embed_buf + (int64_t)(n_input - 1) * H;
        atlas_rmsnorm_f32(x, norm_w, h_norm, H, 1e-6f);
        atlas_lmhead_gemv(m, h_norm, logits, 1);
    }

    int next_token = gumbel_sample(logits, V, temperature, top_k, top_p,
                                    context, n_input, repetition_penalty);
    context[n_input] = next_token;
    callback(next_token, user_data);
    n_gen++;

    const int eos_id = m->eos_id;
    if (next_token == eos_id) {
        atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
        atlas_vfree((uint8_t*)logits); atlas_vfree((uint8_t*)positions);
        atlas_vfree((uint8_t*)context);
        return n_gen;
    }

    // ─── Decode loop ───
    atlas_vfree((uint8_t*)positions);
    int max_possible = max_seq_len - n_input;
    if (max_possible <= 0) {
        atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
        atlas_vfree((uint8_t*)logits); atlas_vfree((uint8_t*)context); return 0;
    }
    if (max_new_tokens > max_possible) max_new_tokens = max_possible;
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
