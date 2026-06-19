// atlas_cli.cpp — Standalone CLI for ATLAS TQ1.0 inference engine
// Usage: atlas.exe <model.atlas> [prompt] [options]
//   If no prompt, reads from stdin (one-shot) or -i for interactive mode.
//
// Build: compile.bat builds both atlas.dll and atlas.exe
// Runtime: atlas.exe loads atlas.dll via LoadLibrary (no import lib needed)

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <shellapi.h>
#else
  #include <dlfcn.h>
  typedef void* HMODULE;
#endif
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <csignal>

#ifdef _WIN32
#include <shellapi.h>
#endif

// ─── ANSI colors ─────────────────────────────────────────────────────────
#define ANSI_RESET  "\033[0m"
#define ANSI_RED    "\033[31m"
#define ANSI_GREEN  "\033[32m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_CYAN   "\033[36m"
#define ANSI_MAGENTA "\033[35m"
#define ANSI_BOLD   "\033[1m"
#define ANSI_DIM    "\033[2m"

// ─── Signal handling ────────────────────────────────────────────────────
static volatile bool g_interrupted = false;

static void handle_sigint(int) {
    g_interrupted = true;
    printf(ANSI_RESET "\n");
}

// ─── FNV-1a hash for cache validation ────────────────────────────────────
static uint64_t fnv1a(const std::string& s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (char c : s) {
        h ^= (uint8_t)c;
        h *= 0x100000001b3ULL;
    }
    return h;
}

// ─── Timing helper ────────────────────────────────────────────────────
struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }
};

// ─── RAII: DLL lifecycle ──────────────────────────────────────────────
struct AtlasDLL {
    HMODULE handle;

    AtlasDLL() : handle(NULL) {}

    bool load(const char* path) {
#ifdef _WIN32
        handle = LoadLibraryA(path);
        if (!handle) {
            fprintf(stderr, ANSI_RED "[Error]" ANSI_RESET
                " Failed to load %s (error %lu)\n", path, GetLastError());
            return false;
        }
#else
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            fprintf(stderr, ANSI_RED "[Error]" ANSI_RESET
                " Failed to load %s: %s\n", path, dlerror());
            return false;
        }
#endif
        return true;
    }

    ~AtlasDLL() { unload(); }

    AtlasDLL(const AtlasDLL&) = delete;
    AtlasDLL& operator=(const AtlasDLL&) = delete;

    AtlasDLL(AtlasDLL&& other) noexcept : handle(other.handle) {
        other.handle = NULL;
    }
    AtlasDLL& operator=(AtlasDLL&& other) noexcept {
        if (this != &other) { unload(); handle = other.handle; other.handle = NULL; }
        return *this;
    }

    void unload() {
        if (handle) {
#ifdef _WIN32
            FreeLibrary(handle);
#else
            dlclose(handle);
#endif
            handle = NULL;
        }
    }

    void* sym(const char* name) const {
#ifdef _WIN32
        return (void*)GetProcAddress(handle, name);
#else
        return dlsym(handle, name);
#endif
    }
};

// ─── Function pointer typedefs ──────────────────────────────────────────
typedef void* (*PFN_atlas_load)(const char* path);
typedef void  (*PFN_atlas_free)(void* model);
typedef void  (*PFN_atlas_get_info)(void* model, int* n_layers, int* hidden_dim,
    int* inter_dim, int* n_heads, int* n_kv_heads,
    int* head_dim, int* vocab_size);
typedef void  (*PFN_atlas_set_seed)(uint64_t seed);
typedef void  (*PFN_atlas_set_num_threads)(void* model, int n);
typedef void  (*PFN_atlas_decompress_all)(void* model);
typedef void  (*PFN_atlas_decompress_ttype5)(void* model);
typedef int   (*PFN_atlas_load_cache)(void* model, const char* path);
typedef void  (*PFN_atlas_save_cache)(void* model, const char* path);
typedef void  (*PFN_atlas_prefetch_int8)(void* model);
typedef void  (*PFN_atlas_set_use_f32_matmul)(void* model, int val);
typedef void  (*PFN_atlas_set_use_hybrid_matmul)(void* model, int val);
typedef void  (*PFN_atlas_set_rope_scale)(void* model, float scale);
typedef void  (*PFN_atlas_set_base_seq_len)(void* model, int seq_len);
typedef void  (*PFN_atlas_reset_cache)(void* model);
typedef void  (*PFN_atlas_ensure_layer_idx)(void* model);
typedef int   (*PFN_atlas_get_tensor_index)(void* model, const char* name);
typedef void  (*PFN_atlas_quantize_lmhead)(void* model, int idx, int keep_data);
typedef int   (*PFN_atlas_tokenizer_preencode)(void* model,
    const char* text, int text_len, int* out_ids, int max_ids);
typedef int   (*PFN_atlas_tokenizer_merge)(void* model, int* ids, int* n_ids);
typedef int   (*PFN_atlas_tokenizer_decode)(void* model,
    const int* ids, int n_ids, char* out_text, int max_out);
typedef int   (*PFN_atlas_generate)(void* model,
    const int* input_ids, int n_input,
    int max_seq_len, int max_new_tokens,
    float temperature, int top_k, float top_p,
    float repetition_penalty,
    int min_new_tokens,
    int cache_offset,
    int* output_ids);
typedef void  (*PFN_atlas_quantize_ffn_to_i4)(void* model);
typedef int   (*PFN_atlas_has_binary_tokenizer)(void* model);

// ─── Dynamic DLL bindings ──────────────────────────────────────────────
static PFN_atlas_load              atlas_load;
static PFN_atlas_free              atlas_free;
static PFN_atlas_get_info          atlas_get_info;
static PFN_atlas_set_seed          atlas_set_seed;
static PFN_atlas_set_num_threads   atlas_set_num_threads;
static PFN_atlas_decompress_all    atlas_decompress_all;
static PFN_atlas_decompress_ttype5 atlas_decompress_ttype5;
static PFN_atlas_load_cache        atlas_load_cache;
static PFN_atlas_save_cache        atlas_save_cache;
static PFN_atlas_prefetch_int8     atlas_prefetch_int8;
static PFN_atlas_set_use_f32_matmul  atlas_set_use_f32_matmul;
static PFN_atlas_set_use_hybrid_matmul atlas_set_use_hybrid_matmul;
static PFN_atlas_set_rope_scale    atlas_set_rope_scale;
static PFN_atlas_set_base_seq_len  atlas_set_base_seq_len;
static PFN_atlas_reset_cache       atlas_reset_cache;
static PFN_atlas_ensure_layer_idx  atlas_ensure_layer_idx;
static PFN_atlas_get_tensor_index  atlas_get_tensor_index;
static PFN_atlas_quantize_lmhead   atlas_quantize_lmhead;
static PFN_atlas_quantize_ffn_to_i4 atlas_quantize_ffn_to_i4;
static PFN_atlas_tokenizer_preencode atlas_tokenizer_preencode;
static PFN_atlas_tokenizer_merge   atlas_tokenizer_merge;
static PFN_atlas_tokenizer_decode  atlas_tokenizer_decode;
static PFN_atlas_generate          atlas_generate;
static PFN_atlas_has_binary_tokenizer atlas_has_binary_tokenizer;

#define LOAD_OR_FAIL(name) do { \
    name = (PFN_##name)dll.sym(#name); \
    if (!name) { \
        fprintf(stderr, ANSI_RED "[Error]" ANSI_RESET " Missing symbol: %s\n", #name); \
        return false; \
    } \
} while(0)

// ─── RAII: Model lifecycle ────────────────────────────────────────────
struct AtlasSession {
    void* model;
    const char* path;

    AtlasSession(const char* p) : model(NULL), path(p) {}

    bool load() {
        model = atlas_load(path);
        if (!model) {
            fprintf(stderr, ANSI_RED "[Error]" ANSI_RESET
                " Failed to load model: %s\n", path);
            return false;
        }
        return true;
    }

    ~AtlasSession() { if (model) atlas_free(model); }

    AtlasSession(const AtlasSession&) = delete;
    AtlasSession& operator=(const AtlasSession&) = delete;

    AtlasSession(AtlasSession&& other) noexcept : model(other.model), path(other.path) {
        other.model = NULL;
    }
    AtlasSession& operator=(AtlasSession&& other) noexcept {
        if (this != &other) {
            if (model) atlas_free(model);
            model = other.model;
            path = other.path;
            other.model = NULL;
        }
        return *this;
    }

    void reset() { if (model) { atlas_free(model); model = NULL; } }

    explicit operator bool() const { return model != NULL; }
};

static bool bind_dll(AtlasDLL& dll) {
    LOAD_OR_FAIL(atlas_load);
    LOAD_OR_FAIL(atlas_free);
    LOAD_OR_FAIL(atlas_get_info);
    LOAD_OR_FAIL(atlas_set_seed);
    LOAD_OR_FAIL(atlas_set_num_threads);
    LOAD_OR_FAIL(atlas_decompress_all);
    LOAD_OR_FAIL(atlas_decompress_ttype5);
    LOAD_OR_FAIL(atlas_load_cache);
    LOAD_OR_FAIL(atlas_save_cache);
    LOAD_OR_FAIL(atlas_prefetch_int8);
    LOAD_OR_FAIL(atlas_set_use_f32_matmul);
    LOAD_OR_FAIL(atlas_set_use_hybrid_matmul);
    LOAD_OR_FAIL(atlas_set_rope_scale);
    LOAD_OR_FAIL(atlas_set_base_seq_len);
    LOAD_OR_FAIL(atlas_reset_cache);
    LOAD_OR_FAIL(atlas_ensure_layer_idx);
    LOAD_OR_FAIL(atlas_get_tensor_index);
    LOAD_OR_FAIL(atlas_quantize_lmhead);
    // Optional: FFN int4 quantization
    atlas_quantize_ffn_to_i4 = (PFN_atlas_quantize_ffn_to_i4)dll.sym("atlas_quantize_ffn_to_i4");
    LOAD_OR_FAIL(atlas_tokenizer_preencode);
    LOAD_OR_FAIL(atlas_tokenizer_merge);
    LOAD_OR_FAIL(atlas_tokenizer_decode);
    LOAD_OR_FAIL(atlas_generate);
    LOAD_OR_FAIL(atlas_has_binary_tokenizer);
    return true;
}

#undef LOAD_OR_FAIL

// ─── Windows UTF-8 helper ────────────────────────────────────────────
#ifdef _WIN32
static std::string wchar_to_utf8(const wchar_t* wstr) {
    if (!wstr) return "";
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
    if (len <= 0) return "";
    std::string result(len - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, &result[0], len, NULL, NULL);
    return result;
}
#endif

// ─── Args ───────────────────────────────────────────────────────────────
struct Config {
    std::string model_path;
    std::string prompt;
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
    int max_new_tokens = 200;
    int max_seq_len = 4096;
    float rep_penalty = 1.0f;
    int min_new_tokens = 20;
    uint64_t seed = 0;
    int num_threads = 0;
    bool interactive = false;
    bool raw = false;
    bool help = false;
    std::string log_path;
};

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            cfg.help = true;
        } else if (arg == "--temp" && i + 1 < argc) {
            cfg.temperature = (float)atof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            cfg.top_k = atoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            cfg.top_p = (float)atof(argv[++i]);
        } else if (arg == "--max-new" && i + 1 < argc) {
            cfg.max_new_tokens = atoi(argv[++i]);
        } else if (arg == "--max-seq" && i + 1 < argc) {
            cfg.max_seq_len = atoi(argv[++i]);
        } else if (arg == "--rep-penalty" && i + 1 < argc) {
            cfg.rep_penalty = (float)atof(argv[++i]);
        } else if (arg == "--min-new" && i + 1 < argc) {
            cfg.min_new_tokens = atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            cfg.seed = (uint64_t)atoll(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            cfg.num_threads = atoi(argv[++i]);
        } else if (arg == "-i") {
            cfg.interactive = true;
        } else if (arg == "--log" && i + 1 < argc) {
            cfg.log_path = argv[++i];
        } else if (arg == "--raw") {
            cfg.raw = true;
        } else if (arg[0] == '-') {
            fprintf(stderr, ANSI_YELLOW "[Warning]" ANSI_RESET " Unknown option: %s\n", arg.c_str());
        } else if (cfg.model_path.empty()) {
            cfg.model_path = arg;
        } else if (cfg.prompt.empty()) {
            cfg.prompt = arg;
        } else {
            cfg.prompt += " " + arg;
        }
    }
    return cfg;
}

static void print_usage() {
    printf(ANSI_BOLD "ATLAS" ANSI_RESET " — TQ1.0 Ternary Inference Engine (CPU)\n");
    printf("Run large language models on a laptop. No GPU needed.\n\n");
    printf(ANSI_BOLD "Usage:" ANSI_RESET "\n");
    printf("  atlas.exe " ANSI_CYAN "<model.atlas>" ANSI_RESET " " ANSI_GREEN "[prompt]" ANSI_RESET " [options]\n\n");
    printf(ANSI_BOLD "Examples:" ANSI_RESET "\n");
    printf("  atlas.exe falcon3-3b-tq1.atlas \"Hello, how are you?\"\n");
    printf("  atlas.exe model.atlas --temp 0.0 \"Capital of France?\"\n");
    printf("  atlas.exe model.atlas -i                          # interactive chat\n\n");
    printf(ANSI_BOLD "Options:" ANSI_RESET "\n");
    printf("  " ANSI_CYAN "--temp <f>" ANSI_RESET "      Temperature (default: 0.7, 0=deterministic)\n");
    printf("  " ANSI_CYAN "--top-k <n>" ANSI_RESET "     Top-k sampling (default: 40, 0=disabled)\n");
    printf("  " ANSI_CYAN "--top-p <f>" ANSI_RESET "     Top-p sampling (default: 0.9, 0=disabled)\n");
    printf("  " ANSI_CYAN "--max-new <n>" ANSI_RESET "   Max tokens to generate (default: 200)\n");
    printf("  " ANSI_CYAN "--max-seq <n>" ANSI_RESET "   KV cache window size (default: 4096)\n");
    printf("  " ANSI_CYAN "--rep-penalty <f>" ANSI_RESET "   Repetition penalty (default: 1.0)\n");
    printf("  " ANSI_CYAN "--min-new <n>" ANSI_RESET "   Min tokens before EOS allowed (default: 20)\n");
    printf("  " ANSI_CYAN "--seed <n>" ANSI_RESET "      RNG seed (default: random)\n");
    printf("  " ANSI_CYAN "--threads <n>" ANSI_RESET "   OpenMP threads (default: auto)\n");
    printf("  " ANSI_CYAN "--raw" ANSI_RESET "           Send prompt as-is (no chat template)\n");
    printf("  " ANSI_CYAN "-i" ANSI_RESET "              Interactive chat mode\n");
    printf("  " ANSI_CYAN "--log <file>" ANSI_RESET "    Log conversation to file\n");
    printf("  " ANSI_CYAN "-h, --help" ANSI_RESET "      Show this help\n");
}

// ─── Chat template ─────────────────────────────────────────────────────
struct ModelInfo {
    int n_layers, hidden, inter, n_heads, n_kv_heads, head_dim, vocab_size;
    float rope_theta;
    bool is_bitnet = false;
    bool is_cann = false;
    bool is_qwen3 = false;
    bool is_falcon3 = false;
};

static ModelInfo get_model_info(void* model) {
    ModelInfo info;
    atlas_get_info(model, &info.n_layers, &info.hidden, &info.inter,
                   &info.n_heads, &info.n_kv_heads, &info.head_dim, &info.vocab_size);
    info.is_bitnet = (atlas_get_tensor_index(model, "model.layers.0.self_attn.attn_sub_norm.weight") >= 0);
    info.is_cann = (info.vocab_size == 73448);
    info.is_qwen3 = (!info.is_bitnet && !info.is_cann && (info.head_dim <= 128 || info.vocab_size > 131072));
    info.is_falcon3 = (!info.is_bitnet && !info.is_cann && !info.is_qwen3);
    return info;
}

static std::string apply_chat_template(const ModelInfo& info, const std::string& user_text) {
    if (info.is_cann || info.is_qwen3) {
        return "<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n";
    }
    if (info.is_bitnet) {
        return "User: " + user_text + "<|eot_id|>\nAssistant: ";
    }
    return "<|user|>\n" + user_text + "\n<|assistant|>\n";
}

// ─── Tokenizer ─────────────────────────────────────────────────────────-
static std::vector<int> encode_text(void* model, const std::string& text) {
    int max_ids = (int)text.size() * 2 + 256;
    std::vector<int> ids(max_ids);
    int n = atlas_tokenizer_preencode(model, text.c_str(), (int)text.size(), ids.data(), max_ids);
    if (n < 0) {
        fprintf(stderr, ANSI_RED "[Error]" ANSI_RESET " Tokenizer preencode failed\n");
        return {};
    }
    ids.resize(n);
    if (atlas_tokenizer_merge(model, ids.data(), &n) != 0) {
        fprintf(stderr, ANSI_RED "[Error]" ANSI_RESET " Tokenizer merge failed\n");
        return {};
    }
    ids.resize(n);
    return ids;
}

static std::string decode_tokens(void* model, const std::vector<int>& ids) {
    if (ids.empty()) return "";
    int max_out = (int)ids.size() * 16 + 64;
    std::string out(max_out, '\0');
    int n = atlas_tokenizer_decode(model, ids.data(), (int)ids.size(), &out[0], max_out);
    if (n < 0) {
        fprintf(stderr, ANSI_RED "[Error]" ANSI_RESET " Tokenizer decode failed\n");
        return "[decode error]";
    }
    out.resize(n);
    return out;
}

// ─── Model setup ────────────────────────────────────────────────────────
static bool setup_model(void* model, const Config& cfg, const ModelInfo& info) {
    printf(ANSI_CYAN "[Atlas]" ANSI_RESET " %dL %dH %dI %d/%d heads | vocab=%d\n",
           info.n_layers, info.hidden, info.inter,
           info.n_heads, info.n_kv_heads, info.vocab_size);

    Timer t;

    int cache_loaded = atlas_load_cache(model, cfg.model_path.c_str());
    if (cache_loaded) {
        printf(ANSI_CYAN "[Atlas]" ANSI_RESET " int8 cache loaded (" ANSI_GREEN "mmap" ANSI_RESET ")\n");
    } else {
        printf(ANSI_CYAN "[Atlas]" ANSI_RESET " Decompressing TQ1 weights to int8...\n");
        atlas_decompress_all(model);
        atlas_decompress_ttype5(model);
        atlas_save_cache(model, cfg.model_path.c_str());
        printf(ANSI_CYAN "[Atlas]" ANSI_RESET " Cache saved (%.1fs)\n", t.elapsed());
    }

    t = Timer();
    atlas_decompress_all(model);
    atlas_decompress_ttype5(model);

    // f32_bypass: for small, block-scaled, or BitNet models
    if (info.is_bitnet || info.hidden <= 2048) {
        atlas_set_use_f32_matmul(model, 1);
        printf(ANSI_CYAN "[Atlas]" ANSI_RESET " f32 bypass\n");
    }

    if (info.is_qwen3) {
        atlas_set_rope_scale(model, 4.0f);
        printf(ANSI_CYAN "[Atlas]" ANSI_RESET " YaRN RoPE scale=4.0\n");
    }

    if (atlas_quantize_ffn_to_i4) {
        atlas_quantize_ffn_to_i4(model);
        printf(ANSI_CYAN "[Atlas]" ANSI_RESET " FFN int4 quantized\n");
    }

    printf(ANSI_CYAN "[Atlas]" ANSI_RESET " Prefetching weights into RAM...\n");
    atlas_prefetch_int8(model);

    atlas_set_base_seq_len(model, cfg.max_seq_len);
    atlas_ensure_layer_idx(model);

    int idx = atlas_get_tensor_index(model, "lm_head.weight");
    if (idx >= 0) {
        printf(ANSI_CYAN "[Atlas]" ANSI_RESET " Quantizing lm_head...\n");
        atlas_quantize_lmhead(model, idx, 0);
    } else {
        idx = atlas_get_tensor_index(model, "model.embed_tokens.weight");
        if (idx >= 0) {
            printf(ANSI_CYAN "[Atlas]" ANSI_RESET " Quantizing lm_head (tied embeddings)...\n");
            atlas_quantize_lmhead(model, idx, 1);
        }
    }

    if (cfg.num_threads > 0) {
        atlas_set_num_threads(model, cfg.num_threads);
        printf(ANSI_CYAN "[Atlas]" ANSI_RESET " Threads: %d\n", cfg.num_threads);
    }

    if (cfg.seed != 0) {
        atlas_set_seed(cfg.seed);
        printf(ANSI_CYAN "[Atlas]" ANSI_RESET " Seed: %llu\n", (unsigned long long)cfg.seed);
    }

    printf(ANSI_CYAN "[Atlas]" ANSI_RESET " Model ready (%.1fs total)\n", t.elapsed());
    return true;
}

// ─── Generation ─────────────────────────────────────────────────────────
static std::string generate(void* model, const std::vector<int>& input_ids,
                             const Config& cfg, int cache_offset = 0)
{
    if (input_ids.empty()) return "";

    std::vector<int> output_ids(cfg.max_new_tokens);
    int n_gen = atlas_generate(
        model, input_ids.data(), (int)input_ids.size(),
        cfg.max_seq_len, cfg.max_new_tokens,
        cfg.temperature, cfg.top_k, cfg.top_p,
        cfg.rep_penalty,
        cfg.min_new_tokens,
        cache_offset,
        output_ids.data());

    if (n_gen < 0) {
        fprintf(stderr, ANSI_RED "[Error]" ANSI_RESET " Generation failed\n");
        return "";
    }

    output_ids.resize(n_gen);
    return decode_tokens(model, output_ids);
}

// ─── Session logging ─────────────────────────────────────────────────────
static FILE* g_log = NULL;

static bool open_log(const std::string& path) {
    if (path.empty()) return true;
    g_log = fopen(path.c_str(), "a");
    if (!g_log) {
        fprintf(stderr, ANSI_YELLOW "[Warning]" ANSI_RESET " Could not open log: %s\n", path.c_str());
        return false;
    }
    // UTF-8 BOM for Windows Notepad compatibility
    fwrite("\xEF\xBB\xBF", 1, 3, g_log);
    time_t now = time(NULL);
    struct tm* tm = localtime(&now);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
    fprintf(g_log, "=== ATLAS Session %s ===\n\n", buf);
    fflush(g_log);
    return true;
}

static void close_log() {
    if (g_log) {
        fprintf(g_log, "\n");
        fclose(g_log);
        g_log = NULL;
    }
}

static void log_turn(const char* prefix, const std::string& text) {
    if (!g_log) return;
    fprintf(g_log, "[%s] %s\n", prefix, text.c_str());
    fflush(g_log);
}

// ─── Interactive loop ───────────────────────────────────────────────────
struct ChatState {
    std::vector<int> all_token_ids;
    std::string history_text;
    uint64_t history_hash = 0;
    int total_input = 0;
};

static void interactive_loop(void* model, const ModelInfo& info, const Config& cfg) {
    signal(SIGINT, handle_sigint);

    printf("\n" ANSI_BOLD "Interactive mode." ANSI_RESET
           " Type " ANSI_YELLOW "/exit" ANSI_RESET " to quit, "
           ANSI_YELLOW "/reset" ANSI_RESET " to clear context.\n\n");

    ChatState chat;
    std::string line;

    while (!g_interrupted) {
        printf(ANSI_BOLD "[YOU]" ANSI_RESET " ");
        fflush(stdout);

        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        if (line == "/exit" || line == "/quit") {
            log_turn("YOU", line);
            break;
        }
        if (line == "/reset") {
            atlas_reset_cache(model);
            chat = ChatState();
            printf(ANSI_YELLOW "[Context cleared]" ANSI_RESET "\n");
            continue;
        }

        std::string full_prompt;
        if (cfg.raw) {
            full_prompt = line;
        } else {
            full_prompt = apply_chat_template(info, line);
        }

        // Cache validity check: hash of history should match
        if (!chat.history_text.empty() && chat.history_hash != 0) {
            uint64_t current_hash = fnv1a(chat.history_text);
            if (current_hash != chat.history_hash) {
                printf(ANSI_YELLOW "[Warning]" ANSI_RESET
                       " History text changed since last generation"
                       " (cache may be stale). Type " ANSI_YELLOW "/reset" ANSI_RESET " if output is garbled.\n");
            }
        }

        std::vector<int> input_ids = encode_text(model, full_prompt);
        if (input_ids.empty()) continue;

        int cache_offset = 0;
        if (chat.history_text.empty()) {
            chat.history_text = full_prompt;
        } else {
            cache_offset = chat.total_input;
        }
        chat.total_input = (int)input_ids.size();

        log_turn("YOU", line);

        Timer t;
        printf(ANSI_GREEN "[ATLAS]" ANSI_RESET " ");
        fflush(stdout);
        std::string response = generate(model, input_ids, cfg, cache_offset);

        if (!response.empty()) {
            printf("%s", response.c_str());
            log_turn("ATLAS", response);
            chat.history_text += response;
            chat.history_hash = fnv1a(chat.history_text);
        }

        printf(ANSI_DIM "\n     ─── %.1fs ───" ANSI_RESET "\n\n", t.elapsed());
    }

    signal(SIGINT, SIG_DFL);
    printf(ANSI_RESET "\n");
}

// ─── Main ───────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    signal(SIGINT, handle_sigint);

#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    int argc_w = 0;
    LPWSTR* argv_w = CommandLineToArgvW(GetCommandLineW(), &argc_w);
    std::vector<std::string> utf8_args;
    std::vector<char*> utf8_ptrs;
    if (argv_w && argc_w > 0) {
        for (int i = 0; i < argc_w; i++)
            utf8_args.push_back(wchar_to_utf8(argv_w[i]));
        for (auto& s : utf8_args)
            utf8_ptrs.push_back(&s[0]);
        argc = argc_w;
        argv = utf8_ptrs.data();
    }
    LocalFree(argv_w);
#else
    (void)argc;
    (void)argv;
#endif

    Config cfg = parse_args(argc, argv);

    if (cfg.help || cfg.model_path.empty()) {
        print_usage();
        return cfg.help ? 0 : 1;
    }

    // ─── Input validation ────────────────────────────────────────────────
    if (cfg.temperature < 0.0f || !std::isfinite(cfg.temperature)) {
        fprintf(stderr, ANSI_RED "[Error]" ANSI_RESET
            " --temp must be a finite value >= 0 (got %g)\n", cfg.temperature);
        return 1;
    }
    if (cfg.max_new_tokens <= 0 || cfg.max_new_tokens > 100000) {
        fprintf(stderr, ANSI_RED "[Error]" ANSI_RESET
            " --max-new must be 1..100000 (got %d)\n", cfg.max_new_tokens);
        return 1;
    }
    if (cfg.max_seq_len < 64 || cfg.max_seq_len > 262144) {
        fprintf(stderr, ANSI_RED "[Error]" ANSI_RESET
            " --max-seq must be 64..262144 (got %d)\n", cfg.max_seq_len);
        return 1;
    }
    if (cfg.top_p < 0.0f || cfg.top_p > 1.0f) {
        fprintf(stderr, ANSI_RED "[Error]" ANSI_RESET
            " --top-p must be 0..1 (got %g)\n", cfg.top_p);
        return 1;
    }
    if (cfg.rep_penalty < 0.0f || !std::isfinite(cfg.rep_penalty)) {
        fprintf(stderr, ANSI_RED "[Error]" ANSI_RESET
            " --rep-penalty must be a finite value >= 0 (got %g)\n", cfg.rep_penalty);
        return 1;
    }
    if (cfg.num_threads < 0) {
        fprintf(stderr, ANSI_RED "[Error]" ANSI_RESET
            " --threads must be >= 0 (got %d)\n", cfg.num_threads);
        return 1;
    }

    const char* env_dll = getenv("ATLAS_DLL");
    std::string dll_path = env_dll ? env_dll : "atlas.dll";

    // RAII: DLL and model both cleaned up on scope exit — no manual free needed
    AtlasDLL dll;
    if (!dll.load(dll_path.c_str())) {
        fprintf(stderr, "  Make sure atlas.dll is in the same directory as atlas.exe\n");
        fprintf(stderr, "  or set " ANSI_YELLOW "ATLAS_DLL" ANSI_RESET " environment variable.\n");
        return 1;
    }

    if (!bind_dll(dll)) return 1;

    AtlasSession session(cfg.model_path.c_str());
    if (!session.load()) return 1;
    open_log(cfg.log_path);

    ModelInfo info = get_model_info(session.model);
    if (!setup_model(session.model, cfg, info)) return 1;

    if (g_interrupted) goto cleanup;

    if (cfg.interactive) {
        interactive_loop(session.model, info, cfg);
    } else if (!cfg.prompt.empty()) {
        std::string full_text = cfg.raw ? cfg.prompt : apply_chat_template(info, cfg.prompt);
        std::vector<int> input_ids = encode_text(session.model, full_text);
        if (input_ids.empty()) goto cleanup;

        log_turn("YOU", cfg.prompt);

        Timer t;
        printf(ANSI_GREEN "[ATLAS]" ANSI_RESET " ");
        fflush(stdout);
        std::string response = generate(session.model, input_ids, cfg, 0);
        printf("%s", response.c_str());
        log_turn("ATLAS", response);
        printf(ANSI_DIM "\n     ─── %.1fs ───" ANSI_RESET "\n", t.elapsed());
    } else {
        std::string stdin_text;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!stdin_text.empty()) stdin_text += "\n";
            stdin_text += line;
        }
        if (stdin_text.empty()) {
            fprintf(stderr, ANSI_RED "[Error]" ANSI_RESET " No prompt provided. Use -h for help.\n");
            goto cleanup;
        }

        std::string full_text = cfg.raw ? stdin_text : apply_chat_template(info, stdin_text);
        std::vector<int> input_ids = encode_text(session.model, full_text);
        if (input_ids.empty()) goto cleanup;

        log_turn("YOU", stdin_text);

        Timer t;
        printf(ANSI_GREEN "[ATLAS]" ANSI_RESET " ");
        fflush(stdout);
        std::string response = generate(session.model, input_ids, cfg, 0);
        printf("%s", response.c_str());
        log_turn("ATLAS", response);
        printf(ANSI_DIM "\n     ─── %.1fs ───" ANSI_RESET "\n", t.elapsed());
    }

cleanup:
    close_log();
    printf(ANSI_RESET);
    return g_interrupted ? 130 : 0;
}
