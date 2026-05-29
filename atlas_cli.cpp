// atlas_cli.cpp — Standalone CLI for ATLAS TQ1.0 inference engine
// Usage: atlas.exe <model.atlas> [prompt] [options]
//   If no prompt, reads from stdin (one-shot) or -i for interactive mode.
//
// Build: compile.bat builds both atlas.dll and atlas.exe
// Runtime: atlas.exe loads atlas.dll via LoadLibrary (no import lib needed)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

// ─── Function pointer typedefs ──────────────────────────────────────────
// Mirrors atlas_ffi.h signatures. Loaded via GetProcAddress at runtime.

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
static HMODULE g_dll = NULL;
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

static bool load_dll(const char* dll_path) {
    g_dll = LoadLibraryA(dll_path);
    if (!g_dll) {
        fprintf(stderr, "[Error] Failed to load %s (error %lu)\n", dll_path, GetLastError());
        return false;
    }
#define LOAD(name) do { \
    name = (PFN_##name)GetProcAddress(g_dll, #name); \
    if (!name) { fprintf(stderr, "[Error] Missing symbol: %s\n", #name); return false; } \
} while(0)
    LOAD(atlas_load);
    LOAD(atlas_free);
    LOAD(atlas_get_info);
    LOAD(atlas_set_seed);
    LOAD(atlas_set_num_threads);
    LOAD(atlas_decompress_all);
    LOAD(atlas_decompress_ttype5);
    LOAD(atlas_load_cache);
    LOAD(atlas_save_cache);
    LOAD(atlas_prefetch_int8);
    LOAD(atlas_set_use_f32_matmul);
    LOAD(atlas_set_use_hybrid_matmul);
    LOAD(atlas_set_rope_scale);
    LOAD(atlas_set_base_seq_len);
    LOAD(atlas_reset_cache);
    LOAD(atlas_ensure_layer_idx);
    LOAD(atlas_get_tensor_index);
    LOAD(atlas_quantize_lmhead);
    // Optional: FFN int4 quantization
    atlas_quantize_ffn_to_i4 = (PFN_atlas_quantize_ffn_to_i4)GetProcAddress(g_dll, "atlas_quantize_ffn_to_i4");
    LOAD(atlas_tokenizer_preencode);
    LOAD(atlas_tokenizer_merge);
    LOAD(atlas_tokenizer_decode);
    LOAD(atlas_generate);
    LOAD(atlas_has_binary_tokenizer);
#undef LOAD
    return true;
}

static void unload_dll() {
    if (g_dll) { FreeLibrary(g_dll); g_dll = NULL; }
}

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
        } else if (arg == "--raw") {
            cfg.raw = true;
        } else if (arg[0] == '-') {
            fprintf(stderr, "[Warning] Unknown option: %s\n", arg.c_str());
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
    printf("ATLAS — TQ1.0 Ternary Inference Engine (CPU)\n");
    printf("Run large language models on a laptop. No GPU needed.\n\n");
    printf("Usage:\n");
    printf("  atlas.exe <model.atlas> [prompt] [options]\n\n");
    printf("Examples:\n");
    printf("  atlas.exe falcon3-3b-tq1.atlas \"Hello, how are you?\"\n");
    printf("  atlas.exe model.atlas --temp 0.0 \"Capital of France?\"\n");
    printf("  atlas.exe model.atlas -i                          # interactive chat\n\n");
    printf("Options:\n");
    printf("  --temp <f>      Temperature (default: 0.7, 0=deterministic)\n");
    printf("  --top-k <n>     Top-k sampling (default: 40, 0=disabled)\n");
    printf("  --top-p <f>     Top-p sampling (default: 0.9, 0=disabled)\n");
    printf("  --max-new <n>   Max tokens to generate (default: 200)\n");
    printf("  --max-seq <n>   KV cache window size (default: 4096)\n");
    printf("  --rep-penalty <f>   Repetition penalty (default: 1.0)\n");
    printf("  --min-new <n>   Min tokens before EOS allowed (default: 20)\n");
    printf("  --seed <n>      RNG seed (default: random)\n");
    printf("  --threads <n>   OpenMP threads (default: auto)\n");
    printf("  --raw           Send prompt as-is (no chat template)\n");
    printf("  -i              Interactive chat mode\n");
    printf("  -h, --help      Show this help\n");
}

// ─── Chat template ─────────────────────────────────────────────────────
// Detects model family and wraps user message in the correct format.
// Mirrors atlas_infer.py _apply_chat_template logic.

struct ModelInfo {
    int n_layers, hidden, inter, n_heads, n_kv_heads, head_dim, vocab_size;
    float rope_theta;
    bool is_bitnet = false;
    bool is_qwen3 = false;
    bool is_falcon3 = false;
};

static ModelInfo get_model_info(void* model) {
    ModelInfo info;
    atlas_get_info(model, &info.n_layers, &info.hidden, &info.inter,
                   &info.n_heads, &info.n_kv_heads, &info.head_dim, &info.vocab_size);
    // Read rope_theta from file header
    // We'll approximate by checking tensor names
    info.is_bitnet = (atlas_get_tensor_index(model, "model.layers.0.self_attn.attn_sub_norm.weight") >= 0);
    info.is_qwen3 = (!info.is_bitnet && (info.head_dim <= 128 || info.vocab_size > 131072));
    info.is_falcon3 = (!info.is_bitnet && !info.is_qwen3);
    return info;
}

static std::string apply_chat_template(const ModelInfo& info, const std::string& user_text) {
    if (info.is_qwen3) {
        return "<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n";
    }
    if (info.is_bitnet) {
        return "User: " + user_text + "<|eot_id|>\nAssistant: ";
    }
    // Falcon3 / default
    return "<|user|>\n" + user_text + "\n<|assistant|>\n";
}

// ─── Tokenizer ─────────────────────────────────────────────────────────-
// Uses C++ preencode + BPE merge from atlas.dll (v6 binary tokenizer).

static std::vector<int> encode_text(void* model, const std::string& text) {
    // Pre-encode: text → byte token IDs
    int max_ids = (int)text.size() * 2 + 256;
    std::vector<int> ids(max_ids);
    int n = atlas_tokenizer_preencode(model, text.c_str(), (int)text.size(), ids.data(), max_ids);
    if (n < 0) {
        fprintf(stderr, "[Error] Tokenizer preencode failed\n");
        return {};
    }
    ids.resize(n);

    // BPE merge loop
    if (atlas_tokenizer_merge(model, ids.data(), &n) != 0) {
        fprintf(stderr, "[Error] Tokenizer merge failed\n");
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
        fprintf(stderr, "[Error] Tokenizer decode failed\n");
        return "[decode error]";
    }
    out.resize(n);
    return out;
}

// ─── Model setup ────────────────────────────────────────────────────────
static bool setup_model(void* model, const Config& cfg, const ModelInfo& info) {
    printf("[Atlas] %dL %dH %dI %d/%d heads | vocab=%d\n",
           info.n_layers, info.hidden, info.inter,
           info.n_heads, info.n_kv_heads, info.vocab_size);

    // Try loading int8 cache; if not found, decompress + save
    int cache_loaded = atlas_load_cache(model, cfg.model_path.c_str());
    if (cache_loaded) {
        printf("[Atlas] Loaded int8 weights from cache (mmap)\n");
    } else {
        printf("[Atlas] Decompressing TQ1 weights to int8...\n");
        atlas_decompress_all(model);
        atlas_decompress_ttype5(model);
        atlas_save_cache(model, cfg.model_path.c_str());
        printf("[Atlas] Cache saved\n");
    }

    // Decompress again even on cache load (FFN tensors not in cache)
    atlas_decompress_all(model);
    atlas_decompress_ttype5(model);

    // f32_bypass: for small, block-scaled, or BitNet models
    if (info.is_bitnet || info.hidden <= 2048) {
        atlas_set_use_f32_matmul(model, 1);
        printf("[Atlas] f32 bypass enabled\n");
    }

    // Qwen3/Bonsai: YaRN RoPE scaling
    if (info.is_qwen3) {
        atlas_set_rope_scale(model, 4.0f);
    }

    // Optional: quantize FFN int8→int4 (v2.8.0, 18-26% faster)
    if (atlas_quantize_ffn_to_i4) {
        atlas_quantize_ffn_to_i4(model);
        printf("[Atlas] FFN quantized to int4\n");
    }

    // Prefetch int8 data into physical RAM
    atlas_prefetch_int8(model);

    // Set base seq len for NTK context extension
    atlas_set_base_seq_len(model, cfg.max_seq_len);

    // Ensure layer indices are built
    atlas_ensure_layer_idx(model);

    // Quantize lm_head to int8 (saves ~1 GB per 131k vocab)
    int idx = atlas_get_tensor_index(model, "lm_head.weight");
    if (idx >= 0) {
        printf("[Atlas] Quantizing lm_head...\n");
        atlas_quantize_lmhead(model, idx, 0);
    } else {
        idx = atlas_get_tensor_index(model, "model.embed_tokens.weight");
        if (idx >= 0) {
            printf("[Atlas] Quantizing lm_head (tied embeddings)...\n");
            atlas_quantize_lmhead(model, idx, 1);
        }
    }

    // Set thread count
    if (cfg.num_threads > 0) {
        atlas_set_num_threads(model, cfg.num_threads);
        printf("[Atlas] Threads: %d\n", cfg.num_threads);
    }

    // Seed RNG
    if (cfg.seed != 0) {
        atlas_set_seed(cfg.seed);
    }

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
        fprintf(stderr, "[Error] Generation failed\n");
        return "";
    }

    output_ids.resize(n_gen);
    return decode_tokens(model, output_ids);
}

// ─── Interactive loop ───────────────────────────────────────────────────
static void interactive_loop(void* model, const ModelInfo& info, const Config& cfg) {
    printf("\nInteractive mode. Type your message, or /exit to quit.\n");
    printf("Model: %s\n\n", cfg.model_path.c_str());

    std::vector<int> history_ids;  // All tokens ever generated (for chat template)
    std::string history_text;      // Full chat template text
    int total_input = 0;

    std::string line;
    while (true) {
        printf(">>> ");
        fflush(stdout);

        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        if (line == "/exit" || line == "/quit") break;
        if (line == "/reset") {
            atlas_reset_cache(model);
            history_ids.clear();
            history_text.clear();
            total_input = 0;
            printf("[Cache reset]\n");
            continue;
        }

        // Build full chat template
        std::string full_prompt;
        if (cfg.raw) {
            full_prompt = line;
        } else {
            // If we have history, reconstruct from text
            if (history_text.empty()) {
                full_prompt = apply_chat_template(info, line);
            } else {
                // Strip the final "<|assistant|>\n" from previous, add user + assistant
                // Simple: rebuild from history text + new user message
                full_prompt = apply_chat_template(info, line);
            }
        }

        // Encode full prompt
        std::vector<int> input_ids = encode_text(model, full_prompt);
        if (input_ids.empty()) continue;

        // Determine cache offset: if history matches, skip prefill for first part
        int cache_offset = 0;
        if (history_text.empty()) {
            // First turn: no cache
            history_text = full_prompt;
            total_input = (int)input_ids.size();
        } else {
            // Subsequent turns: use cache_offset to skip re-prefilling
            // The cache already has previous turns' KV data
            // We send the full input but offset by the number of previously processed tokens
            // For simplicity, just use total_input as offset
            cache_offset = total_input;
            total_input = (int)input_ids.size();
        }

        // Generate
        std::string response = generate(model, input_ids, cfg, cache_offset);
        printf("%s\n\n", response.c_str());

        // Update history
        if (!response.empty()) {
            history_text += response;
        }
    }
}

// ─── Main ───────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    Config cfg = parse_args(argc, argv);

    if (cfg.help || cfg.model_path.empty()) {
        print_usage();
        return cfg.help ? 0 : 1;
    }

    // Determine DLL path (next to EXE, or ATLAS_DLL env var)
    const char* env_dll = getenv("ATLAS_DLL");
    std::string dll_path = env_dll ? env_dll : "atlas.dll";

    if (!load_dll(dll_path.c_str())) {
        fprintf(stderr, "[Error] Could not load atlas.dll.\n");
        fprintf(stderr, "  Make sure atlas.dll is in the same directory as atlas.exe\n");
        fprintf(stderr, "  or set ATLAS_DLL environment variable.\n");
        return 1;
    }

    // Load model
    void* model = atlas_load(cfg.model_path.c_str());
    if (!model) {
        fprintf(stderr, "[Error] Failed to load model: %s\n", cfg.model_path.c_str());
        unload_dll();
        return 1;
    }

    // Get model info + setup
    ModelInfo info = get_model_info(model);
    if (!setup_model(model, cfg, info)) {
        atlas_free(model);
        unload_dll();
        return 1;
    }

    if (cfg.interactive) {
        interactive_loop(model, info, cfg);
    } else if (!cfg.prompt.empty()) {
        // One-shot generation
        std::string full_text;
        if (cfg.raw) {
            full_text = cfg.prompt;
        } else {
            full_text = apply_chat_template(info, cfg.prompt);
        }

        std::vector<int> input_ids = encode_text(model, full_text);
        if (input_ids.empty()) {
            atlas_free(model);
            unload_dll();
            return 1;
        }

        printf("[Atlas] Generating...\n");
        std::string response = generate(model, input_ids, cfg, 0);
        printf("%s\n", response.c_str());
    } else {
        // Read prompt from stdin
        std::string stdin_text;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!stdin_text.empty()) stdin_text += "\n";
            stdin_text += line;
        }
        if (stdin_text.empty()) {
            fprintf(stderr, "[Error] No prompt provided. Use -h for help.\n");
            atlas_free(model);
            unload_dll();
            return 1;
        }

        std::string full_text = cfg.raw ? stdin_text : apply_chat_template(info, stdin_text);
        std::vector<int> input_ids = encode_text(model, full_text);
        if (input_ids.empty()) {
            atlas_free(model);
            unload_dll();
            return 1;
        }

        printf("[Atlas] Generating...\n");
        std::string response = generate(model, input_ids, cfg, 0);
        printf("%s\n", response.c_str());
    }

    atlas_free(model);
    unload_dll();
    return 0;
}
