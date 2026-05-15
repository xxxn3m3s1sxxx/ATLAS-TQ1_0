#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#define WDIR "C:\\dam\\atlas\\falcon3_tq10\\"
#define HIDDEN 3072
#define N_HEADS 12
#define N_KV_HEADS 4
#define HEAD_DIM 256
#define NH_HD (N_HEADS*HEAD_DIM)
#define NKV_HD (N_KV_HEADS*HEAD_DIM)
#define INTERMEDIATE 23040
#define EPS 1e-6
#define CLAMP_HIDDEN 512.0f
#define MAX_SEQ 4096
#define VOCAB 131072
#define N_LAYERS 40
#define BOS_ID 11
#define EOS_ID 11
#define ROPE_THETA 1000042.0f
#define ROPE_HALF (HEAD_DIM/2)

#define TEMPERATURE 0.85f

static int sample_token(float* logits, int n, float temp) {
    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    float max_l = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max_l) max_l = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        logits[i] = expf((logits[i] - max_l) / temp);
        sum += logits[i];
    }
    float r = (float)rand() / (float)RAND_MAX * sum;
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        cum += logits[i];
        if (cum >= r) return i;
    }
    return n - 1;
}

static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static float cos_t[MAX_SEQ][HEAD_DIM];
static float sin_t[MAX_SEQ][HEAD_DIM];

static void init_rope() {
    for (int pos = 0; pos < MAX_SEQ; pos++) {
        for (int i = 0; i < ROPE_HALF; i++) {
            double inv = 1.0 / pow(ROPE_THETA, (2.0 * i) / HEAD_DIM);
            double v = pos * inv;
            float c = (float)cos(v);
            float s = (float)sin(v);
            cos_t[pos][i] = c;
            cos_t[pos][i + ROPE_HALF] = c;
            sin_t[pos][i] = s;
            sin_t[pos][i + ROPE_HALF] = s;
        }
    }
}

static void apply_rope(float* v, int n_heads, int pos) {
    for (int h = 0; h < n_heads; h++) {
        float* hv = &v[h * HEAD_DIM];
        const float* c = cos_t[pos];
        const float* s = sin_t[pos];
        for (int i = 0; i < ROPE_HALF; i++) {
            float a = hv[i], b = hv[i + ROPE_HALF];
            hv[i] = a * c[i] - b * s[i];
            hv[i + ROPE_HALF] = a * s[i] + b * c[i];
        }
    }
}

static float* load_vec(const char* name, int* n) {
    char p[512]; snprintf(p, sizeof(p), "%s%s", WDIR, name);
    FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "ERROR: %s\n", p); exit(1); }
    int r; fread(&r, 4, 1, f); fread(n, 4, 1, f);
    float* d = (float*)_aligned_malloc((*n) * 4, 32);
    fread(d, 4, *n, f); fclose(f);
    return d;
}

static float* load_full(const char* name, long* n) {
    char p[512]; snprintf(p, sizeof(p), "%s%s", WDIR, name);
    FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "ERROR: %s\n", p); exit(1); }
    int r, c; fread(&r, 4, 1, f); fread(&c, 4, 1, f);
    *n = (long)r * c;
    float* d = (float*)_aligned_malloc((*n) * 4, 32);
    fread(d, 4, *n, f); fclose(f);
    return d;
}

typedef struct {
    uint8_t* data;
    float* scales;
    int rows, cols, group_size, num_groups;
    int packed_cols;
    int* group_idx;
} TQ1Mat;

static TQ1Mat load_tq10(const char* name) {
    char p[512]; snprintf(p, sizeof(p), "%s%s.tq10", WDIR, name);
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "ERROR: %s\n", p); exit(1); }
    TQ1Mat m;
    fread(&m.rows, 4, 1, f);
    fread(&m.cols, 4, 1, f);
    fread(&m.group_size, 4, 1, f);
    fread(&m.num_groups, 4, 1, f);
    m.scales = (float*)_aligned_malloc(m.num_groups * 4, 32);
    fread(m.scales, 4, m.num_groups, f);
    m.packed_cols = (m.cols + 3) / 4;
    m.data = (uint8_t*)_aligned_malloc(m.rows * m.packed_cols, 64);
    fread(m.data, 1, m.rows * m.packed_cols, f);
    fclose(f);
    m.group_idx = (int*)_aligned_malloc(m.rows * 4, 32);
    for (int i = 0; i < m.rows; i++) {
        int gi = i / m.group_size;
        if (gi >= m.num_groups) gi = m.num_groups - 1;
        m.group_idx[i] = gi;
    }
    return m;
}

static void mv_tq10_scales_omp(const TQ1Mat* m, const int8_t* a, float* y) {
    int rows = m->rows, cols = m->cols, pc = m->packed_cols;
    const float* scales = m->scales;
    const int* gidx = m->group_idx;
    for (int r = 0; r < rows; r++) {
        int32_t sum = 0;
        const uint8_t* row_data = &m->data[r * pc];
        for (int i = 0; i < cols; i++) {
            int bi = i / 4, sh = (i % 4) * 2;
            int val = (row_data[bi] >> sh) & 3;
            sum += (int32_t)a[i] * ((val & 1) - (val >> 1));
        }
        y[r] = (float)sum * scales[gidx[r]];
    }
}

static void mv_fp32_omp(const float* w, const float* x, float* y, int rows, int cols) {
    #pragma omp for schedule(static)
    for (int r = 0; r < rows; r++) {
        __m256 sum = _mm256_setzero_ps();
        for (int i = 0; i < cols; i += 8)
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(&w[r*cols+i]), _mm256_loadu_ps(&x[i])));
        y[r] = hsum_ps(sum);
    }
}

static float quantize_fast(const float* s, int8_t* d, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(s[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) max_abs = 1e-10f;
    float inv = 127.0f / max_abs;
    for (int i = 0; i < n; i++)
        d[i] = (int8_t)(roundf(s[i] * inv));
    return max_abs / 127.0f;
}

static void rms_norm(float* x, const float* w, int n) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    double inv = 1.0 / sqrt(ss / n + EPS);
    for (int i = 0; i < n; i++) x[i] = (float)((double)x[i] * inv * (double)w[i]);
}

static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }

static void attention_gqa(float* attn_out, const float* q,
                          const float* k_cache, const float* v_cache,
                          int seq_len, int max_s, float* temp) {
    int g = N_HEADS / N_KV_HEADS;
    for (int kv = 0; kv < N_KV_HEADS; kv++) {
        float* kv_temp = &temp[kv * g * MAX_SEQ];
        for (int qh = 0; qh < g; qh++) {
            const float* qv = &q[(kv * g + qh) * HEAD_DIM];
            float* sc = &kv_temp[qh * seq_len];
            for (int s = 0; s < seq_len; s++) {
                __m256 sum = _mm256_setzero_ps();
                const float* kv_vec = &k_cache[(kv * max_s + s) * HEAD_DIM];
                for (int i = 0; i < HEAD_DIM; i += 8)
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(&qv[i]), _mm256_loadu_ps(&kv_vec[i])));
                sc[s] = hsum_ps(sum) / sqrtf(HEAD_DIM);
            }
        }
        for (int qh = 0; qh < g; qh++) {
            float* sc = &kv_temp[qh * seq_len];
            float mx = sc[0]; for (int s = 1; s < seq_len; s++) if (sc[s] > mx) mx = sc[s];
            float sum = 0; for (int s = 0; s < seq_len; s++) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
            float inv_sum = 1.0f / sum; for (int s = 0; s < seq_len; s++) sc[s] *= inv_sum;
        }
        for (int qh = 0; qh < g; qh++) {
            float* out = &attn_out[(kv * g + qh) * HEAD_DIM];
            const float* sc = &kv_temp[qh * seq_len];
            for (int i = 0; i < HEAD_DIM; i++) {
                __m256 sum = _mm256_setzero_ps();
                int s = 0;
                for (; s + 8 <= seq_len; s += 8) {
                    __m256 sv = _mm256_loadu_ps(&sc[s]);
                    __m256 vv = _mm256_loadu_ps(&v_cache[(kv * max_s + s) * HEAD_DIM + i]);
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(sv, vv));
                }
                float total = hsum_ps(sum);
                for (; s < seq_len; s++)
                    total += sc[s] * v_cache[(kv * max_s + s) * HEAD_DIM + i];
                out[i] = total;
            }
        }
    }
}

typedef struct {
    TQ1Mat tq, tk, tv, to, tg, tu, td;
    float *ln1, *ln2;
} LayerW;

void dump_buf(float* buf, int n, const char* name, int layer, int seq) {
    char fn[128]; snprintf(fn, sizeof(fn), "dbg_%s_L%d_pos%d.bin", name, layer, seq);
    FILE* f = fopen(fn, "wb");
    if (f) { fwrite(buf, 4, n, f); fclose(f); }
}

__attribute__((noinline)) static void forward_layer(LayerW* l, float* x,
                          float* scr, int8_t* ai8,
                          float* gate, float* up,
                          float* k_cache, float* v_cache,
                          int seq_len, int max_s) {
    float entry_copy; memcpy(&entry_copy, x, 4);
    if (seq_len == 17) {
        FILE* pf = fopen("fwd_ptr_check.txt", "w");
        if (pf) { fprintf(pf, "x=%p entry_x0=%.10f\n", (void*)x, (double)entry_copy); fclose(pf); }
    }
    float *res = scr, *q = scr + HIDDEN;
    float* kv = scr + HIDDEN + NH_HD;
    memcpy(res, x, HIDDEN * 4);
    rms_norm(x, l->ln1, HIDDEN);
    float qq = quantize_fast(x, ai8, HIDDEN);
    float qo, qg, qd;

    mv_tq10_scales_omp(&l->tq, ai8, q);
    for (int i = 0; i < NH_HD; i++) q[i] *= qq;

    mv_tq10_scales_omp(&l->tk, ai8, kv);
    for (int i = 0; i < NKV_HD; i++) kv[i] *= qq;

    mv_tq10_scales_omp(&l->tv, ai8, kv + NKV_HD);
    for (int i = 0; i < NKV_HD; i++) kv[NKV_HD + i] *= qq;

    apply_rope(q, N_HEADS, seq_len);
    apply_rope(kv, N_KV_HEADS, seq_len);
    for (int kvi = 0; kvi < N_KV_HEADS; kvi++) {
        memcpy(&k_cache[(kvi * max_s + seq_len) * HEAD_DIM], &kv[kvi * HEAD_DIM], HEAD_DIM * 4);
        memcpy(&v_cache[(kvi * max_s + seq_len) * HEAD_DIM], &kv[NKV_HD + kvi * HEAD_DIM], HEAD_DIM * 4);
    }

    attention_gqa(q, q, k_cache, v_cache, seq_len + 1, max_s, scr + HIDDEN + NH_HD + NKV_HD);

    qo = quantize_fast(q, ai8, NH_HD);

    mv_tq10_scales_omp(&l->to, ai8, kv);
    for (int i = 0; i < HIDDEN; i++) kv[i] *= qo;

    for (int i = 0; i < HIDDEN; i++) x[i] = res[i] + kv[i];
    memcpy(res, x, HIDDEN * 4);
    rms_norm(x, l->ln2, HIDDEN);

    qg = quantize_fast(x, ai8, HIDDEN);

    mv_tq10_scales_omp(&l->tg, ai8, gate);
    mv_tq10_scales_omp(&l->tu, ai8, up);
    for (int i = 0; i < INTERMEDIATE; i++) gate[i] *= qg;
    for (int i = 0; i < INTERMEDIATE; i++) up[i] *= qg;

    for (int i = 0; i < INTERMEDIATE; i++)
        gate[i] = silu_f(gate[i]) * up[i];

    qd = quantize_fast(gate, ai8, INTERMEDIATE);

    mv_tq10_scales_omp(&l->td, ai8, q);
    for (int i = 0; i < HIDDEN; i++) q[i] *= qd;

    for (int i = 0; i < HIDDEN; i++) x[i] = res[i] + q[i];
}

int main(int argc, char** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    FILE* dbg = fopen("debug_cpp.txt", "w");
    if (!dbg) { fprintf(stderr, "FATAL: can't open debug file\n"); exit(1); }
    srand(time(NULL));
    float temp = TEMPERATURE;
    char* env_temp = getenv("ATLAS_TEMP");
    if (env_temp) { float v = (float)atof(env_temp); if (v > 0.0f && v < 5.0f) temp = v; }
    if (argc > 1) { float v = (float)atof(argv[1]); if (v > 0.0f && v < 5.0f) temp = v; }
    fprintf(dbg, "TEMP=%.2f\n", (double)temp); fflush(dbg);
    fprintf(dbg, "READY\n"); fflush(dbg);

    clock_t t_load = clock();
    init_rope();

    long n_emb; float* embed = load_full("embed.bin", &n_emb);
    float* lm_head = load_full("lm_head.bin", &n_emb);
    int nn; float* final_norm = load_vec("final_norm.bin", &nn);

    LayerW* layers = (LayerW*)_aligned_malloc(N_LAYERS * sizeof(LayerW), 32);
    for (int li = 0; li < N_LAYERS; li++) {
        char p[64];
        snprintf(p, sizeof(p), "l%d_q_proj", li); layers[li].tq = load_tq10(p);
        snprintf(p, sizeof(p), "l%d_k_proj", li); layers[li].tk = load_tq10(p);
        snprintf(p, sizeof(p), "l%d_v_proj", li); layers[li].tv = load_tq10(p);
        snprintf(p, sizeof(p), "l%d_o_proj", li); layers[li].to = load_tq10(p);
        snprintf(p, sizeof(p), "l%d_gate_proj", li); layers[li].tg = load_tq10(p);
        snprintf(p, sizeof(p), "l%d_up_proj", li); layers[li].tu = load_tq10(p);
        snprintf(p, sizeof(p), "l%d_down_proj", li); layers[li].td = load_tq10(p);
        char bn[128];
        snprintf(bn, sizeof(bn), "l%d_input_layernorm.bin", li); layers[li].ln1 = load_vec(bn, &nn);
        snprintf(bn, sizeof(bn), "l%d_post_attention_layernorm.bin", li); layers[li].ln2 = load_vec(bn, &nn);
    }

    fprintf(dbg, "Load %.2fs\n", (double)(clock()-t_load)/CLOCKS_PER_SEC); fflush(dbg);

    size_t layer_cache_sz = (size_t)N_KV_HEADS * MAX_SEQ * HEAD_DIM;
    size_t arena_sz = HIDDEN * 4 + 262144 * 4 + INTERMEDIATE + INTERMEDIATE * 8 + VOCAB * 4 + N_LAYERS * 2 * layer_cache_sz * 4;
    char* arena_base = (char*)malloc(arena_sz);
    if (!arena_base) { fprintf(stderr, "FATAL: arena alloc failed\n"); exit(1); }
    char* arena = arena_base;
    float* x      = (float*)arena; arena += HIDDEN * 4;
    float* scr    = (float*)arena; arena += 262144 * 4;
    int8_t* ai8   = (int8_t*)arena; arena += INTERMEDIATE;
    float* gate   = (float*)arena; arena += INTERMEDIATE * 4;
    float* up     = (float*)arena; arena += INTERMEDIATE * 4;
    float* logits = (float*)arena; arena += VOCAB * 4;
    float** k_caches = (float**)_aligned_malloc(N_LAYERS * sizeof(float*), 32);
    float** v_caches = (float**)_aligned_malloc(N_LAYERS * sizeof(float*), 32);
    for (int li = 0; li < N_LAYERS; li++) {
        k_caches[li] = (float*)arena; arena += layer_cache_sz * 4;
        v_caches[li] = (float*)arena; arena += layer_cache_sz * 4;
    }

    int stack_dummy = 0;
    fprintf(dbg, "ADDR x=%p scr=%p ai8=%p gate=%p up=%p logits=%p\n", (void*)x, (void*)scr, (void*)ai8, (void*)gate, (void*)up, (void*)logits);
    fprintf(dbg, "ADDR kc0=%p vc0=%p embed=%p lm_head=%p\n", (void*)k_caches[0], (void*)v_caches[0], (void*)embed, (void*)lm_head);
    fprintf(dbg, "ADDR arena_base=%p sz=%llu stack=%p\n", (void*)arena_base, (unsigned long long)arena_sz, (void*)&stack_dummy);

    while (1) {
        int prompt_len;
        if (fread(&prompt_len, 4, 1, stdin) != 1) break;
        if (prompt_len <= 0 || prompt_len >= MAX_SEQ) break;

        for (int li = 0; li < N_LAYERS; li++) {
            memset(k_caches[li], 0, layer_cache_sz * 4);
            memset(v_caches[li], 0, layer_cache_sz * 4);
        }

        for (int pos = 0; pos < prompt_len; pos++) {
            int tid;
            if (fread(&tid, 4, 1, stdin) != 1) goto cleanup;
            for (int _i = 0; _i < HIDDEN; _i++) x[_i] = embed[tid * HIDDEN + _i];
            if (pos == prompt_len - 1) {
                FILE* pb = fopen("post_copy_x0.txt", "w");
                if (pb) { fprintf(pb, "x=%p x0=%.10f\n", (void*)x, (double)x[0]); fclose(pb); }
            }
            for (int li = 0; li < N_LAYERS; li++) {
                forward_layer(&layers[li], x, scr, ai8, gate, up, k_caches[li], v_caches[li], pos, MAX_SEQ);
                if (pos == prompt_len - 1) {
                    double ss = 0; for (int i = 0; i < HIDDEN; i++) ss += (double)x[i]*x[i];
                    fprintf(dbg, "LYR_DBG L%02d rms=%.0f val0=%.1f\n", li, sqrt(ss/HIDDEN), x[0]);
                    fflush(dbg);
                    if (li == 0) {
                        FILE* f = fopen("dump_l0_pos17.bin", "wb");
                        if (f) { fwrite(x, 4, HIDDEN, f); fclose(f); }
                    }
                    if (li == N_LAYERS-1) {
                        FILE* f = fopen("dump_l39_pos17.bin", "wb");
                        if (f) { fwrite(x, 4, HIDDEN, f); fclose(f); }
                    }
                }
            }
        }

        rms_norm(x, final_norm, HIDDEN);
        #pragma omp parallel
        mv_fp32_omp(lm_head, x, logits, VOCAB, HIDDEN);

        {
            int top5_idx[5] = {0,0,0,0,0};
            float top5_val[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
            for (int i = 0; i < VOCAB; i++) {
                for (int j = 0; j < 5; j++) {
                    if (logits[i] > top5_val[j]) {
                        for (int k = 4; k > j; k--) { top5_val[k] = top5_val[k-1]; top5_idx[k] = top5_idx[k-1]; }
                        top5_val[j] = logits[i]; top5_idx[j] = i; break;
                    }
                }
            }
            fprintf(dbg, "TOKENS [");
            for (int j = 0; j < 5; j++) fprintf(dbg, "%d:%.2f%s", top5_idx[j], top5_val[j], j<4?", ":"");
            fprintf(dbg, "]\n"); fflush(dbg);

            int sampled = sample_token(logits, VOCAB, temp);
            fwrite(&sampled, 4, 1, stdout);
            fflush(stdout);

            for (int pos = prompt_len; pos < MAX_SEQ; pos++) {
                if (sampled == EOS_ID) break;
                int next_tok;
                if (fread(&next_tok, 4, 1, stdin) != 1) goto cleanup;
                if (next_tok == EOS_ID) break;
                memcpy(x, &embed[next_tok * HIDDEN], HIDDEN * 4);
                for (int li = 0; li < N_LAYERS; li++) {
                    forward_layer(&layers[li], x, scr, ai8, gate, up, k_caches[li], v_caches[li], pos, MAX_SEQ);
                    if (li == 0 || li == 39) {
                        double ss = 0; for (int i = 0; i < HIDDEN; i++) ss += (double)x[i]*x[i];
                        fprintf(dbg, "GEN_DBG L%d rms=%.6f\n", li, sqrt(ss/HIDDEN));
                    }
                }
                rms_norm(x, final_norm, HIDDEN);
                #pragma omp parallel
                mv_fp32_omp(lm_head, x, logits, VOCAB, HIDDEN);

                for (int i = 0; i < 5; i++) top5_val[i] = -1e30f;
                for (int i = 0; i < VOCAB; i++) {
                    for (int j = 0; j < 5; j++) {
                        if (logits[i] > top5_val[j]) {
                            for (int k = 4; k > j; k--) { top5_val[k] = top5_val[k-1]; top5_idx[k] = top5_idx[k-1]; }
                            top5_val[j] = logits[i]; top5_idx[j] = i; break;
                        }
                    }
                }
                fprintf(dbg, "TOKENS [");
                for (int j = 0; j < 5; j++) fprintf(dbg, "%d:%.2f%s", top5_idx[j], top5_val[j], j<4?", ":"");
                fprintf(dbg, "]\n"); fflush(dbg);

                sampled = sample_token(logits, VOCAB, temp);
                fwrite(&sampled, 4, 1, stdout);
                fflush(stdout);
            }
        }
    }

cleanup:
    _aligned_free(embed); _aligned_free(lm_head); _aligned_free(final_norm);
    free(arena_base); _aligned_free(k_caches); _aligned_free(v_caches);
    for (int li = 0; li < N_LAYERS; li++) {
        _aligned_free(layers[li].tq.data); _aligned_free(layers[li].tq.scales); _aligned_free(layers[li].tq.group_idx);
        _aligned_free(layers[li].tk.data); _aligned_free(layers[li].tk.scales); _aligned_free(layers[li].tk.group_idx);
        _aligned_free(layers[li].tv.data); _aligned_free(layers[li].tv.scales); _aligned_free(layers[li].tv.group_idx);
        _aligned_free(layers[li].to.data); _aligned_free(layers[li].to.scales); _aligned_free(layers[li].to.group_idx);
        _aligned_free(layers[li].tg.data); _aligned_free(layers[li].tg.scales); _aligned_free(layers[li].tg.group_idx);
        _aligned_free(layers[li].tu.data); _aligned_free(layers[li].tu.scales); _aligned_free(layers[li].tu.group_idx);
        _aligned_free(layers[li].td.data); _aligned_free(layers[li].td.scales); _aligned_free(layers[li].td.group_idx);
        _aligned_free(layers[li].ln1); _aligned_free(layers[li].ln2);
    }
    _aligned_free(layers);
    return 0;
}
