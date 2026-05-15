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

#define WDIR "C:\\dam\\atlas\\granite_tq10\\"
#define HIDDEN 2048
#define N_HEADS 32
#define N_KV_HEADS 8
#define HEAD_DIM 64
#define NH_HD (N_HEADS*HEAD_DIM)
#define NKV_HD (N_KV_HEADS*HEAD_DIM)
#define INTERMEDIATE 8192
#define EPS 1e-5
#define CLAMP_HIDDEN 512.0f
#define MAX_SEQ 4096
#define VOCAB 49155
#define N_LAYERS 40
#define BOS_ID 128000
#define EOS_ID 128001
#define ROPE_THETA 10000.0f
#define ROPE_HALF (HEAD_DIM/2)
#define EMBED_MULT 12.0f
#define ATTEN_MULT 0.015625f
#define RESID_MULT 0.22f
#define LOGIT_SCALE 8.0f
#define REP_PENALTY 1.6f
#define MAX_REP_WINDOW 32

static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
static inline int hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo); lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

static float cos_t[MAX_SEQ][HEAD_DIM];
static float sin_t[MAX_SEQ][HEAD_DIM];

static void init_rope() {
    for (int pos = 0; pos < MAX_SEQ; pos++) {
        for (int i = 0; i < ROPE_HALF; i++) {
            double inv = 1.0 / pow(ROPE_THETA, (2.0 * i) / HEAD_DIM);
            double v = pos * inv;
            cos_t[pos][i] = (float)cos(v);
            cos_t[pos][i + ROPE_HALF] = (float)cos(v);
            sin_t[pos][i] = (float)sin(v);
            sin_t[pos][i + ROPE_HALF] = (float)sin(v);
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
    const __m256i v3 = _mm256_set1_epi16(3);
    const __m256i v1 = _mm256_set1_epi16(1);
    const __m256i one16 = _mm256_set1_epi16(1);
    #pragma omp for schedule(static)
    for (int r = 0; r < rows; r++) {
        __m256i s32 = _mm256_setzero_si256();
        const uint8_t* row_data = &m->data[r * pc];
        for (int i = 0; i < cols; i += 32) {
            __m128i packed = _mm_loadl_epi64((__m128i*)&row_data[i / 4]);
            __m256i e = _mm256_cvtepu8_epi16(packed);
            __m256i p0 = _mm256_and_si256(e, v3);
            __m256i p1 = _mm256_and_si256(_mm256_srli_epi16(e, 2), v3);
            __m256i p2 = _mm256_and_si256(_mm256_srli_epi16(e, 4), v3);
            __m256i p3 = _mm256_and_si256(_mm256_srli_epi16(e, 6), v3);
            __m256i r0 = _mm256_sub_epi16(_mm256_and_si256(p0, v1), _mm256_srli_epi16(p0, 1));
            __m256i r1 = _mm256_sub_epi16(_mm256_and_si256(p1, v1), _mm256_srli_epi16(p1, 1));
            __m256i r2 = _mm256_sub_epi16(_mm256_and_si256(p2, v1), _mm256_srli_epi16(p2, 1));
            __m256i r3 = _mm256_sub_epi16(_mm256_and_si256(p3, v1), _mm256_srli_epi16(p3, 1));
            __m256i pa = _mm256_packs_epi16(r0, r1);
            __m256i pb = _mm256_packs_epi16(r2, r3);
            __m256i la = _mm256_unpacklo_epi8(pa, pb);
            __m256i ha = _mm256_unpackhi_epi8(pa, pb);
            __m128i wv_lo128 = _mm256_castsi256_si128(_mm256_unpacklo_epi8(la, ha));
            __m128i wv_hi128 = _mm256_castsi256_si128(_mm256_unpackhi_epi8(la, ha));
            __m256i wv = _mm256_inserti128_si256(_mm256_castsi128_si256(wv_lo128), wv_hi128, 1);
            __m256i av = _mm256_loadu_si256((__m256i*)&a[i]);
            __m256i rv = _mm256_sign_epi8(av, wv);
            __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(rv));
            __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(rv, 1));
            __m256i lo32 = _mm256_madd_epi16(lo, one16);
            __m256i hi32 = _mm256_madd_epi16(hi, one16);
            s32 = _mm256_add_epi32(s32, _mm256_add_epi32(lo32, hi32));
        }
        y[r] = (float)hsum_epi32(s32) * scales[gidx[r]];
    }
}

static void mv_tq10_hybrid(const TQ1Mat* m, const float* a, float* y) {
    int rows = m->rows, cols = m->cols, pc = m->packed_cols;
    const float* scales = m->scales;
    const int* gidx = m->group_idx;
    const __m256i v3 = _mm256_set1_epi16(3);
    const __m256i v1 = _mm256_set1_epi16(1);
    const __m256i one16 = _mm256_set1_epi16(1);
    float max_abs = 0.0f;
    for (int i = 0; i < cols; i++) { float ab = fabsf(a[i]); if (ab > max_abs) max_abs = ab; }
    if (max_abs < 1e-10f) max_abs = 1e-10f;
    float inv_s = 127.0f / max_abs;
    #pragma omp for schedule(static)
    for (int r = 0; r < rows; r++) {
        __m256i s32 = _mm256_setzero_si256();
        const uint8_t* row_data = &m->data[r * pc];
        for (int i = 0; i < cols; i += 32) {
            __m128i packed = _mm_loadl_epi64((__m128i*)&row_data[i / 4]);
            __m256i e = _mm256_cvtepu8_epi16(packed);
            __m256i p0 = _mm256_and_si256(e, v3);
            __m256i p1 = _mm256_and_si256(_mm256_srli_epi16(e, 2), v3);
            __m256i p2 = _mm256_and_si256(_mm256_srli_epi16(e, 4), v3);
            __m256i p3 = _mm256_and_si256(_mm256_srli_epi16(e, 6), v3);
            __m256i r0 = _mm256_sub_epi16(_mm256_and_si256(p0, v1), _mm256_srli_epi16(p0, 1));
            __m256i r1 = _mm256_sub_epi16(_mm256_and_si256(p1, v1), _mm256_srli_epi16(p1, 1));
            __m256i r2 = _mm256_sub_epi16(_mm256_and_si256(p2, v1), _mm256_srli_epi16(p2, 1));
            __m256i r3 = _mm256_sub_epi16(_mm256_and_si256(p3, v1), _mm256_srli_epi16(p3, 1));
            __m256i pa = _mm256_packs_epi16(r0, r1);
            __m256i pb = _mm256_packs_epi16(r2, r3);
            __m256i la = _mm256_unpacklo_epi8(pa, pb);
            __m256i ha = _mm256_unpackhi_epi8(pa, pb);
            __m128i wv_lo = _mm256_castsi256_si128(_mm256_unpacklo_epi8(la, ha));
            __m128i wv_hi = _mm256_castsi256_si128(_mm256_unpackhi_epi8(la, ha));
            __m256i wv = _mm256_inserti128_si256(_mm256_castsi128_si256(wv_lo), wv_hi, 1);
            float as[32]; for (int j = 0; j < 32; j++) as[j] = a[i + j];
            int8_t qi[32]; for (int j = 0; j < 32; j++) qi[j] = (int8_t)(roundf(as[j] * inv_s));
            __m256i av = _mm256_loadu_si256((__m256i*)qi);
            __m256i rv = _mm256_sign_epi8(av, wv);
            __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(rv));
            __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(rv, 1));
            __m256i lo32 = _mm256_madd_epi16(lo, one16);
            __m256i hi32 = _mm256_madd_epi16(hi, one16);
            s32 = _mm256_add_epi32(s32, _mm256_add_epi32(lo32, hi32));
        }
        y[r] = (float)hsum_epi32(s32) * scales[gidx[r]] * max_abs / 127.0f;
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
    for (int i = 0; i < n; i++) { float a = fabsf(s[i]); if (a > max_abs) max_abs = a; }
    if (max_abs < 1e-10f) max_abs = 1e-10f;
    float inv = 127.0f / max_abs;
    for (int i = 0; i < n; i++) d[i] = (int8_t)(roundf(s[i] * inv));
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

static void forward_layer(LayerW* l, float* x,
                          float* scr, int8_t* ai8,
                          float* gate, float* up,
                          float* k_cache, float* v_cache,
                          int seq_len, int max_s) {
    float *res = scr, *q = scr + HIDDEN;
    float* kv = scr + HIDDEN + NH_HD;
    memcpy(res, x, HIDDEN * 4);
    rms_norm(x, l->ln1, HIDDEN);
    float qq = quantize_fast(x, ai8, HIDDEN);

    #pragma omp parallel
    {
        mv_tq10_scales_omp(&l->tq, ai8, q);
        #pragma omp for schedule(static)
        for (int i = 0; i < NH_HD; i++) q[i] *= qq;

        mv_tq10_scales_omp(&l->tk, ai8, kv);
        #pragma omp for schedule(static)
        for (int i = 0; i < NKV_HD; i++) kv[i] *= qq;

        mv_tq10_scales_omp(&l->tv, ai8, kv + NKV_HD);
        #pragma omp for schedule(static)
        for (int i = 0; i < NKV_HD; i++) kv[NKV_HD + i] *= qq;

        #pragma omp single
        {
            apply_rope(q, N_HEADS, seq_len);
            apply_rope(kv, N_KV_HEADS, seq_len);
            for (int kvi = 0; kvi < N_KV_HEADS; kvi++) {
                memcpy(&k_cache[(kvi * max_s + seq_len) * HEAD_DIM], &kv[kvi * HEAD_DIM], HEAD_DIM * 4);
                memcpy(&v_cache[(kvi * max_s + seq_len) * HEAD_DIM], &kv[NKV_HD + kvi * HEAD_DIM], HEAD_DIM * 4);
            }
        }

        #pragma omp single
        attention_gqa(q, q, k_cache, v_cache, seq_len + 1, max_s, scr + HIDDEN + NH_HD + NKV_HD);

        #pragma omp single
        {
            for (int i = 0; i < NH_HD; i++) q[i] *= ATTEN_MULT;
        }

        #pragma omp single
        {
            for (int i = 0; i < HIDDEN; i++) x[i] = res[i] + q[i];
            for (int i = 0; i < HIDDEN; i++)
                x[i] = fmaxf(-CLAMP_HIDDEN, fminf(x[i], CLAMP_HIDDEN));
            memcpy(res, x, HIDDEN * 4);
            rms_norm(x, l->ln2, HIDDEN);
            qq = quantize_fast(x, ai8, HIDDEN);
        }

        mv_tq10_scales_omp(&l->tg, ai8, gate);
        mv_tq10_scales_omp(&l->tu, ai8, up);
        #pragma omp for schedule(static)
        for (int i = 0; i < INTERMEDIATE; i++) gate[i] *= qq;
        #pragma omp for schedule(static)
        for (int i = 0; i < INTERMEDIATE; i++) up[i] *= qq;

        #pragma omp single
        {
            for (int i = 0; i < INTERMEDIATE; i++)
                gate[i] = silu_f(gate[i]) * up[i];
        }

        mv_tq10_hybrid(&l->td, gate, q);
        #pragma omp for schedule(static)
        for (int i = 0; i < HIDDEN; i++) q[i] *= qq;

        #pragma omp single
        {
            for (int i = 0; i < HIDDEN; i++) q[i] *= RESID_MULT;
            for (int i = 0; i < HIDDEN; i++) x[i] = res[i] + q[i];
            for (int i = 0; i < HIDDEN; i++)
                x[i] = fmaxf(-CLAMP_HIDDEN, fminf(x[i], CLAMP_HIDDEN));
        }
    }
}

int main() {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    fprintf(stderr, "READY\n");

    clock_t t_load = clock();
    init_rope();

    long n_emb; float* embed = load_full("embed.bin", &n_emb);
    int nn; float* final_norm = load_vec("final_norm.bin", &nn);

    LayerW* layers = (LayerW*)_aligned_malloc(N_LAYERS * sizeof(LayerW), 32);
    for (int li = 0; li < N_LAYERS; li++) {
        char p[64], bn[128];
        snprintf(p, sizeof(p), "l%d.self_attn.q_proj", li); layers[li].tq = load_tq10(p);
        snprintf(p, sizeof(p), "l%d.self_attn.k_proj", li); layers[li].tk = load_tq10(p);
        snprintf(p, sizeof(p), "l%d.self_attn.v_proj", li); layers[li].tv = load_tq10(p);
        snprintf(p, sizeof(p), "l%d.self_attn.o_proj", li); layers[li].to = load_tq10(p);
        snprintf(p, sizeof(p), "l%d.mlp.gate_proj", li); layers[li].tg = load_tq10(p);
        snprintf(p, sizeof(p), "l%d.mlp.up_proj", li); layers[li].tu = load_tq10(p);
        snprintf(p, sizeof(p), "l%d.mlp.down_proj", li); layers[li].td = load_tq10(p);
        snprintf(bn, sizeof(bn), "l%d.input_layernorm.bin", li); layers[li].ln1 = load_vec(bn, &nn);
        snprintf(bn, sizeof(bn), "l%d.post_attention_layernorm.bin", li); layers[li].ln2 = load_vec(bn, &nn);
    }

    fprintf(stderr, "Load %.2fs\n", (double)(clock()-t_load)/CLOCKS_PER_SEC);

    float* x      = (float*)_aligned_malloc(HIDDEN * 4, 32);
    float* scr    = (float*)_aligned_malloc(262144 * 4, 32);
    int8_t* ai8   = (int8_t*)_aligned_malloc(INTERMEDIATE, 32);
    float* gate   = (float*)_aligned_malloc(INTERMEDIATE * 4, 32);
    float* up     = (float*)_aligned_malloc(INTERMEDIATE * 4, 32);
    float* logits = (float*)_aligned_malloc(VOCAB * 4, 32);
    int rep_hist[MAX_REP_WINDOW] = {0};
    int rep_count = 0;
    size_t layer_cache_sz = (size_t)N_KV_HEADS * MAX_SEQ * HEAD_DIM;
    float* k_cache = (float*)_aligned_malloc(layer_cache_sz * 4, 32);
    float* v_cache = (float*)_aligned_malloc(layer_cache_sz * 4, 32);

    while (1) {
        int prompt_len;
        if (fread(&prompt_len, 4, 1, stdin) != 1) break;
        if (prompt_len <= 0 || prompt_len >= MAX_SEQ) break;

        memset(k_cache, 0, layer_cache_sz * 4);
        memset(v_cache, 0, layer_cache_sz * 4);

        for (int pos = 0; pos < prompt_len; pos++) {
            int tid;
            if (fread(&tid, 4, 1, stdin) != 1) goto cleanup;
            memcpy(x, &embed[tid * HIDDEN], HIDDEN * 4);
            for (int i = 0; i < HIDDEN; i++) x[i] *= EMBED_MULT;
            for (int li = 0; li < N_LAYERS; li++)
                forward_layer(&layers[li], x, scr, ai8, gate, up, k_cache, v_cache, pos, MAX_SEQ);
        }

        rms_norm(x, final_norm, HIDDEN);
        #pragma omp parallel
        mv_fp32_omp(embed, x, logits, VOCAB, HIDDEN);
        #pragma omp for schedule(static)
        for (int i = 0; i < VOCAB; i++) logits[i] *= LOGIT_SCALE;

        for (int j = 0; j < rep_count; j++)
            logits[rep_hist[j]] /= REP_PENALTY;

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
            fprintf(stderr, "TOKENS [");
            for (int j = 0; j < 5; j++) fprintf(stderr, "%d:%.2f%s", top5_idx[j], top5_val[j], j<4?", ":"");
            fprintf(stderr, "]\n");

            float mx = logits[0]; int sampled = 0;
            for (int i = 1; i < VOCAB; i++) { if (logits[i] > mx) { mx = logits[i]; sampled = i; } }
            if (rep_count < MAX_REP_WINDOW) rep_hist[rep_count++] = sampled;
            fwrite(&sampled, 4, 1, stdout);
            fflush(stdout);

            for (int pos = prompt_len; pos < MAX_SEQ; pos++) {
                if (sampled == EOS_ID) break;
                int next_tok;
                if (fread(&next_tok, 4, 1, stdin) != 1) goto cleanup;
                if (next_tok == EOS_ID) break;
                memcpy(x, &embed[next_tok * HIDDEN], HIDDEN * 4);
                for (int i = 0; i < HIDDEN; i++) x[i] *= EMBED_MULT;
                for (int li = 0; li < N_LAYERS; li++)
                    forward_layer(&layers[li], x, scr, ai8, gate, up, k_cache, v_cache, pos, MAX_SEQ);
                rms_norm(x, final_norm, HIDDEN);
                #pragma omp parallel
                mv_fp32_omp(embed, x, logits, VOCAB, HIDDEN);
                #pragma omp for schedule(static)
                for (int i = 0; i < VOCAB; i++) logits[i] *= LOGIT_SCALE;

                for (int j = 0; j < rep_count; j++)
                    logits[rep_hist[j]] /= REP_PENALTY;

                for (int i = 0; i < 5; i++) top5_val[i] = -1e30f;
                for (int i = 0; i < VOCAB; i++) {
                    for (int j = 0; j < 5; j++) {
                        if (logits[i] > top5_val[j]) {
                            for (int k = 4; k > j; k--) { top5_val[k] = top5_val[k-1]; top5_idx[k] = top5_idx[k-1]; }
                            top5_val[j] = logits[i]; top5_idx[j] = i; break;
                        }
                    }
                }
                fprintf(stderr, "TOKENS [");
                for (int j = 0; j < 5; j++) fprintf(stderr, "%d:%.2f%s", top5_idx[j], top5_val[j], j<4?", ":"");
                fprintf(stderr, "]\n");

                mx = logits[0]; sampled = 0;
                for (int i = 1; i < VOCAB; i++) { if (logits[i] > mx) { mx = logits[i]; sampled = i; } }
                if (rep_count < MAX_REP_WINDOW) rep_hist[rep_count++] = sampled;
                else { memmove(rep_hist, rep_hist + 1, (MAX_REP_WINDOW - 1) * 4); rep_hist[MAX_REP_WINDOW - 1] = sampled; }
                fwrite(&sampled, 4, 1, stdout);
                fflush(stdout);
            }
        }
    }

cleanup:
    _aligned_free(embed); _aligned_free(final_norm);
    _aligned_free(x); _aligned_free(scr); _aligned_free(ai8);
    _aligned_free(gate); _aligned_free(up);
    _aligned_free(logits); _aligned_free(k_cache); _aligned_free(v_cache);
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
