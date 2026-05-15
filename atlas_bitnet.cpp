// Atlas Granite — TQ1_0 pure AVX2 engine for Granite 3.0 2B (ternarized)
// clang++ -O3 -mavx2 -lm atlas_bitnet.cpp -o atlas_granite.exe

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>

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
#define EPS 1e-6f
#define MAX_SEQ 4096
#define VOCAB 49155
#define N_LAYERS 40
#define BOS_ID 0
#define EOS_ID 0

// Granite config multipliers
#define EMBED_MULT   12.0f
#define ATTN_SCALE   0.015625f  // attention_multiplier (score scaling before softmax)
#define RESID_MULT   0.22f      // residual_multiplier (same for attn + MLP)
#define LOGIT_SCALE  (1.0f/8.0f)  // logits_scaling=8.0: logits /= 8.0

static uint64_t rng_state = 0;

static inline uint64_t xorshift64() {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

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
} TQ1Mat;

static TQ1Mat load_tq10(const char* name) {
    char p[512]; snprintf(p, sizeof(p), "%s%s.tq10", WDIR, name);
    FILE* f = fopen(p, "rb"); if (!f) {
        snprintf(p, sizeof(p), "%slayer0_%s.tq10", WDIR, name);
        f = fopen(p, "rb");
        if (!f) { fprintf(stderr, "ERROR: %s\n", p); exit(1); }
    }
    TQ1Mat m;
    fread(&m.rows, 4, 1, f);
    fread(&m.cols, 4, 1, f);
    fread(&m.group_size, 4, 1, f);
    fread(&m.num_groups, 4, 1, f);
    m.scales = (float*)_aligned_malloc(m.num_groups * 4, 32);
    fread(m.scales, 4, m.num_groups, f);
    m.packed_cols = (m.cols + 3) / 4;
    m.data = (uint8_t*)_aligned_malloc(m.rows * m.packed_cols, 32);
    fread(m.data, 1, m.rows * m.packed_cols, f);
    fclose(f);
    return m;
}

static void mv_tq10_scales(const TQ1Mat* m, const int8_t* a, float* y) {
    int rows = m->rows, cols = m->cols, pc = m->packed_cols;
    const float* scales = m->scales;
    const __m256i v3 = _mm256_set1_epi16(3);
    const __m256i v1 = _mm256_set1_epi16(1);

    for (int r = 0; r < rows; r++) {
        __m256i s16 = _mm256_setzero_si256();
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
            s16 = _mm256_add_epi16(s16, _mm256_add_epi16(lo, hi));
        }

        int dot = hsum_epi32(_mm256_madd_epi16(s16, _mm256_set1_epi16(1)));
        int gi = r / m->group_size;
        if (gi >= m->num_groups) gi = m->num_groups - 1;
        y[r] = dot * scales[gi];
    }
}

static void mv_fp32(const float* w, const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        __m256 sum = _mm256_setzero_ps();
        for (int i = 0; i < cols; i += 8)
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(&w[r*cols+i]), _mm256_loadu_ps(&x[i])));
        y[r] = hsum_ps(sum);
    }
}

static float quantize(const float* s, int8_t* d, int n) {
    float ma = 1e-10f; for (int i = 0; i < n; i++) { float a = fabsf(s[i]); if (a > ma) ma = a; }
    float q = ma / 127.0f; if (q < 1e-10f) q = 1e-10f;
    float inv_q = 1.0f / q;
    for (int i = 0; i < n; i++) {
        float v = roundf(s[i] * inv_q);
        d[i] = (int8_t)fmaxf(-128.0f, fminf(127.0f, v));
    }
    return q;
}

static void rms_norm(float* x, const float* w, int n) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + EPS);
    for (int i = 0; i < n; i++) x[i] *= inv * w[i];
}

// Precomputed RoPE cos/sin tables
static float* rope_cos = NULL;
static float* rope_sin = NULL;

static void init_rope() {
    rope_cos = (float*)_aligned_malloc(MAX_SEQ * HEAD_DIM * 4, 32);
    rope_sin = (float*)_aligned_malloc(MAX_SEQ * HEAD_DIM * 4, 32);
    float theta = 10000.0f;
    int half = HEAD_DIM / 2;
    for (int pos = 0; pos < MAX_SEQ; pos++) {
        for (int i = 0; i < half; i++) {
            float inv_freq = 1.0f / powf(theta, (float)(2 * i) / HEAD_DIM);
            float val = pos * inv_freq;
            rope_cos[pos * HEAD_DIM + i] = cosf(val);
            rope_sin[pos * HEAD_DIM + i] = sinf(val);
            rope_cos[pos * HEAD_DIM + i + half] = cosf(val);
            rope_sin[pos * HEAD_DIM + i + half] = sinf(val);
        }
    }
}

static inline float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

static void apply_rope(float* v, int n_heads, int pos) {
    const float* cos_t = &rope_cos[pos * HEAD_DIM];
    const float* sin_t = &rope_sin[pos * HEAD_DIM];
    int half = HEAD_DIM / 2;
    for (int h = 0; h < n_heads; h++) {
        float* hv = &v[h * HEAD_DIM];
        for (int i = 0; i < half; i++) {
            float c = cos_t[i], s = sin_t[i];
            float x0 = hv[i], x1 = hv[i + half];
            hv[i] = x0 * c - x1 * s;
            hv[i + half] = x0 * s + x1 * c;
        }
    }
}

static void attention_gqa(float* attn_out, const float* q,
                          const float* k_cache, const float* v_cache,
                          int seq_len, int max_s, float* temp) {
    int g = N_HEADS / N_KV_HEADS;
    for (int kv = 0; kv < N_KV_HEADS; kv++) {
        for (int qh = 0; qh < g; qh++) {
            const float* qv = &q[(kv * g + qh) * HEAD_DIM];
            float* sc = &temp[qh * seq_len];
            for (int s = 0; s < seq_len; s++) {
                __m256 sum = _mm256_setzero_ps();
                const float* kv_vec = &k_cache[(kv * max_s + s) * HEAD_DIM];
                for (int i = 0; i < HEAD_DIM; i += 8)
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(&qv[i]), _mm256_loadu_ps(&kv_vec[i])));
                sc[s] = hsum_ps(sum) * ATTN_SCALE;
            }
        }
        for (int qh = 0; qh < g; qh++) {
            float* sc = &temp[qh * seq_len];
            float mx = sc[0]; for (int s = 1; s < seq_len; s++) if (sc[s] > mx) mx = sc[s];
            float sum = 0; for (int s = 0; s < seq_len; s++) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
            float inv_sum = 1.0f / sum; for (int s = 0; s < seq_len; s++) sc[s] *= inv_sum;
        }
        for (int qh = 0; qh < g; qh++) {
            float* out = &attn_out[(kv * g + qh) * HEAD_DIM];
            const float* sc = &temp[qh * seq_len];
            for (int i = 0; i < HEAD_DIM; i++) {
                float sum = 0; int s = 0;
                for (; s + 8 <= seq_len; s += 8) {
                    __m256 sv = _mm256_loadu_ps(&sc[s]);
                    __m256 vv = _mm256_loadu_ps(&v_cache[(kv * max_s + s) * HEAD_DIM + i]);
                    sum += hsum_ps(_mm256_mul_ps(sv, vv));
                }
                for (; s < seq_len; s++)
                    sum += sc[s] * v_cache[(kv * max_s + s) * HEAD_DIM + i];
                out[i] = sum;
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
    float *res = scr, *q = scr + HIDDEN, *kv = scr + HIDDEN + NH_HD;

    // Layer diagnostics for layer 0
    static int layer_counter = 0;
    int layer_id = layer_counter++;

    // Attention block
    memcpy(res, x, HIDDEN*4); rms_norm(x, l->ln1, HIDDEN);
    float qq = quantize(x, ai8, HIDDEN);
    mv_tq10_scales(&l->tq, ai8, q);  for (int i=0;i<NH_HD;i++) q[i] *= qq;
    mv_tq10_scales(&l->tk, ai8, kv); for (int i=0;i<NKV_HD;i++) kv[i] *= qq;
    mv_tq10_scales(&l->tv, ai8, kv+NKV_HD); for (int i=0;i<NKV_HD;i++) kv[NKV_HD+i] *= qq;

    apply_rope(q, N_HEADS, seq_len);
    apply_rope(kv, N_KV_HEADS, seq_len);

    for (int kvi = 0; kvi < N_KV_HEADS; kvi++) {
        memcpy(&k_cache[(kvi * max_s + seq_len) * HEAD_DIM], &kv[kvi * HEAD_DIM], HEAD_DIM * 4);
        memcpy(&v_cache[(kvi * max_s + seq_len) * HEAD_DIM], &kv[NKV_HD + kvi*HEAD_DIM], HEAD_DIM * 4);
    }

    attention_gqa(q, q, k_cache, v_cache, seq_len + 1, max_s, scr + HIDDEN + NH_HD + NKV_HD);

    float qo = quantize(q, ai8, NH_HD);
    mv_tq10_scales(&l->to, ai8, kv); for (int i=0;i<HIDDEN;i++) kv[i] *= qo;
    for (int i = 0; i < HIDDEN; i++) x[i] = res[i] + kv[i] * RESID_MULT;

    // FFN block (SwiGLU: silu(gate) * up)
    memcpy(res, x, HIDDEN*4); rms_norm(x, l->ln2, HIDDEN);
    float qg = quantize(x, ai8, HIDDEN);
    mv_tq10_scales(&l->tg, ai8, gate); for (int i=0;i<INTERMEDIATE;i++) gate[i] *= qg;
    mv_tq10_scales(&l->tu, ai8, up);   for (int i=0;i<INTERMEDIATE;i++) up[i] *= qg;
    for (int i = 0; i < INTERMEDIATE; i++) gate[i] = silu_f(gate[i]) * up[i];
    float qd = quantize(gate, ai8, INTERMEDIATE);
    mv_tq10_scales(&l->td, ai8, q); for (int i=0;i<HIDDEN;i++) q[i] *= qd;
    for (int i = 0; i < HIDDEN; i++) x[i] = res[i] + q[i] * RESID_MULT;

    if (layer_id == 0) {
        float l0_mean = 0, l0_var = 0;
        for (int i = 0; i < HIDDEN; i++) l0_mean += x[i];
        l0_mean /= HIDDEN;
        for (int i = 0; i < HIDDEN; i++) l0_var += (x[i] - l0_mean) * (x[i] - l0_mean);
        l0_var /= HIDDEN;
        fprintf(stderr, "  [layer0_output] mean=%f var=%f std=%f\n", l0_mean, l0_var, sqrtf(l0_var));
    }
}

static int sample(float* logits, int n, float temp, int top_k) {
    if (temp > 0) {
        float inv = 1.0f / temp;
        for (int i = 0; i < n; i++) logits[i] *= inv;
    }
    if (top_k > 0 && top_k < n) {
        float* copy = (float*)malloc(n * 4);
        memcpy(copy, logits, n * 4);
        for (int i = 0; i < top_k; i++) {
            int best = i;
            for (int j = i + 1; j < n; j++)
                if (copy[j] > copy[best]) best = j;
            float t = copy[i]; copy[i] = copy[best]; copy[best] = t;
        }
        float kth = copy[top_k - 1];
        free(copy);
        for (int i = 0; i < n; i++) if (logits[i] < kth) logits[i] = -INFINITY;
    }
    float mx = logits[0]; for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0; for (int i = 0; i < n; i++) { logits[i] = expf(logits[i] - mx); sum += logits[i]; }
    float inv = 1.0f / sum; float r = (float)(xorshift64() & 0x7FFFFFFF) / 2147483648.0f;
    float cum = 0; for (int i = 0; i < n; i++) { cum += logits[i] * inv; if (r < cum) return i; }
    return n - 1;
}

int main() {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    rng_state = (uint64_t)time(NULL) ^ (uint64_t)(size_t)&rng_state;

    clock_t t0 = clock();
    fprintf(stderr, "Atlas Granite — Loading TQ1_0 packed weights...\n");

    init_rope();

    long n_emb; float* embed = load_full("embed.bin", &n_emb);
    int nn; float* final_norm = load_vec("final_norm.bin", &nn);

    fprintf(stderr, "Loading 40 layers (TQ1_0)...\n");
    LayerW* layers = (LayerW*)_aligned_malloc(N_LAYERS * sizeof(LayerW), 32);
    for (int li = 0; li < N_LAYERS; li++) {
        char p[128];
        snprintf(p, sizeof(p), "l%d.self_attn.q_proj", li);  layers[li].tq = load_tq10(p);
        snprintf(p, sizeof(p), "l%d.self_attn.k_proj", li);  layers[li].tk = load_tq10(p);
        snprintf(p, sizeof(p), "l%d.self_attn.v_proj", li);  layers[li].tv = load_tq10(p);
        snprintf(p, sizeof(p), "l%d.self_attn.o_proj", li);  layers[li].to = load_tq10(p);
        snprintf(p, sizeof(p), "l%d.mlp.gate_proj", li);     layers[li].tg = load_tq10(p);
        snprintf(p, sizeof(p), "l%d.mlp.up_proj", li);       layers[li].tu = load_tq10(p);
        snprintf(p, sizeof(p), "l%d.mlp.down_proj", li);     layers[li].td = load_tq10(p);
        char ln1n[128], ln2n[128];
        snprintf(ln1n, sizeof(ln1n), "l%d.input_layernorm.bin", li);
        snprintf(ln2n, sizeof(ln2n), "l%d.post_attention_layernorm.bin", li);
        layers[li].ln1 = load_vec(ln1n, &nn);
        layers[li].ln2 = load_vec(ln2n, &nn);
    }

    float* x     = (float*)_aligned_malloc(HIDDEN*4, 32);
    float* scr   = (float*)_aligned_malloc(262144*4, 32);
    int8_t* ai8  = (int8_t*)_aligned_malloc(INTERMEDIATE, 32);
    float* gate  = (float*)_aligned_malloc(INTERMEDIATE*4, 32);
    float* up    = (float*)_aligned_malloc(INTERMEDIATE*4, 32);
    float* logits= (float*)_aligned_malloc(VOCAB*4, 32);
    float* k_cache = (float*)_aligned_malloc(N_KV_HEADS * MAX_SEQ * HEAD_DIM * 4, 32);
    float* v_cache = (float*)_aligned_malloc(N_KV_HEADS * MAX_SEQ * HEAD_DIM * 4, 32);

    clock_t t1 = clock();
    fprintf(stderr, "Load time: %.2f sec\n", (double)(t1-t0)/CLOCKS_PER_SEC);
    fprintf(stderr, "Ready. 40 layers, GQA(32,8), SwiGLU, RoPE(theta=10k)\n");

    while (1) {
        int prompt_len;
        if (fread(&prompt_len, 4, 1, stdin) != 1) break;
        if (prompt_len <= 0 || prompt_len >= MAX_SEQ) break;

        memset(k_cache, 0, N_KV_HEADS * MAX_SEQ * HEAD_DIM * 4);
        memset(v_cache, 0, N_KV_HEADS * MAX_SEQ * HEAD_DIM * 4);

        clock_t gen_start = clock();
        long total_tokens = 0;

        for (int pos = 0; pos < prompt_len; pos++) {
            int tid;
            if (fread(&tid, 4, 1, stdin) != 1) goto cleanup;
            memcpy(x, &embed[tid * HIDDEN], HIDDEN * 4);
            for (int i = 0; i < HIDDEN; i++) x[i] *= EMBED_MULT;
            if (pos == 0) {
                float e_mean = 0, e_var = 0;
                for (int i = 0; i < HIDDEN; i++) e_mean += x[i];
                e_mean /= HIDDEN;
                for (int i = 0; i < HIDDEN; i++) e_var += (x[i] - e_mean) * (x[i] - e_mean);
                e_var /= HIDDEN;
                fprintf(stderr, "  [embed] mean=%f var=%f std=%f\n", e_mean, e_var, sqrtf(e_var));
            }
            for (int li = 0; li < N_LAYERS; li++)
                forward_layer(&layers[li], x, scr, ai8, gate, up, k_cache, v_cache, pos, MAX_SEQ);
            total_tokens++;
        }

        // Hidden state diagnostics before final projection
        float h_mean = 0, h_var = 0, h_min = 1e30f, h_max = -1e30f;
        for (int i = 0; i < HIDDEN; i++) {
            h_mean += x[i];
            if (x[i] < h_min) h_min = x[i];
            if (x[i] > h_max) h_max = x[i];
        }
        h_mean /= HIDDEN;
        for (int i = 0; i < HIDDEN; i++) h_var += (x[i] - h_mean) * (x[i] - h_mean);
        h_var /= HIDDEN;
        fprintf(stderr, "  [hidden] mean=%f var=%f std=%f min=%f max=%f\n",
                h_mean, h_var, sqrtf(h_var), h_min, h_max);

        rms_norm(x, final_norm, HIDDEN);
        { float hm = 0, hv = 0; for (int i = 0; i < HIDDEN; i++) hm += x[i]; hm /= HIDDEN;
          for (int i = 0; i < HIDDEN; i++) hv += (x[i] - hm) * (x[i] - hm); hv /= HIDDEN;
          fprintf(stderr, "  [hidden_after_norm] mean=%f var=%f std=%f\n", hm, hv, sqrtf(hv)); }

        mv_fp32(embed, x, logits, VOCAB, HIDDEN);
        for (int i = 0; i < VOCAB; i++) logits[i] *= LOGIT_SCALE;

        // Logit statistics dump on first generation
        float lmax = -1e30f, lmin = 1e30f, lsum = 0;
        int lmax_id = 0, lmin_id = 0;
        for (int i = 0; i < VOCAB; i++) {
            if (logits[i] > lmax) { lmax = logits[i]; lmax_id = i; }
            if (logits[i] < lmin) { lmin = logits[i]; lmin_id = i; }
            lsum += logits[i];
        }
        float lmean = lsum / VOCAB;
        // Find top-5
        int top5_ids[5] = {0};
        float top5_vals[5] = {-1e30f};
        for (int i = 0; i < VOCAB; i++) {
            for (int k = 0; k < 5; k++) {
                if (logits[i] > top5_vals[k]) {
                    for (int k2 = 4; k2 > k; k2--) { top5_ids[k2] = top5_ids[k2-1]; top5_vals[k2] = top5_vals[k2-1]; }
                    top5_ids[k] = i; top5_vals[k] = logits[i]; break;
                }
            }
        }
        fprintf(stderr, "  [logit] max=%f (id=%d) min=%f (id=%d) mean=%f std_est=%f\n",
                lmax, lmax_id, lmin, lmin_id, lmean, lmax - lmean);
        fprintf(stderr, "  [logit] top5: %d:%f %d:%f %d:%f %d:%f %d:%f\n",
                top5_ids[0], top5_vals[0], top5_ids[1], top5_vals[1],
                top5_ids[2], top5_vals[2], top5_ids[3], top5_vals[3],
                top5_ids[4], top5_vals[4]);

        int sampled = sample(logits, VOCAB, 0.7f, 40);
        fwrite(&sampled, 4, 1, stdout);
        fflush(stdout);
        total_tokens++;

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
            mv_fp32(embed, x, logits, VOCAB, HIDDEN);
            for (int i = 0; i < VOCAB; i++) logits[i] *= LOGIT_SCALE;
            sampled = sample(logits, VOCAB, 0.5f, 20);
            fwrite(&sampled, 4, 1, stdout);
            fflush(stdout);
            total_tokens++;
        }

        clock_t gen_end = clock();
        double gen_sec = (double)(gen_end - gen_start) / CLOCKS_PER_SEC;
        fprintf(stderr, "  tokens=%ld  %.2f sec  %.2f t/s\n", total_tokens, gen_sec, total_tokens / gen_sec);
    }

cleanup:
    _aligned_free(embed); _aligned_free(final_norm);
    _aligned_free(x); _aligned_free(scr); _aligned_free(ai8);
    _aligned_free(gate); _aligned_free(up);
    _aligned_free(logits); _aligned_free(k_cache); _aligned_free(v_cache);
    _aligned_free(rope_cos); _aligned_free(rope_sin);
    for (int li = 0; li < N_LAYERS; li++) {
        _aligned_free(layers[li].tq.data); _aligned_free(layers[li].tq.scales);
        _aligned_free(layers[li].tk.data); _aligned_free(layers[li].tk.scales);
        _aligned_free(layers[li].tv.data); _aligned_free(layers[li].tv.scales);
        _aligned_free(layers[li].to.data); _aligned_free(layers[li].to.scales);
        _aligned_free(layers[li].tg.data); _aligned_free(layers[li].tg.scales);
        _aligned_free(layers[li].tu.data); _aligned_free(layers[li].tu.scales);
        _aligned_free(layers[li].td.data); _aligned_free(layers[li].td.scales);
        _aligned_free(layers[li].ln1); _aligned_free(layers[li].ln2);
    }
    _aligned_free(layers);
    return 0;
}
