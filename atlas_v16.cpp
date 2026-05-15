// Atlas v1.6 - Full 28-layer Falcon3 inference with stdin pipe + sampling
// clang++ -O3 -mavx2 -lm atlas_v16.cpp -o atlas_v16.exe
// Protocol:
//   read: [prompt_len:int32] [tok0:int32] ... [tokN-1:int32]
//   write: [sampled:int32]
//   loop: read [next_tok:int32] -> write [sampled:int32] until EOS or pipe close

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

#define WDIR "C:\\dam\\atlas\\weights\\"
#define HIDDEN 3072
#define N_HEADS 12
#define N_KV_HEADS 4
#define HEAD_DIM 256
#define NH_HD (N_HEADS*HEAD_DIM)
#define NKV_HD (N_KV_HEADS*HEAD_DIM)
#define INTERMEDIATE 23040
#define EPS 1e-6f
#define MAX_SEQ 2048
#define VOCAB 131072
#define N_LAYERS 28
#define EOS_ID 11

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

static void* load_bin(const char* name, const char* suf, int esz, int* r, int* c) {
    char p[512]; snprintf(p, sizeof(p), "%s%s_%s", WDIR, name, suf);
    FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "ERROR: %s\n", p); exit(1); }
    fread(r, 4, 1, f); fread(c, 4, 1, f);
    void* d = _aligned_malloc((*r) * (*c) * esz, 32);
    fread(d, esz, (*r) * (*c), f); fclose(f);
    return d;
}
#define load_tern(n, r, c) (int8_t*)load_bin(n, "tern.bin", 1, r, c)

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

static void mv_tern(const int8_t* w, const int8_t* a, int* y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        __m256i s16 = _mm256_setzero_si256();
        for (int i = 0; i < cols; i += 32) {
            __m256i wv = _mm256_loadu_si256((__m256i*)&w[r*cols+i]);
            __m256i av = _mm256_loadu_si256((__m256i*)&a[i]);
            __m256i rv = _mm256_sign_epi8(av, wv);
            __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(rv));
            __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(rv, 1));
            s16 = _mm256_add_epi16(s16, _mm256_add_epi16(lo, hi));
        }
        y[r] = hsum_epi32(_mm256_madd_epi16(s16, _mm256_set1_epi16(1)));
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
    float q = ma / 127.0f; for (int i = 0; i < n; i++) d[i] = (int8_t)(s[i] / q);
    return q;
}
static void dequant(const int* s, float* d, float ws, float qs, int n) {
    float sc = ws * qs; for (int i = 0; i < n; i++) d[i] = sc * s[i];
}

static void rms_norm(float* x, const float* w, int n) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + EPS);
    for (int i = 0; i < n; i++) x[i] *= inv * w[i];
}
static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }

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
                sc[s] = hsum_ps(sum) / sqrtf(HEAD_DIM);
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
    int8_t *tq, *tk, *tv, *to, *tg, *tu, *td;
    float *ln1, *ln2, sq, sk, sv, so, sg, su, sd;
} LayerW;

static float SCALES[N_LAYERS][7] = {
    {48.7500f,36.7500f,26.7500f,27.8750f,11.4375f,11.7500f,10.8750f},
    {10.2500f,9.0625f,13.0625f,12.5000f,10.3125f,11.7500f,11.5000f},
    {10.5000f,9.1875f,11.5625f,12.0625f,10.4375f,11.3125f,11.1250f},
    {10.8125f,10.5000f,10.3125f,11.1250f,9.3750f,11.8125f,11.7500f},
    {10.8125f,10.3125f,10.0625f,10.8750f,9.6875f,11.5000f,11.6250f},
    {10.8750f,10.1875f,10.2500f,11.1875f,10.0625f,11.1875f,11.2500f},
    {10.5000f,10.0000f,10.5000f,10.8750f,10.7500f,10.8125f,10.9375f},
    {10.6875f,10.0000f,11.1875f,11.5000f,11.0625f,11.1250f,11.1875f},
    {10.8125f,10.5000f,10.7500f,11.5000f,11.4375f,11.3125f,11.1875f},
    {11.1250f,10.9375f,10.3125f,10.7500f,11.0625f,11.0000f,11.1875f},
    {10.7500f,10.2500f,10.8750f,11.1250f,11.0000f,11.1250f,11.2500f},
    {10.7500f,9.7500f,11.3125f,11.5000f,11.1250f,11.1250f,11.3125f},
    {11.0625f,11.0000f,10.7500f,11.3125f,11.4375f,11.1250f,11.1875f},
    {11.5625f,11.7500f,9.5000f,10.3750f,11.3125f,10.8750f,11.0000f},
    {11.0000f,10.9375f,10.3125f,10.7500f,11.1250f,10.8750f,10.9375f},
    {11.3125f,11.6250f,10.8125f,11.3125f,11.0625f,10.7500f,10.8750f},
    {11.5000f,11.7500f,10.1250f,10.6250f,10.9375f,10.7500f,10.8125f},
    {11.1250f,10.7500f,10.6875f,11.1250f,10.8750f,10.7500f,10.7500f},
    {11.2500f,11.5625f,10.0625f,10.8750f,10.8125f,10.6875f,10.7500f},
    {12.0000f,13.3125f,9.6875f,10.7500f,10.8125f,10.6250f,10.7500f},
    {12.5625f,13.0625f,9.3750f,10.6875f,10.6875f,10.6875f,10.8125f},
    {12.3125f,12.3125f,9.2500f,10.5625f,10.8125f,10.6250f,10.7500f},
    {12.8750f,14.3125f,9.3750f,10.3125f,10.8125f,10.4375f,10.5625f},
    {13.1875f,14.6250f,7.8750f,9.6250f,10.5625f,10.5000f,10.4375f},
    {13.3125f,15.1875f,7.5312f,9.3125f,10.5625f,10.3125f,10.3750f},
    {13.5625f,14.6250f,8.7500f,10.3125f,10.7500f,10.3125f,10.4375f},
    {14.1250f,14.8125f,7.9375f,9.9375f,11.1875f,10.4375f,11.3125f},
    {15.3125f,16.8750f,10.1875f,12.9375f,12.0625f,11.5625f,14.6250f}
};

static void forward_layer(LayerW* l, int li, float* x,
                          float* scr, int8_t* ai8, int* bi,
                          float* gate, float* up,
                          float* k_cache, float* v_cache,
                          int seq_len, int max_s) {
    float *res = scr, *q = scr + HIDDEN, *kv = scr + HIDDEN + NH_HD;
    float sq=l->sq, sk=l->sk, sv=l->sv, so=l->so, sg=l->sg, su=l->su, sd=l->sd;

    memcpy(res, x, HIDDEN*4); rms_norm(x, l->ln1, HIDDEN);
    float qq = quantize(x, ai8, HIDDEN);
    mv_tern(l->tq, ai8, bi, NH_HD, HIDDEN); dequant(bi, q, sq, qq, NH_HD);
    mv_tern(l->tk, ai8, bi, NKV_HD, HIDDEN); dequant(bi, kv, sk, qq, NKV_HD);
    mv_tern(l->tv, ai8, bi, NKV_HD, HIDDEN); dequant(bi, kv+NKV_HD, sv, qq, NKV_HD);

    for (int kvi = 0; kvi < N_KV_HEADS; kvi++) {
        memcpy(&k_cache[(kvi * max_s + seq_len) * HEAD_DIM], &kv[kvi * HEAD_DIM], HEAD_DIM * 4);
        memcpy(&v_cache[(kvi * max_s + seq_len) * HEAD_DIM], &kv[NKV_HD + kvi*HEAD_DIM], HEAD_DIM * 4);
    }

    attention_gqa(q, q, k_cache, v_cache, seq_len + 1, max_s, scr + HIDDEN + NH_HD + NKV_HD);

    float qo = quantize(q, ai8, NH_HD);
    mv_tern(l->to, ai8, bi, HIDDEN, NH_HD); dequant(bi, kv, so, qo, HIDDEN);
    for (int i = 0; i < HIDDEN; i++) x[i] = res[i] + kv[i];

    memcpy(res, x, HIDDEN*4); rms_norm(x, l->ln2, HIDDEN);
    float qg = quantize(x, ai8, HIDDEN);
    mv_tern(l->tg, ai8, bi, INTERMEDIATE, HIDDEN); dequant(bi, gate, sg, qg, INTERMEDIATE);
    mv_tern(l->tu, ai8, bi, INTERMEDIATE, HIDDEN); dequant(bi, up, su, qg, INTERMEDIATE);
    for (int i = 0; i < INTERMEDIATE; i++) gate[i] = silu_f(gate[i]) * up[i];
    float qd = quantize(gate, ai8, INTERMEDIATE);
    mv_tern(l->td, ai8, bi, HIDDEN, INTERMEDIATE); dequant(bi, q, sd, qd, HIDDEN);
    for (int i = 0; i < HIDDEN; i++) x[i] = res[i] + q[i];
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
    fprintf(stderr, "Atlas v1.6 - Loading weights...\n");

    long n_emb; float* embed = load_full("model_embed_tokens_bin", &n_emb);
    int nn; float* model_norm = load_vec("model_norm.bin", &nn);
    long n_head; float* lm_head = load_full("lm_head_bin", &n_head);

    fprintf(stderr, "Loading 28 layers...\n");
    LayerW* layers = (LayerW*)_aligned_malloc(N_LAYERS * sizeof(LayerW), 32);
    for (int li = 0; li < N_LAYERS; li++) {
        int r, c; char p[64];
        snprintf(p, sizeof(p), "layer%d_q_proj", li);
        layers[li].tq = load_tern(p, &r, &c);
        snprintf(p, sizeof(p), "layer%d_k_proj", li);
        layers[li].tk = load_tern(p, &r, &c);
        snprintf(p, sizeof(p), "layer%d_v_proj", li);
        layers[li].tv = load_tern(p, &r, &c);
        snprintf(p, sizeof(p), "layer%d_o_proj", li);
        layers[li].to = load_tern(p, &r, &c);
        snprintf(p, sizeof(p), "layer%d_gate_proj", li);
        layers[li].tg = load_tern(p, &r, &c);
        snprintf(p, sizeof(p), "layer%d_up_proj", li);
        layers[li].tu = load_tern(p, &r, &c);
        snprintf(p, sizeof(p), "layer%d_down_proj", li);
        layers[li].td = load_tern(p, &r, &c);
        char ln1n[128]; snprintf(ln1n, sizeof(ln1n), "layer%d_input_layernorm.bin", li);
        char ln2n[128]; snprintf(ln2n, sizeof(ln2n), "layer%d_post_attention_layernorm.bin", li);
        layers[li].ln1 = load_vec(ln1n, &r); layers[li].ln2 = load_vec(ln2n, &r);
        layers[li].sq = SCALES[li][0]; layers[li].sk = SCALES[li][1];
        layers[li].sv = SCALES[li][2]; layers[li].so = SCALES[li][3];
        layers[li].sg = SCALES[li][4]; layers[li].su = SCALES[li][5];
        layers[li].sd = SCALES[li][6];
    }

    float* x     = (float*)_aligned_malloc(HIDDEN*4, 32);
    float* scr   = (float*)_aligned_malloc(262144*4, 32);
    int8_t* ai8  = (int8_t*)_aligned_malloc(INTERMEDIATE, 32);
    int* bi      = (int*)_aligned_malloc(INTERMEDIATE*4, 32);
    float* gate  = (float*)_aligned_malloc(INTERMEDIATE*4, 32);
    float* up    = (float*)_aligned_malloc(INTERMEDIATE*4, 32);
    float* logits= (float*)_aligned_malloc(VOCAB*4, 32);
    float* k_cache = (float*)_aligned_malloc(N_KV_HEADS * MAX_SEQ * HEAD_DIM * 4, 32);
    float* v_cache = (float*)_aligned_malloc(N_KV_HEADS * MAX_SEQ * HEAD_DIM * 4, 32);

    clock_t t1 = clock();
    fprintf(stderr, "Load time: %.2f sec\n", (double)(t1-t0)/CLOCKS_PER_SEC);
    fprintf(stderr, "Ready.\n");

    while (1) {
        int prompt_len;
        if (fread(&prompt_len, 4, 1, stdin) != 1) break;
        if (prompt_len <= 0 || prompt_len >= MAX_SEQ) break;

        memset(k_cache, 0, N_KV_HEADS * MAX_SEQ * HEAD_DIM * 4);
        memset(v_cache, 0, N_KV_HEADS * MAX_SEQ * HEAD_DIM * 4);

        // Phase 1: Prompt prefill (build KV cache, no sampling)
        for (int pos = 0; pos < prompt_len; pos++) {
            int tid;
            if (fread(&tid, 4, 1, stdin) != 1) goto cleanup;
            memcpy(x, &embed[tid * HIDDEN], HIDDEN * 4);
            for (int li = 0; li < N_LAYERS; li++)
                forward_layer(&layers[li], li, x, scr, ai8, bi, gate, up, k_cache, v_cache, pos, MAX_SEQ);
        }

        // Sample first generated token from last prompt position
        rms_norm(x, model_norm, HIDDEN);
        mv_fp32(lm_head, x, logits, VOCAB, HIDDEN);
        int sampled = sample(logits, VOCAB, 0.7f, 40);
        fwrite(&sampled, 4, 1, stdout);
        fflush(stdout);

        // Phase 2: Autoregressive generation loop
        for (int pos = prompt_len; pos < MAX_SEQ; pos++) {
            if (sampled == EOS_ID) break;
            int next_tok;
            if (fread(&next_tok, 4, 1, stdin) != 1) goto cleanup;
            if (next_tok == EOS_ID) break;

            memcpy(x, &embed[next_tok * HIDDEN], HIDDEN * 4);
            for (int li = 0; li < N_LAYERS; li++)
                forward_layer(&layers[li], li, x, scr, ai8, bi, gate, up, k_cache, v_cache, pos, MAX_SEQ);

            rms_norm(x, model_norm, HIDDEN);
            mv_fp32(lm_head, x, logits, VOCAB, HIDDEN);
            sampled = sample(logits, VOCAB, 0.7f, 40);
            fwrite(&sampled, 4, 1, stdout);
            fflush(stdout);
        }
    }

cleanup:
    _aligned_free(embed); _aligned_free(lm_head); _aligned_free(model_norm);
    _aligned_free(x); _aligned_free(scr); _aligned_free(ai8);
    _aligned_free(bi); _aligned_free(gate); _aligned_free(up);
    _aligned_free(logits); _aligned_free(k_cache); _aligned_free(v_cache);
    for (int li = 0; li < N_LAYERS; li++) {
        _aligned_free(layers[li].tq); _aligned_free(layers[li].tk);
        _aligned_free(layers[li].tv); _aligned_free(layers[li].to);
        _aligned_free(layers[li].tg); _aligned_free(layers[li].tu);
        _aligned_free(layers[li].td);
        _aligned_free(layers[li].ln1); _aligned_free(layers[li].ln2);
    }
    _aligned_free(layers);
    return 0;
}
