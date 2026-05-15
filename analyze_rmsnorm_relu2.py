"""
Deep dive: RMSNorm weights and relu2 activation stats across all 30 layers
"""
import struct, time, warnings, math
warnings.filterwarnings('ignore')
import numpy as np

WDIR = r'C:\dam\atlas\bitnet_tq10'
HIDDEN, N_HEADS, N_KV_HEADS, HEAD_DIM = 2560, 20, 5, 128
NH_HD, NKV_HD = N_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM
INTERMEDIATE, N_LAYERS, MAX_SEQ, VOCAB = 6912, 30, 4096, 128256
EPS = 1e-5

def load_mat(p):
    a = np.fromfile(p, dtype=np.int32, count=2)
    return np.fromfile(p, dtype=np.float32, offset=8).reshape(a[0], a[1])

def load_vec(p):
    a = np.fromfile(p, dtype=np.int32, count=2)
    return np.fromfile(p, dtype=np.float32, offset=8)

def load_tq10(p):
    with open(p, 'rb') as f:
        r, c, gs, ng = struct.unpack('iiii', f.read(16))
        scales = np.frombuffer(f.read(4*ng), dtype=np.float32)
        data = np.frombuffer(f.read(r*((c+3)//4)), dtype=np.uint8).reshape(r, (c+3)//4)
    return data, scales, gs, c

def unpack_full(data, cols):
    rows = data.shape[0]; pc = (cols + 3) // 4; d = data[:, :pc]
    v0 = (d >> 0) & 3; v1 = (d >> 2) & 3; v2 = (d >> 4) & 3; v3 = (d >> 6) & 3
    w = np.empty((rows, 4 * pc), dtype=np.int8)
    w[:, 0::4] = v0; w[:, 1::4] = v1; w[:, 2::4] = v2; w[:, 3::4] = v3
    w = np.where(w == 2, -1, np.where(w == 1, 1, 0)).astype(np.int8)
    return w[:, :cols]

def quantize_py(x):
    ma = max(float(np.max(np.abs(x))), 1e-10)
    q = ma / 127.0
    return np.round(x / q).clip(-128, 127).astype(np.int8), q

def matmul_tq10(tq, x_q, qq):
    w, scales, gs, cols = tq
    w = unpack_full(w, cols)
    out = np.zeros(w.shape[0], dtype=np.float32)
    for r in range(w.shape[0]):
        dot = int(np.dot(w[r].astype(np.int32), x_q.astype(np.int32)))
        gi = min(r // gs, len(scales) - 1)
        out[r] = dot * scales[gi] * qq
    return out

def rms_norm(x, w):
    ss = np.sum(x.astype(np.float64) ** 2)
    inv = 1.0 / math.sqrt(ss / len(x) + EPS)
    return (x * inv * w).astype(np.float32)

def relu2(x):
    return np.where(x > 0, x * x, 0.0)

half = HEAD_DIM // 2; theta = 500000.0
cos_t = np.zeros((MAX_SEQ, HEAD_DIM), dtype=np.float32)
sin_t = np.zeros((MAX_SEQ, HEAD_DIM), dtype=np.float32)
for pos in range(MAX_SEQ):
    for i in range(half):
        inv = 1.0 / (theta ** ((2 * i) / HEAD_DIM))
        v = pos * inv
        cos_t[pos, i] = cos_t[pos, i + half] = np.float32(math.cos(v))
        sin_t[pos, i] = sin_t[pos, i + half] = np.float32(math.sin(v))

def apply_rope(v, n_heads, pos):
    out = v.copy()
    for h in range(n_heads):
        hv = out[h * HEAD_DIM:(h + 1) * HEAD_DIM]
        for i in range(half):
            c, s = float(cos_t[pos, i]), float(sin_t[pos, i])
            a, b = float(hv[i]), float(hv[i + half])
            hv[i] = a * c - b * s
            hv[i + half] = a * s + b * c
    return out

def attention_gqa(q, k_cache, v_cache, seq_len):
    g = N_HEADS // N_KV_HEADS; out = np.zeros(NH_HD, dtype=np.float32)
    for kv in range(N_KV_HEADS):
        for qh in range(g):
            qv = q[(kv * g + qh) * HEAD_DIM:(kv * g + qh + 1) * HEAD_DIM]
            scores = np.dot(k_cache[kv, :seq_len, :], qv) / math.sqrt(HEAD_DIM)
            scores = np.exp(scores - np.max(scores)); scores /= np.sum(scores)
            out[(kv * g + qh) * HEAD_DIM:(kv * g + qh + 1) * HEAD_DIM] = np.dot(scores, v_cache[kv, :seq_len, :])
    return out


# ========================================================================
print("Loading weights...")
t0 = time.time()
embed = load_mat(f'{WDIR}/embed.bin')
final_norm = load_vec(f'{WDIR}/final_norm.bin')
layers = []
for li in range(N_LAYERS):
    def ltq(n):
        return load_tq10(f'{WDIR}/l{li}_{n}.tq10')
    def lvec(n):
        return load_vec(f'{WDIR}/l{li}_{n}.bin')
    layers.append({
        'tq': ltq('q_proj'), 'tk': ltq('k_proj'), 'tv': ltq('v_proj'),
        'to': ltq('o_proj'), 'tg': ltq('gate_proj'), 'tu': ltq('up_proj'),
        'td': ltq('down_proj'),
        'ln1': lvec('input_layernorm'), 'ln2': lvec('post_attention_layernorm'),
        'attn_sub': lvec('attn_sub_norm'), 'ffn_sub': lvec('ffn_sub_norm'),
    })
print(f"Loaded in {time.time()-t0:.1f}s")

# ========================================================================
# RMSNorm weight analysis
# ========================================================================
print("\n" + "=" * 70)
print("RMSNorm Weight Magnitudes Over 30 Layers")
print("=" * 70)
print(f"{'Layer':>5} {'ln1_mean':>10} {'ln1_std':>10} {'ln1_max':>10} {'ln2_mean':>10} {'ln2_std':>10} {'ln2_max':>10} "
      f"{'attn_sub_mean':>14} {'attn_sub_max':>13} {'ffn_sub_mean':>14} {'ffn_sub_max':>13}")
print("-" * 120)

for li in range(N_LAYERS):
    l = layers[li]
    ln1 = l['ln1']; ln2 = l['ln2']; attn = l['attn_sub']; ffn = l['ffn_sub']
    print(f"  {li:2d}  {np.mean(ln1):10.4f} {np.std(ln1):10.4f} {np.max(np.abs(ln1)):10.4f} "
          f"{np.mean(ln2):10.4f} {np.std(ln2):10.4f} {np.max(np.abs(ln2)):10.4f} "
          f"{np.mean(attn):14.4f} {np.max(np.abs(attn)):13.4f} "
          f"{np.mean(ffn):14.4f} {np.max(np.abs(ffn)):13.4f}")

# RMSNorm overflow tracking: % dims where w > 5, > 10
print("\nRMSNorm weight % outlier dims:")
print(f"{'Layer':>5} {'ln1>5':>8} {'ln1>10':>8} {'ln2>5':>8} {'ln2>10':>8} {'attn>5':>8} {'attn>10':>8} {'ffn>5':>8} {'ffn>10':>8}")
print("-" * 70)
for li in range(N_LAYERS):
    l = layers[li]
    ln1 = l['ln1']; ln2 = l['ln2']; attn = l['attn_sub']; ffn = l['ffn_sub']
    print(f"  {li:2d}  {100*np.mean(np.abs(ln1)>5):7.1f}% {100*np.mean(np.abs(ln1)>10):7.1f}% "
          f"{100*np.mean(np.abs(ln2)>5):7.1f}% {100*np.mean(np.abs(ln2)>10):7.1f}% "
          f"{100*np.mean(np.abs(attn)>5):7.1f}% {100*np.mean(np.abs(attn)>10):7.1f}% "
          f"{100*np.mean(np.abs(ffn)>5):7.1f}% {100*np.mean(np.abs(ffn)>10):7.1f}%")

# ========================================================================
# relu2 activation stats during forward pass
# ========================================================================
print("\n" + "=" * 70)
print("relu2 Activation Stats Per Layer (pre-relu2 and post-relu2)")
print("=" * 70)

from transformers import AutoTokenizer
prompt = "What is the capital of France?"
tok = AutoTokenizer.from_pretrained(r'C:\dam\models\bitnet-b1.58-2B-4T', fix_mistral_regex=True)
ids = tok.encode(prompt)
if ids[0] != 128000: ids = [128000] + ids
x_base = embed[ids[0]].copy().astype(np.float32)

k_cache_py = np.zeros((N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32)
v_cache_py = np.zeros((N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32)

print(f"\n{'Layer':>5} {'gate_pre_std':>13} {'gate_pre_max':>13} {'gate_pre_min':>13} {'relu2_sparsity':>15} "
      f"{'relu2_max':>10} {'gate_post_max':>14} {'quant_clip%':>11}")
print("-" * 95)

xp = x_base.copy()
for li in range(N_LAYERS):
    l = layers[li]
    res = xp.copy()
    xp = rms_norm(xp, l['ln1'])
    x_q, qq = quantize_py(xp)

    q = matmul_tq10(l['tq'], x_q, qq)
    k = matmul_tq10(l['tk'], x_q, qq)
    v = matmul_tq10(l['tv'], x_q, qq)
    q = apply_rope(q, N_HEADS, 0)
    k = apply_rope(k, N_KV_HEADS, 0)
    for kvi in range(N_KV_HEADS):
        k_cache_py[kvi, 0] = k[kvi*HEAD_DIM:(kvi+1)*HEAD_DIM]
        v_cache_py[kvi, 0] = v[kvi*HEAD_DIM:(kvi+1)*HEAD_DIM]
    attn = attention_gqa(q, k_cache_py, v_cache_py, 1)
    attn_q, qo = quantize_py(attn)
    o = matmul_tq10(l['to'], attn_q, qo)
    o = rms_norm(o, l['attn_sub'])
    xp = res + o

    res = xp.copy()
    xp = rms_norm(xp, l['ln2'])
    x_q, qg = quantize_py(xp)

    gate_pre = matmul_tq10(l['tg'], x_q, qg)
    up = matmul_tq10(l['tu'], x_q, qg)

    # relu2 stats
    gate_std = float(np.std(gate_pre))
    gate_max = float(np.max(gate_pre))
    gate_min = float(np.min(gate_pre))
    sparsity = 100 * float(np.mean(gate_pre <= 0))
    gate_r2 = relu2(gate_pre)
    gate_post = gate_r2 * up
    r2_max = float(np.max(gate_r2))
    gp_max = float(np.max(gate_post))
    # quant clipping if any
    q_img, qd = quantize_py(gate_post)
    clip_pct = 100 * float(np.mean(np.abs(q_img) >= 127))

    print(f"  {li:2d}  {gate_std:13.3f} {gate_max:13.3f} {gate_min:13.3f} {sparsity:14.1f}% "
          f"{r2_max:10.3f} {gp_max:14.3f} {clip_pct:10.2f}%")

    gate = relu2(gate_pre) * up
    gate = rms_norm(gate, l['ffn_sub'])
    gate_q, qd = quantize_py(gate)
    d = matmul_tq10(l['td'], gate_q, qd)
    xp = res + d

print("\nDone.")
