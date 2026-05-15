"""
Phase A quality analysis:
1. Activation magnitude over 30 layers (where does it explode/implode?)
2. Double-accumulation matmul vs float32 matmul cosine similarity
3. TQ1.0 scale factors in layers 25-30 — float range limits
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

def matmul_tq10_f32(tq, x_q, qq):
    """float32 accumulation — baseline"""
    w, scales, gs, cols = tq
    w = unpack_full(w, cols)
    out = np.zeros(w.shape[0], dtype=np.float32)
    for r in range(w.shape[0]):
        dot = int(np.dot(w[r].astype(np.int32), x_q.astype(np.int32)))
        gi = min(r // gs, len(scales) - 1)
        out[r] = np.dot(w[r].astype(np.int32), x_q.astype(np.int32)) * scales[gi] * qq
    return out

def matmul_tq10_f64(tq, x_q, qq):
    """float64 accumulation — test if internal precision helps"""
    w, scales, gs, cols = tq
    w = unpack_full(w, cols)
    out = np.zeros(w.shape[0], dtype=np.float64)
    w_f64 = w.astype(np.float64)
    x_f64 = x_q.astype(np.float64)
    scales_f64 = scales.astype(np.float64)
    for r in range(w.shape[0]):
        dot = np.dot(w_f64[r], x_f64)
        gi = min(r // gs, len(scales) - 1)
        out[r] = dot * scales_f64[gi] * np.float64(qq)
    return out.astype(np.float32)

def matmul_tq10_f32_full(tq, x_f32):
    """FULL float32 matmul — dequantize weights to [-1,0,1] float32, no int8 quant on activations.
       This is the 'theoretical FP32' reference for this weight set."""
    w, scales, gs, cols = tq
    w = unpack_full(w, cols).astype(np.float32)
    out = np.zeros(w.shape[0], dtype=np.float32)
    for r in range(w.shape[0]):
        gi = min(r // gs, len(scales) - 1)
        out[r] = np.dot(w[r], x_f32) * scales[gi]
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
# Setup
# ========================================================================
print("=" * 70)
print("PHASE A — QUALITY ANALYSIS")
print("=" * 70)

print("\nLoading TQ1.0 weights...")
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

# Use first token of a short prompt
from transformers import AutoTokenizer
prompt = "What is the capital of France?"
tok = AutoTokenizer.from_pretrained(r'C:\dam\models\bitnet-b1.58-2B-4T', fix_mistral_regex=True)
ids = tok.encode(prompt)
if ids[0] != 128000: ids = [128000] + ids

# ========================================================================
# ANALYSIS 1: Activation magnitude across all 30 layers
# ========================================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: Activation Magnitude Over 30 Layers")
print("=" * 70)

x_base = embed[ids[0]].copy().astype(np.float32)

print(f"\n{'Layer':>5} {'hidden_mean':>12} {'hidden_std':>12} {'hidden_max':>12} {'hidden_min':>12} {'%>100':>8} {'%>1000':>8} {'attn_std':>10} {'ffn_std':>10} {'q_max':>8}")
print("-" * 105)

k_cache_py = np.zeros((N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32)
v_cache_py = np.zeros((N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32)

layer_stats = []
xp = x_base.copy()
for li in range(N_LAYERS):
    l = layers[li]
    res = xp.copy()
    xp = rms_norm(xp, l['ln1'])
    x_q, qq = quantize_py(xp)

    q = matmul_tq10_f32(l['tq'], x_q, qq)
    k = matmul_tq10_f32(l['tk'], x_q, qq)
    v = matmul_tq10_f32(l['tv'], x_q, qq)

    q = apply_rope(q, N_HEADS, 0)
    k = apply_rope(k, N_KV_HEADS, 0)
    for kvi in range(N_KV_HEADS):
        k_cache_py[kvi, 0] = k[kvi*HEAD_DIM:(kvi+1)*HEAD_DIM]
        v_cache_py[kvi, 0] = v[kvi*HEAD_DIM:(kvi+1)*HEAD_DIM]

    attn = attention_gqa(q, k_cache_py, v_cache_py, 1)
    attn_q, qo = quantize_py(attn)
    o = matmul_tq10_f32(l['to'], attn_q, qo)
    o = rms_norm(o, l['attn_sub'])
    xp_after_attn = res + o
    attn_std = float(np.std(xp_after_attn))

    res = xp_after_attn.copy()
    xp = rms_norm(xp_after_attn, l['ln2'])
    x_q, qg = quantize_py(xp)
    gate = matmul_tq10_f32(l['tg'], x_q, qg)
    up = matmul_tq10_f32(l['tu'], x_q, qg)
    gate = relu2(gate) * up
    gate = rms_norm(gate, l['ffn_sub'])
    gate_q, qd = quantize_py(gate)
    d = matmul_tq10_f32(l['td'], gate_q, qd)
    xp = res + d
    ffn_std = float(np.std(xp))

    hm = float(np.mean(xp))
    hs = float(np.std(xp))
    hmax = float(np.max(xp))
    hmin = float(np.min(xp))
    pct100 = 100 * np.mean(np.abs(xp) > 100)
    pct1k = 100 * np.mean(np.abs(xp) > 1000)
    q_max_val = float(np.max(np.abs(q)))

    layer_stats.append((hm, hs, hmax, hmin, pct100, pct1k, attn_std, ffn_std, q_max_val))

    # Flag problematic layers
    flag = " <<<" if (pct1k > 1 or hs > 1000 or hmax > 5000) else ""
    print(f"  {li:2d}  {hm:12.3f} {hs:12.3f} {hmax:12.3f} {hmin:12.3f} {pct100:7.1f}% {pct1k:7.1f}% {attn_std:10.3f} {ffn_std:10.3f} {q_max_val:8.1f}{flag}")

# Summary
print("\nKey findings:")
max_std_idx = max(range(N_LAYERS), key=lambda i: layer_stats[i][1])
print(f"  Highest std: Layer {max_std_idx} (std={layer_stats[max_std_idx][1]:.1f})")
max_mean_idx = max(range(N_LAYERS), key=lambda i: abs(layer_stats[i][0]))
print(f"  Farthest from zero mean: Layer {max_mean_idx} (mean={layer_stats[max_mean_idx][0]:.1f})")
exploding = [i for i in range(N_LAYERS) if layer_stats[i][4] > 1 or layer_stats[i][5] > 1]
if exploding:
    print(f"  Layers with >1% extreme values: {exploding}")


# ========================================================================
# ANALYSIS 2: Double-accumulation matmul vs float32 matmul
# ========================================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: Double-Precision Matmul vs Float32 Matmul")
print("=" * 70)
print("Comparing f32-accum matmul vs f64-accum matmul per layer (cosine sim of full hidden state)")

# Run side-by-side with independent caches
xp_f32 = x_base.copy()
xp_f64 = x_base.copy()

cos_sims = []
for li in range(N_LAYERS):
    l = layers[li]

    # Float32 path (independent cache)
    kc32 = np.zeros((N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32)
    vc32 = np.zeros((N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32)

    res = xp_f32.copy()
    xn = rms_norm(xp_f32, l['ln1'])
    x_q, qq = quantize_py(xn)
    q = matmul_tq10_f32(l['tq'], x_q, qq)
    k = matmul_tq10_f32(l['tk'], x_q, qq)
    v = matmul_tq10_f32(l['tv'], x_q, qq)
    q = apply_rope(q, N_HEADS, 0)
    k = apply_rope(k, N_KV_HEADS, 0)
    for kvi in range(N_KV_HEADS):
        kc32[kvi, 0] = k[kvi*HEAD_DIM:(kvi+1)*HEAD_DIM]
        vc32[kvi, 0] = v[kvi*HEAD_DIM:(kvi+1)*HEAD_DIM]
    attn = attention_gqa(q, kc32, vc32, 1)
    attn_q, qo = quantize_py(attn)
    o = matmul_tq10_f32(l['to'], attn_q, qo)
    o = rms_norm(o, l['attn_sub'])
    xp_f32 = res + o
    res = xp_f32.copy()
    xn = rms_norm(xp_f32, l['ln2'])
    x_q, qg = quantize_py(xn)
    gate = matmul_tq10_f32(l['tg'], x_q, qg)
    up = matmul_tq10_f32(l['tu'], x_q, qg)
    gate = relu2(gate) * up
    gate = rms_norm(gate, l['ffn_sub'])
    gate_q, qd = quantize_py(gate)
    d = matmul_tq10_f32(l['td'], gate_q, qd)
    xp_f32 = res + d

    # Float64 path (independent cache)
    kc64 = np.zeros((N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32)
    vc64 = np.zeros((N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32)

    res = xp_f64.copy()
    xn = rms_norm(xp_f64, l['ln1'])
    x_q, qq = quantize_py(xn)
    q = matmul_tq10_f64(l['tq'], x_q, qq)
    k = matmul_tq10_f64(l['tk'], x_q, qq)
    v = matmul_tq10_f64(l['tv'], x_q, qq)
    q = apply_rope(q, N_HEADS, 0)
    k = apply_rope(k, N_KV_HEADS, 0)
    for kvi in range(N_KV_HEADS):
        kc64[kvi, 0] = k[kvi*HEAD_DIM:(kvi+1)*HEAD_DIM]
        vc64[kvi, 0] = v[kvi*HEAD_DIM:(kvi+1)*HEAD_DIM]
    attn = attention_gqa(q, kc64, vc64, 1)
    attn_q, qo = quantize_py(attn)
    o = matmul_tq10_f64(l['to'], attn_q, qo)
    o = rms_norm(o, l['attn_sub'])
    xp_f64 = res + o
    res = xp_f64.copy()
    xn = rms_norm(xp_f64, l['ln2'])
    x_q, qg = quantize_py(xn)
    gate = matmul_tq10_f64(l['tg'], x_q, qg)
    up = matmul_tq10_f64(l['tu'], x_q, qg)
    gate = relu2(gate) * up
    gate = rms_norm(gate, l['ffn_sub'])
    gate_q, qd = quantize_py(gate)
    d = matmul_tq10_f64(l['td'], gate_q, qd)
    xp_f64 = res + d

    cs = float(np.dot(xp_f32, xp_f64) / (np.linalg.norm(xp_f32) * np.linalg.norm(xp_f64)))
    cos_sims.append(cs)
    diff_std = float(np.std(xp_f32 - xp_f64))
    flag = " <<< diff" if diff_std > 1e-3 else ""
    print(f"  L{li:2d}  cos(f32,f64)={cs:.10f}  |diff|_std={diff_std:.2e}{flag}")

avg_cs = np.mean(cos_sims)
min_cs = np.min(cos_sims)
print(f"\n  Average cosine: {avg_cs:.10f}")
print(f"  Minimum cosine: {min_cs:.10f}")
if min_cs > 0.999999:
    print("  => Double accumulation changes NOTHING significant. f32 matmul is already exact enough.")
else:
    print("  => Double accumulation changes results meaningfully. Investigate further.")


# ========================================================================
# ANALYSIS 3: TQ1.0 Scale Factors in Layers 25-30
# ========================================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: TQ1.0 Scale Factors in Layers 25-30")
print("=" * 70)

for li in range(25, 31):
    print(f"\n  Layer {li}:")
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        tq = layers[li][{'q': 'tq', 'k': 'tk', 'v': 'tv', 'o': 'to', 'gate': 'tg', 'up': 'tu', 'down': 'td'}[name.split('_')[0]]]
        scales = tq[1]  # scales array
        s_min = float(np.min(scales))
        s_max = float(np.max(scales))
        s_mean = float(np.mean(scales))
        is_denorm = np.any(np.abs(scales) < 1.17e-38)
        is_inf = np.any(np.isinf(scales))
        is_nan = np.any(np.isnan(scales))
        near_zero = np.any(np.abs(scales) < 1e-30)
        near_overflow = np.any(np.abs(scales) > 1e30)
        flags = []
        if is_denorm or near_zero: flags.append(f"DENORM(min={s_min:.2e})")
        if is_inf: flags.append("INF")
        if is_nan: flags.append("NAN")
        if near_overflow: flags.append(f"NEAR_OVERFLOW(max={s_max:.2e})")
        flag_str = " *** " + " ".join(flags) if flags else ""
        print(f"    {name:12s} min={s_min:12.3f} max={s_max:12.3f} mean={s_mean:12.3f}  n_groups={len(scales)}{flag_str}")


# Summary of scale factors across all layers (compact)
print("\n  Summary across ALL layers (max |scale| per weight):")
all_max = []
for li in range(N_LAYERS):
    for key, name in [('tq', 'q'), ('tk', 'k'), ('tv', 'v'), ('to', 'o'), ('tg', 'g'), ('tu', 'u'), ('td', 'd')]:
        scales = layers[li][key][1]
        all_max.append(float(np.max(np.abs(scales))))
all_max = np.array(all_max)
print(f"    Global max |scale|: {np.max(all_max):.3f}")
print(f"    Global min |scale| (nonzero): {np.min(all_max[all_max > 0]):.3e}")
print(f"    # near overflow (>1e30): {np.sum(all_max > 1e30)}")
print(f"    # near denorm (<1e-30): {np.sum(all_max < 1e-30)}")

# Check if there's a pattern in scale accumulation
print("\n  Per-weight average |scale| across layers:")
for key, name in [('tq', 'q_proj'), ('tk', 'k_proj'), ('tv', 'v_proj'), ('to', 'o_proj'),
                   ('tg', 'gate_proj'), ('tu', 'up_proj'), ('td', 'down_proj')]:
    vals = [float(np.mean(np.abs(layers[li][key][1]))) for li in range(N_LAYERS)]
    print(f"    {name:12s}: mean={np.mean(vals):.2f}  max={np.max(vals):.2f} (layer {np.argmax(vals)})  min={np.min(vals):.2f} (layer {np.argmin(vals)})")

print("\nDone.")
