"""
Quick scale analysis for layers 25-30
"""
import struct, numpy as np

WDIR = r'C:\dam\atlas\bitnet_tq10'
N_LAYERS = 30

def load_tq10(p):
    with open(p, 'rb') as f:
        r, c, gs, ng = struct.unpack('iiii', f.read(16))
        scales = np.frombuffer(f.read(4*ng), dtype=np.float32)
        data = np.frombuffer(f.read(r*((c+3)//4)), dtype=np.uint8).reshape(r, (c+3)//4)
    return data, scales, gs, c

# Load all layers
layers = []
for li in range(N_LAYERS):
    def ltq(n):
        return load_tq10(f'{WDIR}/l{li}_{n}.tq10')
    layers.append({
        'tq': ltq('q_proj'), 'tk': ltq('k_proj'), 'tv': ltq('v_proj'),
        'to': ltq('o_proj'), 'tg': ltq('gate_proj'), 'tu': ltq('up_proj'),
        'td': ltq('down_proj'),
    })

print("=" * 70)
print("ANALYSIS 3: TQ1.0 Scale Factors in Layers 25-30")
print("=" * 70)

for li in range(25, 30):
    print(f"\n  Layer {li}:")
    for key, name in [('tq', 'q_proj'), ('tk', 'k_proj'), ('tv', 'v_proj'), ('to', 'o_proj'),
                       ('tg', 'gate_proj'), ('tu', 'up_proj'), ('td', 'down_proj')]:
        tq = layers[li][key]
        scales = tq[1]
        s_min = float(np.min(scales))
        s_max = float(np.max(scales))
        s_mean = float(np.mean(scales))
        near_zero = np.any(np.abs(scales) < 1e-30)
        near_overflow = np.any(np.abs(scales) > 1e30)
        flags = []
        if near_zero: flags.append(f"NEAR_ZERO(min={s_min:.2e})")
        if near_overflow: flags.append(f"NEAR_OVERFLOW(max={s_max:.2e})")
        flag_str = " *** " + " ".join(flags) if flags else ""
        print(f"    {name:12s} min={s_min:14.6f} max={s_max:14.6f} mean={s_mean:14.6f}  n_groups={len(scales)}{flag_str}")

# Also check scale magnitude across all layers
print("\n\n  Scale magnitude vs layer depth:")
for key, name in [('tq', 'q_proj'), ('tk', 'k_proj'), ('tv', 'v_proj'), ('to', 'o_proj'),
                   ('tg', 'gate_proj'), ('tu', 'up_proj'), ('td', 'down_proj')]:
    means = []
    for li in range(N_LAYERS):
        means.append(float(np.mean(np.abs(layers[li][key][1]))))
    print(f"    {name:12s}: layer0_mean={means[0]:.3f}  layer14_mean={means[14]:.3f}  layer29_mean={means[29]:.3f}  "
          f"max={np.max(means):.1f}@L{np.argmax(means)}  min={np.min(means):.1f}@L{np.argmin(means)}")

# Check for a pattern: are later-layer scales systematically larger?
print("\n  Mean |scale| per layer (averaged over all 7 weight matrices):")
layer_avg_scale = []
for li in range(N_LAYERS):
    all_scales = []
    for key in ['tq', 'tk', 'tv', 'to', 'tg', 'tu', 'td']:
        all_scales.extend(np.abs(layers[li][key][1]).tolist())
    layer_avg_scale.append(np.mean(all_scales))
for li in range(N_LAYERS):
    bar = '#' * int(layer_avg_scale[li] / max(layer_avg_scale) * 40)
    print(f"    L{li:2d}: {layer_avg_scale[li]:10.3f}  {bar}")

print("\nDone.")
