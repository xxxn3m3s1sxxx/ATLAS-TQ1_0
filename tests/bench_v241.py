#!/usr/bin/env python3
"""v2.4.1 benchmark: all available models, modes."""
import os, sys, time, gc
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from atlas_infer import AtlasModel

MODELS = [
    ('Falcon3-3B',  'falcon3-3b-tq1.atlas',  False),
    ('Bonsai-1.7B', 'bonsai-1.7b-tq1-g128.atlas', True),
    ('Bonsai-4B',   'bonsai-4b-tq1-g128.atlas',  False),
]

PROMPT = "The capital of France is"
N_TOKENS = 30

results = {}
for name, path, small in MODELS:
    if not os.path.exists(path):
        print(f"  [{name}] SKIP — not found"); continue
    print(f"\n--- {name} ---")

    # Default config (hybrid for large, f32 bypass for small)
    t0 = time.time(); m = AtlasModel(path)
    load_t = time.time() - t0
    m.generate_c(PROMPT, max_new_tokens=5, temperature=0.0); gc.collect()
    t0 = time.time()
    out = m.generate_c(PROMPT, max_new_tokens=N_TOKENS, temperature=0.0)
    d_elapsed = time.time() - t0
    d_tok_s = N_TOKENS / d_elapsed
    q = out.replace('\n',' ').strip()[:90]
    mode = "f32-bypass" if small else "hybrid+int8"
    print(f"  default ({mode:12s}): {d_tok_s:.2f} tok/s ({d_elapsed:.1f}s)  load={load_t:.1f}s  q: {q}")
    results[name] = {'default_tok_s': d_tok_s, 'load_s': load_t, 'quality': q, 'mode': mode}
    del m; gc.collect()

    # Non-f32 comparison for small models
    if small:
        t0 = time.time(); m = AtlasModel(path)
        load_t2 = time.time() - t0
        m.set_use_f32_matmul(False)
        m.generate_c(PROMPT, max_new_tokens=5, temperature=0.0); gc.collect()
        t0 = time.time()
        out = m.generate_c(PROMPT, max_new_tokens=N_TOKENS, temperature=0.0)
        nf_elapsed = time.time() - t0
        nf_tok_s = N_TOKENS / nf_elapsed
        print(f"  hybrid+int8            : {nf_tok_s:.2f} tok/s ({nf_elapsed:.1f}s)  load={load_t2:.1f}s")
        results[name]['hybrid_tok_s'] = nf_tok_s
        del m; gc.collect()

    # Packed-only comparison for large models
    if not small:
        t0 = time.time(); m = AtlasModel(path)
        load_t3 = time.time() - t0
        m.set_use_packed_matmul(True)
        m.generate_c(PROMPT, max_new_tokens=5, temperature=0.0); gc.collect()
        t0 = time.time()
        out = m.generate_c(PROMPT, max_new_tokens=N_TOKENS, temperature=0.0)
        p_elapsed = time.time() - t0
        p_tok_s = N_TOKENS / p_elapsed
        print(f"  packed-only            : {p_tok_s:.2f} tok/s ({p_elapsed:.1f}s)")
        results[name]['packed_tok_s'] = p_tok_s
        del m; gc.collect()

print("\n\n=== v2.4.1 BENCHMARKS ===")
for name, d in results.items():
    def_mode = d['mode']
    def_ts = d['default_tok_s']
    hyb = d.get('hybrid_tok_s', 0)
    pck = d.get('packed_tok_s', 0)
    extra = ""
    if hyb: extra += f" | hybrid+int8: {hyb:.2f}"
    if pck: extra += f" | packed-only: {pck:.2f}"
    print(f"  {name:15s} {def_mode}: {def_ts:.2f} tok/s{extra}")
    print(f"    q: {d['quality']}")
