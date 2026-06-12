"""Benchmark: measure tok/s for an atlas model."""
import os, sys, time, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, 'C:/atlas')
from atlas_infer import AtlasModel

def benchmark(model_path, label, force_tq1=False, max_tokens=200, **kwargs):
    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"{'='*60}")
    
    t0 = time.time()
    model = AtlasModel(model_path, force_tq1_native=force_tq1, **kwargs)
    load_time = time.time() - t0
    print(f"Load time: {load_time:.1f}s")
    
    prompt = "What is the capital of France?"
    
    # Warm up
    _ = model.generate_c(prompt, max_new_tokens=5, temperature=0.7, top_k=40)
    
    # Benchmark
    t0 = time.time()
    out = model.generate_c(prompt, max_new_tokens=max_tokens, temperature=0.7, top_k=40)
    elapsed = time.time() - t0
    
    total_tokens = max_tokens + len(model._cpp_encode(prompt))
    tok_s = total_tokens / elapsed
    
    print(f"Output ({len(out)} chars): {out[:100]}...")
    print(f"Time: {elapsed:.1f}s for {max_tokens} gen tokens")
    print(f"Total tok/s: {tok_s:.1f}")
    
    result = {
        "model": label,
        "layers": model.n_layers,
        "hidden": model.hidden,
        "inter": model.inter,
        "load_time_s": round(load_time, 1),
        "total_tok_s": round(tok_s, 1),
        "gen_time_s": round(elapsed, 1),
        "gen_tokens": max_tokens,
        "output_preview": out[:80],
    }
    
    del model
    import gc; gc.collect()
    return result

results = []

# Test models we have on disk
models = [
    ("C:/atlas/models/falcon3-1B-Instruct-tq1.atlas", "Falcon3-1B", {}),
    ("C:/atlas/models/falcon3-3B-Instruct-tq1.atlas", "Falcon3-3B", {}),
]

for path, label, kwargs in models:
    if os.path.exists(path):
        r = benchmark(path, label, **kwargs)
        results.append(r)

# Bonsai-4B: LUT vs baseline
bonsai_path = "C:/atlas/models/Ternary-Bonsai-4B-ATLAS.tq1.atlas"
if os.path.exists(bonsai_path):
    r1 = benchmark(bonsai_path, "Bonsai-4B (LUT)", force_tq1=True)
    results.append(r1)
    
    r2 = benchmark(bonsai_path, "Bonsai-4B (f32_bypass)", max_tokens=200)
    results.append(r2)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for r in results:
    print(f"{r['model']:30s} {r['total_tok_s']:5.1f} tok/s  load={r['load_time_s']:.0f}s  gen={r['gen_time_s']:.0f}s")

import json
with open("C:/atlas/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to benchmark_results.json")
