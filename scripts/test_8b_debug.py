#!/usr/bin/env python3
import sys, os, ctypes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atlas_infer import AtlasModel, dll

d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "models")
cache = os.path.expanduser("~/.cache/huggingface/hub/models--xxxn3m3s1sxxx--Ternary-Bonsai-8B-ATLAS/snapshots")

paths = []
# Pro
pro = os.path.join(d, "Ternary-Bonsai-8B-ATLAS-Pro.atlas")
if os.path.exists(pro):
    paths.append(("8B Pro", pro))
# Baseline
for snap in os.listdir(cache):
    base = os.path.join(cache, snap, "Ternary-Bonsai-8B-ATLAS.tq1.atlas")
    if os.path.exists(base):
        paths.append(("8B Baseline", base))
        break

for name, path in paths:
    print(f"\n=== {name} ===")
    print(f"  Path: {path}")
    m = AtlasModel(path)
    
    # Check model properties
    props = [
        "n_layers", "hidden", "inter", "n_heads", "n_kv_heads",
        "vocab_size", "head_dim", "n_tensors", "rope_theta",
        "_needs_f32_bypass", "_is_qwen3",
        "max_seq_len"
    ]
    for p in props:
        val = getattr(m, p, "N/A")
        print(f"  {p}: {val}")
    
    # Test with f32_bypass
    m.set_use_f32_matmul(True)
    out = m.generate_c("The capital of France is", max_new_tokens=30,
                       temperature=0.0, top_k=1, top_p=1.0, cache_enabled=False)
    clean = out[:80].strip().encode("utf-8", errors="replace").decode("utf-8")
    print(f"  f32_bypass: {clean}")
    
    # Test with hybrid
    m.set_use_f32_matmul(False)
    m.set_use_hybrid_matmul(True)
    out = m.generate_c("The capital of France is", max_new_tokens=30,
                       temperature=0.0, top_k=1, top_p=1.0, cache_enabled=False)
    clean = out[:80].strip().encode("utf-8", errors="replace").decode("utf-8")
    print(f"  hybrid: {clean}")
    
    del m
    import gc; gc.collect()
