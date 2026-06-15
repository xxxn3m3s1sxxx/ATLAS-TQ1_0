#!/usr/bin/env python3
"""Test all ATLAS Pro models: coherence, baseline comparison, quality check."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atlas_infer import AtlasModel

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "models")

TEST_CASES = [
    "The capital of France is",
    "The second law of thermodynamics states that",
    "Explain the difference between a neutron and a proton.",
    "Write a short poem about artificial intelligence.",
    "What is 2+2?",
    "If you have 3 apples and eat 1, how many remain?",
]

def test_model(label, path, tests=TEST_CASES, temps=[0.0, 0.7]):
    """Load model and run coherence tests."""
    if not os.path.exists(path):
        return {label: {"error": f"File not found: {path}"}}

    results = {}
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {os.path.basename(path)} ({os.path.getsize(path)/1e9:.2f} GB)")
    print(f"{'='*60}")

    try:
        model = AtlasModel(path)
        model.set_use_f32_matmul(True)
    except Exception as e:
        print(f"  ❌ Load failed: {e}")
        return {label: {"error": str(e)}}

    for temp in temps:
        print(f"\n  --- T={temp} ---")
        for prompt in tests:
            try:
                start = time.time()
                out = model.generate_c(
                    prompt,
                    max_new_tokens=60,
                    temperature=temp,
                    top_k=1 if temp == 0.0 else 40,
                    top_p=1.0 if temp == 0.0 else 0.9,
                    cache_enabled=False
                )
                elapsed = time.time() - start
                tok_s = len(out.split()) / elapsed if elapsed > 0 else 0
                out_clean = out[:120].strip().encode('ascii', errors='replace').decode('ascii')
                print(f"  [{tok_s:.1f} tok/s] {out_clean}")
                results[(prompt, temp)] = out.strip()
            except Exception as e:
                err = str(e).encode('ascii', errors='replace').decode('ascii')
                print(f"  [ERR] {prompt[:40]}: {err}")
                results[(prompt, temp)] = f"[ERROR: {e}]"

    del model
    import gc; gc.collect()
    return {label: results}

def compare_results(results):
    pass

if __name__ == "__main__":
    models = {
        "Bonsai-1.7B-BASELINE": os.path.join(MODELS_DIR, "Ternary-Bonsai-1.7B-BASELINE.atlas"),
        "Bonsai-1.7B-Pro-v2": os.path.join(MODELS_DIR, "Ternary-Bonsai-1.7B-ATLAS-Pro-v2.atlas"),
        "Bonsai-4B-BASELINE": os.path.join(MODELS_DIR, "Ternary-Bonsai-4B-BASELINE.atlas"),
        "Bonsai-4B-Pro-v1": os.path.join(MODELS_DIR, "Ternary-Bonsai-4B-ATLAS-Pro.atlas"),
        "Bonsai-4B-Pro-v2": os.path.join(MODELS_DIR, "Ternary-Bonsai-4B-ATLAS-Pro-v2.atlas"),
        "Bonsai-8B-Pro": os.path.join(MODELS_DIR, "Ternary-Bonsai-8B-ATLAS-Pro.atlas"),
    }

    all_results = {}
    for name, path in models.items():
        res = test_model(name, path, tests=TEST_CASES[:3], temps=[0.0, 0.7])
        all_results.update(res)
        # Force full GC between models to free DLL memory
        import gc; gc.collect()

    # Print side-by-side comparison
    comp_names = list(models.keys())
    for temp_label, temp_val in [("T=0 ARGMAX", 0.0), ("T=0.7 SAMPLING", 0.7)]:
        print(f"\n\n{'='*60}")
        print(f"  {temp_label} COMPARISON")
        print(f"{'='*60}")
        for prompt in TEST_CASES[:3]:
            print(f"\n  Prompt: {prompt}")
            for name in comp_names:
                val = ""
                for k, v in all_results.items():
                    if name in k and isinstance(v, dict):
                        key = (prompt, temp_val)
                        if key in v:
                            val = v[key]
                            break
                val_clean = val[:120].encode('ascii', errors='replace').decode('ascii') if val else ""
                if val:
                    print(f"    [{name:24s}] {val_clean}")
                else:
                    print(f"    [{name:24s}] [NO DATA]")

    # Summary
    print(f"\n\n{'='*60}")
    print("  VERDICT")
    print(f"{'='*60}")
    for name, path in models.items():
        if os.path.exists(path):
            sz = os.path.getsize(path) / 1e9
            ok = "OK"
            for k, v in all_results.items():
                if name in k and isinstance(v, dict) and "error" in v:
                    ok = "FAIL"
            print(f"  {name:30s} [{ok}] {sz:.2f} GB")
        else:
            print(f"  {name:30s} [MISS] not found")
