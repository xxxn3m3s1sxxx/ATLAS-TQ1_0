"""Compare TriLM-99M vs Pro with optimal config."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from atlas_infer import AtlasModel

MODELS = {
    "baseline": r"C:\atlas\models\TriLM-99M-ATLAS.atlas",
    "pro":      r"C:\atlas\models\TriLM-99M-ATLAS-Pro.atlas",
}

PROMPTS = [
    "The capital of France is",
    "Germany's capital city is",
    "The Eiffel Tower is located in",
    "The best thing about traveling to Paris is",
    "London is the capital of",
]

def run(label, model_path, **kwargs):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    try:
        m = AtlasModel(model_path)
        m.set_num_threads(4)
        m.set_use_f32_matmul(True)
        for prompt in PROMPTS:
            out = m.generate_c(prompt, max_new_tokens=50, **kwargs)
            out_clean = out.replace('\n', '\\n')
            print(f"  '{prompt}'")
            print(f"    -> {out_clean[:150]}")
        del m
    except Exception as e:
        print(f"  ERROR: {e}")

# Best config from diagnosis
cfg = dict(temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.3)

print("TESTING: T=0.7, top_k=40, rep_penalty=1.3, f32_bypass=True")
for name, path in MODELS.items():
    run(name, path, **cfg)
