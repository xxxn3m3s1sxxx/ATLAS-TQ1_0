"""Quick regression test — all models + reset_cache."""
import sys, os, time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, 'C:\\atlas')
from atlas_infer import AtlasModel

models = [
    ('Falcon3-3B',   r'C:\atlas\falcon3-3b-tq1.atlas',   4096),
    ('Bonsai-1.7B',  r'C:\atlas\bonsai-1.7b-tq1-g128.atlas', 2048),
    ('Bonsai-4B',    r'C:\atlas\bonsai-4b-tq1-g128.atlas', 8192),
]

for name, path, base_seq in models:
    t0 = time.time()
    m = AtlasModel(path, max_seq_len=2048)
    m.set_base_seq_len(base_seq)
    m.set_seed(42)
    t1 = time.time()
    out = m.generate_c("What is the capital of France?", max_new_tokens=30, temperature=0.0)
    t2 = time.time()
    out_clean = out.replace("\n", " | ")[:120]
    print(f"{name}: load={t1-t0:.1f}s gen={t2-t1:.1f}s out=\"{out_clean}\"")
    m.reset_cache()
    out2 = m.generate_c("What is 2+2?", max_new_tokens=20, temperature=0.0)
    out2_clean = out2.replace("\n", " | ")[:80]
    print(f"  reset+gen: \"{out2_clean}\"")
    print()
