# Check 8B Pro training telemetry
import os
from huggingface_hub import hf_hub_download
import json, sys

path = hf_hub_download("xxxn3m3s1sxxx/Ternary-Bonsai-8B-ATLAS-Pro", "ste_telemetry.json",
    token=os.environ.get("HF_TOKEN", ""))
with open(path) as f:
    d = json.load(f)
print(f"Records: {len(d)}")
for i in range(0, len(d), 200):
    x = d[i]
    el = x.get("eval_loss", "-")
    ep = x.get("eval_ppl", "-")
    print(f"  step {x['step']:5d}: loss={x['loss']:.4f}  "
          f"grad_b={x.get('grad_norm_b_mean',0):.8f}  "
          f"retention={x.get('ternary_retention_rate',0):.4f}  "
          f"eval_loss={el}  eval_ppl={ep}")
print(f"\nLast record:")
for k, v in d[-1].items():
    print(f"  {k}: {v}")
