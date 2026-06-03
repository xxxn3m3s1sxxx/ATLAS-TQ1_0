#!/usr/bin/env python3
"""Upload only README.md to existing HF model repos (skip atlas binary)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from release_to_hf import _generate_readme, _detect_size, KNOWN_CONFIGS, MODEL_SIZES

CANN_MODELS = [
    ("C:\\atlas\\models\\BitCPM-CANN-0.5B-tq1.atlas", "xxxn3m3s1sxxx/Ternary-BitCPM-CANN-0.5B-ATLAS"),
    ("C:\\atlas\\models\\BitCPM-CANN-1B-tq1.atlas",   "xxxn3m3s1sxxx/Ternary-BitCPM-CANN-1B-ATLAS"),
    ("C:\\atlas\\models\\BitCPM-CANN-3B-tq1.atlas",   "xxxn3m3s1sxxx/Ternary-BitCPM-CANN-3B-ATLAS"),
    ("C:\\atlas\\models\\BitCPM-CANN-8B-tq1.atlas",   "xxxn3m3s1sxxx/Ternary-BitCPM-CANN-8B-ATLAS"),
]

token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

from huggingface_hub import HfApi
api = HfApi()

for atlas_path, repo_id in CANN_MODELS:
    if not os.path.exists(atlas_path):
        print(f"SKIP: {atlas_path} not found")
        continue

    fname = os.path.basename(atlas_path)
    size_key = _detect_size(fname)
    cfg = dict(KNOWN_CONFIGS.get(size_key, {}))
    if not cfg:
        print(f"SKIP: no KNOWN_CONFIGS for {size_key}")
        continue

    model_name = repo_id.split('/')[-1]
    atlas_name = os.path.basename(atlas_path)
    perf, file_sz, desc = MODEL_SIZES.get(size_key, ("varies", "?", ""))
    readme = _generate_readme(model_name, size_key, cfg, atlas_name, perf, file_sz, desc)

    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"OK: README.md uploaded to {repo_id}")
