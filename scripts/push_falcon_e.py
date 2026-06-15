#!/usr/bin/env python3
"""Push all 4 Falcon-E models to HF Hub."""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.release_to_hf import push_to_hub, _generate_readme, _detect_size, MODEL_SIZES

base = r"C:\atlas\models"

models = [
    ("falcon_e_1b_base",    "xxxn3m3s1sxxx/Falcon-E-1B-Base-1.58bit-ATLAS"),
    ("falcon_e_1b_instruct","xxxn3m3s1sxxx/Falcon-E-1B-Instruct-1.58bit-ATLAS"),
    ("falcon_e_3b_base",    "xxxn3m3s1sxxx/Falcon-E-3B-Base-1.58bit-ATLAS"),
    ("falcon_e_3b_instruct","xxxn3m3s1sxxx/Falcon-E-3B-Instruct-1.58bit-ATLAS"),
]

for dirname, repo_id in models:
    model_dir = os.path.join(base, dirname)
    cfg_path = os.path.join(model_dir, "config.json")
    atlas_file = None
    for f in os.listdir(model_dir):
        if f.endswith(".atlas"):
            atlas_file = os.path.join(model_dir, f)
            break

    if not atlas_file:
        print(f"[SKIP] {dirname}: no .atlas file")
        continue

    with open(cfg_path) as f:
        cfg = json.load(f)

    model_name = repo_id.split("/")[-1]
    size = _detect_size(model_name) or "?"
    perf, file_size_str, desc = MODEL_SIZES.get(size, ("varies", "?", ""))
    readme = _generate_readme(model_name, size, cfg, os.path.basename(atlas_file), perf, file_size_str, desc)

    print(f"\n{'='*60}")
    print(f"[PUSH] {repo_id}")
    print(f"  file: {os.path.basename(atlas_file)} ({os.path.getsize(atlas_file)//1024//1024} MB)")
    print(f"  size: {size}")
    print(f"{'='*60}")

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"  Uploading atlas file...")
    api.upload_file(
        path_or_fileobj=atlas_file,
        path_in_repo=os.path.basename(atlas_file),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  Uploading README...")
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  OK https://huggingface.co/{repo_id}")

print(f"\n{'='*60}")
print("ALL 4 FALCON-E MODELS PUSHED OK")
print(f"{'='*60}")
