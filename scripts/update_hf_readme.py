#!/usr/bin/env python3
"""Fix broken GitHub URL in HF model READMEs.

Before: github.com/xxxn3m3s1sxxx/ATLAS → 404
After:  github.com/xxxn3m3s1sxxx/ATLAS-TQ1_0
"""
import os, sys

token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

from huggingface_hub import HfApi
api = HfApi()

REPOS = [
    "xxxn3m3s1sxxx/Falcon3-1B-Instruct-1.58bit-ATLAS",
    "xxxn3m3s1sxxx/Falcon3-3B-Instruct-1.58bit-ATLAS",
    "xxxn3m3s1sxxx/Falcon3-7B-Instruct-1.58bit-ATLAS",
    "xxxn3m3s1sxxx/Falcon3-10B-Instruct-1.58bit-ATLAS",
    "xxxn3m3s1sxxx/Ternary-Bonsai-1.7B-ATLAS",
    "xxxn3m3s1sxxx/Ternary-Bonsai-4B-ATLAS",
    "xxxn3m3s1sxxx/Ternary-Bonsai-8B-ATLAS",
    "xxxn3m3s1sxxx/BitNet-2B4T-b1.58-ATLAS",
    "xxxn3m3s1sxxx/Ternary-BitCPM-CANN-0.5B-ATLAS",
    "xxxn3m3s1sxxx/Ternary-BitCPM-CANN-1B-ATLAS",
    "xxxn3m3s1sxxx/Ternary-BitCPM-CANN-3B-ATLAS",
    "xxxn3m3s1sxxx/Ternary-BitCPM-CANN-8B-ATLAS",
]

OLD = "github.com/xxxn3m3s1sxxx/ATLAS"
NEW = "github.com/xxxn3m3s1sxxx/ATLAS-TQ1_0"

for repo_id in REPOS:
    try:
        readme = api.hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="model")
        with open(readme, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"FAIL download {repo_id}: {e}")
        continue

    if OLD not in content:
        print(f"SKIP {repo_id}: URL not found in README")
        continue

    content = content.replace(OLD, NEW)
    api.upload_file(
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"OK: {repo_id}")
