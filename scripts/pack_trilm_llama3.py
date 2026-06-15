#!/usr/bin/env python3
"""Download, pack, and push TriLM + Llama3 models to HF Hub.
Each model uses a temp dir that gets cleaned up after packing."""
import os, sys, json, time, shutil, tempfile, logging, argparse

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pack_to_atlas import pack_to_atlas

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("pack_trilm")

api = HfApi()

# Ordered by size (smallest first — quick wins)
MODELS = [
    ("SpectraSuite/TriLM_99M_Unpacked",    "TriLM-99M-ATLAS"),
    ("SpectraSuite/TriLM_190M_Unpacked",   "TriLM-190M-ATLAS"),
    ("SpectraSuite/TriLM_390M_Unpacked",   "TriLM-390M-ATLAS"),
    ("SpectraSuite/TriLM_560M_Unpacked",   "TriLM-560M-ATLAS"),
    ("SpectraSuite/TriLM_830M_Unpacked",   "TriLM-830M-ATLAS"),
    ("SpectraSuite/TriLM_1.1B_Unpacked",   "TriLM-1.1B-ATLAS"),
    ("SpectraSuite/TriLM_1.5B_Unpacked",   "TriLM-1.5B-ATLAS"),
    ("SpectraSuite/TriLM_2.4B_Unpacked",   "TriLM-2.4B-ATLAS"),
    ("SpectraSuite/TriLM_3.9B_Unpacked",   "TriLM-3.9B-ATLAS"),
    ("HF1BitLLM/Llama3-8B-1.58-100B-tokens", "Llama3-8B-1.58-ATLAS"),
]


def get_config(source_repo):
    path = api.hf_hub_download(source_repo, "config.json")
    with open(path) as f:
        return json.load(f)


def gen_readme(name, source, config, size_gb, arch_name):
    hs = config.get("hidden_size", "?")
    il = config.get("intermediate_size", "?")
    nl = config.get("num_hidden_layers", "?")
    nh = config.get("num_attention_heads", "?")
    nkv = config.get("num_key_value_heads", nh)
    vs = config.get("vocab_size", "?")
    hd = config.get("head_dim", 128)

    return f"""---
license: mit
base_model: {source}
tags:
- ternary
- quantized
- atlas
- tq1
- cpu-inference
- {arch_name}
- cpu-llm
pipeline_tag: text-generation
---

# {name}

Ternary-quantized 1.58-bit variant of [{source}](https://huggingface.co/{source}), repacked into ATLAS TQ1.0.

| Property | Value |
|----------|-------|
| Architecture | {arch_name} |
| Format | TQ1.0 (~1.58 bits/weight) |
| Size | {size_gb:.2f} GB |
| Layers | {nl} | Hidden | {hs} | FFN | {il} |
| Heads | {nh}/{nkv} | Head Dim | {hd} | Vocab | {vs} |

## Usage
```python
from atlas_infer import AtlasModel
model = AtlasModel("{name}")
print(model.generate_c("The capital of France is", max_new_tokens=50, temperature=0.7, top_k=40))
```
"""


def pack_one(source_repo, target_name):
    target_repo = f"xxxn3m3s1sxxx/{target_name}"
    atlas_file = f"{target_name}.tq1.atlas"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    log.info(f"  {source_repo}  -->  {target_repo}")

    config = get_config(source_repo)
    nl = config.get("num_hidden_layers", "?")
    hs = config.get("hidden_size", "?")
    il = config.get("intermediate_size", "?")
    log.info(f"  Config: {nl}L/{hs}H/{il}I")

    # Download to temp dir
    log.info("  [1/3] Downloading...")
    tmp = tempfile.mkdtemp(prefix="atlas_pack_")
    try:
        local = snapshot_download(source_repo, local_dir=tmp, local_dir_use_symlinks=False)
        log.info(f"  Downloaded: {local}")

        # Pack
        log.info("  [2/3] Packing to TQ1.0...")
        out_path = os.path.join(root, atlas_file)
        pack_to_atlas(model_dir=local, output_path=out_path, ttype=5,
                      block_size=128, use_v6=True, packed_path=False)
        size_gb = os.path.getsize(out_path) / (1024**3)
        log.info(f"  Packed: {size_gb:.2f} GB")
    finally:
        # Clean up temp download
        shutil.rmtree(tmp, ignore_errors=True)
        log.info("  Cleaned up temp dir")

    # Determine arch name
    mt = config.get("model_type", "llama")
    hd = config.get("head_dim", 128)
    if hs <= 256:
        arch_name = "trilm-99m"
    elif hd == 64:
        arch_name = "trilm-subln"
    elif "Llama3" in source_repo:
        arch_name = "llama3"
    else:
        arch_name = mt

    # Upload
    log.info("  [3/3] Uploading...")
    api.create_repo(target_repo, repo_type="model", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=out_path, path_in_repo=atlas_file,
                    repo_id=target_repo, repo_type="model")
    readme = gen_readme(target_name, source_repo, config, size_gb, arch_name)
    api.upload_file(path_or_fileobj=readme.encode("utf-8"),
                    path_in_repo="README.md", repo_id=target_repo, repo_type="model")

    os.remove(out_path)
    log.info(f"  DONE: https://huggingface.co/{target_repo}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-at", default=None)
    parser.add_argument("--skip", default=None)
    args = parser.parse_args()

    started = not args.start_at
    for source_repo, target_name in MODELS:
        if args.skip and args.skip in target_name:
            log.info(f"[SKIP] {target_name}")
            continue
        if args.start_at and args.start_at not in target_name:
            if not started:
                continue
        if not started and args.start_at and args.start_at in target_name:
            started = True

        try:
            pack_one(source_repo, target_name)
        except Exception as e:
            log.error(f"FAILED {target_name}: {e}")
            log.error("Continuing...")
            continue
        time.sleep(2)

    log.info("All done!")


if __name__ == "__main__":
    main()
