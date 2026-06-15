#!/usr/bin/env python3
"""Fix README YAML tags on HF models + rename CANN models."""
import os, sys, json, logging, tempfile, shutil
from huggingface_hub import HfApi

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("fix_readmes")
api = HfApi()

TAG_FIXES = {
    "TriLM-99M-ATLAS":  {"arch": "trilm-99m",    "source": "SpectraSuite/TriLM_99M_Unpacked"},
    "TriLM-190M-ATLAS": {"arch": "trilm-99m",    "source": "SpectraSuite/TriLM_190M_Unpacked"},
    "TriLM-390M-ATLAS": {"arch": "trilm-subln",  "source": "SpectraSuite/TriLM_390M_Unpacked"},
    "TriLM-560M-ATLAS": {"arch": "trilm-subln",  "source": "SpectraSuite/TriLM_560M_Unpacked"},
    "TriLM-830M-ATLAS": {"arch": "trilm-subln",  "source": "SpectraSuite/TriLM_830M_Unpacked"},
    "TriLM-1.1B-ATLAS": {"arch": "trilm-subln",  "source": "SpectraSuite/TriLM_1.1B_Unpacked"},
    "TriLM-1.5B-ATLAS": {"arch": "trilm-subln",  "source": "SpectraSuite/TriLM_1.5B_Unpacked"},
    "TriLM-2.4B-ATLAS": {"arch": "trilm-subln",  "source": "SpectraSuite/TriLM_2.4B_Unpacked"},
    "TriLM-3.9B-ATLAS": {"arch": "trilm",        "source": "SpectraSuite/TriLM_3.9B_Unpacked"},
    "Llama3-8B-1.58-ATLAS": {"arch": "llama3",   "source": "HF1BitLLM/Llama3-8B-1.58-100B-tokens"},
}

CANN_RENAME = [
    ("Ternary-BitCPM-CANN-0.5B-ATLAS", "BitCPM-CANN-0.5B-ATLAS", 0.22),
    ("Ternary-BitCPM-CANN-1B-ATLAS",   "BitCPM-CANN-1B-ATLAS",   0.83),
    ("Ternary-BitCPM-CANN-3B-ATLAS",   "BitCPM-CANN-3B-ATLAS",   1.35),
    ("Ternary-BitCPM-CANN-8B-ATLAS",   "BitCPM-CANN-8B-ATLAS",   2.65),
]

TRILM_SIZES = {
    "TriLM-99M-ATLAS":  {"hs": 512,  "il": 1280,  "nl": 16, "nh": 8,  "nkv": 8,  "hd": 64,  "vs": 50304},
    "TriLM-190M-ATLAS": {"hs": 768,  "il": 2048,  "nl": 16, "nh": 12, "nkv": 12, "hd": 64,  "vs": 50304},
    "TriLM-390M-ATLAS": {"hs": 1024, "il": 2560,  "nl": 24, "nh": 16, "nkv": 16, "hd": 64,  "vs": 50304},
    "TriLM-560M-ATLAS": {"hs": 1280, "il": 3072,  "nl": 24, "nh": 20, "nkv": 20, "hd": 64,  "vs": 50304},
    "TriLM-830M-ATLAS": {"hs": 1536, "il": 4096,  "nl": 24, "nh": 24, "nkv": 24, "hd": 64,  "vs": 50304},
    "TriLM-1.1B-ATLAS": {"hs": 1792, "il": 5120,  "nl": 24, "nh": 28, "nkv": 28, "hd": 64,  "vs": 50432},
    "TriLM-1.5B-ATLAS": {"hs": 2048, "il": 6144,  "nl": 24, "nh": 32, "nkv": 32, "hd": 64,  "vs": 50432},
    "TriLM-2.4B-ATLAS": {"hs": 2304, "il": 7680,  "nl": 30, "nh": 36, "nkv": 36, "hd": 64,  "vs": 50304},
    "TriLM-3.9B-ATLAS": {"hs": 3072, "il": 9216,  "nl": 30, "nh": 24, "nkv": 24, "hd": 128, "vs": 50688},
}

def gen_readme(name, source, arch, size_gb, cfg):
    hs = cfg["hs"]
    il = cfg["il"]
    nl = cfg["nl"]
    nh = cfg["nh"]
    nkv = cfg["nkv"]
    hd = cfg["hd"]
    vs = cfg["vs"]
    return f"""---
license: mit
base_model: {source}
tags:
- ternary
- quantized
- atlas
- tq1
- cpu-inference
- {arch}
- cpu-llm
pipeline_tag: text-generation
---

# {name}

Ternary-quantized 1.58-bit variant of [{source}](https://huggingface.co/{source}), repacked into ATLAS TQ1.0.

| Property | Value |
|----------|-------|
| Architecture | {arch} |
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

def fix_trilm_readmes():
    for name, info in TAG_FIXES.items():
        repo = f"xxxn3m3s1sxxx/{name}"
        log.info(f"Fixing README: {name}")
        atlas_file = f"{name}.tq1.atlas"
        files = list(api.list_repo_files(repo))
        if atlas_file not in files:
            log.warning(f"  {atlas_file} not found, skipping")
            continue
        # Get actual size from files listing
        size_gb = 0
        for f in files:
            if f == atlas_file:
                meta = api.get_paths_info(repo, [atlas_file])
                size_gb = meta[0].size / (1024**3) if meta and meta[0].size else 0

        cfg = TRILM_SIZES.get(name, TRILM_SIZES.get("TriLM-1.5B-ATLAS"))
        readme = gen_readme(name, info["source"], info["arch"], size_gb, cfg)

        api.upload_file(path_or_fileobj=readme.encode("utf-8"),
                        path_in_repo="README.md", repo_id=repo, repo_type="model")
        log.info(f"  Done: {info['arch']}, {size_gb:.2f} GB")

def fix_llama3_readme():
    name = "Llama3-8B-1.58-ATLAS"
    repo = f"xxxn3m3s1sxxx/{name}"
    log.info(f"Fixing README: {name}")
    atlas_file = f"{name}.tq1.atlas"
    cfg = {"hs": 4096, "il": 14336, "nl": 32, "nh": 32, "nkv": 8, "hd": 128, "vs": 131072}
    size_gb = 3.51
    readme = gen_readme(name, "HF1BitLLM/Llama3-8B-1.58-100B-tokens", "llama3", size_gb, cfg)
    api.upload_file(path_or_fileobj=readme.encode("utf-8"),
                    path_in_repo="README.md", repo_id=repo, repo_type="model")
    log.info(f"  Done")

def rename_cann():
    tmp = tempfile.mkdtemp(prefix="cann_rename_")
    try:
        for old_name, new_name, size_gb in CANN_RENAME:
            old_repo = f"xxxn3m3s1sxxx/{old_name}"
            new_repo = f"xxxn3m3s1sxxx/{new_name}"
            atlas_file = f"{old_name}.tq1.atlas"
            new_atlas_file = f"{new_name}.tq1.atlas"

            log.info(f"Rename: {old_name} -> {new_name}")

            # Check if model file exists
            files = list(api.list_repo_files(old_repo))
            if atlas_file not in files:
                log.warning(f"  {atlas_file} not found, skipping")
                continue

            # Download atlas file
            log.info(f"  Downloading...")
            local = api.hf_hub_download(old_repo, atlas_file, local_dir=tmp,
                                        local_dir_use_symlinks=False)

            # Create new repo
            log.info(f"  Creating new repo...")
            api.create_repo(new_repo, repo_type="model", private=False, exist_ok=True)

            # Upload with new name
            log.info(f"  Uploading...")
            api.upload_file(path_or_fileobj=local, path_in_repo=new_atlas_file,
                            repo_id=new_repo, repo_type="model")

            # Upload README
            # CANN arch = "llama" (they use Llama architecture with LongRoPE)
            readme = f"""---
license: mit
base_model: PrismML/{old_name.replace('Ternary-', '')}
tags:
- ternary
- quantized
- atlas
- tq1
- cpu-inference
- llama
- cpu-llm
pipeline_tag: text-generation
---

# {new_name}

Ternary-quantized 1.58-bit variant of PrismML/{old_name.replace('Ternary-', '')}, repacked into ATLAS TQ1.0.

| Property | Value |
|----------|-------|
| Architecture | llama |
| Format | TQ1.0 (~1.58 bits/weight) |
| Size | {size_gb:.2f} GB |
| | |

## Usage
```python
from atlas_infer import AtlasModel
model = AtlasModel("{new_name}")
print(model.generate_c("The capital of France is", max_new_tokens=50, temperature=0.7, top_k=40))
```
"""
            api.upload_file(path_or_fileobj=readme.encode("utf-8"),
                            path_in_repo="README.md", repo_id=new_repo, repo_type="model")

            # Delete old repo
            log.info(f"  Deleting old repo...")
            api.delete_repo(old_repo, repo_type="model")

            log.info(f"  Done: https://huggingface.co/{new_repo}")

        log.info("All CANN renamed!")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def main():
    log.info("=== Step 1: Fix TriLM READMEs ===")
    fix_trilm_readmes()

    log.info("\n=== Step 2: Fix Llama3 README ===")
    fix_llama3_readme()

    log.info("\n=== Step 3: Rename CANN ===")
    rename_cann()

    log.info("\nAll done!")

if __name__ == "__main__":
    main()
