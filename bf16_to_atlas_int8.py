#!/usr/bin/env python3
"""Convert original BF16 model directly to ATLAS int8 format.

Reads original BF16 safetensors, dequantizes to fp32, stores as int8
(ttype=3) with per-tensor scale. No ternarization loss — only 0.4% int8
quantization error.

Usage:
  python bf16_to_atlas_int8.py Qwen/Qwen3-1.7B ./qwen3-1.7b-int8.atlas
  python bf16_to_atlas_int8.py /path/to/model ./out.atlas --no-download
"""
import argparse, json, shutil, sys, tempfile
from pathlib import Path
from safetensors.torch import load_file as st_load_file
from pack_to_atlas import pack_to_atlas


def load_all_tensors(model_dir, tmp):
    """Load all safetensors into in-memory dict (no temp disk I/O)."""
    import torch
    from safetensors import safe_open
    all_tensors = {}
    for sf_path in sorted(Path(model_dir).glob("*.safetensors")):
        if sf_path.name.startswith("."):
            continue
        print(f"  Loading {sf_path.name}...")
        with safe_open(str(sf_path), framework="pt") as f:
            for k in f.keys():
                all_tensors[k] = f.get_tensor(k)
    print(f"  Total: {len(all_tensors)} tensors")
    return all_tensors


def main():
    parser = argparse.ArgumentParser(description="BF16 model -> ATLAS int8")
    parser.add_argument("model", help="HF model ID or local path")
    parser.add_argument("output", help="Output .atlas file path")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--per-row", action="store_true", help="Per-row int8 quantization (ttype=11)")
    args = parser.parse_args()

    if args.no_download:
        model_dir = Path(args.model)
    else:
        print(f"Downloading {args.model}...")
        from huggingface_hub import snapshot_download
        model_dir = Path(snapshot_download(args.model, resume_download=True))
        print(f"Downloaded to {model_dir}")

    with tempfile.TemporaryDirectory(prefix="bf16_atlas_") as tmp:
        print(f"Loading model into memory (config in {tmp})")
        # Copy config files to temp dir
        for f in model_dir.glob("*.json"):
            shutil.copy2(f, tmp)
        for f in model_dir.glob("*.py"):
            shutil.copy2(f, tmp)
        for f in model_dir.glob("*.txt"):
            shutil.copy2(f, tmp)

        all_tensors = load_all_tensors(model_dir, tmp)

        print(f"\nPacking to {args.output} (raw int8, no ternarization)")
        pack_to_atlas(
            model_dir=tmp,
            output_path=args.output,
            ttype=5,  # unused for raw_int8
            block_size=128,
            use_v6=True,
            raw_int8=True,
            per_row_int8=args.per_row,
            in_memory_tensors=all_tensors,
        )

    print(f"\nDone: {args.output}")


if __name__ == "__main__":
    main()
