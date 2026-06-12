#!/usr/bin/env python3
"""Convert TritPlane3 quantized model to ATLAS TQ1.0 format.

Two-phase:
  1. Dequantize tritplane/*.npz to fp16, build in-memory tensor dict
  2. Run pack_to_atlas with in_memory_tensors (no temp disk I/O)

Usage:
  python tritplane_to_atlas.py AsadIsmail/Qwen3-1.7B-ternary ./qwen3-1.7b-tq1.atlas
  python tritplane_to_atlas.py /path/to/local/model ./out.atlas --no-download
"""
import argparse, json, os, shutil, sys, tempfile
import numpy as np
from pathlib import Path

from pack_to_atlas import pack_to_atlas

# ─── TritPlane3 unpacking ──────────────────────────────────────────────

def unpack_tritplane_npz(npz_path):
    """Dequantize a single TritPlane3 .npz to fp32 weight matrix.

    On-disk layout:
      packed_i:  1D uint8, shape (nrows * packed_cols_per_row,)
      group_alpha_i: 2D fp16, shape (nrows, n_groups)
      group_mu_i:    2D fp16, shape (nrows, n_groups)
      group_size_i: scalar, always 32

    Reconstruction per plane:
      Deinterleave packed bytes -> 4 ternary vals/byte (2-bit: 00=-1, 01=0, 10=+1)
      For each group of group_size columns:
        contrib = alpha * ternary + mu
    """
    data = np.load(npz_path, allow_pickle=True)
    n_planes = sum(1 for k in data if k.startswith("packed_"))
    assert n_planes > 0, f"No planes in {npz_path}"

    gs = data["group_size_0"]
    group_size = int(gs) if np.ndim(gs) == 0 else int(gs.flat[0])
    alpha0 = data["group_alpha_0"]
    nrows, n_groups = alpha0.shape
    ncols = n_groups * group_size
    packed_cols = ncols // 4

    result = np.zeros((nrows, ncols), dtype=np.float32)

    for i in range(n_planes):
        packed = data[f"packed_{i}"].reshape(nrows, packed_cols)
        alpha = data[f"group_alpha_{i}"]  # [nrows, n_groups]
        mu = data[f"group_mu_{i}"]        # [nrows, n_groups]

        p = packed.astype(np.int32, copy=False)
        # LSB-first: byte & 3 = element 0, (byte>>2)&3 = element 1, ...
        # Mapping: 0->-1, 1->0, 2->+1, 3->+1 (3 never occurs)
        map_2bit = np.array([-1, 0, 1, 1])
        t0 = map_2bit[p & 3]
        t1 = map_2bit[(p >> 2) & 3]
        t2 = map_2bit[(p >> 4) & 3]
        t3 = map_2bit[(p >> 6) & 3]

        ternary = np.empty((nrows, ncols), dtype=np.int32)
        for j in range(packed_cols):
            ternary[:, j*4+0] = t0[:, j]
            ternary[:, j*4+1] = t1[:, j]
            ternary[:, j*4+2] = t2[:, j]
            ternary[:, j*4+3] = t3[:, j]

        alpha_rpt = alpha.repeat(group_size, axis=1)   # expand to full cols
        mu_rpt = mu.repeat(group_size, axis=1)
        result += alpha_rpt.astype(np.float32) * ternary + mu_rpt.astype(np.float32)

    data.close()
    return result


# ─── TritPlane filename to ATLAS tensor name ───────────────────────────

# Known compound tokens that use underscores internally
COMPOUND_TOKENS = {
    "self_attn", "encoder_attn", "cross_attn",
    "input_layernorm", "post_attention_layernorm",
    "attn_sub_norm", "ffn_sub_norm",
    "q_norm", "k_norm",
    "language_model",  # VLM prefix: model_language_model
}
# All projection suffixes
PROJ_TOKENS = {
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "gate_proj", "up_proj", "down_proj",
}


def npz_stem_to_tensor_name(stem):
    """Convert tritplane .npz stem to ATLAS tensor name.

    Input:  model_layers_0_self_attn_q_proj
    Output: model.layers.0.self_attn.q_proj.weight

    Strategy:
      1. Replace all _ with .
      2. Walk from left: collect parts that form known compound tokens
      3. Append .weight
    """
    parts = stem.split("_")

    # Build tokens greedily from left to right, matching known compounds
    tokens = []
    i = 0
    while i < len(parts):
        # Try longest match first (up to 3 parts)
        matched = False
        for n in range(min(3, len(parts) - i), 0, -1):
            candidate = "_".join(parts[i:i+n])
            if candidate in COMPOUND_TOKENS or candidate in PROJ_TOKENS or n == 1:
                if n == 1 and candidate in ("layers", "model"):
                    # These are dot-separated structural elements
                    pass
                tokens.append(candidate)
                i += n
                matched = True
                break
        if not matched:
            tokens.append(parts[i])
            i += 1

    # Join with dots
    dotted = ".".join(tokens)

    return dotted + ".weight"


def npz_to_tensor_name(filename):
    """Full .npz filename to ATLAS tensor name."""
    stem = filename.replace(".npz", "")
    return npz_stem_to_tensor_name(stem)


# ─── Model reconstruction ──────────────────────────────────────────────

def reconstruct_model(model_dir, orig_model_dir, temp_dir):
    """Dequantize TritPlane model, return (config, in_memory_tensors).

    Builds an in-memory dict of all tensors (dequantized weights + original
    non-weight tensors). Saves only tiny config files to temp_dir.

    Args:
        model_dir: Path to TritPlane3 model (with tritplane/*.npz)
        orig_model_dir: Path to original FP16 model
        temp_dir: Output temp directory (for config files only)

    Returns:
        (config_dict, in_memory_tensor_dict)
    """
    model_path = Path(model_dir)
    orig_model_path = Path(orig_model_dir)
    temp_path = Path(temp_dir)

    # Copy config files (from tritplane model or original)
    copied_configs = set()
    for src_dir in [model_path, orig_model_path]:
        for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                       "vocab.json", "merges.txt", "added_tokens.json",
                       "special_tokens_map.json"]:
            src = src_dir / fname
            if src.exists() and fname not in copied_configs:
                shutil.copy2(str(src), str(temp_path / fname))
                copied_configs.add(fname)

    # Read config from original model
    with open(orig_model_path / "config.json") as f:
        cfg = json.load(f)

    tritplane_dir = model_path / "tritplane"
    if not tritplane_dir.is_dir():
        raise FileNotFoundError(f"No tritplane/ dir in {model_dir}")

    npz_files = sorted(tritplane_dir.glob("*.npz"))
    print(f"[TritPlane] {len(npz_files)} quantized tensors")

    # Dequantize all npz weights into fp16
    import torch
    dequantized = {}
    for npz_path in npz_files:
        tname = npz_to_tensor_name(npz_path.name)
        print(f"  {npz_path.name:55s} -> {tname}")
        fp32 = unpack_tritplane_npz(str(npz_path))
        dequantized[tname] = torch.from_numpy(fp32).half()  # fp16 to save RAM

    print(f"[TritPlane] Dequantized {len(dequantized)} tensors")

    # Build in-memory tensor dict: inject dequantized weights into
    # original safetensors (keeping norms/embed as-is). Entirely in RAM,
    # no temp disk writes for tensors.
    from safetensors import safe_open
    all_tensors = {}
    for sf_path in sorted(orig_model_path.glob("*.safetensors")):
        if sf_path.name.startswith("."):
            continue
        print(f"  Merging {sf_path.name}...")
        with safe_open(str(sf_path), framework="pt") as f:
            for k in f.keys():
                if k in dequantized:
                    all_tensors[k] = dequantized.pop(k)
                else:
                    all_tensors[k] = f.get_tensor(k)

    # Handle any remaining dequantized tensors not in original shards
    if dequantized:
        print(f"  Warning: {len(dequantized)} extra tensors, merging")
        all_tensors.update(dequantized)

    print(f"[TritPlane] Built in-memory tensor dict: {len(all_tensors)} tensors")
    return cfg, all_tensors


# ─── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert TritPlane3 quantized model -> ATLAS TQ1.0")
    parser.add_argument("model", help="HF model ID or local path")
    parser.add_argument("output", help="Output .atlas file path")
    parser.add_argument("--no-download", action="store_true",
                        help="model arg is a local directory")
    parser.add_argument("--original-model", default=None,
                        help="Original FP16 model (HF ID or local path). "
                             "Auto-detected from metadata.json if omitted.")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temp dir after conversion")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--ttype", type=int, default=5, choices=[5, 7])
    parser.add_argument("--no-v6", action="store_false", dest="use_v6")
    args = parser.parse_args()

    if args.no_download:
        trit_dir = Path(args.model)
    else:
        print(f"Downloading {args.model}...")
        from huggingface_hub import snapshot_download
        trit_dir = Path(snapshot_download(args.model, resume_download=True))
        print(f"Downloaded to {trit_dir}")

    # Determine original model
    orig_model = args.original_model
    if orig_model is None:
        meta_file = trit_dir / "metadata.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_bytes())
            orig_model = meta.get("model_name")
            print(f"Auto-detected original model: {orig_model}")

    if orig_model is None:
        print("ERROR: Cannot determine original model. "
              "Specify --original-model or provide metadata.json with model_name.")
        sys.exit(1)

    if Path(orig_model).exists():
        orig_dir = Path(orig_model)
    else:
        print(f"Downloading original model {orig_model}...")
        from huggingface_hub import snapshot_download
        orig_dir = Path(snapshot_download(orig_model, resume_download=True))
        print(f"Downloaded to {orig_dir}")

    # Use temp dir for config files only (tensors stay in memory)
    with tempfile.TemporaryDirectory(prefix="tritplane_") as tmp:
        print(f"Building in-memory model (config in {tmp})")
        cfg, all_tensors = reconstruct_model(str(trit_dir), str(orig_dir), tmp)

        print(f"\nPacking to {args.output} (no temp tensor I/O)")
        pack_to_atlas(
            model_dir=tmp,
            output_path=args.output,
            ttype=args.ttype,
            block_size=args.block_size,
            use_v6=args.use_v6,
            raw_int8=True,
            per_row_int8=True,
            in_memory_tensors=all_tensors,
        )

    print(f"\nDone: {args.output}")


if __name__ == "__main__":
    main()
