#!/usr/bin/env python3
"""Validate TQ1 block scales from .atlas file against HuggingFace reference."""

import os, sys, struct
import numpy as np

ATLAS_PATH = r"C:\atlas\trlm-1.1b-tq1-g128.atlas"

def fp16_bits_to_f32(h):
    """IEEE 754 binary16 -> float32"""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        return 0.0
    if exp == 31:
        return float('nan')
    return (1 - 2 * sign) * (2 ** (exp - 15)) * (1 + mant / 1024)

def parse_atlas_tensor(atlas_path, target_name_pattern="o_proj.weight"):
    with open(atlas_path, "rb") as f:
        hdr = f.read(64)
        magic = hdr[:5]
        ver = struct.unpack('<H', hdr[5:7])[0]
        n_layers = struct.unpack('<H', hdr[7:9])[0]
        hidden = struct.unpack('<H', hdr[9:11])[0]
        inter = struct.unpack('<H', hdr[11:13])[0]
        n_heads = hdr[13]
        n_kv_heads = hdr[14]
        head_dim = struct.unpack('<H', hdr[15:17])[0]
        vocab = struct.unpack('<I', hdr[17:21])[0]
        rope_theta = struct.unpack('<d', hdr[21:29])[0]
        name_block_size = struct.unpack('<i', hdr[56:60])[0]
        n_tensors = struct.unpack('<i', hdr[60:64])[0]

        print("=" * 70)
        print("ATLAS FILE HEADER")
        print("=" * 70)
        print(f"  Magic:              {magic}")
        print(f"  Version:            {ver}")
        print(f"  Layers:             {n_layers}")
        print(f"  Hidden:             {hidden}")
        print(f"  Intermediate:       {inter}")
        print(f"  Heads:              {n_heads}")
        print(f"  KV Heads:           {n_kv_heads}")
        print(f"  Head dim:           {head_dim}")
        print(f"  Vocab:              {vocab}")
        print(f"  RoPE theta:         {rope_theta}")
        print(f"  Tensors:            {n_tensors}")
        print()

        # Read directory
        dir_size = n_tensors * 12
        f.seek(64)
        dir_bytes = f.read(dir_size)

        # Read name block
        name_block_data = f.read(name_block_size)
        name_block_len = struct.unpack('<I', name_block_data[:4])[0]
        names_raw = name_block_data[4:]
        names = [n.decode('utf-8', errors='replace') for n in names_raw.split(b'\x00') if n]

        # Find target tensor (first matching o_proj at layer 0)
        target_idx = None
        target_info = None
        for i, name in enumerate(names):
            if target_name_pattern in name and "0.self_attn" in name:
                e = dir_bytes[i*12:(i+1)*12]
                ttype = e[0]
                offset = struct.unpack('<I', e[1:5])[0]
                row_dim = struct.unpack('<I', e[5:9])[0]
                packed_cols = (e[9] | (e[10] << 8) | (e[11] << 16))
                target_idx = i
                target_info = (ttype, offset, row_dim, packed_cols)
                break

        if target_idx is None:
            print("ERROR: Could not find o_proj tensor at layer 0")
            sys.exit(1)

        ttype, data_off, row_dim, packed_cols = target_info
        tensor_name = names[target_idx]

        print("TENSOR DIRECTORY ENTRY")
        print("=" * 70)
        print(f"  Name:               {tensor_name}")
        print(f"  Index:              {target_idx}")
        print(f"  ttype:              {ttype}")
        print(f"  File offset:        {data_off}")
        print(f"  row_dim:            {row_dim}")
        print(f"  packed_cols:        {packed_cols}")
        print()

        # Seek and parse ttype=5
        f.seek(data_off)
        prefix = f.read(3)
        block_size = prefix[0]
        n_blocks = struct.unpack('<H', prefix[1:3])[0]

        print("TTYPE=5 PREFIX (block-scaled TQ1.0)")
        print("=" * 70)
        print(f"  block_size:         {block_size}")
        print(f"  n_blocks:           {n_blocks}")
        expected_blocks = (1792 + block_size - 1) // block_size
        print(f"  expected n_blocks:  {expected_blocks} (for 1792 cols)")
        print(f"  scales per row:     {n_blocks} fp16 values = {n_blocks * 2} bytes")
        print(f"  total scales:       {row_dim * n_blocks} fp16 = {row_dim * n_blocks * 2} bytes")
        print()

        # Read first row scales only
        scales_first_row = f.read(n_blocks * 2)
        uint16_vals = list(struct.unpack('<'+'H'*n_blocks, scales_first_row))

        print("FIRST ROW BLOCK SCALES (raw uint16 -> fp32)")
        print("=" * 70)
        print(f"  {'Block':>6s}  {'Uint16':>8s}  {'FP32':>14s}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*14}")
        for i, u16 in enumerate(uint16_vals):
            f32 = fp16_bits_to_f32(u16)
            print(f"  {i:6d}  {u16:8d}  {f32:14.8f}")

        # Check if all first 7 blocks share a value and last 7 share another
        first_half = uint16_vals[:7]
        second_half = uint16_vals[7:]
        print()
        print(f"  First  7 blocks:  {first_half}  ->  {[fp16_bits_to_f32(x) for x in first_half]}")
        print(f"  Last   7 blocks:  {second_half}  ->  {[fp16_bits_to_f32(x) for x in second_half]}")
        unique_vals = set(uint16_vals)
        print(f"  Unique scale values in first row: {len(unique_vals)}")
        for v in sorted(unique_vals):
            print(f"    {v:8d} (0x{v:04X}) -> {fp16_bits_to_f32(v):.8f}")

        packed_trits_size = row_dim * packed_cols
        print()
        print("TENSOR SIZE BREAKDOWN")
        print("=" * 70)
        print(f"  Prefix (block_size + n_blocks):  3 bytes")
        print(f"  Scales ({row_dim} x {n_blocks} x 2):      {row_dim * n_blocks * 2} bytes")
        print(f"  Packed trits ({row_dim} x {packed_cols}):  {packed_trits_size} bytes")
        total = 3 + row_dim * n_blocks * 2 + packed_trits_size
        print(f"  Total:                             {total} bytes")
        print()

        return {
            "n_layers": n_layers,
            "hidden": hidden,
            "inter": inter,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "vocab": vocab,
            "block_size": block_size,
            "n_blocks": n_blocks,
            "row_dim": row_dim,
            "packed_cols": packed_cols,
            "scales_uint16": uint16_vals,
            "scales_fp32": [fp16_bits_to_f32(v) for v in uint16_vals],
        }

def try_load_hf_model(atlas_scales_uint16):
    """Attempt to load HuggingFace reference model."""
    print("=" * 70)
    print("HUGGINGFACE REFERENCE MODEL")
    print("=" * 70)

    model = None
    found_model = None
    n_blocks = len(atlas_scales_uint16)

    # Try local model first
    local_model_path = r"C:\atlas\models\TriLM_1.1B_Unpacked"
    if os.path.exists(local_model_path):
        print(f"  Found local model at: {local_model_path}")
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
            import torch
            cfg = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
            print(f"  Model config: layers={cfg.num_hidden_layers}, hidden={cfg.hidden_size}")
            if cfg.hidden_size == 1792 and cfg.num_hidden_layers == 24:
                print(f"  Matched architecture!")
                model = AutoModelForCausalLM.from_pretrained(
                    local_model_path, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
                )
                found_model = local_model_path
        except Exception as e:
            print(f"  Local load failed: {e}")
            model = None

    if model is not None:
        print(f"\n  ** Model loaded: {found_model}")

        # Get O_proj from layer 0
        try:
            o_proj = model.model.layers[0].self_attn.o_proj.weight
        except (AttributeError, IndexError):
            try:
                o_proj = model.transformer.h[0].self_attention.o_proj.weight
            except (AttributeError, IndexError):
                o_proj = None

        if o_proj is not None:
            import torch
            w = o_proj.detach().cpu().to(torch.float32).numpy()
            print(f"  O_proj shape: {w.shape}")

            # Compute block scales for first row the same way packer does
            block_size = 128
            ncols = w.shape[1]

            row0 = w[0, :]
            print(f"  First row (first 16 values): {row0[:16]}")
            print(f"  First row absmax: {np.max(np.abs(row0)):.6f}")

            # Compute per-block scales
            w_3d = w[0, :].reshape(n_blocks, block_size)
            block_scales32 = np.max(np.abs(w_3d), axis=1)
            block_scales32 = np.where(block_scales32 < 1e-10, 1.0, block_scales32)
            hf_scales_fp16 = block_scales32.astype(np.float16)
            hf_uint16 = hf_scales_fp16.view(np.uint16).tolist()
            hf_fp32 = hf_scales_fp16.astype(np.float32).tolist()

            print()
            print("COMPARISON: ATLAS vs HF first-row block scales")
            print("=" * 70)
            print(f"  {'Block':>6s}  {'ATLAS_u16':>10s}  {'ATLAS_f32':>12s}  {'HF_u16':>8s}  {'HF_f32':>12s}  {'Match'}")
            print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*12}  {'-'*5}")

            all_match = True
            for i in range(n_blocks):
                a_u16 = atlas_scales_uint16[i]
                a_f32 = fp16_bits_to_f32(a_u16)
                h_u16 = int(hf_uint16[i])
                h_f32 = float(hf_fp32[i])
                match = "OK" if a_u16 == h_u16 else "MISMATCH"
                if a_u16 != h_u16:
                    all_match = False
                print(f"  {i:6d}  {a_u16:10d}  {a_f32:12.8f}  {h_u16:8d}  {h_f32:12.8f}  {match:>8s}")

            print()
            if all_match:
                print("  ** ALL SCALES MATCH!")
            else:
                print("  ** SCALES DIFFER - see above for mismatches")
                print()
                print("  HF weight first row range:")
                print(f"    min={np.min(row0):.6f} max={np.max(row0):.6f} mean={np.mean(row0):.6f}")
                print(f"    absmax={np.max(np.abs(row0)):.6f}")
        else:
            print("  ** Could not find O_proj weight in model")
    else:
        print("  ** Could not load HuggingFace model from any source")

    return model

if __name__ == "__main__":
    info = parse_atlas_tensor(ATLAS_PATH)
    print()
    try:
        try_load_hf_model(info["scales_uint16"])
    except ImportError:
        print("\n  ** transformers not installed - skipping HF comparison")
        print("  Install with: pip install transformers torch")
    print()
    print("Done.")
