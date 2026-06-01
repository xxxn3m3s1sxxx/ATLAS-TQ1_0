#!/usr/bin/env python3
"""Synthetic ATLAS model generator for CI testing.

Produces tiny .atlas files (2 layers, hidden=128, vocab=256) exercising
all dispatch paths: ttype=1 norms/embeds, ttype=5 TQ1-packed weights.

Usage:
    python tests/atlas_mock_model.py [out_base]
    python -c "import atlas_infer; m = atlas_infer.AtlasModel('mock/out-qwen3.atlas')"
"""
import json, os, struct, sys
import numpy as np

TQ1_MUL = np.array([1, 3, 9, 27, 81], dtype=np.uint32)

ARCHES = {
    "falcon3": dict(
        n_layers=2, hidden=128, inter=512,
        n_heads=4, n_kv_heads=2, head_dim=256,
        vocab=256, rope_theta=1000042.0,
        rope_interleaved=True, stride=9,
        arch="falcon3",
        layer_tensors=[
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ],
    ),
    "qwen3": dict(
        n_layers=2, hidden=128, inter=512,
        n_heads=4, n_kv_heads=2, head_dim=128,
        vocab=256, rope_theta=1000000.0,
        rope_interleaved=True, stride=11,
        arch="qwen3",
        layer_tensors=[
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        ],
    ),
    "bitnet": dict(
        n_layers=2, hidden=128, inter=512,
        n_heads=4, n_kv_heads=2, head_dim=128,
        vocab=256, rope_theta=500000.0,
        rope_interleaved=False, stride=11,
        arch="bitnet",
        layer_tensors=[
            "input_layernorm.weight",
            "self_attn.attn_sub_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "mlp.ffn_sub_norm.weight",
        ],
    ),
    "falcon3-ttype0": dict(
        n_layers=2, hidden=128, inter=512,
        n_heads=4, n_kv_heads=2, head_dim=256,
        vocab=256, rope_theta=1000042.0,
        rope_interleaved=True, stride=9,
        use_tq1="ttype0",
        arch="falcon3",
        layer_tensors=[
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ],
    ),
    "turboquant": dict(
        n_layers=2, hidden=256, inter=1024,
        n_heads=4, n_kv_heads=2, head_dim=128,
        vocab=512, rope_theta=1000000.0,
        rope_interleaved=True, stride=11,
        arch="turboquant",
        use_tq1="ttype7",
        layer_tensors=[
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ],
    ),
}

GLOBAL_TENSORS = ["model.embed_tokens.weight", "model.norm.weight"]

# Dispatch corridors for coverage-driven testing.
# Each entry specifies how to generate + configure the model to exercise a
# specific C++ dispatch path in forward_layer_internal.
#
# Fields:
#   qkv_ttype / ffn_ttype  — tensor encoding for attention / FFN weights
#   use_f32_bypass          — meta block value; 0 = production dispatch, 1 = f32
#   post_init               — list of (dll_func_name, args) to call after AtlasModel()
#                             e.g. ("atlas_set_use_f32_matmul", [0]) resets to production
#   requires_hidden_gt_2048 — if True, mock uses hidden=4096 to skip auto-f32 in Python
CORRIDORS = {
    # Production path (7B/10B): all decompressed to int8, then FFN quantized to int4.
    # QKV/O → matmul_i8_f32, FFN → matmul_i4_reorder_deq (covers 5.6% gap).
    "production_int8": dict(
        qkv_ttype=5, ffn_ttype=3,
        use_f32_bypass=0,
        post_init=[("atlas_set_use_f32_matmul", [0]),
                    ("atlas_quantize_ffn_to_i4", [])],
        requires_hidden_gt_2048=False,
    ),
    # f32 bypass (1B, Bonsai): everything decompressed → f32_reorder.
    "f32_bypass": dict(
        qkv_ttype=5, ffn_ttype=3,
        use_f32_bypass=1,
        post_init=[],
        requires_hidden_gt_2048=False,
    ),
    # Ternary dispatch: no f32, no int4, pure sign-of-int8.
    # QKV/O → matmul_ternary_add_reorder, FFN → matmul_ternary_add_reorder.
    "ternary_dispatch": dict(
        qkv_ttype=5, ffn_ttype=3,
        use_f32_bypass=0,
        post_init=[("atlas_set_use_f32_matmul", [0]),
                    ("atlas_set_use_ternary_matmul", [1])],
        requires_hidden_gt_2048=False,
    ),
    # Direct TQ1-packed: no decompress, no quantize, block-fused kernel.
    # QKV/O → matmul_tq1_block_fused_s8, FFN → matmul_tq1_block_fused_s8.
    "packed_direct": dict(
        qkv_ttype=5, ffn_ttype=5,
        use_f32_bypass=0,
        post_init=[("atlas_set_use_packed_matmul", [1])],
        requires_hidden_gt_2048=False,
    ),
}


def _shape_of(name, cfg):
    h = cfg["hidden"]
    i = cfg["inter"]
    nh = cfg["n_heads"]
    nk = cfg["n_kv_heads"]
    hd = cfg["head_dim"]
    v = cfg["vocab"]
    qd = nh * hd
    kvd = nk * hd

    if "embed_tokens" in name:
        return (v, h)
    if "lm_head" in name:
        return (v, h)
    if name == "model.norm.weight":
        return (h,)
    if "input_layernorm" in name or "post_attention_layernorm" in name:
        return (h,)
    if "attn_sub_norm" in name:
        return (h,)

    if "ffn_sub_norm" in name:
        return (i,)
    if "ffn_out_norm" in name:
        return (h,)
    if "q_norm" in name or "k_norm" in name:
        return (hd,)
    if "q_proj" in name:
        return (qd, h)
    if "k_proj" in name or "v_proj" in name:
        return (kvd, h)
    if "o_proj" in name:
        return (h, qd)
    if "gate_proj" in name or "up_proj" in name:
        return (i, h)
    if "down_proj" in name:
        return (h, i)
    return (v, h)


def pack_tq1_g128(weights_fp16, block_size=128):
    """TQ1 g128 block-scaled packing (ttype=5)."""
    w = weights_fp16.astype(np.float32)
    nrows, ncols = w.shape
    n_blocks = (ncols + block_size - 1) // block_size
    pad_len = n_blocks * block_size - ncols
    w_pad = np.pad(w, ((0, 0), (0, pad_len)), constant_values=0) if pad_len else w
    w_3d = w_pad.reshape(nrows, n_blocks, block_size)
    block_scales32 = np.max(np.abs(w_3d), axis=2)
    block_scales32 = np.where(block_scales32 < 1e-10, 1.0, block_scales32)
    scales_expanded = np.repeat(block_scales32[:, :, np.newaxis], block_size, axis=2)
    ternary_3d = np.clip(np.round(w_3d / scales_expanded).astype(np.int32), -1, 1)
    ternary_flat = ternary_3d.reshape(nrows, n_blocks * block_size)[:, :ncols].astype(np.int8)
    packed_per_row = (ncols + 4) // 5
    full_len = packed_per_row * 5
    out = np.empty(nrows * packed_per_row, dtype=np.uint8)
    for r in range(nrows):
        row = ternary_flat[r, :].astype(np.int32) + 1
        if ncols < full_len:
            row = np.pad(row, (0, full_len - ncols), constant_values=1)
        t5 = row[:full_len].reshape(packed_per_row, 5)
        out[r * packed_per_row: (r + 1) * packed_per_row] = (
            (t5 * TQ1_MUL).sum(axis=1).astype(np.uint8)
        )
    header = struct.pack("<BH", block_size, n_blocks) + block_scales32.astype(np.float16).tobytes()
    return header + out.tobytes(), packed_per_row, n_blocks, block_size


def pack_turboquant(weights_fp16, block_size=32):
    """TurboQuant 2-bit packed ternary (ttype=7). 4 vals/byte, block-scaled."""
    w = weights_fp16.astype(np.float32)
    nrows, ncols = w.shape
    n_blocks = (ncols + block_size - 1) // block_size
    pad_len = n_blocks * block_size - ncols
    w_pad = np.pad(w, ((0, 0), (0, pad_len)), constant_values=0) if pad_len else w
    w_3d = w_pad.reshape(nrows, n_blocks, block_size)
    block_scales32 = np.max(np.abs(w_3d), axis=2)
    block_scales32 = np.where(block_scales32 < 1e-10, 1.0, block_scales32)
    scales_expanded = np.repeat(block_scales32[:, :, np.newaxis], block_size, axis=2)
    ternary_3d = np.clip(np.round(w_3d / scales_expanded).astype(np.int32), -1, 1)
    ternary_flat = ternary_3d.reshape(nrows, n_blocks * block_size)[:, :ncols].astype(np.int8)
    packed_per_row = (ncols + 3) // 4
    full_len = packed_per_row * 4
    out = np.empty(nrows * packed_per_row, dtype=np.uint8)
    for r in range(nrows):
        row = ternary_flat[r, :].astype(np.int32) + 1
        if ncols < full_len:
            row = np.pad(row, (0, full_len - ncols), constant_values=1)
        for i in range(packed_per_row):
            out[r * packed_per_row + i] = (
                (row[i * 4 + 0] << 0) |
                (row[i * 4 + 1] << 2) |
                (row[i * 4 + 2] << 4) |
                (row[i * 4 + 3] << 6)
            )
    header = struct.pack("<BH", block_size, n_blocks) + block_scales32.astype(np.float16).tobytes()
    return header + out.tobytes(), packed_per_row, n_blocks, block_size


def pack_tq1_per_tensor(weights_fp16):
    """TQ1 per-tensor packed (ttype=0, Falcon3 I2_S style).

    Stores fp16 scale + Base-3 packed ternary values.
    Returns (data_bytes, packed_per_row).
    """
    w = weights_fp16.astype(np.float32)
    nrows, ncols = w.shape
    scale = max(np.abs(w).max(), 1e-10)
    ternary = np.clip(np.round(w / scale).astype(np.int32), -1, 1).astype(np.int8)
    packed_per_row = (ncols + 4) // 5
    full_len = packed_per_row * 5
    out = np.empty(nrows * packed_per_row, dtype=np.uint8)
    for r in range(nrows):
        row = (ternary[r, :].astype(np.int32) + 1)  # -1→0, 0→1, 1→2
        if ncols < full_len:
            row = np.pad(row, (0, full_len - ncols), constant_values=1)
        t5 = row[:full_len].reshape(packed_per_row, 5)
        out[r * packed_per_row: (r + 1) * packed_per_row] = (
            (t5 * TQ1_MUL).sum(axis=1).astype(np.uint8)
        )
    header = struct.pack("<e", float(scale))
    return header + out.tobytes(), packed_per_row


def _is_qkv(name):
    return any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"])

def _is_ffn(name):
    return any(x in name for x in ["gate_proj", "up_proj", "down_proj"])


def make(output_path, arch_name, use_tq1=True, corridor=None, head_dim=None):
    """Generate a synthetic ATLAS model file.

    Args:
        output_path: Path to write .atlas file.
        arch_name: Key into ARCHES dict.
        use_tq1: True = ttype=5 for weights, "ttype7" = ttype=7 for QKV/O,
                 False = all fp16.
        corridor: Key into CORRIDORS dict. Overrides tensor types + meta flags.
    """
    cfg = dict(ARCHES[arch_name])
    if head_dim is not None:
        cfg["head_dim"] = head_dim
    core = dict(CORRIDORS[corridor]) if corridor else None
    n_layers = cfg["n_layers"]
    n_layers = cfg["n_layers"]
    n_tensors = n_layers * len(cfg["layer_tensors"]) + len(GLOBAL_TENSORS)
    h = cfg["hidden"]
    i = cfg["inter"]
    v = cfg["vocab"]

    # Ordered tensor names
    tensor_names = []
    for L in range(n_layers):
        for tname in cfg["layer_tensors"]:
            tensor_names.append(f"model.layers.{L}.{tname}")
    tensor_names.extend(GLOBAL_TENSORS)

    rng = np.random.RandomState(42)

    # Generate tensor data
    tensor_entries = []
    for name in tensor_names:
        shape = _shape_of(name, cfg)
        is_norm = len(shape) == 1

        if is_norm:
            data = np.ones(shape, dtype=np.float32) + rng.randn(*shape).astype(np.float32) * 0.01
            tensor_entries.append((name, data, 1))
        elif core is not None:
            if _is_qkv(name):
                data = rng.randn(*shape).astype(np.float32) * 0.1
                tensor_entries.append((name, data, core["qkv_ttype"]))
            elif _is_ffn(name):
                data = rng.randn(*shape).astype(np.float32) * 0.1
                tensor_entries.append((name, data, core["ffn_ttype"]))
            else:
                data = rng.randn(*shape).astype(np.float32) * 0.1
                tensor_entries.append((name, data, 1))
        elif use_tq1 == "ttype7":
            if _is_qkv(name):
                data = rng.randn(*shape).astype(np.float32) * 0.1
                tensor_entries.append((name, data, 7))
            elif _is_ffn(name):
                data = rng.randn(*shape).astype(np.float32) * 0.1
                tensor_entries.append((name, data, 3))
            else:
                data = rng.randn(*shape).astype(np.float32) * 0.1
                tensor_entries.append((name, data, 1))
        elif use_tq1 == "ttype0":
            if "proj" in name:
                data = rng.randn(*shape).astype(np.float32) * 0.1
                tensor_entries.append((name, data, 0))
            else:
                data = rng.randn(*shape).astype(np.float32) * 0.1
                tensor_entries.append((name, data, 1))
        elif use_tq1 and "proj" in name:
            data = rng.randn(*shape).astype(np.float32) * 0.1
            tensor_entries.append((name, data, 5))
        else:
            data = rng.randn(*shape).astype(np.float32) * 0.1
            tensor_entries.append((name, data, 1))

    # ─── Write file ────────────────────────────────────────────────────
    use_f32 = (1 if arch_name == "bitnet" else
               (0 if use_tq1 == "ttype7" else
                (core["use_f32_bypass"] if core else 0)))
    meta = {
        "arch": arch_name,
        "rope_interleaved": cfg["rope_interleaved"],
        "use_f32_bypass": use_f32,
        "rope_theta": cfg["rope_theta"],
        "head_dim": cfg["head_dim"],
        "rope_scale": 1.0,
        "base_seq_len": 2048,
    }
    meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
    meta_size = 4 + len(meta_bytes)

    # Name block
    name_bytes = b"\0".join(n.encode() for n, _, _ in tensor_entries) + b"\0"
    name_block = struct.pack("<I", 4 + len(name_bytes)) + name_bytes

    dir_start = 64 + meta_size
    data_start = dir_start + n_tensors * 12 + len(name_block)

    with open(output_path, "wb") as f:
        # Header (64 bytes)
        hdr = bytearray(64)
        hdr[0:5] = b"ATLAS"
        struct.pack_into("<H", hdr, 5, 8)
        struct.pack_into("<H", hdr, 7, n_layers)
        struct.pack_into("<H", hdr, 9, h)
        struct.pack_into("<H", hdr, 11, i)
        struct.pack_into("<B", hdr, 13, cfg["n_heads"])
        struct.pack_into("<B", hdr, 14, cfg["n_kv_heads"])
        struct.pack_into("<H", hdr, 15, cfg["head_dim"])
        struct.pack_into("<I", hdr, 17, v)
        struct.pack_into("<d", hdr, 21, cfg["rope_theta"])
        struct.pack_into("<I", hdr, 56, len(name_block))
        struct.pack_into("<I", hdr, 60, n_tensors)

        # v8 meta block (right after header)
        f.write(hdr)
        f.write(struct.pack("<I", meta_size))
        f.write(meta_bytes)

        # Directory placeholder
        dir_buf = bytearray(n_tensors * 12)
        f.write(dir_buf)
        # Name block
        f.write(name_block)

        # Write tensor data (32-byte aligned)
        offsets = []
        for idx, (name, data_f32, ttype) in enumerate(tensor_entries):
            offset = f.tell()
            if offset % 32 != 0:
                pad = 32 - (offset % 32)
                f.write(b"\x00" * pad)
                offset += pad

            row_dim = data_f32.shape[0]

            if ttype == 0:
                packed, ppr = pack_tq1_per_tensor(data_f32)
                tens_ttype = 0
            elif ttype == 5:
                packed, ppr, nb, bs = pack_tq1_g128(data_f32)
                tens_ttype = 5
            elif ttype == 7:
                packed, ppr, nb, bs = pack_turboquant(data_f32)
                tens_ttype = 7
            elif ttype == 3:
                # int8 format: [fp16_scale:2][i8_data:rows*cols][row_sums:rows*4]
                max_abs = max(np.max(np.abs(data_f32)), 1e-10)
                scale = max_abs / 127.0
                i8 = np.clip(np.round(data_f32 / scale).astype(np.int32), -128, 127).astype(np.int8)
                row_sums = np.sum(i8.astype(np.int64), axis=1).astype(np.int32)
                packed = (np.float16(scale).tobytes() +
                          i8.tobytes() +
                          row_sums.tobytes())
                ppr = 0
                tens_ttype = 3
            else:
                packed = data_f32.astype(np.float16).tobytes()
                ppr = 0
                tens_ttype = 1

            # Write directory entry
            dir_buf[idx * 12] = tens_ttype
            struct.pack_into("<I", dir_buf, idx * 12 + 1, offset)
            struct.pack_into("<I", dir_buf, idx * 12 + 5, row_dim)
            ppr_clamped = ppr & 0xFFFFFF
            dir_buf[idx * 12 + 9] = ppr_clamped & 0xFF
            dir_buf[idx * 12 + 10] = (ppr_clamped >> 8) & 0xFF
            dir_buf[idx * 12 + 11] = (ppr_clamped >> 16) & 0xFF

            f.write(packed)

        # Rewrite directory
        f.seek(dir_start)
        f.write(dir_buf)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Wrote {output_path}  ({size_kb:.0f} KB, {n_tensors} tensors)")
    return output_path


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "mock/out"
    out_dir = os.path.dirname(base)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for arch in ARCHES:
        path = f"{base}-{arch}.atlas"
        make(path, arch, use_tq1=True)
        make(path.replace(".atlas", "-fp16.atlas"), arch, use_tq1=False)
        print(f"    $ python -c \"import atlas_infer; m = atlas_infer.AtlasModel('{path}')\"")


if __name__ == "__main__":
    main()
