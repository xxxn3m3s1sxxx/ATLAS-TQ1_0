#!/usr/bin/env python3
"""Generate minimal .atlas test fixtures for all supported architectures.

Produces 6 files: {falcon3,qwen3,bitnet}_{v6,v8}.atlas
Each is ~100 KB with 1 layer of random data.

Usage:
  python tests/generate_test_fixtures.py [outdir]
"""

import struct, os, sys, json

H, I = 64, 128
nH, nKV, hd = 2, 1, 32
V = 67

def _pc(cols):
    return (cols + 4) // 5

def _tq1(rows, cols):
    pc = _pc(cols)
    return struct.pack("<H", 15360) + bytes(rows * pc)

def _fp16(rows, cols=None):
    n = rows * cols if cols else rows
    return bytes(n * 2)

ARCHES = {
    "trilm": {
        "stride": 11,
        "meta": {"arch": "trilm", "use_f32_bypass": 1},
        "tensors": lambda L: [
            (f"model.layers.{L}.input_layernorm.weight", 1, H, lambda: _fp16(H)),
            (f"model.layers.{L}.self_attn.attn_sub_norm.weight", 1, H, lambda: _fp16(H)),
            (f"model.layers.{L}.self_attn.q_proj.weight", 0, nH*hd, lambda: _tq1(nH*hd, H)),
            (f"model.layers.{L}.self_attn.k_proj.weight", 0, nKV*hd, lambda: _tq1(nKV*hd, H)),
            (f"model.layers.{L}.self_attn.v_proj.weight", 0, nKV*hd, lambda: _tq1(nKV*hd, H)),
            (f"model.layers.{L}.self_attn.o_proj.weight", 0, H, lambda: _tq1(H, nH*hd)),
            (f"model.layers.{L}.post_attention_layernorm.weight", 1, H, lambda: _fp16(H)),
            (f"model.layers.{L}.mlp.gate_proj.weight", 0, I, lambda: _tq1(I, H)),
            (f"model.layers.{L}.mlp.up_proj.weight", 0, I, lambda: _tq1(I, H)),
            (f"model.layers.{L}.mlp.down_proj.weight", 0, H, lambda: _tq1(H, I)),
            (f"model.layers.{L}.mlp.ffn_sub_norm.weight", 1, I, lambda: _fp16(I)),
        ],
    },

    "falcon3": {
        "stride": 9,
        "meta": {"arch": "falcon3", "rope_interleaved": 1},
        "tensors": lambda L: [
            (f"model.layers.{L}.input_layernorm.weight", 1, H, lambda: _fp16(H)),
            (f"model.layers.{L}.self_attn.q_proj.weight", 0, nH*hd, lambda: _tq1(nH*hd, H)),
            (f"model.layers.{L}.self_attn.k_proj.weight", 0, nKV*hd, lambda: _tq1(nKV*hd, H)),
            (f"model.layers.{L}.self_attn.v_proj.weight", 0, nKV*hd, lambda: _tq1(nKV*hd, H)),
            (f"model.layers.{L}.self_attn.o_proj.weight", 0, H, lambda: _tq1(H, nH*hd)),
            (f"model.layers.{L}.post_attention_layernorm.weight", 1, H, lambda: _fp16(H)),
            (f"model.layers.{L}.mlp.gate_proj.weight", 0, I, lambda: _tq1(I, H)),
            (f"model.layers.{L}.mlp.up_proj.weight", 0, I, lambda: _tq1(I, H)),
            (f"model.layers.{L}.mlp.down_proj.weight", 0, H, lambda: _tq1(H, I)),
        ],
    },
    "qwen3": {
        "stride": 11,
        "meta": {"arch": "qwen3", "has_qk_norm": 1, "use_f32_bypass": 1, "rope_scale": 4.0},
        "tensors": lambda L: [
            (f"model.layers.{L}.input_layernorm.weight", 1, H, lambda: _fp16(H)),
            (f"model.layers.{L}.self_attn.q_proj.weight", 0, nH*hd, lambda: _tq1(nH*hd, H)),
            (f"model.layers.{L}.self_attn.k_proj.weight", 0, nKV*hd, lambda: _tq1(nKV*hd, H)),
            (f"model.layers.{L}.self_attn.v_proj.weight", 0, nKV*hd, lambda: _tq1(nKV*hd, H)),
            (f"model.layers.{L}.self_attn.o_proj.weight", 0, H, lambda: _tq1(H, nH*hd)),
            (f"model.layers.{L}.self_attn.q_norm.weight", 1, hd, lambda: _fp16(hd)),
            (f"model.layers.{L}.self_attn.k_norm.weight", 1, hd, lambda: _fp16(hd)),
            (f"model.layers.{L}.post_attention_layernorm.weight", 1, H, lambda: _fp16(H)),
            (f"model.layers.{L}.mlp.gate_proj.weight", 0, I, lambda: _tq1(I, H)),
            (f"model.layers.{L}.mlp.up_proj.weight", 0, I, lambda: _tq1(I, H)),
            (f"model.layers.{L}.mlp.down_proj.weight", 0, H, lambda: _tq1(H, I)),
        ],
    },
    "bitnet": {
        "stride": 11,
        "meta": {"arch": "bitnet", "use_relu2": 1, "use_f32_bypass": 1},
        "tensors": lambda L: [
            (f"model.layers.{L}.input_layernorm.weight", 1, H, lambda: _fp16(H)),
            (f"model.layers.{L}.self_attn.attn_sub_norm.weight", 1, H, lambda: _fp16(H)),
            (f"model.layers.{L}.self_attn.q_proj.weight", 0, nH*hd, lambda: _tq1(nH*hd, H)),
            (f"model.layers.{L}.self_attn.k_proj.weight", 0, nKV*hd, lambda: _tq1(nKV*hd, H)),
            (f"model.layers.{L}.self_attn.v_proj.weight", 0, nKV*hd, lambda: _tq1(nKV*hd, H)),
            (f"model.layers.{L}.self_attn.o_proj.weight", 0, H, lambda: _tq1(H, nH*hd)),
            (f"model.layers.{L}.post_attention_layernorm.weight", 1, H, lambda: _fp16(H)),
            (f"model.layers.{L}.mlp.gate_proj.weight", 0, I, lambda: _tq1(I, H)),
            (f"model.layers.{L}.mlp.up_proj.weight", 0, I, lambda: _tq1(I, H)),
            (f"model.layers.{L}.mlp.down_proj.weight", 0, H, lambda: _tq1(H, I)),
            (f"model.layers.{L}.mlp.ffn_sub_norm.weight", 1, I, lambda: _fp16(I)),
        ],
    },
}

def generate(arch_key, version, outdir):
    cfg = ARCHES[arch_key]
    stride = cfg["stride"]
    n_layers = 1

    tensors = []
    tensors.append(("model.embed_tokens.weight", 1, V, lambda: _fp16(V, H)))
    for L in range(n_layers):
        tensors.extend(cfg["tensors"](L))
    tensors.append(("model.norm.weight", 1, H, lambda: _fp16(H)))
    tensors.append(("lm_head.weight", 2, V, lambda: _fp16(V, H)))

    N = len(tensors)

    # Build name block
    name_data = bytearray()
    for name, *_ in tensors:
        name_data.extend(name.encode("utf-8") + b"\0")
    name_block = struct.pack("<I", len(name_data)) + name_data
    name_block_sz = len(name_block)

    # Build directory entries + data
    data_blobs = []
    entries = []
    for name, ttype, row_dim, data_fn in tensors:
        blob = data_fn()
        if ttype == 0:
            pcol = _pc(row_dim)  # NOTE: we need cols, not rows — but for mock both are H/I
            # Actually packed_cols = number of 5-trit groups = cols / 5
            # The engine reads row_dim as the first dimension of the weight matrix
            # For q_proj: weight[TQ1: out_dim × (in_dim/5)] → row_dim = out_dim, col_dim = in_dim
            # We need the actual column dimension for packed_cols
            # Re-derive from the tensor size
            data_sz = len(blob)
            actual_bytes = data_sz - 2  # minus scale
            pcol = actual_bytes // row_dim  # packed bytes per row
        else:
            pcol = 0
        data_blobs.append(blob)
        entries.append((ttype, row_dim, pcol))

    # Compute offsets
    if version == 8:
        meta = cfg["meta"]
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        meta_sz = 4 + len(meta_bytes)
        dir_offset = 64 + meta_sz
    else:
        meta_sz = 0
        dir_offset = 64

    name_offset = dir_offset + N * 12
    data_offset = name_offset + name_block_sz

    offsets = []
    off = data_offset
    for blob in data_blobs:
        offsets.append(off)
        off += len(blob)

    # Write file
    name = f"{arch_key}_v{version}.atlas"
    path = os.path.join(outdir, name)
    with open(path, "wb") as f:
        hdr = bytearray(64)
        hdr[0:5] = b"ATLAS"
        struct.pack_into("<H", hdr, 5, version)
        struct.pack_into("<H", hdr, 7, n_layers)
        struct.pack_into("<H", hdr, 9, H)
        struct.pack_into("<H", hdr, 11, I)
        hdr[13] = nH
        hdr[14] = nKV
        struct.pack_into("<H", hdr, 15, hd)
        struct.pack_into("<I", hdr, 17, V)
        struct.pack_into("<d", hdr, 21, 10000.0)
        struct.pack_into("<i", hdr, 29, 0)
        struct.pack_into("<I", hdr, 33, 0)
        struct.pack_into("<i", hdr, 37, 0)
        struct.pack_into("<I", hdr, 41, 0)
        struct.pack_into("<I", hdr, 45, 0)
        struct.pack_into("<I", hdr, 49, 0)
        hdr[53] = 0
        struct.pack_into("<i", hdr, 56, name_block_sz)
        struct.pack_into("<i", hdr, 60, N)
        f.write(hdr)

        # v8 meta block
        if version == 8:
            f.write(struct.pack("<I", meta_sz))
            f.write(meta_bytes)

        # Directory
        for i, (ttype, row_dim, pcol) in enumerate(entries):
            f.write(struct.pack("<B", ttype))
            f.write(struct.pack("<I", offsets[i]))
            f.write(struct.pack("<I", row_dim))
            f.write(struct.pack("<I", pcol)[:3])

        # Name block
        f.write(name_block)

        # Data
        for blob in data_blobs:
            f.write(blob)

    kb = os.path.getsize(path) / 1024
    print(f"  {name}: {N} tensors, {kb:.1f} KB, stride={stride}, version={version}")
    return path

def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
    os.makedirs(outdir, exist_ok=True)
    print(f"Generating test fixtures in {outdir}...")
    for arch in ["falcon3", "qwen3", "bitnet", "trilm"]:
        for ver in [6, 8]:
            generate(arch, ver, outdir)
    print("Done.")

if __name__ == "__main__":
    main()
