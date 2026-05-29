#!/usr/bin/env python3
"""Create a minimal valid .atlas file for CI smoke testing.

Generates a 1-layer model with random TQ1-packed weights + fp16 embed/norm.
Output is ~60 KB — fast to download and load in CI.
"""
import struct, os, sys
import numpy as np

def _tq1_packed_cols(cols):
    return (cols + 4) // 5

def _tq1_ternary(rows, cols):
    """Generate random ternary (±1,0) TQ1-packed bytes + fp16 scale."""
    pc = _tq1_packed_cols(cols)
    scale_f16 = np.float16(1.0 / max(rows, cols))
    scale = struct.pack('<H', scale_f16.view(np.uint16))
    n = rows * pc
    tern = np.random.randint(0, 3, n * 5, dtype=np.uint8)
    packed = np.zeros(n, dtype=np.uint8)
    mul = np.array([1, 3, 9, 27, 81], dtype=np.uint32)
    packed[:] = (tern.reshape(-1, 5) * mul).sum(axis=1).astype(np.uint8)
    return scale + packed.tobytes()

def _fp16_data(arr):
    """Serialize float32 array as raw fp16 bytes."""
    return arr.astype(np.float16).tobytes()

def create_mock_atlas(path, hidden=64, inter=128, n_heads=2, n_kv_heads=1,
                       head_dim=32, vocab=67, n_layers=1):
    np.random.seed(42)
    rng = np.random.RandomState(42)

    # ─── Build tensor list ─────────────────────────────────────────────
    # Each entry: (name, ttype, row_dim, packed_cols, data_bytes)
    # ttype 0 = TQ1 packed (2-byte scale + row_dim*packed_cols bytes)
    # ttype 1 = fp16 vector or matrix (norm = 1D, embed = 2D)
    # ttype 2 = fp16 matrix (lm_head)
    H, I = hidden, inter
    nH, nKV, hd = n_heads, n_kv_heads, head_dim
    V = vocab

    tensors = []

    def add(name, ttype, row_dim, data_func):
        tensors.append((name, ttype, row_dim, data_func))

    # Embed tokens: fp16 [V, H]
    add("model.embed_tokens.weight", 1, V,
        lambda: _fp16_data(rng.randn(V, H).astype(np.float32)))

    # Per-layer tensors (Falcon3: 9 per layer)
    for L in range(n_layers):
        pre = f"model.layers.{L}."
        add(pre + "input_layernorm.weight", 1, H,
            lambda: _fp16_data(rng.randn(H).astype(np.float32)))
        add(pre + "self_attn.q_proj.weight", 0, nH * hd,
            lambda: _tq1_ternary(nH * hd, H))
        add(pre + "self_attn.k_proj.weight", 0, nKV * hd,
            lambda: _tq1_ternary(nKV * hd, H))
        add(pre + "self_attn.v_proj.weight", 0, nKV * hd,
            lambda: _tq1_ternary(nKV * hd, H))
        add(pre + "self_attn.o_proj.weight", 0, H,
            lambda: _tq1_ternary(H, nH * hd))
        add(pre + "post_attention_layernorm.weight", 1, H,
            lambda: _fp16_data(rng.randn(H).astype(np.float32)))
        add(pre + "mlp.gate_proj.weight", 0, I,
            lambda: _tq1_ternary(I, H))
        add(pre + "mlp.up_proj.weight", 0, I,
            lambda: _tq1_ternary(I, H))
        add(pre + "mlp.down_proj.weight", 0, H,
            lambda: _tq1_ternary(H, I))

    # Final RMSNorm + LM head
    add("model.norm.weight", 1, H,
        lambda: _fp16_data(rng.randn(H).astype(np.float32)))
    add("lm_head.weight", 2, V,
        lambda: _fp16_data(rng.randn(V, H).astype(np.float32)))

    N = len(tensors)

    # ─── Compute file layout ──────────────────────────────────────────
    name_block_data = bytearray()
    for name, *_ in tensors:
        name_bytes = name.encode('utf-8')
        name_block_data.extend(name_bytes)
        name_block_data.append(0)
    name_block_size = 4 + len(name_block_data)
    name_block = struct.pack('<I', len(name_block_data)) + name_block_data

    dir_offset = 64
    name_offset = dir_offset + N * 12
    data_offset = name_offset + name_block_size

    # Compute tensor data sizes
    data_sizes = []
    data_blobs = []
    for _, ttype, row_dim, data_func in tensors:
        blob = data_func()
        if ttype == 0:
            # blob = scale(2) + packed bytes
            used_cols = len(blob) - 2
            packed_cols = used_cols // row_dim
        else:
            packed_cols = 0
        data_sizes.append(len(blob))
        data_blobs.append((blob, packed_cols))

    # Assign file offsets (sequential after name block)
    offsets = []
    off = data_offset
    for blob, _ in data_blobs:
        offsets.append(off)
        off += len(blob)

    # ─── Write file ──────────────────────────────────────────────────
    with open(path, 'wb') as f:
        # Header (64 bytes)
        hdr = bytearray(64)
        hdr[0:5] = b'ATLAS'
        struct.pack_into('<H', hdr, 5, 6)        # version = 6
        struct.pack_into('<H', hdr, 7, n_layers)
        struct.pack_into('<H', hdr, 9, H)
        struct.pack_into('<H', hdr, 11, I)
        hdr[13] = nH
        hdr[14] = nKV
        struct.pack_into('<H', hdr, 15, hd)
        struct.pack_into('<I', hdr, 17, V)
        struct.pack_into('<d', hdr, 21, 10000.0)  # rope_theta
        # tokenizer_size=0, tokenizer_offset=0 (no embedded tokenizer)
        struct.pack_into('<i', hdr, 29, 0)
        struct.pack_into('<I', hdr, 33, 0)
        # v6 binary tokenizer: none
        struct.pack_into('<i', hdr, 37, 0)
        struct.pack_into('<I', hdr, 41, 0)
        # EOS=0, PAD=0 (use defaults)
        struct.pack_into('<I', hdr, 45, 0)  # eos
        struct.pack_into('<I', hdr, 49, 0)  # pad
        hdr[53] = 0  # model_flags
        struct.pack_into('<i', hdr, 56, name_block_size)
        struct.pack_into('<i', hdr, 60, N)
        f.write(hdr)

        # Directory (N × 12 bytes)
        for i, (_, ttype, row_dim, _) in enumerate(tensors):
            blob, packed_cols = data_blobs[i]
            f.write(struct.pack('<B', ttype))
            f.write(struct.pack('<I', offsets[i]))
            f.write(struct.pack('<I', row_dim))
            # packed_cols as 3-byte little-endian
            f.write(struct.pack('<I', packed_cols)[:3])

        # Name block
        f.write(name_block)

        # Tensor data
        for blob, _ in data_blobs:
            f.write(blob)

    size_kb = os.path.getsize(path) / 1024
    print(f"[Mock] Created {path} ({N} tensors, {size_kb:.1f} KB, "
          f"{n_layers}L {H}H {I}I {nH}/{nKV}h {hd}hd {V}vocab)")
    return path


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mock.atlas")
    create_mock_atlas(out)
