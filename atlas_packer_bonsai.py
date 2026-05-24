#!/usr/bin/env python3
"""Atlas Packer for Qwen3/Bonsai Ternary — FP16 → TQ1.0"""
import struct, numpy as np, json, os, sys
from safetensors import safe_open

TQ1_MUL = np.array([1, 3, 9, 27, 81], dtype=np.uint32)

def pre_shuffle_rows(tensor):
    """Pre-shuffle weight rows to cancel C++ matmul SIMD reorder.

    C++ reorder: output[(r%4)*rows_packed + r//4] = W[r] · act
    We want:     output[r] = W_natural[r] · act

    Store W_shuffled[target] = W_natural[r] where:
      target = (r % rows_packed) * 4 + r // rows_packed
    """
    out_dim = tensor.shape[0]
    assert out_dim % 4 == 0
    rows_packed = out_dim // 4
    out = np.empty_like(tensor)
    for r in range(out_dim):
        target = (r % rows_packed) * 4 + r // rows_packed
        out[target] = tensor[r]
    return out

def ternarize(weights_fp16):
    """Extract scale and quantize FP16 weights to ternary {-1,0,1}.
    Uses 95th percentile for scale (avoids outlier domination).
    Returns (scale_fp16, packed_bytes, packed_per_row)."""
    w = weights_fp16.astype(np.float32)
    scale = float(np.percentile(np.abs(w), 95))
    if scale < 1e-10:
        scale = 1.0
    ternary = np.clip(np.round(w / scale).astype(np.int32), -1, 1).astype(np.int8)  # -1, 0, 1
    in_cols = w.shape[1]
    packed_per_row = (in_cols + 4) // 5
    out_rows = w.shape[0]
    out = np.empty(out_rows * packed_per_row, dtype=np.uint8)
    for r in range(out_rows):
        row = ternary[r, :].astype(np.int32) + 1  # -1→0, 0→1, 1→2
        full_len = packed_per_row * 5
        if in_cols < full_len:
            row = np.pad(row, (0, full_len - in_cols), constant_values=1)
        t5 = row[:full_len].reshape(packed_per_row, 5)
        out[r * packed_per_row : (r + 1) * packed_per_row] = (
            (t5 * TQ1_MUL).sum(axis=1).astype(np.uint8)
        )
    return scale, out.tobytes(), packed_per_row

def ternarize_block_scaled(weights_fp16, block_size=128):
    """Per-block ternary quantization (Bonsai g128 format).

    Per-row block scales: each output row has its own FP16 scales,
    one per group of `block_size` columns. Vectorized numpy.
    Returns (block_scales_bytes, packed_bytes, packed_per_row, n_blocks, block_size).
    """
    w = weights_fp16.astype(np.float32)
    nrows, ncols = w.shape
    n_blocks = (ncols + block_size - 1) // block_size

    # Pad cols to n_blocks * block_size for uniform reshape
    pad_len = n_blocks * block_size - ncols
    w_pad = np.pad(w, ((0, 0), (0, pad_len)), constant_values=0) if pad_len else w

    # Reshape to (nrows, n_blocks, block_size) and compute per-row per-block scales
    w_3d = w_pad.reshape(nrows, n_blocks, block_size)
    block_scales32 = np.max(np.abs(w_3d), axis=2)
    block_scales32 = np.where(block_scales32 < 1e-10, 1.0, block_scales32)

    # Expand scales to full padded shape and quantize
    scales_expanded = np.repeat(block_scales32[:, :, np.newaxis], block_size, axis=2)
    ternary_3d = np.clip(np.round(w_3d / scales_expanded).astype(np.int32), -1, 1)
    # Trim padding and reshape to (nrows, ncols)
    ternary_flat = ternary_3d.reshape(nrows, n_blocks * block_size)[:, :ncols].astype(np.int8)

    # Pack ternary to TQ1 (5 trits/byte)
    packed_per_row = (ncols + 4) // 5
    full_len = packed_per_row * 5
    out = np.empty(nrows * packed_per_row, dtype=np.uint8)
    for r in range(nrows):
        row = ternary_flat[r, :].astype(np.int32) + 1
        if ncols < full_len:
            row = np.pad(row, (0, full_len - ncols), constant_values=1)
        t5 = row[:full_len].reshape(packed_per_row, 5)
        out[r * packed_per_row : (r + 1) * packed_per_row] = (
            (t5 * TQ1_MUL).sum(axis=1).astype(np.uint8)
        )

    # Wire format: [block_size:1][n_blocks:2][scales: nrows*n_blocks*2 fp16][packed_TQ1]
    block_scales = block_scales32.astype(np.float16)
    header = struct.pack('<BH', block_size, n_blocks) + block_scales.tobytes()
    return header + out.tobytes(), packed_per_row, n_blocks, block_size

def get_shard_path(weight_map, tensor_name, model_dir):
    """Resolve which shard file contains a tensor."""
    shard = weight_map.get(tensor_name)
    if shard:
        return os.path.join(model_dir, shard)
    return None

def create_atlas_qwen(model_dir, output_path):
    with open(os.path.join(model_dir, 'config.json')) as f:
        cfg = json.load(f)

    hidden = cfg['hidden_size']
    n_layers = cfg['num_hidden_layers']
    n_heads = cfg['num_attention_heads']
    n_kv_heads = cfg['num_key_value_heads']
    inter = cfg['intermediate_size']
    vocab = cfg['vocab_size']
    head_dim = cfg.get('head_dim', hidden // n_heads)
    rope_theta = cfg.get('rope_theta', 10000.0)
    tie_emb = cfg.get('tie_word_embeddings', False)

    print(f"  Layers:{n_layers} Hidden:{hidden} Heads:{n_heads}/{n_kv_heads}")
    print(f"  Intermediate:{inter} Vocab:{vocab} Head_dim:{head_dim}")
    print(f"  RoPE theta:{rope_theta} Tie embeddings:{tie_emb}")

    idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        weight_map = idx['weight_map']
    else:
        weight_map = {}
        sf_path = os.path.join(model_dir, 'model.safetensors')
        with safe_open(sf_path, framework='np') as sf:
            for k in sf.keys():
                weight_map[k] = 'model.safetensors'

    # Build ordered tensor list (Qwen3 -> ATLAS naming)
    # Order: per-layer tensors, then global tensors
    tensor_names = []
    for L in range(n_layers):
        for tname in [
            f"model.layers.{L}.input_layernorm.weight",
            f"model.layers.{L}.self_attn.q_proj.weight",
            f"model.layers.{L}.self_attn.k_proj.weight",
            f"model.layers.{L}.self_attn.v_proj.weight",
            f"model.layers.{L}.self_attn.o_proj.weight",
            f"model.layers.{L}.self_attn.q_norm.weight",
            f"model.layers.{L}.self_attn.k_norm.weight",
            f"model.layers.{L}.post_attention_layernorm.weight",
            f"model.layers.{L}.mlp.gate_proj.weight",
            f"model.layers.{L}.mlp.up_proj.weight",
            f"model.layers.{L}.mlp.down_proj.weight",
        ]:
            if tname in weight_map:
                tensor_names.append(tname)

    # Global tensors
    for tname in ["model.embed_tokens.weight", "model.norm.weight"]:
        if tname in weight_map:
            tensor_names.append(tname)

    # lm_head: tied embeddings — only add if separate tensor exists
    if "lm_head.weight" in weight_map:
        tensor_names.append("lm_head.weight")

    print(f"  Tensors: {len(tensor_names)}")

    # Load tokenizer
    tokenizer_block = b''
    tok_path = os.path.join(model_dir, 'tokenizer.json')
    if os.path.exists(tok_path):
        with open(tok_path, 'rb') as tf:
            tok_data = tf.read()
        tokenizer_block += struct.pack('<I', len(tok_data)) + tok_data
        cfg_path = os.path.join(model_dir, 'tokenizer_config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'rb') as cf:
                cfg_data = cf.read()
        tokenizer_block += struct.pack('<I', len(cfg_data)) + cfg_data

    # Cache open shard handles: shard_path -> safe_open object
    shard_handles = {}
    def get_tensor(tname):
        sp = get_shard_path(weight_map, tname, model_dir)
        if sp not in shard_handles:
            shard_handles[sp] = safe_open(sp, framework='np')
        return shard_handles[sp].get_tensor(tname)

    with open(output_path, 'wb') as out:
        # Header
        header = bytearray(64)
        header[0:5] = b'ATLAS'
        struct.pack_into('<H', header, 5, 5)  # v5
        struct.pack_into('<H', header, 7, n_layers)
        struct.pack_into('<H', header, 9, hidden)
        struct.pack_into('<H', header, 11, inter)
        struct.pack_into('<B', header, 13, n_heads)
        struct.pack_into('<B', header, 14, n_kv_heads)
        struct.pack_into('<H', header, 15, head_dim)
        struct.pack_into('<I', header, 17, vocab)
        struct.pack_into('<d', header, 21, rope_theta)
        struct.pack_into('<I', header, 56, 0)  # name_block_size placeholder
        struct.pack_into('<I', header, 60, len(tensor_names))
        # EOS/PAD in header bytes 45-52 (reserved area, use bytes 45-48 for eos, 49-52 for pad)
        eos_id = cfg.get('eos_token_id')
        pad_id = cfg.get('pad_token_id')
        if isinstance(eos_id, list): eos_id = eos_id[0] if eos_id else 0
        if isinstance(pad_id, list): pad_id = pad_id[0] if pad_id else 0
        if eos_id is not None: struct.pack_into('<I', header, 45, eos_id)
        if pad_id is not None: struct.pack_into('<I', header, 49, pad_id)

        # Name block
        name_bytes = b''.join(n.encode() + b'\0' for n in tensor_names)
        name_block = struct.pack('<I', 4 + len(name_bytes)) + name_bytes
        struct.pack_into('<I', header, 56, len(name_block))

        out.write(header)

        # Directory placeholder
        data_start = 64 + len(tensor_names) * 12 + len(name_block)
        directory = bytearray(len(tensor_names) * 12)
        current_offset = data_start

        # Write name block
        out.seek(64 + len(tensor_names) * 12)
        out.write(name_block)
        out.seek(data_start)

        for idx, name in enumerate(tensor_names):
            tensor = get_tensor(name)
            is_ternary = 'proj' in name and 'weight' in name

            if is_ternary:
                tensor = pre_shuffle_rows(tensor)
                data_bytes, packed_per_row, n_blocks, block_size = ternarize_block_scaled(tensor)
                # No scale prefix — scales are embedded in ternarize_block_scaled output
                # data_bytes already contains [block_size:1][n_blocks:2][scales][packed_TQ1]
            else:
                packed_per_row = 0
                n_blocks = 0
                block_size = 0
                data_bytes = tensor.astype(np.float16).tobytes()

            if current_offset % 32 != 0:
                pad = 32 - (current_offset % 32)
                current_offset += pad
                out.write(b'\x00' * pad)

            if is_ternary:
                ttype = 5  # TQ1 packed with per-block scales (g128)
                row_dim = tensor.shape[0]
            elif 'norm' in name or 'embed' in name:
                ttype = 1  # FP16 vector
                row_dim = tensor.shape[0]
            else:
                ttype = 2  # FP16 matrix
                row_dim = tensor.shape[0]

            directory[idx*12] = ttype
            struct.pack_into('<I', directory, idx*12 + 1, current_offset)
            struct.pack_into('<I', directory, idx*12 + 5, row_dim)
            ppr = packed_per_row & 0xFFFFFF
            directory[idx*12 + 9] = ppr & 0xFF
            directory[idx*12 + 10] = (ppr >> 8) & 0xFF
            directory[idx*12 + 11] = (ppr >> 16) & 0xFF

            out.write(data_bytes)
            current_offset += len(data_bytes)

            if idx % 8 == 0:
                print(f"  [{idx}/{len(tensor_names)}] {name[:55]:55s} {len(data_bytes)/1024:7.1f}KB")

        # Tokenizer block
        if tokenizer_block:
            tokenizer_offset = current_offset
            if tokenizer_offset % 32 != 0:
                pad = 32 - (tokenizer_offset % 32)
                tokenizer_offset += pad
                out.write(b'\x00' * pad)
            out.write(tokenizer_block)
            struct.pack_into('<I', header, 29, len(tokenizer_block))
            struct.pack_into('<I', header, 33, tokenizer_offset)

        out.flush()
        out.seek(64)
        out.write(directory)
        out.seek(0)
        out.write(header)

    for h in shard_handles.values():
        pass  # safe_open is context-manager only, no close
    total_gb = current_offset / 1024**3
    print(f"[ATLAS] Done! {total_gb:.2f} GB | {len(tensor_names)} tensors")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python atlas_packer_bonsai.py <model_dir> <output.atlas>")
        sys.exit(1)
    create_atlas_qwen(sys.argv[1], sys.argv[2])
