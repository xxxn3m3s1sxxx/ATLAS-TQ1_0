#!/usr/bin/env python3
"""Atlas Packer for BitNet b1.58 — BF16 → TQ1.0 g128 block-scaled ternary.
BitNet models from Microsoft: ab initio QAT ternary LLMs with LLaMA-style
architecture + SubLN (attn_sub_norm, ffn_sub_norm) + ReLU² activation.
Stride=11: (ln1, q, k, v, o, ln2, gate, up, down, attn_sub_norm, ffn_sub_norm)
"""
import struct, numpy as np, json, os, sys

TQ1_MUL = np.array([1, 3, 9, 27, 81], dtype=np.uint32)

def pre_shuffle_rows(tensor):
    out_dim = tensor.shape[0]
    assert out_dim % 4 == 0
    rows_packed = out_dim // 4
    out = np.empty_like(tensor)
    for r in range(out_dim):
        target = (r % rows_packed) * 4 + r // rows_packed
        out[target] = tensor[r]
    return out

def ternarize_per_tensor_absmean(weights_fp16, block_size=128, gamma=None):
    """BitNet per-tensor absmean ternarization.
    gamma = mean(abs(W)) [or use provided gamma], round(W/gamma) -> {-1, 0, +1}
    All blocks share the same gamma, stored in g128 block format.
    """
    w = weights_fp16.astype(np.float32)
    nrows, ncols = w.shape
    n_blocks = (ncols + block_size - 1) // block_size

    # Per-tensor absmean gamma (use provided or compute)
    if gamma is None:
        gamma = np.abs(w).mean()
    if gamma < 1e-10:
        gamma = 1.0

    print(f"    gamma={gamma:.6f} range=[{w.min():.4f},{w.max():.4f}]")

    pad_len = n_blocks * block_size - ncols
    w_pad = np.pad(w, ((0, 0), (0, pad_len)), constant_values=0) if pad_len else w
    w_3d = w_pad.reshape(nrows, n_blocks, block_size)
    ternary_3d = np.clip(np.round(w_3d / gamma).astype(np.int32), -1, 1)
    ternary_flat = ternary_3d.reshape(nrows, n_blocks * block_size)[:, :ncols].astype(np.int8)

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

    # All blocks share the same gamma
    block_scales = np.full((nrows, n_blocks), gamma, dtype=np.float16)
    header = struct.pack('<BH', block_size, n_blocks) + block_scales.tobytes()
    return header + out.tobytes(), packed_per_row, n_blocks, block_size

def ternarize_per_row_max(weights_fp16, block_size=128):
    w = weights_fp16.astype(np.float32)
    nrows, ncols = w.shape

    packed_per_row = (ncols + 4) // 5
    full_len = packed_per_row * 5
    n_blocks = (full_len + block_size - 1) // block_size

    row_scales = np.max(np.abs(w), axis=1, keepdims=True)
    row_scales = np.where(row_scales < 1e-10, 1.0, row_scales)

    w_ternary = np.clip(np.round(w / row_scales).astype(np.int32), -1, 1)

    out = np.empty(nrows * packed_per_row, dtype=np.uint8)
    for r in range(nrows):
        row = w_ternary[r, :].astype(np.int32) + 1
        if ncols < full_len:
            row = np.pad(row, (0, full_len - ncols), constant_values=1)
        t5 = row[:full_len].reshape(packed_per_row, 5)
        out[r * packed_per_row : (r + 1) * packed_per_row] = (
            (t5 * TQ1_MUL).sum(axis=1).astype(np.uint8)
        )
    block_scales32 = np.tile(row_scales.astype(np.float32), (1, n_blocks))
    block_scales = block_scales32.astype(np.float16)
    header = struct.pack('<BH', block_size, n_blocks) + block_scales.tobytes()
    return header + out.tobytes(), packed_per_row, n_blocks, block_size

def ternarize_block_scaled(weights_fp16, block_size=128, mp_shards=1):
    w = weights_fp16.astype(np.float32)
    nrows, ncols = w.shape
    n_blocks = (ncols + block_size - 1) // block_size

    if mp_shards > 1:
        shard_width = ncols // mp_shards
        blocks_per_shard = shard_width // block_size
        shard_scales = np.array([
            np.mean(np.abs(w[:, s*shard_width:(s+1)*shard_width]))
            for s in range(mp_shards)
        ])
        shard_scales = np.maximum(shard_scales, 1e-10)
        block_scales32 = np.zeros((nrows, n_blocks), dtype=np.float32)
        for s in range(mp_shards):
            b_start = s * blocks_per_shard
            b_end = (s + 1) * blocks_per_shard if s < mp_shards - 1 else n_blocks
            block_scales32[:, b_start:b_end] = shard_scales[s]
    else:
        pad_len = n_blocks * block_size - ncols
        w_pad = np.pad(w, ((0, 0), (0, pad_len)), constant_values=0) if pad_len else w
        w_3d = w_pad.reshape(nrows, n_blocks, block_size)
        block_scales32 = np.max(np.abs(w_3d), axis=2)
        block_scales32 = np.where(block_scales32 < 1e-10, 1.0, block_scales32)

    scales_expanded = np.repeat(block_scales32[:, :, np.newaxis], block_size, axis=2)
    pad_len = n_blocks * block_size - ncols
    w_pad = np.pad(w, ((0, 0), (0, pad_len)), constant_values=0) if pad_len else w
    w_3d = w_pad.reshape(nrows, n_blocks, block_size)
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
        out[r * packed_per_row : (r + 1) * packed_per_row] = (
            (t5 * TQ1_MUL).sum(axis=1).astype(np.uint8)
        )
    block_scales = block_scales32.astype(np.float16)
    header = struct.pack('<BH', block_size, n_blocks) + block_scales.tobytes()
    return header + out.tobytes(), packed_per_row, n_blocks, block_size

def get_shard_path(weight_map, tensor_name, model_dir):
    shard = weight_map.get(tensor_name)
    return os.path.join(model_dir, shard) if shard else None

def create_atlas_bitnet(model_dir, output_path):
    with open(os.path.join(model_dir, 'config.json')) as f:
        cfg = json.load(f)

    hidden = cfg['hidden_size']
    n_layers = cfg['num_hidden_layers']
    n_heads = cfg['num_attention_heads']
    n_kv_heads = cfg.get('num_key_value_heads', n_heads)
    inter = cfg['intermediate_size']
    vocab = cfg['vocab_size']
    head_dim = cfg.get('head_dim', hidden // n_heads)
    rope_theta = cfg.get('rope_theta', 10000.0)
    tie_emb = cfg.get('tie_word_embeddings', False)
    mp_shards = cfg.get('pretraining_model_parallel', 1)

    print(f"[BitNet] {n_layers}L:{hidden}:{inter} Heads:{n_heads}/{n_kv_heads}")
    print(f"  Vocab:{vocab} Head_dim:{head_dim} RoPE theta:{rope_theta}")
    print(f"  Tie embeddings:{tie_emb} MP shards:{mp_shards}")

    # Build weight map from safetensors index or single file
    idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    weight_map = {}
    has_bf16 = True
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        weight_map = idx['weight_map']
    else:
        sf_path = os.path.join(model_dir, 'model.safetensors')
        if os.path.exists(sf_path):
            with open(sf_path, 'rb') as sf:
                hl = struct.unpack('<Q', sf.read(8))[0]
                h0 = json.loads(sf.read(hl))
                for k in h0:
                    if k != '__metadata__':
                        weight_map[k] = 'model.safetensors'
        else:
            print("Error: no safetensors file found")
            sys.exit(1)

    # BF16 reader
    def read_tensor(tname):
        sp = weight_map.get(tname)
        if not sp:
            raise KeyError(f'Tensor {tname} not found')
        spath = os.path.join(model_dir, sp) if os.sep not in sp else sp
        with open(spath, 'rb') as f:
            hl = struct.unpack('<Q', f.read(8))[0]
            hdr = json.loads(f.read(hl))
            info = hdr[tname]
            start, end = info['data_offsets']
            f.seek(8 + hl + start)
            data = f.read(end - start)
        if info['dtype'] == 'BF16':
            arr = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16
            fp32 = arr.view(np.float32).reshape(info['shape'])
            # Zero out garbage rows (BF16 exponent >= 250, value > 1e10)
            if 'embed_tokens' in tname and fp32.ndim == 2:
                H = fp32.shape[1]
                for r in range(fp32.shape[0]):
                    overflow_mask = np.abs(fp32[r]) > 1e10
                    n_overflow = np.sum(overflow_mask)
                    if n_overflow > 0:
                        good_rms = np.sqrt(np.mean(fp32[r][~overflow_mask] ** 2))
                        print(f"  Clean row {r}: {n_overflow}/{H} overflow"
                              f", good RMS={good_rms:.4f}, zeroing overflow")
                        fp32[r][overflow_mask] = 0.0
            fp32 = np.clip(fp32, -65504.0, 65504.0)
            return fp32.astype(np.float16)
        elif info['dtype'] == 'F16':
            return np.frombuffer(data, dtype=np.float16).reshape(info['shape'])
        elif info['dtype'] == 'F32':
            return np.frombuffer(data, dtype=np.float32).reshape(info['shape'])
        raise TypeError(f'Unsupported dtype: {info["dtype"]} for {tname}')

    # Ordered tensor list: stride=11 (BitNet SubLN variant)
    tensor_names = []
    for L in range(n_layers):
        for tname in [
            f"model.layers.{L}.input_layernorm.weight",
            f"model.layers.{L}.self_attn.q_proj.weight",
            f"model.layers.{L}.self_attn.k_proj.weight",
            f"model.layers.{L}.self_attn.v_proj.weight",
            f"model.layers.{L}.self_attn.o_proj.weight",
            f"model.layers.{L}.post_attention_layernorm.weight",
            f"model.layers.{L}.mlp.gate_proj.weight",
            f"model.layers.{L}.mlp.up_proj.weight",
            f"model.layers.{L}.mlp.down_proj.weight",
            f"model.layers.{L}.self_attn.attn_sub_norm.weight",
            f"model.layers.{L}.mlp.ffn_sub_norm.weight",
        ]:
            if tname in weight_map:
                tensor_names.append(tname)
            else:
                print(f"  WARNING: {tname} not found in weight map")

    # Global tensors
    for tname in ["model.embed_tokens.weight", "model.norm.weight"]:
        if tname in weight_map:
            tensor_names.append(tname)

    # No lm_head if tie_word_embeddings
    if not tie_emb and "lm_head.weight" in weight_map:
        tensor_names.append("lm_head.weight")

    print(f"  Tensors: {len(tensor_names)}")

    # Preload weight_scales (used as per-tensor gamma for QAT-trained ternary)
    scales = {}
    for tname in weight_map:
        if tname.endswith('weight_scale'):
            sc = read_tensor(tname)
            scales[tname] = float(sc) if np.isscalar(sc) else float(sc.flat[0])
    print(f"  Scales loaded: {len(scales)}")

    # Load tokenizer
    tokenizer_block = b''
    tok_path = os.path.join(model_dir, 'tokenizer.json')
    if os.path.exists(tok_path):
        with open(tok_path, 'rb') as tf:
            tok_data = tf.read()
        tokenizer_block += struct.pack('<I', len(tok_data)) + tok_data
        cfg_path = os.path.join(model_dir, 'tokenizer_config.json')
        cfg_data = b''
        if os.path.exists(cfg_path):
            with open(cfg_path, 'rb') as cf:
                cfg_data = cf.read()
        tokenizer_block += struct.pack('<I', len(cfg_data)) + cfg_data
    else:
        print("  WARNING: no tokenizer.json found, trying .model...")
        tok_path = os.path.join(model_dir, 'tokenizer.model')
        if os.path.exists(tok_path):
            with open(tok_path, 'rb') as tf:
                tok_data = tf.read()
            tokenizer_block += struct.pack('<I', len(tok_data)) + tok_data

    with open(output_path, 'wb') as out:
        header = bytearray(64)
        header[0:5] = b'ATLAS'
        struct.pack_into('<H', header, 5, 5)           # format version
        struct.pack_into('<H', header, 7, n_layers)
        struct.pack_into('<H', header, 9, hidden)
        struct.pack_into('<H', header, 11, inter)
        struct.pack_into('<B', header, 13, n_heads)
        struct.pack_into('<B', header, 14, n_kv_heads)
        struct.pack_into('<H', header, 15, head_dim)
        struct.pack_into('<I', header, 17, vocab)
        struct.pack_into('<d', header, 21, rope_theta)
        struct.pack_into('<I', header, 56, 0)          # name_block_size placeholder
        struct.pack_into('<I', header, 60, len(tensor_names))
        eos_id = cfg.get('eos_token_id')
        pad_id = cfg.get('pad_token_id')
        if isinstance(eos_id, list): eos_id = eos_id[0] if eos_id else 0
        if isinstance(pad_id, list): pad_id = pad_id[0] if pad_id else 0
        if eos_id is not None: struct.pack_into('<I', header, 45, eos_id)
        if pad_id is not None: struct.pack_into('<I', header, 49, pad_id)
        # Byte 53: model_flags — BitNet b1.58 uses ReLU² (hidden_act: "relu2" per Microsoft config)
        struct.pack_into('<B', header, 53, 1 << 3)

        name_bytes = b''.join(n.encode() + b'\0' for n in tensor_names)
        name_block = struct.pack('<I', 4 + len(name_bytes)) + name_bytes
        struct.pack_into('<I', header, 56, len(name_block))

        out.write(header)
        data_start = 64 + len(tensor_names) * 12 + len(name_block)
        directory = bytearray(len(tensor_names) * 12)
        current_offset = data_start

        out.seek(64 + len(tensor_names) * 12)
        out.write(name_block)
        out.seek(data_start)

        for idx, name in enumerate(tensor_names):
            tensor = read_tensor(name)
            is_ternary = 'proj' in name and 'weight' in name

            if is_ternary:
                tensor = pre_shuffle_rows(tensor)
                sname = name.replace('.weight', '.weight_scale')
                gamma = scales.get(sname, None)
                data_bytes, packed_per_row, n_blocks, block_size = ternarize_per_tensor_absmean(tensor, gamma=gamma)
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
                ttype = 5
                row_dim = tensor.shape[0]
            elif 'norm' in name or 'embed' in name:
                ttype = 1
                row_dim = tensor.shape[0]
            else:
                ttype = 2
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

    total_gb = current_offset / 1024**3
    print(f"[ATLAS] Done! {total_gb:.2f} GB | {len(tensor_names)} tensors")

# ─── Packed model converter (reads microsoft/bitnet-b1.58-2B-4T U8 format) ───

def unpack_2bit_ternary_to_int8(packed, per_tensor_gamma=1.0, out_dim=None):
    """Unpack 2-bit packed ternary → int8 {-1, 0, +1} 2D array.
    
    packed: np.uint8 array of shape [ceil(out/4), in]
    Returns: np.int8 array of shape [out, in]
    """
    B, C = packed.shape
    out = out_dim or (B * 4)
    unpacked = np.zeros((out, C), dtype=np.int8)
    for ur in range(B):
        row_vals = packed[ur].astype(np.uint32)
        for k in range(4):
            shift = k * 2  # bits 1-0 (k=0) to 7-6 (k=3) — HF I2_S order
            v = (row_vals >> shift) & 3
            unpacked[k * B + ur, :] = v.astype(np.int8) - 1  # {0,1,2} → {-1,0,+1}
    return unpacked

def pack_int8_ternary_to_tq1(ternary, per_tensor_scale, block_size=128):
    """Pack int8 {-1, 0, +1} ternary to TQ1.0 format with block scales.
    
    All blocks share the same per-tensor scale. Lossless if values are {-1,0,+1}.
    Returns: (data_bytes, packed_per_row, n_blocks, block_size)
    """
    nrows, ncols = ternary.shape
    n_blocks = (ncols + block_size - 1) // block_size
    packed_per_row = (ncols + 4) // 5
    full_len = packed_per_row * 5
    out = np.empty(nrows * packed_per_row, dtype=np.uint8)
    for r in range(nrows):
        row = ternary[r, :].astype(np.int32) + 1  # {-1,0,+1} → {0,1,2}
        if ncols < full_len:
            row = np.pad(row, (0, full_len - ncols), constant_values=1)
        t5 = row[:full_len].reshape(packed_per_row, 5)
        out[r * packed_per_row : (r + 1) * packed_per_row] = (
            (t5 * TQ1_MUL).sum(axis=1).astype(np.uint8)
        )
    block_scales = np.full((nrows, n_blocks), per_tensor_scale, dtype=np.float16)
    header = struct.pack('<BH', block_size, n_blocks) + block_scales.tobytes()
    return header + out.tobytes(), packed_per_row, n_blocks, block_size

def create_atlas_bitnet_from_packed(model_dir, output_path):
    """Convert the packed microsoft/bitnet-b1.58-2B-4T model to TQ1.0 format.
    
    This reads the already-quantized U8 packed format (2-bit ternary per byte)
    and converts losslessly to TQ1.0 (5-trit Base-3 per byte) with block scales.
    """
    sf_path = os.path.join(model_dir, 'model.safetensors')
    if not os.path.exists(sf_path):
        print(f"Error: {sf_path} not found")
        sys.exit(1)

    with open(sf_path, 'rb') as sf:
        hl = struct.unpack('<Q', sf.read(8))[0]
        hdr = json.loads(sf.read(hl))
        all_names = sorted([k for k in hdr if k != '__metadata__'])

    with open(os.path.join(model_dir, 'config.json')) as f:
        cfg = json.load(f)

    hidden = cfg['hidden_size']
    n_layers = cfg['num_hidden_layers']
    n_heads = cfg['num_attention_heads']
    n_kv_heads = cfg.get('num_key_value_heads', n_heads)
    inter = cfg['intermediate_size']
    vocab = cfg['vocab_size']
    head_dim = cfg.get('head_dim', hidden // n_heads)
    rope_theta = cfg.get('rope_theta', 10000.0)
    tie_emb = cfg.get('tie_word_embeddings', False)

    print(f"[BitNet-Packed] {n_layers}L:{hidden}:{inter} Heads:{n_heads}/{n_kv_heads}")
    print(f"  Vocab:{vocab} Head_dim:{head_dim} RoPE theta:{rope_theta}")
    print(f"  Tie embeddings:{tie_emb}")

    # Preload scales (tiny, all BF16 shape=[1])
    scales = {}
    for name in all_names:
        if name.endswith('weight_scale'):
            info = hdr[name]
            s, e = info['data_offsets']
            with open(sf_path, 'rb') as f:
                f.seek(8 + hl + s)
                sdata = f.read(e - s)
            scale_arr = np.frombuffer(sdata, dtype=np.uint16).astype(np.uint32) << 16
            scales[name] = scale_arr.view(np.float32)[0]
    print(f"  Scales loaded: {len(scales)}")

    # Ordered tensor list: stride=11 (BitNet SubLN variant)
    tensor_names = []
    for L in range(n_layers):
        for tname in [
            f"model.layers.{L}.input_layernorm.weight",
            f"model.layers.{L}.self_attn.q_proj.weight",
            f"model.layers.{L}.self_attn.k_proj.weight",
            f"model.layers.{L}.self_attn.v_proj.weight",
            f"model.layers.{L}.self_attn.o_proj.weight",
            f"model.layers.{L}.post_attention_layernorm.weight",
            f"model.layers.{L}.mlp.gate_proj.weight",
            f"model.layers.{L}.mlp.up_proj.weight",
            f"model.layers.{L}.mlp.down_proj.weight",
            f"model.layers.{L}.self_attn.attn_sub_norm.weight",
            f"model.layers.{L}.mlp.ffn_sub_norm.weight",
        ]:
            if tname in all_names:
                tensor_names.append(tname)
            else:
                print(f"  WARNING: {tname} not found")
    for tname in ["model.embed_tokens.weight", "model.norm.weight"]:
        if tname in all_names:
            tensor_names.append(tname)
    if not tie_emb and "lm_head.weight" in all_names:
        tensor_names.append("lm_head.weight")

    print(f"  Tensors: {len(tensor_names)}")

    # Load tokenizer
    tokenizer_block = b''
    for tokfn in ['tokenizer.json', 'tokenizer.model']:
        tp = os.path.join(model_dir, tokfn)
        if os.path.exists(tp):
            with open(tp, 'rb') as tf:
                tok_data = tf.read()
            tokenizer_block += struct.pack('<I', len(tok_data)) + tok_data
            break
    cfg_path = os.path.join(model_dir, 'tokenizer_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'rb') as cf:
            tokenizer_block += struct.pack('<I', len(cf.read())) + cf.read()

    with open(output_path, 'wb') as out:
        header = bytearray(64)
        header[0:5] = b'ATLAS'
        struct.pack_into('<H', header, 5, 5)
        struct.pack_into('<H', header, 7, n_layers)
        struct.pack_into('<H', header, 9, hidden)
        struct.pack_into('<H', header, 11, inter)
        struct.pack_into('<B', header, 13, n_heads)
        struct.pack_into('<B', header, 14, n_kv_heads)
        struct.pack_into('<H', header, 15, head_dim)
        struct.pack_into('<I', header, 17, vocab)
        struct.pack_into('<d', header, 21, rope_theta)
        struct.pack_into('<I', header, 56, 0)
        struct.pack_into('<I', header, 60, len(tensor_names))
        eos_id = cfg.get('eos_token_id')
        pad_id = cfg.get('pad_token_id')
        if isinstance(eos_id, list): eos_id = eos_id[0] if eos_id else 0
        if isinstance(pad_id, list): pad_id = pad_id[0] if pad_id else 0
        if eos_id is not None: struct.pack_into('<I', header, 45, eos_id)
        if pad_id is not None: struct.pack_into('<I', header, 49, pad_id)
        # Byte 53: model_flags — BitNet b1.58 uses ReLU² (hidden_act: "relu2" per Microsoft config)
        struct.pack_into('<B', header, 53, 1 << 3)

        name_bytes = b''.join(n.encode() + b'\0' for n in tensor_names)
        name_block = struct.pack('<I', 4 + len(name_bytes)) + name_bytes
        struct.pack_into('<I', header, 56, len(name_block))

        out.write(header)
        data_start = 64 + len(tensor_names) * 12 + len(name_block)
        directory = bytearray(len(tensor_names) * 12)
        current_offset = data_start

        out.seek(64 + len(tensor_names) * 12)
        out.write(name_block)
        out.seek(data_start)

        with open(sf_path, 'rb') as sf:
            for idx, name in enumerate(tensor_names):
                info = hdr[name]
                s, e = info['data_offsets']
                sf.seek(8 + hl + s)
                data = sf.read(e - s)
                dtype = info['dtype']
                shape = info['shape']

                is_weight = 'proj' in name and 'weight' in name

                if dtype == 'U8' and is_weight:
                    packed = np.frombuffer(data, dtype=np.uint8).reshape(shape)
                    sname = name.replace('.weight', '.weight_scale')
                    gamma = float(scales.get(sname, 1.0))
                    out_dim = shape[0] * 4
                    # Unpack 2-bit → int8 ternary → shuffle rows → pack TQ1.0
                    ternary = unpack_2bit_ternary_to_int8(packed, gamma, out_dim)
                    ternary = pre_shuffle_rows(ternary)
                    data_bytes, packed_per_row, n_blocks, block_size = pack_int8_ternary_to_tq1(ternary, gamma)
                    ttype = 5
                    row_dim = out_dim
                elif dtype == 'BF16' and not is_weight:
                    arr = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16
                    fp32 = arr.view(np.float32).reshape(shape)
                    # Clean embed_tokens garbage rows
                    if 'embed_tokens' in name and fp32.ndim == 2:
                        Hc = fp32.shape[1]
                        for r in range(fp32.shape[0]):
                            overflow_mask = np.abs(fp32[r]) > 1e10
                            n_ov = np.sum(overflow_mask)
                            if n_ov > 0:
                                fp32[r][overflow_mask] = 0.0
                                print(f"    Clean row {r}: {n_ov}/{Hc} overflow")
                    fp32 = np.clip(fp32, -65504.0, 65504.0)
                    data_bytes = fp32.astype(np.float16).tobytes()
                    packed_per_row = 0
                    n_blocks = 0
                    block_size = 0
                    ttype = 1 if ('norm' in name or 'embed' in name) else 2
                    row_dim = shape[0]
                else:
                    print(f"  SKIP {name}: dtype={dtype} shape={shape}")
                    continue

                if current_offset % 32 != 0:
                    pad = 32 - (current_offset % 32)
                    current_offset += pad
                    out.write(b'\x00' * pad)

                directory[idx*12] = ttype
                struct.pack_into('<I', directory, idx*12 + 1, current_offset)
                struct.pack_into('<I', directory, idx*12 + 5, row_dim)
                ppr = packed_per_row & 0xFFFFFF
                directory[idx*12 + 9] = ppr & 0xFF
                directory[idx*12 + 10] = (ppr >> 8) & 0xFF
                directory[idx*12 + 11] = (ppr >> 16) & 0xFF

                out.write(data_bytes)
                current_offset += len(data_bytes)
                print(f"  [{idx}/{len(tensor_names)}] {name[:55]:55s} {len(data_bytes)/1024:7.1f}KB ttype={ttype}")

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

    total_gb = current_offset / 1024**3
    print(f"[ATLAS] Done! {total_gb:.2f} GB | {len(tensor_names)} tensors")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python atlas_packer_bitnet.py <model_dir> [output.atlas]")
        print("    BF16 model → ternarize + pack (old path)")
        print("  python atlas_packer_bitnet.py --packed <model_dir> [output.atlas]")
        print("    Packed U8 model → lossless TQ1.0 conversion")
        sys.exit(1)

    is_packed = sys.argv[1] == '--packed'
    if is_packed:
        model_dir = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else 'bitnet-2B4T-tq1-g128.atlas'
        create_atlas_bitnet_from_packed(model_dir, output_path)
    else:
        model_dir = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'bitnet-2B4T-tq1-g128.atlas'
        create_atlas_bitnet(model_dir, output_path)
