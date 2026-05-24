#!/usr/bin/env python3
"""Atlas Packer for Phi-3 — FP16 → TQ1.0 g128 block-scaled ternary.
Phi-3 has fused qkv_proj and gate_up_proj — split them before packing.
LLaMA-like, MHA, head_dim=96, stride=7 (no QK-Norm). 
Handles bfloat16 by converting to float16 on the fly.
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

def ternarize_block_scaled(weights_fp16, block_size=128):
    w = weights_fp16.astype(np.float32)
    nrows, ncols = w.shape
    n_blocks = (ncols + block_size - 1) // block_size
    pad_len = n_blocks * block_size - ncols
    if pad_len:
        w_pad = np.pad(w, ((0, 0), (0, pad_len)), constant_values=0)
    else:
        w_pad = w
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
        out[r * packed_per_row : (r + 1) * packed_per_row] = (
            (t5 * TQ1_MUL).sum(axis=1).astype(np.uint8)
        )
    block_scales = block_scales32.astype(np.float16)
    header = struct.pack('<BH', block_size, n_blocks) + block_scales.tobytes()
    return header + out.tobytes(), packed_per_row, n_blocks, block_size

def read_safetensors_header(path):
    """Read header from a safetensors file, return (header_dict, file_handle)."""
    f = open(path, 'rb')
    hdr_len = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hdr_len))
    return hdr, f, hdr_len

def load_tensor_from_shard(shard_path, tname):
    """Load a single tensor from a safetensors shard. Handles bfloat16."""
    hdr, f, hdr_len = read_safetensors_header(shard_path)
    if tname not in hdr:
        f.close()
        return None
    info = hdr[tname]
    dtype = info['dtype']
    shape = info['shape']
    start, end = info['data_offsets']
    f.seek(8 + hdr_len + start)
    data = f.read(end - start)
    f.close()

    if dtype == 'BF16':
        arr = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16
        arr = arr.view(np.float32).reshape(shape).astype(np.float16)
        return arr
    elif dtype == 'F16':
        return np.frombuffer(data, dtype=np.float16).reshape(shape)
    elif dtype == 'F32':
        return np.frombuffer(data, dtype=np.float32).reshape(shape)
    else:
        raise TypeError(f"Unsupported dtype: {dtype} for {tname}")

def create_atlas_phi3(model_dir, output_path):
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

    print(f"[Phi3] Layers:{n_layers} Hidden:{hidden} Heads:{n_heads}/{n_kv_heads}")
    print(f"  Intermediate:{inter} Vocab:{vocab} Head_dim:{head_dim} RoPE theta:{rope_theta}")

    # Build weight_map: tensor_name -> shard_path
    idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        weight_map = {}
        for k, v in idx['weight_map'].items():
            weight_map[k] = os.path.join(model_dir, v)
    else:
        sf = os.path.join(model_dir, 'model.safetensors')
        if os.path.exists(sf):
            hdr, f, _ = read_safetensors_header(sf)
            f.close()
            weight_map = {k: sf for k in hdr.keys()}
        else:
            weight_map = {}

    has_fused_qkv = any('qkv_proj' in k for k in weight_map)
    has_fused_gate_up = any('gate_up_proj' in k for k in weight_map)

    # Build expanded tensor list (splitting fused tensors)
    per_layer_templates = [
        "{prefix}.input_layernorm.weight",
        "{prefix}.self_attn.q_proj.weight",
        "{prefix}.self_attn.k_proj.weight",
        "{prefix}.self_attn.v_proj.weight",
        "{prefix}.self_attn.o_proj.weight",
        "{prefix}.post_attention_layernorm.weight",
        "{prefix}.mlp.gate_proj.weight",
        "{prefix}.mlp.up_proj.weight",
        "{prefix}.mlp.down_proj.weight",
    ]

    tensor_sources = {}  # output_name -> {source_name, split_op or None}

    for L in range(n_layers):
        prefix = f"model.layers.{L}"
        for tmpl in per_layer_templates:
            out_name = tmpl.format(prefix=prefix)
            # Check for fused variants
            if 'q_proj' in out_name and out_name not in weight_map and has_fused_qkv:
                src_name = f"{prefix}.self_attn.qkv_proj.weight"
                if src_name not in tensor_sources:  # first time → add q/k/v
                    tensor_sources[f"{prefix}.self_attn.q_proj.weight"] = {'src': src_name, 'split': ('slice', 0)}
                    tensor_sources[f"{prefix}.self_attn.k_proj.weight"] = {'src': src_name, 'split': ('slice', 1)}
                    tensor_sources[f"{prefix}.self_attn.v_proj.weight"] = {'src': src_name, 'split': ('slice', 2)}
                continue
            elif 'gate_proj' in out_name and out_name not in weight_map and has_fused_gate_up:
                src_name = f"{prefix}.mlp.gate_up_proj.weight"
                if src_name not in tensor_sources:
                    tensor_sources[f"{prefix}.mlp.gate_proj.weight"] = {'src': src_name, 'split': ('split_half', 0)}
                    tensor_sources[f"{prefix}.mlp.up_proj.weight"] = {'src': src_name, 'split': ('split_half', 1)}
                continue
            elif out_name in weight_map:
                tensor_sources[out_name] = {'src': out_name, 'split': None}

    for tname in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]:
        if tname in weight_map:
            tensor_sources[tname] = {'src': tname, 'split': None}

    tensor_names = list(tensor_sources.keys())
    print(f"  Tensors: {len(tensor_names)}")

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

    # Tensor cache to avoid reloading fused sources
    tensor_cache = {}
    def load_tensor(tname):
        if tname in tensor_cache:
            return tensor_cache[tname]
        info = tensor_sources.get(tname)
        if info is None:
            raise KeyError(f"Tensor {tname} not found in weight_map")
        src = info['src']
        raw = load_tensor_from_shard(weight_map[src], src)
        if info['split'] is None:
            t = raw
        elif info['split'][0] == 'slice':
            _, part = info['split']
            nH = n_heads
            hd = head_dim
            q_dim = nH * hd
            t = raw[q_dim*part : q_dim*(part+1)]
        elif info['split'][0] == 'split_half':
            _, half = info['split']
            half_len = raw.shape[0] // 2
            t = raw[half_len*half : half_len*(half+1)]
        tensor_cache[tname] = t
        return t

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
            tensor = load_tensor(name)
            is_ternary = 'proj' in name and 'weight' in name

            if is_ternary:
                tensor = pre_shuffle_rows(tensor)
                data_bytes, packed_per_row, n_blocks, block_size = ternarize_block_scaled(tensor)
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
            elif 'norm' in name or 'embed' in name:
                ttype = 1
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

def auto_output_name(model_dir):
    cfg_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    hidden = cfg.get('hidden_size', 0)
    sizes = {3072: "3.8B", 2560: "2.8B", 2048: "1.5B"}
    label = sizes.get(hidden, f"{hidden}d")
    return f"phi3-{label}-tq1-g128.atlas"

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python atlas_packer_phi3.py <model_dir> [output.atlas]")
        sys.exit(1)
    model_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else auto_output_name(model_dir)
    if not output_path:
        print("Error: could not auto-detect output name")
        sys.exit(1)
    create_atlas_phi3(model_dir, output_path)
