#!/usr/bin/env python3
"""Atlas Packer for TriLM — FP16 → TQ1.0 g128 block-scaled ternary.
TriLM models from NolanoOrg/SpectraSuite: BitNet-style natively ternary LLMs
with LLaMA architecture (MHA, SwiGLU, RoPE, RMSNorm, no bias, no QK-Norm).
"""
import struct, numpy as np, json, os, sys
from safetensors import safe_open

TQ1_MUL = np.array([1, 3, 9, 27, 81], dtype=np.uint32)

TRILM_SIZES = {
    512: "99M", 768: "190M", 1024: "390M", 1280: "560M",
    1536: "830M", 1792: "1.1B", 2048: "1.5B", 2304: "2.4B", 3072: "3.9B",
}


def build_tokenizer_binary(model_dir):
    """Build v6 binary tokenizer block from tokenizer.json."""
    tok_path = os.path.join(model_dir, 'tokenizer.json')
    if not os.path.exists(tok_path):
        print("[ATLAS] ERROR: tokenizer.json not found — cannot build v6 binary tokenizer")
        return b''

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    vocab = tok.get_vocab()
    V = len(vocab)
    sorted_items = sorted(vocab.items(), key=lambda kv: kv[1])

    offsets = np.empty(V, dtype=np.uint32)
    lengths = np.empty(V, dtype=np.uint16)
    pool_parts = []
    offset_acc = 0
    for i, (token_str, tid) in enumerate(sorted_items):
        token_bytes = token_str.encode('utf-8')
        offsets[i] = offset_acc
        lengths[i] = len(token_bytes)
        pool_parts.append(token_bytes)
        offset_acc += len(token_bytes)
    pool = b''.join(pool_parts)
    max_token_length = int(max(lengths))

    merge_left = np.full(V, 0xFFFFFFFF, dtype=np.uint32)
    merge_right = np.full(V, 0xFFFFFFFF, dtype=np.uint32)
    merge_rank = np.zeros(V, dtype=np.uint32)

    with open(tok_path, 'r', encoding='utf-8') as jf:
        tok_json_data = json.load(jf)
    merges_list = tok_json_data.get('model', {}).get('merges', [])
    for i, merge_pair in enumerate(merges_list):
        if isinstance(merge_pair, list) and len(merge_pair) == 2:
            left_str, right_str = merge_pair
        elif isinstance(merge_pair, str):
            parts = merge_pair.split()
            if len(parts) != 2: continue
            left_str, right_str = parts
        else:
            continue
        left_id = vocab.get(left_str)
        right_id = vocab.get(right_str)
        if left_id is not None and right_id is not None:
            merged_str = left_str + right_str
            merged_id = vocab.get(merged_str)
            if merged_id is not None and merged_id < V:
                merge_left[merged_id] = left_id
                merge_right[merged_id] = right_id
                merge_rank[merged_id] = i + 1

    byte_encoder = np.full(256, 0xFFFF, dtype=np.uint16)
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    byte_to_token = {}
    n = 0
    for b in range(256):
        if b in printable:
            byte_to_token[b] = chr(b)
        else:
            byte_to_token[b] = chr(256 + n)
            n += 1
    for b in range(256):
        tid = vocab.get(byte_to_token[b])
        if tid is not None:
            byte_encoder[b] = tid
    missing = [b for b in range(256) if byte_encoder[b] == 0xFFFF]
    if missing:
        for b in missing:
            tid = vocab.get(chr(256 + b))
            if tid is not None:
                byte_encoder[b] = tid
    missing = [b for b in range(256) if byte_encoder[b] == 0xFFFF]
    if missing:
        print(f'  WARNING: {len(missing)} bytes unmapped in byte_encoder (first 8: {missing[:8]})')

    byte_decoder = np.full(256, 0xFFFF, dtype=np.uint16)
    for b in range(256):
        if byte_encoder[b] != 0xFFFF:
            byte_decoder[b] = b

    eos_id = 0
    cfg_path = os.path.join(model_dir, 'tokenizer_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as cf:
            tcfg = json.load(cf)
        eos_cfg = tcfg.get('eos_token')
        if eos_cfg and isinstance(eos_cfg, dict) and 'id' in eos_cfg:
            eos_id = eos_cfg['id']
    special_arr = np.array([eos_id, 0xFFFFFFFF, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF], dtype=np.uint32)

    off = 128
    off_offsets = off; off += V * 4
    off_lengths = off; off += V * 2
    off_pool = off; off += len(pool)
    pool_pad = (4 - len(pool) % 4) % 4; off += pool_pad
    off_merge_left = off; off += V * 4
    off_merge_right = off; off += V * 4
    off_merge_rank = off; off += V * 4
    off_byte_enc = off; off += 512
    off_byte_dec = off; off += 512
    off_special = off; off += 28
    total_size = off

    buf = bytearray(total_size)
    struct.pack_into('<I', buf, 0, 0x544F4B42)
    struct.pack_into('<I', buf, 4, 1)
    struct.pack_into('<I', buf, 8, V)
    struct.pack_into('<I', buf, 12, max_token_length)
    struct.pack_into('<I', buf, 20, 0)
    offs_list = [off_offsets, off_lengths, off_pool, len(pool),
                 off_merge_left, off_merge_right, off_merge_rank,
                 off_byte_enc, off_byte_dec, off_special]
    for i, val in enumerate(offs_list):
        struct.pack_into('<Q', buf, 24 + i * 8, val)
    buf[off_offsets:off_offsets + V * 4] = offsets.tobytes()
    buf[off_lengths:off_lengths + V * 2] = lengths.tobytes()
    buf[off_pool:off_pool + len(pool)] = pool
    buf[off_merge_left:off_merge_left + V * 4] = merge_left.tobytes()
    buf[off_merge_right:off_merge_right + V * 4] = merge_right.tobytes()
    buf[off_merge_rank:off_merge_rank + V * 4] = merge_rank.tobytes()
    buf[off_byte_enc:off_byte_enc + 512] = byte_encoder.tobytes()
    buf[off_byte_dec:off_byte_dec + 512] = byte_decoder.tobytes()
    buf[off_special:off_special + 28] = special_arr.tobytes()

    print(f'  Binary tokenizer: {total_size/1024/1024:.2f} MB (pool={len(pool)/1024:.1f} KB, V={V})')
    return bytes(buf)

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
        out[r * packed_per_row : (r + 1) * packed_per_row] = (
            (t5 * TQ1_MUL).sum(axis=1).astype(np.uint8)
        )
    block_scales = block_scales32.astype(np.float16)
    header = struct.pack('<BH', block_size, n_blocks) + block_scales.tobytes()
    return header + out.tobytes(), packed_per_row, n_blocks, block_size

def get_shard_path(weight_map, tensor_name, model_dir):
    shard = weight_map.get(tensor_name)
    return os.path.join(model_dir, shard) if shard else None

def create_atlas_trilm(model_dir, output_path):
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

    size_label = TRILM_SIZES.get(hidden, f"{hidden}d")
    print(f"[TriLM] {size_label}  Layers:{n_layers} Hidden:{hidden} Heads:{n_heads}/{n_kv_heads}")
    print(f"  Intermediate:{inter} Vocab:{vocab} Head_dim:{head_dim} RoPE theta:{rope_theta}")
    print(f"  Tie embeddings:{tie_emb}")

    idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    weight_map = {}
    has_bf16 = False
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        weight_map = idx['weight_map']
        # Check dtype from first tensor
        shard0 = os.path.join(model_dir, list(weight_map.values())[0])
        if os.path.exists(shard0):
            with open(shard0, 'rb') as sf:
                hl = struct.unpack('<Q', sf.read(8))[0]
                h0 = json.loads(sf.read(hl))
                # Skip __metadata__ key
                first_k = next(k for k in h0 if k != '__metadata__')
                has_bf16 = h0[first_k]['dtype'] == 'BF16'
    else:
        sf_path = os.path.join(model_dir, 'model.safetensors')
        if os.path.exists(sf_path):
            with open(sf_path, 'rb') as sf:
                hl = struct.unpack('<Q', sf.read(8))[0]
                h0 = json.loads(sf.read(hl))
                for k in h0:
                    if k != '__metadata__':
                        weight_map[k] = 'model.safetensors'
                first_k = next(k for k in h0 if k != '__metadata__')
                has_bf16 = h0[first_k]['dtype'] == 'BF16'

    # BF16 reader using raw safetensors format
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
            f.seek(start)
            data = f.read(end - start)
        if info['dtype'] == 'BF16':
            arr = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16
            return arr.view(np.float32).reshape(info['shape']).astype(np.float16)
        elif info['dtype'] == 'F16':
            return np.frombuffer(data, dtype=np.float16).reshape(info['shape'])
        elif info['dtype'] == 'F32':
            return np.frombuffer(data, dtype=np.float32).reshape(info['shape'])
        raise TypeError(f'Unsupported dtype: {info["dtype"]} for {tname}')

    # Ordered tensor list: stride=9 (no QK-Norm), LLaMA naming
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
        ]:
            if tname in weight_map:
                tensor_names.append(tname)

    for tname in ["model.embed_tokens.weight", "model.norm.weight"]:
        if tname in weight_map:
            tensor_names.append(tname)
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
        cfg_data = b''
        if os.path.exists(cfg_path):
            with open(cfg_path, 'rb') as cf:
                cfg_data = cf.read()
        tokenizer_block += struct.pack('<I', len(cfg_data)) + cfg_data

    shard_handles = {}
    def get_tensor(tname):
        if has_bf16:
            return read_tensor(tname)
        sp = weight_map.get(tname)
        spath = os.path.join(model_dir, sp) if os.sep not in sp else sp
        if spath not in shard_handles:
            shard_handles[spath] = safe_open(spath, framework='np')
        return shard_handles[spath].get_tensor(tname)

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
            tensor = get_tensor(name)
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

        # v6: Append binary tokenizer block (REQUIRED)
        binary_tok_block = build_tokenizer_binary(model_dir)
        if len(binary_tok_block) == 0:
            print("[ATLAS] ERROR: Could not build v6 binary tokenizer block")
            sys.exit(1)
        struct.pack_into('<H', header, 5, 7)
        if tokenizer_block:
            binary_offset = tokenizer_offset + len(tokenizer_block)
        else:
            binary_offset = current_offset
        if binary_offset % 32 != 0:
            pad = 32 - (binary_offset % 32)
            binary_offset += pad
            out.write(b'\x00' * pad)
        out.write(binary_tok_block)
        current_offset = binary_offset + len(binary_tok_block)
        struct.pack_into('<I', header, 37, len(binary_tok_block))
        struct.pack_into('<I', header, 41, binary_offset)

        # model_flags byte 53
        mt = cfg.get('model_type', '')
        is_qwen = 1 if mt == 'qwen2' else 0
        tie_emb = 1 if cfg.get('tie_word_embeddings', False) else 0
        thinking = 0
        gate_act = 1 if cfg.get('hidden_act', 'silu') == 'relu2' else 0
        model_flags = (is_qwen << 0) | (tie_emb << 1) | (thinking << 2) | (gate_act << 3)
        struct.pack_into('<B', header, 53, model_flags)

        out.flush()
        out.seek(64)
        out.write(directory)
        out.seek(0)
        out.write(header)

    for h in shard_handles.values():
        pass
    total_gb = current_offset / 1024**3
    print(f"[ATLAS] Done! {total_gb:.2f} GB | {len(tensor_names)} tensors")

def auto_output_name(model_dir):
    cfg_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    mt = cfg.get('model_type', '')
    if mt not in ('llama', 'qwen2'):
        return None
    hidden = cfg.get('hidden_size', 0)
    size_label = TRILM_SIZES.get(hidden, f"{hidden}d")
    prefix = 'trilm' if mt == 'llama' else 'qwen25'
    return f"{prefix}-{size_label}-tq1-g128.atlas"

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python atlas_packer_trilm.py <model_dir> [output.atlas]")
        sys.exit(1)
    model_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else auto_output_name(model_dir)
    if not output_path:
        print(f"Error: could not auto-detect output name from {model_dir}/config.json")
        sys.exit(1)
    create_atlas_trilm(model_dir, output_path)
