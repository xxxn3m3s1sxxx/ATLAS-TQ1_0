#!/usr/bin/env python3
"""Atlas Packer for Qwen3/Bonsai/BitNet — FP16 → TQ1.0"""
import struct, numpy as np, json, os, sys
from safetensors import safe_open
import torch

TQ1_MUL = np.array([1, 3, 9, 27, 81], dtype=np.uint32)


def build_tokenizer_binary(model_dir):
    """Build v6 binary tokenizer block from tokenizer.json.

    Returns bytes of the binary block, or b'' if tokenizer.json not found.
    Format: [128-byte header][offsets][lengths][pool][merge_left][merge_right][merge_rank][byte_encoder][byte_decoder][special_tokens]
    """
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

    special_ids = {'eos': 0xFFFFFFFF, 'bos': 0xFFFFFFFF, 'pad': 0xFFFFFFFF,
                   'unk': 0xFFFFFFFF, 'mask': 0xFFFFFFFF, 'sep': 0xFFFFFFFF, 'cls': 0xFFFFFFFF}
    cfg_path = os.path.join(model_dir, 'tokenizer_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as cf:
            cfg = json.load(cf)
        for key in ['eos_token', 'bos_token', 'pad_token', 'unk_token', 'mask_token', 'sep_token', 'cls_token']:
            val = cfg.get(key)
            if val and isinstance(val, dict) and 'id' in val:
                special_ids[key.split('_')[0]] = val['id']
    for token_str, tid in vocab.items():
        if tid == 0:
            special_ids['eos'] = tid if special_ids['eos'] == 0xFFFFFFFF else special_ids['eos']
        for pattern, idx_key in [('<|endoftext|>', 'eos'), ('<|im_end|>', 'eos'),
                                  ('<|pad|>', 'pad'), ('<unk>', 'unk')]:
            if token_str == pattern and special_ids[idx_key] == 0xFFFFFFFF:
                special_ids[idx_key] = tid
    if special_ids['pad'] == 0xFFFFFFFF: special_ids['pad'] = 0
    if special_ids['unk'] == 0xFFFFFFFF: special_ids['unk'] = 0
    if special_ids['eos'] == 0xFFFFFFFF: special_ids['eos'] = 0
    special_arr = np.array([
        special_ids['eos'], special_ids['bos'], special_ids['pad'],
        special_ids['unk'], special_ids['mask'], special_ids['sep'], special_ids['cls']
    ], dtype=np.uint32)

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


def unpack_ternary_weight(packed, scale):
    """Unpack 2-bit ternary codes from uint8 tensor.

    HF BitNet stores weights as packed 2-bit ternary: 4 values/byte.
    Mapping: 0b00=-1, 0b01=0, 0b10=+1
    Shape (packed_rows, cols) -> (packed_rows*4, cols)

    Returns float32 matrix (packed_rows*4 x cols) = ternary * scale.
    """
    pr, c = packed.shape  # packed_rows, cols
    out_rows = pr * 4
    out = np.zeros((out_rows, c), dtype=np.float32)

    # Bit decode: shift right by 0,2,4,6 then mask 0x03
    # Mapping: 0->-1, 1->0, 2->+1
    for j in range(4):
        bits = (packed >> (j * 2)) & 0x03
        val = np.where(bits == 1, 0.0, np.where(bits == 2, 1.0, -1.0))
        out[j::4, :] = val * scale

    return out

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


def ternarize_block_scaled_2bit(weights_fp16, block_size=128):
    """Per-block ternary quantization → 2-bit packed format (ttype=7).

    Same block-scaling as ternarize_block_scaled, but packs 4 ternary
    weights/byte (2-bit) instead of 5 trits/byte (TQ1 Base-3).
    2-bit encoding: 00=-1, 01=0, 10=+1 (simpler SIMD decode).
    Returns (block_scales_bytes, packed_bytes, packed_per_row, n_blocks, block_size).
    """
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

    # Pack ternary to 2-bit: 4 weights per byte, vectorized
    # Map {-1,0,1} → {0,1,2}, pack 4 per byte
    packed_per_row = (ncols + 3) // 4
    full_len = packed_per_row * 4
    row_padded = np.ones((nrows, full_len), dtype=np.uint8)  # pad with 1 (ternary 0)
    row_padded[:, :ncols] = ternary_flat.astype(np.int32) + 1  # -1→0, 0→1, 1→2

    # Reshape to (nrows, packed_per_row, 4) and pack with shifts
    reshaped = row_padded.reshape(nrows, packed_per_row, 4).astype(np.uint8)
    out = (reshaped[:, :, 0] << 0) | (reshaped[:, :, 1] << 2) | \
          (reshaped[:, :, 2] << 4) | (reshaped[:, :, 3] << 6)

    # Same wire format as ttype=5: [block_size:1][n_blocks:2][scales...][packed_data]
    block_scales = block_scales32.astype(np.float16)
    header = struct.pack('<BH', block_size, n_blocks) + block_scales.tobytes()
    return header + out.tobytes(), packed_per_row, n_blocks, block_size

def get_shard_path(weight_map, tensor_name, model_dir):
    """Resolve which shard file contains a tensor."""
    shard = weight_map.get(tensor_name)
    if shard:
        return os.path.join(model_dir, shard)
    return None

def create_atlas_qwen(model_dir, output_path, ttype=5):
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

    # BitNet SubLN tensors (only in BitNet models, skip for Qwen3/Bonsai)
    for L in range(n_layers):
        for bname in [
            f"model.layers.{L}.self_attn.attn_sub_norm.weight",
            f"model.layers.{L}.mlp.ffn_sub_norm.weight",
        ]:
            if bname in weight_map:
                tensor_names.append(bname)

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
            shard_handles[sp] = safe_open(sp, framework='pt')
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

        # model_flags byte 53: bit0=is_qwen3, bit1=tie_emb, bit2=thinking, bit3=gate_act
        mt = cfg.get('model_type', '')
        is_qwen = 1 if mt in ('qwen2', 'qwen3') else 0
        tie_emb = 1 if cfg.get('tie_word_embeddings', False) else 0
        thinking = 1 if cfg.get('enable_thinking', False) else 0
        gate_act = 1 if cfg.get('hidden_act', 'silu') == 'relu2' else 0
        model_flags = (is_qwen << 0) | (tie_emb << 1) | (thinking << 2) | (gate_act << 3)
        struct.pack_into('<B', header, 53, model_flags)

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
                # Handle HF BitNet packed uint8 weights
                if tensor.dtype == torch.uint8:
                    base = name.rsplit('.', 1)[0]
                    scale_tname = f"{base}.weight_scale"
                    scale_val = get_tensor(scale_tname).item()
                    tensor = unpack_ternary_weight(tensor.numpy(), scale_val)
                else:
                    tensor = tensor.cpu().to(torch.float32).numpy()

                tensor = pre_shuffle_rows(tensor) if ttype != 7 else tensor
                if ttype == 7:
                    data_bytes, packed_per_row, n_blocks, block_size = ternarize_block_scaled_2bit(tensor)
                else:
                    data_bytes, packed_per_row, n_blocks, block_size = ternarize_block_scaled(tensor)
            else:
                packed_per_row = 0
                n_blocks = 0
                block_size = 0
                data_bytes = tensor.cpu().to(torch.float16).numpy().tobytes()

            if current_offset % 32 != 0:
                pad = 32 - (current_offset % 32)
                current_offset += pad
                out.write(b'\x00' * pad)

            if is_ternary:
                # packed format: 5=TQ1 Base-3, 7=TurboQuant 2-bit
                tens_ttype = ttype if ttype == 7 else 5
                row_dim = tensor.shape[0]
            elif 'norm' in name or 'embed' in name:
                tens_ttype = 1  # FP16 vector
                row_dim = tensor.shape[0]
            else:
                tens_ttype = 2  # FP16 matrix
                row_dim = tensor.shape[0]

            directory[idx*12] = tens_ttype
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

        # Tokenizer block (v5+)
        if tokenizer_block:
            tokenizer_offset = current_offset
            if tokenizer_offset % 32 != 0:
                pad = 32 - (tokenizer_offset % 32)
                tokenizer_offset += pad
                out.write(b'\x00' * pad)
            out.write(tokenizer_block)
            struct.pack_into('<I', header, 29, len(tokenizer_block))
            struct.pack_into('<I', header, 33, tokenizer_offset)

        # v6: Append binary tokenizer block
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
    import argparse
    parser = argparse.ArgumentParser(description='Pack a model to ATLAS format')
    parser.add_argument('model_dir', help='HF model directory')
    parser.add_argument('output', help='Output .atlas file')
    parser.add_argument('--ttype', type=int, default=5, choices=[5, 7],
                        help='Ternary tensor format: 5=TQ1 Base-3, 7=TurboQuant 2-bit (default: 5)')
    args = parser.parse_args()
    create_atlas_qwen(args.model_dir, args.output, ttype=args.ttype)
