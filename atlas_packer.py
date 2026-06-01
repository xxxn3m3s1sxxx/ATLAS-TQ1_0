#!/usr/bin/env python3
"""Atlas Packer v2/v6 - Streaming TQ1.0 packer for Falcon3 BitNet"""
import struct, torch, numpy as np, json, os, sys
from safetensors import safe_open

BITNET_MUL = np.array([1, 3, 9, 27], dtype=np.uint32)
TQ1_MUL    = np.array([1, 3, 9, 27, 81], dtype=np.uint32)

def build_tokenizer_binary(model_dir):
    """Build v6 binary tokenizer block from tokenizer.json.

    Returns bytes of the binary block, or b'' if tokenizer.json not found.
    Format: [128-byte header][offsets][lengths][pool][merge_left][merge_right][merge_rank][byte_encoder][byte_decoder][special_tokens]
    """
    tok_path = os.path.join(model_dir, 'tokenizer.json')
    if not os.path.exists(tok_path):
        return b''

    from tokenizers import Tokenizer, decoders
    tok = Tokenizer.from_file(tok_path)

    # vocab: dict token_str → id, sorted by id
    vocab = tok.get_vocab()
    V = len(vocab)
    sorted_items = sorted(vocab.items(), key=lambda kv: kv[1])

    # Build decoder arrays: offsets[], lengths[], pool
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

    # Get merges from tokenizer.json directly (model object doesn't expose them)
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

    # Build byte_encoder[256]: byte value → base token ID (0xFFFF = unmapped)
    # GPT-2 ByteLevel uses bytes_to_unicode() mapping: bytes → Unicode chars (U+0100-U+01FF)
    byte_encoder = np.full(256, 0xFFFF, dtype=np.uint16)

    # Replicate bytes_to_unicode() from GPT-2/transformers
    printable = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    byte_to_token = {}
    n = 0
    for b in range(256):
        if b in printable:
            byte_to_token[b] = chr(b)
        else:
            byte_to_token[b] = chr(256 + n)
            n += 1

    # Map each byte through its Unicode representation to vocabulary
    for b in range(256):
        token_str = byte_to_token[b]
        tid = vocab.get(token_str)
        if tid is not None:
            byte_encoder[b] = tid
        elif b == 32:
            # Space is a special case — it's often Ġ (U+0120) in the byte vocabulary
            # Try the Ġ mapping: space → U+0120 → UTF-8 [0xC4, 0xA0]
            # Actually, in bytes_to_unicode, space (0x20) = chr(0x20) = ' '
            # This should map normally via the printable list
            pass

    # Verify and report gaps
    missing = [b for b in range(256) if byte_encoder[b] == 0xFFFF]
    if missing:
        # Last resort: try Ġ-based mapping (GPT-2 ByteLevel alternative)
        # Ġ = chr(288) = U+0120, used as space prefix in pre-tokenizer
        for b in missing:
            # Bytes 0-255 may be mapped to Unicode range 256-511
            unicode_char = chr(256 + b)
            tid = vocab.get(unicode_char)
            if tid is not None:
                byte_encoder[b] = tid

    missing = [b for b in range(256) if byte_encoder[b] == 0xFFFF]
    if missing:
        print(f'  WARNING: {len(missing)} bytes unmapped in byte_encoder (first 8: {missing[:8]})')

    # byte_decoder[256]: reverse mapping — token ID → byte value
    # For GPT-2 ByteLevel, the decoder processes UTF-8 bytes through the ByteLevel decoder
    # which converts Ġ back to space. So byte_decoder is identity for all mapped bytes.
    byte_decoder = np.full(256, 0xFFFF, dtype=np.uint16)
    for b in range(256):
        if byte_encoder[b] != 0xFFFF:
            byte_decoder[b] = b

    # Detect special tokens by scanning token strings
    special_map = {
        '<|endoftext|>': 0, '<|im_end|>': 0,
        '<|pad|>': 2, '<unk>': 3, '<|startoftext|>': 1,
    }
    # More robust: check tokenizer_config.json for special token IDs
    cfg_path = os.path.join(model_dir, 'tokenizer_config.json')
    special_ids = {'eos': 0xFFFFFFFF, 'bos': 0xFFFFFFFF, 'pad': 0xFFFFFFFF,
                   'unk': 0xFFFFFFFF, 'mask': 0xFFFFFFFF, 'sep': 0xFFFFFFFF, 'cls': 0xFFFFFFFF}
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        for key in ['eos_token', 'bos_token', 'pad_token', 'unk_token', 'mask_token', 'sep_token', 'cls_token']:
            val = cfg.get(key)
            if val and isinstance(val, dict) and 'id' in val:
                special_ids[key.split('_')[0]] = val['id']
    # Fallback: scan by token string
    for token_str, tid in vocab.items():
        if tid == 0:
            special_ids['eos'] = tid if special_ids['eos'] == 0xFFFFFFFF else special_ids['eos']
        for pattern, idx_key in [('<|endoftext|>', 'eos'), ('<|im_end|>', 'eos'),
                                  ('<|pad|>', 'pad'), ('<unk>', 'unk')]:
            if token_str == pattern and special_ids[idx_key] == 0xFFFFFFFF:
                special_ids[idx_key] = tid
    # Set defaults if not found
    if special_ids['eos'] == 0xFFFFFFFF: special_ids['eos'] = 0
    if special_ids['pad'] == 0xFFFFFFFF: special_ids['pad'] = 0
    if special_ids['unk'] == 0xFFFFFFFF: special_ids['unk'] = 0
    if special_ids['bos'] == 0xFFFFFFFF: special_ids['bos'] = 0xFFFFFFFF  # not mapped

    special_tokens_arr = np.array([
        special_ids['eos'], special_ids['bos'], special_ids['pad'],
        special_ids['unk'], special_ids['mask'], special_ids['sep'], special_ids['cls']
    ], dtype=np.uint32)

    # ─── Pack binary block ──────────────────────────────────────────────
    # Layout:
    #   128 bytes header
    #   offsets[V]: uint32 × V
    #   lengths[V]: uint16 × V
    #   pool: raw bytes
    #   (padding to 4 bytes)
    #   merge_left[V]: uint32 × V
    #   merge_right[V]: uint32 × V
    #   merge_rank[V]: uint32 × V
    #   byte_encoder[256]: uint16 × 256
    #   byte_decoder[256]: uint16 × 256
    #   special_tokens[7]: uint32 × 7

    # Compute offsets relative to block start
    off = 128
    off_offsets = off
    off += V * 4
    off_lengths = off
    off += V * 2
    off_pool = off
    off += len(pool)
    # Align to 4
    pool_pad = (4 - len(pool) % 4) % 4
    off += pool_pad
    off_merge_left = off
    off += V * 4
    off_merge_right = off
    off += V * 4
    off_merge_rank = off
    off += V * 4
    off_byte_enc = off
    off += 512  # 256 * 2
    off_byte_dec = off
    off += 512
    off_special = off
    total_size = off + 28  # 7 * 4

    # Build header (128 bytes)
    header = bytearray(128)
    struct.pack_into('<I', header, 0, 0x544F4B42)  # magic "TOKB"
    struct.pack_into('<I', header, 4, 1)             # version
    struct.pack_into('<I', header, 8, V)              # vocab_size
    struct.pack_into('<I', header, 12, int(max_token_length))
    struct.pack_into('<I', header, 16, 0)             # special_count (reserved)
    struct.pack_into('<I', header, 20, 0)             # flags
    struct.pack_into('<Q', header, 24, off_offsets)
    struct.pack_into('<Q', header, 32, off_lengths)
    struct.pack_into('<Q', header, 40, off_pool)
    struct.pack_into('<Q', header, 48, len(pool))
    struct.pack_into('<Q', header, 56, off_merge_left)
    struct.pack_into('<Q', header, 64, off_merge_right)
    struct.pack_into('<Q', header, 72, off_merge_rank)
    struct.pack_into('<Q', header, 80, off_byte_enc)
    struct.pack_into('<Q', header, 88, off_byte_dec)
    struct.pack_into('<Q', header, 96, off_special)
    # reserved[16] at 104-119 stays zero

    # Assemble binary block
    buf = bytearray(total_size)
    buf[:128] = header
    buf[off_offsets:off_offsets + V * 4] = offsets.tobytes()
    buf[off_lengths:off_lengths + V * 2] = lengths.tobytes()
    buf[off_pool:off_pool + len(pool)] = pool
    # pool padding already zero-initialized
    buf[off_merge_left:off_merge_left + V * 4] = merge_left.tobytes()
    buf[off_merge_right:off_merge_right + V * 4] = merge_right.tobytes()
    buf[off_merge_rank:off_merge_rank + V * 4] = merge_rank.tobytes()
    buf[off_byte_enc:off_byte_enc + 512] = byte_encoder.tobytes()
    buf[off_byte_dec:off_byte_dec + 512] = byte_decoder.tobytes()
    buf[off_special:off_special + 28] = special_tokens_arr.tobytes()

    print(f"  Binary tokenizer: {total_size/1024/1024:.2f} MB "
          f"(pool={len(pool)/1024:.1f} KB, V={V})")
    return bytes(buf)

def pack_tensor_row_wise(tensor):
    """BitNet uint8 [OUT/4, IN] → TQ1.0 [OUT, IN/4*5] row-major.
    
    Each uint8 byte packs 4 consecutive OUTPUT rows at one INPUT column.
    We de-interleave: uint8_row[ur] → W[4*ur+0,:], W[4*ur+1,:], W[4*ur+2,:], W[4*ur+3,:]
    Then repack each weight row separately: [W[r,0..4]]→1 byte, [W[r,5..9]]→1 byte, ...
    """
    arr = tensor.numpy()
    u8_rows, in_cols = arr.shape        # [OUT/4, IN]
    weight_cols = in_cols                # each col = 1 ternary per output row
    packed_per_row = (weight_cols + 4) // 5
    packed = np.empty(u8_rows * 4 * packed_per_row, dtype=np.uint8)

    for ur in range(u8_rows):
        row = arr[ur].astype(np.uint32)
        # Unpack 4 ternary values per byte (2-bit packing, NOT Base-3!)
        # byte = v0 + v1*4 + v2*16 + v3*64  where v_i ∈ {0,1,2,3}
        # v_i ∈ {0→-1, 1→0, 2→+1}; clamp 3→2 (should not appear for real data)
        t0 = np.minimum(row & 3, 2)
        t1 = np.minimum((row >> 2) & 3, 2)
        t2 = np.minimum((row >> 4) & 3, 2)
        t3 = np.minimum((row >> 6) & 3, 2)

        # De-interleave: each of 4 output rows gets all ternary values for that row
        # row_tern[0] = W[4*ur+0, :], row_tern[1] = W[4*ur+1, :], etc.
        for sub in range(4):
            wt = [t0, t1, t2, t3][sub]  # weight row sub, m ∈ {0,1,2}
            full_len = packed_per_row * 5
            if weight_cols < full_len:
                wt_pad = np.pad(wt, (0, full_len - weight_cols), constant_values=1)
            else:
                wt_pad = wt[:full_len]
            t5 = wt_pad.reshape(packed_per_row, 5)
            v = (t5 * TQ1_MUL).sum(axis=1)
            start = (ur * 4 + sub) * packed_per_row
            packed[start : start + packed_per_row] = np.minimum(v, 242).astype(np.uint8)

    return packed.tobytes(), packed_per_row


def pack_ffn_to_int4(tensor, scale_val):
    """FFN uint8 [OUT/4, IN] → persistent int4 (ttype=8).
    
    Like pack_tensor_row_wise but packs 2 int8 values per byte (4-bit nibbles)
    instead of 5 trits/byte TQ1. This matches the ttype=8 format expected
    by atlas_matmul_i4_f32 with u8+128 activation quantization.
    
    Format: [fp16_scale:2][nibbles:OUT*packed_cols][row_sums:OUT*4]
    packed_cols = (IN + 1) // 2
    Each nibble stores int8 ∈ {-1,0,1} directly (clipped to [−8,7]).
    """
    arr = tensor.numpy()
    u8_rows, in_cols = arr.shape
    out_rows = u8_rows * 4
    packed_cols = (in_cols + 1) // 2

    # Unpack 4 ternary rows per uint8 byte, map {0,1,2}→{-1,0,+1}
    packed = np.zeros(out_rows * packed_cols, dtype=np.uint8)
    row_sums = np.zeros(out_rows, dtype=np.int32)

    for ur in range(u8_rows):
        row = arr[ur].astype(np.int32)
        t0 = np.minimum(row & 3, 2)
        t1 = np.minimum((row >> 2) & 3, 2)
        t2 = np.minimum((row >> 4) & 3, 2)
        t3 = np.minimum((row >> 6) & 3, 2)

        for sub, src in enumerate([t0, t1, t2, t3]):
            r = ur * 4 + sub
            # Map: 0→-1, 1→0, 2→+1 (same as TQ1 decode)
            tern = np.where(src == 0, -1, np.where(src == 2, 1, 0)).astype(np.int8)
            row_sum = 0
            for c in range(0, in_cols, 2):
                v0 = int(tern[c]) & 0x0F
                v1 = int(tern[c + 1]) & 0x0F if c + 1 < in_cols else 0
                packed[r * packed_cols + c // 2] = np.uint8(v0 | (v1 << 4))
                row_sum += int(tern[c])
                if c + 1 < in_cols:
                    row_sum += int(tern[c + 1])
            row_sums[r] = row_sum

    scale_bytes = struct.pack('<e', float(scale_val))
    data = scale_bytes + packed.tobytes() + row_sums.tobytes()
    return data, packed_cols


def _is_ffn_tensor(name):
    """Check if a tensor is an FFN weight (gate/down/up projection).
    Matches the same patterns as quantize_ffn_to_i4() in C++.
    """
    return any(kw in name for kw in ('gate', 'down', 'up'))

def create_atlas_from_config(safetensors_path, output_path):
    print(f"[ATLAS] Opening {safetensors_path}...")
    model_dir = os.path.dirname(safetensors_path)
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

    print(f"  Layers:{n_layers} Hidden:{hidden} Heads:{n_heads}/{n_kv_heads}")
    print(f"  Intermediate:{inter} Vocab:{vocab} Head_dim:{head_dim}")

    with safe_open(safetensors_path, framework='pt', device='cpu') as f:
        names = list(f.keys())

    print(f"  Tensors: {len(names)}")

    # Preload all weight_scale tensors (they're tiny: shape [1] bfloat16 each)
    scales = {}
    with safe_open(safetensors_path, framework='pt', device='cpu') as f:
        for n in names:
            if n.endswith('weight_scale'):
                scales[n] = f.get_tensor(n).item()

    print(f"  Scales loaded: {len(scales)}")

    # Load tokenizer.json + tokenizer_config.json for embedding (v5+)
    # Format: [tokenizer_json_size:4][tokenizer_json bytes][config_json_size:4][config_json bytes]
    tokenizer_block = b''
    tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'rb') as tf:
            tok_data = tf.read()
        tokenizer_block += struct.pack('<I', len(tok_data)) + tok_data
        cfg_path = os.path.join(model_dir, 'tokenizer_config.json')
        cfg_data = b''
        if os.path.exists(cfg_path):
            with open(cfg_path, 'rb') as cf:
                cfg_data = cf.read()
        tokenizer_block += struct.pack('<I', len(cfg_data)) + cfg_data
        print(f"  Tokenizer: {len(tokenizer_block)/1024/1024:.1f} MB (json={len(tok_data)/1024/1024:.1f} MB, config={len(cfg_data)/1024/1024:.1f} MB)")

    with open(output_path, 'wb') as out:
        n_tensors = len(names)
        header = bytearray(64)
        header[0:5] = b'ATLAS'
        struct.pack_into('<H', header, 5, 5)  # v5: embedded tokenizer
        struct.pack_into('<H', header, 7, n_layers)
        struct.pack_into('<H', header, 9, hidden)
        struct.pack_into('<H', header, 11, inter)
        struct.pack_into('<B', header, 13, n_heads)
        struct.pack_into('<B', header, 14, n_kv_heads)
        struct.pack_into('<H', header, 15, head_dim)
        struct.pack_into('<I', header, 17, vocab)
        struct.pack_into('<d', header, 21, rope_theta)
        struct.pack_into('<I', header, 60, n_tensors)

        # EOS/PAD in header bytes 45-52
        tcfg_path = os.path.join(model_dir, 'tokenizer_config.json')
        if os.path.exists(tcfg_path):
            with open(tcfg_path) as tcf:
                tcfg = json.load(tcf)
            eos_cfg = tcfg.get('eos_token')
            if isinstance(eos_cfg, dict) and 'id' in eos_cfg:
                struct.pack_into('<I', header, 45, eos_cfg['id'])
            pad_cfg = tcfg.get('pad_token')
            if isinstance(pad_cfg, dict) and 'id' in pad_cfg:
                struct.pack_into('<I', header, 49, pad_cfg['id'])

        # model_flags byte 53: bit0=is_qwen3, bit1=tie_emb, bit2=thinking, bit3=gate_act
        model_flags = 0  # Falcon3
        struct.pack_into('<B', header, 53, model_flags)

        # Bytes 54-55: format_version (v2.10.0+), 2 = TQ1.0 baseline (no persistent int4 on disk)
        struct.pack_into('<H', header, 54, 2)

        # Build name block: [name_block_size:4] [name_0\0] [name_1\0] ...
        name_bytes = b''.join(n.encode() + b'\0' for n in names)
        name_block = struct.pack('<I', 4 + len(name_bytes)) + name_bytes
        struct.pack_into('<I', header, 56, len(name_block))

        out.write(header)

        data_start = 64 + len(names) * 12 + len(name_block)
        directory = bytearray(len(names) * 12)
        current_offset = data_start

        # Write name block after directory
        out.seek(64 + len(names) * 12)
        out.write(name_block)

        out.seek(data_start)

        with safe_open(safetensors_path, framework='pt', device='cpu') as f:
            for idx, name in enumerate(names):
                tensor = f.get_tensor(name)

                if tensor.dtype == torch.uint8:
                    raw_bytes, packed_per_row = pack_tensor_row_wise(tensor)
                    sname = name.replace('.weight', '.weight_scale')
                    scale_val = scales.get(sname, 1.0)
                    scale_bytes = struct.pack('<e', float(scale_val))
                    data_bytes = scale_bytes + raw_bytes
                else:
                    packed_per_row = 0
                    data_bytes = tensor.to(torch.float16).numpy().tobytes()

                if current_offset % 32 != 0:
                    pad = 32 - (current_offset % 32)
                    current_offset += pad
                    out.write(b'\x00' * pad)

                ttype = 0 if tensor.dtype == torch.uint8 else (1 if 'norm' in name or 'embed' in name else 2)
                row_dim = tensor.shape[0] * 4 if tensor.dtype == torch.uint8 else tensor.shape[0]

                # Entry: [ttype:1][offset:4][row_dim:4][packed_per_row:3] = 12 bytes
                directory[idx*12] = ttype
                struct.pack_into('<I', directory, idx*12 + 1, current_offset)
                struct.pack_into('<I', directory, idx*12 + 5, row_dim)
                ppr = packed_per_row & 0xFFFFFF
                directory[idx*12 + 9] = ppr & 0xFF
                directory[idx*12 + 10] = (ppr >> 8) & 0xFF
                directory[idx*12 + 11] = (ppr >> 16) & 0xFF

                out.write(data_bytes)
                current_offset += len(data_bytes)
                del tensor

                if idx % 16 == 0:
                    print(f"  [{idx}/{len(names)}] {name[:55]:55s} {len(data_bytes)/1024:7.1f}KB")

        # Append tokenizer block at end of file (v5+)
        if tokenizer_block:
            tokenizer_offset = current_offset
            if tokenizer_offset % 32 != 0:
                pad = 32 - (tokenizer_offset % 32)
                tokenizer_offset += pad
                out.write(b'\x00' * pad)
            out.write(tokenizer_block)
            current_offset = tokenizer_offset + len(tokenizer_block)
            struct.pack_into('<I', header, 29, len(tokenizer_block))
            struct.pack_into('<I', header, 33, tokenizer_offset)

        # v6: Append binary tokenizer block (REQUIRED)
        binary_tok_block = build_tokenizer_binary(model_dir)
        if len(binary_tok_block) == 0:
            print("[ATLAS] ERROR: Could not build v6 binary tokenizer block")
            sys.exit(1)
        struct.pack_into('<H', header, 5, 7)
        binary_offset = current_offset
        if binary_offset % 32 != 0:
            pad = 32 - (binary_offset % 32)
            binary_offset += pad
            out.write(b'\x00' * pad)
        out.write(binary_tok_block)
        current_offset = binary_offset + len(binary_tok_block)
        struct.pack_into('<I', header, 37, len(binary_tok_block))
        struct.pack_into('<I', header, 41, binary_offset)

        # Flush before seeking back — Windows MSVCRT can't seek past 2GB
        # without an explicit flush (buffer contains unwritten data).
        out.flush()
        out.seek(64)
        out.write(directory)
        out.seek(0)
        out.write(header)

    total_gb = current_offset / 1024**3
    print(f"[ATLAS] Done! {total_gb:.2f} GB")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python atlas_packer.py <model.safetensors> <output.atlas>")
        sys.exit(1)
    create_atlas_from_config(sys.argv[1], sys.argv[2])
