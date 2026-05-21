"""Fast v6 block addition to existing atlas file."""
import struct, json, os, sys, time
import numpy as np
from tokenizers import Tokenizer

def add_v6_block_fast(model_dir, atlas_path):
    t0 = time.time()
    tok = Tokenizer.from_file(os.path.join(model_dir, 'tokenizer.json'))
    vocab = tok.get_vocab()
    V = len(vocab)
    sorted_items = sorted(vocab.items(), key=lambda kv: kv[1])
    print(f'Vocab: {V} tokens ({time.time()-t0:.1f}s)')

    # Decoder arrays
    t1 = time.time()
    offsets = np.empty(V, dtype=np.uint32)
    lengths = np.empty(V, dtype=np.uint16)
    pool_parts = []
    for i, (token_str, tid) in enumerate(sorted_items):
        token_bytes = token_str.encode('utf-8')
        offsets[i] = sum(len(p) for p in pool_parts)
        lengths[i] = len(token_bytes)
        pool_parts.append(token_bytes)
    pool = b''.join(pool_parts)
    print(f'Decoder arrays: pool={len(pool)/1024:.1f} KB ({time.time()-t1:.1f}s)')

    # Merge arrays (read from tokenizer.json directly)
    t1 = time.time()
    merge_left = np.full(V, 0xFFFFFFFF, dtype=np.uint32)
    merge_right = np.full(V, 0xFFFFFFFF, dtype=np.uint32)
    merge_rank = np.zeros(V, dtype=np.uint32)

    import json as _json
    tok_path = os.path.join(model_dir, 'tokenizer.json')
    with open(tok_path, 'r', encoding='utf-8') as jf:
        tok_json_data = _json.load(jf)
    merges_list = tok_json_data.get('model', {}).get('merges', [])
    base_vocab_size = V - len(merges_list)
    for i, merge_pair in enumerate(merges_list):
        if isinstance(merge_pair, list) and len(merge_pair) == 2:
            left_str, right_str = merge_pair
        elif isinstance(merge_pair, str):
            parts = merge_pair.split()
            if len(parts) != 2: continue
            left_str, right_str = parts
        else:
            continue
        lid = vocab.get(left_str, -1)
        rid = vocab.get(right_str, -1)
        if lid >= 0 and rid >= 0:
            merged_id = base_vocab_size + i
            if merged_id < V:
                merge_left[merged_id] = lid
                merge_right[merged_id] = rid
                merge_rank[merged_id] = i + 1
    print(f'Merge arrays: {len(merges_list)} merges ({time.time()-t1:.1f}s)')

    # Byte encoder via bytes_to_unicode
    t1 = time.time()
    bs = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    byte_encoder = np.full(256, 0xFFFF, dtype=np.uint16)
    n_count = 0
    for b in range(256):
        token_str = chr(b) if b in bs else chr(256 + n_count)
        if b not in bs: n_count += 1
        tid = vocab.get(token_str)
        if tid is not None: byte_encoder[b] = tid
    # Fill remaining via id_to_token
    for b in range(256):
        if byte_encoder[b] != 0xFFFF: continue
        for tid in range(min(V, 512)):
            tstr = tok.id_to_token(tid)
            if tstr and len(tstr) == 1 and (ord(tstr) - 256) == b:
                byte_encoder[b] = tid
                break
    byte_decoder = np.full(256, 0xFFFF, dtype=np.uint16)
    for b in range(256):
        if byte_encoder[b] != 0xFFFF: byte_decoder[b] = b
    print(f'Byte encoder: {np.sum(byte_encoder != 0xFFFF)}/256 mapped ({time.time()-t1:.1f}s)')

    # Special tokens
    t1 = time.time()
    eos_id = 0
    cfg_path = os.path.join(model_dir, 'tokenizer_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as cf:
            cfg = json.load(cf)
        eos_cfg = cfg.get('eos_token')
        if eos_cfg and isinstance(eos_cfg, dict) and 'id' in eos_cfg:
            eos_id = eos_cfg['id']
    special_arr = np.array([eos_id, 0xFFFFFFFF, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF], dtype=np.uint32)
    print(f'Special tokens ({time.time()-t1:.1f}s)')

    # Pack binary block
    t1 = time.time()
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
    # Header
    struct.pack_into('<I', buf, 0, 0x544F4B42)
    struct.pack_into('<I', buf, 4, 1)
    struct.pack_into('<I', buf, 8, V)
    struct.pack_into('<I', buf, 12, int(max(lengths)))
    struct.pack_into('<I', buf, 20, 0)
    offs_list = [off_offsets, off_lengths, off_pool, len(pool),
                 off_merge_left, off_merge_right, off_merge_rank,
                 off_byte_enc, off_byte_dec, off_special]
    for i, val in enumerate(offs_list):
        struct.pack_into('<Q', buf, 24 + i * 8, val)
    # Data sections
    buf[off_offsets:off_offsets + V * 4] = offsets.tobytes()
    buf[off_lengths:off_lengths + V * 2] = lengths.tobytes()
    buf[off_pool:off_pool + len(pool)] = pool
    buf[off_merge_left:off_merge_left + V * 4] = merge_left.tobytes()
    buf[off_merge_right:off_merge_right + V * 4] = merge_right.tobytes()
    buf[off_merge_rank:off_merge_rank + V * 4] = merge_rank.tobytes()
    buf[off_byte_enc:off_byte_enc + 512] = byte_encoder.tobytes()
    buf[off_byte_dec:off_byte_dec + 512] = byte_decoder.tobytes()
    buf[off_special:off_special + 28] = special_arr.tobytes()
    print(f'Binary block: {total_size/1024/1024:.2f} MB ({time.time()-t1:.1f}s)')

    # Write to file
    t1 = time.time()
    with open(atlas_path, 'r+b') as f:
        f.seek(0, 2)
        current_offset = f.tell()
        if current_offset % 32 != 0:
            pad = 32 - (current_offset % 32)
            f.write(b'\x00' * pad)
            current_offset += pad
        f.write(buf)
        binary_offset = current_offset
        binary_size = len(buf)

        # Update header
        f.seek(0)
        orig_hdr = bytearray(f.read(64))
        struct.pack_into('<H', orig_hdr, 5, 6)
        struct.pack_into('<I', orig_hdr, 37, binary_size)
        struct.pack_into('<I', orig_hdr, 41, binary_offset)
        f.seek(0)
        f.write(orig_hdr)
    print(f'File updated: +{binary_size/1024/1024:.2f} MB at offset {binary_offset/1024**3:.2f} GB')
    print(f'Total time: {time.time()-t0:.1f}s')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python add_v6_block.py <model_dir> <atlas_path>")
        sys.exit(1)
    add_v6_block_fast(sys.argv[1], sys.argv[2])
