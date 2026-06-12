import os, ctypes, struct, numpy as np, pytest
from atlas_infer import AtlasModel, dll
from tests.atlas_mock_model import make, ARCHES, CORRIDORS

HERE = os.path.dirname(os.path.abspath(__file__))
MOCK_DIR = os.path.join(HERE, "..", "mock")
os.makedirs(MOCK_DIR, exist_ok=True)

def _packing(arch):
    return ARCHES[arch].get("use_tq1", True)

@pytest.mark.parametrize("arch", list(ARCHES.keys()))
def test_load(arch):
    path = os.path.join(MOCK_DIR, f"ci-{arch}.atlas")
    if not os.path.exists(path):
        make(path, arch, use_tq1=_packing(arch))
    m = AtlasModel(path)
    assert m.n_layers == 2

@pytest.mark.parametrize("arch", list(ARCHES.keys()))
def test_forward(arch):
    path = os.path.join(MOCK_DIR, f"ci-{arch}.atlas")
    if not os.path.exists(path):
        make(path, arch, use_tq1=_packing(arch))
    m = AtlasModel(path)
    ids = np.array([[1, 2, 3]], dtype=np.int32)
    logits = m.forward(ids)
    assert logits.shape == (1, 3, m.vocab_size)

@pytest.mark.parametrize("arch", list(ARCHES.keys()))
def test_forward_batch(arch):
    path = os.path.join(MOCK_DIR, f"ci-{arch}.atlas")
    if not os.path.exists(path):
        make(path, arch, use_tq1=_packing(arch))
    m = AtlasModel(path)
    ids = np.array([[1], [2], [5]], dtype=np.int32)
    logits = m.forward(ids)
    assert logits.shape == (3, 1, m.vocab_size)

def test_turboquant_decompress():
    path = os.path.join(MOCK_DIR, "ci-turboquant.atlas")
    if not os.path.exists(path):
        make(path, "turboquant", use_tq1="ttype7")
    m = AtlasModel(path)
    ids = np.array([[1, 2]], dtype=np.int32)
    logits = m.forward(ids)
    assert logits.shape == (1, 2, m.vocab_size)
    assert not np.any(np.isnan(logits))
    assert not np.any(np.isinf(logits))

def test_falcon3_decompress():
    path = os.path.join(MOCK_DIR, "ci-falcon3.atlas")
    if not os.path.exists(path):
        make(path, "falcon3")
    m = AtlasModel(path)
    ids = np.array([[1, 2]], dtype=np.int32)
    logits = m.forward(ids)
    assert logits.shape == (1, 2, m.vocab_size)
    assert not np.any(np.isnan(logits))
    assert not np.any(np.isinf(logits))

def test_sampling_coverage():
    path = os.path.join(MOCK_DIR, "ci-falcon3.atlas")
    if not os.path.exists(path):
        make(path, "falcon3", use_tq1=_packing("falcon3"))
    m = AtlasModel(path)
    V = m.vocab_size
    logits = np.random.randn(max(V, 256)).astype(np.float32) * 0.1
    logits_ptr = logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out = ctypes.c_int()
    dll.atlas_set_seed(42)
    dll.atlas_sample(m.model_ptr, logits_ptr, ctypes.byref(out), 0.0, 1, 0.0)
    assert 0 <= out.value < V
    dll.atlas_set_seed(42)
    dll.atlas_sample(m.model_ptr, logits_ptr, ctypes.byref(out), 0.7, 40, 0.0)
    assert 0 <= out.value < V
    dll.atlas_set_seed(42)
    dll.atlas_sample(m.model_ptr, logits_ptr, ctypes.byref(out), 0.7, 40, 0.5)
    assert 0 <= out.value < V
    dll.atlas_set_seed(42)
    dll.atlas_sample(m.model_ptr, logits_ptr, ctypes.byref(out), 0.7, 0, 0.0)
    assert 0 <= out.value < V
    dll.atlas_set_seed(42)
    dll.atlas_sample(m.model_ptr, logits_ptr, ctypes.byref(out), 1.5, 256, 0.0)
    assert 0 <= out.value < V

@pytest.mark.parametrize("key", list(CORRIDORS.keys()))
def test_corridor_load(key):
    core = CORRIDORS[key]
    path = os.path.join(MOCK_DIR, f"corridor-{key}.atlas")
    if not os.path.exists(path):
        make(path, "falcon3", corridor=key)
    m = AtlasModel(path)
    for func_name, args in core["post_init"]:
        fn = getattr(dll, func_name)
        fn(m.model_ptr, *args)
    assert m.n_layers == 2
    ids = np.array([[1, 2, 3]], dtype=np.int32)
    logits = m.forward(ids)
    assert logits.shape == (1, 3, m.vocab_size)
    assert not np.any(np.isnan(logits))
    assert not np.any(np.isinf(logits))

BONSAI_CORRIDORS = {
    "bonsai_1_7b": "f32_bypass",
    "bonsai_4b": "production_int8",
    "bonsai_8b": "f32_bypass",
    "bitnet_2b4t": "f32_bypass",
    "bitcpm": "f32_bypass",
}

@pytest.mark.parametrize("arch,corridor", list(BONSAI_CORRIDORS.items()))
def test_bonsai_corridor_load(arch, corridor):
    core = CORRIDORS[corridor]
    path = os.path.join(MOCK_DIR, f"bonsai-{arch}.atlas")
    if not os.path.exists(path):
        make(path, arch, corridor=corridor)
    m = AtlasModel(path)
    for func_name, args in core["post_init"]:
        fn = getattr(dll, func_name)
        fn(m.model_ptr, *args)
    assert m.n_layers == 2
    ids = np.array([[1, 2, 3]], dtype=np.int32)
    logits = m.forward(ids)
    assert logits.shape == (1, 3, m.vocab_size)
    assert not np.any(np.isnan(logits))
    assert not np.any(np.isinf(logits))

TRILM_CORRIDORS = {
    "trilm_1_5b": "f32_bypass",
    "trilm_2_4b": "production_int8",
    "trilm_3_9b": "production_int8",
}

@pytest.mark.parametrize("arch,corridor", list(TRILM_CORRIDORS.items()))
def test_trilm_corridor_load(arch, corridor):
    core = CORRIDORS[corridor]
    path = os.path.join(MOCK_DIR, f"trilm-{arch}.atlas")
    if not os.path.exists(path):
        make(path, arch, corridor=corridor)
    m = AtlasModel(path)
    for func_name, args in core["post_init"]:
        fn = getattr(dll, func_name)
        fn(m.model_ptr, *args)
    assert m.n_layers == 2
    ids = np.array([[1, 2, 3]], dtype=np.int32)
    logits = m.forward(ids)
    assert logits.shape == (1, 3, m.vocab_size)
    assert not np.any(np.isnan(logits))
    assert not np.any(np.isinf(logits))

# ─── v6 Binary Tokenizer with Added Tokens ─────────────────────────────

def _build_v6_binary(vocab_size=256, added_tokens=None):
    """Build a v6 binary tokenizer block with optional added_tokens."""
    if added_tokens is None:
        added_tokens = []
    # Sort added tokens by length descending (longest-match-first)
    added_tokens = sorted(set(added_tokens), key=lambda x: -len(x[0]))
    V = vocab_size

    # Build dummy vocab
    offsets = np.arange(V, dtype=np.uint32) * 2
    lengths = np.full(V, 2, dtype=np.uint16)
    pool = b"".join(chr(256 + i & 0xFF).encode("utf-8") for i in range(V))
    max_token_length = max(lengths) if V > 0 else 0

    merge_left = np.full(V, 0xFFFFFFFF, dtype=np.uint32)
    merge_right = np.full(V, 0xFFFFFFFF, dtype=np.uint32)
    merge_rank = np.zeros(V, dtype=np.uint32)

    byte_encoder = np.full(256, 0xFFFF, dtype=np.uint16)
    for b in range(256):
        if b < V:
            byte_encoder[b] = b

    byte_decoder = np.full(256, 0xFFFF, dtype=np.uint16)
    for b in range(256):
        byte_decoder[b] = b

    special_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)  # eos/bos/pad/unk/...

    off = 128
    off_offsets = off
    off += V * 4
    off_lengths = off
    off += V * 2
    off_pool = off
    off += len(pool)
    pool_pad = (4 - len(pool) % 4) % 4
    off += pool_pad
    off_merge_left = off
    off += V * 4
    off_merge_right = off
    off += V * 4
    off_merge_rank = off
    off += V * 4
    off_byte_enc = off
    off += 512
    off_byte_dec = off
    off += 512
    off_special = off
    off += 28
    off_added_tokens = off if added_tokens else 0

    added_data = b""
    if added_tokens:
        added_data += struct.pack("<I", len(added_tokens))
        for tbytes, tid in added_tokens:
            slen = len(tbytes)
            added_data += struct.pack("<I", slen)
            added_data += tbytes
            pad_len = (4 - slen % 4) % 4
            if pad_len:
                added_data += b"\x00" * pad_len
            added_data += struct.pack("<I", tid)
        off += len(added_data)

    total_size = off
    buf = bytearray(total_size)
    struct.pack_into("<I", buf, 0, 0x544F4B42)
    struct.pack_into("<I", buf, 4, 1)
    struct.pack_into("<I", buf, 8, V)
    struct.pack_into("<I", buf, 12, max_token_length)
    struct.pack_into("<I", buf, 16, len(added_tokens))
    struct.pack_into("<I", buf, 20, 0)
    offs_list = [off_offsets, off_lengths, off_pool, len(pool),
                 off_merge_left, off_merge_right, off_merge_rank,
                 off_byte_enc, off_byte_dec, off_special,
                 off_added_tokens]
    for i, val in enumerate(offs_list):
        struct.pack_into("<Q", buf, 24 + i * 8, val)
    buf[off_offsets:off_offsets + V * 4] = offsets.tobytes()
    buf[off_lengths:off_lengths + V * 2] = lengths.tobytes()
    buf[off_pool:off_pool + len(pool)] = pool
    buf[off_merge_left:off_merge_left + V * 4] = merge_left.tobytes()
    buf[off_merge_right:off_merge_right + V * 4] = merge_right.tobytes()
    buf[off_merge_rank:off_merge_rank + V * 4] = merge_rank.tobytes()
    buf[off_byte_enc:off_byte_enc + 512] = byte_encoder.tobytes()
    buf[off_byte_dec:off_byte_dec + 512] = byte_decoder.tobytes()
    buf[off_special:off_special + 28] = special_arr.tobytes()
    if added_tokens and off_added_tokens:
        buf[off_added_tokens:off_added_tokens + len(added_data)] = added_data
    return bytes(buf)


def _write_minimal_atlas_v6(path, v6_binary):
    """Write a minimal .atlas file (v7) with a v6 binary tokenizer block.
    
    The model has 1 layer, 2 tensors (norm + embed) — enough for atlas_load
    to succeed. Used to test the C++ tokenizer APIs (preencode/decode with
    added tokens).
    """
    V = 256
    hidden = 64
    inter = 128
    n_layers = 1
    n_tensors = 2  # model.norm.weight + model.embed_tokens.weight
    
    # Tiny tensor data (32-byte aligned)
    norm_data = struct.pack("<e", 1.0) * hidden  # fp16 norm weights
    embed_data = struct.pack("<e", 0.01) * (V * hidden)  # fp16 embed
    
    header = bytearray(64)
    header[0:5] = b"ATLAS"
    struct.pack_into("<H", header, 5, 7)   # version=7 (v6 binary)
    struct.pack_into("<H", header, 7, n_layers)
    struct.pack_into("<H", header, 9, hidden)
    struct.pack_into("<H", header, 11, inter)
    struct.pack_into("<B", header, 13, 2)   # n_heads
    struct.pack_into("<B", header, 14, 1)   # n_kv_heads
    struct.pack_into("<H", header, 15, 64)  # head_dim
    struct.pack_into("<I", header, 17, V)   # vocab
    struct.pack_into("<d", header, 21, 10000.0)  # rope_theta

    # Meta (empty)
    meta_bytes = b'{"arch":"falcon3","rope_interleaved":true,"use_f32_bypass":1,"rope_theta":10000.0,"head_dim":64,"rope_scale":1.0,"base_seq_len":2048}'
    meta_block = struct.pack("<I", 4 + len(meta_bytes)) + meta_bytes
    
    # Name block
    name_bytes = b"model.norm.weight\x00model.embed_tokens.weight\x00"
    name_block = struct.pack("<I", 4 + len(name_bytes)) + name_bytes
    
    dir_start = 64 + len(meta_block)
    data_start = dir_start + n_tensors * 12 + len(name_block)

    with open(path, "wb") as f:
        f.write(header)
        f.write(meta_block)
        
        # Directory
        dir_buf = bytearray(n_tensors * 12)
        idx = 0
        
        # model.norm.weight — ttype=1 (fp16)
        off = data_start
        dir_buf[idx * 12] = 1  # ttype
        struct.pack_into("<I", dir_buf, idx * 12 + 1, off)
        struct.pack_into("<I", dir_buf, idx * 12 + 5, hidden)  # row_dim
        dir_buf[idx * 12 + 9] = 0; dir_buf[idx * 12 + 10] = 0; dir_buf[idx * 12 + 11] = 0
        idx += 1
        
        # model.embed_tokens.weight — ttype=1 (fp16)
        off += len(norm_data)
        # Align to 32
        if off % 32 != 0:
            off += 32 - (off % 32)
        dir_buf[idx * 12] = 1
        struct.pack_into("<I", dir_buf, idx * 12 + 1, off)
        struct.pack_into("<I", dir_buf, idx * 12 + 5, V)
        dir_buf[idx * 12 + 9] = 0; dir_buf[idx * 12 + 10] = 0; dir_buf[idx * 12 + 11] = 0
        
        f.write(dir_buf)
        f.write(name_block)
        
        # Norm data
        cur = data_start
        f.write(norm_data)
        cur += len(norm_data)
        
        # Embed data (aligned)
        if cur % 32 != 0:
            pad = 32 - (cur % 32)
            f.write(b"\x00" * pad)
            cur += pad
        f.write(embed_data)
        cur += len(embed_data)
        
        # v6 binary tokenizer block (aligned)
        binary_offset = cur
        if binary_offset % 32 != 0:
            pad = 32 - (binary_offset % 32)
            f.write(b"\x00" * pad)
            binary_offset += pad
        f.write(v6_binary)
        
        # Update header
        struct.pack_into("<I", header, 29, 0)  # tokenizer_size=0 (no v5 JSON)
        struct.pack_into("<I", header, 33, 0)
        struct.pack_into("<I", header, 37, len(v6_binary))
        struct.pack_into("<I", header, 41, binary_offset)
        struct.pack_into("<I", header, 56, len(name_block))
        struct.pack_into("<I", header, 60, n_tensors)
        f.seek(0)
        f.write(header)


def test_v6_tokenizer_added_tokens():
    """Verify C++ preencode/decode correctly handles out-of-vocab added tokens.
    Uses direct ctypes calls to bypass Python AtlasModel init (which needs real tensors)."""
    added = [
        (b"<|eot_id|>", 128009),
        (b"<|start_header_id|>", 128006),
        (b"<|begin_of_text|>", 128000),
    ]
    v6bin = _build_v6_binary(vocab_size=256, added_tokens=added)
    path = os.path.join(MOCK_DIR, "test_v6_added_tokens.atlas")
    _write_minimal_atlas_v6(path, v6bin)

    # Load directly via C API
    model_ptr = dll.atlas_load(path.encode("utf-8"))
    assert model_ptr, "atlas_load failed"
    assert dll.atlas_has_binary_tokenizer(model_ptr) == 1

    # Test preencode: added tokens should be resolved as atomic IDs
    text = b"hello<|start_header_id|>world<|eot_id|>"
    max_ids = len(text) + 32
    ids_arr = (ctypes.c_int * max_ids)()
    n_ids = dll.atlas_tokenizer_preencode(model_ptr, text, len(text), ids_arr, max_ids)
    assert n_ids > 0, f"preencode returned {n_ids}"
    ids = list(ids_arr[:n_ids])

    # Added tokens (128006, 128009) must appear as atomic IDs
    assert 128006 in ids, f"<|start_header_id|> not found in {ids}"
    assert 128009 in ids, f"<|eot_id|> not found in {ids}"
    assert ids.index(128006) < ids.index(128009), "Token order wrong"

    # Test decode round-trip: added tokens survive decode
    max_out = n_ids * 16 + 64
    out_buf = ctypes.create_string_buffer(max_out)
    n_bytes = dll.atlas_tokenizer_decode(model_ptr, ids_arr, n_ids, out_buf, max_out)
    assert n_bytes > 0, f"decode returned {n_bytes}"
    decoded = bytes(out_buf[:n_bytes]).decode("utf-8")
    assert "<|start_header_id|>" in decoded, f"Decode lost: {decoded!r}"
    assert "<|eot_id|>" in decoded, f"Decode lost: {decoded!r}"

    # Test decode of individual added token IDs
    for tid, expected in [(128000, "<|begin_of_text|>"), (128006, "<|start_header_id|>"), (128009, "<|eot_id|>")]:
        single = (ctypes.c_int * 1)(tid)
        out = ctypes.create_string_buffer(64)
        nb = dll.atlas_tokenizer_decode(model_ptr, single, 1, out, 64)
        result = bytes(out[:nb]).decode("utf-8")
        assert result == expected, f"decode({tid}) = {result!r}, expected {expected!r}"

    # Skip both atlas_free and os.remove — the minimal model has enough to load+bind
    # tokenizer but is structurally incomplete (missing layer tensors), so cleanup
    # would crash and the file is mmap'd. OS reaps all on exit.
