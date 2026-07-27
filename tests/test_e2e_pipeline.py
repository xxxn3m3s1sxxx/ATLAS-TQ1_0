"""End-to-end pipeline test — completes coverage for all remaining atlas_* API functions.

Covers the full cycle across all 4 dispatch corridors:
1. atlas_get_config        — struct return type query
2. atlas_get_tensor_index  — tensor lookup by name  
3. atlas_tensor_info       — tensor metadata query
4. atlas_reset_cache       — KV cache zeroing (called twice: fresh + mid-stream)
5. atlas_set_rope_interleaved — RoPE pair order toggle
6. atlas_set_rope_scale       — YaRN NTK scale
7. atlas_set_rope_theta        — frequency override
8. atlas_set_use_hybrid_matmul — matmul dispatch toggle
9. atlas_set_layer_stride      — layer index stride
10. atlas_generate_stream    — streaming generation with callback
11. atlas_tokenizer_preencode — byte→ID mapping (error path: -1 on mock)
12. atlas_tokenizer_merge     — BPE merge (error path: -1 on mock)
13. atlas_tokenizer_decode    — ID→text decode (error path: -1 on mock)
14. atlas_rope_f32            — standalone RoPE kernel
15. atlas_sample              — standalone sampling
16. atlas_decompress_ffn      — FFN decompress

Pipeline sequence: load → get_config → setters → generate_stream → 
                    reset_cache → generate_stream_again → tokenizer_apis →
                    rope_kernel → sample_kernel → decompress → done
"""
import os
import ctypes
import numpy as np
import pytest

from atlas_infer import AtlasModel, dll
from tests.atlas_mock_model import make, CORRIDORS

HERE = os.path.dirname(os.path.abspath(__file__))
MOCK_DIR = os.path.join(HERE, "..", "mock")
os.makedirs(MOCK_DIR, exist_ok=True)

TEST_CORRIDORS = list(CORRIDORS.keys())


class AtlasModelConfig(ctypes.Structure):
    _fields_ = [
        ("n_layers", ctypes.c_int),
        ("hidden_dim", ctypes.c_int),
        ("inter_dim", ctypes.c_int),
        ("n_heads", ctypes.c_int),
        ("n_kv_heads", ctypes.c_int),
        ("head_dim", ctypes.c_int),
        ("vocab_size", ctypes.c_int),
        ("rope_theta", ctypes.c_float),
    ]


dll.atlas_get_config.restype = AtlasModelConfig
dll.atlas_get_config.argtypes = [ctypes.c_void_p]

TOKEN_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p)
LOGIT_PROCESSOR_CB = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p)
TOKEN_NOTIFY_CB = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p)

_NOOP_LOGIT_CB = LOGIT_PROCESSOR_CB(lambda logits, n, data: None)
_NOOP_TOKEN_NOTIFY_CB = TOKEN_NOTIFY_CB(lambda tid, data: None)


def _apply_corridor_post_init(m, core):
    for func_name, args in core["post_init"]:
        fn = getattr(dll, func_name)
        fn(m.model_ptr, *args)


@pytest.mark.parametrize("corridor", TEST_CORRIDORS)
def test_e2e_pipeline(corridor):
    core = CORRIDORS[corridor]
    path = os.path.join(MOCK_DIR, f"e2e-{corridor}.atlas")
    if not os.path.exists(path):
        make(path, "falcon3", corridor=corridor)

    os.environ["OMP_NUM_THREADS"] = "1"
    m = AtlasModel(path)
    rc = core.get("requires_hidden_gt_2048", False)
    if rc:
        dll.atlas_set_use_f32_matmul(m.model_ptr, 0)
    _apply_corridor_post_init(m, core)
    m.set_seed(42)

    # ── 1. atlas_get_config ──
    cfg = dll.atlas_get_config(m.model_ptr)
    assert cfg.n_layers == 2
    assert cfg.hidden_dim == 128
    assert cfg.vocab_size == 256

    # ── 2. atlas_get_tensor_index ──
    idx = dll.atlas_get_tensor_index(m.model_ptr, b"model.norm.weight")
    assert idx >= 0
    idx_embed = dll.atlas_get_tensor_index(m.model_ptr, b"model.embed_tokens.weight")
    assert idx_embed >= 0

    # ── 3. atlas_tensor_info ──
    tt = ctypes.c_int()
    rd = ctypes.c_int()
    cd = ctypes.c_int()
    dll.atlas_tensor_info(m.model_ptr, idx, tt, rd, cd)
    assert rd.value == 128

    # ── 4–9. Setter APIs ──
    dll.atlas_set_rope_interleaved(m.model_ptr, 1)
    dll.atlas_set_rope_interleaved(m.model_ptr, 0)
    dll.atlas_set_rope_theta(m.model_ptr, 1000000.0)
    dll.atlas_set_rope_scale(m.model_ptr, 4.0)
    dll.atlas_set_layer_stride(m.model_ptr, 9)
    dll.atlas_set_use_hybrid_matmul(m.model_ptr, 0)
    dll.atlas_set_use_hybrid_matmul(m.model_ptr, 1)

    # ── 10. atlas_generate_stream (first cycle) ──
    collected = []
    cb = TOKEN_CALLBACK(lambda tid, _: collected.append(tid))
    in_ids = (ctypes.c_int * 3)(1, 2, 3)
    n_gen = dll.atlas_generate_stream(
        m.model_ptr, in_ids, 3, 512, 10, 0.0, 1, 0.0, 1.0, 0, 0, cb, None, 0,
        _NOOP_LOGIT_CB, None, _NOOP_TOKEN_NOTIFY_CB, None)
    assert n_gen > 0, f"generate_stream returned {n_gen}"
    assert len(collected) == n_gen
    first_batch = list(collected)

    # ── 4b. atlas_reset_cache (mid-stream) ──
    m.reset_cache()

    # ── 10b. atlas_generate_stream (second cycle, post-reset) ──
    collected2 = []
    cb2 = TOKEN_CALLBACK(lambda tid, _: collected2.append(tid))
    n_gen2 = dll.atlas_generate_stream(
        m.model_ptr, in_ids, 3, 512, 5, 0.0, 1, 0.0, 1.0, 0, 0, cb2, None, 0,
        _NOOP_LOGIT_CB, None, _NOOP_TOKEN_NOTIFY_CB, None)
    assert n_gen2 > 0
    assert len(collected2) == n_gen2
    assert collected2 != first_batch  # reset changes generation

    # ── 11–13. Tokenizer APIs (error path: no binary tokenizer in mock) ──
    text = b"hello world"
    ids_arr = (ctypes.c_int * 256)()
    ret = dll.atlas_tokenizer_preencode(m.model_ptr, text, len(text), ids_arr, 256)
    assert ret == -1, f"preencode expected -1, got {ret}"
    n_ids = ctypes.c_int(3)
    ret = dll.atlas_tokenizer_merge(m.model_ptr, ids_arr, n_ids)
    assert ret == -1, f"merge expected -1, got {ret}"
    out_buf = ctypes.create_string_buffer(64)
    ret = dll.atlas_tokenizer_decode(m.model_ptr, ids_arr, 3, out_buf, 64)
    assert ret == -1, f"decode expected -1, got {ret}"

    # ── 14. atlas_rope_f32 (standalone kernel) ──
    H = cfg.hidden_dim
    nh = cfg.n_heads
    nk = cfg.n_kv_heads
    hd = cfg.head_dim
    q = np.ones(nh * hd, dtype=np.float32)
    k = np.ones(nk * hd, dtype=np.float32)
    dll.atlas_rope_f32(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        nh, nk, hd, 0, 10000.0, True)
    assert not np.any(np.isnan(q))
    assert not np.any(np.isinf(q))

    # ── 15. atlas_sample (standalone sampling) ──
    V = m.vocab_size
    logits = np.random.randn(V).astype(np.float32) * 0.1
    logits[1] = 10.0  # deterministic argmax → 1
    token = ctypes.c_int()
    dll.atlas_set_seed(42)
    dll.atlas_sample(m.model_ptr,
        logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(token), 0.0, 0, 0.0)
    assert token.value == 1

    # ── 16. atlas_decompress_ffn (FFN decompress) ──
    dll.atlas_decompress_ffn(m.model_ptr)

    m.reset_cache()
    del m
