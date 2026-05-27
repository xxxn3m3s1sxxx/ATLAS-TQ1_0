"""Verify DLL loads and all C API symbols are exported."""
import ctypes, os, sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DLL_PATH = os.path.join(PROJECT_ROOT, "atlas.dll")
pytestmark = pytest.mark.skipif(not os.path.exists(DLL_PATH), reason="atlas.dll not found")


@pytest.fixture(scope="module")
def dll():
    return ctypes.CDLL(DLL_PATH)


API_SYMBOLS = [
    "atlas_load",
    "atlas_free",
    "atlas_generate",
    "atlas_set_seed",
    "atlas_set_num_threads",
    "atlas_reset_cache",
    "atlas_set_base_seq_len",
    "atlas_set_rope_scale",
    "atlas_set_use_hybrid_matmul",
    "atlas_set_use_f32_matmul",
    "atlas_set_use_packed_matmul",
    "atlas_set_use_ternary_matmul",
    "atlas_tensor_info",
    "atlas_tensor_data",
    "atlas_get_tensor_count",
    "atlas_get_tensor_name",
    "atlas_get_tensor_index",
]


def test_all_api_symbols_exported(dll):
    missing = [s for s in API_SYMBOLS if not hasattr(dll, s)]
    assert not missing, f"Missing DLL symbols: {missing}"


def test_reset_cache_argtypes(dll):
    """atlas_reset_cache should accept void* and return void."""
    dll.atlas_reset_cache.argtypes = [ctypes.c_void_p]
    dll.atlas_reset_cache.restype = None
    # Calling with NULL should not crash
    dll.atlas_reset_cache(None)


def test_generate_argtypes(dll):
    """atlas_generate should have correct signature."""
    sig = dll.atlas_generate
    sig.argtypes = [
        ctypes.c_void_p,                                 # model
        ctypes.POINTER(ctypes.c_int),                    # input_ids
        ctypes.c_int,                                    # n_input
        ctypes.c_int,                                    # max_seq_len
        ctypes.c_int,                                    # max_new_tokens
        ctypes.c_float,                                  # temperature
        ctypes.c_int,                                    # top_k
        ctypes.c_float,                                  # top_p
        ctypes.c_float,                                  # repetition_penalty
        ctypes.c_int,                                    # min_new_tokens
        ctypes.c_int,                                    # cache_offset
        ctypes.POINTER(ctypes.c_int),                    # output_ids
    ]
    sig.restype = ctypes.c_int


def test_atlas_infer_import():
    """atlas_infer should import without errors."""
    import atlas_infer
    assert atlas_infer.AtlasModel


def test_atlas_server_import():
    """atlas_server should import without errors."""
    import atlas_server
    assert atlas_server.app
