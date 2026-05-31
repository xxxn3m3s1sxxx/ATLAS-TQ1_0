"""CI smoke test: generate mock models and run forward pass for all arches."""
import os, ctypes, numpy as np, pytest

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
    """Verify ttype=7 tensors are decompressed in-C++ (coverage for atlas_decompress_ttype7 body)."""
    path = os.path.join(MOCK_DIR, "ci-turboquant.atlas")
    if not os.path.exists(path):
        make(path, "turboquant", use_tq1="ttype7")
    m = AtlasModel(path)
    # Load triggers full decompress; forward confirms decompressed tensors produce valid logits
    ids = np.array([[1, 2]], dtype=np.int32)
    logits = m.forward(ids)
    assert logits.shape == (1, 2, m.vocab_size)
    # Verify no crash or NaN
    assert not np.any(np.isnan(logits))
    assert not np.any(np.isinf(logits))


def test_sampling_coverage():
    """Cover xoshiro RNG, gumbel_sample, top-k/top-p/argmax paths."""
    path = os.path.join(MOCK_DIR, "ci-falcon3.atlas")
    if not os.path.exists(path):
        make(path, "falcon3", use_tq1=True)
    m = AtlasModel(path)
    V = m.vocab_size

    logits = np.random.randn(max(V, 256)).astype(np.float32) * 0.1
    logits_ptr = logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out = ctypes.c_int()

    # 1. Argmax path (T=0)
    dll.atlas_set_seed(42)
    dll.atlas_sample(m.model_ptr, logits_ptr, ctypes.byref(out), 0.0, 1, 0.0)
    assert 0 <= out.value < V

    # 2. Softmax+multinomial (T=0.7, top_k=40)
    dll.atlas_set_seed(42)
    dll.atlas_sample(m.model_ptr, logits_ptr, ctypes.byref(out), 0.7, 40, 0.0)
    assert 0 <= out.value < V

    # 3. Top-p probability path (T=0.7, top_k=40, top_p=0.5)
    dll.atlas_set_seed(42)
    dll.atlas_sample(m.model_ptr, logits_ptr, ctypes.byref(out), 0.7, 40, 0.5)
    assert 0 <= out.value < V

    # 4. No top-k filtering (top_k=0)
    dll.atlas_set_seed(42)
    dll.atlas_sample(m.model_ptr, logits_ptr, ctypes.byref(out), 0.7, 0, 0.0)
    assert 0 <= out.value < V

    # 5. Temperature scaling only (T=1.5)
    dll.atlas_set_seed(42)
    dll.atlas_sample(m.model_ptr, logits_ptr, ctypes.byref(out), 1.5, 256, 0.0)
    assert 0 <= out.value < V


@pytest.mark.parametrize("key", list(CORRIDORS.keys()))
def test_corridor_load(key):
    """Load model with corridor dispatch config; verify post_init C API calls."""
    core = CORRIDORS[key]
    path = os.path.join(MOCK_DIR, f"corridor-{key}.atlas")
    if not os.path.exists(path):
        make(path, "falcon3", corridor=key)
    m = AtlasModel(path)
    # Post-init C API calls to override dispatch state (e.g. reset f32, re-run quantize)
    for func_name, args in core["post_init"]:
        fn = getattr(dll, func_name)
        fn(m.model_ptr, *args)
    assert m.n_layers == 2
    ids = np.array([[1, 2, 3]], dtype=np.int32)
    logits = m.forward(ids)
    assert logits.shape == (1, 3, m.vocab_size)
    assert not np.any(np.isnan(logits))
    assert not np.any(np.isinf(logits))
