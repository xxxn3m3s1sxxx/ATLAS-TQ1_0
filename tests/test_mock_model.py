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
