"""Edge-case fuzz tests: head_dim boundaries, odd sizes, and TQ2 conversion."""
import os, ctypes, numpy as np, pytest

from atlas_infer import AtlasModel, dll
from tests.atlas_mock_model import make


HERE = os.path.dirname(os.path.abspath(__file__))
FUZZ_DIR = os.path.join(HERE, "..", "fuzz")
os.makedirs(FUZZ_DIR, exist_ok=True)


HEAD_DIMS = [
    1,   # Minimum — single-element per-head attention
    17,  # Odd, not power of 2, not SIMD-aligned
    63,  # 1 below 64 (VNNI chunk boundary)
    64,  # 64 boundary (AVX2 / VNNI chunk size)
    65,  # 1 above 64
    127, # 1 below 128 (block_size boundary)
    128, # Block size boundary
    129, # 1 above 128
    256, # Standard Falcon3 value
]


@pytest.mark.parametrize("hd", HEAD_DIMS)
def test_head_dim_forward(hd):
    """Load model with non-standard head_dim; verify forward pass."""
    path = os.path.join(FUZZ_DIR, f"hd-{hd}.atlas")
    make(path, "falcon3", use_tq1=True, head_dim=hd)
    m = AtlasModel(path)
    assert m.head_dim == hd
    ids = np.array([[1, 2, 3]], dtype=np.int32)
    logits = m.forward(ids)
    assert logits.shape == (1, 3, m.vocab_size)
    assert not np.any(np.isnan(logits))
    assert not np.any(np.isinf(logits))


@pytest.mark.parametrize("hd", [63, 64, 65, 127, 128, 129, 256])
def test_head_dim_tq2(hd):
    """Load + convert-to-tq2 at head_dim boundaries; verify forward pass."""
    path = os.path.join(FUZZ_DIR, f"tq2-hd-{hd}.atlas")
    make(path, "falcon3", use_tq1=True, head_dim=hd)
    m = AtlasModel(path, convert_to_tq2=True)
    assert m.head_dim == hd
    ids = np.array([[1, 2, 3]], dtype=np.int32)
    logits = m.forward(ids)
    assert logits.shape == (1, 3, m.vocab_size)
    assert not np.any(np.isnan(logits))
    assert not np.any(np.isinf(logits))


@pytest.mark.parametrize("hd", [1, 17, 63, 64, 65, 127, 128, 129, 256])
def test_head_dim_batch(hd):
    """Batch forward with edge-case head_dim."""
    path = os.path.join(FUZZ_DIR, f"batch-hd-{hd}.atlas")
    make(path, "falcon3", use_tq1=True, head_dim=hd)
    m = AtlasModel(path)
    ids = np.array([[1], [2], [5]], dtype=np.int32)
    logits = m.forward(ids)
    assert logits.shape == (3, 1, m.vocab_size)
    assert not np.any(np.isnan(logits))
