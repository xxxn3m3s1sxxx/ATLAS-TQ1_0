"""CI smoke test: generate mock models and run forward pass for all arches."""
import os, numpy as np, pytest

from atlas_infer import AtlasModel
from tests.atlas_mock_model import make, ARCHES

HERE = os.path.dirname(os.path.abspath(__file__))
MOCK_DIR = os.path.join(HERE, "..", "mock")
os.makedirs(MOCK_DIR, exist_ok=True)


@pytest.mark.parametrize("arch", list(ARCHES.keys()))
def test_load(arch):
    path = os.path.join(MOCK_DIR, f"ci-{arch}.atlas")
    if not os.path.exists(path):
        make(path, arch, use_tq1=True)
    m = AtlasModel(path)
    assert m.n_layers == 2
    assert m.hidden == 128
    assert m.vocab_size == 256


@pytest.mark.parametrize("arch", list(ARCHES.keys()))
def test_forward(arch):
    path = os.path.join(MOCK_DIR, f"ci-{arch}.atlas")
    if not os.path.exists(path):
        make(path, arch, use_tq1=True)
    m = AtlasModel(path)
    ids = np.array([[1, 2, 3]], dtype=np.int32)
    logits = m.forward(ids)
    assert logits.shape == (1, 3, 256)


@pytest.mark.parametrize("arch", list(ARCHES.keys()))
def test_forward_batch(arch):
    path = os.path.join(MOCK_DIR, f"ci-{arch}.atlas")
    if not os.path.exists(path):
        make(path, arch, use_tq1=True)
    m = AtlasModel(path)
    ids = np.array([[1], [2], [5]], dtype=np.int32)
    logits = m.forward(ids)
    assert logits.shape == (3, 1, 256)
