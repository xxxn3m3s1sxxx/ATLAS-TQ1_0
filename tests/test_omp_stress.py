"""OpenMP thread-safety stress test for ATLAS dispatch corridors.

Runs 50 consecutive forward passes per corridor under OMP_NUM_THREADS=N_CORES,
then bit-verifies each output against a single-thread reference.

Corridors tested:
  - production_int8:  int8 QKV + int4-packed FFN (7B/10B production path)
  - pure_ternary:     ternary sign-of-int8 matmul (vpsignb, no multiply)
  - int4_ffn_packed:  int4-packed FFN via atlas_quantize_ffn_to_i4
"""
import os
import numpy as np
import pytest

from atlas_infer import AtlasModel, dll
from tests.atlas_mock_model import make, CORRIDORS

HERE = os.path.dirname(os.path.abspath(__file__))
MOCK_DIR = os.path.join(HERE, "..", "mock")
os.makedirs(MOCK_DIR, exist_ok=True)

N_CORES = os.cpu_count() or 8
N_STRESS_RUNS = 50

# Map user-facing corridor names to CORRIDORS keys
KEY_TO_CORRIDOR = {
    "production_int8": "production_int8",
    "pure_ternary":    "ternary_dispatch",
    "int4_ffn_packed": "production_int8",
}
STRESS_KEYS = list(KEY_TO_CORRIDOR.keys())


def _apply_corridor(m, core):
    for func_name, args in core["post_init"]:
        fn = getattr(dll, func_name)
        fn(m.model_ptr, *args)


def _make_model(key):
    corr_key = KEY_TO_CORRIDOR[key]
    path = os.path.join(MOCK_DIR, f"stress-{key}.atlas")
    if not os.path.exists(path):
        make(path, "falcon3", corridor=corr_key)
    return path


@pytest.mark.parametrize("key", STRESS_KEYS)
def test_corridor_omp_stress(key):
    """50 forward passes at N_CORES threads, verify bit-exact with single-thread reference."""
    core = CORRIDORS[KEY_TO_CORRIDOR[key]]
    path = _make_model(key)

    # Single-thread reference
    os.environ["OMP_NUM_THREADS"] = "1"
    m_ref = AtlasModel(path)
    _apply_corridor(m_ref, core)
    m_ref.set_seed(42)
    m_ref.set_num_threads(1)
    m_ref.reset_cache()
    input_ids = np.array([[1, 2, 3]], dtype=np.int32)
    ref = m_ref.forward(input_ids)
    del m_ref

    # Multi-thread stress
    os.environ["OMP_NUM_THREADS"] = str(N_CORES)
    m_stress = AtlasModel(path)
    _apply_corridor(m_stress, core)
    m_stress.set_seed(42)

    failures = []
    for i in range(N_STRESS_RUNS):
        m_stress.reset_cache()
        m_stress.set_num_threads(N_CORES)
        out = m_stress.forward(input_ids)
        try:
            np.testing.assert_array_equal(out, ref)
        except AssertionError as e:
            failures.append((i, str(e)))
            if len(failures) >= 3:
                break

    del m_stress

    if failures:
        pytest.fail(f"{key}: {len(failures)}/{N_STRESS_RUNS} runs mismatched "
                    f"under {N_CORES}-thread stress. "
                    f"First failure run #{failures[0][0]}: {failures[0][1]}")


@pytest.mark.parametrize("key", STRESS_KEYS)
def test_corridor_omp_cache_persistence(key):
    """Sequential forward passes (cache accumulation) must produce identical state under OMP load."""
    core = CORRIDORS[KEY_TO_CORRIDOR[key]]
    path = _make_model(key)

    os.environ["OMP_NUM_THREADS"] = "1"
    m_ref = AtlasModel(path)
    _apply_corridor(m_ref, core)
    m_ref.set_seed(42)
    m_ref.set_num_threads(1)
    m_ref.reset_cache()

    n_pos = 5
    ref_logits = []
    for pos in range(n_pos):
        t = np.array([[pos + 1]], dtype=np.int32)
        logits = m_ref.forward(t, start_pos=pos)
        ref_logits.append(logits.copy())
    del m_ref

    # Multi-thread with dynamic thread count
    os.environ["OMP_NUM_THREADS"] = str(N_CORES)
    m_stress = AtlasModel(path)
    _apply_corridor(m_stress, core)
    m_stress.set_seed(42)
    m_stress.reset_cache()

    for pos in range(n_pos):
        threads = N_CORES if pos % 2 == 0 else 1
        m_stress.set_num_threads(threads)
        t = np.array([[pos + 1]], dtype=np.int32)
        logits = m_stress.forward(t, start_pos=pos)
        try:
            np.testing.assert_array_equal(logits, ref_logits[pos])
        except AssertionError as e:
            del m_stress
            pytest.fail(f"{key}: cache persistence mismatch at position {pos} "
                        f"under dynamic thread count: {e}")

    del m_stress


def test_omp_dynamic_thread_count():
    """Mid-stream thread count changes should not corrupt state or produce non-deterministic output."""
    core = CORRIDORS["production_int8"]
    path = os.path.join(MOCK_DIR, "stress-dynamic-threads.atlas")
    if not os.path.exists(path):
        make(path, "falcon3", corridor="production_int8")

    os.environ["OMP_NUM_THREADS"] = "1"
    m = AtlasModel(path)
    _apply_corridor(m, core)
    m.set_seed(42)

    input_ids = np.array([[1]], dtype=np.int32)
    ref = None
    for threads in [1, 2, 4, 8, 1, 8, 4, 2, 1]:
        m.reset_cache()
        m.set_num_threads(threads)
        out = m.forward(input_ids)
        if ref is None:
            ref = out
        else:
            np.testing.assert_array_equal(
                out, ref,
                err_msg=f"Dynamic thread count {threads}: output mismatch")
    del m
