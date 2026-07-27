#!/usr/bin/env python3
"""CI matrix test: load + generate + golden regression for all fixtures.

Tests 6 scenarios: {falcon3,qwen3,bitnet} × {v6,v8}.
Each fixture is a minimal 1-layer .atlas (~25 KB).

Golden mode:
  GENERATE_GOLDEN=1  — overwrite golden_registry.json with current outputs
  GENERATE_GOLDEN=0  — compare generated tokens against frozen golden data
"""

import ctypes, json, os, sys, pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, HERE)
from abi_verify import verify_abi  # noqa: E402

GENERATE_GOLDEN = os.environ.get("GENERATE_GOLDEN", "").strip() in ("1", "true", "yes")
GOLDEN_PATH = os.path.join(HERE, "golden_registry.json")

# Deterministic generation parameters for golden reference
PROMPT = [1]
SEED = 42
MAX_NEW = 10
TEMPERATURE = 0.0  # greedy
TOP_K = 1  # argmax

if sys.platform == "win32":
    LIB = os.path.join(ROOT, "atlas.dll")
elif sys.platform == "linux":
    LIB = os.path.join(ROOT, "libatlas.so")
else:
    LIB = os.path.join(ROOT, "libatlas.so")

ALL_FIXTURES = []
for arch in ["falcon3", "qwen3", "bitnet", "trilm"]:
    for ver in [6, 8]:
        path = os.path.join(HERE, f"{arch}_v{ver}.atlas")
        ALL_FIXTURES.append((f"{arch}_v{ver}", path))

pytestmark = [
    pytest.mark.skipif(not os.path.exists(LIB), reason="Library not found"),
]


def _load_golden():
    if not os.path.exists(GOLDEN_PATH):
        return {}
    try:
        with open(GOLDEN_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _save_golden(golden):
    with open(GOLDEN_PATH, "w") as f:
        json.dump(golden, f, indent=2)
        f.write("\n")


def _bind_api(dll):
    dll.atlas_load.restype = ctypes.c_void_p
    dll.atlas_load.argtypes = [ctypes.c_char_p]
    dll.atlas_free.argtypes = [ctypes.c_void_p]
    dll.atlas_decompress_all.restype = None
    dll.atlas_decompress_all.argtypes = [ctypes.c_void_p]
    dll.atlas_decompress_ttype5.restype = None
    dll.atlas_decompress_ttype5.argtypes = [ctypes.c_void_p]
    dll.atlas_set_num_threads.argtypes = [ctypes.c_void_p, ctypes.c_int]
    dll.atlas_set_seed.argtypes = [ctypes.c_uint64]
    dll.atlas_quantize_lmhead.restype = None
    dll.atlas_quantize_lmhead.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    dll.atlas_prefetch_int8.restype = None
    dll.atlas_prefetch_int8.argtypes = [ctypes.c_void_p]
    dll.atlas_get_tensor_count.restype = ctypes.c_int
    dll.atlas_get_tensor_count.argtypes = [ctypes.c_void_p]
    dll.atlas_generate.restype = ctypes.c_int
    dll.atlas_generate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p, ctypes.c_void_p,  # logit_cb, logit_cb_data
        ctypes.c_void_p, ctypes.c_void_p,  # token_notify_cb, token_notify_data
    ]
    # ABI guard: abort on Python-vs-C signature mismatch
    header_path = os.path.join(ROOT, "atlas_ffi.h")
    verify_abi(header_path, dll)


@pytest.fixture(scope="session")
def dll():
    return ctypes.CDLL(LIB)


def _run_fixture(dll, name, path):
    """Load fixture, quantize, generate. Returns (n_gen, output[:n_gen] list)."""
    model = dll.atlas_load(path.encode())
    assert model, f"{name}: atlas_load returned NULL"
    try:
        dll.atlas_decompress_all(model)
        dll.atlas_decompress_ttype5(model)
        dll.atlas_set_num_threads(model, 2)
        dll.atlas_set_seed(SEED)

        n_tensors = dll.atlas_get_tensor_count(model)
        dll.atlas_quantize_lmhead(model, n_tensors - 1, 0)
        dll.atlas_prefetch_int8(model)

        input_ids = (ctypes.c_int * len(PROMPT))(*PROMPT)
        output = (ctypes.c_int * MAX_NEW)()
        n_gen = dll.atlas_generate(
            model, input_ids, len(PROMPT), 512, MAX_NEW,
            TEMPERATURE, TOP_K, 1.0, 1.0, 0, 0, output,
            None, None, None, None,
        )
        assert n_gen > 0, f"{name}: atlas_generate returned {n_gen}"
        return n_gen, [output[i] for i in range(n_gen)]
    finally:
        dll.atlas_free(model)


# ─── Test: load + generate (survival check) ────────────────────────────────
@pytest.mark.parametrize("name,path", ALL_FIXTURES)
def test_fixture_load_and_generate(dll, name, path):
    if not os.path.exists(path):
        pytest.skip(f"{name}.atlas not found — run generate_test_fixtures.py first")
    _bind_api(dll)
    n_gen, tokens = _run_fixture(dll, name, path)
    assert n_gen > 0


# ─── Test: golden regression (mathematical integrity) ──────────────────────
@pytest.mark.parametrize("name,path", ALL_FIXTURES)
def test_golden_regression(dll, name, path):
    if not os.path.exists(path):
        pytest.skip(f"{name}.atlas not found — run generate_test_fixtures.py first")
    _bind_api(dll)
    n_gen, tokens = _run_fixture(dll, name, path)

    golden = _load_golden()
    entry = golden.get(name)

    if GENERATE_GOLDEN:
        golden[name] = {
            "prompt": PROMPT,
            "seed": SEED,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "max_new": MAX_NEW,
            "n_gen": n_gen,
            "tokens": tokens,
        }
        _save_golden(golden)
        pytest.skip(f"Golden data generated for {name}")
        return

    if entry is None:
        pytest.skip(f"No golden data for {name} — set GENERATE_GOLDEN=1 first")
        return

    # ── Token ID match (hard gate) ──
    expected = entry["tokens"]
    assert len(tokens) == len(expected), (
        f"{name}: token count mismatch — got {len(tokens)}, expected {len(expected)}"
    )
    for i, (got_tok, exp_tok) in enumerate(zip(tokens, expected)):
        assert got_tok == exp_tok, (
            f"{name}: token {i} mismatch — got {got_tok}, expected {exp_tok}"
        )
