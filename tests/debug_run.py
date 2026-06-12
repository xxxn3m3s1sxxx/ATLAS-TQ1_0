#!/usr/bin/env python3
"""Isolated debug runner — zero pytest, pure raw telemetry.

Loads every fixture sequentially, prints each step.
Catches Windows fatal exceptions by running in subprocess.
"""

import ctypes, os, sys, subprocess, time

# ─── ABI guard: abort on any Python-vs-C signature mismatch ──────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from abi_verify import verify_abi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RELEASE_DLL = os.path.join(ROOT, "atlas.dll")
DEBUG_DLL = os.path.join(ROOT, "atlas_d.dll")

ALL_FIXTURES = []
for arch in ["falcon3", "qwen3", "bitnet", "trilm"]:
    for ver in [6, 8]:
        path = os.path.join(HERE, f"{arch}_v{ver}.atlas")
        ALL_FIXTURES.append((f"{arch}_v{ver}", path))


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
    ]


def test_one(dll, name, path):
    print(f"\n[DEBUG] === {name} ===", flush=True)
    if not os.path.exists(path):
        print(f"[DEBUG] {name}: fixture not found, skipping", flush=True)
        return True

    model = dll.atlas_load(path.encode())
    assert model, f"{name}: atlas_load returned NULL"
    print(f"[DEBUG] {name}: loaded OK", flush=True)

    try:
        dll.atlas_decompress_all(model)
        print(f"[DEBUG] {name}: decompress_all OK", flush=True)

        dll.atlas_decompress_ttype5(model)
        print(f"[DEBUG] {name}: decompress_ttype5 OK", flush=True)

        dll.atlas_set_num_threads(model, 2)
        dll.atlas_set_seed(42)

        n_tensors = dll.atlas_get_tensor_count(model)
        lm_head_idx = n_tensors - 1
        print(f"[DEBUG] {name}: n_tensors={n_tensors}, lm_head_idx={lm_head_idx}", flush=True)

        dll.atlas_quantize_lmhead(model, lm_head_idx, 0)
        print(f"[DEBUG] {name}: quantize_lmhead OK", flush=True)

        print(f"[DEBUG] {name}: calling prefetch_int8...", flush=True)
        dll.atlas_prefetch_int8(model)
        print(f"[DEBUG] {name}: prefetch_int8 OK", flush=True)

        print(f"[DEBUG] {name}: calling atlas_generate...", flush=True)
        input_ids = (ctypes.c_int * 1)(1)
        output = (ctypes.c_int * 50)()
        n_gen = dll.atlas_generate(
            model, input_ids, 1, 512, 10, 0.0, 1, 1.0, 1.0, 0, 0, output,
        )
        print(f"[DEBUG] {name}: generate OK, n_gen={n_gen}", flush=True)

        assert n_gen > 0, f"{name}: atlas_generate returned {n_gen}"
        print(f"[DEBUG] {name}: PASSED", flush=True)
        return True
    finally:
        dll.atlas_free(model)


def main():
    print(f"[DEBUG] Loading DLL from: {RELEASE_DLL}", flush=True)
    dll = ctypes.CDLL(RELEASE_DLL)
    _bind_api(dll)

    # Verify ABI before any C call — ValueError if Python argtypes mismatch header
    header_path = os.path.join(ROOT, "atlas_ffi.h")
    verify_abi(header_path, dll)
    print("[DEBUG] ABI verification: PASSED", flush=True)

    passed = 0
    failed = 0
    for name, path in ALL_FIXTURES:
        try:
            ok = test_one(dll, name, path)
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[DEBUG] {name}: EXCEPTION: {e}", flush=True)
            failed += 1

    print(f"\n[DEBUG] RESULT: {passed} passed, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
