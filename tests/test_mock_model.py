#!/usr/bin/env python3
"""CI smoke test: load a mock .atlas model and run atlas_generate."""
import ctypes, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if sys.platform == "win32":
    LIB = os.path.join(HERE, "..", "atlas.dll")
else:
    LIB = os.path.join(HERE, "..", "libatlas.so")
MOCK = os.path.join(HERE, "mock.atlas")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def main():
    if not os.path.exists(LIB):
        print(f"[SKIP] Library not found at {LIB}")
        return 0
    if not os.path.exists(MOCK):
        print(f"[SKIP] mock.atlas not found at {MOCK}")
        return 0

    dll = ctypes.CDLL(LIB)

    # Bind C API
    dll.atlas_load.restype = ctypes.c_void_p
    dll.atlas_load.argtypes = [ctypes.c_char_p]
    dll.atlas_free.argtypes = [ctypes.c_void_p]
    dll.atlas_set_seed.argtypes = [ctypes.c_uint64]
    dll.atlas_decompress_all.restype = None
    dll.atlas_decompress_all.argtypes = [ctypes.c_void_p]
    dll.atlas_decompress_ttype5.restype = None
    dll.atlas_decompress_ttype5.argtypes = [ctypes.c_void_p]
    dll.atlas_set_base_seq_len.argtypes = [ctypes.c_void_p, ctypes.c_int]
    dll.atlas_set_use_f32_matmul.argtypes = [ctypes.c_void_p, ctypes.c_int]
    dll.atlas_set_num_threads.argtypes = [ctypes.c_int]
    dll.atlas_quantize_lmhead.restype = None
    dll.atlas_quantize_lmhead.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    dll.atlas_prefetch_int8.restype = None
    dll.atlas_prefetch_int8.argtypes = [ctypes.c_void_p]
    dll.atlas_generate.restype = ctypes.c_int
    dll.atlas_generate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_int, ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]

    print("[Test] Loading mock model...")
    t0 = time.time()
    model = dll.atlas_load(MOCK.encode())
    if not model:
        print("[FAIL] atlas_load returned NULL")
        return 1
    print(f"[Test] Load OK ({time.time() - t0:.2f}s)")

    # Unpack ttype=5 (embed_tokens, norm → int8)
    dll.atlas_decompress_all(model)
    dll.atlas_decompress_ttype5(model)
    dll.atlas_set_use_f32_matmul(model, 1)
    dll.atlas_set_base_seq_len(model, 512)
    dll.atlas_set_num_threads(2)
    dll.atlas_set_seed(42)

    # Quantize lm_head (index 11 = last tensor)
    try:
        dll.atlas_quantize_lmhead(model, 11, 0)
    except Exception as e:
        print(f"[WARN] lm_head quantize failed (non-fatal): {e}")

    dll.atlas_prefetch_int8(model)

    # Generate with 1 input token
    input_ids = (ctypes.c_int * 1)(1)
    output = (ctypes.c_int * 50)()
    max_seq_len = 512
    max_new = 10

    print("[Test] Running atlas_generate (1 input, T=0, top_k=1)...")
    t0 = time.time()
    n_gen = dll.atlas_generate(
        model, input_ids, 1,
        max_seq_len, max_new,
        0.0, 1, 1.0,  # T=0, top_k=1, top_p=1
        1.0,           # repetition_penalty
        0,             # min_new_tokens
        0,             # cache_offset
        output,
    )
    elapsed = time.time() - t0
    dll.atlas_free(model)

    if n_gen < 0:
        print(f"[FAIL] atlas_generate returned {n_gen}")
        return 1

    out_tokens = list(output[:n_gen])
    print(f"[Test] Generated {n_gen} tokens in {elapsed:.2f}s ({n_gen/elapsed:.1f} tok/s): {out_tokens}")

    if n_gen == 0:
        print("[FAIL] atlas_generate returned 0 tokens")
        return 1

    print("[PASS] Mock model smoke test passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
