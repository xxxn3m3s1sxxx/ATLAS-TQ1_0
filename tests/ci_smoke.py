"""Quick CI smoke test — verify DLL exports critical symbols."""
import os, sys, ctypes

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    dll_path = os.path.join(PROJECT_ROOT, "atlas.dll")
elif sys.platform == "linux":
    dll_path = os.path.join(PROJECT_ROOT, "libatlas.so")
elif sys.platform == "darwin":
    dll_path = os.path.join(PROJECT_ROOT, "libatlas.so")
else:
    print(f"Unsupported platform: {sys.platform}")
    sys.exit(1)

if not os.path.exists(dll_path):
    print(f"FAIL: {dll_path} not found")
    sys.exit(1)

dll = ctypes.CDLL(dll_path)

required = ["atlas_load", "atlas_free", "atlas_generate", "atlas_reset_cache",
            "atlas_set_seed", "atlas_set_num_threads", "atlas_get_info"]
missing = [s for s in required if not hasattr(dll, s)]
if missing:
    print(f"FAIL: Missing symbols: {missing}")
    sys.exit(1)

print(f"PASS: {len(required)} symbols OK")
