"""Debug test: confirm LUT path fires with debug DLL."""
import os, sys, ctypes, numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, 'C:\\atlas')

from atlas_infer import AtlasModel, dll

path = 'C:/atlas/mock/ci-bonsai_8b.atlas'
print(f"Loading {path}...")
m = AtlasModel(path)

# Check ttypes
ttype_counts = {}
for i in range(m.n_tensors):
    tt = ctypes.c_int(); rd = ctypes.c_int(); cd = ctypes.c_int()
    dll.atlas_tensor_info(m.model_ptr, i, tt, rd, cd)
    ttype_counts[tt.value] = ttype_counts.get(tt.value, 0) + 1

print(f"ttype=5 count: {ttype_counts.get(5, 0)}")
print(f"needs_f32_bypass: {m._needs_f32_bypass}")

# Test batch=1 (below threshold — fused_s8)
print("\n--- Batch=1 (fused_s8 path) ---")
ids_1 = np.array([[1, 2, 3]], dtype=np.int32)
out_1 = m.forward(ids_1)
print(f"OK: shape={out_1.shape}, range=[{out_1.min():.4f}, {out_1.max():.4f}]")

# Test batch=8 (below threshold — fused_s8)
print("\n--- Batch=8 (fused_s8, B < LUT_THRESHOLD=16) ---")
ids_8 = np.zeros((8, 3), dtype=np.int32)
for b in range(8):
    ids_8[b] = [1, (b % 254) + 1, 3]
out_8 = m.forward(ids_8)
print(f"OK: shape={out_8.shape}, no NaN/inf")

# Test batch=16 (at threshold — LUT should fire if ttype=5)
print("\n--- Batch=16 (LUT path should fire for ttype=5) ---")
ids_16 = np.zeros((16, 3), dtype=np.int32)
for b in range(16):
    ids_16[b] = [1, (b % 254) + 1, 3]
out_16 = m.forward(ids_16)
print(f"OK: shape={out_16.shape}, no NaN/inf")

print("\nDone. Check output above for [LUT] markers in stderr.")
