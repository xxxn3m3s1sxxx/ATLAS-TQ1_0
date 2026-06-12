"""E2E test: compare LUT path (B >= 16) vs standard path (B=1) for same tokens."""
import os, sys, numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, 'C:\\atlas')

from atlas_infer import AtlasModel

# Load mock model with TQ1 format (needs cache clear)
import shutil
cache_dir = 'C:/atlas/tests/.i8_cache'
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# Also try to find a model that retains ttype=5
# The mock model might immediately convert to int8. Let's check.
m = AtlasModel('C:\\atlas\\tests\\mock-falcon3.atlas')

# Check tensor types in loaded model
print("\nTensor types in loaded model:")
n = m.n_tensors
ttypes_seen = set()
for i in range(n):
    ti = m.tensors[i]
    ttypes_seen.add(ti.ttype)
    if i < 5 or 'down' in m.get_tensor_name(i).lower():
        print(f"  [{i}] {m.get_tensor_name(i)} ttype={ti.ttype} row={ti.row_dim} pc={ti.packed_cols}")
print(f"Unique ttypes: {sorted(ttypes_seen)}")

# If all tensors are ttype=3 (int8 decompressed), the LUT dispatch won't trigger.
# We need a model that preserves ttype=5.
# Try loading WITHOUT int8 cache
print("\nChecking if model preserves ttype=5 tensor format...")
for i in range(n):
    ti = m.tensors[i]
    if ti.ttype == 5:
        print(f"  FOUND ttype=5: [{i}] {m.get_tensor_name(i)}")
        break
else:
    print("  No ttype=5 tensor found - model was decompressed to int8")
    print("  LUT kernel only works with raw TQ1 format (ttype=5)")
    print("  To test: need a model loaded WITHOUT cache/auto-decompress")

# Check the initial tensor header bytes from the raw file
with open('C:/atlas/tests/mock-falcon3.atlas', 'rb') as f:
    raw = f.read(64)

# Parse v5/v6 header: find directory entries
n_tensors = int.from_bytes(raw[6:8], 'little')
print(f"\nRaw file: n_tensors={n_tensors}")
print(f"Header hex: {raw[:32].hex()}")

# Read directory
dir_data = raw[8:8+min(n_tensors*12, 56)]  # first few entries
print(f"Dir entries (first 4):")
for i in range(min(4, n_tensors)):
    off = 8 + i * 12
    if off + 12 <= len(raw):
        entry = raw[off:off+12]
        data_off = int.from_bytes(entry[0:4], 'little')
        data_sz = int.from_bytes(entry[4:8], 'little')
        name_len = entry[8]
        ttype = entry[9]
        print(f"  [{i}] off={data_off} sz={data_sz} name_len={name_len} ttype={ttype}")
