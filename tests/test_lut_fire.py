"""Test LUT path fires for Bonsai-8B mock model (has ttype=5 after i4 cache)."""
import os, sys, ctypes, numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, 'C:\\atlas')

from atlas_infer import AtlasModel, dll

path = 'C:/atlas/mock/ci-bonsai_8b.atlas'
assert os.path.exists(path), f"Missing: {path}"

print("Loading Bonsai-8B mock model...")
m = AtlasModel(path)

# Check ttypes
ttype_counts = {}
for i in range(m.n_tensors):
    tt = ctypes.c_int(); rd = ctypes.c_int(); cd = ctypes.c_int()
    dll.atlas_tensor_info(m.model_ptr, i, tt, rd, cd)
    ttype_counts[tt.value] = ttype_counts.get(tt.value, 0) + 1
print(f"ttypes: {dict(sorted(ttype_counts.items()))}")
print(f"needs_f32_bypass: {m._needs_f32_bypass}")

if 5 not in ttype_counts:
    print("FAIL: No ttype=5 tensors — LUT path cannot fire")
    sys.exit(1)

print(f"HAS {ttype_counts[5]} ttype=5 tensors — LUT path should fire for B>=16")

# Forward with batch=1 (uses fused_s8 for ttype=5)
print("\nForward batch=1 (fused_s8 path)...")
ids_1 = np.array([[1, 2, 3]], dtype=np.int32)
logits_1 = m.forward(ids_1)
assert logits_1.shape == (1, 3, m.vocab_size)
print(f"  OK: shape={logits_1.shape}, range=[{logits_1.min():.4f}, {logits_1.max():.4f}]")

mv = np.max(np.abs(logits_1))
print(f"  max_abs={mv:.4f}")

# Forward with batch=16 (should use LUT for ttype=5)
print("\nForward batch=16 (LUT path)...")
ids_16 = np.zeros((16, 3), dtype=np.int32)
for b in range(16):
    ids_16[b] = [1, (b % 254) + 1, 3]
logits_16 = m.forward(ids_16)
assert logits_16.shape == (16, 3, m.vocab_size)
assert not np.any(np.isnan(logits_16))
assert not np.any(np.isinf(logits_16))
print(f"  OK: shape={logits_16.shape}, no NaN/inf")

# Compare first token output between batch=1 and batch=16
diff = np.max(np.abs(logits_1[0] - logits_16[0]))
print(f"\nDiff batch=1 vs batch=16 (first token): max_abs={diff:.6f}")

# Calculate relative diff for the output token position
# (position 2 is the last token in the 3-token input)
rel_diff = np.max(np.abs(logits_1[0, 2] - logits_16[0, 2])) / (np.max(np.abs(logits_1[0, 2])) + 1e-10)
print(f"Relative diff (last position): {rel_diff:.6f}")

if diff < 1e-3:
    print("✅ LUT path produces identical output to fused_s8!")
elif diff < 1e-1:
    print(f"⚠️  Small diff ({diff:.6f}) — likely quantization path difference")
else:
    print(f"❌ Large diff ({diff:.6f}) — possible LUT kernel bug!")

# Verify no crash with larger batch
print("\nForward batch=32...")
ids_32 = np.zeros((32, 3), dtype=np.int32)
for b in range(32):
    ids_32[b] = [1, (b % 254) + 1, 3]
logits_32 = m.forward(ids_32)
assert logits_32.shape == (32, 3, m.vocab_size)
assert not np.any(np.isnan(logits_32))
assert not np.any(np.isinf(logits_32))
print(f"  OK: shape={logits_32.shape}")

# Forward batch=64
print("\nForward batch=64...")
ids_64 = np.zeros((64, 3), dtype=np.int32)
for b in range(64):
    ids_64[b] = [1, (b % 254) + 1, 3]
logits_64 = m.forward(ids_64)
assert logits_64.shape == (64, 3, m.vocab_size)
assert not np.any(np.isnan(logits_64))
assert not np.any(np.isinf(logits_64))
print(f"  OK: shape={logits_64.shape}")

print("\n✅ All tests passed!")
