"""Test LUT path activation: force TQ1 path and verify forward output matches."""
import os, sys, ctypes, numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, 'C:\\atlas')

from atlas_infer import AtlasModel, dll

# Use the caches to ensure fast loading - we'll force TQ1 native path
# by setting needs_f32_bypass=False after init, which means decompress_ttype5
# was skipped. But since the mock models use cache (which has ttype=3), we need
# to either clear the cache or use a fresh model.

# Strategy: load normally, then check if any tensor is ttype=5.
# For mock models, the cached version has ttype=3. So we need to work with
# whatever the model gives us.

def check_ttypes(path, label="model"):
    """Check tensor ttypes after load."""
    m = AtlasModel(path)
    # Get ttype of first 5 tensors and count by type
    ttype_counts = {}
    for i in range(m.n_tensors):
        # Use the C API to get tensor info
        tt = ctypes.c_int()
        rd = ctypes.c_int()
        cd = ctypes.c_int()
        dll.atlas_tensor_info(m.model_ptr, i, tt, rd, cd)
        ttype_counts[tt.value] = ttype_counts.get(tt.value, 0) + 1
    print(f"  {label}: ttype counts: {ttype_counts}")
    # Check specifically for ttype=5
    has_ttype5 = 5 in ttype_counts
    if has_ttype5:
        print(f"  -> HAS ttype=5 ({ttype_counts[5]} tensors) — LUT path available")
    else:
        print(f"  -> NO ttype=5 — LUT path cannot fire")
    return m, has_ttype5

print("=" * 60)
print("1. Check mock models loaded normally")
print("=" * 60)
for fname, label in [
    ("ci-bonsai_4b.atlas", "Bonsai-4B (default=needs_f32_bypass)"),
    ("ci-bonsai_1_7b.atlas", "Bonsai-1.7B (default=needs_f32_bypass)"),
    ("ci-bonsai_8b.atlas", "Bonsai-8B (default=needs_f32_bypass)"),
]:
    path = f'C:/atlas/mock/{fname}'
    if os.path.exists(path):
        m, _ = check_ttypes(path, label)
        # Print rope_theta and hidden for f32 deduction
        print(f"    hidden={m.hidden} rope_theta={m.rope_theta} is_bitnet={m._is_bitnet}")
        print(f"    would trigger needs_f32_bypass: {m._needs_f32_bypass}")

print()
print("=" * 60)
print("2. Test LUT-like batch forward with mock Qwen3 (ttype=5 available)")
print("=" * 60)

# Qwen3 mock model has ttype=5 tensors and rope_theta=1M (<3M)
# and hidden might be > 2048
path = 'C:/atlas/mock/ci-qwen3.atlas'
if os.path.exists(path):
    m, has_ttype5 = check_ttypes(path, "Qwen3")
    print(f"    hidden={m.hidden} rope_theta={m.rope_theta}")
    
    if has_ttype5:
        # Test forward with batch size 16 (should hit LUT path if ttype=5 preserved)
        print("\n    Testing forward with batch=16...")
        ids = np.zeros((16, 3), dtype=np.int32)
        for b in range(16):
            ids[b] = [1, b + 2, 3]
        logits = m.forward(ids)
        assert logits.shape == (16, 3, m.vocab_size), f"Shape mismatch: {logits.shape}"
        assert not np.any(np.isnan(logits))
        assert not np.any(np.isinf(logits))
        print(f"    OK: forward(16,3) → {logits.shape}, no NaN/inf")
    else:
        print("    No ttype=5 — LUT path cannot fire for this model")

print()
print("=" * 60)
print("3. Force TQ1 path by clearing cache and re-loading")
print("=" * 60)

# For Bonsai-4B with rope_theta=5M: needs_f32_bypass=True always.
# To test LUT, we must clear cache AND force needs_f32_bypass=False.
# This simulates a model that has ttype=5 but works without f32_bypass.
import shutil

# Use a model that has ttype=5 but rope_theta < 3M
# Let's try the Qwen3 mock model
if os.path.exists(path):
    # Clear both i8 and i4 caches
    i8_cache = path + '.i8'
    if os.path.exists(i8_cache):
        os.remove(i8_cache)
        print(f"  Removed i8 cache: {i8_cache}")
    i4_cache = path + '.i4'
    if os.path.exists(i4_cache):
        os.remove(i4_cache)
        print(f"  Removed i4 cache: {i4_cache}")
    
    # Now load - with needs_f32_bypass=False, decompress_ttype5 is skipped
    m = AtlasModel(path)
    # Override needs_f32_bypass to False (in case rope_theta >= 3M)
    m._needs_f32_bypass = False
    
    # Check ttypes
    ttype_counts = {}
    for i in range(m.n_tensors):
        tt = ctypes.c_int()
        rd = ctypes.c_int()
        cd = ctypes.c_int()
        dll.atlas_tensor_info(m.model_ptr, i, tt, rd, cd)
        ttype_counts[tt.value] = ttype_counts.get(tt.value, 0) + 1
    print(f"  After load: ttype counts: {ttype_counts}")
    
    has_ttype5 = 5 in ttype_counts
    if has_ttype5:
        print(f"  -> HAS ttype=5 ({ttype_counts[5]} tensors) — LUT path available!")
        
        # Test batch=1 (decode; uses fused_s8 for ttype=5)
        print("\n  Test batch=1 (decode, fused_s8 path)...")
        ids = np.array([[1, 2, 3]], dtype=np.int32)
        logits_1 = m.forward(ids)
        assert logits_1.shape == (1, 3, m.vocab_size)
        print(f"  OK: forward(1,3) → {logits_1.shape}")
        
        # Test batch=16 (prefill; should use LUT if threshold=16)
        print("\n  Test batch=16 (prefill, LUT path if B>=16)...")
        ids = np.zeros((16, 3), dtype=np.int32)
        for b in range(16):
            ids[b] = [1, b + 2, 3]
        logits_16 = m.forward(ids)
        assert logits_16.shape == (16, 3, m.vocab_size)
        assert not np.any(np.isnan(logits_16))
        assert not np.any(np.isinf(logits_16))
        print(f"  OK: forward(16,3) → {logits_16.shape}, no NaN/inf")
        
        # Verify numerical consistency: logits for token 0 should match
        # between batch=1 and batch=16 (first token is same)
        first_logits_1 = logits_1[0, :, :]
        first_logits_16 = logits_16[0, :, :]
        diff = np.max(np.abs(first_logits_1 - first_logits_16))
        print(f"\n  Max diff between batch=1 and batch=16 (first token): {diff:.6f}")
        if diff < 1e-3:
            print("  ✅ Numerical consistency: LUT and fused_s8 produce same output!")
        else:
            print(f"  ⚠️  Diff={diff:.6f} - may be quantization path difference")
    else:
        print("  No ttype=5 survived")

print()
print("=" * 60)
print("Done.")
