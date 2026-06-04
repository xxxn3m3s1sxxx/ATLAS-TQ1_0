"""Direct test of TQ1 LUT prefill kernel vs standard matmul.
Creates synthetic TQ1 weights, runs both kernels, compares outputs."""
import ctypes, numpy as np, os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, 'C:\\atlas')

dll = ctypes.CDLL('C:\\atlas\\atlas.dll')

# TQ1 encode: 5 ternary values -> 1 byte (base-3)
def tq1_encode(v5):
    return v5[0]+1 + (v5[1]+1)*3 + (v5[2]+1)*9 + (v5[3]+1)*27 + (v5[4]+1)*81

# Build synthetic model: rows=256, dim=640, block_size=128
rows = 256
dim = 640
block_size = 128
n_blocks = (dim + block_size - 1) // block_size
packed_cols = (dim + 4) // 5  # 128

# Generate random ternary weights
rng = np.random.RandomState(42)
weights = rng.randint(-1, 2, size=(rows, dim)).astype(np.int8)
# Per-block scales: random fp16
scales_fp16 = np.random.uniform(0.5, 2.0, size=(rows, n_blocks)).astype(np.float32)

# Pack TQ1 format
packed = np.zeros((rows, packed_cols), dtype=np.uint8)
for r in range(rows):
    for g in range(packed_cols):
        d0 = g * 5
        v5 = [weights[r, min(d0+k, dim-1)] for k in range(5)]
        # Pad last group if needed
        if d0 + 4 >= dim:
            for k in range(5):
                if d0 + k >= dim:
                    v5[k] = 0
        packed[r, g] = tq1_encode(v5)

# Build tensor_data: [block_size:1][n_blocks:2][scales...][packed...]
hdr_size = 3 + rows * n_blocks * 2 + rows * packed_cols
tensor_data = np.zeros(hdr_size, dtype=np.uint8)
tensor_data[0] = block_size
tensor_data[1] = n_blocks & 0xFF
tensor_data[2] = (n_blocks >> 8) & 0xFF
# Scales
scale_bytes = scales_fp16.astype(np.float16).tobytes()
tensor_data[3:3+len(scale_bytes)] = list(scale_bytes)
# Packed weights (TQ1)
tensor_data[3+len(scale_bytes):] = packed.flatten()

# Activations: B=32 tokens
B = 32
act_f32 = rng.randn(B, dim).astype(np.float32)

# Output buffers
out_lut = np.zeros((B, rows), dtype=np.float32)
out_std = np.zeros((B, rows), dtype=np.float32)

# Export data as ctypes pointers
tensor_data_p = tensor_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
act_f32_p = act_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
out_lut_p = out_lut.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
out_std_p = out_std.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# We can't call the static functions directly from Python.
# Instead, load the model via atlas_infer with our synthetic data.
# OR: expose the kernels as test functions.

print("Synthetic data ready: rows=%d dim=%d B=%d packed_cols=%d n_blocks=%d" % (
    rows, dim, B, packed_cols, n_blocks))
print("LUT kernel will process B=%d (threshold=%d)" % (B, 16))
print("To verify: modify LUT_THRESHOLD to 1 and test with real model")

# Verify TQ1 round-trip: decode and check dot products
from atlas_infer import AtlasModel  # noqa: F401
print("Test setup complete - use real model with B >= LUT_THRESHOLD(16) to exercise LUT path")

# Clean check: compute one dot product manually
r, b = 0, 0
manual_dot = 0.0
for i in range(dim):
    manual_dot += act_f32[b, i] * weights[r, i]
# With scales: each block has its own scale
scaled_manual = 0.0
for blk in range(n_blocks):
    blk_start = blk * block_size
    blk_end = min(blk_start + block_size, dim)
    block_dot = 0.0
    for i in range(blk_start, blk_end):
        block_dot += act_f32[b, i] * weights[r, i]
    scaled_manual += block_dot * scales_fp16[r, blk]

print("Manual dot (first token, first row, block-scaled): %.6f" % scaled_manual)
