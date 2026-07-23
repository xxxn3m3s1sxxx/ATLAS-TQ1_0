"""Load and test DeepSeek-V2-Lite ATLAS model."""
import os, sys, numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.pop("ATLAS_DLL", None)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from atlas_infer import AtlasModel

path = r"D:\models\deepseek-v2-lite-atlas-i8.atlas"
print(f"Loading {os.path.getsize(path) / 1024**3:.2f} GB model...")
m = AtlasModel(path, use_f32_matmul=True)
print(f"n_layers={m.n_layers} hidden={m.hidden} vocab={m.vocab_size}")

# Minimal test: feed 4 token IDs
ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
print(f"Input shape: {ids.shape}")
logits = m.forward(ids)
print(f"Output shape: {logits.shape}")
print(f"Logits[0,:5]: {logits[0, 0, :5]}")
print("SUCCESS - model loads and runs forward pass")
