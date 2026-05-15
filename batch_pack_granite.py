"""Batch MSE-packer: Full Granite 2B -> TQ1_0 + FP32 norm extraction.
Output: C:\dam\atlas\granite_tq10\ layer_{N}_{name}.tq10 / .bin
"""
import os, sys, json, struct, time
import numpy as np
import torch
from safetensors.torch import load_file

MODEL_DIR = r"C:\dam\models\granite-3.0-2b-instruct"
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")
OUT_DIR = r"C:\dam\atlas\granite_tq10"
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from mse_packer import optimize_fast, write_tq10, mse_loss, optimize_alpha

with open(INDEX_PATH) as f:
    index = json.load(f)
weight_map = index["weight_map"]

shards_loaded = {}
def get_weight(key):
    shard = weight_map[key]
    if shard not in shards_loaded:
        path = os.path.join(MODEL_DIR, shard)
        sz = os.path.getsize(path) / (1024*1024*1024)
        print(f"  Loading shard {shard} ({sz:.1f} GB)...")
        t0 = time.time()
        shards_loaded[shard] = load_file(path, device="cpu")
        print(f"    done in {time.time()-t0:.1f}s")
    arr = shards_loaded[shard][key]
    return arr.to(torch.float32).numpy()

total_start = time.time()

# --- Global weights: embed + final_norm + lm_head ---
print("\n=== Global weights ===")

global_keys = [
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
]
for key in global_keys:
    if key not in weight_map:
        print(f"  {key}: not found, skipping")
        continue
    print(f"  Loading {key}...")
    W = get_weight(key)
    if W.ndim == 2:
        rows, cols = W.shape
        # Embed + lm_head: store as FP32 .bin (not ternarized)
        out_name = key.split(".")[-2] if key.startswith("model.") else key.split(".")[0]
        if key == "model.embed_tokens.weight":
            out_name = "embed"
        elif key == "lm_head.weight":
            out_name = "lm_head"
        out_path = os.path.join(OUT_DIR, f"{out_name}.bin")
        with open(out_path, 'wb') as f:
            f.write(struct.pack('<i', rows))
            f.write(struct.pack('<i', cols))
            f.write(W.astype(np.float32).tobytes())
        print(f"  -> {out_name}.bin ({rows}x{cols}, {os.path.getsize(out_path)/1024/1024:.1f} MB)")
    elif W.ndim == 1:
        n = W.shape[0]
        out_name = "final_norm" if "norm" in key else key.split(".")[0]
        out_path = os.path.join(OUT_DIR, f"{out_name}.bin")
        with open(out_path, 'wb') as f:
            f.write(struct.pack('<i', 1))
            f.write(struct.pack('<i', n))
            f.write(W.astype(np.float32).tobytes())
        print(f"  -> {out_name}.bin (1D, n={n})")

# --- Per-layer weights ---
N_LAYERS = 40
layer_keys_map = {}
for key in weight_map:
    parts = key.split(".")
    if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
        layer_num = int(parts[2])
        if layer_num not in layer_keys_map:
            layer_keys_map[layer_num] = []
        layer_keys_map[layer_num].append(key)

for layer_num in sorted(layer_keys_map.keys()):
    keys = sorted(layer_keys_map[layer_num])
    print(f"\n=== Layer {layer_num}/{N_LAYERS-1} ===")

    for key in keys:
        name_tag = key.replace("model.layers.", "l")
        out_name = name_tag.split(".weight")[0]
        out_path_tq10 = os.path.join(OUT_DIR, f"{out_name}.tq10")
        out_path_bin = os.path.join(OUT_DIR, f"{out_name}.bin")

        # Skip if already processed
        if os.path.exists(out_path_tq10) or os.path.exists(out_path_bin):
            continue

        W = get_weight(key)
        short = ".".join(key.split(".")[3:-1])  # e.g. "self_attn.q_proj"

        if W.ndim != 2:
            # 1D norm vector
            n = W.shape[0]
            with open(out_path_bin, 'wb') as f:
                f.write(struct.pack('<i', 1))
                f.write(struct.pack('<i', n))
                f.write(W.astype(np.float32).tobytes())
            print(f"  {short}: norm ({n}) -> {out_name}.bin")
        else:
            rows, cols = W.shape
            t0 = time.time()

            # Use sample capped at 2048 rows for speed
            sample = min(2048, rows)
            t_tern, alpha, mse = optimize_fast(W, sample_rows=sample)

            # Reference for comparison
            t_s, a_s = optimize_alpha(W)
            mse_s = mse_loss(W, t_s, a_s)
            imp = 100 * (1 - mse / mse_s)
            zeros = np.sum(t_tern == 0)
            n_total = t_tern.size

            write_tq10(out_path_tq10, t_tern, alpha)
            elapsed = time.time() - t0
            print(f"  {short}: ({rows}x{cols}) alpha={alpha:.6f} MSE={mse:.8f} imp={imp:.1f}% zeros={100*zeros/n_total:.0f}% [{elapsed:.1f}s]")

total_elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

# Cleanup
del shards_loaded
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Verify output
n_tq10 = len([f for f in os.listdir(OUT_DIR) if f.endswith('.tq10')])
n_bin = len([f for f in os.listdir(OUT_DIR) if f.endswith('.bin')])
total_mb = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR)) / (1024*1024)
print(f"Output: {n_tq10} TQ1_0 files + {n_bin} FP32 files = {total_mb:.0f} MB in {OUT_DIR}")
print("Done.")
