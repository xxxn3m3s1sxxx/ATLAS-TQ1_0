#!/usr/bin/env python3
"""Download Qwen3-1.7B-ternary + original model files."""
from huggingface_hub import snapshot_download, hf_hub_download
import threading, time

results = {}
errors = {}

def dl_tritplane():
    try:
        print("[TRIT] Starting TritPlane download (~1 GB)...")
        d = snapshot_download("AsadIsmail/Qwen3-1.7B-ternary",
            allow_patterns=["tritplane/*", "metadata.json"])
        results["tritplane"] = d
        print(f"[TRIT] Done! -> {d}")
    except Exception as e:
        errors["tritplane"] = str(e)

def dl_original():
    try:
        print("[ORIG] Starting original model download (~2.3 GB)...")
        files = ["config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors.index.json"]
        for f in files:
            print(f"[ORIG] {f}")
            hf_hub_download("Qwen/Qwen3-1.7B", f)
        print("[ORIG] model-00001-of-00002.safetensors (~1.7 GB)...")
        hf_hub_download("Qwen/Qwen3-1.7B", "model-00001-of-00002.safetensors")
        print("[ORIG] model-00002-of-00002.safetensors (~600 MB)...")
        hf_hub_download("Qwen/Qwen3-1.7B", "model-00002-of-00002.safetensors")
        results["original"] = "done"
        print("[ORIG] Done!")
    except Exception as e:
        errors["original"] = str(e)

t1 = threading.Thread(target=dl_tritplane, daemon=True)
t2 = threading.Thread(target=dl_original, daemon=True)
t1.start(); t2.start()

while t1.is_alive() or t2.is_alive():
    time.sleep(5)
    s1 = "RUNNING" if t1.is_alive() else "DONE"
    s2 = "RUNNING" if t2.is_alive() else "DONE"
    print(f"  Status: tritplane={s1}  original={s2}")

t1.join(); t2.join()
print()
print("=== DOWNLOAD SUMMARY ===")
if "tritplane" in results: print(f"TritPlane: OK -> {results['tritplane']}")
if "original" in results: print("Original: OK")
for k, v in errors.items():
    print(f"ERROR [{k}]: {v}")
print()
if not errors:
    print("Alles da! Run: python test_qwen3_tritplane.py")
