"""Inspect DeepSeek tiny MLA model tensors."""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
from safetensors import safe_open
import json

f = safe_open(r"C:\llm-models\deepseek-tiny-mla\model.safetensors", framework="pt")
for k in sorted(f.keys()):
    t = f.get_tensor(k)
    print(f"  {k}: {t.shape} {t.dtype}")
