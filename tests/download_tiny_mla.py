"""Download ChrisMcCormick/deepseek-tiny-mla-o-v0.1 from HuggingFace."""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

from huggingface_hub import snapshot_download

path = snapshot_download(
    "ChrisMcCormick/deepseek-tiny-mla-o-v0.1",
    local_dir=r"C:\llm-models\deepseek-tiny-mla",
)
print(f"Downloaded to: {path}")
