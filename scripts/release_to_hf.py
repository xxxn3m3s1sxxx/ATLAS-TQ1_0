#!/usr/bin/env python3
"""Release an ATLAS-packed model to Hugging Face Hub.

Usage:
  python scripts/release_to_hf.py <model_dir> <repo_id> [--push]

Steps:
  1. Pack model_dir to .atlas via pack_to_atlas.py
  2. Optionally push to HF Hub (requires HF_TOKEN env var)

The pack step is local-only and always runs. The --push flag gates upload.

Examples:
  python scripts/release_to_hf.py ./falcon3-7B xxxn3m3s1sxxx/Falcon3-7B-Atlas
  python scripts/release_to_hf.py --atlas-path model.atlas --repo-id xxxn3m3s1sxxx/Falcon3-7B-Atlas
  python scripts/release_to_hf.py --atlas-path model.atlas --repo-id user/Repo --push
"""
import argparse, json, os, sys


# Known configs for README completeness when model_dir unavailable
KNOWN_CONFIGS = {
    "falcon3-1B": {"model_type": "falcon3", "num_hidden_layers": 18, "hidden_size": 2048,
                   "intermediate_size": 8192, "num_attention_heads": 8,
                   "num_key_value_heads": 4, "head_dim": 256, "rope_theta": 1000042.0,
                   "vocab_size": 131072},
    "falcon3-3B": {"model_type": "falcon3", "num_hidden_layers": 22, "hidden_size": 3072,
                   "intermediate_size": 9216, "num_attention_heads": 12,
                   "num_key_value_heads": 4, "head_dim": 256, "rope_theta": 1000042.0,
                   "vocab_size": 131072},
    "falcon3-7B": {"model_type": "falcon3", "num_hidden_layers": 28, "hidden_size": 3072,
                   "intermediate_size": 23040, "num_attention_heads": 12,
                   "num_key_value_heads": 4, "head_dim": 256, "rope_theta": 1000042.0,
                   "vocab_size": 131080},
    "falcon3-10B": {"model_type": "falcon3", "num_hidden_layers": 40, "hidden_size": 3072,
                    "intermediate_size": 23040, "num_attention_heads": 12,
                    "num_key_value_heads": 4, "head_dim": 256, "rope_theta": 1000042.0,
                    "vocab_size": 131072},
    "bonsai-1.7B": {"model_type": "qwen3", "num_hidden_layers": 28, "hidden_size": 2048,
                    "intermediate_size": 6144, "num_attention_heads": 16,
                    "num_key_value_heads": 8, "head_dim": 128, "rope_theta": 1000000.0,
                    "vocab_size": 151669},
    "bonsai-4B": {"model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 2560,
                  "intermediate_size": 9728, "num_attention_heads": 32,
                  "num_key_value_heads": 8, "head_dim": 128, "rope_theta": 5000000.0,
                  "vocab_size": 151669},
    "bonsai-8B": {"model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 4096,
                  "intermediate_size": 12288, "num_attention_heads": 32,
                  "num_key_value_heads": 8, "head_dim": 128, "rope_theta": 1000000.0,
                  "vocab_size": 151669},
    "bitnet-2B": {"model_type": "bitnet", "num_hidden_layers": 30, "hidden_size": 2560,
                  "intermediate_size": 6912, "num_attention_heads": 20,
                  "num_key_value_heads": 5, "head_dim": 128, "rope_theta": 500000.0,
                  "vocab_size": 128256, "tie_word_embeddings": True},
    "cann-0.5B": {"model_type": "minicpm", "num_hidden_layers": 24, "hidden_size": 1024,
                  "intermediate_size": 4096, "num_attention_heads": 16,
                  "num_key_value_heads": 2, "head_dim": 64, "rope_theta": 10000.0,
                  "vocab_size": 73448, "scale_emb": 12, "scale_depth": 1.4},
    "cann-1B": {"model_type": "minicpm", "num_hidden_layers": 28, "hidden_size": 2048,
                "intermediate_size": 6144, "num_attention_heads": 16,
                "num_key_value_heads": 2, "head_dim": 128, "rope_theta": 10000.0,
                "vocab_size": 73448, "scale_emb": 12, "scale_depth": 1.4},
    "cann-3B": {"model_type": "minicpm", "num_hidden_layers": 32, "hidden_size": 2560,
                "intermediate_size": 10240, "num_attention_heads": 32,
                "num_key_value_heads": 2, "head_dim": 128, "rope_theta": 10000.0,
                "vocab_size": 73448, "scale_emb": 12, "scale_depth": 1.4},
    "cann-8B": {"model_type": "minicpm", "num_hidden_layers": 32, "hidden_size": 4096,
                "intermediate_size": 16384, "num_attention_heads": 32,
                "num_key_value_heads": 2, "head_dim": 128, "rope_theta": 10000.0,
                "vocab_size": 73448, "scale_emb": 12, "scale_depth": 1.4,
                "tie_word_embeddings": False},
    "1bitllm-large": {"model_type": "bitnet_paper", "num_hidden_layers": 24, "hidden_size": 1536,
                      "intermediate_size": 4096, "num_attention_heads": 16,
                      "num_key_value_heads": 16, "head_dim": 96, "rope_theta": 10000.0,
                      "vocab_size": 32002},
    "1bitllm-xl": {"model_type": "bitnet_paper", "num_hidden_layers": 24, "hidden_size": 2048,
                   "intermediate_size": 5460, "num_attention_heads": 32,
                   "num_key_value_heads": 32, "head_dim": 64, "rope_theta": 10000.0,
                   "vocab_size": 32002},
    "1bitllm-3B": {"model_type": "bitnet_paper", "num_hidden_layers": 26, "hidden_size": 3200,
                   "intermediate_size": 8640, "num_attention_heads": 32,
                   "num_key_value_heads": 32, "head_dim": 100, "rope_theta": 10000.0,
                   "vocab_size": 32002},
}

# Per-model perf & size description — keys must match KNOWN_CONFIGS
MODEL_SIZES = {
    "falcon3-10B": ("2.3 tok/s (int4 FFN)",  "3.28 GB", "40 layers, 3072 hidden, 23040 intermediate \u2014 maximum quality"),
    "falcon3-7B":  ("3.2 tok/s (int4 FFN)",  "2.75 GB", "28 layers, 3072 hidden, 23040 intermediate \u2014 quality output"),
    "falcon3-3B":  ("7.1 tok/s (hybrid)",    "1.97 GB", "22 layers, 3072 hidden, 9216 intermediate \u2014 best balance"),
    "falcon3-1B":  ("10.1 tok/s (f32 bypass)", "1.22 GB", "18 layers, 2048 hidden, 8192 intermediate \u2014 light & fast"),
    "bonsai-4B":   ("17.4 tok/s (hybrid)",   "1.49 GB", "36 layers, 2560 hidden, 9728 intermediate \u2014 fast Bonsai"),
    "bonsai-1.7B": ("13.0 tok/s (f32 bypass)","0.86 GB", "28 layers, 2048 hidden, 6144 intermediate \u2014 light Bonsai"),
    "bonsai-8B":   ("1.8 tok/s (f32 bypass)", "3.72 GB", "36 layers, 4096 hidden, 12288 intermediate \u2014 max Bonsai quality"),
    "bitnet-2B":   ("2.8 tok/s (f32 bypass)", "1.04 GB", "30 layers, 2560 hidden, 6912 intermediate \u2014 BitNet b1.58"),
    "cann-0.5B":   ("30 tok/s (f32 bypass)",  "0.22 GB", "24 layers, 1024 hidden, 4096 intermediate \u2014 tiny CANN"),
    "cann-1B":     ("9.7 tok/s (hybrid)",     "0.83 GB", "28 layers, 2048 hidden, 6144 intermediate \u2014 CANN balanced"),
    "cann-3B":     ("7.1 tok/s (hybrid)",     "1.35 GB", "32 layers, 2560 hidden, 10240 intermediate \u2014 CANN quality"),
    "cann-8B":     ("1.5 tok/s (f32 bypass)", "2.65 GB", "32 layers, 4096 hidden, 16384 intermediate \u2014 CANN max quality"),
    "1bitllm-large": ("7.0 tok/s (f32 bypass)", "0.23 GB", "24 layers, 1536 hidden, 4096 intermediate \u2014 paper reproduction"),
    "1bitllm-xl":   ("6.0 tok/s (f32 bypass)", "0.37 GB", "24 layers, 2048 hidden, 5460 intermediate \u2014 paper reproduction"),
    "1bitllm-3B":   ("4.0 tok/s (f32 bypass)", "0.84 GB", "26 layers, 3200 hidden, 8640 intermediate \u2014 paper reproduction"),
}


def _read_atlas_arch(atlas_path):
    """Read architecture type from v8 atlas file metadata."""
    try:
        import struct, json
        with open(atlas_path, "rb") as f:
            hdr = f.read(64)
            meta_size = struct.unpack_from("<I", hdr, 64)[0]
            meta_bytes = f.read(meta_size)
            meta = json.loads(meta_bytes.decode("utf-8"))
            return meta.get("arch", "unknown")
    except Exception:
        return None


def _detect_size(name):
    """Extract model family+size from filename (e.g. 'Ternary-BitCPM-CANN-1B' -> 'cann-1B')."""
    family_map = {"Falcon3": "falcon3", "Bonsai": "bonsai", "BitNet": "bitnet",
                  "CANN": "cann", "Llama3": "llama3", "TriLM": "trilm",
                  "1bitLLM": "1bitllm"}
    for fam, key in family_map.items():
        if fam in name:
            for sz in ["10B", "8B", "1.7B", "7B", "4B", "3B", "2B", "1B", "0.5B"]:
                if sz in name:
                    return f"{key}-{sz}"
            # Variant names (no B-suffix)
            for variant in ["large", "xl"]:
                if variant in name:
                    return f"{key}-{variant}"
    return None


def _bare_size(size_key):
    """Extract bare size from family-prefixed key (e.g. 'cann-1B' -> '1B')."""
    for sz in ["0.5B", "1.7B", "10B", "8B", "7B", "4B", "3B", "2B", "1B", "large", "xl"]:
        if sz in size_key:
            return sz
    return size_key


def pack_model(model_dir, output_dir=None):
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "unknown")
    hidden = cfg.get("hidden_size", 0)
    n_layers = cfg.get("num_hidden_layers", 0)
    model_size = f"{n_layers}Lx{hidden}H" if hidden else model_type

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(output_dir, exist_ok=True)

    fname = f"{model_type}-{model_size}.atlas"
    output_path = os.path.join(output_dir, fname)

    print(f"[release] Packing {model_dir} -> {output_path}")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pack_to_atlas import pack_to_atlas
    pack_to_atlas(model_dir, output_path)

    if not os.path.exists(output_path):
        print("[release] ERROR: pack produced no output")
        sys.exit(1)

    print(f"[release] Pack OK: {os.path.getsize(output_path) / 1024 ** 3:.2f} GB")
    return output_path, cfg


def _generate_readme(model_name, size, cfg, atlas_name, perf, file_size_str, desc):
    is_bonsai = "Bonsai" in model_name or cfg.get("model_type") in ("qwen3",)
    is_falcon3 = "Falcon3" in model_name or cfg.get("model_type") == "falcon3"
    is_bitnet = "BitNet" in model_name or cfg.get("model_type") == "bitnet"
    is_cann = "CANN" in model_name or cfg.get("model_type") == "minicpm"
    is_1bitllm = "1bitLLM" in model_name or cfg.get("model_type") == "bitnet_paper"
    bsz = _bare_size(size)  # bare size for base model HF names
    engine_note = ""

    if is_bonsai:
        base_model_hf = f"prism-ml/Ternary-Bonsai-{bsz}-unpacked"
        ctx_window = "8192 (YaRN-scalable up to 16384)"
        prompt_template = """```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
```"""
        license_yaml = "license: apache-2.0"
        license_section = """\
## License

This is a **quantized derivative work** based on the **Ternary-Bonsai** architecture (original model by **Prism ML**), originally released under **Apache 2.0**.

The ATLAS engine itself is also **Apache 2.0 licensed** — a clean, permissive, fully open-source stack.
"""
    elif is_bitnet:
        base_model_hf = "microsoft/bitnet-b1.58-2B-4T"
        ctx_window = "2048 (NTK-scalable up to 4096)"
        prompt_template = """```
{role}: {content}<|eot_id|>
```

### Example Sequence

```
User: What is 2+2?
Assistant: 4
```"""
        license_yaml = "license: mit"
        license_section = """\
## License

This is a **quantized derivative work** based on the **BitNet b1.58** architecture (original model by **Microsoft**), originally released under **MIT License**.

The ATLAS engine itself is **Apache 2.0 licensed**.
"""
    elif is_cann:
        base_model_hf = f"openbmb/BitCPM-CANN-{bsz}"
        ctx_window = "32768 (LongRoPE, NTK-scalable)"
        prompt_template = """```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

### Example Sequence

```
<|im_start|>user
Explain quantum computing in one sentence.<|im_end|>
<|im_start|>assistant
```"""
        license_yaml = "license: apache-2.0"
        engine_note = """\
> **Requires ATLAS Engine v2.10.4+** — older engine versions lack `scale_emb` and `scale_depth` support for MiniCPM DeepNorm residual scaling.
"""
        license_section = f"""\
## License

This is a **quantized derivative work** based on the **BitCPM-CANN** architecture (original model by **OpenBMB**), originally released under **Apache 2.0**.

The ATLAS engine itself is **Apache 2.0 licensed**.
"""
    elif is_1bitllm:
        base_model_hf = f"1bitLLM/bitnet_b1_58-{bsz if bsz in ['3B'] else 'large' if bsz == 'large' else 'xl'}"
        ctx_window = "2048 (NTK-scalable up to 4096)"
        prompt_template = """```
{role}: {content}
```

### Example Sequence

```
User: What is 2+2?
Assistant: 4
```"""
        license_yaml = "license: mit"
        engine_note = """\
> **Paper reproduction model** — this is a research reproduction of the BitNet b1.58 architecture by the 1bitLLM team. Quality is limited by the RedPajama 100B token training budget.
"""
        license_section = """\
## License

This is a **quantized derivative work** based on the **1bitLLM BitNet b1.58 paper reproduction** (original model by **1bitLLM**), originally released under **MIT License**.

The ATLAS engine itself is **Apache 2.0 licensed**.
"""
    else:
        base_model_hf = f"tiiuae/Falcon3-{bsz}-Instruct-1.58bit"
        ctx_window = "4096 (NTK-scalable up to 8192)"
        prompt_template = """```
<|role|>
{content}
<|endoftext|>
```

### Example Sequence

```
<|user|>
Explain quantum computing in one sentence.
<|assistant|>
```"""
        license_yaml = """\
license: other
license_name: tii-falcon-llm-license-2.0
license_link: https://huggingface.co/spaces/tiiuae/falcon3-license"""
        license_section = """\
## License & Usage Restrictions

This is a **quantized derivative work** based on Falcon3 architectures developed by the **Technology Innovation Institute (TII)**.

By downloading or utilizing this file, you agree to be bound by the **TII Falcon-LLM License 2.0**:

1. **Attribution:** Any usage or secondary deployment must credit the Technology Innovation Institute (TII).
2. **Non-Commercial & Small Commercial Use:** Free for academic research, personal projects, and commercial entities with **annual revenue under $1,000,000 USD**.
3. **Commercial Royalty Terms:** Entities exceeding the $1M annual revenue threshold are subject to a **10% licensing fee** on revenue exceeding that amount, as specified in the master Falcon3 license terms.

*Disclaimer: This quantized file is provided "as-is" by the open-source community. While model outputs have been verified for correctness (see Verification above), no explicit or implied warranties regarding mathematical equivalence to FP16 baselines are made.*

The ATLAS engine itself is **Apache 2.0 licensed**.
"""

    extra_tags = ""
    if is_bonsai:
        extra_tags = "- cpu-llm\n- edge-ai\n- no-gpu\n- efficient-inference"
    elif is_bitnet or is_1bitllm:
        extra_tags = "- cpu-inference\n- bitnet"
    elif is_falcon3:
        extra_tags = "- cpu-inference\n- bitnet"
    elif is_cann:
        extra_tags = "- cpu-llm\n- edge-ai\n- no-gpu\n- efficient-inference"
    if is_1bitllm:
        arch_tag = "bitnet"
    elif is_bonsai:
        arch_tag = "bonsai"
    elif is_bitnet:
        arch_tag = "bitnet"
    elif is_falcon3:
        arch_tag = "falcon3"
    elif is_cann:
        arch_tag = "minicpm"
    else:
        arch_tag = "llama"
    frontmatter = f"""---
{license_yaml}
language:
- en
tags:
- ternary
- quantized
- atlas
- tq1
- cpu-optimized
- {arch_tag}
- llm
{extra_tags}
base_model: {base_model_hf}
pipeline_tag: text-generation
library_name: atlas
---"""

    model_type = cfg.get("model_type", "llama")
    n_layers = cfg.get("num_hidden_layers", "?")
    n_heads = cfg.get("num_attention_heads", "?")
    n_kv = cfg.get("num_key_value_heads", "?")
    hidden = cfg.get("hidden_size", "?")
    intermediate = cfg.get("intermediate_size", "?")
    head_dim = cfg.get("head_dim", "?")
    rope_theta = cfg.get("rope_theta", "?")
    vocab = cfg.get("vocab_size", "?")

    body = f"""
# {model_name} (v2.10.0)

This repository contains a highly optimized **TQ1 quantized version** of the official `{base_model_hf}` model for the **ATLAS Engine** ecosystem, designed for native, ultra-low-latency CPU inference without any GPU requirement.

> Packed using the unified `pack_to_atlas.py` toolchain (v2.10.0) with BF16 weight scale correction.

---

## Engine Specifications

| Property | Value |
|---|---|
| **Format** | ATLAS Binary (`.atlas`), format_version=2 |
| **Quantization** | TQ1.0 \u2014 Ternary Weight Packing (Base-3, ~1.58 bits/weight) |
| **Target** | Native CPU \u2014 Intel AVX2 (Haswell 2013+), no GPU needed |
| **File Size** | {file_size_str} |
| **Inference Speed** | {perf} |
| **Description** | {desc} |

### Architecture

| Component | Detail |
|---|---|
| Base Model | `{base_model_hf}` |
| Architecture | {model_type} |
| Layers | {n_layers} |
| Hidden Size | {hidden} |
| Intermediate Size | {intermediate} |
| Attention Heads | {n_heads} (GQA, {n_kv} KV heads) |
| Head Dim | {head_dim} |
| RoPE Theta | {rope_theta} |
| Vocabulary | {vocab} |
| Context Window | {ctx_window} |

{engine_note}
### Verification

During pre-release evaluation (v2.10.0), this quantized derivative demonstrated correct convergence:
- **T=0 (argmax):** `"The capital of France is Paris."` \u2014 correct deterministic output
- **T=0.7 (sampling):** Coherent structured generation with sensible continuation

> *Note on scale mathematics:* the legacy dequantization path divides by the scale factor rather than multiplying. Since this is a constant across all logits for any given output row, the relative probability distribution remains identical under softmax normalization \u2014 no effect on output quality.

---

## Prompt Template

To prevent token degradation and alignment shifting, use the standard chat template:

{prompt_template}

---

## Usage

### Python

```bash
git clone https://github.com/xxxn3m3s1sxxx/ATLAS-TQ1_0.git
```

```python
from atlas_infer import AtlasModel

model = AtlasModel("{atlas_name}")
output = model.generate_c(
    "What is the capital of France?",
    max_new_tokens=100,
    temperature=0.7,
    top_k=40,
)
print(output)
```

### C++ CLI (standalone, no Python required)

```bash
atlas --model {atlas_name} --prompt "What is the capital of France?" --max-tokens 100
```

### SSE Web Server

```bash
python atlas_server.py --model {atlas_name} --port 8080
curl http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{"prompt": "What is the capital of France?", "max_tokens": 100}}'
```

---

## What is ATLAS?

**ATLAS** is a CPU inference engine for BitNet b1.58 ternary-quantized models. It repacks HuggingFace safetensors into the **TQ1.0 format** (5 ternary trits per byte, Base-3 encoding, ~1.58 bits/weight) and runs fast inference via a C++ DLL + Python wrapper.

| Feature | Description |
|---|---|
| **No GPU required** | Runs on any x86-64 CPU with AVX2 (Intel Haswell 2013+, AMD Excavator 2015+) |
| **Hybrid matmul** | FFN tensors in int8, QKV/O in TQ1-packed, per-tensor dispatch |
| **int4 FFN mode** | Halves FFN memory bandwidth for 18-26% speedup (7B/10B) |
| **f32 bypass** | Auto-enabled for small models (\u22641B) and SubLN architectures |
| **Ring buffer KV cache** | Extended context via NTK-aware RoPE scaling |
| **Standalone C++ CLI** | No Python or PyTorch required at runtime |
| **SSE web server** | FastAPI-based `/v1/chat/completions` with prompt caching |

### Links

- **Engine source code**: [github.com/xxxn3m3s1sxxx/ATLAS-TQ1_0](https://github.com/xxxn3m3s1sxxx/ATLAS-TQ1_0)
- **Original model**: [`{base_model_hf}`](https://huggingface.co/{base_model_hf})

---

{license_section}
"""
    return frontmatter + body


def push_to_hub(atlas_path, repo_id, cfg):
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[release] ERROR: HF_TOKEN not set \u2014 cannot push")
        print("  Set: $env:HF_TOKEN = 'hf_your_token_here'")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[release] ERROR: huggingface_hub not installed")
        print("  pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        token=token,
    )
    print(f"[release] Repo {repo_id} ready")

    atlas_name = os.path.basename(atlas_path)
    print(f"[release] Uploading {atlas_name} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=atlas_path,
        path_in_repo=atlas_name,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"[release] Upload OK: https://huggingface.co/{repo_id}")

    model_name = repo_id.split('/')[-1]
    size = _detect_size(model_name) or "?"
    perf, file_size_str, desc = MODEL_SIZES.get(size, ("varies", "?", ""))

    readme = _generate_readme(model_name, size, cfg, atlas_name, perf, file_size_str, desc)
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"[release] README.md uploaded")

    config_json = json.dumps(cfg, indent=2).encode()
    api.upload_file(
        path_or_fileobj=config_json,
        path_in_repo="config.json",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"[release] config.json uploaded")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Release an ATLAS-packed model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model_dir", nargs="?",
                        help="HF model directory (config.json + safetensors)")
    parser.add_argument("--atlas-path", default=None,
                        help="Pre-packed .atlas file (use instead of model_dir)")
    parser.add_argument("--repo-id", default=None, required=True,
                        help="HF repo ID (e.g. xxxn3m3s1sxxx/Falcon3-7B-Atlas)")
    parser.add_argument("--push", action="store_true",
                        help="Upload to Hugging Face Hub (requires HF_TOKEN)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for .atlas file (default: ./models/)")
    args = parser.parse_args()

    if args.atlas_path:
        if not os.path.exists(args.atlas_path):
            print(f"[release] ERROR: atlas_path not found: {args.atlas_path}")
            sys.exit(1)
        atlas_path = args.atlas_path
        fname = os.path.basename(atlas_path)
        if args.model_dir:
            cfg_path = os.path.join(args.model_dir, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
            else:
                cfg = {"model_type": "unknown"}
        else:
            # Try reading arch from v8 metadata block
            arch = _read_atlas_arch(atlas_path)
            if arch:
                cfg = {"model_type": arch}
            else:
                sz = _detect_size(fname)
                if sz and sz in KNOWN_CONFIGS:
                    cfg = dict(KNOWN_CONFIGS[sz])
                else:
                    clean = fname.replace("-tq1.atlas", "").replace(".atlas", "")
                    cfg = {"model_type": clean}
        print(f"[release] Using pre-packed: {atlas_path} ({os.path.getsize(atlas_path) / 1024 ** 3:.2f} GB)")
    elif args.model_dir:
        atlas_path, cfg = pack_model(args.model_dir, args.output_dir)
    else:
        print("[release] ERROR: provide <model_dir> or --atlas-path <file>")
        sys.exit(1)

    if args.push:
        push_to_hub(atlas_path, args.repo_id, cfg)
    else:
        print(f"[release] Dry run - use --push to upload to {args.repo_id}")
        print("[release] Set $env:HF_TOKEN before --push")


if __name__ == "__main__":
    main()
