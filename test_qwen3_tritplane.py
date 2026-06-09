#!/usr/bin/env python3
"""Smoke test: load TritPlane3-dequantized Qwen3-1.7B into PyTorch and generate.

BUG HISTORY — LSB-first fix (cos 0 → 0.98):
  The 4 ternary values per byte are stored LSB-first, not MSB-first:
    byte & 3       = element 0
    (byte >> 2) & 3 = element 1
    (byte >> 4) & 3 = element 2
    (byte >> 6) & 3 = element 3
  Mapping: 0→-1, 1→0, 2→+1, 3→+1 (3 never occurs).
  Old code used MSB-first mapping which destroyed spatial structure → cos ~0.

Usage:
  python test_qwen3_tritplane.py

Requires:
  - models/Qwen3-1.7B/        (orig config + embed_tokens + norms)
  - models/Qwen3-1.7B-ternary/ (tritplane/*.npz)
  - pip install transformers torch safetensors
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

from huggingface_hub import snapshot_download

# Resolve to local cache paths (will use cache if already downloaded)
TRIT_CACHE = Path(snapshot_download("AsadIsmail/Qwen3-1.7B-ternary",
    allow_patterns=["tritplane/*", "metadata.json"]))
ORIG_CACHE = Path(snapshot_download("Qwen/Qwen3-1.7B",
    allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                     "model.safetensors.index.json",
                     "model-00001-of-00002.safetensors",
                     "model-00002-of-00002.safetensors"]))


def unpack_tritplane_npz(npz_path):
    """Dequantize TritPlane3 .npz to fp32.

    TODO: cos ~0 with original — suspect wrong unpacking layout.
    See hypothesis in module docstring. This function assumes column-wise
    packing (4 consecutive columns per byte within each row). One of the
    three hypotheses (A: row-packing, B: group interleaving, C: transpose)
    is likely the fix.
    """
    data = np.load(npz_path, allow_pickle=True)
    n_planes = sum(1 for k in data if k.startswith("packed_"))
    gs = data["group_size_0"]
    group_size = int(gs) if np.ndim(gs) == 0 else int(gs.flat[0])
    alpha0 = data["group_alpha_0"]
    nrows, n_groups = alpha0.shape
    ncols = n_groups * group_size
    packed_cols = ncols // 4
    result = np.zeros((nrows, ncols), dtype=np.float32)
    for i in range(n_planes):
        pk = data[f"packed_{i}"].reshape(nrows, packed_cols)
        al = data[f"group_alpha_{i}"]
        mu = data[f"group_mu_{i}"]
        p = pk.astype(np.int32, copy=False)
        # LSB-first: byte & 3 = element 0, (byte>>2)&3 = element 1, ...
        # Mapping: 0→-1, 1→0, 2→+1, 3→+1 (3 never occurs in practice)
        map_2bit = np.array([-1, 0, 1, 1])
        t0 = map_2bit[p & 3]
        t1 = map_2bit[(p >> 2) & 3]
        t2 = map_2bit[(p >> 4) & 3]
        t3 = map_2bit[(p >> 6) & 3]
        tern = np.empty((nrows, ncols), dtype=np.int32)
        for j in range(packed_cols):
            tern[:, j*4+0] = t0[:, j]
            tern[:, j*4+1] = t1[:, j]
            tern[:, j*4+2] = t2[:, j]
            tern[:, j*4+3] = t3[:, j]
        result += al.repeat(group_size, axis=1).astype(np.float32) * tern + mu.repeat(group_size, axis=1).astype(np.float32)
    data.close()
    return result


def npz_to_tensor_name(filename):
    """tritplane filename -> ATLAS tensor name."""
    from tritplane_to_atlas import npz_to_tensor_name as _conv
    return _conv(filename)


def load_norms_and_embed(orig_dir, device="cpu"):
    """Load non-quantized tensors from original model safetensors.

    We need: embed_tokens, model.norm, layer norms (input, post_attn, q, k).
    Each norm is just [hidden_size] fp16.
    """
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download

    needed = [
        "model.embed_tokens.weight",
        "model.norm.weight",
    ]
    # Layer norms
    n_layers = 28
    needed += [f"model.layers.{i}.input_layernorm.weight" for i in range(n_layers)]
    needed += [f"model.layers.{i}.post_attention_layernorm.weight" for i in range(n_layers)]
    needed += [f"model.layers.{i}.self_attn.q_norm.weight" for i in range(n_layers)]
    needed += [f"model.layers.{i}.self_attn.k_norm.weight" for i in range(n_layers)]

    # Load weight map
    idx = json.loads(Path(orig_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]

    # Group by shard
    shard_tensors = {}
    for tname in needed:
        shard = wm.get(tname)
        if shard:
            shard_tensors.setdefault(shard, []).append(tname)

    result = {}
    for shard, tnames in sorted(shard_tensors.items()):
        sf_path = orig_dir / shard
        if not sf_path.exists():
            print(f"  Downloading {shard}...")
            sf_path = Path(hf_hub_download("Qwen/Qwen3-1.7B", shard))
        print(f"  Loading {len(tnames)} tensors from {shard}")
        with safe_open(str(sf_path), framework="pt") as f:
            for tname in tnames:
                result[tname] = f.get_tensor(tname).to(device=device, dtype=torch.float16)

    return result


@torch.no_grad()
def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Device: {device}")

    # 1. Load config
    print("\n[1] Loading config...")
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(str(ORIG_CACHE), trust_remote_code=True)
    print(f"  {config.model_type}, {config.num_hidden_layers}L, {config.hidden_size}H, {config.intermediate_size}I")

    # 2. Dequantize all tritplane weights
    print("\n[2] Dequantizing tritplane weights...")
    npz_files = sorted((TRIT_CACHE / "tritplane").glob("*.npz"))
    print(f"  Found {len(npz_files)} .npz files")
    deq = {}
    t0 = time.time()
    for i, npz_path in enumerate(npz_files):
        tname = npz_to_tensor_name(npz_path.name)
        deq[tname] = unpack_tritplane_npz(str(npz_path))
        if (i+1) % 50 == 0 or i == len(npz_files)-1:
            print(f"  [{i+1}/{len(npz_files)}] {time.time()-t0:.0f}s")
    print(f"  Dequantized {len(deq)} tensors in {time.time()-t0:.0f}s")

    # 3. Load norms + embed from original model
    print("\n[3] Loading original model tensors (norms, embed)...")
    orig = load_norms_and_embed(ORIG_CACHE, device=device)
    print(f"  Loaded {len(orig)} tensors")

    # 4. Build model skeleton
    print("\n[4] Building model skeleton...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = model.to(dtype=torch.float16, device=device).eval()
    print(f"  Skeleton created in {time.time()-t0:.0f}s")

    # 5. Set embeddings
    print("\n[5] Loading weights into model...")
    model.model.embed_tokens.weight.data.copy_(orig["model.embed_tokens.weight"])

    # Final norm
    model.model.norm.weight.data.copy_(orig["model.norm.weight"])

    # LM head: tied to embed_tokens (tie_word_embeddings=True)

    # Per-layer: replace weight matrices with dequantized, set norms from original
    for i in range(config.num_hidden_layers):
        layer = model.model.layers[i]
        prefix = f"model.layers.{i}"

        if (i+1) % 7 == 0:
            print(f"  Layer {i}...")

        # Dequantized weight matrices (fp32 -> fp16)
        layer.self_attn.q_proj.weight.data.copy_(torch.from_numpy(deq[f"{prefix}.self_attn.q_proj.weight"]).to(device=device, dtype=torch.float16))
        layer.self_attn.k_proj.weight.data.copy_(torch.from_numpy(deq[f"{prefix}.self_attn.k_proj.weight"]).to(device=device, dtype=torch.float16))
        layer.self_attn.v_proj.weight.data.copy_(torch.from_numpy(deq[f"{prefix}.self_attn.v_proj.weight"]).to(device=device, dtype=torch.float16))
        layer.self_attn.o_proj.weight.data.copy_(torch.from_numpy(deq[f"{prefix}.self_attn.o_proj.weight"]).to(device=device, dtype=torch.float16))

        layer.mlp.gate_proj.weight.data.copy_(torch.from_numpy(deq[f"{prefix}.mlp.gate_proj.weight"]).to(device=device, dtype=torch.float16))
        layer.mlp.up_proj.weight.data.copy_(torch.from_numpy(deq[f"{prefix}.mlp.up_proj.weight"]).to(device=device, dtype=torch.float16))
        layer.mlp.down_proj.weight.data.copy_(torch.from_numpy(deq[f"{prefix}.mlp.down_proj.weight"]).to(device=device, dtype=torch.float16))

        # Norms from original model
        layer.input_layernorm.weight.data.copy_(orig[f"{prefix}.input_layernorm.weight"])
        layer.post_attention_layernorm.weight.data.copy_(orig[f"{prefix}.post_attention_layernorm.weight"])
        layer.self_attn.q_norm.weight.data.copy_(orig[f"{prefix}.self_attn.q_norm.weight"])
        layer.self_attn.k_norm.weight.data.copy_(orig[f"{prefix}.self_attn.k_norm.weight"])

    # 6. Generate!
    print("\n[6] Generating...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ORIG_CACHE), trust_remote_code=True)

    prompts = [
        "The capital of France is",
        "The Eiffel Tower is located in",
        "2 + 2 =",
    ]

    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        t0 = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        elapsed = time.time() - t0

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
        print(f"  Output: {generated}")
        print(f"  Speed: {tokens_generated / elapsed:.1f} tok/s ({tokens_generated} tok in {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
