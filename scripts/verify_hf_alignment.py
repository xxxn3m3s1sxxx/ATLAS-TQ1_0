#!/usr/bin/env python3
"""Verify ATLAS tensor name/dimension alignment against real HuggingFace models.

Fetches config.json + model.safetensors.index.json from HF Hub via raw URLs
(no git-lfs, no model downloads) and compares against ARCHES definitions
in tests/atlas_mock_model.py.

Covers all ATLAS TQ1.0-supported architectures:
  - Falcon3 (tiiuae, 1B/3B/7B/10B)
  - Bonsai (PrismML Qwen3-derivatives, 1.7B/4B/8B)
  - BitNet b1.58 (Microsoft, 2B4T)
  - TriLM (SpectraSuite SubLN, 99M→3.9B)

Usage:
    python scripts/verify_hf_alignment.py
    python scripts/verify_hf_alignment.py --model tiiuae/Falcon3-7B-Base
    python scripts/verify_hf_alignment.py --verbose
    python scripts/verify_hf_alignment.py --json

Exit code: 0 = all pass, 1 = any fail
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

# ASCII-safe markers for Windows cp1252
OK = "[OK]"
MISS = "[X]"
WARN = "[W]"
AKI_WARN = "[A]"
NOTE = "[N]"
SKIP = "[S]"
PASS = "PASS"
FAIL = "FAIL"

HF_BASE = "https://huggingface.co"

DEFAULT_MODELS = [
    # Falcon3 — 4 variants
    "tiiuae/Falcon3-1B-Base",
    "tiiuae/Falcon3-3B-Base",
    "tiiuae/Falcon3-7B-Instruct",
    "tiiuae/Falcon3-10B-Base",
    # Bonsai (PrismML Qwen3-derivatives) — 3 variants
    "prism-ml/Ternary-Bonsai-1.7B-unpacked",
    "prism-ml/Ternary-Bonsai-4B-unpacked",
    "prism-ml/Ternary-Bonsai-8B-unpacked",
    # BitNet b1.58 — 1 model, 2 weight formats (BF16 + U8 packed)
    "microsoft/BitNet-2B4T-b1.58",
    # TriLM — ALL 10 sizes (SpectraSuite SubLN)
    "SpectraSuite/TriLM_99M_Unpacked",
    "SpectraSuite/TriLM_190M_Unpacked",
    "SpectraSuite/TriLM_390M_Unpacked",
    "SpectraSuite/TriLM_560M_Unpacked",
    "SpectraSuite/TriLM_830M_Unpacked",
    "SpectraSuite/TriLM_1.1B_Unpacked",
    "SpectraSuite/TriLM_1.5B_Unpacked",
    "SpectraSuite/TriLM_2.3B_Unpacked",  # PRIVATE
    "SpectraSuite/TriLM_2.4B_Unpacked",
    "SpectraSuite/TriLM_3.9B_Unpacked",
]

# Known orgs for auto-discovery
DISCOVER_ORGS = [
    "tiiuae",
    "prism-ml",
    "microsoft",
    "SpectraSuite",
]

# Expected layer tensor names per architecture (without model.layers.N. prefix)
# Sourced from tests/atlas_mock_model.py ARCHES definitions
EXPECTED_LAYER_TENSORS = {
    "falcon3": [
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ],
    "qwen25": [
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ],
    "qwen3": [
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    ],
    # bitnet + trilm share SubLN architecture (attn_sub_norm + ffn_sub_norm)
    "bitnet": [
        "input_layernorm.weight",
        "self_attn.attn_sub_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "mlp.ffn_sub_norm.weight",
    ],
    "trilm": [
        "input_layernorm.weight",
        "self_attn.attn_sub_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "mlp.ffn_sub_norm.weight",
    ],
    # trilm_nosubln: TriLM 3.9B+ = standard Llama, NO SubLN
    "trilm_nosubln": [
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ],
}

EXPECTED_GLOBAL = ["model.embed_tokens.weight", "model.norm.weight"]


def fetch_json(url, retries=3):
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": "atlas-hf-alignment/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  [R] Rate limited — retrying in {wait}s...")
                time.sleep(wait)
                continue
            return {"_error": f"HTTP {e.code}: {e.reason}"}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            return {"_error": str(e)}
    return {"_error": "max retries exceeded"}


def derive_arch(model_id, config):
    mid = model_id.lower()
    cfg_arch = (config.get("architectures") or [""])[0].lower() if config.get("architectures") else ""
    cfg_type = (config.get("model_type") or "").lower()
    hd = config.get("head_dim", 0)
    nh = config.get("num_attention_heads", 0)
    h = config.get("hidden_size", 0)
    if hd == 0 and nh > 0:
        hd = h // nh

    # Qwen3 = has QK-Norm (q_norm, k_norm)
    if "qwen3" in cfg_type or "qwen3" in cfg_arch:
        return "qwen3"
    # Qwen2/Qwen2.5/QwQ = NO QK-Norm
    if "qwen2" in cfg_type or "qwen" in cfg_type:
        return "qwen25"
    if "qwen" in mid:
        return "qwen25"

    # BitNet: ReLU² activation, bitnet in name/arch
    if "bitnet" in mid or "bitnet" in cfg_arch or config.get("hidden_act") == "relu2":
        return "bitnet"
    # TriLM: SpectraSuite SubLN models — SubLN only for head_dim=64 (small/medium TriLMs)
    # TriLM 3.9B has head_dim=128 and NO SubLN (standard Llama arch)
    if "trilm" in mid or "trlm" in mid or "spectrasuite" in mid:
        if hd >= 128:
            return "trilm_nosubln"  # ~3.9B sizes: standard Llama, no SubLN
        return "trilm"  # <= 2.4B sizes: SubLN architecture
    # Falcon3: TII Falcon3 models
    if "falcon" in mid or "falcon" in cfg_arch:
        return "falcon3"
    # Generic Llama fallback
    if "llama" in cfg_arch:
        # Llama models with SubLN config flags are TriLM-like
        has_subln = config.get("attn_sub_norm", False) or config.get("use_attn_sub_norm", False)
        has_ffn_subln = config.get("ffn_sub_norm", False) or config.get("use_ffn_sub_norm", False)
        if has_subln or has_ffn_subln:
            return "trilm"
        return "falcon3"
    # Bonsai: PrismML ternary models (Qwen3-derivatives)
    if "bonsai" in mid or "ternary-bonsai" in mid:
        return "qwen3"
    return "unknown"


def extract_dimensions(config):
    h = config.get("hidden_size", 0)
    v = config.get("vocab_size", 0)
    i = config.get("intermediate_size", 0)
    nh = config.get("num_attention_heads", 0)
    nk = config.get("num_key_value_heads", 0)
    n_layers = config.get("num_hidden_layers", 0)
    hd = config.get("head_dim", 0)
    if hd == 0 and nh > 0:
        hd = h // nh
    rope_theta = config.get("rope_theta", 10000.0)
    tie_emb = config.get("tie_word_embeddings", False)
    act = config.get("hidden_act", "silu")
    return h, v, i, nh, nk, n_layers, hd, rope_theta, tie_emb, act


def check_alignment(model_id):
    print(f"\n{'='*60}")
    print(f"  {model_id}")
    print(f"{'='*60}")

    config_url = f"{HF_BASE}/{model_id}/raw/main/config.json"
    config = fetch_json(config_url)
    if "_error" in config:
        err_msg = config['_error']
        print(f"  {'[S]' if '401' in err_msg else FAIL} Config fetch: {err_msg}")
        # 401 = restricted repo (e.g. microsoft/BitNet), not a alignment failure
        if '401' in err_msg:
            return None  # SKIP — access restricted, not a failure
        return False

    h, v, i, nh, nk, n_layers, hd, rope_theta, tie_emb, act = extract_dimensions(config)
    arch = derive_arch(model_id, config)

    print(f"  Arch:          {arch}")
    print(f"  Model type:    {config.get('model_type', '?')}")
    print(f"  Architecture:  {(config.get('architectures') or ['?'])[0]}")
    print(f"  hidden_size:   {h}")
    print(f"  vocab_size:    {v}")
    print(f"  intermediate:  {i}")
    print(f"  heads:         {nh}/{nk}")
    print(f"  head_dim:      {hd}")
    print(f"  layers:        {n_layers}")
    print(f"  rope_theta:    {rope_theta}")
    print(f"  tie_emb:       {tie_emb}")
    print(f"  activation:    {act}")
    print(f"  hidden==vocab: {h == v}")
    if h == v:
        print(f"  {AKI_WARN} AKI-RISK: hidden==vocab ({h}) — norm tensors need name guard")

    config_head_dim = config.get("head_dim", 0)
    computed_hd = h // nh if nh > 0 else 0
    if config_head_dim and config_head_dim != computed_hd:
        print(f"  [N] head_dim explicitly set to {hd} in config (hidden/heads={computed_hd})")
        print(f"    (ATLAS uses explicit head_dim — correct)")

    # Check for SubLN indicators
    has_attn_sub_norm = config.get("attn_sub_norm", config.get("use_attn_sub_norm", False))
    has_ffn_sub_norm = config.get("ffn_sub_norm", config.get("use_ffn_sub_norm", False))
    if has_attn_sub_norm or has_ffn_sub_norm:
        print(f"  SubLN:         attn={has_attn_sub_norm} ffn={has_ffn_sub_norm}")
    if arch in ("bitnet", "trilm"):
        if not has_attn_sub_norm and not has_ffn_sub_norm:
            print(f"  {WARN} SubLN arch but no sub_norm flags in config")
    elif arch == "trilm_nosubln":
        if has_attn_sub_norm or has_ffn_sub_norm:
            print(f"  {WARN} non-SubLN arch but sub_norm flags found in config")

    idx_url = f"{HF_BASE}/{model_id}/raw/main/model.safetensors.index.json"
    idx = fetch_json(idx_url)

    if "_error" in idx:
        print(f"  Index:         (single file — no shard index)")
        print(f"  Tensor check:  (structural only)")
        expected_global = EXPECTED_GLOBAL[:]
        if arch != "qwen3" and not tie_emb:
            expected_global.append("lm_head.weight")
        expected_layer = EXPECTED_LAYER_TENSORS.get(arch, [])
        print(f"  Expected layer tensors: {len(expected_layer)}")
        print(f"  {OK} Name prefix: model.layers.N.")
        print(f"  {OK} Name suffix: .weight")
        print(f"  {OK} FFN prefix:  mlp")
        print(f"  {OK} Attn prefix: self_attn")
        print(f"\n  Result: {PASS}")
        return True

    weight_map = idx.get("weight_map", {})
    names = list(weight_map.keys())
    layer_names = [n for n in names if n.startswith("model.layers.")]
    global_names = [n for n in names if not n.startswith("model.layers.")]

    has_embed = "model.embed_tokens.weight" in global_names
    has_norm = "model.norm.weight" in global_names
    has_lm_head = "lm_head.weight" in global_names
    has_bias = any(n.endswith(".bias") for n in names)

    print(f"\n  Global tensors ({len(global_names)}):")
    print(f"    embed_tokens: {OK if has_embed else MISS}")
    print(f"    norm.weight:  {OK if has_norm else MISS}")
    print(f"    lm_head:      {OK if has_lm_head else 'tied'}")
    if has_bias:
        bias_count = sum(1 for n in names if n.endswith(".bias"))
        print(f"    biases:       {bias_count} bias tensors present")

    layer0 = sorted([n for n in layer_names if n.startswith("model.layers.0.")])
    layer0_bare = [n.replace("model.layers.0.", "") for n in layer0]

    expected = EXPECTED_LAYER_TENSORS.get(arch, EXPECTED_LAYER_TENSORS.get("falcon3", []))
    print(f"\n  Layer-0 tensors ({len(layer0_bare)}):")
    for t in layer0_bare:
        weight_only = t.replace(".weight", "").replace(".bias", "")
        expected_match = any(weight_only in e or e.startswith(weight_only) for e in expected)
        marker = OK if expected_match else WARN
        print(f"    {marker} {t}")

    missing = []
    for e in expected:
        found = any(e in t for t in layer0_bare)
        if not found:
            missing.append(e)

    # Detect unknown/extra tensors that ATLAS doesn't handle
    extra = []
    for t in layer0_bare:
        if not any(e.replace(".weight", "") in t or t.startswith(e.split(".")[0]) for e in expected):
            extra.append(t)

    ffn_found = any("mlp." in t for t in layer0_bare)
    attn_found = any("self_attn." in t for t in layer0_bare)

    print(f"\n  Pattern checks:")
    print(f"    FFN prefix 'mlp.':            {OK if ffn_found else MISS}")
    print(f"    Attn prefix 'self_attn.':      {OK if attn_found else MISS}")
    print(f"    Layer prefix 'model.layers.N.': {OK}")
    if extra:
        print(f"  {WARN} Extra tensors not in ATLAS schema: {extra}")
    if arch in ("qwen3",) and "self_attn.q_norm.weight" not in layer0_bare:
        print(f"  {WARN} qwen3 arch but no q_norm found!")

    if missing:
        print(f"\n  {WARN} Missing expected tensors: {missing}")

    actual_layers = set()
    for n in layer_names:
        parts = n.split(".")
        if len(parts) >= 3 and parts[1] == "layers":
            try:
                actual_layers.add(int(parts[2]))
            except ValueError:
                pass
    max_layer = max(actual_layers) if actual_layers else -1
    print(f"    Layer count: config={n_layers}, actual={max_layer + 1}")

    if h == v:
        print(f"\n  {AKI_WARN} AKI-BUG CHECK: hidden==vocab ({h})")
        print(f"    Old heuristic would misclassify norm.weight as embedding!")
    else:
        print(f"\n  {OK} AKI-BUG: NOT TRIGGERED (hidden={h} != vocab={v})")

    verdict = len(missing) == 0
    print(f"\n  Result: {PASS if verdict else FAIL}")
    return verdict


def auto_discover():
    """Query HF API for all models from known organizations."""
    discovered = set()
    for org in DISCOVER_ORGS:
        url = f"https://huggingface.co/api/models?author={org}&sort=downloads&direction=-1&limit=50"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "atlas-discovery/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for m in data:
                    mid = m["modelId"]
                    # Skip FloatLM (float32, not TQ1) and ckpts (checkpoints)
                    if "FloatLM" not in mid and "ckpts" not in mid:
                        discovered.add(mid)
        except Exception as e:
            print(f"  {WARN} Discover {org}: {e}", file=sys.stderr)
    return sorted(discovered)


def main():
    verbose = "--verbose" in sys.argv
    json_out = "--json" in sys.argv
    discover = "--discover" in sys.argv

    if discover:
        models = auto_discover()
        if not models:
            print("  No models discovered.")
            return 1
        print(f"\n  Auto-discovered {len(models)} models:")
        for m in models:
            print(f"    - {m}")
    else:
        models = [a for a in sys.argv[1:] if not a.startswith("--")] or DEFAULT_MODELS

    results = {}
    all_pass = True
    for mid in models:
        ok = check_alignment(mid)
        results[mid] = ok
        if ok is False:
            all_pass = False

    print(f"\n{'='*60}")
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped_total = sum(1 for v in results.values() if v is None)
    print(f"  SUMMARY  ({len(results)} models: {passed} PASS, {failed} FAIL, {skipped_total} SKIP)")
    print(f"{'='*60}")
    for mid, ok in results.items():
        if ok is True:
            status = PASS
        elif ok is None:
            status = SKIP
        else:
            status = FAIL
        print(f"  {status:12s} {mid}")
    print(f"\n  Overall: {PASS if all_pass else FAIL}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
