"""Architecture definitions for the unified ATLAS packer (pack_to_atlas.py).

Each entry defines a model architecture: tensor naming, quantization rules,
and header flags. Auto-detected from config.json model_type.
"""

# ─── Falcon3 (stride=9, ttype=0 uint8 pre-packed ternaries) ─────────────
FALCON3 = {
    "model_types": {"falcon3"},
    "layer_tensors": [
        "model.layers.{}.input_layernorm.weight",
        "model.layers.{}.self_attn.q_proj.weight",
        "model.layers.{}.self_attn.k_proj.weight",
        "model.layers.{}.self_attn.v_proj.weight",
        "model.layers.{}.self_attn.o_proj.weight",
        "model.layers.{}.post_attention_layernorm.weight",
        "model.layers.{}.mlp.gate_proj.weight",
        "model.layers.{}.mlp.up_proj.weight",
        "model.layers.{}.mlp.down_proj.weight",
    ],
    "stride": 9,
    "global_tensors": ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"],
    "conditional_layer": [],
    "input_format": "uint8_packed",
    "requires_pre_shuffle": False,
    "quant_weight": "tq1_repack",
    "has_weight_scale": True,
    "flags_fn": lambda cfg: 0,
}

# ─── Qwen3/Bonsai (stride=11, ttype=5 g128 block-scaled) ───────────────
QWEN3 = {
    "model_types": {"qwen3", "qwen2", "qwen25"},
    "layer_tensors": [
        "model.layers.{}.input_layernorm.weight",
        "model.layers.{}.self_attn.q_proj.weight",
        "model.layers.{}.self_attn.k_proj.weight",
        "model.layers.{}.self_attn.v_proj.weight",
        "model.layers.{}.self_attn.o_proj.weight",
        "model.layers.{}.self_attn.q_norm.weight",
        "model.layers.{}.self_attn.k_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight",
        "model.layers.{}.mlp.gate_proj.weight",
        "model.layers.{}.mlp.up_proj.weight",
        "model.layers.{}.mlp.down_proj.weight",
    ],
    "stride": 11,
    "global_tensors": ["model.embed_tokens.weight", "model.norm.weight"],
    "conditional_layer": ["lm_head.weight"],
    "input_format": "bf16",
    "requires_pre_shuffle": True,
    "quant_weight": "tq1_block_scaled",
    "has_weight_scale": False,
    "flags_fn": lambda cfg: (
        (1 if cfg.get("model_type", "") in ("qwen2", "qwen3") else 0) << 0 |
        (1 if cfg.get("tie_word_embeddings", False) else 0) << 1 |
        (1 if cfg.get("enable_thinking", False) else 0) << 2 |
        (1 if cfg.get("hidden_act", "silu") == "relu2" else 0) << 3
    ),
}

# ─── BitNet b1.58 (stride=11, ttype=5 per-tensor absmean) ──────────────
BITNET = {
    "model_types": {"bitnet", "bitnet_b1_58"},
    "layer_tensors": [
        "model.layers.{}.input_layernorm.weight",
        "model.layers.{}.self_attn.q_proj.weight",
        "model.layers.{}.self_attn.k_proj.weight",
        "model.layers.{}.self_attn.v_proj.weight",
        "model.layers.{}.self_attn.o_proj.weight",
        "model.layers.{}.post_attention_layernorm.weight",
        "model.layers.{}.mlp.gate_proj.weight",
        "model.layers.{}.mlp.up_proj.weight",
        "model.layers.{}.mlp.down_proj.weight",
        "model.layers.{}.self_attn.attn_sub_norm.weight",
        "model.layers.{}.mlp.ffn_sub_norm.weight",
    ],
    "stride": 11,
    "global_tensors": ["model.embed_tokens.weight", "model.norm.weight"],
    "conditional_layer": ["lm_head.weight"],
    "input_format": "auto",
    "requires_pre_shuffle": True,
    "quant_weight": "tq1_absmean",
    "has_weight_scale": True,
    "flags_fn": lambda cfg: 1 << 3,
}

# ─── TriLM ≤2.4B (stride=11 SubLN, head_dim=64) ────────────────────────
TRILM = {
    "model_types": {"trilm"},
    "layer_tensors": list(BITNET["layer_tensors"]),  # same SubLN layout
    "stride": 11,
    "global_tensors": ["model.embed_tokens.weight", "model.norm.weight"],
    "conditional_layer": ["lm_head.weight"],
    "input_format": "bf16",
    "requires_pre_shuffle": True,
    "quant_weight": "tq1_absmean",
    "has_weight_scale": False,
    "flags_fn": lambda cfg: 0,
}

# ─── TriLM 3.9B / Llama-style (stride=9, NO SubLN, head_dim=128) ──────
TRILM_NOSUBLN = {
    "model_types": {"trilm_nosubln"},
    "layer_tensors": list(FALCON3["layer_tensors"]),  # same basic layout
    "stride": 9,
    "global_tensors": ["model.embed_tokens.weight", "model.norm.weight"],
    "conditional_layer": ["lm_head.weight"],
    "input_format": "bf16",
    "requires_pre_shuffle": True,
    "quant_weight": "tq1_block_scaled",
    "has_weight_scale": False,
    "flags_fn": lambda cfg: 0,
}

# ─── Generic Llama (stride=9, no extras) ────────────────────────────────
LLAMA = {
    "model_types": {"llama", "mistral", "gemma", "gemma2", "starcoder2", "phi3", "phi4"},
    "layer_tensors": list(FALCON3["layer_tensors"]),
    "stride": 9,
    "global_tensors": ["model.embed_tokens.weight", "model.norm.weight"],
    "conditional_layer": ["lm_head.weight"],
    "input_format": "bf16",
    "requires_pre_shuffle": True,
    "quant_weight": "tq1_block_scaled",
    "has_weight_scale": False,
    "flags_fn": lambda cfg: 0,
}

# ─── Registry: model_type → arch entry ──────────────────────────────────
ARCH_REGISTRY = {}
for _entry in [FALCON3, QWEN3, BITNET, TRILM, TRILM_NOSUBLN, LLAMA]:
    for _mt in _entry["model_types"]:
        ARCH_REGISTRY[_mt] = _entry


def detect_arch(config, available_tensors=()):
    """Auto-detect architecture from config.json and available tensor names.

    Returns an arch entry dict. Falls back to heuristics if model_type
    is unknown.
    """
    mt = config.get("model_type", "")

    # Falcon3 models from TII ship with model_type="llama" but have
    # uint8 I2_S weights, head_dim=256, rope_theta=1000042.
    # Route these to the Falcon3 entry (uint8 repack, NOT re-quantize).
    if mt == "llama" and config.get("head_dim") == 256 and config.get("rope_theta") == 1000042:
        print(f"  (detected Falcon3 variant via head_dim=256, rope_theta=1000042)")
        return FALCON3

    # Llama models with uint8 I2_S weights (e.g. HF1BitLLM/Llama3-8B-1.58-100B-tokens)
    # have weight_scale tensors and U8 weight dtype. Route to Falcon3 uint8 repack path.
    # TII Falcon3 (head_dim=256) is caught by the check above; remaining Llama I2_S models
    # use Microsoft bit order (sub-row 0 in high bits) — set invert flag.
    if mt in ARCH_REGISTRY:
        entry = ARCH_REGISTRY[mt]
        # Standard Llama expects BF16; if weight_scale tensors exist, this is I2_S format
        if entry.get("input_format") == "bf16" and not entry.get("has_weight_scale"):
            has_ws = any(t.endswith("weight_scale") for t in available_tensors)
            if has_ws:
                invert = config.get("is_bitnet_config", False)
                print(f"  (detected I2_S variant: {mt} with weight_scale tensors, using uint8 repack, invert_subrows={invert})")
                return {**FALCON3, "sub_row_bit_invert": invert}
        return entry

    # Heuristic fallback: check for SubLN tensors
    has_sub_ln = any("attn_sub_norm" in t or "ffn_sub_norm" in t for t in available_tensors)
    has_qk_norm = any("q_norm" in t or "k_norm" in t for t in available_tensors)
    head_dim = config.get("head_dim", config.get("hidden_size", 0) // max(config.get("num_attention_heads", 1), 1))

    if has_sub_ln:
        return TRILM
    if has_qk_norm:
        return QWEN3

    # Check if TriLM large (head_dim >= 128 → no SubLN)
    if mt.startswith("trilm") or mt == "trilm_nosubln":
        return TRILM_NOSUBLN if head_dim >= 128 else TRILM

    print(f"  WARNING: unknown model_type={mt!r}, falling back to Llama-like stride=9")
    return LLAMA
