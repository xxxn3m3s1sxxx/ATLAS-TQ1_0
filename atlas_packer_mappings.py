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

# ─── MiniCPM (stride=9, Llama-like, scale_emb/scale_depth) ─────────────
MINICPM = {
    "model_types": {"minicpm"},
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

# ─── Falcon-E I2_S with per-channel weight_scale (stride=9, ttype=5) ──
# I2_S uint8 weights + per-channel weight_scale tensors.
# Block-scaled ttype=5 preserves per-row scales and multiplies (correct direction).
FALCON_E = {
    "model_types": {},
    "layer_tensors": list(FALCON3["layer_tensors"]),
    "stride": 9,
    "global_tensors": ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"],
    "conditional_layer": [],
    "input_format": "uint8_packed",
    "requires_pre_shuffle": True,
    "quant_weight": "tq1_block_scaled",
    "has_weight_scale": True,
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

# ─── DeepSeek-V2 / V2-Lite / V3 (MoE + MLA) ─────────────────────────────
# DeepSeek-V2 uses Multi-head Latent Attention (MLA) with joint KV compression
# and Mixture-of-Experts FFN. Layer 0 is dense; layers 1+ are MoE.
# V2-Lite: q_proj (no query compression), simple o_proj
# V3: q_a_proj + q_b_proj (query compression), gated o_proj (o_proj.0/1/2)
# MLA tensors: kv_a_proj_with_mqa, kv_b_proj
# MoE tensors: mlp.gate (router), mlp.experts.{E}.gate/up/down, shared_experts
DEEPSEEK_V2 = {
    "model_types": {"deepseek_v2", "deepseek_v3"},
    # Dense layer tensors (all layers have these)
    "layer_tensors": [
        "model.layers.{}.input_layernorm.weight",
        # MLA attention — V2-Lite uses q_proj, V3 uses q_a_proj (detected dynamically)
        "model.layers.{}.self_attn.q_proj.weight",
        "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.{}.self_attn.kv_a_layernorm.weight",
        "model.layers.{}.self_attn.kv_b_proj.weight",
        "model.layers.{}.self_attn.o_proj.weight",
        "model.layers.{}.post_attention_layernorm.weight",
    ],
    # V3 extra attention tensors (q_a, q_b, q_a_layernorm, gated o_proj)
    "layer_tensors_v3": [
        "model.layers.{}.self_attn.q_a_proj.weight",
        "model.layers.{}.self_attn.q_b_proj.weight",
        "model.layers.{}.self_attn.q_a_layernorm.weight",
        "model.layers.{}.self_attn.o_proj.0.weight",
        "model.layers.{}.self_attn.o_proj.1.weight",
        "model.layers.{}.self_attn.o_proj.2.weight",
    ],
    "stride": 7,  # base stride (dense attention tensors per layer)
    "global_tensors": ["model.embed_tokens.weight", "model.norm.weight"],
    "conditional_layer": ["lm_head.weight"],
    "input_format": "bf16",
    "requires_pre_shuffle": True,
    "quant_weight": "tq1_block_scaled",
    "has_weight_scale": False,
    # MoE-specific fields
    "is_moe": True,
    "moe_expert_patterns": [
        "model.layers.{}.mlp.experts.{}.gate_proj.weight",
        "model.layers.{}.mlp.experts.{}.up_proj.weight",
        "model.layers.{}.mlp.experts.{}.down_proj.weight",
    ],
    "moe_shared_patterns": [
        "model.layers.{}.mlp.shared_experts.gate_proj.weight",
        "model.layers.{}.mlp.shared_experts.up_proj.weight",
        "model.layers.{}.mlp.shared_experts.down_proj.weight",
    ],
    "moe_router_pattern": "model.layers.{}.mlp.gate.weight",
    "moe_dense_ffn_patterns": [
        "model.layers.{}.mlp.gate_proj.weight",
        "model.layers.{}.mlp.up_proj.weight",
        "model.layers.{}.mlp.down_proj.weight",
    ],
    "flags_fn": lambda cfg: 0,
}

# ─── Registry: model_type → arch entry ──────────────────────────────────
ARCH_REGISTRY = {}
for _entry in [FALCON3, QWEN3, BITNET, TRILM, TRILM_NOSUBLN, MINICPM, LLAMA, DEEPSEEK_V2]:
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
        # Llama models with SubLN tensors (attn_sub_norm, ffn_sub_norm) → TriLM
        if mt == "llama" and available_tensors:
            has_sub_ln = any("attn_sub_norm" in t or "ffn_sub_norm" in t for t in available_tensors)
            if has_sub_ln:
                print(f"  (detected TriLM SubLN variant via attn_sub_norm/ffn_sub_norm tensors)")
                return TRILM
        # Standard Llama expects BF16; if weight_scale tensors exist, this is I2_S format
        if entry.get("input_format") == "bf16" and not entry.get("has_weight_scale"):
            has_ws = any(t.endswith("weight_scale") for t in available_tensors)
            if has_ws:
                is_falcon_e = (config.get("quantization_config") or {}).get("quant_method") == "bitnet"
                if is_falcon_e:
                    print(f"  (detected Falcon-E: {mt} with TII bit order, no invert)")
                    return FALCON3
                else:
                    print(f"  (detected I2_S variant: {mt} with Microsoft bit order, invert)")
                    return {**FALCON3, "sub_row_bit_invert": True, "requires_pre_shuffle": True}
        return entry

    # MiniCPM detection: scale_depth/scale_emb/dim_model_base in config, or architectures
    if ("scale_depth" in config or "scale_emb" in config or "dim_model_base" in config
            or any("MiniCPMForCausalLM" in a for a in config.get("architectures", []))):
        print(f"  (detected MiniCPM via scale_depth/scale_emb/dim_model_base/architectures)")
        return MINICPM

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
