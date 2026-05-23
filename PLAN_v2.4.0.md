# ATLAS v2.4.0 — Qwen/Bonsai-Ökosystem-Upgrade

## Ziel

Bonsai-8B (Qwen-2.5 Topologie) in TQ1.0 format packen und auf ATLAS inferieren.
~2.6 GB, 8B Parameter, 151.936 Vocab, RoPE Base 1M, SwiGLU FFN.

## Teil 1: Packer (`atlas_packer_qwen.py`)

### Tensor-Mapping (Qwen → ATLAS)

| Qwen Name | ATLAS Rolle |
|---|---|
| `model.layers.N.input_layernorm.weight` | `input_layernorm` (RMSNorm) |
| `model.layers.N.self_attn.q_proj.weight` | Q projection |
| `model.layers.N.self_attn.k_proj.weight` | K projection |
| `model.layers.N.self_attn.v_proj.weight` | V projection |
| `model.layers.N.self_attn.o_proj.weight` | O projection |
| `model.layers.N.post_attention_layernorm.weight` | post_attention_layernorm |
| `model.layers.N.mlp.gate_proj.weight` | gate projection (SwiGLU) |
| `model.layers.N.mlp.up_proj.weight` | up projection (SwiGLU) |
| `model.layers.N.mlp.down_proj.weight` | down projection (SwiGLU) |
| `model.embed_tokens.weight` | embed_tokens |
| `model.norm.weight` | final norm |
| `lm_head.weight` | lm_head |

### Skalierungsfaktor

Jeder Tensor enthält ternär-quantisierte Werte im FP16/BF16 Master:
- `scale = max(abs(weights))` pro Tensor
- `ternary_weight = round(w / scale)` → exakt -1, 0, +1
- Scale im Tensor-Header für SIMD-Dequantisierung

### 5-Trit-Packing

Gleiches Verfahren wie `atlas_packer.py`: 5 Ternary-Werte → 1 Byte (Base-3, 3^5 = 243 Zustände).

## Teil 2: C++ Engine (`atlas_api.cpp`)

### Dynamisches Vocab

- `vocab_size` aus Datei-Header statt hardcoded 131072
- Betrifft: `lm_head` Allokation, `embed_tokens` Allokation, Sampling-Loop-Grenzen
- Bonsai: 151.936 Tokens

### SwiGLU-Hotpath

Aktuelle FFN-Struktur (Falcon3):
```
hidden → gate (SiLU) * up → down → output
```

Bereits fusioniert in `forward_layer_internal`. SwiGLU-Struktur ist identisch — kein Umbau nötig. Gate/Up werden parallel berechnet, SiLU fusioniert, Down-Projektor logisch getrennt.

### RoPE Base 1M

- `rope_theta` aus Datei-Header lesen (statt 1000042)
- Qwen: `rope_theta = 1000000`
- Kein struktureller Umbau — Parameter-Swap in der C++ RoPE-Funktion

## Abhängigkeiten

- Int8 KV-Cache (v2.3.0) ✅ vorhanden
- Dynamisches Vocab-Handling ⬜ neu
- Qwen-Packer ⬜ neu
- RoPE Base Parameter ⬜ minimal

## Target

Bonsai-8B TQ1.0: ~2.6 GB at 5 trits/byte (8B * 1.58 bits / 8 = ~1.58 GB reine Gewichte + Overhead).
