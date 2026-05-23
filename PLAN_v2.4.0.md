# ATLAS v2.4.0 — Qwen3/Bonsai-Ökosystem-Upgrade

## Ziel

Bonsai-4B (Qwen3 Topologie, 36 Layers) in TQ1.0 format packen und auf ATLAS inferieren.
~2.6 GB, 4B Parameter, 151.669 Vocab, RoPE 5M+YaRN, SwiGLU FFN, QK-Norm.

## Architektur (Bonsai-4B)

| Parameter | Bonsai-4B | Falcon3-3B (Referenz) |
|---|---|---|
| hidden_size | 2560 | 3072 |
| intermediate_size | 9728 | 9216 |
| num_hidden_layers | 36 | 22 |
| num_attention_heads | 32 | 12 |
| num_key_value_heads | 8 | 4 |
| head_dim | **128** | **256** |
| vocab_size | 151669 | 131072 |
| rope_theta | 5,000,000 (YaRN) | 1,000,042 |
| max_seq_len | 32,768 | 4,096 |
| hidden_act | silu (SwiGLU) | silu (SwiGLU) |
| rms_norm_eps | 1e-6 | 1e-6 |
| tie_word_embeddings | **True** (no lm_head) | False |
| QK-Norm | **Ja** (q_norm, k_norm per layer) | Nein |
| EOS/PAD | 151645 / 151643 | 0 / 0 |

## Teil 1: Packer (`atlas_packer_qwen.py`)

### Tensor-Mapping (Qwen3 → ATLAS)

| Qwen3 Name | ATLAS Rolle | Notiz |
|---|---|---|
| `model.layers.N.input_layernorm.weight` | input_layernorm | RMSNorm |
| `model.layers.N.self_attn.q_proj.weight` | Q | GQA, 32 heads |
| `model.layers.N.self_attn.k_proj.weight` | K | GQA, 8 heads |
| `model.layers.N.self_attn.v_proj.weight` | V | GQA, 8 heads |
| `model.layers.N.self_attn.o_proj.weight` | O | |
| `model.layers.N.self_attn.q_norm.weight` | **neues Tensor** | QK-Norm (RMSNorm) |
| `model.layers.N.self_attn.k_norm.weight` | **neues Tensor** | QK-Norm (RMSNorm) |
| `model.layers.N.post_attention_layernorm.weight` | post_attention_layernorm | RMSNorm |
| `model.layers.N.mlp.gate_proj.weight` | gate | SwiGLU |
| `model.layers.N.mlp.up_proj.weight` | up | SwiGLU |
| `model.layers.N.mlp.down_proj.weight` | down | |
| `model.embed_tokens.weight` | embed_tokens | Auch lm_head (tied) |
| `model.norm.weight` | final norm | RMSNorm |

### Skalierungsfaktor + Packing

Wie PLAN: `scale = max(abs(w))`, `round(w/scale)`, 5-Trit-Packing.

**Wichtig**: `tie_word_embeddings=True` → `lm_head` existiert nicht als separater Tensor. `embed_tokens.weight` dient als Klassifikationskopf. ATLAS muss das erkennen und `embed_tokens` duplizieren oder als lm_head referenzieren.

## Teil 2: C++ Engine (`atlas_api.cpp`)

### head_dim = 128 (KRITISCH)

Falcon3 nutzt head_dim=256. Alle Attention-Pfade müssen auf 128 umgestellt werden:
- RoPE: Rotations-Paare halbieren sich (head_dim/2 = 64 statt 128)
- Attention Scores: `(B×n_heads×seq) × head_dim` Matrix
- Weighted Sum: `(B×n_heads×head_dim) × seq`
- KV-Cache: `n_kv_heads × max_seq × head_dim × 2` (int8)

**Maßnahme**: `head_dim` aus Datei-Header lesen, alle Attention-Puffer danach dimensionieren.

### QK-Norm (NEU)

Nach RoPE und vor Attention-Score-Berechnung:
```cpp
// Apply QK-norm after RoPE
atlas_rmsnorm_f32(q, m->q_norm[layer], q, n_heads * head_dim, rms_norm_eps);
atlas_rmsnorm_f32(k, m->k_norm[layer], k, n_kv_heads * head_dim, rms_norm_eps);
```

**Maßnahme**: Zwei neue RMSNorm-Gewichts-Tensoren pro Layer laden (`q_norm.weight`, `k_norm.weight`).

### Dynamisches Vocab

- `vocab_size = 151669` aus Header
- `embed_tokens` auf vocabsize × hidden_dim allozieren
- Sampling auf vocabsize begrenzen (Obergrenze für top-k)
- EOS/PAD IDs aus Header statt hardcoded 0

### YaRN RoPE (KOMPLEX)

Qwen3 nutzt **YaRN** (Yet another RoPE scaling method):
```python
rope_theta = 5000000.0
scale_factor = 4.0
original_max = 8192
max_position = 32768
```

YaRN modifiziert die Frequenzen:
- High frequencies beibehalten (kurze Distanzen)
- Low frequencies interpolieren (lange Distanzen)
- Rampenfunktion zwischen den Bändern

**Vereinfachter Ansatz**: NTK-aware scaling als erste Approximation:
```cpp
float base = rope_theta;
float scale = powf(base / (base * powf(scale_factor, head_dim/(head_dim-2))), head_dim/(head_dim-2));
```

Alternativ: Nur NTK-scaling ohne YaRN-Rampe — ~95% der Qualität, deutlich simpler.

### SwiGLU-Hotpath

FFN-Struktur identisch mit Falcon3 (gate+up parallel, SiLU fusioniert). Kein Umbau nötig.

### Tie Word Embeddings

`lm_head` existiert nicht. `embed_tokens.weight` wird für Logit-Berechnung verwendet:
```cpp
// lm_head: reuse embed_tokens weights
int idx_embed = get_tensor_index("model.embed_tokens.weight");
Tensor* embed = &m->tensors[idx_embed];
// embed->data IS the lm_head weights
```

**Int8-Quantisierung**: `embed_tokens` muss int8-quantisiert werden wie lm_head.

## Summary

| Änderung | Komplexität | Status |
|---|---|---|
| head_dim=128 | **HOCH** (alle Attention-Pfade) | ⬜ |
| QK-Norm | **MITTEL** (2 neue Tensoren/Layer) | ⬜ |
| Vocab 151669 + EOS/PAD | **NIEDRIG** (Header-Parameter) | ⬜ |
| YaRN RoPE | **HOCH** (Frequenz-Skalierung) | ⬜ |
| Tie embeddings | **MITTEL** (lm_head = embed) | ⬜ |
| Packer (Qwen3 Mapping) | **MITTEL** (neues Skript) | ⬜ |
| SwiGLU | **KEINE** (identisch) | ✅ |
| Int8 KV-Cache | **KEINE** (v2.3.0, head_dim-agnostic) | ✅ |
