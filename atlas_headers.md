# Atlas TQ1_0 — Tatsächliche Header & Offsets (validiert)

## TQ1_0 Weight-Datei (*.tq10)

```
Offset  Größe  Typ    Feld
────────────────────────────────────
 0       4     i32    rows         (output_dim)
 4       4     i32    cols         (input_dim)
 8       4     i32    group_size   (laut extract: = rows → 1 Gruppe)
12       4     i32    num_groups   (= 1 bei BitNet b1.58)
16       4     f32    scale[0]     (erster/gleich einziger Scale)
20       N     u8[]   data         (rows × packed_cols, packed_cols = (cols+3)/4)

Gesamt-Header: 20 Bytes (KEIN Magic, KEIN Version-Feld)
```

### Packung (4er 2-bit, NICHT 5er base-3!)
```
Byte = v0 | v1<<2 | v2<<4 | v3<<6
v ∈ {0,1,2} → ternary {0, 1, -1}
```
Entpackt im C++ Code (`mv_tq10_scales`):
```
p0 = byte & 3          → {0,1,2,3}
r0 = (p0 & 1) - (p0>>1) → {0,1,-1,0}
```
- code 0 → 0
- code 1 → 1  
- code 2 → -1
- code 3 → 0 (sollte nie vorkommen)

---

## FP32 Vektor/Matrix-Datei (*.bin)

### Vektor (z.B. `l0_input_layernorm.bin`)
```
Offset  Größe  Typ   Feld
────────────────────────────
 0       4     i32   rows = 1
 4       4     i32   n    (Länge)
 8       4*n   f32   data[n]

Gesamt: 8 + 4*n Bytes
```

### Matrix (z.B. `embed.bin`)
```
Offset  Größe  Typ    Feld
─────────────────────────────
 0       4     i32    rows
 4       4     i32    cols
 8       4*N   f32    data[rows][cols]  (row-major)

Gesamt: 8 + 4*rows*cols Bytes
```

---

## BitNet b1.58 2B4T — Modell-Dimensionen

| Parameter | Wert |
|-----------|------|
| hidden_size | 2560 |
| intermediate_size | 6912 |
| num_attention_heads | 20 |
| num_key_value_heads | 5 |
| head_dim | 128 |
| num_hidden_layers | 30 |
| vocab_size | 128256 |
| max_seq_len | 4096 |
| rms_norm_eps | 1e-5 |
| rope_theta | 500000.0 |
| hidden_act | relu2 |
| tie_word_embeddings | true |

### BitNet — Weight-Matrizen pro Layer (7 Stück)

| Name | Shape | Size (TQ1_0) |
|------|-------|-------------|
| `q_proj` | [2560, 2560] | 6.6 MB |
| `k_proj` | [640, 2560] | 1.6 MB |
| `v_proj` | [640, 2560] | 1.6 MB |
| `o_proj` | [2560, 2560] | 6.6 MB |
| `gate_proj` | [6912, 2560] | 17.7 MB |
| `up_proj` | [6912, 2560] | 17.7 MB |
| `down_proj` | [2560, 6912] | 17.7 MB |

Typische scales: 0.96–2.30

### BitNet — Norm-Vektoren pro Layer (4 Stück)
- `input_layernorm` [2560]
- `post_attention_layernorm` [2560]
- `attn_sub_norm` [2560]  (aka inner_attn_ln)
- `ffn_sub_norm` [6912]

### BitNet — Global
- `embed` [128256, 2560] = 1.31 GB (FP32)
- `final_norm` [2560]

---

## Granite 3.0 2B — Modell-Dimensionen

| Parameter | Wert |
|-----------|------|
| hidden_size | 2048 |
| intermediate_size | 8192 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 |
| head_dim | 64 |
| num_hidden_layers | 40 |
| vocab_size | 49155 |
| max_seq_len | 4096 |
| rms_norm_eps | 1e-5 |
| rope_theta | 10000 |
| hidden_act | silu (SwiGLU) |
| tie_word_embeddings | true |

### Granite — Custom Scaling Multipliers
| Multiplier | Wert | Anwendung |
|-----------|------|-----------|
| `embedding_multiplier` | 12.0 | embed_out *= embed_mult |
| `attention_multiplier` | 0.015625 | attn_out *= attn_mult (≈1/64) |
| `residual_multiplier` | 0.22 | hidden = resid_mult * (hidden + layer_out) |
| `logits_scaling` | 8.0 | logits *= logit_scale |

### Granite — Weight-Matrizen pro Layer (7 Stück)

| Name | Shape | Size (TQ1_0) |
|------|-------|-------------|
| `q_proj` | [2048, 2048] | 1.0 MB |
| `k_proj` | [512, 2048] | 0.25 MB |
| `v_proj` | [512, 2048] | 0.25 MB |
| `o_proj` | [2048, 2048] | 1.0 MB |
| `gate_proj` | [8192, 2048] | 4.0 MB |
| `up_proj` | [8192, 2048] | 4.0 MB |
| `down_proj` | [2048, 8192] | 4.0 MB |

Typische scales (MSE-optimized): 0.012–0.025
Typische Sparsity: 47–53%

### Granite — Norm-Vektoren pro Layer (2 Stück)
- `input_layernorm` [2048]
- `post_attention_layernorm` [2048]

### Granite — Global
- `embed` [49155, 2048] = 384 MB (FP32)
- `final_norm` [2048]
- `lm_head` [49155, 2048] = 384 MB (FP32, tied → embed only)

### Granite — Order of Operations (Wichtig für C++ Engine!)
```
# Embedding
x = embed[token_ids] * embedding_multiplier            # *12

# Per Layer
x_residual = x
x = rms_norm(x)
x = attention(x)                                        # QKV → attn_out
x = attn_sub_norm(x)                                    # NOT present in Granite!
x = x_residual + x * attention_multiplier                # *0.015625 am Attn-Output

x_residual = x
x = rms_norm(x)
gate = silu(linear(x, gate_proj))                       # SwiGLU statt ReLU²
up   = linear(x, up_proj)
x   = gate * up
x   = linear(x, down_proj)
x = x_residual + x * residual_multiplier                 # *0.22 am MLP-Output

# Final
x = rms_norm(x)
logits = linear(x, lm_head) * logits_scaling             # lm_head = embed (tied)
```

---

## Binary Protocol (C++ ↔ Test Script)

```
ENGINE liest von stdin:

[int32: prompt_len]
[int32: token_id] × prompt_len
[int32: token_id] × MAX_GEN  (loopback der generierten Token)

ENGINE schreibt auf stdout:

[int32: token_id] × MAX_GEN  (generierte Token)
```

EOS = 128001 terminiert die Generation.
