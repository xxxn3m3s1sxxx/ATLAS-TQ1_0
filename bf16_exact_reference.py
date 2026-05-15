"""bf16 exact reference — memory efficient. Process one layer at a time."""
import torch, numpy as np, time, gc
from safetensors.torch import load_file

MODEL = r'C:\dam\models\bitnet-b1.58-2B-4T'
HIDDEN, INTERMEDIATE = 2560, 6912
N_LAYERS, EPS = 30, 1e-3

def bf16(t): return t.float().numpy()

def unpack_bitnet_f32(u8):
    """Unpack BitNet uint8 to float32 {-1,0,1}. Returns (rows*4, cols) float32.
    Memory efficient: no intermediate int8 array."""
    rows = u8.shape[0] * 4; u = u8.numpy().astype(np.int8)
    v0 = ((u >> 0) & 3).astype(np.float32)
    v1 = ((u >> 2) & 3).astype(np.float32)
    v2 = ((u >> 4) & 3).astype(np.float32)
    v3 = ((u >> 6) & 3).astype(np.float32)
    # Map 0->-1, 1->0, 2->1: subtract 1
    v0 -= 1.0; v1 -= 1.0; v2 -= 1.0; v3 -= 1.0
    r = np.empty((rows, u8.shape[1]), dtype=np.float32)
    r[0::4] = v0; r[1::4] = v1; r[2::4] = v2; r[3::4] = v3
    return r

sd = load_file(f'{MODEL}/model.safetensors')

print("Loading embed + final_norm...")
embed = bf16(sd['model.embed_tokens.weight'])
fn = bf16(sd['model.norm.weight'])
del sd
gc.collect()

def rms_norm(x, w):
    ss = np.sum(x.astype(np.float64) ** 2)
    inv = 1.0 / np.sqrt(ss / len(x) + EPS)
    return (x * inv * w).astype(np.float32)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, fix_mistral_regex=True)
prompt = "What is the capital of France?"
ids = tok.encode(prompt)
print(f"Prompt ({len(ids)} tokens): {prompt}")

# Process first token (position 0)
x = embed[ids[0]].copy().astype(np.float32)

# Reload safetensors layer by layer
t0 = time.time()
for li in range(N_LAYERS):
    sd = load_file(f'{MODEL}/model.safetensors')
    
    l = {}
    for key, short in [("self_attn.q_proj","q"),("self_attn.k_proj","k"),("self_attn.v_proj","v"),
                       ("self_attn.o_proj","o"),("mlp.gate_proj","g"),("mlp.up_proj","u"),("mlp.down_proj","d")]:
        w = unpack_bitnet_f32(sd[f'model.layers.{li}.{key}.weight'])
        sc = sd[f'model.layers.{li}.{key}.weight_scale'].float().item()
        l[short] = (w, sc)
    for key, short in [("input_layernorm","ln1"),("post_attention_layernorm","ln2"),
                       ("self_attn.attn_sub_norm","attn_sub"),("mlp.ffn_sub_norm","ffn_sub")]:
        l[short] = bf16(sd[f'model.layers.{li}.{key}.weight'])
    del sd
    gc.collect()
    
    res = x.copy()
    x = rms_norm(x, l['ln1'])
    q = l['q'][0] @ x * l['q'][1]; del l['q']
    k = l['k'][0] @ x * l['k'][1]; del l['k']
    v = l['v'][0] @ x * l['v'][1]; del l['v']
    o = l['o'][0] @ q * l['o'][1]; del l['o']
    del q, k, v
    o = rms_norm(o, l['attn_sub']); del l['attn_sub']
    x = res + o; del res, o
    
    res = x.copy()
    x = rms_norm(x, l['ln2']); del l['ln2']
    gate = l['g'][0] @ x * l['g'][1]; del l['g']
    up = l['u'][0] @ x * l['u'][1]; del l['u']
    gate = np.where(gate > 0, gate * gate, 0.0) * up; del up
    gate = rms_norm(gate, l['ffn_sub']); del l['ffn_sub']
    d = l['d'][0] @ gate * l['d'][1]; del l['d'], gate
    x = res + d; del res, d
    
    log_norm = np.linalg.norm(x)
    log_std = x.std()
    if li < 3 or li == N_LAYERS - 1 or li % 5 == 0:
        print(f"  Layer {li:2d}: norm={log_norm:.4f} std={log_std:.4f}")

print(f"\nAfter {N_LAYERS} layers ({time.time()-t0:.1f}s):")
print(f"  std={x.std():.4f} mean={x.mean():.4f}")

x = rms_norm(x, fn)
logits = embed.astype(np.float32) @ x.astype(np.float32)
top5 = np.argsort(logits)[-5:][::-1]
print(f"\nTop-5 tokens:")
for t in top5:
    token = tok.convert_ids_to_tokens(int(t))
    safe = token.encode('unicode_escape').decode(errors='replace')
    print(f"  {t:6d} -> '{safe}' (logit={logits[t]:.4f})")

# Generate
print(f"\nGenerating 10 tokens...")
for step in range(10):
    x_final = rms_norm(x, fn)
    logits = embed.astype(np.float32) @ x_final.astype(np.float32)
    next_id = int(np.argmax(logits))
    token = tok.convert_ids_to_tokens(next_id)
    safe = token.encode('unicode_escape').decode(errors='replace')
    print(f"  {step}: {next_id:6d} -> '{safe}' (logit={logits[next_id]:.4f})")
    if next_id == 128001:
        print("  [EOS]"); break
    x = embed[next_id].copy().astype(np.float32)
    # Re-process from scratch for each new token (simplified, no KV cache)
    for li in range(N_LAYERS):
        sd = load_file(f'{MODEL}/model.safetensors')
        l = {}
        for key, short in [("self_attn.q_proj","q"),("self_attn.k_proj","k"),("self_attn.v_proj","v"),
                           ("self_attn.o_proj","o"),("mlp.gate_proj","g"),("mlp.up_proj","u"),("mlp.down_proj","d")]:
            w = unpack_bitnet_f32(sd[f'model.layers.{li}.{key}.weight'])
            sc = sd[f'model.layers.{li}.{key}.weight_scale'].float().item()
            l[short] = (w, sc)
        for key, short in [("input_layernorm","ln1"),("post_attention_layernorm","ln2"),
                           ("self_attn.attn_sub_norm","attn_sub"),("mlp.ffn_sub_norm","ffn_sub")]:
            l[short] = bf16(sd[f'model.layers.{li}.{key}.weight'])
        del sd; gc.collect()
        
        res = x.copy()
        x = rms_norm(x, l['ln1'])
        q = l['q'][0] @ x * l['q'][1]; k = l['k'][0] @ x * l['k'][1]; v = l['v'][0] @ x * l['v'][1]
        o = l['o'][0] @ q * l['o'][1]; o = rms_norm(o, l['attn_sub'])
        x = res + o
        res = x.copy()
        x = rms_norm(x, l['ln2'])
        gate = l['g'][0] @ x * l['g'][1]; up = l['u'][0] @ x * l['u'][1]
        gate = np.where(gate > 0, gate * gate, 0.0) * up
        gate = rms_norm(gate, l['ffn_sub'])
        d = l['d'][0] @ gate * l['d'][1]; x = res + d
