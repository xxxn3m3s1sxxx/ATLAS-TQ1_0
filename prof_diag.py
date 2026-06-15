#!/usr/bin/env python3
"""Diagnostic profiler: measure per-operation timing for 4B Pro vs 1.7B Pro."""
import os, sys, time, json, subprocess, re, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas_infer import AtlasModel, dll

PROMPT = "The capital of France is"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def load_and_profile(path, label="model", use_f32=False, max_new=100):
    print(f"\n{'='*60}", flush=True)
    print(f"[DIAG] Loading {label}", flush=True)
    print(f"       Path: {path}", flush=True)
    print(f"       f32_bypass forced: {use_f32}", flush=True)
    print(f"{'='*60}", flush=True)
    if not os.path.exists(path):
        print(f"[DIAG] SKIP — file not found: {path}", flush=True)
        return None
    t0 = time.time()
    m = AtlasModel(path, use_f32_matmul=use_f32, max_seq_len=4096)
    load_t = time.time() - t0
    print(f"[DIAG] Load time: {load_t:.1f}s", flush=True)
    # Redirect FD-level stderr to capture C profile output
    stderr_fd = os.dup(2)  # save original
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)
    t1 = time.time()
    out = m.generate_c(PROMPT, max_new_tokens=max_new, temperature=0.7, top_k=40)
    gen_t = time.time() - t1
    os.dup2(stderr_fd, 2)  # restore
    os.close(stderr_fd)
    stderr_text = os.read(r_fd, 65536).decode('utf-8', errors='replace')
    os.close(r_fd)
    # Parse: PROFILE (XXXX layers × YYY tokens, ZZZ.ZZZs total)
    m_profile = re.search(r'PROFILE\s*\((\d+)\s*layers\s*×\s*(\d+)\s*tokens', stderr_text)
    if m_profile:
        n_total = int(m_profile.group(2))
        n_gen = max(0, n_total - 1)  # subtract prefill lm_head
    else:
        n_gen = len(out) // 4  # fallback: rough char→token estimate
    tok_s = n_gen / gen_t if gen_t > 0 else 0
    print(f"[DIAG] Generated {n_gen} tokens in {gen_t:.1f}s = {tok_s:.1f} tok/s", flush=True)
    text = out
    print(f"[DIAG] Output: {text[:200]!r}", flush=True)
    return {"label": label, "path": path, "use_f32": use_f32,
            "load_t": load_t, "gen_t": gen_t, "n_tokens": n_gen,
            "tok_s": tok_s, "text": text[:200]}

results = []
# 1) 4B Pro hybrid (default path)
r = load_and_profile(
    os.path.join(MODELS_DIR, "Ternary-Bonsai-4B-ATLAS-Pro.atlas"),
    label="4B Pro (hybrid)", use_f32=False, max_new=100)
if r: results.append(r)

# 2) 4B Pro forced f32_bypass
r = load_and_profile(
    os.path.join(MODELS_DIR, "Ternary-Bonsai-4B-ATLAS-Pro.atlas"),
    label="4B Pro (f32 bypass)", use_f32=True, max_new=100)
if r: results.append(r)

# 3) 1.7B Pro hybrid (baseline)
for p in [os.path.join(MODELS_DIR, "Ternary-Bonsai-1.7B-ATLAS-Pro.atlas"),
          os.path.join(MODELS_DIR, "models", "Ternary-Bonsai-1.7B-ATLAS-Pro-v2.atlas")]:
    if os.path.exists(p):
        r = load_and_profile(p, label="1.7B Pro (hybrid)", use_f32=False, max_new=100)
        if r: results.append(r)
        break

# 4) 4B baseline (non-Pro) if available
for p in [os.path.join(MODELS_DIR, "models", "Ternary-Bonsai-4B-BASELINE.atlas"),
          os.path.join(MODELS_DIR, "Ternary-Bonsai-4B-BASELINE.atlas"),
          os.path.join(MODELS_DIR, "Ternary-Bonsai-4B-ATLAS.atlas")]:
    if os.path.exists(p):
        r = load_and_profile(p, label="4B baseline (hybrid)", use_f32=False, max_new=100)
        if r: results.append(r)
        break

# 5) 4B Pro v2 (if available in subfolder)
for p in [os.path.join(MODELS_DIR, "models", "Ternary-Bonsai-4B-ATLAS-Pro-v2.atlas")]:
    if os.path.exists(p):
        r = load_and_profile(p, label="4B Pro v2 (hybrid)", use_f32=False, max_new=100)
        if r: results.append(r)
        break

print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
for r in results:
    print(f"  {r['label']:30s}  {r['tok_s']:5.1f} tok/s  {r['n_tokens']:3d} tok  {r['text'][:80]}", flush=True)
print(flush=True)
print("[DIAG] Done — profile output above in stderr", flush=True)
