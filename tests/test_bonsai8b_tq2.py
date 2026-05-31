"""Bonsai-8B TQ2 Langlauf: 2000 Token mit Signalqualität."""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas_infer import AtlasModel

path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "bonsai-8b-tq1-g128-v8.atlas")

print(f"[LANGLAUF] Loading {path} with TQ2...")
t0 = time.time()
m = AtlasModel(path, max_seq_len=4096, convert_to_tq2=True)
t_load = time.time() - t0
print(f"[LANGLAUF] Load + TQ2 complete: {t_load:.1f}s")

prompt = "The capital of France is"
print(f"[LANGLAUF] Prompt: {prompt!r}")
print(f"[LANGLAUF] Generating 2000 tokens (T=0.7, top_k=40)...")

tokens_out = m.generate_c(
    prompt=prompt,
    max_new_tokens=2000,
    temperature=0.7,
    top_k=40,
)

if isinstance(tokens_out, str):
    text = tokens_out
else:
    text = tokens_out

t_gen = time.time() - t0 - t_load
n_tok = len(text.split())
tok_s = n_tok / t_gen if t_gen > 0 else 0

print(f"[LANGLAUF] Tokens: {n_tok} in {t_gen:.1f}s ({tok_s:.1f} tok/s)")

lines = text.strip().split("\n")
print(f"[LANGLAUF] Output lines: {len(lines)}")
for i, line in enumerate(lines[:20]):
    print(f"  [{i}] {line[:120]}")

if len(lines) > 20:
    print(f"  ... ({len(lines) - 20} more lines)")

last = text[-500:]
print(f"[LANGLAUF] Last 500 chars: {last!r}")
print(f"[LANGLAUF] Text length: {len(text)} chars")

oversized_ok = len(text) > 800
print(f"[LANGLAUF] Output >= 800 chars: {'OK' if oversized_ok else 'SHORT'}")

has_repeats = text.count("|") > 20
print(f"[LANGLAUF] Repeats (>20 pipe chars): {'DEGRADED' if has_repeats else 'OK'}")

if not oversized_ok:
    print("[LANGLAUF] FAILED: output too short (EOS early or collapse)")
    sys.exit(1)
if has_repeats and tok_s < 0.5:
    print("[LANGLAUF] WARNING: degradiert + sehr langsam")
elif has_repeats:
    print("[LANGLAUF] WARNING: Ausgabe degradiert (pipe-repetition)")
else:
    print("[LANGLAUF] PASSED: Signal stabil, keine Degradation")

print(f"\n[LANGLAUF] Gesamt: {t_load:.1f}s load + {t_gen:.1f}s gen = {time.time()-t0:.1f}s total")
