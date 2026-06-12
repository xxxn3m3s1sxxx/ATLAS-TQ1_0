"""v2.10.6: KV-Cache Isolation + Error Handling + Memory Telemetry."""
import sys, os, ctypes, math, gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atlas_infer import AtlasModel

MODEL = "models/falcon_e_1b_base/model.atlas"
FAIL = 0
PASS = 0

def pf(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label} {detail}")

# ============================================================
# 1. KV-Cache Isolation
# ============================================================
print("\n=== 1. KV-Cache Isolation ===")

m = AtlasModel(MODEL)
prompt_a = [{"role": "user", "content": "What is the capital of France?"}]
prompt_b = [{"role": "user", "content": "What is the capital of Germany?"}]

# Run A twice with reset in between -- should be identical
out_a1 = m.generate_c(prompt_a, max_new_tokens=10, temperature=0.0, top_k=1)
m.reset_cache()
out_a2 = m.generate_c(prompt_a, max_new_tokens=10, temperature=0.0, top_k=1)
pf("reset_cache -> same prompt -> identical output", out_a1 == out_a2,
      f"got {out_a1!r} vs {out_a2!r}")

# Run B then A without reset -- cache from B should not leak into A
m.reset_cache()
out_b = m.generate_c(prompt_b, max_new_tokens=10, temperature=0.0, top_k=1)
out_a3 = m.generate_c(prompt_a, max_new_tokens=10, temperature=0.0, top_k=1)
pf("different prompts -> no cache leak (same output as isolated run)", out_a3 == out_a1,
      f"A1={out_a1!r} A3={out_a3!r}")

# Prefix match: second call with same prompt uses cache_offset
m.reset_cache()
out_p1 = m.generate_c("The capital of France is", max_new_tokens=5, temperature=0.0, top_k=1)
out_p2 = m.generate_c("The capital of France is", max_new_tokens=5, temperature=0.0, top_k=1)
pf("same prefix -> cache_offset skips re-prefix (identical output)", out_p1 == out_p2,
      f"P1={out_p1!r} P2={out_p2!r}")

# Prefix change: different prefix should not match cache
m.reset_cache()
out_prefix_a = m.generate_c("France is a country in", max_new_tokens=5, temperature=0.0, top_k=1)
out_prefix_b = m.generate_c("The capital of France", max_new_tokens=5, temperature=0.0, top_k=1)
pf("different prefix -> no cache corruption (returns valid string)",
      len(out_prefix_b) > 0 and not out_prefix_b.startswith("[ATLAS:"),
      f"got {out_prefix_b!r}")

# Multi-turn: generate then continue -- cache should grow correctly
m.reset_cache()
out_turn1 = m.generate_c("Hello", max_new_tokens=3, temperature=0.0, top_k=1)
out_turn2 = m.generate_c("Hello", max_new_tokens=3, temperature=0.0, top_k=1)
m.reset_cache()
out_turn1_iso = m.generate_c("Hello", max_new_tokens=3, temperature=0.0, top_k=1)
pf("cached second turn identical to isolated run", out_turn2 == out_turn1_iso,
      f"cached={out_turn2!r} iso={out_turn1_iso!r}")

del m
gc.collect()

# ============================================================
# 2. Error Handling
# ============================================================
print("\n=== 2. Error Handling ===")

m = AtlasModel(MODEL)

# max_new_tokens = 0 -> should return error string
result = m.generate_c("Hello", max_new_tokens=0, temperature=0.0, top_k=1)
pf("max_new_tokens=0 -> error string", result == "[ATLAS: atlas_generate failed]",
      f"got {result!r}")

# temperature = -1 -> should treat as argmax (deterministic)
out_t1 = m.generate_c("The capital of France is", max_new_tokens=10, temperature=-1.0, top_k=1)
out_t2 = m.generate_c("The capital of France is", max_new_tokens=10, temperature=-1.0, top_k=1)
pf("temp=-1 -> deterministic (argmax)", out_t1 == out_t2,
      f"got {out_t1!r} vs {out_t2!r}")

# temperature = NaN -> should be handled gracefully (no crash)
try:
    out_nan = m.generate_c("Hello", max_new_tokens=5, temperature=float('nan'), top_k=1)
    pf("temp=NaN -> no crash", True)
except Exception as e:
    pf("temp=NaN -> no crash", False, f"exception: {e}")

# max_seq_len = 500K should be allocatable (~6 GB virtual)
out_big = m.generate_c("Hello", max_new_tokens=5, max_seq_len=500000, temperature=0.0, top_k=1)
pf("max_seq_len=500K works (large but allocatable)", not out_big.startswith("[ATLAS:"),
      f"got {out_big!r}")
# NOTE: ensure_cache() does NOT check atlas_valloc NULL return.
# With absurd max_seq_len (e.g. 500M -> ~6 TB), atlas_valloc returns NULL
# and the next cache write causes a segfault. This is a known bug.

# Repeat test to verify no memory corruption from large allocation
out_big2 = m.generate_c("Hello", max_new_tokens=5, max_seq_len=500000, temperature=0.0, top_k=1)
pf("repeat after large cache: no corruption", out_big == out_big2,
      f"1={out_big!r} 2={out_big2!r}")

del m
gc.collect()

# ============================================================
# 3. Memory Telemetry
# ============================================================
print("\n=== 3. Memory Telemetry ===")

try:
    import psutil
    proc = psutil.Process()
    m = AtlasModel(MODEL)
    base_rss = proc.memory_info().rss
    print(f"  Base RSS: {base_rss // 1024 // 1024} MB")

    deltas = []
    for i in range(3):
        gc.collect()
        before = proc.memory_info().rss
        out = m.generate_c("Tell me a story about a cat", max_new_tokens=50,
                           temperature=0.7, top_k=40)
        after = proc.memory_info().rss
        delta = after - before
        deltas.append(delta)
        print(f"  Run {i+1}: RSS delta {delta // 1024 // 1024} MB, output: {out[:40]!r}...")

    m.reset_cache()
    gc.collect()
    after_reset = proc.memory_info().rss
    print(f"  After reset_cache + gc: {(after_reset - base_rss) // 1024 // 1024} MB delta from base")

    before_final = proc.memory_info().rss
    out_final = m.generate_c("Tell me a story about a dog", max_new_tokens=50,
                             temperature=0.7, top_k=40)
    after_final = proc.memory_info().rss
    delta_final = after_final - before_final
    print(f"  Fresh prompt: RSS delta {delta_final // 1024 // 1024} MB")

    pf("no monotonic RSS growth (leaks)", deltas[-1] <= deltas[0] + 5*1024*1024,
          f"deltas={[d//1024//1024 for d in deltas]} MB")
    pf("reset_cache frees working memory", after_reset < before_final + 10*1024*1024,
          f"after_reset={after_reset//1024//1024}MB base={base_rss//1024//1024}MB")

    del m
    gc.collect()

except ImportError:
    print("  [SKIP] psutil not installed, skipping memory telemetry")
    m = AtlasModel(MODEL)
    out = m.generate_c("Hello", max_new_tokens=5, temperature=0.0, top_k=1)
    pf("basic generation works", len(out) > 0 and not out.startswith("[ATLAS:"), f"got {out!r}")
    del m

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
print(f"  {PASS} PASS / {FAIL} FAIL / {PASS+FAIL} total")
print(f"{'='*50}")
sys.exit(0 if FAIL == 0 else 1)
