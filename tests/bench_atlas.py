#!/usr/bin/env python3
"""Benchmark ATLAS: prefill vs decode speed, per-operation breakdown."""
import ctypes, os, time, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from atlas_infer import AtlasModel

def profile_decode_step(model, seq_len):
    """Run one decode step with per-operation timing."""
    times = {}
    h = model._get_embedding(0)

    # Prewarm: run through all layers once to page in int8 data
    warmup_h = h.copy()
    for l in range(model.n_layers):
        warmup_h = model.forward_layer(warmup_h, l, [seq_len])

    # Warmup layer 0
    for _ in range(2):
        model.forward_layer(h, 0, [seq_len])

    # Timed decode
    n_runs = 3
    for run in range(n_runs):
        # Random input
        x = np.random.randn(1, model.hidden).astype(np.float32)
        for layer in range(model.n_layers):
            t0 = time.perf_counter()
            x = model.forward_layer(x, layer, [seq_len])
            t = time.perf_counter() - t0
            times.setdefault(f"layer_{layer}", []).append(t)
    for k in times:
        times[k] = np.mean(times[k]) * 1000  # ms
    return times

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python bench_atlas.py <atlas.tq1> <model_dir> [prompt]")
        sys.exit(1)
    atlas_path = sys.argv[1]
    model_dir = sys.argv[2]
    print(f"Loading {atlas_path}...")
    t0 = time.perf_counter()
    model = AtlasModel(atlas_path, model_dir=model_dir)
    print(f"Load: {time.perf_counter()-t0:.1f}s")

    # Profile single decode step
    print("\nProfiling decode step (per-layer)...")
    times = profile_decode_step(model, 10)

    layers_ms = np.array([times[f"layer_{l}"] for l in range(model.n_layers)])
    print(f"  Per-layer: min={layers_ms.min():.1f}ms "
          f"max={layers_ms.max():.1f}ms "
          f"mean={layers_ms.mean():.1f}ms "
          f"total={layers_ms.sum():.1f}ms")
    print(f"  Fastest layer: {layers_ms.argmin()} ({layers_ms.min():.1f}ms)")
    print(f"  Slowest layer: {layers_ms.argmax()} ({layers_ms.max():.1f}ms)")

    # Full benchmark with real text
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Hello"
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    input_ids = tok.encode(text, return_tensors='np')[0]
    seq_len = len(input_ids)
    print(f"\n=== Benchmark (seq_len={seq_len}, gen=10) ===")

    t0 = time.perf_counter()
    full_logits = model.forward(input_ids[None])
    pt = time.perf_counter() - t0
    print(f"PRE: {seq_len} tok in {pt:.1f}s ({seq_len/pt:.1f} tok/s)")

    next_token = int(np.argmax(full_logits[0, -1, :]))
    output = [next_token]
    eos_id = tok.eos_token_id
    dt = 0.0
    t0 = time.perf_counter()
    for step in range(10):
        if next_token == eos_id: break
        h = model._get_embedding(next_token)
        for layer in range(model.n_layers):
            h = model.forward_layer(h, layer, [seq_len + step])
        h_norm = model._rmsnorm(h.flatten(), "model.norm.weight")
        logits = model._matmul_f16("lm_head.weight", h_norm.reshape(1, -1)).flatten()
        next_token = int(np.argmax(logits))
        output.append(next_token)
    dt = time.perf_counter() - t0
    ng = len(output) - 1
    print(f"GEN: {ng} tok in {dt:.1f}s ({ng/dt:.1f} tok/s, {dt/ng*1000:.0f}ms/tok)")
    print(f"Output: {tok.decode(output, skip_special_tokens=True)[:100]}")
