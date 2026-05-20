"""Bottleneck analysis: OMP thread scaling for int8 vs ternary paths.
Measures tok/s at 1,2,4,8 threads to determine compute vs memory bound."""
import os, sys, time, subprocess, json
import numpy as np

MODEL = r"C:\atlas\falcon3-10b-tq1.atlas"
PROMPT = "The capital of France is"
GEN_TOKENS = 30
WARMUP = 2  # prompt tokens for warmup

# Python script template - run in subprocess to control OMP
SCRIPT_TEMPLATE = r'''
import os, sys, time, ctypes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, r"{atlas_dir}")
import atlas_infer as ai

model_path = r"{model_path}"
prompt = "{prompt}"
gen_tokens = {gen_tokens}
use_ternary = {use_ternary}

t0 = time.time()
m = ai.AtlasModel(model_path)
load_time = time.time() - t0
print("LOAD_TIME:" + str(round(load_time, 3)))

m.set_use_ternary_matmul(use_ternary)
m.set_seed(42)

t0 = time.time()
out = m.generate_c(prompt, max_new_tokens=gen_tokens, temperature=0.7, top_k=40, top_p=0.9)
gen_time = time.time() - t0
tok_per_s = gen_tokens / gen_time
print("GEN_TIME:" + str(round(gen_time, 3)))
print("TOK_PER_S:" + str(round(tok_per_s, 3)))
print("OUTPUT:" + out[:80])
'''

def run_bench(n_threads, use_ternary):
    atlas_dir = os.path.dirname(os.path.abspath(__file__))
    script = SCRIPT_TEMPLATE.format(
        model_path=MODEL,
        prompt=PROMPT,
        gen_tokens=GEN_TOKENS,
        use_ternary="True" if use_ternary else "False",
        atlas_dir=atlas_dir
    )
    # Write to temp file
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_bench_scale_temp.py')
    with open(script_path, 'w') as f:
        f.write(script)
    
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(n_threads)
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=300,
        env=env
    )
    elapsed = time.time() - t0
    
    results = {}
    for line in result.stdout.strip().split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            try:
                results[key] = float(val) if '.' in val else val
            except:
                results[key] = val
    
    # Check for errors
    if result.returncode != 0:
        print(f"  ERROR (threads={n_threads}, ternary={use_ternary}):")
        for line in result.stderr.split('\n')[-5:]:
            print(f"    {line}")
        return None
    
    return results

def main():
    print(f"═══ THREAD SCALING BENCHMARK ═══")
    print(f"Model: 10B, Gen: {GEN_TOKENS} tokens")
    print(f"{'Threads':>8s} {'Path':>8s} {'Load(s)':>8s} {'Gen(s)':>8s} {'tok/s':>8s} {'Speedup':>8s}")
    print("-" * 56)
    
    for path_name, use_ternary in [("int8", False), ("ternary", True)]:
        for n_threads in [1, 2, 4, 8]:
            # Cold run (first time): include load time
            res = run_bench(n_threads, use_ternary)
            if res is None:
                continue
            tok_per_s = res.get('TOK_PER_S', 0)
            gen_time = res.get('GEN_TIME', 0)
            load_time = res.get('LOAD_TIME', 0)
            speedup = tok_per_s / 0.35 if n_threads > 1 else 1.0  # relative to 1-thread
            
            print(f"{n_threads:>8d} {path_name:>8s} {load_time:>8.2f} {gen_time:>8.2f} {tok_per_s:>8.2f} {speedup:>7.2f}x")
        
        # Warm cache run at 8 threads (2nd run)
        if not use_ternary:
            print("-" * 56)
    
    # Memory bandwidth estimation (warm cache, 8 threads)
    print(f"\n═══ MEMORY BANDWIDTH ESTIMATE ═══")
    # 10B model: ~8.86 GB int8 weights read per token (decompressed)
    # For generate with gen_tokens tokens, weights read = 8.86 * gen_tokens GB
    # But only the first token does prefill (batch forward), subsequent do decode
    # Approx: each token reads ~8.86 GB of int8 weights
    weight_bytes_int8 = 8.86e9  # 8.86 GB
    
    for n_threads in [1, 8]:
        res_int8 = run_bench(n_threads, False)
        res_tern = run_bench(n_threads, True)
        if res_int8 and res_tern:
            bw_int8 = weight_bytes_int8 * res_int8['TOK_PER_S'] / 1e9
            bw_tern = weight_bytes_int8 * res_tern['TOK_PER_S'] / 1e9
            print(f"  {n_threads} threads, int8:    {bw_int8:.1f} GB/s (est. weight read BW)")
            print(f"  {n_threads} threads, ternary: {bw_tern:.1f} GB/s (est. weight read BW)")
    
    print(f"\n═══ DONE ═══")

if __name__ == '__main__':
    main()
