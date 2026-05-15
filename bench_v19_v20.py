import subprocess, struct, time, sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def bench(exe, label, prompt_ids, max_gen=30):
    proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0)
    t0 = time.time()
    proc.stdin.write(struct.pack("i", len(prompt_ids)))
    for tid in prompt_ids:
        proc.stdin.write(struct.pack("i", tid))
    tokens = []
    for step in range(max_gen):
        data = proc.stdout.read(4)
        if not data or len(data) < 4:
            break
        tok = struct.unpack("i", data)[0]
        tokens.append(tok)
        if tok == 128001:
            break
        proc.stdin.write(struct.pack("i", tok))
    t1 = time.time()
    try:
        proc.stdin.write(struct.pack("i", -1))
    except:
        pass
    proc.stdin.close()
    stderr_data = proc.stderr.read().decode('utf-8', errors='replace')
    proc.wait(timeout=30)

    elapsed = t1 - t0
    tps = len(tokens) / max(elapsed, 0.001)
    load_line = [l for l in stderr_data.splitlines() if l.startswith('Load')]
    load_time = load_line[0].split()[1].rstrip('s') if load_line else '?'
    return tokens, elapsed, tps, load_time, stderr_data

prompt = [128000, 3923, 374, 279, 6864, 315, 9822, 30]
print(f"Prompt: {prompt}")
print(f"{'Engine':>12} {'Tok':>4} {'Time(s)':>8} {'t/s':>8} {'Load(s)':>8}  {'Toks'}")
print("-"*80)

for exe, label in [("atlas_v19_speed.exe", "v19"), ("atlas_v20.exe", "v20")]:
    if not os.path.exists(exe):
        print(f"{label:>12}: (not found)")
        continue
    toks, elapsed, tps, load, stderr = bench(exe, label, prompt, max_gen=20)
    tok_str = " ".join(str(t) for t in toks[:8])
    print(f"{label:>12} {len(toks):>4} {elapsed:>8.3f} {tps:>8.2f} {load:>8}  {tok_str}...")

print("\n--- Full stderr v19 ---")
toks_v19, _, _, _, stderr_v19 = bench("atlas_v19_speed.exe", "v19", prompt, max_gen=5)
for l in stderr_v19.splitlines():
    if l.strip():
        print(l)

print("\n--- Full stderr v20 ---")
toks_v20, _, _, _, stderr_v20 = bench("atlas_v20.exe", "v20", prompt, max_gen=5)
for l in stderr_v20.splitlines():
    if l.strip():
        print(l)
