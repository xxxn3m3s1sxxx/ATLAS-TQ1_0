import subprocess, struct, time, sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

prompt = [128000, 3923, 374, 279, 6864, 315, 9822, 30]

def warmup(exe, label):
    proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    proc.stdin.write(struct.pack("i", len(prompt)))
    for tid in prompt:
        proc.stdin.write(struct.pack("i", tid))
    proc.stdin.flush()
    for _ in range(50):
        data = proc.stdout.read(4)
        if not data:
            break
        tok = struct.unpack("i", data)[0]
        if tok == 128001:
            break
        proc.stdin.write(struct.pack("i", tok))
        proc.stdin.flush()
    proc.kill()
    proc.wait()
    print(f"{label:>12}: 50 toks (warmup)")

for exe, label in [("atlas_v19_speed.exe", "v19"), ("atlas_v20.exe", "v20")]:
    if os.path.exists(exe):
        warmup(exe, label)

print()

def run_bench(exe, label, max_gen=50):
    proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    proc.stdin.write(struct.pack("i", len(prompt)))
    for tid in prompt:
        proc.stdin.write(struct.pack("i", tid))
    proc.stdin.flush()

    data = proc.stdout.read(4)
    if not data:
        proc.kill(); proc.wait()
        return 0, 0
    tok = struct.unpack("i", data)[0]

    tokens = [tok]
    t0 = time.perf_counter()
    for step in range(max_gen - 1):
        if tok == 128001:
            break
        proc.stdin.write(struct.pack("i", tok))
        proc.stdin.flush()
        data = proc.stdout.read(4)
        if not data:
            break
        tok = struct.unpack("i", data)[0]
        tokens.append(tok)
    t1 = time.perf_counter()
    gen_time = t1 - t0
    gen_toks = len(tokens) - 1

    proc.kill()
    proc.wait()
    return gen_toks, gen_time

for exe, label in [("atlas_v19_speed.exe", "v19"), ("atlas_v20.exe", "v20")]:
    if not os.path.exists(exe):
        continue
    times = []
    counts = []
    for run in range(3):
        n, t = run_bench(exe, label, max_gen=50)
        if n > 0:
            times.append(t)
            counts.append(n)
    if times:
        avg_t = sum(times) / len(times)
        avg_n = sum(counts) / len(counts)
        avg_tps = avg_n / avg_t
        print(f"{label:>12}: {avg_n:.0f} gen-toks avg {avg_t:.3f}s = {avg_tps:.3f} t/s  (runs: {[f'{t:.3f}s' for t in times]})")
