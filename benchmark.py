import subprocess, struct, threading, time
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(r'C:\dam\models\bitnet-b1.58-2B-4T')
prompt_ids = tokenizer.encode('The capital of France is')

proc = subprocess.Popen(
    [r'C:\dam\atlas\atlas_v20.exe'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
)

stderr_lines = []
def drain():
    for line in iter(proc.stderr.readline, b''):
        if not line: break
        stderr_lines.append(line.decode('utf-8', errors='replace').rstrip())
threading.Thread(target=drain, daemon=True).start()

# Wait for READY signal
import time as _time
_time.sleep(0.5)

t0 = time.time()
proc.stdin.write(struct.pack('i', len(prompt_ids)))
for tok in prompt_ids:
    proc.stdin.write(struct.pack('i', tok))
proc.stdin.flush()

tokens = []
for step in range(10):
    if step == 0:
        tok = struct.unpack('i', proc.stdout.read(4))[0]
    else:
        proc.stdin.write(struct.pack('i', tokens[-1]))
        proc.stdin.flush()
        tok = struct.unpack('i', proc.stdout.read(4))[0]
    tokens.append(tok)

proc.stdin.close()
proc.wait()
elapsed = time.time() - t0

# Parse load time from stderr
load_time = 0
for line in stderr_lines:
    if 'Load' in line:
        parts = line.split()
        for p in parts:
            try:
                load_time = float(p.replace('s', ''))
            except:
                pass

print(f'Load time: {load_time:.2f}s')
print(f'Prompt tokens: {len(prompt_ids)}, Gen tokens: {len(tokens)}')
print(f'Total time: {elapsed:.1f}s')
per_step = elapsed / (len(prompt_ids) + len(tokens))
print(f'Per step: {per_step:.3f}s ({1/per_step:.1f} tok/s)')
print(f'Output: {tokenizer.decode(prompt_ids + tokens, skip_special_tokens=False)}')

# Also print stderr timing lines
for line in stderr_lines[-3:]:
    print(f'  {line}')
