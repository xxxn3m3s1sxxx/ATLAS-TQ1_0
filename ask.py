"""Ask Falcon3 a question via C++ inference engine.
Usage:
  python ask.py "What is the capital of France?"
  python ask.py --temp 0.5 "Write a short poem" --7b
  set ATLAS_TEMP=0.3 && python ask.py "Explain gravity"
"""
import subprocess, struct, time, threading, sys, os
from transformers import AutoTokenizer

MODEL = r'C:\dam\models\Falcon3-10B-Instruct-1.58bit'
EXE_10B = r'C:\dam\atlas\atlas_falcon3_avx2.exe'
EXE_7B = r'C:\dam\atlas\atlas_falcon3_7b_avx2.exe'

# Parse args
args = sys.argv[1:]
temp = None
use_7b = False
prompt_parts = []
i = 0
while i < len(args):
    if args[i] == '--temp' and i + 1 < len(args):
        temp = float(args[i + 1]); i += 2
    elif args[i] == '--7b':
        use_7b = True; i += 1
    elif args[i] == '--10b':
        use_7b = False; i += 1
    else:
        prompt_parts.append(args[i]); i += 1

exe = EXE_7B if use_7b else EXE_10B
if not os.path.exists(exe):
    exe = EXE_7B if not os.path.exists(EXE_10B) else EXE_7B

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

if prompt_parts:
    user_text = ' '.join(prompt_parts)
else:
    user_text = "What is the capital of France? Answer in one word."

prompt_text = tok.apply_chat_template(
    [{"role": "user", "content": user_text}],
    tokenize=False, add_generation_prompt=True
)
print(f"\033[90mPrompt ({len(tok.encode(prompt_text))} tok)\033[0m")
ids = tok.encode(prompt_text)

# Build command
cmd = [exe]
if temp is not None:
    cmd.append(str(temp))
    print(f"\033[90mTemperature: {temp}\033[0m")

proc = subprocess.Popen(
    cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE, cwd=r'C:\dam\atlas'
)
threading.Thread(target=lambda: [line for line in iter(proc.stderr.readline, b'')], daemon=True).start()
time.sleep(0.1)

MAX_TOKENS = 200

t0 = time.perf_counter()
proc.stdin.write(struct.pack('i', len(ids)))
for tid in ids: proc.stdin.write(struct.pack('i', tid))
proc.stdin.flush()

gen_ids = []
for _ in range(MAX_TOKENS):
    data = proc.stdout.read(4)
    if not data: break
    gid = struct.unpack('i', data)[0]
    gen_ids.append(gid)
    if gid == 11: break
    proc.stdin.write(struct.pack('i', gid))
    proc.stdin.flush()

proc.stdin.close()
t_end = time.perf_counter()
try: proc.wait(timeout=3)
except: proc.kill()

text = tok.decode(gen_ids)
print(text)
n = len(ids) + len(gen_ids)
print(f"\033[90m{len(gen_ids)} tok | {t_end-t0:.1f}s | {(t_end-t0)/n*1000:.0f}ms/tok\033[0m")
