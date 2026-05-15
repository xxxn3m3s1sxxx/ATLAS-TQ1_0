"""
Baseline long generation — simple factual prompt.
Tests if model degrades after many tokens of generation.
"""
import struct, subprocess, time, warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer

TOKENIZER_PATH = r'C:\dam\models\bitnet-b1.58-2B-4T'
MODEL_EXE = r'C:\dam\atlas\atlas_v19.exe'
EOS_ID, BOS_ID = 128001, 128000

# Simple prompt — known to produce argmax 9334
PROMPT = "What is the capital of France?"

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, fix_mistral_regex=True)
ids = tok.encode(PROMPT)
if ids[0] != BOS_ID: ids = [BOS_ID] + ids
print(f"Prompt: {len(ids)} tokens -> {ids}")

proc = subprocess.Popen([MODEL_EXE], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
proc.stdin.write(struct.pack('i', len(ids)))
for tid in ids: proc.stdin.write(struct.pack('i', tid))
proc.stdin.flush()

gen = []
t0 = time.time()
for step in range(100):
    data = proc.stdout.read(4)
    if not data or len(data) < 4: break
    gid = struct.unpack('i', data)[0]
    gen.append(gid)
    if gid == EOS_ID: break
    proc.stdin.write(struct.pack('i', gid))
    proc.stdin.flush()
t1 = time.time()

try: proc.stdin.write(struct.pack('i', -1))
except: pass
proc.wait(timeout=5)

print(f"\nGenerated {len(gen)} tokens in {t1-t0:.2f}s")
print(f"First 5: {gen[:5]}")
print(f"Last 5:  {gen[-5:]}")
print(f"\nDecoded: {tok.decode(gen, skip_special_tokens=True)}")
