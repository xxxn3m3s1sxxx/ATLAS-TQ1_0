"""Analyze what tokens 9334 and friends actually decode to."""
import warnings; warnings.filterwarnings('ignore')
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(r'C:\dam\models\bitnet-b1.58-2B-4T', fix_mistral_regex=True)

tokens = [9334, 951, 75352, 37356, 21878, 61798, 72574, 27637, 31643, 74894]
for i, t in enumerate(tokens[:15]):
    d = tok.decode([t])
    print(f"  token {t:>6} -> '{d}'")

print(f"\nAll together: '{tok.decode(tokens, skip_special_tokens=True)}'")

# Try reference: what should "What is the capital of France?" answer look like?
ref = tok.encode("Paris is the capital of France.")
print(f"\nReference 'Paris is the capital of France.': {ref}")

# Check BOS
print(f"\nBOS token: {tok.encode('')[0] if tok.encode('') else 'none'}")

# Run the exact feed_v19.py protocol
import struct, subprocess
exe = r'C:\dam\atlas\atlas_v19.exe'
ids = [128000, 3923, 374, 279, 6864, 315, 9822, 30]

proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
proc.stdin.write(struct.pack('<i', len(ids)))
for tid in ids: proc.stdin.write(struct.pack('<i', tid))
proc.stdin.flush()
proc.stdin.close()

raw = proc.stdout.read()
stderr = proc.stderr.read().decode('utf-8', errors='replace')
proc.wait(timeout=30)

if raw and len(raw) >= 4:
    tok_id = struct.unpack('<i', raw[:4])[0]
    decoded = tok.decode([tok_id])
    print(f"\nfeed_v19.py result: token={tok_id} -> '{decoded}'")
