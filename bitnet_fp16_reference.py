"""Run FP16 reference for BitNet b1.58 (uses custom modeling code)."""
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\dam\atlas")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = r"C:\dam\models\bitnet-b1.58-2B-4T"
MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"
OUT = r"C:\dam\atlas\bitnet_fp16_output.txt"

tok = AutoTokenizer.from_pretrained(MODEL_ID)

PROMPTS = [
    "What is the capital of France?",
    "The capital of France is",
    "Paris is the capital of",
]

with open(OUT, 'w', encoding='utf-8') as f:
    f.write("BitNet b1.58 Reference Output\n")
    f.write("=" * 60 + "\n")

print("Loading BitNet b1.58 model (trust_remote_code)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
print(f"Loaded in {time.time()-t0:.1f}s")
model.eval()

with open(OUT, 'a', encoding='utf-8') as f:
    for pi, prompt in enumerate(PROMPTS):
        f.write(f"\nPrompt {pi}: {prompt}\n")
        ids = tok.encode(prompt, return_tensors="pt")
        f.write(f"  Input IDs: {ids.tolist()}\n")
        
        with torch.no_grad():
            outputs = model(ids)
            logits = outputs.logits[0, -1, :]
            
            top5 = torch.topk(logits, 5)
            f.write(f"  Top-5 next-token logits:\n")
            for i in range(5):
                idx = top5.indices[i].item()
                val = top5.values[i].item()
                piece = tok.convert_ids_to_tokens(idx)
                f.write(f"    {idx:6d} -> '{piece}' (logit={val:.4f})\n")
            
            greedy = torch.argmax(logits).item()
            piece = tok.convert_ids_to_tokens(greedy)
            f.write(f"  Greedy: {greedy} -> '{piece}'\n")
        
        with torch.no_grad():
            outputs = model.generate(
                ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tok.pad_token_id or 0,
            )
        generated = outputs[0][len(ids[0]):]
        text = tok.decode(generated, skip_special_tokens=True)
        f.write(f"  Greedy generation:\n")
        for tid in generated.tolist():
            piece = tok.convert_ids_to_tokens(tid)
            f.write(f"    {tid:6d} -> '{piece}'\n")
        f.write(f"  Decoded: {text}\n")
        print(f"  Prompt {pi} done")

with open(OUT, 'a', encoding='utf-8') as f:
    f.write("\nDone.\n")
print(f"Output written to {OUT}")
