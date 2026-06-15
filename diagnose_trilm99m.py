"""Diagnose TriLM-99M: understand repetition pattern and find optimal settings."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from atlas_infer import AtlasModel

MODEL_PATH = r"C:\atlas\models\TriLM-99M-ATLAS.atlas"
PROMPTS = [
    "The capital of France is",
    "What is the capital of France?",
    "Paris is the capital of",
    "Once upon a time,",
    "The Eiffel Tower is located in",
    "The best thing about France is",
    "Germany's capital is",
]

def test_config(model_path, label, **kwargs):
    print(f"\n{'='*60}")
    print(f"CONFIG: {label}")
    print(f"  kwargs: {kwargs}")
    print(f"{'='*60}")
    try:
        m = AtlasModel(model_path)
        for k, v in kwargs.items():
            setattr(m, k, v)
        m.set_num_threads(4)
        for prompt in PROMPTS[:3]:
            out = m.generate_c(prompt, max_new_tokens=50, **kwargs)
            tokens = m._cpp_encode(out) if isinstance(out, str) else []
            print(f"  prompt='{prompt}' -> '{out}'")
            if tokens:
                print(f"    last 5 tokens: {tokens[-5:]}" if len(tokens) >= 5 else f"    tokens: {tokens}")
        del m
    except Exception as e:
        print(f"  ERROR: {e}")

# Test 1: Default (T=0.7, top_k=40)
test_config(MODEL_PATH, "DEFAULT: T=0.7, top_k=40",
    temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.0)

# Test 2: T=0 argmax
test_config(MODEL_PATH, "T=0 argmax",
    temperature=0.0, top_k=0, top_p=0.0, repetition_penalty=1.0)

# Test 3: T=0.7 with no top_k
test_config(MODEL_PATH, "T=0.7, no top_k",
    temperature=0.7, top_k=0, top_p=0.9, repetition_penalty=1.0)

# Test 4: Higher temperature
test_config(MODEL_PATH, "T=1.0, top_k=40",
    temperature=1.0, top_k=40, top_p=0.9, repetition_penalty=1.0)

# Test 5: Repetition penalty
test_config(MODEL_PATH, "T=0.7, top_k=40, rep_penalty=1.2",
    temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.2)

# Test 6: Strong repetition penalty
test_config(MODEL_PATH, "T=0.7, top_k=40, rep_penalty=1.5",
    temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.5)

# Test 7: Low temperature with rep penalty
test_config(MODEL_PATH, "T=0.3, top_k=40, rep_penalty=1.2",
    temperature=0.3, top_k=40, top_p=0.9, repetition_penalty=1.2)

# Test 8: No chat template (raw prompt)
test_config(MODEL_PATH, "NO CHAT TEMPLATE",
    temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.0)

# Test 9: f32_bypass
print(f"\n{'='*60}")
print("TEST: f32_bypass=True vs hybrid")
print(f"{'='*60}")
for use_f32 in [True, False]:
    try:
        m = AtlasModel(MODEL_PATH)
        m.set_use_f32_matmul(use_f32)
        m.set_num_threads(4)
        out = m.generate_c("The capital of France is", max_new_tokens=50,
            temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.0)
        print(f"  f32_bypass={use_f32}: '{out}'")
        del m
    except Exception as e:
        print(f"  f32_bypass={use_f32}: ERROR: {e}")

# Test 10: Very long generation to see pattern
print(f"\n{'='*60}")
print("LONG GENERATION (200 tokens) - analyze pattern")
print(f"{'='*60}")
try:
    m = AtlasModel(MODEL_PATH)
    m.set_num_threads(4)
    out = m.generate_c("The capital of France is", max_new_tokens=200,
        temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.0)
    print(f"  Output: '{out}'")
    print(f"  Length: {len(out)} chars")
    # Show unique tokens
    tokens = m._cpp_encode(out) if isinstance(out, str) else []
    if tokens:
        unique = len(set(tokens))
        total = len(tokens)
        print(f"  Tokens: {total} total, {unique} unique")
        if total > 10:
            print(f"  First 10:  {tokens[:10]}")
            print(f"  Last 10:   {tokens[-10:]}")
            # Check for exact repeats
            for window in [2, 3, 4]:
                repeats = 0
                seen = set()
                for i in range(len(tokens) - window + 1):
                    chunk = tuple(tokens[i:i+window])
                    if chunk in seen:
                        repeats += 1
                    seen.add(chunk)
                print(f"  Repeated {window}-grams: {repeats}/{len(tokens)-window+1}")
    del m
except Exception as e:
    print(f"  ERROR: {e}")
