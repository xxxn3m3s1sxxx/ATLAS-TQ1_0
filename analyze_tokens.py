"""Analyze generated tokens — write to file to avoid console encoding issues."""
import sys
sys.path.insert(0, r"C:\dam\atlas")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(r"C:\dam\models\granite-3.0-2b-instruct")
OUT = r"C:\dam\atlas\token_analysis.txt"

ids = [36238, 36291, 116, 25250, 5184, 38022, 31250, 6075, 17380, 
       42405, 42625, 915, 308, 6941, 34, 45801, 9837, 1817, 323, 8603, 
       20063, 34, 308, 11664, 11385, 40827, 307, 373, 9828, 34, 19492, 28358,
       34, 11969, 11909, 281, 25959, 10167, 7097, 31, 48238, 281, 34, 23614,
       34787, 36837, 6587, 12588, 5024, 34, 1248, 34, 14004, 11927, 36304,
       34, 9367, 39573, 32735, 19117, 308, 37610, 17641, 308, 5800, 1975,
       47410, 20579]

with open(OUT, 'w', encoding='utf-8') as f:
    f.write("Token analysis of Granite 2B output\n")
    f.write("="*50 + "\n\n")
    f.write("Individual token decodes:\n")
    for tid in ids:
        token = tok.decode([tid], skip_special_tokens=False)
        piece = tok.convert_ids_to_tokens(tid) if hasattr(tok, 'convert_ids_to_tokens') else str(tid)
        f.write(f"  {tid:5d} -> |{token}| (piece: {piece})\n")
    
    f.write("\nFull decode:\n")
    full = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    f.write(full + "\n")
    f.write(f"\n(Total length: {len(full)} chars)\n")
    
    # Check tokenizer type
    f.write(f"\nTokenizer type: {type(tok).__name__}\n")
    f.write(f"Vocab size: {tok.vocab_size}\n")
    
    # Check special tokens
    f.write(f"\nSpecial tokens:\n")
    f.write(f"  BOS: {tok.bos_token_id} -> {repr(tok.bos_token)}\n")
    f.write(f"  EOS: {tok.eos_token_id} -> {repr(tok.eos_token)}\n")
    f.write(f"  PAD: {tok.pad_token_id} -> {repr(tok.pad_token)}\n")
    f.write(f"  UNK: {tok.unk_token_id} -> {repr(tok.unk_token)}\n")
    
    # Check if 36238 is in vocab
    try:
        piece = tok.convert_ids_to_tokens(36238)
        f.write(f"\nToken 36238 piece: {piece}\n")
    except Exception as e:
        f.write(f"\nToken 36238 error: {e}\n")

    # Check first few token ranges for structure
    f.write(f"\nFirst 10 tokens (0-9):\n")
    for i in range(10):
        t = tok.decode([i], skip_special_tokens=False)
        p = tok.convert_ids_to_tokens(i) if hasattr(tok, 'convert_ids_to_tokens') else '?'
        f.write(f"  {i:5d} -> |{t}| (piece: {p})\n")

print(f"Analysis written to {OUT}")
