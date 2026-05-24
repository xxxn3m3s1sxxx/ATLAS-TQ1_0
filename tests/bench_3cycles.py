import time, os, ctypes, gc, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from atlas_infer import AtlasModel, dll
from transformers import PreTrainedTokenizerFast

def run_gen(model, tok, prompt="What is the capital of France?", tokens=30):
    text = tok.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text)
    in_arr = (ctypes.c_int * len(ids))(*ids)
    kc = model.k_cache.ctypes.data_as(ctypes.c_void_p)
    vc = model.v_cache.ctypes.data_as(ctypes.c_void_p)
    out_arr = (ctypes.c_int * tokens)()
    t0 = time.time()
    n = dll.atlas_generate(model.model_ptr, in_arr, len(ids),
        ctypes.cast(kc, ctypes.POINTER(ctypes.c_uint16)),
        ctypes.cast(vc, ctypes.POINTER(ctypes.c_uint16)),
        model.max_seq_len, tokens, 1.0, 40, 0.9, out_arr)
    return time.time() - t0, list(out_arr[:n])

def bench(atlas_path, label, runs=3):
    print(f'\n--- {label} ---')
    model = AtlasModel(atlas_path)
    tok = PreTrainedTokenizerFast(tokenizer_object=model._tok, chat_template=model._chat_template)
    tok.add_special_tokens({'pad_token':'<|pad|>'})

    times = []
    for r in range(1, runs+1):
        gt, out = run_gen(model, tok)
        times.append(gt)
        print(f'  Run {r}: {gt:.1f}s | {30.0/gt:.2f} tok/s | "{tok.decode(out, skip_special_tokens=True)[:50]}"')

    avg = sum(times)/len(times)
    print(f'  ** {label} cycle avg: {avg:.1f}s | {30.0/avg:.2f} tok/s')
    del model, tok
    gc.collect()
    return avg

models = [
    (r'C:\dam\models\Falcon3-1B-Instruct-1.58bit\falcon3-1b-tq1.atlas', 'Falcon3-1B'),
    (r'C:\dam\models\Falcon3-3B-Instruct-1.58bit\falcon3-3b-tq1.atlas', 'Falcon3-3B'),
    (r'C:\dam\models\Falcon3-7B-Instruct-1.58bit\falcon3-7b-tq1.atlas', 'Falcon3-7B'),
    (r'C:\dam\models\Falcon3-10B-Instruct-1.58bit\falcon3-10b-tq1.atlas', 'Falcon3-10B'),
]

print('='*60)
print('ATLAS v1.2.0  --  i7-7700T  --  3 Cycles  --  Sequential')
print('='*60)

results = {}
for cycle in range(1, 4):
    print(f'\n[===== CYCLE {cycle}/3 =====]')
    results[cycle] = {}
    for atlas_path, label in models:
        avg = bench(atlas_path, label, runs=3)
        results[cycle][label] = avg
        # Free disk space before next model
        i8 = atlas_path + '.i8'
        if os.path.exists(i8):
            sz = os.path.getsize(i8)/1e9
            os.remove(i8)
            print(f'  [Freed {sz:.1f} GB cache]')
        gc.collect()
        time.sleep(3)
    if cycle < 3:
        print(f'\n[Cooldown 60s...]')
        time.sleep(60)

print('\n' + '='*60)
print('FINAL RESULTS  (3 cycles x 3 runs each)')
print('='*60)
print('{:<16} {:>10} {:>10} {:>10} {:>10} | {:>12}'.format(
    'Model','Cycle1 (s)','Cycle2 (s)','Cycle3 (s)','AVG (s)','tok/s'))
print('-'*60)
for atlas_path, label in models:
    c1 = results[1][label]
    c2 = results[2][label]
    c3 = results[3][label]
    avg = (c1+c2+c3)/3
    print('{:<16} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f} | {:>12.2f}'.format(
        label, c1, c2, c3, avg, 30.0/avg))