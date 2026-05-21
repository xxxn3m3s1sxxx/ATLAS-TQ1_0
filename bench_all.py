import time, os, ctypes, glob
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from atlas_infer import AtlasModel
from atlas_infer import dll
from transformers import PreTrainedTokenizerFast

def bench_model(atlas_path, label, runs=3):
    print(f'\n=== {label} ===')
    results = []
    for i in range(runs):
        t0 = time.time()
        model = AtlasModel(atlas_path)
        load_time = time.time() - t0

        t0 = time.time()
        tok = PreTrainedTokenizerFast(tokenizer_object=model._tok, chat_template=model._chat_template)
        tok.add_special_tokens({'pad_token': '<|pad|>', 'bos_token': '<s>'})

        prompt = [{'role': 'user', 'content': 'What is the capital of France?'}]
        text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_ids = tok.encode(text)
        n_input = len(input_ids)
        in_arr = (ctypes.c_int * n_input)(*input_ids)
        k_cache = model.k_cache.ctypes.data_as(ctypes.c_void_p)
        v_cache = model.v_cache.ctypes.data_as(ctypes.c_void_p)
        out_arr = (ctypes.c_int * 30)()
        n_gen = dll.atlas_generate(
            model.model_ptr, in_arr, n_input,
            ctypes.cast(k_cache, ctypes.POINTER(ctypes.c_uint16)),
            ctypes.cast(v_cache, ctypes.POINTER(ctypes.c_uint16)),
            model.max_seq_len, 30,
            1.0, 40, 0.9,
            out_arr)
        gen_time = time.time() - t0
        output = list(out_arr[:n_gen])
        decoded = tok.decode(output, skip_special_tokens=True)
        toks_per_sec = 30.0 / gen_time if gen_time > 0 else 0

        results.append((load_time, gen_time, toks_per_sec, decoded[:50]))
        print(f'  Run {i+1}: load={load_time:.1f}s | gen={gen_time:.1f}s | {toks_per_sec:.2f} tok/s | "{decoded[:40]}"')

    avg_load = sum(r[0] for r in results) / len(results)
    avg_gen = sum(r[1] for r in results) / len(results)
    avg_tok = sum(r[2] for r in results) / len(results)
    print(f'  AVG: load={avg_load:.1f}s | gen={avg_gen:.1f}s | {avg_tok:.2f} tok/s')
    return label, avg_load, avg_gen, avg_tok

# Cleanup helper: remove .i8 caches to free space between runs
def cleanup_caches(base_dir):
    for f in glob.glob(os.path.join(base_dir, '**', '*.i8'), recursive=True):
        try:
            os.remove(f)
            print(f'  Removed cache: {os.path.basename(f)}')
        except:
            pass

all_results = []
models = [
    (r'C:\dam\models\Falcon3-1B-Instruct-1.58bit\falcon3-1b-tq1.atlas', 'Falcon3-1B', r'C:\dam\models\Falcon3-1B-Instruct-1.58bit'),
    (r'C:\dam\models\Falcon3-3B-Instruct-1.58bit\falcon3-3b-tq1.atlas', 'Falcon3-3B', r'C:\dam\models\Falcon3-3B-Instruct-1.58bit'),
    (r'C:\dam\models\Falcon3-7B-Instruct-1.58bit\falcon3-7b-tq1.atlas', 'Falcon3-7B', r'C:\dam\models\Falcon3-7B-Instruct-1.58bit'),
    (r'C:\dam\models\Falcon3-10B-Instruct-1.58bit\falcon3-10b-tq1.atlas', 'Falcon3-10B', r'C:\dam\models\Falcon3-10B-Instruct-1.58bit'),
]

for idx, (path, label, cache_dir) in enumerate(models):
    result = bench_model(path, label, runs=3)
    all_results.append(result)
    # Clean up cache after this model to free space for the next
    if idx < len(models) - 1:
        cleanup_caches(cache_dir)

print('')
print('='*70)
print('FINAL BENCHMARK SUMMARY (i7-7700T, 3 runs avg)')
print('='*70)
header = '{:<16} {:<14} {:<18} {:<14}'.format('Model', 'Load (avg)', 'Gen 30tok (avg)', 'Tok/s (avg)')
print(header)
for label, ld, gt, ts in all_results:
    row = '{:<16} {:<14.1f} {:<18.1f} {:<14.2f}'.format(label, ld, gt, ts)
    print(row)
