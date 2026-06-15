"""Bug hunt: test Falcon-E models load + generate."""
import os, sys, time, tempfile
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\atlas')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['ATLAS_DLL'] = r'C:\atlas\atlas_d.dll'

from huggingface_hub import hf_hub_download, HfApi
from atlas_infer import AtlasModel

api = HfApi()
tmp = tempfile.mkdtemp(prefix='atlas_dbg_')
print(f'Temp: {tmp}')

targets = [
    ('xxxn3m3s1sxxx/Falcon-E-1B-Base-1.58bit-ATLAS',    'Falcon-E-1B-Base'),
    ('xxxn3m3s1sxxx/Falcon-E-1B-Instruct-1.58bit-ATLAS', 'Falcon-E-1B-Inst'),
]

for rid, label in targets:
    print(f'\n=== {label} ===')
    try:
        files = api.list_repo_files(rid, repo_type='model')
        atlas_file = [f for f in files if f.endswith('.atlas')][0]
        local = hf_hub_download(rid, atlas_file, local_dir=tmp)
        t0 = time.time()
        m = AtlasModel(local)
        load_t = time.time() - t0
        print(f'  {m.n_layers}L {m.hidden}H {m.inter}I v={m.vocab_size} loaded in {load_t:.1f}s')
        t0 = time.time()
        out = m.generate_c('The capital of France is', max_new_tokens=20, temperature=0.0, top_k=1)
        gen_t = time.time() - t0
        out = out.replace('\n',' | ').strip()
        print(f'  gen={gen_t:.1f}s (20 tok/s)  "{out[:100]}"')
        if 'Paris' in out:
            print(f'  ✅ Paris')
        elif len(out) > 10:
            print(f'  ⚠️  no Paris: "{out[:60]}"')
        else:
            print(f'  ❌ short: "{out[:60]}"')
        del m; import gc; gc.collect(); time.sleep(1)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'  FAIL: {e}')
    finally:
        if 'local' in dir() and os.path.exists(local):
            for _ in range(3):
                try: os.remove(local); break
                except PermissionError: time.sleep(2)
