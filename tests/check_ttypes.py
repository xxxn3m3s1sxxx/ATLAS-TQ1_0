import os, json

for fname in ['C:/atlas/tests/mock-falcon3.atlas', 'C:/atlas/mock/ci-falcon3.atlas',
              'C:/atlas/mock/out-falcon3.atlas', 'C:/atlas/mock/out-falcon3-fp16.atlas']:
    with open(fname, 'rb') as f:
        raw = f.read(4096)
    text = raw.decode('utf-8', errors='replace')
    end_marker = text.find('\n@\n')
    if end_marker > 0:
        json_str = text[:end_marker]
        hdr = json.loads(json_str)
        ttypes = set()
        for t in hdr.get('tensors', []):
            ttypes.add(t.get('ttype'))
        print(f'{os.path.basename(fname)}: {len(hdr.get("tensors",[]))} tensors, ttypes={sorted(ttypes)}')
    else:
        print(f'{os.path.basename(fname)}: not V8 format')
