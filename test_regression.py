import sys, os
os.environ.pop('ATLAS_DLL', None)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, '.')
for name in list(sys.modules.keys()):
    if 'atlas' in name: del sys.modules[name]
import atlas_infer as ai

tests = []

m = ai.AtlasModel('falcon3-3b-tq1.atlas')
out = m.generate_c('The capital of France is', max_new_tokens=30, temperature=0.0, top_k=1)
ok = 'Paris' in out
tests.append(('Falcon3-3B T=0', repr(out[:60]), ok))

m = ai.AtlasModel('trilm-1.5b-tq1-g128.atlas')
m.set_use_f32_matmul(1)
out = m.generate_c('The capital of France is', max_new_tokens=30, temperature=0.0, top_k=1)
ok = len(out) > 10
tests.append(('TriLM-1.5B T=0', repr(out[:80]), ok))

m = ai.AtlasModel('bonsai-8b-tq1-g128-v8.atlas')
out = m.generate_c('The capital of France is', max_new_tokens=20, temperature=0.0, top_k=1)
ok = 'Paris' in out
tests.append(('Bonsai-8B T=0', repr(out[:60]), ok))

print('\n=== REGRESSION ===')
for name, output, ok in tests:
    print('  %s %s: %s' % ('OK' if ok else 'FAIL', name, output))
n = sum(1 for _,_,ok in tests if ok)
print('\n%d/%d passed' % (n, len(tests)))
