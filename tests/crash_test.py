import ctypes, os
os.chdir(r'C:\atlas')
d = ctypes.CDLL(os.path.join(os.getcwd(),'atlas.dll'))
d.atlas_load.restype=ctypes.c_void_p; d.atlas_load.argtypes=[ctypes.c_char_p]
d.atlas_free.argtypes=[ctypes.c_void_p]
d.atlas_decompress_all.restype=None; d.atlas_decompress_all.argtypes=[ctypes.c_void_p]
d.atlas_decompress_ttype5.restype=None; d.atlas_decompress_ttype5.argtypes=[ctypes.c_void_p]
d.atlas_set_num_threads.argtypes=[ctypes.c_int]; d.atlas_set_seed.argtypes=[ctypes.c_uint64]
d.atlas_quantize_lmhead.restype=None; d.atlas_quantize_lmhead.argtypes=[ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
d.atlas_prefetch_int8.restype=None; d.atlas_prefetch_int8.argtypes=[ctypes.c_void_p]
d.atlas_get_tensor_count.restype=ctypes.c_int; d.atlas_get_tensor_count.argtypes=[ctypes.c_void_p]
d.atlas_set_base_seq_len.restype=None; d.atlas_set_base_seq_len.argtypes=[ctypes.c_void_p, ctypes.c_int]
d.atlas_generate.restype=ctypes.c_int
d.atlas_generate.argtypes=[ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

# TEST 1: inline literal (working)
print("TEST 1: inline literal")
m = d.atlas_load(b'tests/falcon3_v6.atlas')
assert m
d.atlas_decompress_all(m)
d.atlas_decompress_ttype5(m)
nt=d.atlas_get_tensor_count(m)
d.atlas_quantize_lmhead(m, nt-1, 0)
d.atlas_prefetch_int8(m)
d.atlas_set_num_threads(2); d.atlas_set_seed(42)
inp=(ctypes.c_int*1)(1); out=(ctypes.c_int*10)()
n=d.atlas_generate(m,inp,1,512,10,0.0,1,1.0,1.0,0,0,out)
assert n>0
d.atlas_free(m)
print(f"  OK: {n} tokens")

# TEST 2: variable (was crashing?)
print("TEST 2: variable bytes")
name = b'tests/falcon3_v6.atlas'
m = d.atlas_load(name)
assert m
d.atlas_decompress_all(m)
d.atlas_decompress_ttype5(m)
nt=d.atlas_get_tensor_count(m)
d.atlas_quantize_lmhead(m, nt-1, 0)
d.atlas_prefetch_int8(m)
d.atlas_set_num_threads(2); d.atlas_set_seed(42)
inp=(ctypes.c_int*1)(1); out=(ctypes.c_int*10)()
n=d.atlas_generate(m,inp,1,512,10,0.0,1,1.0,1.0,0,0,out)
assert n>0
d.atlas_free(m)
print(f"  OK: {n} tokens")

# TEST 3: for loop
print("TEST 3: for loop")
for name in ['falcon3_v6']:
    m = d.atlas_load(b'tests/falcon3_v6.atlas')
    assert m
    d.atlas_decompress_all(m)
    d.atlas_decompress_ttype5(m)
    nt=d.atlas_get_tensor_count(m)
    d.atlas_quantize_lmhead(m, nt-1, 0)
    d.atlas_prefetch_int8(m)
    d.atlas_set_num_threads(2); d.atlas_set_seed(42)
    inp=(ctypes.c_int*1)(1); out=(ctypes.c_int*10)()
    n=d.atlas_generate(m,inp,1,512,10,0.0,1,1.0,1.0,0,0,out)
    assert n>0
    d.atlas_free(m)
    print(f"  {name}: {n} OK")

print("ALL TESTS PASSED")
