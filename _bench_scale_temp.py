
import os, sys, time, ctypes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, r"C:\atlas")
import atlas_infer as ai

model_path = r"C:\atlas\falcon3-10b-tq1.atlas"
prompt = "The capital of France is"
gen_tokens = 30
use_ternary = True

t0 = time.time()
m = ai.AtlasModel(model_path)
load_time = time.time() - t0
print("LOAD_TIME:" + str(round(load_time, 3)))

m.set_use_ternary_matmul(use_ternary)
m.set_seed(42)

t0 = time.time()
out = m.generate_c(prompt, max_new_tokens=gen_tokens, temperature=0.7, top_k=40, top_p=0.9)
gen_time = time.time() - t0
tok_per_s = gen_tokens / gen_time
print("GEN_TIME:" + str(round(gen_time, 3)))
print("TOK_PER_S:" + str(round(tok_per_s, 3)))
print("OUTPUT:" + out[:80])
