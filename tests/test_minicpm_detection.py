import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atlas_packer_mappings import detect_arch, MINICPM, LLAMA

# Simulate MiniCPM config
cfg = {"hidden_size": 2304, "scale_depth": 1.4, "scale_emb": 12, "dim_model_base": 256, "architectures": ["MiniCPMForCausalLM"]}
arch = detect_arch(cfg, ["model.layers.0.self_attn.q_proj.weight"])
assert arch is MINICPM, f"Expected MINICPM, got {arch}"
print(f"MiniCPM detected OK | stride={arch['stride']} quant={arch['quant_weight']}")

# Test without MiniCPM fields → should fall through to LLAMA
cfg2 = {"hidden_size": 4096, "num_hidden_layers": 32, "architectures": ["LlamaForCausalLM"]}
arch2 = detect_arch(cfg2, [])
assert arch2 is LLAMA, f"Expected LLAMA, got {arch2}"
print(f"Standard Llama OK | stride={arch2['stride']}")

# Verify MINICPM is in ARCH_REGISTRY
assert "minicpm" in detect_arch.__globals__.get("ARCH_REGISTRY", {}), "MINICPM not in ARCH_REGISTRY"
print("MINICPM in ARCH_REGISTRY OK")

print("\nAll checks passed!")
