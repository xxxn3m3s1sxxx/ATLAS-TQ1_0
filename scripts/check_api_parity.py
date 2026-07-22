#!/usr/bin/env python3
"""API Parity Scanner — prüft, ob alle C-ATLAS_API-Funktionen in atlas_ffi.h
a) als ctypes-Binding in atlas_infer.py registriert und
b) als Python-Methode in der AtlasModel-Klasse gewrappt sind.

Exit-Code: 0 = alle OK, 1 = blinde Flecken gefunden.
"""
import re, sys, os

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FFI_PATH = os.path.join(PROJECT, "atlas_ffi.h")
PY_PATH = os.path.join(PROJECT, "atlas_infer.py")

INTERNAL_C_APIS = {
    "atlas_attention_f32",
    "atlas_rmsnorm_f32",
    "atlas_rope_f32",
    "atlas_matmul_i8_f32",
    "atlas_lmhead_gemv",
    "atlas_sample",
    "atlas_get_int8",
    "atlas_tensor_info",
    "atlas_tensor_data",
    "atlas_get_config",
    "atlas_get_info",
    "atlas_get_tensor_count",
    "atlas_get_tensor_name",
    "atlas_get_tensor_index",
    "atlas_get_tokenizer",
    "atlas_has_binary_tokenizer",
    "atlas_tokenizer_preencode",
    "atlas_tokenizer_merge",
    "atlas_tokenizer_decode",
}

NO_PYTHON_WRAPPER_NEEDED = INTERNAL_C_APIS | {
    "atlas_decompress_all",
    "atlas_decompress_ttype5",
    "atlas_decompress_ttype7",
    "atlas_decompress_ffn",
    "atlas_quantize_ffn_to_i4",
    "atlas_save_cache",
    "atlas_load_cache",
    "atlas_prefetch_int8",
    "atlas_quantize_lmhead",
    "atlas_repack_experts",
    "atlas_set_layer_stride",
    "atlas_ensure_layer_idx",
    "atlas_forward",
    "atlas_set_seed",
    "atlas_load",
    "atlas_free",
    "atlas_get_config",
    "atlas_get_info",
}

CUSTOM_PYTHON_NAMES = {
    "atlas_load": "__init__",
    "atlas_free": "__del__",
    "atlas_set_seed": "set_seed",
    "atlas_generate": "generate_c",
    "atlas_generate_stream": "generate_stream",
    "atlas_forward": "forward",
    "atlas_tokenizer_decode": "_cpp_decode",
    "atlas_lmhead_gemv": "_lmhead_gemv",
    "atlas_quantize_lmhead": "_quantize_lmhead",
    "atlas_get_scale_emb": "scale_emb",
    "atlas_set_scale_depth_factor": "set_scale_depth",
}


def parse_c_apis(path):
    apis = {}
    with open(path, encoding="utf-8") as f:
        for m in re.finditer(r"ATLAS_API\s+\S+\s+(atlas_\w+)\s*\(", f.read()):
            apis[m.group(1)] = None
    return apis


def parse_dll_bindings(path):
    bindings = set()
    with open(path, encoding="utf-8") as f:
        for m in re.finditer(r"dll\.(atlas_\w+)\.(restype|argtypes)", f.read()):
            bindings.add(m.group(1))
    return bindings


def parse_python_methods(path):
    methods = set()
    with open(path, encoding="utf-8") as f:
        for m in re.finditer(r"^\s+def (\w+)\(self", f.read(), re.MULTILINE):
            methods.add(m.group(1))
    return methods


def main():
    errors = []

    c_apis = parse_c_apis(FFI_PATH)
    dll_bindings = parse_dll_bindings(PY_PATH)
    py_methods = parse_python_methods(PY_PATH)

    # 1) Jede C-API braucht ein DLL-Binding (ausser INTERNAL)
    for api, _ in sorted(c_apis.items()):
        if api in INTERNAL_C_APIS:
            continue
        if api not in dll_bindings:
            errors.append(f"  [NO BINDING] {api} in atlas_ffi.h hat kein ctypes-Binding in atlas_infer.py")

    # 2) Jedes DLL-Binding braucht eine Python-Methode (aussber NO_PYTHON_WRAPPER_NEEDED)
    for api in sorted(dll_bindings):
        if api in NO_PYTHON_WRAPPER_NEEDED:
            continue
        py_name = CUSTOM_PYTHON_NAMES.get(api, api[6:])
        if py_name not in py_methods:
            py_guess = api[6:]
            errors.append(f"  [NO METHOD] {api} ist per ctypes gebunden, aber AtlasModel.{py_guess}() existiert nicht")

    if errors:
        print("=" * 72)
        print("  API PARITY SCANNER — BLINDE FLECKEN ENTDECKT")
        print("=" * 72)
        for e in errors:
            print(e)
        print()
        print(f"  {len(errors)} Blindspot(s) — bitte Python-Wrapper nachreichen.")
        print("=" * 72)
        sys.exit(1)

    print("[API Parity] OK — alle C-APIs haben Python-Binding + Wrapper.")
    sys.exit(0)


if __name__ == "__main__":
    main()
