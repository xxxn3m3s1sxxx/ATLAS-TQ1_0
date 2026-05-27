#!/usr/bin/env python3
"""Atlas Inference Engine v2.6.4 — Falcon3/BitNet/Bonsai TQ1.0 inference."""
import ctypes, struct, os, sys, time, json, queue, threading, numpy as np
# v2.0.0: No more AutoTokenizer dependency — C++ binary tokenizer handles encode/decode

# ─── Load C++ DLL ────────────────────────────────────────────────────────
# Resolve OpenMP runtime conflicts (numpy MKL vs libomp)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_dll_name = "atlas.dll" if sys.platform == "win32" else "libatlas.so"
_dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _dll_name)
if not os.path.exists(_dll_path):
    _dll_path = os.environ.get("ATLAS_DLL", _dll_path)
dll = ctypes.CDLL(_dll_path)

dll.atlas_load.restype = ctypes.c_void_p
dll.atlas_load.argtypes = [ctypes.c_char_p]

dll.atlas_free.argtypes = [ctypes.c_void_p]

dll.atlas_get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]

dll.atlas_tensor_info.argtypes = [ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int)]

dll.atlas_tensor_data.restype = ctypes.POINTER(ctypes.c_uint8)
dll.atlas_tensor_data.argtypes = [ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int)]

dll.atlas_matmul_i8_f32.restype = None
dll.atlas_matmul_i8_f32.argtypes = [ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int]

dll.atlas_decompress_all.restype = None
dll.atlas_decompress_all.argtypes = [ctypes.c_void_p]

try:
    dll.atlas_decompress_ttype5.restype = None
    dll.atlas_decompress_ttype5.argtypes = [ctypes.c_void_p]
    _HAS_TTYPE5_DECOMPRESS = True
except AttributeError:
    _HAS_TTYPE5_DECOMPRESS = False

dll.atlas_decompress_ffn.restype = None
dll.atlas_decompress_ffn.argtypes = [ctypes.c_void_p]

dll.atlas_save_cache.restype = None
dll.atlas_save_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

dll.atlas_load_cache.restype = ctypes.c_int
dll.atlas_load_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

dll.atlas_prefetch_int8.restype = None
dll.atlas_prefetch_int8.argtypes = [ctypes.c_void_p]

dll.atlas_quantize_lmhead.restype = None
dll.atlas_quantize_lmhead.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

dll.atlas_set_use_f32_matmul.restype = None
dll.atlas_set_use_f32_matmul.argtypes = [ctypes.c_void_p, ctypes.c_int]

dll.atlas_set_use_ternary_matmul.restype = None
dll.atlas_set_use_ternary_matmul.argtypes = [ctypes.c_void_p, ctypes.c_int]

dll.atlas_set_use_packed_matmul.restype = None
dll.atlas_set_use_packed_matmul.argtypes = [ctypes.c_void_p, ctypes.c_int]

dll.atlas_set_use_hybrid_matmul.restype = None
dll.atlas_set_use_hybrid_matmul.argtypes = [ctypes.c_void_p, ctypes.c_int]

dll.atlas_set_rope_scale.restype = None
dll.atlas_set_rope_scale.argtypes = [ctypes.c_void_p, ctypes.c_float]

dll.atlas_set_base_seq_len.restype = None
dll.atlas_set_base_seq_len.argtypes = [ctypes.c_void_p, ctypes.c_int]

dll.atlas_reset_cache.restype = None
dll.atlas_reset_cache.argtypes = [ctypes.c_void_p]

dll.atlas_set_layer_stride.restype = None
dll.atlas_set_layer_stride.argtypes = [ctypes.c_void_p, ctypes.c_int]

dll.atlas_ensure_layer_idx.restype = None
dll.atlas_ensure_layer_idx.argtypes = [ctypes.c_void_p]

dll.atlas_set_num_threads.restype = None
dll.atlas_set_num_threads.argtypes = [ctypes.c_void_p, ctypes.c_int]

dll.atlas_lmhead_gemv.restype = None
dll.atlas_lmhead_gemv.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]

dll.atlas_forward.restype = None
dll.atlas_forward.argtypes = [
    ctypes.c_void_p,                              # model
    ctypes.POINTER(ctypes.c_float),               # hidden_states (in-place)
    ctypes.c_int,                                  # B
    ctypes.POINTER(ctypes.c_int),                  # positions
    ctypes.c_int,                                  # max_seq_len
    ctypes.c_int,                                  # seq_now
    ctypes.POINTER(ctypes.c_int),                  # layer_idx [n_layers * 9]
    ctypes.c_int,                                  # n_layers
]

dll.atlas_get_int8.restype = ctypes.POINTER(ctypes.c_int8)
dll.atlas_get_int8.argtypes = [ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))]

dll.atlas_get_tensor_count.restype = ctypes.c_int
dll.atlas_get_tensor_count.argtypes = [ctypes.c_void_p]
dll.atlas_get_tensor_name.restype = ctypes.c_int
dll.atlas_get_tensor_name.argtypes = [ctypes.c_void_p, ctypes.c_int,
    ctypes.c_char_p, ctypes.c_int]
dll.atlas_get_tensor_index.restype = ctypes.c_int
dll.atlas_get_tensor_index.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

dll.atlas_get_tokenizer.restype = ctypes.POINTER(ctypes.c_uint8)
dll.atlas_get_tokenizer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]

dll.atlas_has_binary_tokenizer.restype = ctypes.c_int
dll.atlas_has_binary_tokenizer.argtypes = [ctypes.c_void_p]

dll.atlas_tokenizer_preencode.restype = ctypes.c_int
dll.atlas_tokenizer_preencode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.c_int]

dll.atlas_tokenizer_merge.restype = ctypes.c_int
dll.atlas_tokenizer_merge.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int)]

dll.atlas_tokenizer_decode.restype = ctypes.c_int
dll.atlas_tokenizer_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
    ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

dll.atlas_rmsnorm_f32.restype = None
dll.atlas_rmsnorm_f32.argtypes = [ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_float]

dll.atlas_rope_f32.restype = None
dll.atlas_rope_f32.argtypes = [ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_float]

dll.atlas_attention_f32.restype = None
dll.atlas_attention_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # q
    ctypes.POINTER(ctypes.c_float),   # k
    ctypes.POINTER(ctypes.c_float),   # v
    ctypes.POINTER(ctypes.c_int),     # positions
    ctypes.POINTER(ctypes.c_int8),    # k_cache (int8)
    ctypes.POINTER(ctypes.c_float),   # k_scale_cache
    ctypes.POINTER(ctypes.c_int8),    # v_cache (int8)
    ctypes.POINTER(ctypes.c_float),   # v_scale_cache
    ctypes.c_int,     # max_seq_len
    ctypes.c_int,     # seq_now
    ctypes.c_int,     # B
    ctypes.c_int,     # n_heads
    ctypes.c_int,     # n_kv_heads
    ctypes.c_int,     # head_dim
    ctypes.c_float,   # rope_theta
    ctypes.c_float,   # rope_scale
    ctypes.POINTER(ctypes.c_float),   # output
    ctypes.c_void_p,  # q_norm_w (NULL = skip)
    ctypes.c_void_p,  # k_norm_w (NULL = skip)
]

dll.atlas_set_seed.restype = None
dll.atlas_set_seed.argtypes = [ctypes.c_uint64]

dll.atlas_sample.restype = None
dll.atlas_sample.argtypes = [ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
    ctypes.c_float, ctypes.c_int, ctypes.c_float]

dll.atlas_generate.restype = ctypes.c_int
dll.atlas_generate.argtypes = [ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_int, ctypes.c_float,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int)]

# v2.1.0: Streaming callback type + repetition_penalty + BPE-PQ
TOKEN_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p)
dll.atlas_generate_stream.argtypes = [ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_int, ctypes.c_float,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    TOKEN_CALLBACK, ctypes.c_void_p]
dll.atlas_generate_stream.restype = ctypes.c_int

# ─── Model class ─────────────────────────────────────────────────────────
class AtlasModel:
    def __init__(self, atlas_path, safetensors_path=None, model_dir=None,
                 use_packed_matmul=False, use_hybrid_matmul=False,
                 max_seq_len=4096, base_seq_len=None):
        self._safe_path = safetensors_path
        self._model_dir = model_dir
        self._atlas_path = atlas_path
        self._base_seq_len = base_seq_len or max_seq_len
        self.model_ptr = dll.atlas_load(atlas_path.encode())
        if not self.model_ptr:
            raise RuntimeError("Failed to load model")

        # Get model info
        nl = ctypes.c_int(); hd = ctypes.c_int(); id_ = ctypes.c_int()
        nh = ctypes.c_int(); nk = ctypes.c_int(); hdm = ctypes.c_int()
        vs = ctypes.c_int()
        dll.atlas_get_info(self.model_ptr, nl, hd, id_, nh, nk, hdm, vs)
        self.n_layers, self.hidden, self.inter = nl.value, hd.value, id_.value
        self.n_heads, self.n_kv_heads, self.head_dim = nh.value, nk.value, hdm.value
        self.vocab_size = vs.value
        self.rope_theta = self._get_rope_theta()

        # Get tensor names from C++ (v4+), fallback to safetensors for v3
        n = dll.atlas_get_tensor_count(self.model_ptr)
        if n > 0:
            self.tensor_names = []
            buf = ctypes.create_string_buffer(256)
            for i in range(n):
                dll.atlas_get_tensor_name(self.model_ptr, i, buf, 256)
                self.tensor_names.append(buf.value.decode())
        elif safetensors_path:
            # v3 fallback: read from safetensors
            from safetensors import safe_open
            with safe_open(safetensors_path, framework='np', device='cpu') as f:
                self.tensor_names = list(f.keys())
        else:
            raise RuntimeError("Model has no embedded names and no safetensors file provided")
        self.n_tensors = len(self.tensor_names)

        # Seed C++ PRNG
        dll.atlas_set_seed(np.random.randint(0, 2**31, dtype=np.int64) ^ int(time.time_ns()))

        print(f"[Atlas] {self.n_layers}L {self.hidden}H {self.inter}I "
              f"{self.n_heads}/{self.n_kv_heads} heads | "
              f"{self.n_tensors} tensors")

        # Cache tensor indices for fast lookup
        self._cache_indices()

        # v2.8.0: Persistent KV cache tracking for multi-turn
        self._cached_input_ids = []
        self._cache_valid = False

        # v2.4.0: Detect Qwen3 (Bonsai) models by rope_theta — YaRN NTK scaling
        if self.rope_theta >= 3000000.0:
            self.set_rope_scale(4.0)
            print(f"[Atlas] Qwen3 detected: YaRN RoPE scale=4.0 (theta={self.rope_theta:.0f})")

        if use_packed_matmul:
            # v1.3.1: Direct TQ1-packed matmul — keep raw TQ1 data, skip cache/decomp
            dll.atlas_set_use_packed_matmul(self.model_ptr, 1)
            print("[Atlas] Using direct TQ1-packed matmul (no int8 decompression)")
        elif use_hybrid_matmul:
            # v1.3.2: FFN int8 cache, QKV packed — best speed/RAM balance
            # For small models (1B, hidden<=2048) and block-scaled models (Bonsai),
            # decompress ALL tensors for f32_bypass (avoids uint8+128 signal collapse)
            dll.atlas_set_use_hybrid_matmul(self.model_ptr, 1)
            is_bitnet = any('attn_sub_norm' in name for name in self.idx)
            needs_f32 = is_bitnet or self.hidden <= 2048 or self.rope_theta >= 3000000.0
            if needs_f32:
                dll.atlas_decompress_all(self.model_ptr)
                if _HAS_TTYPE5_DECOMPRESS:
                    dll.atlas_decompress_ttype5(self.model_ptr)
                print("[Atlas] Full int8 (ttype=5 decompressed for f32 bypass)")
                dll.atlas_set_use_f32_matmul(self.model_ptr, 1)
            else:
                dll.atlas_decompress_ffn(self.model_ptr)
                print("[Atlas] Hybrid: FFN int8, QKV packed")
            dll.atlas_prefetch_int8(self.model_ptr)
        else:
            # Try loading int8 cache (mmap'd, instant). If not found, decompress + save.
            # Always decompress ttype=5 to int8 (no-op if none exist). Enables fast int8
            # matmul + f32_bypass to avoid uint8+128 signal collapse on block-scaled models.
            cache_loaded = dll.atlas_load_cache(self.model_ptr, self._atlas_path.encode())
            if cache_loaded:
                print("[Atlas] Loaded int8 weights from cache (mmap)")
            else:
                dll.atlas_decompress_all(self.model_ptr)
                if _HAS_TTYPE5_DECOMPRESS:
                    dll.atlas_decompress_ttype5(self.model_ptr)
                print("[Atlas] TQ1 tensors decoded to int8")
                dll.atlas_save_cache(self.model_ptr, self._atlas_path.encode())
            # Prefetch int8 data into physical RAM (page-in mmap or fresh decompress)
            dll.atlas_prefetch_int8(self.model_ptr)
            # f32_bypass: for small, block-scaled, or BitNet models
            is_bitnet = any('attn_sub_norm' in name for name in self.idx)
            if is_bitnet or self.hidden <= 2048 or self.rope_theta >= 3000000.0:
                dll.atlas_set_use_f32_matmul(self.model_ptr, 1)

        # Max sequence length for KV cache ring buffer (v2.5.0: configurable)
        self.max_seq_len = max_seq_len
        # Set base sequence length for NTK context extension
        dll.atlas_set_base_seq_len(self.model_ptr, self._base_seq_len)

        # Quantize lm_head to per-row int8 in C++ (saves ~1.1 GB vs full fp32)
        # For tie embeddings (Qwen3/Bonsai), lm_head = embed_tokens
        print("[Atlas] Quantizing lm_head to int8...")
        t0 = time.time()
        idx_lm_head = self.idx.get("lm_head.weight", -1)
        if idx_lm_head >= 0:
            dll.atlas_quantize_lmhead(self.model_ptr, idx_lm_head, 0)
        else:
            idx_embed = self.idx.get("model.embed_tokens.weight", -1)
            if idx_embed >= 0:
                dll.atlas_quantize_lmhead(self.model_ptr, idx_embed, 1)  # keep embed data
        print(f"[Atlas] lm_head quantized ({time.time()-t0:.1f}s)")

        # Load tokenizer — prefer v6 binary (C++), fallback to v5 JSON (Python)
        self._tok = None
        self._use_cpp_tokenizer = False
        self._eos_id = 0
        self._chat_template = None
        # Phi-3: instruct model, head_dim=96, small vocab, <|role|> format with <|end|>
        self._is_phi3 = (self.head_dim == 96 and self.vocab_size <= 40000)
        # TriLM: base LLaMA with no chat format
        self._is_trilm = (not self._is_phi3 and self.vocab_size <= 60000 and self.head_dim <= 128)
        # Qwen3/Bonsai detection: not TriLM, and head_dim<=128 or large vocab (>131k)
        self._is_qwen3 = (not self._is_trilm and not self._is_phi3 and (self.head_dim <= 128 or self.vocab_size > 131072))
        self._enable_thinking = True  # Qwen3 supports thinking; Bonsai does not

        # Read chat_template from embedded config JSON (present in both v5 and v6)
        tok_size = ctypes.c_int()
        tok_ptr = dll.atlas_get_tokenizer(self.model_ptr, tok_size)
        if tok_ptr and tok_size.value > 0:
            raw = ctypes.string_at(tok_ptr, tok_size.value)
            try:
                pos = 0
                js_size = struct.unpack_from('<I', raw, pos)[0]; pos += 4
                pos += js_size  # skip tokenizer.json raw bytes
                cfg_size = struct.unpack_from('<I', raw, pos)[0]; pos += 4
                if cfg_size > 0:
                    cfg = json.loads(raw[pos:pos+cfg_size].decode('utf-8'))
                    self._chat_template = cfg.get('chat_template')
                    # Bonsai has no embedded template → no thinking support
                    if not self._chat_template and self._is_qwen3:
                        self._enable_thinking = False
                    # Get EOS token ID from config
                    eos_cfg = cfg.get('eos_token')
                    if eos_cfg and isinstance(eos_cfg, dict) and 'id' in eos_cfg:
                        self._eos_id = eos_cfg['id']
            except Exception:
                pass

        # Fallback: read EOS/PAD from file header bytes 45-52
        if self._eos_id == 0:
            try:
                with open(self._atlas_path, 'rb') as f:
                    f.seek(45)
                    hdr = f.read(8)
                    if len(hdr) == 8:
                        self._eos_id = struct.unpack('<I', hdr[:4])[0]
            except Exception:
                pass

        # Check for v6 binary tokenizer (C++ decode)
        if dll.atlas_has_binary_tokenizer(self.model_ptr):
            self._use_cpp_tokenizer = True
            print("[Atlas] Using C++ tokenizer (v6 binary)")
        # Always load Python tokenizer for encoding (handles GPT-2 ByteLevel pre-tokenization)
        if tok_ptr and tok_size.value > 0:
            if not self._use_cpp_tokenizer:
                print("[Atlas] Also loading Python tokenizer for encode (v5 fallback)")
            raw = ctypes.string_at(tok_ptr, tok_size.value)
            try:
                pos = 0
                js_size = struct.unpack_from('<I', raw, pos)[0]; pos += 4
                from tokenizers import Tokenizer
                self._tok = Tokenizer.from_buffer(raw[pos:pos+js_size])
            except Exception as e:
                print(f"[Atlas] Python tokenizer load failed: {e}")
        else:
            if not self._use_cpp_tokenizer:
                print("[Atlas] No embedded tokenizer found")


    def __del__(self):
        if self.model_ptr:
            dll.atlas_free(self.model_ptr)
            self.model_ptr = None

    def set_seed(self, seed):
        """Set C++ PRNG seed for deterministic sampling."""
        dll.atlas_set_seed(ctypes.c_uint64(seed))

    def set_use_f32_matmul(self, enable=True):
        """v1.3.2: Enable/disable f32 bypass (no activation quantization)."""
        dll.atlas_set_use_f32_matmul(self.model_ptr, 1 if enable else 0)

    def set_rope_scale(self, scale=1.0):
        """v2.4.0: Set YaRN NTK RoPE scaling factor (4.0 for Bonsai-4B)."""
        dll.atlas_set_rope_scale(self.model_ptr, ctypes.c_float(scale))

    def set_base_seq_len(self, seq_len):
        """v2.5.0: Set base sequence length for NTK context extension.
        E.g., 4096 for Falcon3, 2048 for Bonsai-1.7B, 8192 for Bonsai-4B.
        When max_seq_len > base_seq_len, NTK-aware frequency scaling is applied."""
        dll.atlas_set_base_seq_len(self.model_ptr, seq_len)
        self._base_seq_len = seq_len

    def reset_cache(self):
        """v2.6.0: Reset KV cache — zeros all cache data without freeing allocation.
        Call between conversations to prevent context leakage across sessions."""
        dll.atlas_reset_cache(self.model_ptr)
        self._cached_input_ids = []
        self._cache_valid = False

    def set_max_seq_len(self, seq_len):
        """v2.5.0: Set max sequence length (ring buffer window size).
        Larger values allocate more KV cache memory but enable longer context.
        When > base_seq_len, NTK context extension is auto-applied."""
        self.max_seq_len = seq_len

    def set_use_ternary_matmul(self, enable=True):
        """v1.3.0: Enable/disable ternary-add kernel (vpsignb, no multiplication)."""
        dll.atlas_set_use_ternary_matmul(self.model_ptr, 1 if enable else 0)

    def set_use_packed_matmul(self, enable=True):
        """v1.3.1: Enable/disable direct TQ1-packed matmul (no decompression, 5× less weight reads)."""
        dll.atlas_set_use_packed_matmul(self.model_ptr, 1 if enable else 0)

    def set_use_hybrid_matmul(self, enable=True):
        """v1.3.2: Enable/disable hybrid mode (FFN int8, QKV packed)."""
        dll.atlas_set_use_hybrid_matmul(self.model_ptr, 1 if enable else 0)

    def set_num_threads(self, n):
        """v1.3.1: Set OpenMP thread count (0 = default). Lowers CPU load on shared systems."""
        dll.atlas_set_num_threads(self.model_ptr, n)

    def _cache_indices(self):
        self.idx = {}
        for i, name in enumerate(self.tensor_names):
            self.idx[name] = i
        # Cache frequently-used tensors
        self._embed_w = self._load_embed("model.embed_tokens.weight")
        self._norm_w = self._load_weight_f16("model.norm.weight")
        self._tq1_cache = {}
        self._f16_cache = {}
        self._i8_cache = {}
        # Call C ensure_layer_idx which auto-detects stride + model_arch
        dll.atlas_ensure_layer_idx(self.model_ptr)
        # Read back the stride
        has_qk_norm = "model.layers.0.self_attn.q_norm.weight" in self.idx
        has_sub_norm = "model.layers.0.self_attn.attn_sub_norm.weight" in self.idx
        stride = 11 if (has_qk_norm or has_sub_norm) else 9
        # Build flat index array for atlas_forward (fused C++ layer loop)
        idx = self.idx
        per_layer = ['input_layernorm.weight',
            'self_attn.q_proj.weight', 'self_attn.k_proj.weight',
            'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
            'post_attention_layernorm.weight',
            'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight',
            'self_attn.q_norm.weight', 'self_attn.k_norm.weight']
        if has_sub_norm:
            # BitNet: attn_sub_norm + ffn_sub_norm instead of QK-Norm
            per_layer[9] = 'self_attn.attn_sub_norm.weight'
            per_layer[10] = 'mlp.ffn_sub_norm.weight'
        elif not has_qk_norm:
            # Falcon3: remove q_norm/k_norm from per_layer
            per_layer = [n for n in per_layer if 'q_norm' not in n and 'k_norm' not in n]
        arrs = []
        for L in range(self.n_layers):
            for n in per_layer:
                arrs.append(idx[f'model.layers.{L}.{n}'])
        self._layer_idx_arr = np.array(arrs, dtype=np.int32)
        self._stride = stride

    def _get_rope_theta(self):
        with open(self._atlas_path, 'rb') as f:
            f.read(21)
            return struct.unpack('<d', f.read(8))[0]

    def _load_weight_f16(self, name, shape=None):
        """Load a float16 weight tensor from atlas, return as float32 numpy."""
        idx = self.idx.get(name)
        if idx is None: return None
        sz = ctypes.c_int()
        ptr = dll.atlas_tensor_data(self.model_ptr, idx, sz)
        if not ptr: return None
        arr = np.ctypeslib.as_array(ptr, shape=(sz.value,)).view(np.float16)
        if shape:
            arr = arr.reshape(shape)
        return arr.copy().astype(np.float32)

    def _load_embed(self, name):
        """Load embedding as fp16 (not fp32) to save 1.6GB RAM."""
        idx = self.idx.get(name)
        if idx is None: return None
        sz = ctypes.c_int()
        ptr = dll.atlas_tensor_data(self.model_ptr, idx, sz)
        if not ptr: return None
        flat = np.ctypeslib.as_array(ptr, shape=(sz.value,)).view(np.float16).copy()
        return flat.reshape(self.vocab_size, self.hidden)

    def _load_tq1(self, name):
        """Load TQ1 packed data and scale for a weight tensor. Cached."""
        if name in self._tq1_cache:
            return self._tq1_cache[name]
        idx = self.idx.get(name)
        if idx is None: return None, None, None
        sz = ctypes.c_int()
        ptr = dll.atlas_tensor_data(self.model_ptr, idx, sz)
        if not ptr: return None, None, None
        raw = np.ctypeslib.as_array(ptr, shape=(sz.value,))
        scale_raw = struct.unpack('<H', raw[:2].tobytes())[0]
        scale = np.frombuffer(struct.pack('<H', scale_raw), dtype=np.float16)[0].item()
        tt = ctypes.c_int(); rd = ctypes.c_int(); cd = ctypes.c_int()
        dll.atlas_tensor_info(self.model_ptr, idx, tt, rd, cd)
        packed_cols = cd.value // 5
        packed = np.frombuffer(raw[2:].tobytes(), dtype=np.uint8).copy()
        result = (packed, packed_cols, rd.value, scale)
        self._tq1_cache[name] = result
        return result

    def _matmul_tq1(self, data_flat, packed_cols, rows, act, scale):
        """TQ1 matmul with BitNet activation quantization (pure numpy)."""
        orig_shape = act.shape
        act = act.reshape(-1, orig_shape[-1])
        max_abs = np.max(np.abs(act), axis=-1, keepdims=True)
        max_abs = np.maximum(max_abs, 1e-5)
        act_scale = 127.0 / max_abs
        act_q = np.round(act * act_scale).astype(np.float32)

        need = packed_cols * 5
        if act_q.shape[1] < need:
            pad = np.zeros((act_q.shape[0], need), dtype=np.float32)
            pad[:, :act_q.shape[1]] = act_q
            act_q = pad

        # Decode TQ1 bytes and compute dot product in Python
        # tq1_decode[byte] = [t0, t1, t2, t3, t4] ∈ {-1,0,1}
        dec = np.array([((b // 3**i) % 3) - 1 for b in range(256) for i in range(5)],
                       dtype=np.int8).reshape(256, 5)
        out = np.zeros((act_q.shape[0], rows), dtype=np.float32)
        for r in range(rows):
            row_bytes = data_flat[r * packed_cols:(r + 1) * packed_cols]
            w = dec[row_bytes].reshape(-1)  # int8 weight row
            dot = act_q @ w  # [B] float32 dot product
            out[:, r] = dot

        # Reorder: atlas order (ur*4+q) → HF unpack order (q*rows_packed+ur)
        rows_packed = rows // 4
        out = out.reshape(out.shape[0], rows_packed, 4).transpose(0, 2, 1).reshape(out.shape[0], rows)

        out *= max_abs / (127.0 * scale)
        out = out.reshape(*orig_shape[:-1], rows)
        return out

    def _load_int8(self, name):
        """Load int8 data from C++ (decompressed at load time)."""
        if name in self._i8_cache:
            return self._i8_cache[name]
        idx = self.idx.get(name)
        if idx is None:
            self._i8_cache[name] = (None, None, None, None, None)
            return self._i8_cache[name]
        rows = ctypes.c_int()
        input_dim = ctypes.c_int()
        scale = ctypes.c_float()
        rs_ptr = ctypes.POINTER(ctypes.c_int32)()
        i8_ptr = dll.atlas_get_int8(
            self.model_ptr, idx, rows, input_dim, scale, rs_ptr)
        if not i8_ptr:
            self._i8_cache[name] = (None, None, None, None, None)
            return self._i8_cache[name]
        r, d = rows.value, input_dim.value
        # Create numpy views into DLL memory (no copy)
        i8 = np.ctypeslib.as_array(i8_ptr, shape=(r, d))
        row_sums = np.ctypeslib.as_array(rs_ptr, shape=(r,))
        result = (i8, r, d, scale.value, row_sums)
        self._i8_cache[name] = result
        return result

    def _matmul_int8(self, i8_data, rows, input_dim, scale, row_sums, act):
        """Int8 matmul via C++ AVX2 maddubs kernel."""
        orig_shape = act.shape
        act = act.reshape(-1, orig_shape[-1])

        # Pad activation to packed_cols*5 (TQ1 packer pads to multiple of 5)
        if act.shape[1] < input_dim:
            pad = np.zeros((act.shape[0], input_dim), dtype=np.float32)
            pad[:, :act.shape[1]] = act
            act = pad

        # Quantize activations to int8 [-127, 127]
        max_abs = np.max(np.abs(act), axis=-1, keepdims=True)
        max_abs = np.maximum(max_abs, 1e-5)
        act_scale = 127.0 / max_abs
        act_q = np.clip(np.round(act * act_scale), -127, 127).astype(np.int8)

        # Convert to uint8 with +128 offset (for maddubs unsigned operand)
        act_u8 = (act_q.astype(np.int32) + 128).clip(0, 255).astype(np.uint8)

        out = np.zeros((act.shape[0], rows), dtype=np.float32)
        dll.atlas_matmul_i8_f32(
            rows, input_dim,
            i8_data.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            act_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            row_sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            act.shape[0])

        # Reorder: atlas order → HF order
        rows_packed = rows // 4
        out = out.reshape(out.shape[0], rows_packed, 4).transpose(0, 2, 1).reshape(out.shape[0], rows)

        # Dequantize
        out *= max_abs / (127.0 * scale)
        out = out.reshape(*orig_shape[:-1], rows)
        return out

    def _matmul_f16(self, name, act):
        """Float32 activation × cached weight. Falls back to int8 GEMV for quantized lm_head."""
        if name not in self._f16_cache:
            idx = self.idx.get(name)
            if idx is None: return None
            tt = ctypes.c_int(); rd = ctypes.c_int(); cd = ctypes.c_int()
            dll.atlas_tensor_info(self.model_ptr, idx, tt, rd, cd)
            if tt.value == 0:
                return self._matmul_tq1(*self._load_tq1(name), act)
            sz = ctypes.c_int()
            ptr = dll.atlas_tensor_data(self.model_ptr, idx, sz)
            if not ptr:  # tensor consumed (quantized lm_head)
                return self._lmhead_gemv(act)
            w = np.ctypeslib.as_array(ptr, shape=(sz.value,)).view(np.float16).reshape(rd.value, act.shape[-1])
            if w.shape[0] * w.shape[1] > 100000:  # lm_head: 131072x3072 -> convert to fp32
                self._f16_cache[name] = np.ascontiguousarray(w, dtype=np.float32)
            else:
                self._f16_cache[name] = w
        w = self._f16_cache[name]
        if act.ndim == 1:
            return act @ w.T
        B = act.shape[0]
        flat = act.reshape(-1, act.shape[-1])
        result = flat @ w.T
        return result.reshape(B, -1, w.shape[0]) if act.ndim > 2 else result

    def _rmsnorm(self, x, weight_name, eps=1e-6):
        if weight_name not in self._f16_cache:
            idx = self.idx.get(weight_name)
            if idx is None: return x
            sz = ctypes.c_int()
            ptr = dll.atlas_tensor_data(self.model_ptr, idx, sz)
            if not ptr: return x
            self._f16_cache[weight_name] = ptr
        out = np.zeros_like(x)
        dll.atlas_rmsnorm_f32(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._f16_cache[weight_name],
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            x.shape[-1], ctypes.c_float(eps))
        return out

    def _apply_rope(self, q, k, position):
        """Apply RoPE to q and k in-place."""
        dll.atlas_rope_f32(
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.n_heads, self.n_kv_heads, self.head_dim,
            position, ctypes.c_float(self.rope_theta))

    def _silu(self, x):
        return x * (1.0 / (1.0 + np.exp(-x)))

    def forward_layer(self, x, layer_idx, positions, use_kvcache=True):
        """Forward one transformer layer via atlas_forward (fused, n_layers=1)."""
        B = len(positions) if isinstance(positions, (list, np.ndarray)) else positions.shape[0]
        positions_arr = np.array(positions, dtype=np.int32)
        seq_now = int(positions_arr.max()) + 1 if use_kvcache else B

        out = x.copy()
        stride = self._stride if hasattr(self, '_stride') else 9
        idx_slice = self._layer_idx_arr[layer_idx * stride : (layer_idx + 1) * stride].copy()
        dll.atlas_forward(
            self.model_ptr,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B,
            positions_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.max_seq_len, seq_now,
            idx_slice.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            1)
        return out

    def forward(self, tokens, start_pos=0):
        """Full forward pass — all layers fused in C++, final RMSNorm + LM head in Python."""
        B, seq_len = tokens.shape
        h = self._embed_w[tokens].astype(np.float32)
        h = h.reshape(-1, self.hidden)  # [B*seq_len, H]
        n = B * seq_len

        positions = np.array([start_pos + p for b in range(B) for p in range(seq_len)], dtype=np.int32)
        seq_now = start_pos + seq_len

        dll.atlas_forward(
            self.model_ptr,
            h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.max_seq_len, seq_now,
            self._layer_idx_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.n_layers)

        h_norm = np.array([self._rmsnorm(h[b], "model.norm.weight") for b in range(n)])
        output_logits = self._lmhead_gemv(h_norm).reshape(B, seq_len, self.vocab_size)
        return output_logits

    @staticmethod
    def _sample(logits, temperature=1.0, top_k=0, top_p=0.0):
        """Sample next token from logits with temperature, top-k, top-p."""
        if temperature == 0.0:
            return int(np.argmax(logits))

        probs = np.exp(logits / temperature)
        probs /= probs.sum()

        if top_k > 0:
            idx = np.argpartition(probs, -top_k)[-top_k:]
            mask = np.zeros_like(probs)
            mask[idx] = 1.0
            probs *= mask
            probs /= probs.sum()

        if top_p > 0.0:
            idx = np.argsort(probs)[::-1]
            cum = np.cumsum(probs[idx])
            cutoff = np.searchsorted(cum, top_p, side='right') + 1
            mask = np.zeros_like(probs)
            mask[idx[:cutoff]] = 1.0
            probs *= mask
            probs /= probs.sum()

        return int(np.random.choice(len(probs), p=probs))

    def generate(self, prompt, max_new_tokens=50, temperature=1.0,
                 top_k=40, top_p=0.9):
        try:
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            text = self._apply_chat_template(prompt)
            if self._use_cpp_tokenizer:
                input_ids = self._cpp_encode(text)
                eos_id = self._eos_id
            elif self._tok is not None:
                input_ids = self._tok.encode(text).ids
                eos_id = self._eos_id
            else:
                return "[TOKENIZER ERROR: No tokenizer available]"
        except Exception as e:
            return f"[TOKENIZER ERROR: {e}]"

        stop_tokens = ['<|user|>', '<|system|>', '<|end|>']

        full_logits = self.forward(np.array([input_ids], dtype=np.int32))
        logits = full_logits[0, -1, :]

        output = []
        for step in range(max_new_tokens):
            if step == 0:
                current_logits = logits
            else:
                h = self._get_embedding(next_token)
                pos = np.array([len(input_ids) + step - 1], dtype=np.int32)
                dll.atlas_forward(
                    self.model_ptr,
                    h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    1, pos.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    self.max_seq_len, len(input_ids) + step,
                    self._layer_idx_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    self.n_layers)
                h_norm = self._rmsnorm(h.flatten(), "model.norm.weight")
                current_logits = self._lmhead_gemv(h_norm.reshape(1, -1)).flatten()

            next_token = self._sample(current_logits, temperature, top_k, top_p)
            output.append(next_token)

            if eos_id is not None and next_token == eos_id:
                break
            if self._use_cpp_tokenizer:
                decoded = self._cpp_decode(output, skip_special=False)
            else:
                decoded = self._tok.decode(output, skip_special_tokens=False)
            if any(stop in decoded for stop in stop_tokens):
                break

        if self._use_cpp_tokenizer:
            text = self._cpp_decode(output)
        else:
            text = self._tok.decode(output, skip_special_tokens=True)
        for stop in stop_tokens:
            idx = text.find(stop)
            if idx >= 0:
                text = text[:idx]
        return text

    def set_system_prompt(self, prompt):
        """Set system prompt injected before every user message."""
        self._system_prompt = prompt

    def generate_c(self, prompt, max_new_tokens=50, temperature=0.7,
                   top_k=40, top_p=0.9, repetition_penalty=1.0,
                   max_seq_len=None, min_new_tokens=20, cache_enabled=True):
        """Generate via atlas_generate (single C call, v1.2.1).
        max_seq_len: override KV cache window (default: self.max_seq_len).
        min_new_tokens: suppress EOS for first N generated tokens.
        cache_enabled: persistent KV cache for multi-turn (default: True)."""
        try:
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            sysp = getattr(self, '_system_prompt', None)
            if sysp and not any(m.get('role') == 'system' for m in prompt):
                prompt = [{"role": "system", "content": sysp}] + prompt
            text = self._apply_chat_template(prompt)
            input_ids = self._cpp_encode(text)
        except Exception as e:
            return f"[TOKENIZER ERROR: {e}]"

        n_input = len(input_ids)
        in_arr = (ctypes.c_int * n_input)(*input_ids)
        out_arr = (ctypes.c_int * max_new_tokens)()

        # Determine cache_offset from persistent KV cache
        cache_offset = 0
        if cache_enabled and self._cache_valid:
            old_len = len(self._cached_input_ids)
            if n_input > old_len and list(input_ids[:old_len]) == self._cached_input_ids:
                cache_offset = old_len

        n_gen = dll.atlas_generate(
            self.model_ptr, in_arr, n_input,
            max_seq_len if max_seq_len is not None else self.max_seq_len,
            max_new_tokens,
            float(temperature), int(top_k), float(top_p),
            float(repetition_penalty),
            int(min_new_tokens),
            int(cache_offset),
            out_arr)

        if n_gen < 0:
            return "[ATLAS: atlas_generate failed]"

        output = list(out_arr[:n_gen])
        decoded = self._cpp_decode(output)
        stops = ['<|user|>', '<|system|>', '<|end|>', '<|im_end|>']
        for stop in stops:
            idx = decoded.find(stop)
            if idx >= 0:
                decoded = decoded[:idx]

        # Update persistent cache state
        if cache_enabled and n_gen > 0:
            self._cached_input_ids = list(input_ids) + output
            self._cache_valid = True
        elif not cache_enabled:
            self._cached_input_ids = []
            self._cache_valid = False

        return decoded

    def generate_stream(self, prompt, max_new_tokens=200, temperature=0.7,
                        top_k=40, top_p=0.9, repetition_penalty=1.0,
                        max_seq_len=None, min_new_tokens=20, cache_enabled=True):
        """Streaming generator — yields token IDs as they are produced.
        max_seq_len: override KV cache window (default: self.max_seq_len).
        min_new_tokens: suppress EOS for first N generated tokens.
        cache_enabled: persistent KV cache for multi-turn (default: True).

        Usage:
            for token_id in model.generate_stream("Write an article"):
                token_text = model._cpp_decode([token_id])
                send_to_frontend(token_text)

        Yields int token IDs. Caller is responsible for decoding.
        """
        try:
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            sysp = getattr(self, '_system_prompt', None)
            if sysp and not any(m.get('role') == 'system' for m in prompt):
                prompt = [{"role": "system", "content": sysp}] + prompt
            text = self._apply_chat_template(prompt)
            input_ids = self._cpp_encode(text)
        except Exception as e:
            return

        n_input = len(input_ids)
        in_arr = (ctypes.c_int * n_input)(*input_ids)

        # Determine cache_offset from persistent KV cache
        cache_offset = 0
        if cache_enabled and self._cache_valid:
            old_len = len(self._cached_input_ids)
            if n_input > old_len and list(input_ids[:old_len]) == self._cached_input_ids:
                cache_offset = old_len

        # Thread-safe token queue
        q = queue.Queue()
        collected = []
        def on_token(tid, _):
            q.put(tid)
            collected.append(tid)
        cb = TOKEN_CALLBACK(on_token)

        t = threading.Thread(target=dll.atlas_generate_stream,
            args=(self.model_ptr, in_arr, n_input,
                  max_seq_len if max_seq_len is not None else self.max_seq_len,
                  max_new_tokens,
                  temperature, top_k, top_p,
                  float(repetition_penalty),
                  int(min_new_tokens),
                  int(cache_offset),
                  cb, None))
        t.start()

        n_gen = 0
        while t.is_alive() or not q.empty():
            try:
                tid = q.get(timeout=0.1)
                yield tid
                n_gen += 1
            except queue.Empty:
                continue

        # Update persistent cache state
        if cache_enabled and collected:
            self._cached_input_ids = list(input_ids) + collected
            self._cache_valid = True
        elif not cache_enabled:
            self._cached_input_ids = []
            self._cache_valid = False

    def _cpp_encode(self, text):
        """Encode text via Python tokenizers library (no transformers dependency).

        Uses `tokenizers.Tokenizer` directly for the full encode (pre-tokenization + BPE),
        then C++ for the decode path. This eliminates the `transformers` package dependency
        while keeping the lightweight `tokenizers` C extension for correct GPT-2 ByteLevel
        pre-tokenization.
        """
        if self._tok is None:
            # Fallback: use pure C++ preencode (no BPE, raw byte encoding)
            text_bytes = text.encode('utf-8')
            max_ids = len(text_bytes) + 256
            ids_arr = (ctypes.c_int * max_ids)()
            n = dll.atlas_tokenizer_preencode(
                self.model_ptr, text_bytes, len(text_bytes), ids_arr, max_ids)
            if n < 0:
                raise RuntimeError("C++ tokenizer preencode failed")
            return list(ids_arr[:n])

        # Use Python tokenizers library for full encode (handles GPT-2 ByteLevel correctly)
        return self._tok.encode(text).ids

    def _cpp_decode(self, ids, skip_special=True):
        """Decode token IDs. Uses C++ v6 binary tokenizer; falls back to Python for v5."""
        if not ids:
            return ""
        if not self._use_cpp_tokenizer and self._tok is not None:
            return self._tok.decode(ids, skip_special_tokens=skip_special)
        n = len(ids)
        ids_arr = (ctypes.c_int * n)(*ids)
        max_out = n * 16 + 64
        out_buf = ctypes.create_string_buffer(max_out)
        n_bytes = dll.atlas_tokenizer_decode(
            self.model_ptr, ids_arr, n, out_buf, max_out)
        if n_bytes < 0:
            raise RuntimeError("C++ tokenizer decode failed")
        raw = bytes(out_buf[:n_bytes])
        # GPT-2 ByteLevel decoder: convert Unicode byte tokens back to raw bytes
        # Build reverse bytes_to_unicode() mapping
        bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
        unicode_to_byte = {}
        n_count = 0
        for b in range(256):
            if b in bs:
                unicode_to_byte[chr(b)] = b
            else:
                unicode_to_byte[chr(256 + n_count)] = b
                n_count += 1
        # Decode: map each char through unicode_to_byte, then interpret as Latin-1
        result_bytes = bytearray()
        for ch in raw.decode('utf-8', errors='replace'):
            b = unicode_to_byte.get(ch)
            if b is not None:
                result_bytes.append(b)
            else:
                result_bytes.append(ord(ch) if ord(ch) < 256 else 32)  # fallback to space
        text = result_bytes.decode('utf-8', errors='replace')
        if skip_special:
            stops = ['<|endoftext|>', '<|im_end|>', '<|pad|>']
            for s in stops:
                text = text.replace(s, '')
        return text

    def _apply_chat_template(self, messages, add_generation_prompt=True):
        """Chat template (no Jinja2/transformers dependency).

        Matches model-specific format:
        - Falcon3: <|role|>\ncontent\n  (EOS: <|endoftext|>)
        - Qwen3:   <|im_start|>role\ncontent<|im_end|>\n
        - TriLM:   plain text (base model, no chat format)
        Bonsai: Qwen3 format + enable_thinking=False (empty <think> block).
        """
        # TriLM: base LLaMA model — no chat formatting, just concatenate content
        if self._is_trilm:
            result = '\n'.join(m.get('content', '') for m in messages)
            return result + '\n' if add_generation_prompt else result

        # Phi-3: <|role|>\ncontent<|end|>\n
        if self._is_phi3:
            eos = '<|end|>'
            result = ''
            for msg in messages:
                role = msg['role']
                content = msg.get('content', '')
                result += f"<|{role}|>\n{content}{eos}\n"
            if add_generation_prompt:
                result += '<|assistant|>\n'
            return result

        if self._is_qwen3:
            eos = '<|im_end|>'
            result = ""
            for msg in messages:
                role = msg['role']
                content = msg.get('content', '')
                result += f"<|im_start|>{role}\n{content}{eos}\n"
            if add_generation_prompt:
                result += '<|im_start|>assistant\n'
                # Suppress thinking for non-thinking models (Bonsai)
                enable = getattr(self, '_enable_thinking', True)
                if not enable:
                    result += '<think>\n\n</think>\n\n'
            return result

        eos = '<|endoftext|>'
        result = ""
        n = len(messages)
        for i, msg in enumerate(messages):
            role = msg['role']
            content = msg.get('content', '')
            if role == 'system':
                result += f"<|system|>\n{content}\n"
            elif role == 'user':
                result += f"<|user|>\n{content}\n"
            elif role == 'assistant':
                is_last = (i == n - 1)
                if not is_last:
                    result += f"<|assistant|>\n{content}{eos}\n"
                else:
                    result += f"<|assistant|>\n{content}{eos}"
        if add_generation_prompt:
            result += '<|assistant|>\n'
        return result

    def _lmhead_gemv(self, h_norm):
        """Logits via C++ int8 GEMV (per-row symmetric int8 lm_head + u8 activation)."""
        orig = h_norm.shape
        h = h_norm.reshape(-1, self.hidden)
        B = h.shape[0]
        out = np.empty(B * self.vocab_size, dtype=np.float32)
        dll.atlas_lmhead_gemv(
            self.model_ptr,
            h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B)
        return out.reshape(*orig[:-1], self.vocab_size)

    def _get_embedding(self, token_id):
        return self._embed_w[token_id].astype(np.float32).reshape(1, self.hidden)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python atlas_infer.py <atlas.tq1> [prompt]")
        print("  atlas.tq1  — path to packed .atlas file (v5, embedded tokenizer)")
        print("  prompt     — optional prompt (default: 'Say hello')")
        sys.exit(1)
    atlas_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Say hello"

    print(f"[Atlas] Loading {atlas_path}...")
    t0 = time.time()
    model = AtlasModel(atlas_path)
    print(f"[Atlas] Loaded in {time.time() - t0:.1f}s")

    print(f"[Atlas] Generating: {prompt}")
    t0 = time.time()
    text = model.generate_c(prompt, max_new_tokens=20, temperature=0.0)
    print(f"[Atlas] Output: {text}")
    print(f"[Atlas] Time: {time.time() - t0:.1f}s")
