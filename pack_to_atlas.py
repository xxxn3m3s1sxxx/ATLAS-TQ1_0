#!/usr/bin/env python3
"""ATLAS packer — pack supported HF models to TQ1.0 format.

Usage:
  python pack_to_atlas.py <model_dir> <output.atlas>
  python pack_to_atlas.py <model_dir> <output.atlas> --block-size 128
  python pack_to_atlas.py <model_dir> <output.atlas> --ttype 5
  python pack_to_atlas.py <model_dir> <output.atlas> --no-v6

Auto-detects architecture from config.json model_type.
Supports: Falcon3, Qwen3/Bonsai, BitNet b1.58, TriLM, Llama, Mistral, Gemma, Phi.
"""
import argparse, json, os, struct, sys
import numpy as np
import torch
from safetensors import safe_open

from atlas_packer_mappings import detect_arch

TQ1_MUL = np.array([1, 3, 9, 27, 81], dtype=np.uint32)


# ═══════════════════════════════════════════════════════════════════════
# Tensor readers
# ═══════════════════════════════════════════════════════════════════════

def _read_bf16_bytes(data):
    """Convert raw BF16 bytes → float16 numpy array."""
    u32 = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16
    return u32.view(np.float32)


def _clean_embed_rows(fp32, name):
    """Zero overflow values in embed_tokens tensors (BitNet quirk)."""
    if "embed_tokens" not in name or fp32.ndim != 2:
        return fp32
    H = fp32.shape[1]
    for r in range(fp32.shape[0]):
        overflow = np.abs(fp32[r]) > 1e10
        n_ov = np.sum(overflow)
        if n_ov > 0:
            fp32[r][overflow] = 0.0
    return fp32


class SafetensorsReader:
    """Read tensors from safetensors or PyTorch .bin files."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._shards_np = {}  # path → numpy safe_open handle
        self._shards_pt = {}  # path → torch safe_open handle
        self._torch_state = None  # torch full state_dict (for .bin fallback)

        # Build weight map
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                idx = json.load(f)
            self.weight_map = idx["weight_map"]
            return

        sf_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(sf_path):
            self.weight_map = {}
            with safe_open(sf_path, framework="np") as sf:
                for k in sf.keys():
                    self.weight_map[k] = "model.safetensors"
            return

        # Fallback: PyTorch .bin file (MiniCPM, etc.)
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(bin_path):
            import torch
            print(f"  Loading PyTorch .bin: {bin_path}")
            self._torch_state = torch.load(bin_path, map_location="cpu", weights_only=True)
            self.weight_map = {k: "pytorch_model.bin" for k in self._torch_state.keys()}
            return

        raise FileNotFoundError(f"No safetensors or pytorch_model.bin found in {model_dir}")

    def get_tensor_np(self, tname):
        """Read tensor as numpy array."""
        if self._torch_state is not None:
            return self._torch_state[tname].cpu().numpy()
        sp = self.weight_map.get(tname)
        if not sp:
            raise KeyError(f"Tensor {tname} not found in weight map")
        shard = os.path.join(self.model_dir, sp)
        with safe_open(shard, framework="np") as sf:
            arr = sf.get_tensor(tname)
        return np.array(arr)

    def get_tensor_pt(self, tname):
        """Read tensor via PyTorch."""
        if self._torch_state is not None:
            return self._torch_state[tname]
        sp = self.weight_map.get(tname)
        if not sp:
            raise KeyError(f"Tensor {tname} not found in weight map")
        shard = os.path.join(self.model_dir, sp)
        if shard not in self._shards_pt:
            self._shards_pt[shard] = safe_open(shard, framework="pt")
        return self._shards_pt[shard].get_tensor(tname)

    def get_bf16_manual(self, tname):
        """Manually parse BF16 tensor from raw data.

        Handles safetensors (BitNet) and torch .bin (MiniCPM) formats.
        """
        if self._torch_state is not None:
            t = self._torch_state[tname]
            arr = t.cpu().to(torch.float32).numpy()
            arr = _clean_embed_rows(arr, tname)
            arr = np.clip(arr, -65504.0, 65504.0)
            return arr
        sp = self.weight_map.get(tname)
        if not sp:
            raise KeyError(f"Tensor {tname} not found")
        shard = os.path.join(self.model_dir, sp)

        # Parse safetensors header manually
        with open(shard, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(header_len))
            info = hdr[tname]
            start, end = info["data_offsets"]
            dtype = info["dtype"]
            shape = info["shape"]
            f.seek(0)
            f.seek(8 + header_len + start)
            raw = f.read(end - start)

        if dtype == "BF16":
            arr = _read_bf16_bytes(raw)
            fp32 = _clean_embed_rows(arr.reshape(shape), tname)
            fp32 = np.clip(fp32, -65504.0, 65504.0)
            return fp32
        elif dtype == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(shape)
        elif dtype == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shape)
        elif dtype in ("U8", "I8"):
            return np.frombuffer(raw, dtype=np.uint8 if dtype == "U8" else np.int8).reshape(shape)
        else:
            raise TypeError(f"Unsupported dtype {dtype} for {tname}")

    def has_tensor(self, tname):
        return tname in self.weight_map

    def close(self):
        self._shards_np.clear()
        self._shards_pt.clear()
        self._torch_state = None


# ═══════════════════════════════════════════════════════════════════════
# Quantization functions
# ═══════════════════════════════════════════════════════════════════════

def pack_tq1_base3(ternary_int8, ncols):
    """Pack int8 {-1,0,+1} tensor to TQ1 Base-3 (5 trits/byte).

    Args:
        ternary_int8: np.int8 array of shape [OUT, IN], values in {-1,0,+1}
        ncols: number of input columns (may be less than shape[1])

    Returns: (packed_bytes, packed_per_row)
    """
    nrows = ternary_int8.shape[0]
    packed_per_row = (ncols + 4) // 5
    full_len = packed_per_row * 5
    out = np.empty(nrows * packed_per_row, dtype=np.uint8)

    for r in range(nrows):
        row = ternary_int8[r, :ncols].astype(np.int32) + 1  # {-1,0,+1} → {0,1,2}
        if ncols < full_len:
            row = np.pad(row, (0, full_len - ncols), constant_values=1)
        t5 = row[:full_len].reshape(packed_per_row, 5)
        packed = (t5 * TQ1_MUL).sum(axis=1).astype(np.uint8)
        out[r * packed_per_row:(r + 1) * packed_per_row] = np.minimum(packed, 242)

    return out.tobytes(), packed_per_row


def _decode_i2s_to_fp32(tensor_uint8, weight_scale, sub_row_bit_invert=False):
    """I2_S uint8 → fp32 with per-row weight_scale.

    Decodes 4 ternary values from each uint8 byte, multiplies each row
    by its per-row scale from weight_scale tensor.

    Args:
        tensor_uint8: uint8 array of shape [u8_rows, in_cols]
        weight_scale: per-row scale, shape [nrows] or [nrows, 1] or scalar
        sub_row_bit_invert: True for Microsoft bit order (sub 0 in high bits)

    Returns:
        fp32 array of shape [nrows, in_cols]
    """
    u8_rows, in_cols = tensor_uint8.shape
    nrows = u8_rows * 4
    arr = tensor_uint8.astype(np.uint32)
    ws = np.asarray(weight_scale, dtype=np.float32).reshape(-1)
    out = np.empty((nrows, in_cols), dtype=np.float32)

    for ur in range(u8_rows):
        row = arr[ur]
        if sub_row_bit_invert:
            t0 = np.minimum((row >> 6) & 3, 2).astype(np.int8) - 1
            t1 = np.minimum((row >> 4) & 3, 2).astype(np.int8) - 1
            t2 = np.minimum((row >> 2) & 3, 2).astype(np.int8) - 1
            t3 = np.minimum(row & 3, 2).astype(np.int8) - 1
        else:
            t0 = np.minimum(row & 3, 2).astype(np.int8) - 1
            t1 = np.minimum((row >> 2) & 3, 2).astype(np.int8) - 1
            t2 = np.minimum((row >> 4) & 3, 2).astype(np.int8) - 1
            t3 = np.minimum((row >> 6) & 3, 2).astype(np.int8) - 1

        out[ur * 4 + 0] = t0
        out[ur * 4 + 1] = t1
        out[ur * 4 + 2] = t2
        out[ur * 4 + 3] = t3

    if ws.size > 1:
        out *= ws[:, np.newaxis]
    else:
        out *= ws[0]

    return out


def pre_shuffle_rows(tensor):
    """Pre-shuffle rows to cancel C++ matmul SIMD reorder.

    C++: output[(r%4)*rows_packed + r//4] = W[r] · act
    Store: W_shuffled[target] = W_natural[r]
      where target = (r % rows_packed) * 4 + r // rows_packed
    """
    out_dim = tensor.shape[0]
    assert out_dim % 4 == 0, f"out_dim {out_dim} not divisible by 4"
    rows_packed = out_dim // 4
    out = np.empty_like(tensor)
    for r in range(out_dim):
        target = (r % rows_packed) * 4 + r // rows_packed
        out[target] = tensor[r]
    return out


def quantize_tq1_repack(tensor_uint8, scale_val=1.0, sub_row_bit_invert=False, pre_shuffle=False):
    """I2_S uint8 [OUT/4, IN] → TQ1 Base-3 with fp16 scale.

    Each uint8 byte packs 4 consecutive output rows at one input column.
    De-interleave → clamp → repack as TQ1.

    TII Falcon3 format (default): sub-row 0 in low bits [0-1].
    Microsoft format (invert=True): sub-row 0 in high bits [6-7].

    If pre_shuffle=True, output rows are permuted so that the C++ matmul's
    dst[sub*rows_packed+ur] = W[ur*4+sub] · input produces correct
    dst[r] = W_natural[r] · input.

    Returns (data_bytes, packed_per_row, row_dim).
    """
    arr = tensor_uint8.astype(np.uint32)
    u8_rows, in_cols = arr.shape
    packed_per_row = (in_cols + 4) // 5
    nrows = u8_rows * 4
    out = np.empty(nrows * packed_per_row, dtype=np.uint8)

    for ur in range(u8_rows):
        row = arr[ur]
        if sub_row_bit_invert:
            # Microsoft I2_S: sub-row 0 in high bits
            t0 = np.minimum((row >> 6) & 3, 2)
            t1 = np.minimum((row >> 4) & 3, 2)
            t2 = np.minimum((row >> 2) & 3, 2)
            t3 = np.minimum(row & 3, 2)
        else:
            # TII Falcon3: sub-row 0 in low bits
            t0 = np.minimum(row & 3, 2)
            t1 = np.minimum((row >> 2) & 3, 2)
            t2 = np.minimum((row >> 4) & 3, 2)
            t3 = np.minimum((row >> 6) & 3, 2)
        for sub, src in enumerate([t0, t1, t2, t3]):
            wt = src.astype(np.int32)
            full_len = packed_per_row * 5
            if in_cols < full_len:
                wt = np.pad(wt, (0, full_len - in_cols), constant_values=1)
            else:
                wt = wt[:full_len]
            t5 = wt.reshape(packed_per_row, 5)
            packed = (t5 * TQ1_MUL).sum(axis=1).astype(np.uint8)
            start = (ur * 4 + sub) * packed_per_row
            out[start:start + packed_per_row] = np.minimum(packed, 242)

    if pre_shuffle:
        # Reorder logical rows to cancel C++ matmul SIMD output reorder.
        # Logical row r → position (r % u8_rows) * 4 + r // u8_rows
        shuffled = np.empty_like(out)
        for r in range(nrows):
            target = (r % u8_rows) * 4 + r // u8_rows
            shuffled[target * packed_per_row:(target + 1) * packed_per_row] = \
                out[r * packed_per_row:(r + 1) * packed_per_row]
        out = shuffled

    data = struct.pack("<e", float(scale_val)) + out.tobytes()
    return data, packed_per_row, nrows


def quantize_tq1_block_scaled(weights_fp16, block_size=128):
    """Per-block ternary quantization → ttype=5.

    Per-row block scales (max(abs) per block), stored as fp16.
    Returns (data_bytes, packed_per_row, n_blocks, row_dim).
    """
    w = weights_fp16.astype(np.float32)
    nrows, ncols = w.shape
    n_blocks = (ncols + block_size - 1) // block_size

    pad_len = n_blocks * block_size - ncols
    w_pad = np.pad(w, ((0, 0), (0, pad_len)), constant_values=0) if pad_len else w
    w_3d = w_pad.reshape(nrows, n_blocks, block_size)
    block_scales32 = np.max(np.abs(w_3d), axis=2)
    block_scales32 = np.where(block_scales32 < 1e-10, 1.0, block_scales32)

    scales_expanded = np.repeat(block_scales32[:, :, np.newaxis], block_size, axis=2)
    ternary_3d = np.clip(np.round(w_3d / scales_expanded).astype(np.int32), -1, 1)
    ternary_flat = ternary_3d.reshape(nrows, n_blocks * block_size)[:, :ncols].astype(np.int8)

    packed_bytes, packed_per_row = pack_tq1_base3(ternary_flat, ncols)

    block_scales = block_scales32.astype(np.float16)
    header = struct.pack("<BH", block_size, n_blocks) + block_scales.tobytes()
    return header + packed_bytes, packed_per_row, n_blocks, block_size, nrows


def quantize_tq1_absmean(weights_fp16, gamma=None, block_size=128):
    """Per-tensor absmean ternary quantization → ttype=5.

    All blocks share the same gamma = mean(abs(W)).
    Returns (data_bytes, packed_per_row, n_blocks, row_dim).
    """
    w = weights_fp16.astype(np.float32)
    nrows, ncols = w.shape
    n_blocks = (ncols + block_size - 1) // block_size

    if gamma is None:
        gamma_val = float(np.abs(w).mean())
    else:
        gamma_val = float(gamma)
    if gamma_val < 1e-10:
        gamma_val = 1.0

    pad_len = n_blocks * block_size - ncols
    w_pad = np.pad(w, ((0, 0), (0, pad_len)), constant_values=0) if pad_len else w
    w_3d = w_pad.reshape(nrows, n_blocks, block_size)
    ternary_3d = np.clip(np.round(w_3d / gamma_val).astype(np.int32), -1, 1)
    ternary_flat = ternary_3d.reshape(nrows, n_blocks * block_size)[:, :ncols].astype(np.int8)

    packed_bytes, packed_per_row = pack_tq1_base3(ternary_flat, ncols)

    block_scales = np.full((nrows, n_blocks), gamma_val, dtype=np.float16)
    header = struct.pack("<BH", block_size, n_blocks) + block_scales.tobytes()
    return header + packed_bytes, packed_per_row, n_blocks, block_size, nrows


def quantize_tq1_from_u8_packed(tensor_uint8, gamma=1.0, out_dim=None, block_size=128):
    """BitNet --packed path: uint8 I2_S [ceil(OUT/4), IN] → TQ1 Base-3.

    Returns (data_bytes, packed_per_row, n_blocks, row_dim).
    """
    B, C = tensor_uint8.shape
    out = out_dim or (B * 4)

    # Unpack I2_S: row r at position (r % B) with shift (r // B) * 2
    ternary = np.zeros((out, C), dtype=np.int8)
    packed_u32 = tensor_uint8.astype(np.uint32)
    for k in range(4):
        shift = k * 2
        v = (packed_u32 >> shift) & 3
        ternary[k * B:(k + 1) * B, :] = v.astype(np.int8) - 1

    # Pre-shuffle rows to cancel SIMD reorder
    ternary = pre_shuffle_rows(ternary)

    nrows, ncols = ternary.shape
    n_blocks = (ncols + block_size - 1) // block_size
    packed_bytes, packed_per_row = pack_tq1_base3(ternary, ncols)

    block_scales = np.full((nrows, n_blocks), float(gamma), dtype=np.float16)
    header = struct.pack("<BH", block_size, n_blocks) + block_scales.tobytes()
    return header + packed_bytes, packed_per_row, n_blocks, block_size, out


# ═══════════════════════════════════════════════════════════════════════
# Tokenizer builder (v6 binary)
# ═══════════════════════════════════════════════════════════════════════

def build_tokenizer_binary(model_dir):
    """Build v6 binary tokenizer block from tokenizer.json.

    Returns bytes of the binary block, or b'' if not found.
    """
    tok_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        return b""

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)

    # Get raw vocab from JSON to exclude added tokens (IDs >= 128000).
    # Tokenizer.get_vocab() includes added tokens, which would inflate V
    # and cause merge_lookup to mis-handle out-of-vocab IDs.
    with open(tok_path, "r", encoding="utf-8") as jf:
        tok_json_data = json.load(jf)
    raw_vocab = tok_json_data.get("model", {}).get("vocab", {})
    # Detect SentencePiece-style tokenizer (pre_tokenizer is null)
    pre_tok = tok_json_data.get("pre_tokenizer")
    if pre_tok is None or pre_tok.get("type") is None:
        print("  SentencePiece tokenizer detected — v6 binary block not supported, using v5 JSON fallback")
        return b""
    # raw_vocab contains only BPE tokens (model.vocab), NOT added tokens.
    # Using it directly ensures V matches the true BPE vocab size — no filter needed.
    # This correctly handles Falcon3 (V=131072), Llama3 (V=128000), and all other models.
    vocab = raw_vocab if raw_vocab else {k: v for k, v in tok.get_vocab().items() if v < 128000}
    V = len(vocab)
    sorted_items = sorted(vocab.items(), key=lambda kv: kv[1])

    offsets = np.empty(V, dtype=np.uint32)
    lengths = np.empty(V, dtype=np.uint16)
    pool_parts = []
    offset_acc = 0
    for i, (token_str, tid) in enumerate(sorted_items):
        token_bytes = token_str.encode("utf-8")
        offsets[i] = offset_acc
        lengths[i] = len(token_bytes)
        pool_parts.append(token_bytes)
        offset_acc += len(token_bytes)
    pool = b"".join(pool_parts)
    max_token_length = int(max(lengths))

    merge_left = np.full(V, 0xFFFFFFFF, dtype=np.uint32)
    merge_right = np.full(V, 0xFFFFFFFF, dtype=np.uint32)
    merge_rank = np.zeros(V, dtype=np.uint32)
    merges_list = tok_json_data.get("model", {}).get("merges", [])
    for i, merge_pair in enumerate(merges_list):
        if isinstance(merge_pair, list) and len(merge_pair) == 2:
            left_str, right_str = merge_pair
        elif isinstance(merge_pair, str):
            parts = merge_pair.split()
            if len(parts) != 2:
                continue
            left_str, right_str = parts
        else:
            continue
        left_id = vocab.get(left_str)
        right_id = vocab.get(right_str)
        if left_id is not None and right_id is not None:
            merged_str = left_str + right_str
            merged_id = vocab.get(merged_str)
            if merged_id is not None and merged_id < V:
                merge_left[merged_id] = left_id
                merge_right[merged_id] = right_id
                merge_rank[merged_id] = i + 1

    byte_encoder = np.full(256, 0xFFFF, dtype=np.uint16)
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    byte_to_token = {}
    n = 0
    for b in range(256):
        if b in printable:
            byte_to_token[b] = chr(b)
        else:
            byte_to_token[b] = chr(256 + n)
            n += 1
    for b in range(256):
        tid = vocab.get(byte_to_token[b])
        if tid is not None:
            byte_encoder[b] = tid
    missing = [b for b in range(256) if byte_encoder[b] == 0xFFFF]
    if missing:
        for b in missing:
            tid = vocab.get(chr(256 + b))
            if tid is not None:
                byte_encoder[b] = tid

    byte_decoder = np.full(256, 0xFFFF, dtype=np.uint16)
    for b in range(256):
        if byte_encoder[b] != 0xFFFF:
            byte_decoder[b] = b

    special_ids = {
        "eos": 0xFFFFFFFF, "bos": 0xFFFFFFFF, "pad": 0xFFFFFFFF,
        "unk": 0xFFFFFFFF, "mask": 0xFFFFFFFF, "sep": 0xFFFFFFFF, "cls": 0xFFFFFFFF,
    }
    cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as cf:
            tcfg = json.load(cf)
        for key in ["eos_token", "bos_token", "pad_token", "unk_token", "mask_token", "sep_token", "cls_token"]:
            val = tcfg.get(key)
            idx_key = key.split("_")[0]
            if val:
                if isinstance(val, dict) and "id" in val:
                    special_ids[idx_key] = val["id"]
                elif isinstance(val, str):
                    tid = tok.token_to_id(val)
                    if tid is not None:
                        special_ids[idx_key] = tid
    # Fallback: pattern matching in BPE vocab + full tokenizer
    for pattern, idx_key in [("<|endoftext|>", "eos"), ("<|im_end|>", "eos"),
                              ("<|eot_id|>", "eos"),
                              ("<|pad|>", "pad"), ("<unk>", "unk")]:
        if special_ids[idx_key] == 0xFFFFFFFF:
            tid = tok.token_to_id(pattern)
            if tid is not None:
                special_ids[idx_key] = tid
    if special_ids["eos"] == 0xFFFFFFFF:
        special_ids["eos"] = 0
    if special_ids["pad"] == 0xFFFFFFFF:
        special_ids["pad"] = 0
    if special_ids["unk"] == 0xFFFFFFFF:
        special_ids["unk"] = 0
    special_arr = np.array([
        special_ids["eos"], special_ids["bos"], special_ids["pad"],
        special_ids["unk"], special_ids["mask"], special_ids["sep"], special_ids["cls"],
    ], dtype=np.uint32)

    # Extract out-of-vocab added tokens (IDs >= V, not in BPE vocab).
    # These cannot be produced by BPE merges and must be handled as atomic units
    # by the pre-encode step (e.g. Llama3 special tokens 128000-128255).
    # Include ALL out-of-vocab tokens — even EOS/BOS — since the C++ preencode
    # only scans added_specs, not the special array.
    added_tokens_sorted = []
    added_tok_json = tok_json_data.get("added_tokens", [])
    # Only include tokens whose ID >= V (out of BPE vocab range)
    out_of_vocab = [a for a in added_tok_json if a["id"] >= V]
    # Sort by length descending for longest-match-first scanning in C++
    out_of_vocab.sort(key=lambda a: -len(a["content"]))
    for a in out_of_vocab:
        tbytes = a["content"].encode("utf-8")
        added_tokens_sorted.append((tbytes, a["id"]))
    num_added = len(added_tokens_sorted)

    off = 128
    off_offsets = off
    off += V * 4
    off_lengths = off
    off += V * 2
    off_pool = off
    off += len(pool)
    pool_pad = (4 - len(pool) % 4) % 4
    off += pool_pad
    off_merge_left = off
    off += V * 4
    off_merge_right = off
    off += V * 4
    off_merge_rank = off
    off += V * 4
    off_byte_enc = off
    off += 512
    off_byte_dec = off
    off += 512
    off_special = off
    off += 28
    off_added_tokens = off if num_added > 0 else 0
    added_data = b""
    if num_added > 0:
        added_data += struct.pack("<I", num_added)
        for tbytes, tid in added_tokens_sorted:
            slen = len(tbytes)
            added_data += struct.pack("<I", slen)
            added_data += tbytes
            pad_len = (4 - slen % 4) % 4
            if pad_len:
                added_data += b"\x00" * pad_len
            added_data += struct.pack("<I", tid)
        off += len(added_data)
    total_size = off

    buf = bytearray(total_size)
    struct.pack_into("<I", buf, 0, 0x544F4B42)
    struct.pack_into("<I", buf, 4, 1)
    struct.pack_into("<I", buf, 8, V)
    struct.pack_into("<I", buf, 12, max_token_length)
    struct.pack_into("<I", buf, 16, num_added)  # h[4] = num_added_tokens
    struct.pack_into("<I", buf, 20, 0)
    offs_list = [off_offsets, off_lengths, off_pool, len(pool),
                 off_merge_left, off_merge_right, off_merge_rank,
                 off_byte_enc, off_byte_dec, off_special,
                 off_added_tokens]  # offs[10] at byte 104
    for i, val in enumerate(offs_list):
        struct.pack_into("<Q", buf, 24 + i * 8, val)
    buf[off_offsets:off_offsets + V * 4] = offsets.tobytes()
    buf[off_lengths:off_lengths + V * 2] = lengths.tobytes()
    buf[off_pool:off_pool + len(pool)] = pool
    buf[off_merge_left:off_merge_left + V * 4] = merge_left.tobytes()
    buf[off_merge_right:off_merge_right + V * 4] = merge_right.tobytes()
    buf[off_merge_rank:off_merge_rank + V * 4] = merge_rank.tobytes()
    buf[off_byte_enc:off_byte_enc + 512] = byte_encoder.tobytes()
    buf[off_byte_dec:off_byte_dec + 512] = byte_decoder.tobytes()
    buf[off_special:off_special + 28] = special_arr.tobytes()
    if num_added > 0 and off_added_tokens:
        buf[off_added_tokens:off_added_tokens + len(added_data)] = added_data

    return bytes(buf)


# ═══════════════════════════════════════════════════════════════════════
# Main packing function
# ═══════════════════════════════════════════════════════════════════════

def pack_to_atlas(model_dir, output_path, ttype=5, block_size=128, use_v6=True,
                  packed_path=False, thinking=False):
    """Pack a HuggingFace model directory to ATLAS TQ1.0 format.

    Args:
        model_dir: Path to HF model directory (config.json + safetensors)
        output_path: Output .atlas file path
        ttype: Ternary tensor format (5=TQ1 Base-3, 7=2-bit, default: 5)
        block_size: Block size for block-scaled quantization (default: 128)
        use_v6: Embed v6 binary tokenizer (default: True)
        packed_path: Use U8 pre-quantized path (BitNet --packed, default: False)
        thinking: Set enable_thinking flag (default: False)
    """
    # ── Read config ────────────────────────────────────────────────
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    hidden = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = cfg.get("num_key_value_heads", n_heads)
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    head_dim = cfg.get("head_dim", hidden // n_heads)
    rope_theta = cfg.get("rope_theta", 10000.0)

    print(f"[ATLAS] {model_dir}")
    print(f"  Layers:{n_layers} Hidden:{hidden} Heads:{n_heads}/{n_kv_heads}")
    print(f"  Intermediate:{inter} Vocab:{vocab} Head_dim:{head_dim}")

    # ── Open reader ────────────────────────────────────────────────
    reader = SafetensorsReader(model_dir)

    # Auto-detect which tensors are available (for arch detection)
    available = list(reader.weight_map.keys())
    arch = detect_arch(cfg, available)
    mt = cfg.get("model_type", "unknown")
    print(f"  Stride: {arch['stride']} tensors/layer")

    is_uint8_input = any(reader.weight_map.get(n, "") and
                         os.path.exists(os.path.join(model_dir, reader.weight_map[n]))
                         for n in available if n.endswith(".weight") and "embed" not in n)

    # ── Preload weight scales ──────────────────────────────────────
    scales = {}
    scales_raw = {}
    if arch["has_weight_scale"]:
        for tname in available:
            if tname.endswith("weight_scale"):
                try:
                    tensor = reader.get_tensor_np(tname)
                except Exception:
                    try:
                        tensor = reader.get_bf16_manual(tname)
                    except Exception:
                        continue
                t_flat = tensor.reshape(-1)
                scales[tname] = float(t_flat[0])
                if t_flat.size > 1:
                    scales_raw[tname] = tensor
        print(f"  Scales loaded: {len(scales)} ({len(scales_raw)} per-channel)")

    # ── Build ordered tensor list ──────────────────────────────────
    tensor_names = []

    # Layer tensors
    for L in range(n_layers):
        for pattern in arch["layer_tensors"]:
            tname = pattern.format(L)
            if reader.has_tensor(tname):
                tensor_names.append(tname)

    # Conditional layer tensors (optional, detected by presence)
    for L in range(n_layers):
        for pattern in arch.get("conditional_layer", []):
            if "{}" in pattern:
                tname = pattern.format(L)
                if reader.has_tensor(tname):
                    tensor_names.append(tname)

    # Global tensors
    for pattern in arch["global_tensors"]:
        if reader.has_tensor(pattern):
            tensor_names.append(pattern)

    # Conditional globals (lm_head tied emb etc.)
    for pattern in arch.get("conditional_layer", []):
        if "{}" not in pattern:
            # Bare name like "lm_head.weight" — treat as conditional global
            if pattern not in tensor_names and reader.has_tensor(pattern):
                tensor_names.append(pattern)

    # SubLN tensors (BitNet/TriLM) — detect by presence
    for L in range(n_layers):
        for extra in [
            f"model.layers.{L}.self_attn.attn_sub_norm.weight",
            f"model.layers.{L}.mlp.ffn_sub_norm.weight",
        ]:
            if extra not in tensor_names and reader.has_tensor(extra):
                tensor_names.append(extra)

    print(f"  Tensors: {len(tensor_names)}")

    # ── Load tokenizer ─────────────────────────────────────────────
    tokenizer_block = b""
    tok_path = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        with open(tok_path, "rb") as tf:
            tok_data = tf.read()
        tokenizer_block += struct.pack("<I", len(tok_data)) + tok_data
        cfg_path = os.path.join(model_dir, "tokenizer_config.json")
        cfg_data = b""
        if os.path.exists(cfg_path):
            with open(cfg_path, "rb") as cf:
                cfg_data = cf.read()
        tokenizer_block += struct.pack("<I", len(cfg_data)) + cfg_data
    else:
        # Fallback: try tokenizer.model (BitNet legacy)
        tok_path = os.path.join(model_dir, "tokenizer.model")
        if os.path.exists(tok_path):
            with open(tok_path, "rb") as tf:
                tok_data = tf.read()
            tokenizer_block += struct.pack("<I", len(tok_data)) + tok_data

    # ── Write output file ─────────────────────────────────────────
    with open(output_path, "wb") as out:
        header = bytearray(64)
        header[0:5] = b"ATLAS"
        struct.pack_into("<H", header, 5, 5)  # v5 (may upgrade to v7 later)
        struct.pack_into("<H", header, 7, n_layers)
        struct.pack_into("<H", header, 9, hidden)
        struct.pack_into("<H", header, 11, inter)
        struct.pack_into("<B", header, 13, n_heads)
        struct.pack_into("<B", header, 14, n_kv_heads)
        struct.pack_into("<H", header, 15, head_dim)
        struct.pack_into("<I", header, 17, vocab)
        struct.pack_into("<d", header, 21, rope_theta)
        struct.pack_into("<I", header, 56, 0)  # name_block_size placeholder
        struct.pack_into("<I", header, 60, len(tensor_names))

        # EOS/PAD from config
        eos_id = cfg.get("eos_token_id")
        pad_id = cfg.get("pad_token_id")
        if isinstance(eos_id, list):
            eos_id = eos_id[0] if eos_id else 0
        if isinstance(pad_id, list):
            pad_id = pad_id[0] if pad_id else 0
        if eos_id is not None:
            struct.pack_into("<I", header, 45, eos_id)
        if pad_id is not None:
            struct.pack_into("<I", header, 49, pad_id)
        # Prefer tokenizer_config.json eos over config.json (handles Llama-3/BitNet
        # where eos_token_id in config.json is wrong/points to BOS instead of EOT)
        tcfg_path = os.path.join(model_dir, "tokenizer_config.json")
        if os.path.exists(tcfg_path):
            with open(tcfg_path) as tcf:
                tcfg = json.load(tcf)
            ec = tcfg.get("eos_token")
            if isinstance(ec, dict) and "id" in ec:
                struct.pack_into("<I", header, 45, ec["id"])
            elif isinstance(ec, str):
                tok_json_path = os.path.join(model_dir, "tokenizer.json")
                if os.path.exists(tok_json_path):
                    from tokenizers import Tokenizer as Tk
                    tk = Tk.from_file(tok_json_path)
                    tvid = tk.token_to_id(ec)
                    if tvid is not None and tvid != 0:
                        struct.pack_into("<I", header, 45, tvid)

        # Byte 53: model_flags
        model_flags = arch["flags_fn"](cfg)
        if thinking:
            model_flags |= 1 << 2
        struct.pack_into("<B", header, 53, model_flags)

        # Bytes 54-55: format_version (v2.10.0+)
        struct.pack_into("<H", header, 54, 2)

        # v8 meta block (for MiniCPM scale_emb/scale_depth)
        meta_bytes = b""
        is_minicpm = cfg.get("scale_emb") is not None or cfg.get("scale_depth") is not None
        if is_minicpm:
            meta = {
                "arch": "minicpm",
                "scale_emb": cfg.get("scale_emb", 1.0),
                "scale_depth": cfg.get("scale_depth", 1.0),
                "rope_interleaved": False,
                "use_f32_bypass": False,
                "rope_theta": rope_theta,
            }
            meta_json = json.dumps(meta).encode("utf-8")
            meta_bytes = struct.pack("<I", 4 + len(meta_json)) + meta_json
            struct.pack_into("<H", header, 5, 8)  # upgrade to v8

        # Name block
        name_bytes = b"".join(n.encode() + b"\0" for n in tensor_names)
        name_block = struct.pack("<I", 4 + len(name_bytes)) + name_bytes
        struct.pack_into("<I", header, 56, len(name_block))

        out.write(header)
        out.write(meta_bytes)

        dir_offset = 64 + len(meta_bytes)
        data_start = dir_offset + len(tensor_names) * 12 + len(name_block)
        directory = bytearray(len(tensor_names) * 12)
        current_offset = data_start

        out.seek(dir_offset + len(tensor_names) * 12)
        out.write(name_block)
        out.seek(data_start)

        for idx, name in enumerate(tensor_names):
            is_weight = "proj" in name and "weight" in name

            # ── Determine quantization path ──────────────────────────
            if is_weight and arch["input_format"] == "uint8_packed":
                # Falcon3 path: uint8 pre-packed → TQ1
                tensor_pt = reader.get_tensor_pt(name)
                sname = name.replace(".weight", ".weight_scale")
                scale_raw = scales_raw.get(sname)
                if scale_raw is not None:
                    # Per-channel weight_scale: decode → multiply → block-scaled ttype=5
                    fp32_weights = _decode_i2s_to_fp32(
                        tensor_pt.numpy(), scale_raw,
                        sub_row_bit_invert=arch.get("sub_row_bit_invert", False))
                    if arch["requires_pre_shuffle"]:
                        fp32_weights = pre_shuffle_rows(fp32_weights)
                    result = quantize_tq1_block_scaled(fp32_weights, block_size)
                    data_bytes, packed_per_row, n_blocks, _, row_dim = result
                    tens_ttype = ttype if ttype == 7 else 5
                else:
                    # Scalar weight_scale: existing repack path (ttype=0)
                    scale_val = scales.get(sname, 1.0)
                    data_bytes, packed_per_row, row_dim = quantize_tq1_repack(
                        tensor_pt.numpy(), scale_val,
                        sub_row_bit_invert=arch.get("sub_row_bit_invert", False),
                        pre_shuffle=arch["requires_pre_shuffle"])
                    tens_ttype = 0
                    n_blocks = 0

            elif is_weight and packed_path:
                # BitNet --packed path: uint8 I2_S → TQ1
                tensor_np = reader.get_tensor_np(name)
                sname = name.replace(".weight", ".weight_scale")
                gamma = scales.get(sname, 1.0)
                out_dim = tensor_np.shape[0] * 4
                result = quantize_tq1_from_u8_packed(
                    tensor_np, gamma, out_dim, block_size)
                data_bytes, packed_per_row, n_blocks, _, row_dim = result
                tens_ttype = ttype if ttype == 7 else 5

            elif is_weight:
                # BF16/F32 path: read → ternarize
                try:
                    # Try manual BF16 first (handles BitNet BF16 correctly)
                    tensor_fp32 = reader.get_bf16_manual(name)
                except (TypeError, KeyError):
                    # Fallback: safe_open with torch
                    tensor_pt = reader.get_tensor_pt(name)
                    tensor_fp32 = tensor_pt.cpu().to(torch.float32).numpy()

                # Pre-shuffle rows if needed
                if arch["requires_pre_shuffle"]:
                    tensor_fp32 = pre_shuffle_rows(tensor_fp32)

                quant = arch["quant_weight"]
                if quant == "tq1_block_scaled":
                    result = quantize_tq1_block_scaled(tensor_fp32, block_size)
                    data_bytes, packed_per_row, n_blocks, _, row_dim = result
                elif quant == "tq1_absmean":
                    sname = name.replace(".weight", ".weight_scale")
                    gamma = scales.get(sname, None)
                    result = quantize_tq1_absmean(tensor_fp32, gamma, block_size)
                    data_bytes, packed_per_row, n_blocks, _, row_dim = result
                else:
                    # Fallback: block-scaled
                    result = quantize_tq1_block_scaled(tensor_fp32, block_size)
                    data_bytes, packed_per_row, n_blocks, _, row_dim = result
                tens_ttype = ttype if ttype == 7 else 5

            else:
                # Non-weight tensor (norm, embed, lm_head) → fp16
                try:
                    tensor = reader.get_tensor_np(name)
                except Exception:
                    tensor = reader.get_bf16_manual(name)

                if "embed" in name:
                    tensor_fp32 = _clean_embed_rows(
                        tensor.astype(np.float32), name)
                    np.clip(tensor_fp32, -65504.0, 65504.0, out=tensor_fp32)
                    tensor = tensor_fp32.astype(np.float16)
                elif tensor.dtype in (np.float32, np.float64):
                    tensor = tensor.astype(np.float16)
                elif tensor.dtype == np.uint16:
                    # BF16 via numpy
                    tensor = tensor.astype(np.float16)

                if tensor.dtype != np.float16:
                    tensor = tensor.astype(np.float16)

                data_bytes = tensor.tobytes()
                packed_per_row = 0
                n_blocks = 0
                row_dim = tensor.shape[0]
                tens_ttype = 1 if ("norm" in name or "embed" in name) else 2

            # ── 32-byte alignment ────────────────────────────────────
            if current_offset % 32 != 0:
                pad = 32 - (current_offset % 32)
                current_offset += pad
                out.write(b"\x00" * pad)

            # ── Directory entry ──────────────────────────────────────
            directory[idx * 12] = tens_ttype
            struct.pack_into("<I", directory, idx * 12 + 1, current_offset)
            struct.pack_into("<I", directory, idx * 12 + 5, row_dim)
            if n_blocks:
                ppr_entry = (packed_per_row & 0x1FFFFF) | (n_blocks << 21)
            else:
                ppr_entry = packed_per_row & 0xFFFFFF
            directory[idx * 12 + 9] = ppr_entry & 0xFF
            directory[idx * 12 + 10] = (ppr_entry >> 8) & 0xFF
            directory[idx * 12 + 11] = (ppr_entry >> 16) & 0xFF

            out.write(data_bytes)
            current_offset += len(data_bytes)

            if idx % 8 == 0:
                print(f"  [{idx}/{len(tensor_names)}] {name[:55]:55s} {len(data_bytes)/1024:7.1f}KB ttype={tens_ttype}")

        # ── Tokenizer block (v5 JSON) ───────────────────────────────
        if tokenizer_block:
            tokenizer_offset = current_offset
            if tokenizer_offset % 32 != 0:
                pad = 32 - (tokenizer_offset % 32)
                tokenizer_offset += pad
                out.write(b"\x00" * pad)
            out.write(tokenizer_block)
            struct.pack_into("<I", header, 29, len(tokenizer_block))
            struct.pack_into("<I", header, 33, tokenizer_offset)
            current_offset = tokenizer_offset + len(tokenizer_block)

        # ── v6 binary tokenizer ─────────────────────────────────────
        if use_v6:
            binary_tok_block = build_tokenizer_binary(model_dir)
            if len(binary_tok_block) > 0:
                struct.pack_into("<H", header, 5, 7)
                binary_offset = current_offset
                if binary_offset % 32 != 0:
                    pad = 32 - (binary_offset % 32)
                    binary_offset += pad
                    out.write(b"\x00" * pad)
                out.write(binary_tok_block)
                current_offset = binary_offset + len(binary_tok_block)
                struct.pack_into("<I", header, 37, len(binary_tok_block))
                struct.pack_into("<I", header, 41, binary_offset)
            else:
                print("  WARNING: v6 binary tokenizer block empty, skipping")

        # ── Finalize header + directory ─────────────────────────────
        out.flush()
        out.seek(dir_offset)
        out.write(directory)
        out.seek(0)
        out.write(header)

    reader.close()
    total_gb = current_offset / 1024 ** 3
    print(f"[ATLAS] Done! {total_gb:.2f} GB | {len(tensor_names)} tensors")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ATLAS packer — pack supported HF models to TQ1.0 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pack_to_atlas.py falcon3-1B/ falcon3-1B-tq1.atlas
  python pack_to_atlas.py bonsai-4B/ bonsai-4B-tq1.atlas --block-size 128
  python pack_to_atlas.py bitnet-2B4T/ bitnet-2B4T-tq1.atlas --packed
  python pack_to_atlas.py trilm-1.5B/ trilm-1.5B-tq1.atlas
  python pack_to_atlas.py llama-3B/ llama-3B-tq1.atlas --ttype 7

Supported architectures (auto-detected):
  Falcon3, Qwen3/Bonsai, BitNet b1.58, TriLM, Llama, Mistral, Phi, Gemma
        """,
    )
    parser.add_argument("model_dir", help="HF model directory (config.json + safetensors)")
    parser.add_argument("output", help="Output .atlas file path")
    parser.add_argument("--ttype", type=int, default=5, choices=[5, 7],
                        help="Ternary tensor format: 5=TQ1 Base-3, 7=2-bit (default: 5)")
    parser.add_argument("--block-size", type=int, default=128,
                        help="Block size for block-scaled quantization (default: 128)")
    parser.add_argument("--no-v6", action="store_false", dest="use_v6",
                        help="Skip v6 binary tokenizer embedding")
    parser.add_argument("--packed", action="store_true",
                        help="BitNet packed U8 path (--packed flag)")
    parser.add_argument("--thinking", action="store_true",
                        help="Set enable_thinking flag in header")
    args = parser.parse_args()

    pack_to_atlas(
        model_dir=args.model_dir,
        output_path=args.output,
        ttype=args.ttype,
        block_size=args.block_size,
        use_v6=args.use_v6,
        packed_path=args.packed,
        thinking=args.thinking,
    )


if __name__ == "__main__":
    main()
