"""Parity tests: AVX2 int8/int4 kernels vs numpy reference.

Calls atlas_matmul_i8_f32 and atlas_matmul_i4_f32 from atlas.dll via ctypes.
All quantization logic replicates engine behavior (u8+128 activations,
int8 weights, row_sum correction, int4 nibble packing).
"""

import ctypes
import numpy as np
import pytest
from pathlib import Path

# ─── DLL setup ────────────────────────────────────────────────────────────

DLL = None
for candidate in ["atlas.dll", "atlas_d.dll", "atlas.so", "libatlas.so"]:
    p = Path(candidate)
    if p.exists():
        DLL = ctypes.cdll.LoadLibrary(str(p.resolve()))
        break

if DLL is None:
    raise RuntimeError("No atlas DLL/SO found in current directory")


def _bind(name, argtypes, restype):
    fn = getattr(DLL, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


matmul_i8_f32 = _bind("atlas_matmul_i8_f32", [
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
], None)

matmul_i4_f32 = _bind("atlas_matmul_i4_f32", [
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
], None)


# ─── Quantization helpers (replicate engine logic) ────────────────────────

def quantize_act_f32(act_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """act_f32: [B, cols] float32 -> act_u8: [B, cols] uint8 + max_abs: [B]"""
    B, cols = act_f32.shape
    max_abs = np.max(np.abs(act_f32), axis=1)
    max_abs = np.maximum(max_abs, 1e-9)
    act_u8 = np.zeros_like(act_f32, dtype=np.uint8)
    for b in range(B):
        scaled = act_f32[b] * (127.0 / max_abs[b]) + 128.0
        act_u8[b] = np.clip(np.round(scaled), 0, 255).astype(np.uint8)
    return act_u8, max_abs


def quantize_weights_i8(w_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """w_f32: [rows, cols] float32 -> w_i8: [rows, cols] int8 + scale + row_sums"""
    rows, cols = w_f32.shape
    max_abs = np.max(np.abs(w_f32), axis=1, keepdims=True)
    max_abs = np.maximum(max_abs, 1e-9)
    w_i8 = np.clip(np.round(w_f32 * (127.0 / max_abs)), -128, 127).astype(np.int8)
    row_sums = w_i8.astype(np.int32).sum(axis=1).astype(np.int32)
    return w_i8, max_abs.flatten(), row_sums


def quantize_weights_i4(w_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w_f32: [rows, cols] float32 -> packed: [rows, (cols+1)//2] uint8 + row_sums
    Int4 packing: clip to [-8,7], low nibble first, sign-extend via (nibble ^ 8) - 8.
    """
    rows, cols = w_f32.shape
    max_abs = np.max(np.abs(w_f32), axis=1, keepdims=True)
    max_abs = np.maximum(max_abs, 1e-9)
    w_i4_scaled = np.clip(np.round(w_f32 * (7.0 / max_abs)), -8, 7).astype(np.int8)
    packed_cols = (cols + 1) // 2
    packed = np.zeros((rows, packed_cols), dtype=np.uint8)
    for r in range(rows):
        for c in range(0, cols, 2):
            lo = int(w_i4_scaled[r, c]) & 0x0F
            hi = int(w_i4_scaled[r, c + 1]) & 0x0F if c + 1 < cols else 0
            packed[r, c // 2] = (hi << 4) | lo
    # row_sums for int4: sign-extend each nibble via (nibble ^ 8) - 8, then sum
    row_sums = np.zeros(rows, dtype=np.int32)
    for r in range(rows):
        s = 0
        for c in range(cols):
            nibble = int((packed[r, c // 2] >> (4 * (c % 2))) & 0x0F)
            val = (nibble ^ 8) - 8
            s += val
        row_sums[r] = s
    return packed, row_sums, max_abs.flatten()


def reference_u8xi8(act_u8: np.ndarray, w_i8: np.ndarray,
                    row_sums: np.ndarray) -> np.ndarray:
    """Naive uint8xint8 dot product with row_sum correction.
    Returns [B, rows] float32 (raw accumulator, no dequantization).
    """
    B, cols = act_u8.shape
    rows = w_i8.shape[0]
    out = np.zeros((B, rows), dtype=np.float32)
    for b in range(B):
        for r in range(rows):
            dot = 0
            for c in range(cols):
                dot += int(act_u8[b, c]) * int(w_i8[r, c])
            out[b, r] = float(dot - 128 * row_sums[r])
    return out


def reference_u8xi4_centered(act_u8: np.ndarray,
                              packed_w: np.ndarray) -> np.ndarray:
    """Naive centered reference for u8xi4 matmul.
    Decompresses int4 nibbles, computes sum((act[c]-128) * w_i4[c]).
    No row_sum correction needed — matches centered kernel.
    """
    B, cols = act_u8.shape
    rows = packed_w.shape[0]
    out = np.zeros((B, rows), dtype=np.float32)
    for b in range(B):
        for r in range(rows):
            dot = 0
            for c in range(cols):
                nibble = int((packed_w[r, c // 2] >> (4 * (c % 2))) & 0x0F)
                w_val = (nibble ^ 8) - 8
                dot += (int(act_u8[b, c]) - 128) * w_val
            out[b, r] = float(dot)
    return out


# ─── Test shapes ──────────────────────────────────────────────────────────

SHAPES = [
    (32, 64),
    (64, 128),
    (128, 256),
    (256, 1024),
    (64, 64),
    (1, 128),
    (128, 1),
]

B_VALUES = [1, 4]


# ─── Tests ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rows,cols", SHAPES)
@pytest.mark.parametrize("B", B_VALUES)
def test_matmul_i8_parity(rows: int, cols: int, B: int):
    """atlas_matmul_i8_f32 matches naive u8xi8 reference."""
    rng = np.random.default_rng(42)
    w_f32 = rng.standard_normal((rows, cols), dtype=np.float32) * 0.5
    act_f32 = rng.standard_normal((B, cols), dtype=np.float32)

    w_i8, _, row_sums = quantize_weights_i8(w_f32)
    act_u8, _ = quantize_act_f32(act_f32)

    expected = reference_u8xi8(act_u8, w_i8, row_sums)

    out = np.zeros((B, rows), dtype=np.float32)
    matmul_i8_f32(
        ctypes.c_int(rows), ctypes.c_int(cols),
        w_i8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        act_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        row_sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(B),
    )

    assert np.allclose(out, expected, atol=0.5, rtol=1e-5), (
        f"i8 mismatch for ({rows}, {cols}) B={B}\n"
        f"  max diff: {np.max(np.abs(out - expected)):.4f}\n"
        f"  first diff at: {np.unravel_index(np.argmax(np.abs(out - expected)), out.shape)}"
    )


@pytest.mark.parametrize("rows,cols", SHAPES)
@pytest.mark.parametrize("B", B_VALUES)
def test_matmul_i4_parity(rows: int, cols: int, B: int):
    """atlas_matmul_i4_f32 matches naive u8xi4 reference using packed int4 weights."""
    if cols < 2:
        pytest.skip("int4 needs at least 2 columns for packing")
    rng = np.random.default_rng(42)
    w_f32 = rng.standard_normal((rows, cols), dtype=np.float32) * 0.5
    act_f32 = rng.standard_normal((B, cols), dtype=np.float32)

    packed_w, row_sums, _ = quantize_weights_i4(w_f32)
    act_u8, _ = quantize_act_f32(act_f32)

    expected = reference_u8xi4_centered(act_u8, packed_w)

    out = np.zeros((B, rows), dtype=np.float32)
    matmul_i4_f32(
        ctypes.c_int(rows), ctypes.c_int(cols),
        packed_w.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        act_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        row_sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(B),
    )

    assert np.allclose(out, expected, atol=0.5, rtol=1e-5), (
        f"i4 mismatch for ({rows}, {cols}) B={B}\n"
        f"  max diff: {np.max(np.abs(out - expected)):.4f}\n"
        f"  first diff at: {np.unravel_index(np.argmax(np.abs(out - expected)), out.shape)}"
    )


def test_matmul_i8_vs_i4_consistency():
    """i8 and i4 kernel dequantized outputs agree (within quantization tolerance).

    i8 weights quantize f32 × 127/|w|_max, int4 weights quantize f32 × 7/|w|_max.
    Raw accumulators differ by ×127/7 ≈ 18×. Compare in f32 space.
    """
    rng = np.random.default_rng(99)
    rows, cols, B = 128, 256, 4
    w_f32 = rng.standard_normal((rows, cols), dtype=np.float32) * 0.5
    act_f32 = rng.standard_normal((B, cols), dtype=np.float32)

    w_i8, w_scale_i8, row_sums = quantize_weights_i8(w_f32)
    packed_w, row_sums_i4, w_scale_i4 = quantize_weights_i4(w_f32)
    act_u8, act_scale = quantize_act_f32(act_f32)

    out_i8 = np.zeros((B, rows), dtype=np.float32)
    matmul_i8_f32(
        ctypes.c_int(rows), ctypes.c_int(cols),
        w_i8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        act_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        row_sums.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_i8.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(B),
    )

    out_i4 = np.zeros((B, rows), dtype=np.float32)
    matmul_i4_f32(
        ctypes.c_int(rows), ctypes.c_int(cols),
        packed_w.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        act_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        row_sums_i4.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_i4.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(B),
    )

    # Dequantize: raw = (act_i8 @ w_quant) → f32 = raw / (127/w_scale) / (127/act_scale)
    for b in range(B):
        out_i8[b] *= w_scale_i8 / 127.0 * act_scale[b] / 127.0
        out_i4[b] *= w_scale_i4 / 7.0 * act_scale[b] / 127.0

    diff = np.max(np.abs(out_i8 - out_i4))
    assert diff < 5.0, (
        f"i8 vs i4 dequantized max diff = {diff:.4f} (int4 coarser by 18×)"
    )
