import struct, sys, numpy as np

def read_tensor_info(path, tensor_name):
    with open(path, "rb") as f:
        h = f.read(64)
        n_tensors = struct.unpack_from("<I", h, 60)[0]
        name_block_size = struct.unpack_from("<I", h, 56)[0]
        directory = f.read(n_tensors * 12)
        name_block = f.read(name_block_size)
        names = name_block[4:].split(b'\x00')
        for i in range(n_tensors):
            nm = names[i].decode(errors='replace') if i < len(names) else ''
            if nm == tensor_name:
                entry = directory[i*12:i*12+12]
                ttype = entry[0]; off = struct.unpack_from("<I", entry, 1)[0]
                row_dim = struct.unpack_from("<I", entry, 5)[0]
                ppr = entry[9] | (entry[10] << 8) | (entry[11] << 16)
                print(f"  {nm}: ttype={ttype} row={row_dim} ppr={ppr} off={off}")
                f.seek(off)
                if ttype != 5: raise ValueError(f"Not ttype=5: {ttype}")
                data = f.read(3)
                bs = data[0]; nb = struct.unpack_from("<H", data, 1)[0]
                print(f"  block_size={bs} n_blocks={nb}")
                scale_bytes = row_dim * nb * 2
                packed_bytes = row_dim * ppr
                f.seek(off + 3)
                scales_raw = f.read(scale_bytes)
                packed_raw = f.read(packed_bytes)
                scales = np.frombuffer(scales_raw, dtype=np.float16).reshape(row_dim, nb).astype(np.float32)
                # Decode trits
                tq1_decode = np.zeros((256, 5), dtype=np.int8)
                for b in range(256):
                    t = b
                    tq1_decode[b, 0] = (t % 3) - 1; t //= 3
                    tq1_decode[b, 1] = (t % 3) - 1; t //= 3
                    tq1_decode[b, 2] = (t % 3) - 1; t //= 3
                    tq1_decode[b, 3] = (t % 3) - 1; t //= 3
                    tq1_decode[b, 4] = (t % 3) - 1
                input_dim = ppr * 5
                ternary = np.zeros((row_dim, input_dim), dtype=np.int8)
                for r in range(row_dim):
                    for c in range(ppr):
                        b = packed_raw[r*ppr + c]
                        trits = tq1_decode[b]
                        col = c * 5
                        for k in range(5):
                            if col + k < input_dim:
                                ternary[r, col + k] = trits[k]
                return scales, ternary, bs, nb, row_dim, input_dim, ppr
    return None

def compute_tq1(act, scales, ternary, bs, nb, input_dim):
    """Replicate C++ matmul_tq1_block_fused_s8 in Python."""
    act = act.astype(np.float32).flatten()
    if len(act) < input_dim:
        act = np.pad(act, (0, input_dim - len(act)))
    max_val = np.max(np.abs(act))
    if max_val < 1e-5: max_val = 1e-5
    scale_x = max_val / 127.0
    inv = 127.0 / max_val
    act_q = np.round(act * inv).clip(-127, 127).astype(np.int8)
    rows = scales.shape[0]
    output = np.zeros(rows, dtype=np.float32)
    for r in range(rows):
        rscales = scales[r]; w_ter = ternary[r]
        dot = 0.0
        for blk in range(nb):
            blk_start = blk * bs
            blk_end = min(blk_start + bs, input_dim)
            d = int(np.sum(act_q[blk_start:blk_end].astype(np.int32) * w_ter[blk_start:blk_end].astype(np.int32)))
            dot += d * rscales[blk]
        output[r] = dot * scale_x
    return output, scale_x, act_q, max_val

def compute_f32_ref(act, scales, ternary, bs, nb, input_dim):
    """Full f32 matmul (no int8 quant)."""
    act = act.astype(np.float32).flatten()
    if len(act) < input_dim:
        act = np.pad(act, (0, input_dim - len(act)))
    rows = scales.shape[0]
    output = np.zeros(rows, dtype=np.float32)
    for r in range(rows):
        rscales = scales[r]; w_ter = ternary[r].astype(np.float32)
        for blk in range(nb):
            blk_start = blk * bs
            blk_end = min(blk_start + bs, input_dim)
            d = np.dot(act[blk_start:blk_end], w_ter[blk_start:blk_end])
            output[r] += d * rscales[blk]
    return output

path = sys.argv[1]

# Test: Attention output -> O_proj
np.random.seed(42)
attn_out = np.random.randn(1792).astype(np.float32)
attn_out *= 0.2846 / np.linalg.norm(attn_out)  # target norm 0.2846
print("attn_out norm:", np.linalg.norm(attn_out))
print("\n=== O_proj ===")
scales, ternary, bs, nb, rows, input_dim, ppr = read_tensor_info(path, "model.layers.0.self_attn.o_proj.weight")
out_tq1, sx, aq, mv = compute_tq1(attn_out, scales, ternary, bs, nb, input_dim)
out_ref = compute_f32_ref(attn_out, scales, ternary, bs, nb, input_dim)
print(f"max_val={mv:.6f} scale_x={sx:.8f}")
print(f"act_q range: [{aq.min()}, {aq.max()}]")
print(f"TQ1 output norm: {np.linalg.norm(out_tq1):.4f}")
print(f"F32 ref norm: {np.linalg.norm(out_ref):.4f}")
print(f"TQ1/F32 ratio: {np.linalg.norm(out_tq1)/np.linalg.norm(out_ref):.4f}")

# Scale stats
print(f"scales: min={scales.min():.6f} max={scales.max():.6f} mean={scales.mean():.6f}")

# Test: ln1_out -> Q_proj
np.random.seed(42)
ln1_out = np.random.randn(1792).astype(np.float32)
ln1_out *= 0.5866 / np.linalg.norm(ln1_out)
print("\n=== Q_proj ===")
scales_q, ternary_q, bs_q, nb_q, rows_q, input_dim_q, ppr_q = read_tensor_info(path, "model.layers.0.self_attn.q_proj.weight")
out_tq1_q, sx_q, aq_q, mv_q = compute_tq1(ln1_out, scales_q, ternary_q, bs_q, nb_q, input_dim_q)
out_ref_q = compute_f32_ref(ln1_out, scales_q, ternary_q, bs_q, nb_q, input_dim_q)
print(f"max_val={mv_q:.6f} scale_x={sx_q:.8f}")
print(f"act_q range: [{aq_q.min()}, {aq_q.max()}]")
print(f"TQ1 output norm: {np.linalg.norm(out_tq1_q):.4f}")
print(f"F32 ref norm: {np.linalg.norm(out_ref_q):.4f}")
print(f"scales: min={scales_q.min():.6f} max={scales_q.max():.6f} mean={scales_q.mean():.6f}")

# Test: Silu(gate*up) -> down_proj
np.random.seed(42)
ffn_in = np.random.randn(5120).astype(np.float32)
ffn_in *= 68.0 / np.linalg.norm(ffn_in)
print("\n=== Down_proj ===")
scales_d, ternary_d, bs_d, nb_d, rows_d, input_dim_d, ppr_d = read_tensor_info(path, "model.layers.0.mlp.down_proj.weight")
out_tq1_d, sx_d, aq_d, mv_d = compute_tq1(ffn_in, scales_d, ternary_d, bs_d, nb_d, input_dim_d)
out_ref_d = compute_f32_ref(ffn_in, scales_d, ternary_d, bs_d, nb_d, input_dim_d)
print(f"max_val={mv_d:.6f} scale_x={sx_d:.8f}")
print(f"act_q range: [{aq_d.min()}, {aq_d.max()}]")
print(f"TQ1 output norm: {np.linalg.norm(out_tq1_d):.4f}")
print(f"F32 ref norm: {np.linalg.norm(out_ref_d):.4f}")
print(f"scales: min={scales_d.min():.6f} max={scales_d.max():.6f} mean={scales_d.mean():.6f}")
