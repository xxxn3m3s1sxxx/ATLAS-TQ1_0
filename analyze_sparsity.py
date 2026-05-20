"""Analyze zero-weight distribution in TQ1-packed tensors for N:M sparsity planning."""
import struct, numpy as np, os, sys

ATLAS_PATH = r"C:\atlas\falcon3-10b-tq1.atlas"

TQ1_MUL = np.array([1, 3, 9, 27, 81], dtype=np.uint32)

def decode_tq1_row(packed_bytes):
    """Decode TQ1 bytes to ternary values. Each byte → 5 ternary values {-1,0,+1}."""
    arr = np.frombuffer(packed_bytes, dtype=np.uint8).astype(np.uint32)
    t = arr[:, None]
    vals = ((t // TQ1_MUL) % 3).astype(np.int8) - 1
    return vals.ravel()

def read_atlas_header(path):
    with open(path, 'rb') as f:
        hdr = f.read(64)
    magic = hdr[0:5]
    version = struct.unpack_from('<H', hdr, 5)[0]
    n_layers = struct.unpack_from('<H', hdr, 7)[0]
    hidden = struct.unpack_from('<H', hdr, 9)[0]
    inter = struct.unpack_from('<H', hdr, 11)[0]
    n_heads = struct.unpack_from('<B', hdr, 13)[0]
    n_kv = struct.unpack_from('<B', hdr, 14)[0]
    head_dim = struct.unpack_from('<H', hdr, 15)[0]
    vocab = struct.unpack_from('<I', hdr, 17)[0]
    rope_theta = struct.unpack_from('<d', hdr, 21)[0]
    name_block_size = struct.unpack_from('<I', hdr, 56)[0]
    n_tensors = struct.unpack_from('<I', hdr, 60)[0]
    return {
        'magic': magic, 'version': version, 'n_layers': n_layers,
        'hidden': hidden, 'inter': inter, 'n_heads': n_heads,
        'n_kv': n_kv, 'head_dim': head_dim, 'vocab': vocab,
        'rope_theta': rope_theta, 'name_block_size': name_block_size,
        'n_tensors': n_tensors
    }

def read_dir_and_names(path, hdr):
    n = hdr['n_tensors']
    dir_size = n * 12
    with open(path, 'rb') as f:
        f.seek(64)
        dir_bytes = f.read(dir_size)
        name_block_size_bytes = f.read(4)
        name_block_size = struct.unpack('<I', name_block_size_bytes)[0]
        name_bytes = f.read(name_block_size - 4)
        names = name_bytes.rstrip(b'\x00').split(b'\x00')
        names = [s.decode() for s in names if s]

    entries = []
    for i in range(n):
        off = i * 12
        ttype = dir_bytes[off]
        offset = struct.unpack_from('<I', dir_bytes, off + 1)[0]
        row_dim = struct.unpack_from('<I', dir_bytes, off + 5)[0]
        packed_per_row = dir_bytes[off + 9] | (dir_bytes[off + 10] << 8) | (dir_bytes[off + 11] << 16)
        entries.append({
            'ttype': ttype, 'offset': offset, 'row_dim': row_dim,
            'packed_per_row': packed_per_row, 'name': names[i] if i < len(names) else f"tensor_{i}"
        })
    return entries

def main():
    print(f"Opening {ATLAS_PATH}...")
    hdr = read_atlas_header(ATLAS_PATH)
    print(f"Magic: {hdr['magic']}, Version: {hdr['version']}, Layers: {hdr['n_layers']}")
    print(f"Hidden: {hdr['hidden']}, Inter: {hdr['inter']}, Heads: {hdr['n_heads']}/{hdr['n_kv']}")
    print(f"Vocab: {hdr['vocab']}, Head_dim: {hdr['head_dim']}")
    print(f"Tensors: {hdr['n_tensors']}")

    entries = read_dir_and_names(ATLAS_PATH, hdr)
    tq1_entries = [e for e in entries if e['ttype'] == 0]
    print(f"\nTQ1 tensors: {len(tq1_entries)}")

    # Categorize by tensor type
    categories = {
        'gate_proj': [], 'up_proj': [], 'down_proj': [],
        'q_proj': [], 'k_proj': [], 'v_proj': [], 'o_proj': [],
        'lm_head': []
    }

    for e in tq1_entries:
        name = e['name']
        for cat in categories:
            if cat in name:
                categories[cat].append(e)
                break

    print(f"\nPer-category tensor counts:")
    for cat, tensors in categories.items():
        print(f"  {cat:15s}: {len(tensors)} tensors")
    uncategorized = [e for e in tq1_entries if not any(c in e['name'] for c in categories)]
    if uncategorized:
        print(f"  {'other':15s}: {len(uncategorized)} tensors")
        for e in uncategorized[:5]:
            print(f"    {e['name']}")

    # Sample analysis: analyze first layer's weights per category
    print(f"\n═══ LAYER 0 SPARSITY ANALYSIS ═══")
    with open(ATLAS_PATH, 'rb') as f:
        # Get layer 0 entries
        layer0 = [e for e in tq1_entries if 'layers.0.' in e['name'] and not e['name'].endswith('weight_scale')]

        results = {}
        for e in layer0:
            n_rows = e['row_dim']
            n_packed = e['packed_per_row']
            f.seek(e['offset'])
            data = f.read(n_rows * n_packed)

            # Decode all ternary values
            all_vals = np.frombuffer(data, dtype=np.uint8).astype(np.uint32)
            all_t = all_vals.reshape(-1, 1)
            decoded = ((all_t // TQ1_MUL) % 3).astype(np.int8) - 1
            total_vals = decoded.size
            n_zero = int(np.sum(decoded == 0))
            n_pos = int(np.sum(decoded == 1))
            n_neg = int(np.sum(decoded == -1))
            frac_zero = n_zero / total_vals

            # Per-block analysis (N=32 blocks = 32 * 5 = 160 values)
            block_size_vals = 160  # 32 bytes × 5 values per byte
            blocks = decoded.reshape(-1, block_size_vals) if total_vals >= block_size_vals else decoded.reshape(1, -1)
            block_zeros = np.sum(blocks == 0, axis=1)
            n_m_zero = np.sum(block_zeros == block_size_vals)  # fully zero blocks
            n_full_blocks = blocks.shape[0]
            frac_full_zero = n_m_zero / n_full_blocks if n_full_blocks > 0 else 0

            short_name = e['name'].split('.')[-2] if 'layers' in e['name'] else e['name']
            results[short_name] = {
                'n_rows': n_rows, 'total_vals': total_vals,
                'n_zero': n_zero, 'frac_zero': frac_zero,
                'n_neg': n_neg, 'n_pos': n_pos,
                'block32_full_zero': frac_full_zero
            }
            print(f"  {short_name:15s}: rows={n_rows:5d} total={total_vals:8d} "
                  f"zero={frac_zero:6.4f} (-:{n_neg:6d} +:{n_pos:6d}) "
                  f"block32_zero={frac_full_zero:.4f}")

        # Also analyze per-category across all layers (sampled or all)
        print(f"\n═══ FULL MODEL SPARSITY (ALL LAYERS) ═══")
        for cat, tensors in categories.items():
            if not tensors:
                continue
            all_fracs = []
            all_block_zeros = []
            for e in tensors:
                n_rows = e['row_dim']
                n_packed = e['packed_per_row']
                f.seek(e['offset'])
                data = f.read(n_rows * n_packed)
                all_t = np.frombuffer(data, dtype=np.uint8).astype(np.uint32)
                all_t_2d = all_t.reshape(-1, 1)
                decoded = ((all_t_2d // TQ1_MUL) % 3).astype(np.int8) - 1
                decoded = decoded.ravel()
                frac = float(np.mean(decoded == 0))
                all_fracs.append(frac)

                # Block analysis (32-byte blocks)
                total_vals = decoded.size
                block_size = 160
                if total_vals >= block_size:
                    blocks = decoded[:total_vals - (total_vals % block_size)].reshape(-1, block_size)
                    block_zeros = np.sum(np.all(blocks == 0, axis=1))
                    n_blocks = blocks.shape[0]
                else:
                    block_zeros = 0
                    n_blocks = 0
                all_block_zeros.append(block_zeros / n_blocks if n_blocks > 0 else 0)

            mean_frac = np.mean(all_fracs)
            min_frac = np.min(all_fracs)
            max_frac = np.max(all_fracs)
            mean_block_zero = np.mean(all_block_zeros)
            print(f"  {cat:15s}: zero={mean_frac:.4f} [{min_frac:.4f}–{max_frac:.4f}] "
                  f"block32_zero={mean_block_zero:.4f}")

        # Check N:M sparsity specifically: are zero weights randomly distributed or clustered?
        print(f"\n═══ N:M SPARSITY PATTERN ANALYSIS (Layer 0 gate_proj) ═══")
        # Find gate_proj for layer 0
        gate_e = next((e for e in layer0 if 'gate_proj' in e['name']), None)
        if gate_e:
            n_rows = gate_e['row_dim']
            n_packed = gate_e['packed_per_row']
            f.seek(gate_e['offset'])
            data = f.read(n_rows * n_packed)
            all_t = np.frombuffer(data, dtype=np.uint8).astype(np.uint32)
            all_t_2d = all_t.reshape(-1, 1)
            decoded = ((all_t_2d // TQ1_MUL) % 3).astype(np.int8) - 1
            decoded = decoded.ravel()
            decoded_2d = decoded.reshape(n_rows, -1)

            # N:M: for each group of N consecutive weights, how many are zero?
            for n_size in [2, 4, 8]:
                total_groups = decoded_2d.shape[1] // n_size
                if total_groups == 0:
                    continue
                groups = decoded_2d[:, :total_groups * n_size].reshape(n_rows, -1, n_size)
                group_nz = np.sum(groups != 0, axis=2)  # [rows, groups] - non-zero count per group
                # For N:M sparsity: we skip groups where all M are zero
                # M = N (fully sparse group)
                frac_full_zero = np.mean(group_nz == 0)
                # M = ceil(N/2) (mostly sparse)
                frac_mostly_zero = np.mean(group_nz <= (n_size // 2))
                print(f"  N={n_size:2d}: groups={total_groups:6d} "
                      f"full_zero={frac_full_zero:.4f} half_zero={frac_mostly_zero:.4f}")

        # Actual ternary weight distribution (not just zeros)
        print(f"\n═══ TERNARY VALUE DISTRIBUTION (Layer 0, all TQ1) ═══")
        all_vals = []
        for e in layer0:
            n_rows = e['row_dim']
            n_packed = e['packed_per_row']
            f.seek(e['offset'])
            data = f.read(n_rows * n_packed)
            all_t = np.frombuffer(data, dtype=np.uint8).astype(np.uint32)
            all_t_2d = all_t.reshape(-1, 1)
            decoded = ((all_t_2d // TQ1_MUL) % 3).astype(np.int8) - 1
            decoded = decoded.ravel()
            all_vals.append(decoded)
        combined = np.concatenate(all_vals)
        n_total = combined.size
        n_m1 = int(np.sum(combined == -1))
        n_0 = int(np.sum(combined == 0))
        n_p1 = int(np.sum(combined == 1))
        print(f"  Total: {n_total:,}")
        print(f"  -1: {n_m1:12,} ({n_m1/n_total*100:.2f}%)")
        print(f"   0: {n_0:12,} ({n_0/n_total*100:.2f}%)")
        print(f"  +1: {n_p1:12,} ({n_p1/n_total*100:.2f}%)")

    print(f"\n═══ DONE ═══")

if __name__ == '__main__':
    main()
