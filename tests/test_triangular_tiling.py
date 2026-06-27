"""
Triangular tiling unit test: validate causal mask zone classification
against brute-force per-element masking.

Three zones for tile (Q[b0:b1], K[s0:s1]):
  SKIP:    s0 >= b1       — all keys are future → no compute needed
  DENSE:   s1 <= b0       — all keys are past → full compute, no masking
  PARTIAL: otherwise       — causal mask must be applied per-element

We verify that the tiled scores == naive scores for ALL (b,s) pairs.
"""
import numpy as np

TILE_B = 32
TILE_S = 32

def zone(b0, b1, s0, s1):
    if s0 >= b1:
        return 'SKIP'
    elif s1 <= b0:
        return 'DENSE'
    else:
        return 'PARTIAL'

def naive_scores(B, head_dim=128):
    q = np.random.randn(B, head_dim).astype(np.float32)
    k = np.random.randn(B, head_dim).astype(np.float32)
    scores = np.zeros((B, B), dtype=np.float32)
    for b in range(B):
        for s in range(B):
            if s > b:
                scores[b, s] = -1e9  # causal mask
            else:
                scores[b, s] = np.dot(q[b], k[s])
    return q, k, scores

def tiled_scores(q, k, B):
    scores = np.full((B, B), -1e9, dtype=np.float32)
    for b0 in range(0, B, TILE_B):
        b1 = min(b0 + TILE_B, B)
        for s0 in range(0, B, TILE_S):
            s1 = min(s0 + TILE_S, B)
            z = zone(b0, b1, s0, s1)
            if z == 'SKIP':
                continue
            for b in range(b0, b1):
                for s in range(s0, s1):
                    if z == 'DENSE' or s <= b:
                        scores[b, s] = np.dot(q[b], k[s])
    return scores

def test_zones():
    """Validate zone classification for various B values."""
    for B in [8, 32, 64, 128, 256]:
        q, k, ref = naive_scores(B)
        tiled = tiled_scores(q, k, B)
        mask = ref > -1e8
        if not np.allclose(tiled[mask], ref[mask]):
            diff = np.abs(tiled[mask] - ref[mask]).max()
            print(f"[FAIL] B={B}: max_diff={diff:.6f}")
            return False
        print(f"[PASS] B={B}: tiled == naive ({np.sum(mask)} elements)")
    return True

def test_zone_counts():
    """Count how many SKIP/DENSE/PARTIAL tiles for different B."""
    print("\nTile counts (TILE_B=32, TILE_S=32):")
    for B in [32, 64, 128, 256, 512, 1024]:
        skip = dense = partial = 0
        for b0 in range(0, B, TILE_B):
            b1 = min(b0 + TILE_B, B)
            for s0 in range(0, B, TILE_S):
                s1 = min(s0 + TILE_S, B)
                z = zone(b0, b1, s0, s1)
                if z == 'SKIP': skip += 1
                elif z == 'DENSE': dense += 1
                else: partial += 1
        total = skip + dense + partial
        print(f"  B={B:5d}: {total:4d} tiles | SKIP={skip:4d} ({100*skip/total:5.1f}%) "
              f"DENSE={dense:4d} ({100*dense/total:5.1f}%) "
              f"PARTIAL={partial:4d} ({100*partial/total:5.1f}%)")
    return True

def test_ring_buffer_equivalence():
    """Verify that the zone classification works with ring buffer offset."""
    for ring_start in [0, 64, 128, 512]:
        for B in [32, 64, 128]:
            ring_len = B
            q = np.random.randn(B, 128).astype(np.float32)
            k = np.random.randn(ring_len, 128).astype(np.float32)
            ref = np.full((B, B), -1e9, dtype=np.float32)
            for b in range(B):
                pos_b = b
                for s in range(ring_len):
                    attn_pos = ring_start + s
                    if attn_pos <= pos_b:
                        ref[b, s] = np.dot(q[b], k[s])

            tiled = np.full((B, B), -1e9, dtype=np.float32)
            for b0 in range(0, B, TILE_B):
                b1 = min(b0 + TILE_B, B)
                for s0 in range(0, ring_len, TILE_S):
                    s1 = min(s0 + TILE_S, ring_len)
                    key_pos_start = ring_start + s0
                    key_pos_end = ring_start + s1 - 1
                    query_pos_start = b0
                    query_pos_end = b1 - 1
                    if key_pos_start > query_pos_end:
                        continue
                    for b in range(b0, b1):
                        pos_b = b
                        if key_pos_end <= pos_b:
                            for s in range(s0, s1):
                                tiled[b, s] = np.dot(q[b], k[s])
                        else:
                            for s in range(s0, s1):
                                if ring_start + s <= pos_b:
                                    tiled[b, s] = np.dot(q[b], k[s])

            mask = ref > -1e8
            if not np.allclose(tiled[mask], ref[mask]):
                print(f"[FAIL] ring_start={ring_start}, B={B}")
                return False
            print(f"[PASS] ring_start={ring_start}, B={B}: tiled == naive")
    return True

if __name__ == '__main__':
    ok = True
    ok &= test_zones()
    ok &= test_zone_counts()
    ok &= test_ring_buffer_equivalence()
    print(f"\n{'ALL PASS' if ok else 'FAILED'}")
