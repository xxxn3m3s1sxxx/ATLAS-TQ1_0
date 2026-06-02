import struct, sys
def check(path):
    with open(path, "rb") as f:
        h = f.read(64)
        version = struct.unpack_from("<H", h, 5)[0]
        n_layers = struct.unpack_from("<H", h, 7)[0]
        hidden = struct.unpack_from("<H", h, 9)[0]
        inter = struct.unpack_from("<H", h, 11)[0]
        n_heads = struct.unpack_from("<B", h, 13)[0]
        n_kv_heads = struct.unpack_from("<B", h, 14)[0]
        head_dim = struct.unpack_from("<H", h, 15)[0]
        vocab = struct.unpack_from("<I", h, 17)[0]
        n_tensors = struct.unpack_from("<I", h, 60)[0]
        name_block_size = struct.unpack_from("<I", h, 56)[0]
        print(f"Version:{version} Layers:{n_layers} Hidden:{hidden} Inter:{inter}")
        print(f"Heads:{n_heads} KV:{n_kv_heads} Hdim:{head_dim} Vocab:{vocab}")
        print(f"Tensors:{n_tensors} NameBlock:{name_block_size}")
        dir_size = n_tensors * 12
        directory = f.read(dir_size)
        name_block = f.read(name_block_size)
        names = name_block[4:].split(b'\x00')
        for i in range(min(40, n_tensors)):
            entry = directory[i*12:i*12+12]
            ttype = entry[0]
            off = struct.unpack_from("<I", entry, 1)[0]
            row_dim = struct.unpack_from("<I", entry, 5)[0]
            ppr = entry[9] | (entry[10] << 8) | (entry[11] << 16)
            nm = names[i].decode(errors='replace') if i < len(names) else '?'
            f.seek(off)
            if ttype == 5:
                data = f.read(3)
                bs = data[0]; nb = struct.unpack_from("<H", data, 1)[0]
                print(f"[{i:2d}] {nm:55s} ttype=5 row={row_dim:5d} ppr={ppr:4d} bs={bs} nb={nb} input_dim={ppr*5}")
            elif ttype == 1:
                print(f"[{i:2d}] {nm:55s} ttype=1 row={row_dim:5d}")
            else:
                print(f"[{i:2d}] {nm:55s} ttype={ttype} row={row_dim:5d} ppr={ppr}")
        print("\n--- Layer 0 attention tensor sizes ---")
        for i in range(n_tensors):
            entry = directory[i*12:i*12+12]
            ttype = entry[0]
            off = struct.unpack_from("<I", entry, 1)[0]
            row_dim = struct.unpack_from("<I", entry, 5)[0]
            ppr = entry[9] | (entry[10] << 8) | (entry[11] << 16)
            nm = names[i].decode(errors='replace') if i < len(names) else '?'
            if 'h.0.' in nm and any(x in nm for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                f.seek(off)
                if ttype == 5:
                    data = f.read(3)
                    bs = data[0]; nb = struct.unpack_from("<H", data, 1)[0]
                    print(f"  {nm:55s} ttype=5 row={row_dim:5d} ppr={ppr:4d} bs={bs} nb={nb} scale_bytes={row_dim*nb*2} packed={row_dim*ppr}")
check(sys.argv[1])
