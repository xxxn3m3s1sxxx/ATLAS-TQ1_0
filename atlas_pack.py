"""Unified CLI — autodetects model family (Falcon3, Bonsai/Qwen3, TriLM, Phi-3).
"""
import sys, os

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python atlas_pack.py <model_dir> [output.atlas]")
        sys.exit(1)

    model_dir = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None

    cfg_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(cfg_path):
        print(f"[ATLAS] config.json not found in {model_dir}")
        sys.exit(1)

    import json
    with open(cfg_path) as f:
        cfg = json.load(f)

    n_heads = cfg.get('num_attention_heads', 0)
    n_kv = cfg.get('num_key_value_heads', n_heads)
    hidden = cfg.get('hidden_size', 0)
    model_type = cfg.get('model_type', '')
    inter = cfg.get('intermediate_size', 0)
    vocab = cfg.get('vocab_size', 0)

    print(f"[ATLAS] Config: model_type={model_type} hidden={hidden} heads={n_heads}/{n_kv} inter={inter} vocab={vocab}")

    # Detection logic
    is_falcon = ('falcon' in str(model_type).lower() or
                 (n_kv != n_heads and hidden >= 2048 and vocab > 130000 and inter > 0))
    is_phi3 = model_type == 'phi3'

    if is_phi3:
        print(f"[ATLAS] Detected: PHI3 family")
        from atlas_packer_phi3 import create_atlas_phi3, auto_output_name
        if not output:
            output = auto_output_name(model_dir) or f"phi3-{hidden}d-tq1-g128.atlas"
        print(f"[ATLAS] Auto output: {output}")
        create_atlas_phi3(model_dir, output)
    elif is_falcon:
        print("[ATLAS] Detected: FALCON family")
        from atlas_packer import create_atlas
        create_atlas(model_dir, output)
    else:
        # TriLM/Bonsai detection via packer modules
        from atlas_packer_bonsai import create_atlas_bonsai, auto_output_name as bonsai_name
        from atlas_packer_trilm import create_atlas_trilm, auto_output_name as trilm_name

        bn = bonsai_name(model_dir)
        tn = trilm_name(model_dir)

        if bn and not tn:
            print("[ATLAS] Detected: BONSAI/QWEN3 family")
            if not output:
                output = bn
            print(f"[ATLAS] Auto output: {output}")
            create_atlas_bonsai(model_dir, output)
        elif tn and not bn:
            print("[ATLAS] Detected: TRILM family, ~{0}B params".format(
                {1792: "1", 2048: "1.5", 2304: "2.4", 3072: "4", 512: "0.1", 768: "0.2",
                 1024: "0.4", 1280: "0.6", 1536: "0.8"}.get(hidden, "?")))
            if not output:
                output = tn
            print(f"[ATLAS] Auto output: {output}")
            create_atlas_trilm(model_dir, output)
        elif bn or tn:
            print(f"[ATLAS] Ambiguous detection — trying BONSAI first")
            if bn and not output:
                output = bn
            create_atlas_bonsai(model_dir, output)
        else:
            print(f"[ATLAS] Unknown model type {model_type}")
            print("  Try one of: atlas_packer.py, atlas_packer_bonsai.py, atlas_packer_trilm.py, atlas_packer_phi3.py")
            sys.exit(1)

if __name__ == '__main__':
    main()
