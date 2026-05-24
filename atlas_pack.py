#!/usr/bin/env python3
"""Unified ATLAS Packer — autodetect Falcon3 vs Bonsai/Qwen3, pack to .atlas."""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def detect_model(model_dir):
    """Read config.json and return (family, size_label)."""
    cfg_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(cfg_path):
        print(f"[ERROR] {cfg_path} not found")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = json.load(f)

    hidden = cfg.get('hidden_size', 0)
    inter = cfg.get('intermediate_size', 0)
    n_heads = cfg.get('num_attention_heads', 1)
    n_layers = cfg.get('num_hidden_layers', 0)
    head_dim = cfg.get('head_dim', hidden // n_heads if n_heads else 128)
    vocab = cfg.get('vocab_size', 0)

    # Bonsai/Qwen3 detection: head_dim=128, vocab>131k, or qwen2 model_type
    is_bonsai = (head_dim == 128 or vocab > 131072 or
                 cfg.get('model_type', '').startswith('qwen'))

    # Estimate model size from params
    # Rough: ~2 * hidden * inter * n_layers (FFN) + 4 * hidden^2 * n_layers (attn)
    ffn_params = 3 * hidden * inter * n_layers
    attn_params = 4 * hidden * (n_heads * head_dim) * n_layers
    embed_params = hidden * vocab
    tie_embeddings = cfg.get('tie_word_embeddings', True)
    lm_head_params = 0 if tie_embeddings else hidden * vocab
    total_params = (ffn_params + attn_params + embed_params + lm_head_params) / 1e9
    # Round to nearest standard size (tight tolerance to avoid overlap)
    sizes = [0.5, 1, 1.7, 3, 4, 7, 8, 10, 14, 32, 70]
    best = min(sizes, key=lambda s: abs(total_params - s))
    size_label = f"{total_params:.1f}B"
    if abs(total_params - best) < best * 0.2:
        size_label = f"{best:.0f}B".replace('.0', '')
        if best < 1:
            size_label = f"{best}B".replace('0.', '').replace('5B', '0.5B')

    family = "bonsai" if is_bonsai else "falcon3"

    print(f"[ATLAS] Detected: {family.upper()} family, ~{size_label} params")
    print(f"         {n_layers}L {hidden}H {inter}I {n_heads}h hd={head_dim} vocab={vocab}")

    return family, size_label


def auto_output_name(model_dir, family, size_label):
    """Generate standard output filename."""
    name = os.path.basename(model_dir.rstrip('/\\')).lower()
    if family == "bonsai":
        # Normalize: e.g. "bonsai-8b-tq1-g128.atlas"
        out = f"bonsai-{size_label.lower()}-tq1-g128.atlas"
    else:
        out = f"falcon3-{size_label.lower()}-tq1.atlas"
    return os.path.join(os.path.dirname(model_dir), out)


def main():
    parser = argparse.ArgumentParser(
        description="ATLAS Packer — Convert ternary models to TQ1.0 format")
    parser.add_argument('model_dir', help='Path to model directory (with config.json)')
    parser.add_argument('-o', '--output', help='Output .atlas file path (auto-generated if omitted)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Skip .i8 cache (always decompress from packed)')
    args = parser.parse_args()

    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        print(f"[ERROR] Not a directory: {model_dir}")
        sys.exit(1)

    # Detect model family
    family, size_label = detect_model(model_dir)

    # Auto-generate output path if not specified
    if args.output:
        output_path = args.output
    else:
        output_path = auto_output_name(model_dir, family, size_label)
        print(f"[ATLAS] Auto output: {output_path}")

    # Check if output already exists
    if os.path.exists(output_path):
        print(f"[ATLAS] Output exists: {output_path}")
        answer = input("  Overwrite? [y/N] ").strip().lower()
        if answer != 'y':
            print("[ATLAS] Aborted")
            sys.exit(0)

    print(f"[ATLAS] Packing {family.upper()} model...")
    print(f"[ATLAS] This may take several minutes.\n")

    if family == "bonsai":
        from atlas_packer_bonsai import create_atlas_qwen
        create_atlas_qwen(model_dir, output_path)
    else:
        from atlas_packer import create_atlas_from_config
        # Find the safetensors file(s)
        idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
        if os.path.exists(idx_path):
            safetensors_path = model_dir  # create_atlas_from_config handles sharded
        else:
            single = os.path.join(model_dir, 'model.safetensors')
            if not os.path.exists(single):
                print(f"[ERROR] No model.safetensors found in {model_dir}")
                sys.exit(1)
            safetensors_path = single
        create_atlas_from_config(safetensors_path, output_path)

    print(f"\n[ATLAS] Done → {output_path}")

    # Friendly next step
    print(f"\n  Try it:")
    print(f'  python -c "from atlas_infer import AtlasModel; m = AtlasModel(\'{os.path.basename(output_path)}\'); print(m.generate_c(\'Hello\', temperature=0.0))"')


if __name__ == '__main__':
    main()
