#!/usr/bin/env python3
"""
infer_bonsai.py - Bonsai Image Tern�rer Diffusions-Pipeline Driver.

L�dt Text-Embeddings via PyTorch/Qwen3, injiziert sie in die C++ Engine,
f�hrt den Denoise-Loop aus und gibt die Latent-Rohdaten zur�ck.

Usage:
    python infer_bonsai.py                            # Default: 4 steps, 64x64
    python infer_bonsai.py --prompt "A cat" --steps 4 --size 64 --out out.raw
"""

import os, sys, ctypes, time, struct, argparse
import numpy as np
from pathlib import Path

# ─── Pfade ────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
DLL_PATH = HERE / "atlas_diffusion.dll"
ATLAS_PATH = HERE / "models" / "bonsai-image-4B.atlas"
ATLAS_DLL_PATH = HERE / "atlas.dll"
QWEN_PATH = HERE / "models" / "bonsai-qwen3-4B.atlas"

# ─── Ctypes - C++ API ─────────────────────────────────────────────────────

def load_dll(path: str):
    """Load atlas_diffusion.dll and set argtypes/restypes."""
    if not os.path.exists(path):
        print(f"[ERROR] DLL not found: {path}")
        sys.exit(1)
    dll = ctypes.CDLL(str(path))

    # diffusion_create() -> void*
    dll.diffusion_create.restype = ctypes.c_void_p
    dll.diffusion_create.argtypes = []

    # diffusion_destroy(void*)
    dll.diffusion_destroy.restype = None
    dll.diffusion_destroy.argtypes = [ctypes.c_void_p]

    # diffusion_init(void*, char*, char*, char*) -> int
    dll.diffusion_init.restype = ctypes.c_int
    dll.diffusion_init.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]

    # diffusion_set_shift(void*, double)
    dll.diffusion_set_shift.restype = None
    dll.diffusion_set_shift.argtypes = [ctypes.c_void_p, ctypes.c_double]

    # diffusion_denoise(void*, np.ndarray, np.ndarray, int, int, int, int, int, np.ndarray)
    # We use ctypes.c_void_p for pointer args to support numpy arrays
    dll.diffusion_denoise.restype = ctypes.c_int
    dll.diffusion_denoise.argtypes = [
        ctypes.c_void_p,  # dm
        ctypes.c_void_p,  # latent_seq (float*)
        ctypes.c_void_p,  # txt_emb (float*)
        ctypes.c_int,     # txt_dim
        ctypes.c_int,     # n_tokens
        ctypes.c_int,     # latent_hw
        ctypes.c_int,     # latent_dim
        ctypes.c_int,     # n_steps
        ctypes.c_void_p,  # timesteps (float*) or NULL
    ]

    # diffusion_generate(void*, char*, int, int, int, np.ndarray) -> int
    dll.diffusion_generate.restype = ctypes.c_int
    dll.diffusion_generate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]

    return dll


# ─── Qwen3 Text Embeddings ────────────────────────────────────────────────

def get_text_embeddings(prompt: str, model, tokenizer, device: str = "cpu",
                        layers: list = None) -> np.ndarray:
    """
    Extrahiert Text-Embeddings aus Qwen3: Layer 9/18/27 -> concat -> [seq, 7680].

    Returns: float32 numpy array [n_tokens, 7680] (C-contiguous)
    """
    import torch
    if layers is None:
        layers = [9, 18, 27]

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=256).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is tuple of (embedding, layer1, ..., layerN)
    # Each is [1, seq_len, 2560]
    hs_list = [outputs.hidden_states[i] for i in layers]
    # Stack along last dim: [1, seq_len, 7680]
    cat = torch.cat(hs_list, dim=-1).float()
    cat = cat.squeeze(0)

    return cat.cpu().numpy().astype(np.float32)


def load_text_encoder(qwen_path: str, device: str = "cpu"):
    """
    L�dt Qwen3 Modell und Tokenizer von lokalem Pfad oder HuggingFace.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"[Qwen] Loading from: {qwen_path}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True, subfolder="tokenizer")
    model = AutoModelForCausalLM.from_pretrained(
        qwen_path,
        trust_remote_code=True,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        device_map=device,
        output_hidden_states=True,
    )
    model.eval()
    print(f"[Qwen] Loaded in {time.time()-t0:.1f}s ({sum(p.numel() for p in model.parameters())/1e9:.1f}B params)")
    return model, tokenizer


# ─── Denoise ───────────────────────────────────────────────────────────────

def run_denoise(dll, dm_ptr, txt_emb: np.ndarray, latent_hw: int = 64,
                latent_dim: int = 128, n_steps: int = 4, shift: float = 3.0,
                seed: int = 42) -> np.ndarray:
    """
    F�hre Denoise-Loop aus.

    Args:
        txt_emb: [n_tokens, 7680] float32 - Text-Embeddings von Qwen3
        latent_hw: sqrt der Anzahl Patches (z.B. 64 = 4096 Patches)
        latent_dim: Kanalzahl des Latents (128 f�r FLUX)

    Returns: latent [latent_hw*latent_hw, latent_dim] float32
    """
    n_patches = latent_hw * latent_hw
    latent_size = n_patches * latent_dim
    n_tokens = txt_emb.shape[0]
    txt_dim = txt_emb.shape[1]  # 7680

    # Initialisiere Latent mit Rauschen N(0,1)
    rng = np.random.RandomState(seed)
    latent = rng.randn(latent_size).astype(np.float32)

    # Set shift
    if shift != 3.0:
        dll.diffusion_set_shift(dm_ptr, ctypes.c_double(shift))

    print(f"[DENOISE] Running {n_steps} steps on {latent_hw}x{latent_hw} latent "
          f"({n_patches} patches, {n_tokens} text tokens)")
    print(f"[DENOISE] txt_emb: shape={txt_emb.shape}, "
          f"mean={txt_emb.mean():.4f}, std={txt_emb.std():.4f}")
    t0 = time.time()

    latent_ptr = latent.ctypes.data_as(ctypes.c_void_p)
    txt_ptr = txt_emb.ctypes.data_as(ctypes.c_void_p)

    ret = dll.diffusion_denoise(
        dm_ptr,
        latent_ptr,
        txt_ptr,
        ctypes.c_int(txt_dim),
        ctypes.c_int(n_tokens),
        ctypes.c_int(latent_hw),
        ctypes.c_int(latent_dim),
        ctypes.c_int(n_steps),
        None,  # timesteps (auto-compute)
    )

    elapsed = time.time() - t0
    if ret != 0:
        print(f"[ERROR] diffusion_denoise returned {ret}")
        return latent.reshape(n_patches, latent_dim)

    print(f"[DENOISE] Done in {elapsed:.1f}s ({elapsed/max(n_steps,1):.1f}s/step)")
    lat_mean = latent.mean()
    lat_std = latent.std()
    lat_min = latent.min()
    lat_max = latent.max()
    print(f"[DENOISE] Output latent: mean={lat_mean:.4f}, std={lat_std:.4f}, "
          f"range=[{lat_min:.4f}, {lat_max:.4f}]")

    return latent.reshape(n_patches, latent_dim)


# ─── Save Raw Latent ───────────────────────────────────────────────────────

def save_latent_raw(latent: np.ndarray, path: str, latent_hw: int, latent_dim: int):
    """Save as .raw file (flattened float32)."""
    flat = latent.ravel().astype(np.float32)
    with open(path, 'wb') as f:
        f.write(flat.tobytes())
    print(f"[SAVE] Wrote {len(flat)*4/1024**2:.1f} MB to {path}")

    # Also save a sidecar with shape info
    meta_path = path + ".meta"
    with open(meta_path, 'w') as f:
        f.write(f"latent_hw={latent_hw}\nlatent_dim={latent_dim}\n"
                f"n_patches={latent_hw*latent_hw}\n")
    print(f"[SAVE] Meta: {meta_path}")


# ─── Simple VAE decode helper (placeholder) ───────────────────────────────

def decode_latent_placeholder(latent: np.ndarray, latent_hw: int) -> np.ndarray:
    """
    Placeholder: zeigt nur, was die VAE erwarten w�rde.
    FLUX VAE: latent [128, h/8, w/8] -> RGB [3, h, w]
    """
    # Reshape to [C, H, W]
    c = 128
    h = latent_hw
    img = latent.reshape(c, h, h)
    print(f"[VAE] Latent reshaped: {img.shape}, mean={img.mean():.4f}, std={img.std():.4f}")
    return img


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Bonsai Image Diffusion Pipeline Driver")
    parser.add_argument("--prompt", default="A photorealistic mountain landscape in autumn, highly detailed",
                        help="Text prompt")
    parser.add_argument("--steps", type=int, default=4,
                        help="Number of denoising steps")
    parser.add_argument("--size", type=int, default=64,
                        help="Latent HW (64 = 512x512 output)")
    parser.add_argument("--shift", type=float, default=1.0,
                        help="Timestep shift (1.0=exponential dynamic, 3.0=legacy linear)")
    parser.add_argument("--out", default="output.raw",
                        help="Output .raw file path")
    parser.add_argument("--device", default="cpu",
                        help="Device for Qwen3 ('cpu' or 'cuda')")
    parser.add_argument("--qwen-path", default=None,
                        help="Path to Qwen3 model (default: HuggingFace cache)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initial latent")
    parser.add_argument("--no-text", action="store_true",
                        help="Run with zero text embeddings (baseline test)")
    args = parser.parse_args()

    print("=" * 60)
    print("Bonsai Image Diffusion Inference")
    print("=" * 60)
    print(f"Prompt:  {args.prompt}")
    print(f"Steps:   {args.steps}")
    print(f"Size:    {args.size}x{args.size} latent")
    print(f"Shift:   {args.shift}")
    print(f"Device:  {args.device}")

    # ── 1. Load DLL ──
    dll = load_dll(str(DLL_PATH if os.path.exists(str(DLL_PATH)) else ATLAS_DLL_PATH))

    # ── 2. Create + Init Model ──
    dm_ptr = dll.diffusion_create()
    if not dm_ptr:
        print("[ERROR] diffusion_create failed")
        sys.exit(1)

    atlas_p = str(ATLAS_PATH).encode('utf-8')
    dll_p = str(ATLAS_DLL_PATH).encode('utf-8') if os.path.exists(str(ATLAS_DLL_PATH)) else b""
    qwen_p = str(QWEN_PATH).encode('utf-8') if os.path.exists(str(QWEN_PATH)) else b""

    ret = dll.diffusion_init(dm_ptr, atlas_p, dll_p, qwen_p)
    if ret != 0:
        print(f"[ERROR] diffusion_init returned {ret}")
        dll.diffusion_destroy(dm_ptr)
        sys.exit(1)
    print("[INIT] Model loaded")

    # ── 3. Set Shift ──
    dll.diffusion_set_shift(dm_ptr, ctypes.c_double(args.shift))

    # ── 4. Get Text Embeddings ──
    if args.no_text:
        # Zero embedding for baseline test
        n_tokens = 1
        txt_emb = np.zeros((n_tokens, 7680), dtype=np.float32)
        print(f"[TEXT] Using zero embeddings (baseline mode)")
    else:
        global torch
        import torch
        qwen_path = args.qwen_path or str(HERE / "models" / "bonsai-qwen3-4B")
        # Try HuggingFace cache if local path not found
        if not os.path.exists(qwen_path):
            qwen_path = "prism-ml/bonsai-image-ternary-4B-unpacked"
            print(f"[TEXT] Using HuggingFace: {qwen_path}")

        model, tokenizer = load_text_encoder(qwen_path, args.device)
        txt_emb = get_text_embeddings(args.prompt, model, tokenizer, args.device)
        print(f"[TEXT] Embeddings: shape={txt_emb.shape}, "
              f"mean={txt_emb.mean():.4f}, std={txt_emb.std():.4f}")

    # ── 5. Run Denoise ──
    latent = run_denoise(dll, dm_ptr, txt_emb,
                         latent_hw=args.size,
                         latent_dim=128,
                         n_steps=args.steps,
                         shift=args.shift,
                         seed=args.seed)

    # ── 6. Save ──
    if args.out:
        save_latent_raw(latent, args.out, args.size, 128)

    # ── 7. Placeholder VAE check ──
    decode_latent_placeholder(latent, args.size)

    # ── Cleanup ──
    dll.diffusion_destroy(dm_ptr)
    print("[DONE]")


if __name__ == "__main__":
    main()
