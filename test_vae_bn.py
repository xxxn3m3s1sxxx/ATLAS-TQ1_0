"""Check VAE batch norm statistics and unpatcify function."""
import torch
from diffusers import AutoencoderKLFlux2

vae = AutoencoderKLFlux2.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", subfolder="vae", torch_dtype=torch.float32
)

# Check BatchNorm
print("=== VAE BatchNorm ===")
bn = vae.bn
print(f"  num_features: {bn.num_features}")
print(f"  running_mean shape: {bn.running_mean.shape}")
print(f"  running_mean: {bn.running_mean}")
print(f"  running_var: {bn.running_var}")
print(f"  running_mean stats: mean={bn.running_mean.mean():.4f}, std={bn.running_mean.std():.4f}")
print(f"  range: [{bn.running_mean.min():.4f}, {bn.running_mean.max():.4f}]")
print(f"  running_var range: [{bn.running_var.min():.4f}, {bn.running_var.max():.4f}]")

# Now let's decode our model output with proper BN denormalization
import numpy as np
latent = np.fromfile("output.raw", dtype=np.float32).reshape(1, 128, 16, 16)
print(f"\nModel output: mean={latent.mean():.3f}, std={latent.std():.3f}")

# Apply BN denormalization
# The model output is in BN-normalized space (model predicts v in BN space)
# So we need: latents = output * bn_std + bn_mean
latents_t = torch.from_numpy(latent)
bn_running_var = bn.running_var.view(1, -1, 1, 1)
bn_running_mean = bn.running_mean.view(1, -1, 1, 1)
bn_eps = vae.config.batch_norm_eps

latents_bn_std = torch.sqrt(bn_running_var + bn_eps)
denorm = latents_t * latents_bn_std + bn_running_mean
print(f"\nAfter BN denormalization:")
print(f"  shape: {denorm.shape}, mean={denorm.mean():.3f}, std={denorm.std():.3f}")

# Now unpatchify: 128→32 channels, 16→32 spatial
def _unpatchify_latents(latents):
    batch_size, num_channels_4x, height, width = latents.shape
    num_channels = num_channels_4x // 4
    latents = latents.reshape(batch_size, num_channels, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels, height * 2, width * 2)
    return latents

unpatched = _unpatchify_latents(denorm)
print(f"\nAfter unpatchify: shape={unpatched.shape}")
print(f"  mean={unpatched.mean():.3f}, std={unpatched.std():.3f}")

# Decode
with torch.no_grad():
    decoded = vae.decode(unpatched.float()).sample
img = (decoded * 0.5 + 0.5).clamp(0, 1)
print(f"\nDecoded: shape={decoded.shape}")
print(f"  range=[{decoded.min():.3f}, {decoded.max():.3f}]")
print(f"  mean={decoded.mean():.3f}, std={decoded.std():.3f}")
print(f"  img range=[{img.min():.4f}, {img.max():.4f}]")

from PIL import Image
img_pil = Image.fromarray((img[0].permute(1,2,0).numpy() * 255).astype(np.uint8))
img_pil.save("output_bn_corrected.png")
print(f"Saved to output_bn_corrected.png")
