import json
import os
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage

normalized_directory = Path(os.environ["KREA2_VAE_DIRECTORY"])
result_path = Path(os.environ["KREA2_VAE_RESULT"])

vae = AutoencoderKLQwenImage.from_pretrained(str(normalized_directory))
vae = vae.to("cuda:0", dtype=torch.float32)
vae.eval()

latent_channels = int(getattr(vae.config, "z_dim", getattr(vae.config, "latent_channels", 16)))
temporal_flags = getattr(vae.config, "temperal_downsample", [False])
temporal = sum(1 for flag in temporal_flags if flag)
frames = 2**temporal + 1 if temporal > 0 else 1
sample = torch.randn(1, 3, frames, 64, 64, device="cuda:0", dtype=torch.float32)

with torch.no_grad():
    encoded = vae.encode(sample)
    latent = encoded.latent_dist.sample() if hasattr(encoded, "latent_dist") else encoded.sample()
    decoded = vae.decode(latent)
    reconstruction = decoded.sample if hasattr(decoded, "sample") else decoded

if not torch.isfinite(latent).all():
    raise RuntimeError("The Krea 2 default VAE produced non-finite latents during the smoke test.")
if not torch.isfinite(reconstruction).all():
    raise RuntimeError(
        "The Krea 2 default VAE produced a non-finite reconstruction during the smoke test."
    )

result = {
    "architecture": "AutoencoderKLQwenImage",
    "latent_channels": latent_channels,
    "latent_shape": list(latent.shape),
    "reconstruction_shape": list(reconstruction.shape),
    "encode_decode_smoke_test": "passed",
}
result_path.write_text(json.dumps(result), encoding="utf-8")
print(json.dumps(result))
