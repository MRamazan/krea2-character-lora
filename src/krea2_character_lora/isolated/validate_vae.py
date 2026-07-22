import json
import os
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage
from safetensors.torch import load_file

source_path = Path(os.environ["KREA2_VAE_SOURCE"])
normalized_directory = Path(os.environ["KREA2_VAE_DIRECTORY"])
result_path = Path(os.environ["KREA2_VAE_RESULT"])
mapping_document = json.loads(Path(os.environ["KREA2_VAE_KEY_MAPPING"]).read_text(encoding="utf-8"))
conversion = os.environ.get("KREA2_VAE_CONVERSION", mapping_document.get("conversion", "identity"))
mapping = mapping_document["mapping"]

state_dict = load_file(str(source_path))
if set(mapping) != set(state_dict):
    raise RuntimeError("The custom VAE key mapping does not cover the checkpoint keys exactly.")
converted = {mapping[source_key]: tensor for source_key, tensor in state_dict.items()}
if len(converted) != len(state_dict):
    raise RuntimeError("The custom VAE key mapping is not injective.")

config = json.loads((normalized_directory / "config.json").read_text(encoding="utf-8"))
vae = AutoencoderKLQwenImage.from_config(config)
missing, unexpected = vae.load_state_dict(converted, strict=False)
if missing or unexpected:
    raise RuntimeError(
        "Strict AutoencoderKLQwenImage validation failed after conversion. "
        f"Missing keys: {list(missing)[:10]}. Unexpected keys: {list(unexpected)[:10]}."
    )

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
    raise RuntimeError("The custom VAE produced non-finite latents during the smoke test.")
if not torch.isfinite(reconstruction).all():
    raise RuntimeError("The custom VAE produced a non-finite reconstruction during the smoke test.")

vae.save_pretrained(str(normalized_directory))

result = {
    "architecture": "AutoencoderKLQwenImage",
    "conversion": conversion,
    "converted_tensor_count": len(converted),
    "latent_channels": latent_channels,
    "latent_shape": list(latent.shape),
    "reconstruction_shape": list(reconstruction.shape),
    "load_state_dict_strict": True,
    "encode_decode_smoke_test": "passed",
}
result_path.write_text(json.dumps(result), encoding="utf-8")
print(json.dumps(result))
