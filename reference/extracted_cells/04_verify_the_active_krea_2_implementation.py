import importlib.metadata
import json
import os
import subprocess

source_environment_manifest = json.loads(
    (PATHS["config"] / "source_environment_manifest.json").read_text(encoding="utf-8")
)
expected_diffusers_commit = source_environment_manifest.get("expected_diffusers_commit")

verification_script = r"""
import importlib.metadata
import inspect
import json
import torch
import diffusers
import transformers
import accelerate
from diffusers import AutoencoderKLQwenImage
from toolkit.config_modules import ModelConfig
from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model, QWEN3_VL_PATH, QWEN_IMAGE_VAE_PATH

distribution = importlib.metadata.distribution("diffusers")
direct_url_text = distribution.read_text("direct_url.json")
direct_url = json.loads(direct_url_text) if direct_url_text else {}
probe_configuration = ModelConfig(
    name_or_path="unused",
    arch="krea2",
    dtype="bf16",
    vae_dtype="bf16",
    te_dtype="bf16",
    model_kwargs={},
)
probe = Krea2Model(device="cpu", model_config=probe_configuration, dtype="bf16")
result = {
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "diffusers": diffusers.__version__,
    "diffusers_commit": direct_url.get("vcs_info", {}).get("commit_id"),
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "model_class": f"{Krea2Model.__module__}.{Krea2Model.__name__}",
    "arch": probe.arch,
    "target_lora_modules": list(probe.target_lora_modules),
    "vae_scale_factor": probe.vae_scale_factor,
    "text_encoder_default": QWEN3_VL_PATH,
    "vae_default": QWEN_IMAGE_VAE_PATH,
    "vae_class": f"{AutoencoderKLQwenImage.__module__}.{AutoencoderKLQwenImage.__name__}",
    "has_save_conversion": hasattr(Krea2Model, "convert_lora_weights_before_save"),
    "has_load_conversion": hasattr(Krea2Model, "convert_lora_weights_before_load"),
    "source_file": inspect.getsourcefile(Krea2Model),
}
print(json.dumps(result))
"""

environment = os.environ.copy()
environment["PYTHONUNBUFFERED"] = "1"
result = subprocess.run(
    [str(PATHS["venv_python"]), "-c", verification_script],
    cwd=str(PATHS["ai_toolkit"]),
    env=environment,
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    raise RuntimeError(f"Krea 2 implementation verification failed.\n{result.stdout}\n{result.stderr}")

verification = json.loads(result.stdout.strip().splitlines()[-1])
expected_torch_prefix = USER_CONFIG["torch_version"]
if not verification["torch"].startswith(expected_torch_prefix):
    raise RuntimeError(
        f"PyTorch version mismatch. Expected {expected_torch_prefix}, received {verification['torch']}."
    )
if expected_diffusers_commit is not None and verification["diffusers_commit"] != expected_diffusers_commit:
    raise RuntimeError(
        "Diffusers commit mismatch.\n"
        f"Expected: {expected_diffusers_commit}\n"
        f"Received: {verification['diffusers_commit']}"
    )
if verification["arch"] != "krea2":
    raise RuntimeError(f"Unexpected Krea 2 architecture identifier: {verification['arch']}")
if verification["target_lora_modules"] != ["SingleStreamDiT"]:
    raise RuntimeError(f"Unexpected Krea 2 LoRA targets: {verification['target_lora_modules']}")
if verification["text_encoder_default"] != "Qwen/Qwen3-VL-4B-Instruct":
    raise RuntimeError(f"Unexpected Krea 2 text encoder default: {verification['text_encoder_default']}")
if verification["vae_default"] != "Qwen/Qwen-Image":
    raise RuntimeError(f"Unexpected Krea 2 VAE default: {verification['vae_default']}")
if verification["vae_scale_factor"] != 8:
    raise RuntimeError(f"Unexpected Krea 2 VAE scale factor: {verification['vae_scale_factor']}")
if not verification["has_save_conversion"] or not verification["has_load_conversion"]:
    raise RuntimeError("The installed Krea 2 implementation lacks required LoRA conversion hooks.")
if not verification["cuda_available"] or not verification["bf16_supported"]:
    raise RuntimeError("CUDA or BF16 is unavailable inside the isolated environment.")

verification["ai_toolkit_commit"] = source_environment_manifest["resolved_commit"]
verification_path = PATHS["config"] / "krea2_implementation_verification.json"
verification_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
print(json.dumps(verification, indent=2))
print(f"Verification manifest: {verification_path}")
