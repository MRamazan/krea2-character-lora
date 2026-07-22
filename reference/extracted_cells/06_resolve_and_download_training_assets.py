import json
import os
import shutil
import subprocess
from pathlib import Path

asset_manifest_path = PATHS["config"] / "training_asset_manifest.json"
minimum_free_bytes = 42 * 1024 ** 3
free_bytes = shutil.disk_usage(PROJECT_ROOT).free
if free_bytes < minimum_free_bytes:
    raise RuntimeError(
        f"At least {minimum_free_bytes / (1024 ** 3):.0f} GiB of free disk is required before downloading training assets. "
        f"Available: {free_bytes / (1024 ** 3):.2f} GiB."
    )

raw_directory = PATHS["models"] / "krea_2_raw"
text_directory = PATHS["models"] / "qwen3_vl_text_encoder"
vae_directory = PATHS["models"] / "qwen_image_vae"

script = r"""
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

config = json.loads(Path(os.environ["KREA2_USER_CONFIG"]).read_text(encoding="utf-8"))
raw_directory = Path(os.environ["KREA2_RAW_DIRECTORY"])
text_directory = Path(os.environ["KREA2_TEXT_DIRECTORY"])
vae_directory = Path(os.environ["KREA2_VAE_DIRECTORY"])
token = os.environ["HF_TOKEN"]
api = HfApi(token=token)

def resolve(repo_id, revision):
    information = api.model_info(repo_id=repo_id, revision=revision or "main", files_metadata=True)
    return information.sha

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

raw_revision = resolve(config["training_model_repository"], config["training_model_revision"])
text_revision = resolve(config["text_encoder_repository"], config["text_encoder_revision"])
vae_revision = resolve(config["vae_repository"], config["vae_revision"])

raw_path = Path(hf_hub_download(
    repo_id=config["training_model_repository"],
    filename=config["training_checkpoint_filename"],
    revision=raw_revision,
    local_dir=str(raw_directory),
    token=token,
))

snapshot_download(
    repo_id=config["text_encoder_repository"],
    revision=text_revision,
    local_dir=str(text_directory),
    token=token,
)

snapshot_download(
    repo_id=config["vae_repository"],
    revision=vae_revision,
    local_dir=str(vae_directory),
    allow_patterns=["vae/*"],
    token=token,
)

result = {
    "training_model": {
        "repository": config["training_model_repository"],
        "revision": raw_revision,
        "checkpoint_path": str(raw_path),
        "checkpoint_filename": raw_path.name,
        "size_bytes": raw_path.stat().st_size,
        "sha256": sha256_file(raw_path),
    },
    "text_encoder": {
        "repository": config["text_encoder_repository"],
        "revision": text_revision,
        "local_directory": str(text_directory),
    },
    "vae": {
        "repository": config["vae_repository"],
        "revision": vae_revision,
        "local_directory": str(vae_directory),
        "subfolder": "vae",
    },
}
print(json.dumps(result))
"""

environment = os.environ.copy()
environment["KREA2_USER_CONFIG"] = str(PATHS["config"] / "user_configuration.json")
environment["KREA2_RAW_DIRECTORY"] = str(raw_directory)
environment["KREA2_TEXT_DIRECTORY"] = str(text_directory)
environment["KREA2_VAE_DIRECTORY"] = str(vae_directory)
environment["PYTHONUNBUFFERED"] = "1"
result = subprocess.run(
    [str(PATHS["venv_python"]), "-c", script],
    cwd=str(PATHS["ai_toolkit"]),
    env=environment,
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    raise RuntimeError(f"Training asset download failed.\n{result.stdout}\n{result.stderr}")
manifest = json.loads(result.stdout.strip().splitlines()[-1])
asset_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
print(f"Training asset manifest: {asset_manifest_path}")