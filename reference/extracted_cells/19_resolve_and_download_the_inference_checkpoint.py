import json
import os
import shutil
import subprocess
from pathlib import Path

if not USER_CONFIG["run_inference"]:
    print("Inference asset preparation skipped by configuration.")
else:
    training_assets = json.loads((PATHS["config"] / "training_asset_manifest.json").read_text(encoding="utf-8"))
    turbo_directory = PATHS["models"] / "krea_2_turbo"
    manifest_path = PATHS["config"] / "inference_asset_manifest.json"
    free_bytes = shutil.disk_usage(PROJECT_ROOT).free
    if free_bytes < 27 * 1024 ** 3 and not (turbo_directory / USER_CONFIG["inference_checkpoint_filename"]).is_file():
        raise RuntimeError(f"Insufficient free disk for Krea-2-Turbo. Available: {free_bytes / (1024 ** 3):.2f} GiB.")
    script = r"""
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
config = json.loads(Path(os.environ["KREA2_USER_CONFIG"]).read_text(encoding="utf-8"))
turbo_directory = Path(os.environ["KREA2_TURBO_DIRECTORY"])
token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
information = api.model_info(
    repo_id=config["inference_model_repository"],
    revision=config["inference_model_revision"] or "main",
    files_metadata=True,
)
revision = information.sha
path = Path(hf_hub_download(
    repo_id=config["inference_model_repository"],
    filename=config["inference_checkpoint_filename"],
    revision=revision,
    local_dir=str(turbo_directory),
    token=token,
))
digest = hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
        digest.update(block)
print(json.dumps({
    "repository": config["inference_model_repository"],
    "revision": revision,
    "checkpoint_path": str(path),
    "checkpoint_filename": path.name,
    "size_bytes": path.stat().st_size,
    "sha256": digest.hexdigest(),
}))
"""
    environment = os.environ.copy()
    environment["KREA2_USER_CONFIG"] = str(PATHS["config"] / "user_configuration.json")
    environment["KREA2_TURBO_DIRECTORY"] = str(turbo_directory)
    result = subprocess.run(
        [str(PATHS["venv_python"]), "-c", script],
        cwd=str(PATHS["ai_toolkit"]),
        env=environment,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Inference asset preparation failed.\n{result.stdout}\n{result.stderr}")
    turbo = json.loads(result.stdout.strip().splitlines()[-1])
    manifest = {
        "inference_model": turbo,
        "text_encoder": training_assets["text_encoder"],
        "vae": training_assets["vae"],
        "official_turbo_defaults": {"num_inference_steps": 8, "guidance_scale": 0.0},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))