from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

from .constants import (
    INFERENCE_CHECKPOINT_FILENAME,
    INFERENCE_MODEL_REPOSITORY,
    MINIMUM_INFERENCE_DISK_BYTES,
    MINIMUM_TRAINING_DISK_BYTES,
    TEXT_ENCODER_REPOSITORY,
    TRAINING_CHECKPOINT_FILENAME,
    TRAINING_MODEL_REPOSITORY,
    VAE_ARCHITECTURE,
    VAE_REPOSITORY,
    VAE_SUBFOLDER,
)
from .errors import AssetError, EnvironmentPreparationError, VaeValidationError
from .hashing import sha256_file
from .isolated import isolated_script
from .manifests import read_json, write_json_atomic
from .paths import ProjectPaths


def hf_model_info(repo_id: str, revision: str | None = None) -> Any:
    import huggingface_hub

    return huggingface_hub.HfApi().model_info(
        repo_id=repo_id, revision=revision or "main", token=None
    )


def hf_download_file(
    repo_id: str,
    filename: str,
    revision: str,
    local_dir: Path,
    subfolder: str | None = None,
) -> Path:
    import huggingface_hub

    return Path(
        huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            subfolder=subfolder,
            local_dir=str(local_dir),
            token=None,
        )
    )


def hf_snapshot(
    repo_id: str, revision: str, local_dir: Path, allow_patterns: list[str] | None = None
) -> Path:
    import huggingface_hub

    return Path(
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
            token=None,
        )
    )


def resolve_revision(repo_id: str, revision: str | None = None) -> str:
    information = hf_model_info(repo_id, revision)
    resolved = getattr(information, "sha", None)
    if not resolved:
        raise AssetError(f"Unable to resolve a commit for repository {repo_id}.")
    return resolved


def read_safetensors_header(path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    with Path(path).open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise AssetError(f"The safetensors file is truncated: {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise AssetError(f"The safetensors header is truncated: {path}")
    header = json.loads(header_bytes.decode("utf-8"))
    metadata = header.pop("__metadata__", {})
    return metadata, header


def prepare_vae(paths: ProjectPaths) -> dict[str, Any]:
    download_root = paths.models / "krea2_vae"
    if download_root.exists():
        shutil.rmtree(download_root)
    download_root.mkdir(parents=True, exist_ok=True)
    revision = resolve_revision(VAE_REPOSITORY)
    hf_snapshot(VAE_REPOSITORY, revision, download_root, allow_patterns=[f"{VAE_SUBFOLDER}/*"])
    vae_directory = download_root / VAE_SUBFOLDER
    config = vae_directory / "config.json"
    weights = next(vae_directory.glob("*.safetensors"), None)
    if not config.is_file() or weights is None:
        raise AssetError(
            f"The Krea 2 default VAE repository {VAE_REPOSITORY} did not provide a "
            f"{VAE_SUBFOLDER}/config.json and safetensors file."
        )
    manifest = {
        "architecture": VAE_ARCHITECTURE,
        "source": "krea2_default",
        "repository": VAE_REPOSITORY,
        "revision": revision,
        "subfolder": VAE_SUBFOLDER,
        "weights_filename": weights.name,
        "normalized_directory": str(vae_directory),
        "weights_sha256": sha256_file(weights),
        "strict_validation": "pending",
    }
    write_json_atomic(vae_directory / "vae_manifest.json", manifest)
    return manifest


def validate_vae(paths: ProjectPaths, vae_manifest: dict[str, Any]) -> dict[str, Any]:
    if not paths.venv_python.is_file():
        raise EnvironmentPreparationError(
            "The isolated environment is required to validate the Krea 2 default VAE."
        )
    normalized_directory = Path(vae_manifest["normalized_directory"])
    result_path = normalized_directory / "vae_smoke_result.json"
    environment = os.environ.copy()
    environment["KREA2_VAE_DIRECTORY"] = str(normalized_directory)
    environment["KREA2_VAE_RESULT"] = str(result_path)
    environment["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [str(paths.venv_python), str(isolated_script("validate_vae.py"))],
        cwd=str(paths.ai_toolkit),
        env=environment,
        capture_output=True,
        text=True,
    )
    log_path = paths.logs / "vae_validation.log"
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise VaeValidationError(
            f"The Krea 2 default VAE failed validation with {VAE_ARCHITECTURE}. "
            f"Repository: {vae_manifest['repository']}. Validation log: {log_path}"
        )
    smoke = json.loads(result.stdout.strip().splitlines()[-1])
    vae_manifest["strict_validation"] = "passed"
    vae_manifest["smoke_test"] = smoke
    write_json_atomic(normalized_directory / "vae_manifest.json", vae_manifest)
    return vae_manifest


def prepare_training_assets(paths: ProjectPaths, strict_vae: bool = True) -> dict[str, Any]:
    models = paths.models
    models.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(paths.root).free
    if free_bytes < MINIMUM_TRAINING_DISK_BYTES:
        raise AssetError(
            f"At least {MINIMUM_TRAINING_DISK_BYTES / 1024**3:.0f} GiB of free disk is required "
            f"before downloading training assets. Available: {free_bytes / 1024**3:.2f} GiB."
        )
    raw_directory = models / "krea_2_raw"
    text_directory = models / "qwen3_vl_text_encoder"
    raw_revision = resolve_revision(TRAINING_MODEL_REPOSITORY)
    raw_path = hf_download_file(
        TRAINING_MODEL_REPOSITORY, TRAINING_CHECKPOINT_FILENAME, raw_revision, raw_directory
    )
    text_revision = resolve_revision(TEXT_ENCODER_REPOSITORY)
    hf_snapshot(TEXT_ENCODER_REPOSITORY, text_revision, text_directory)
    vae_manifest = prepare_vae(paths)
    if strict_vae:
        vae_manifest = validate_vae(paths, vae_manifest)
    manifest = {
        "training_model": {
            "repository": TRAINING_MODEL_REPOSITORY,
            "revision": raw_revision,
            "checkpoint_path": str(raw_path),
            "checkpoint_filename": raw_path.name,
            "size_bytes": raw_path.stat().st_size,
            "sha256": sha256_file(raw_path),
        },
        "text_encoder": {
            "repository": TEXT_ENCODER_REPOSITORY,
            "revision": text_revision,
            "local_directory": str(text_directory),
        },
        "vae": vae_manifest,
    }
    write_json_atomic(paths.training_asset_manifest, manifest)
    return manifest


def prepare_inference_assets(paths: ProjectPaths) -> dict[str, Any]:
    if not paths.training_asset_manifest.is_file():
        raise AssetError(
            "Training assets must be prepared before inference assets so the VAE and "
            "text encoder can be reused."
        )
    training = read_json(paths.training_asset_manifest)
    turbo_directory = paths.models / "krea_2_turbo"
    free_bytes = shutil.disk_usage(paths.root).free
    turbo_present = (turbo_directory / INFERENCE_CHECKPOINT_FILENAME).is_file()
    if free_bytes < MINIMUM_INFERENCE_DISK_BYTES and not turbo_present:
        raise AssetError(
            f"Insufficient free disk for Krea 2 Turbo. Available: {free_bytes / 1024**3:.2f} GiB."
        )
    turbo_revision = resolve_revision(INFERENCE_MODEL_REPOSITORY)
    turbo_path = hf_download_file(
        INFERENCE_MODEL_REPOSITORY, INFERENCE_CHECKPOINT_FILENAME, turbo_revision, turbo_directory
    )
    manifest = {
        "inference_model": {
            "repository": INFERENCE_MODEL_REPOSITORY,
            "revision": turbo_revision,
            "checkpoint_path": str(turbo_path),
            "checkpoint_filename": turbo_path.name,
            "size_bytes": turbo_path.stat().st_size,
            "sha256": sha256_file(turbo_path),
        },
        "text_encoder": training["text_encoder"],
        "vae": training["vae"],
        "official_turbo_defaults": {"num_inference_steps": 8, "guidance_scale": 0.0},
    }
    write_json_atomic(paths.inference_asset_manifest, manifest)
    return manifest
