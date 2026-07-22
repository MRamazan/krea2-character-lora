from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

from .constants import (
    CUSTOM_VAE_REPOSITORY,
    INFERENCE_CHECKPOINT_FILENAME,
    INFERENCE_MODEL_REPOSITORY,
    TEXT_ENCODER_REPOSITORY,
    TRAINING_CHECKPOINT_FILENAME,
    TRAINING_MODEL_REPOSITORY,
    VAE_ARCHITECTURE,
    VAE_CONFIG_REPOSITORY,
    VAE_CONFIG_SUBFOLDER,
    VAE_CONVERSION_IDENTITY,
    VAE_CONVERSION_ORIGINAL_TO_DIFFUSERS,
    VAE_FORMAT_DIFFUSERS,
    VAE_FORMAT_ORIGINAL,
    VAE_FORMAT_UNKNOWN,
    VAE_STATE_DICT_PREFIX,
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


def hf_list_repo_files(repo_id: str, revision: str | None = None) -> list[str]:
    import huggingface_hub

    return list(
        huggingface_hub.list_repo_files(repo_id=repo_id, revision=revision or "main", token=None)
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


def state_dict_shapes(path: str | Path) -> dict[str, list[int]]:
    _, header = read_safetensors_header(path)
    return {name: list(info["shape"]) for name, info in header.items()}


def select_vae_safetensors(file_names: list[str]) -> str:
    candidates = sorted(name for name in file_names if name.lower().endswith(".safetensors"))
    if not candidates:
        raise AssetError("No safetensors file was found in the custom VAE repository.")
    for name in candidates:
        if Path(name).name == "diffusion_pytorch_model.safetensors":
            return name
    vae_named = [name for name in candidates if "vae" in Path(name).name.lower()]
    pool = vae_named or candidates
    return sorted(pool, key=lambda name: (len(name), name))[0]


def detect_removable_prefix(
    keys: list[str], reference_keys: list[str], prefix: str = VAE_STATE_DICT_PREFIX
) -> bool:
    key_set = set(keys)
    return (
        bool(key_set)
        and all(key.startswith(prefix) for key in key_set)
        and not any(reference.startswith(prefix) for reference in reference_keys)
    )


def normalize_keys(
    shapes: dict[str, list[int]], remove_prefix: bool, prefix: str
) -> dict[str, list[int]]:
    if not remove_prefix:
        return dict(shapes)
    return {
        name[len(prefix) :] if name.startswith(prefix) else name: shape
        for name, shape in shapes.items()
    }


def analyze_state_dict(
    custom_shapes: dict[str, list[int]], reference_shapes: dict[str, list[int]]
) -> dict[str, list[str]]:
    custom_keys = set(custom_shapes)
    reference_keys = set(reference_shapes)
    missing = sorted(reference_keys - custom_keys)
    unexpected = sorted(custom_keys - reference_keys)
    shape_mismatches = sorted(
        f"{key}: {custom_shapes[key]} != {reference_shapes[key]}"
        for key in (custom_keys & reference_keys)
        if custom_shapes[key] != reference_shapes[key]
    )
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
    }


_MID_BLOCK_MAP = {"0": "resnets.0", "1": "attentions.0", "2": "resnets.1"}


def convert_original_vae_key(key: str) -> str:
    k = key
    if k.startswith("conv1."):
        return "quant_conv." + k[len("conv1.") :]
    if k.startswith("conv2."):
        return "post_quant_conv." + k[len("conv2.") :]
    k = re.sub(r"^encoder\.conv1\.", "encoder.conv_in.", k)
    k = re.sub(r"^decoder\.conv1\.", "decoder.conv_in.", k)
    k = k.replace("encoder.downsamples.", "encoder.down_blocks.")
    match = re.match(r"decoder\.upsamples\.(\d+)\.(.*)", k)
    if match is not None:
        index = int(match.group(1))
        rest = match.group(2)
        block, position = divmod(index, 4)
        if position < 3:
            k = f"decoder.up_blocks.{block}.resnets.{position}.{rest}"
        else:
            k = f"decoder.up_blocks.{block}.upsamplers.0.{rest}"
    k = re.sub(
        r"(encoder|decoder)\.middle\.([012])\.",
        lambda mo: f"{mo.group(1)}.mid_block.{_MID_BLOCK_MAP[mo.group(2)]}.",
        k,
    )
    k = k.replace(".residual.0.", ".norm1.").replace(".residual.2.", ".conv1.")
    k = k.replace(".residual.3.", ".norm2.").replace(".residual.6.", ".conv2.")
    k = k.replace(".shortcut.", ".conv_shortcut.")
    k = re.sub(r"\.head\.0\.", ".norm_out.", k)
    k = re.sub(r"\.head\.2\.", ".conv_out.", k)
    return k


def detect_vae_format(keys: list[str]) -> str:
    key_set = set(keys)
    has_original = (
        {"conv1.weight", "conv2.weight"} <= key_set
        and any(name.startswith("encoder.downsamples.") for name in key_set)
        and any(name.startswith("decoder.upsamples.") for name in key_set)
    )
    if has_original:
        return VAE_FORMAT_ORIGINAL
    has_diffusers = bool({"quant_conv.weight", "post_quant_conv.weight"} & key_set) or any(
        name.startswith("encoder.down_blocks.") for name in key_set
    )
    if has_diffusers:
        return VAE_FORMAT_DIFFUSERS
    return VAE_FORMAT_UNKNOWN


def plan_vae_normalization(
    custom_shapes: dict[str, list[int]],
    reference_shapes: dict[str, list[int]],
    prefix: str = VAE_STATE_DICT_PREFIX,
) -> dict[str, Any]:
    remove_prefix = detect_removable_prefix(list(custom_shapes), list(reference_shapes), prefix)
    stripped = normalize_keys(custom_shapes, remove_prefix, prefix)
    detected_format = detect_vae_format(list(stripped))
    if detected_format == VAE_FORMAT_ORIGINAL:
        conversion = VAE_CONVERSION_ORIGINAL_TO_DIFFUSERS
        key_mapping = {name: convert_original_vae_key(name) for name in stripped}
    else:
        conversion = VAE_CONVERSION_IDENTITY
        key_mapping = {name: name for name in stripped}
    converted = {key_mapping[name]: shape for name, shape in stripped.items()}
    analysis = analyze_state_dict(converted, reference_shapes)
    compatible = (
        detected_format != VAE_FORMAT_UNKNOWN
        and len(converted) == len(stripped)
        and not analysis["missing_keys"]
        and not analysis["unexpected_keys"]
        and not analysis["shape_mismatches"]
    )
    return {
        "remove_prefix": remove_prefix,
        "prefix": prefix,
        "detected_format": detected_format,
        "conversion": conversion,
        "stripped_key_count": len(stripped),
        "converted_key_count": len(converted),
        "key_mapping": key_mapping,
        "analysis": analysis,
        "compatible": compatible,
    }


def _write_validation_log(paths: ProjectPaths, name: str, payload: dict[str, Any]) -> Path:
    log_path = paths.logs / name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return log_path


def prepare_custom_vae(paths: ProjectPaths) -> dict[str, Any]:
    vae_source_directory = paths.models / "custom_vae_source"
    reference_directory = paths.models / "reference_vae"
    normalized_directory = paths.models / "krea2_normalized_vae"
    vae_source_directory.mkdir(parents=True, exist_ok=True)
    reference_directory.mkdir(parents=True, exist_ok=True)
    if normalized_directory.exists():
        shutil.rmtree(normalized_directory)
    normalized_directory.mkdir(parents=True, exist_ok=False)

    custom_revision = resolve_revision(CUSTOM_VAE_REPOSITORY)
    file_names = hf_list_repo_files(CUSTOM_VAE_REPOSITORY, custom_revision)
    selected_name = select_vae_safetensors(file_names)
    custom_path = hf_download_file(
        CUSTOM_VAE_REPOSITORY, selected_name, custom_revision, vae_source_directory
    )

    config_revision = resolve_revision(VAE_CONFIG_REPOSITORY)
    hf_snapshot(
        VAE_CONFIG_REPOSITORY,
        config_revision,
        reference_directory,
        allow_patterns=[f"{VAE_CONFIG_SUBFOLDER}/*"],
    )
    reference_vae_directory = reference_directory / VAE_CONFIG_SUBFOLDER
    reference_config = reference_vae_directory / "config.json"
    reference_weights = next(reference_vae_directory.glob("*.safetensors"), None)
    if not reference_config.is_file() or reference_weights is None:
        raise AssetError(
            "The reference Qwen Image VAE repository did not provide a config and safetensors file."
        )

    custom_shapes = state_dict_shapes(custom_path)
    reference_shapes = state_dict_shapes(reference_weights)
    plan = plan_vae_normalization(custom_shapes, reference_shapes)
    if not plan["compatible"]:
        log_path = _write_validation_log(
            paths,
            "custom_vae_validation.json",
            {
                "custom_repository": CUSTOM_VAE_REPOSITORY,
                "custom_revision": custom_revision,
                "source_filename": selected_name,
                "plan": plan,
            },
        )
        raise VaeValidationError(
            f"The custom VAE is incompatible with {VAE_ARCHITECTURE}. "
            f"Discovered file: {selected_name}. Detected format: {plan['detected_format']}. "
            f"Missing keys: {plan['analysis']['missing_keys'][:5]}. "
            f"Unexpected keys: {plan['analysis']['unexpected_keys'][:5]}. "
            f"Shape mismatches: {plan['analysis']['shape_mismatches'][:5]}. "
            f"Validation log: {log_path}"
        )

    source_to_diffusers = _build_source_mapping(custom_shapes, plan)
    key_mapping_path = normalized_directory / "vae_key_mapping.json"
    write_json_atomic(
        key_mapping_path, {"conversion": plan["conversion"], "mapping": source_to_diffusers}
    )
    shutil.copy2(reference_config, normalized_directory / "config.json")
    manifest = {
        "architecture": VAE_ARCHITECTURE,
        "source_repository": CUSTOM_VAE_REPOSITORY,
        "source_revision": custom_revision,
        "source_filename": selected_name,
        "source_path": str(custom_path),
        "source_sha256": sha256_file(custom_path),
        "config_source_repository": VAE_CONFIG_REPOSITORY,
        "config_source_revision": config_revision,
        "config_source_subfolder": VAE_CONFIG_SUBFOLDER,
        "normalized_directory": str(normalized_directory),
        "detected_format": plan["detected_format"],
        "conversion": plan["conversion"],
        "remove_prefix": plan["remove_prefix"],
        "prefix": plan["prefix"],
        "source_tensor_count": len(custom_shapes),
        "converted_key_count": plan["converted_key_count"],
        "key_mapping_path": str(key_mapping_path),
        "key_analysis": plan["analysis"],
        "static_compatibility": plan["compatible"],
        "strict_validation": "pending",
    }
    write_json_atomic(normalized_directory / "vae_normalization_manifest.json", manifest)
    return manifest


def _build_source_mapping(
    custom_shapes: dict[str, list[int]], plan: dict[str, Any]
) -> dict[str, str]:
    prefix = plan["prefix"]
    remove_prefix = plan["remove_prefix"]
    stripped_mapping = plan["key_mapping"]
    mapping: dict[str, str] = {}
    for raw_key in custom_shapes:
        stripped = (
            raw_key[len(prefix) :] if remove_prefix and raw_key.startswith(prefix) else raw_key
        )
        mapping[raw_key] = stripped_mapping[stripped]
    return mapping


def strict_validate_vae(paths: ProjectPaths, vae_manifest: dict[str, Any]) -> dict[str, Any]:
    if not paths.venv_python.is_file():
        raise EnvironmentPreparationError(
            "The isolated environment is required to strictly validate the custom VAE."
        )
    normalized_directory = Path(vae_manifest["normalized_directory"])
    result_path = normalized_directory / "vae_smoke_result.json"
    environment = os.environ.copy()
    environment["KREA2_VAE_SOURCE"] = vae_manifest["source_path"]
    environment["KREA2_VAE_DIRECTORY"] = str(normalized_directory)
    environment["KREA2_VAE_RESULT"] = str(result_path)
    environment["KREA2_VAE_KEY_MAPPING"] = vae_manifest["key_mapping_path"]
    environment["KREA2_VAE_CONVERSION"] = vae_manifest["conversion"]
    environment["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [str(paths.venv_python), str(isolated_script("validate_vae.py"))],
        cwd=str(paths.ai_toolkit),
        env=environment,
        capture_output=True,
        text=True,
    )
    log_path = paths.logs / "custom_vae_strict_validation.log"
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise VaeValidationError(
            "The custom VAE failed strict validation with "
            f"{VAE_ARCHITECTURE}. Source file: {vae_manifest['source_filename']}. "
            f"Validation log: {log_path}"
        )
    smoke = json.loads(result.stdout.strip().splitlines()[-1])
    vae_manifest["strict_validation"] = "passed"
    vae_manifest["smoke_test"] = smoke
    vae_manifest["normalized_weights_sha256"] = sha256_file(
        normalized_directory / "diffusion_pytorch_model.safetensors"
    )
    write_json_atomic(normalized_directory / "vae_normalization_manifest.json", vae_manifest)
    return vae_manifest


def prepare_training_assets(paths: ProjectPaths, strict_vae: bool = True) -> dict[str, Any]:
    models = paths.models
    models.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(paths.root).free
    from .constants import MINIMUM_TRAINING_DISK_BYTES

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
    vae_manifest = prepare_custom_vae(paths)
    if strict_vae:
        vae_manifest = strict_validate_vae(paths, vae_manifest)
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
    from .constants import MINIMUM_INFERENCE_DISK_BYTES

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
