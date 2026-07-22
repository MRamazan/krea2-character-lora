from __future__ import annotations

import json
import shutil
import stat
import uuid
import zipfile
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path, PurePosixPath
from typing import Any

from .constants import (
    BUNDLE_FORMAT_VERSION,
    BUNDLE_MANIFEST_NAME,
    BUNDLE_TYPE,
    CUSTOM_VAE_REPOSITORY,
    FALLBACK_PACKAGE_VERSION,
    INFERENCE_MODEL_REPOSITORY,
    MAX_BUNDLE_FILES,
    MAX_BUNDLE_UNCOMPRESSED_BYTES,
    MAX_LORA_FILE_BYTES,
    PACKAGE_DISTRIBUTION,
    TEXT_ENCODER_REPOSITORY,
    TRAINING_MODEL_REPOSITORY,
)
from .errors import (
    BundleValidationError,
    ExportError,
    UnsupportedBundleVersionError,
)
from .hashing import sha256_file
from .manifests import read_json, write_json_atomic
from .paths import ProjectPaths
from .secrets import assert_no_secret_fields, find_secret_field
from .types import ExportBundle, ImportedRun

_REQUIRED_MANIFEST_FIELDS = (
    "bundle_type",
    "bundle_format_version",
    "run_name",
    "trigger_word",
    "selected_checkpoint_step",
    "capabilities",
    "files",
)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def package_version() -> str:
    try:
        return version(PACKAGE_DISTRIBUTION)
    except PackageNotFoundError:
        return FALLBACK_PACKAGE_VERSION


def source_git_revision() -> str | None:
    try:
        text = distribution(PACKAGE_DISTRIBUTION).read_text("direct_url.json")
    except (PackageNotFoundError, FileNotFoundError):
        return None
    if not text:
        return None
    try:
        info = json.loads(text)
    except json.JSONDecodeError:
        return None
    return info.get("vcs_info", {}).get("commit_id")


def _relpath_posix(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _scrub_workspace_paths(value: Any, root: str) -> Any:
    if isinstance(value, dict):
        return {key: _scrub_workspace_paths(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_workspace_paths(item, root) for item in value]
    if isinstance(value, str) and value.startswith(root):
        return PurePosixPath(Path(value).relative_to(root)).as_posix()
    return value


def _find_selected_record(
    checkpoints: list[dict[str, Any]], selection: dict[str, Any]
) -> dict[str, Any]:
    for record in checkpoints:
        if record["step"] == selection["checkpoint_step"]:
            return dict(record)
    return {
        "step": selection["checkpoint_step"],
        "rank": selection.get("rank"),
        "sha256": selection.get("checkpoint_sha256"),
        "is_final_name": False,
    }


def _bundle_checkpoint_path(step: int, selected_step: int) -> str:
    if step == selected_step:
        return "checkpoints/selected.safetensors"
    return f"checkpoints/step_{int(step):08d}.safetensors"


def _build_provenance(
    paths: ProjectPaths, run_manifest: dict[str, Any], selected_sha256: str
) -> dict[str, Any]:
    existing = run_manifest.get("provenance")
    if existing:
        provenance = deepcopy(existing)
        provenance["selected_lora_sha256"] = selected_sha256
        return provenance
    training = (
        read_json(paths.training_asset_manifest) if paths.training_asset_manifest.is_file() else {}
    )
    inference = (
        read_json(paths.inference_asset_manifest)
        if paths.inference_asset_manifest.is_file()
        else {}
    )
    vae = training.get("vae", {})
    return {
        "training_model": {
            "repository": TRAINING_MODEL_REPOSITORY,
            "revision": run_manifest.get("model_revision")
            or training.get("training_model", {}).get("revision"),
        },
        "evaluation_model": {
            "repository": INFERENCE_MODEL_REPOSITORY,
            "revision": inference.get("inference_model", {}).get("revision"),
        },
        "text_encoder": {
            "repository": TEXT_ENCODER_REPOSITORY,
            "revision": training.get("text_encoder", {}).get("revision"),
        },
        "vae": {
            "repository": CUSTOM_VAE_REPOSITORY,
            "revision": run_manifest.get("vae_revision") or vae.get("source_revision"),
            "source_filename": vae.get("source_filename"),
            "source_sha256": vae.get("source_sha256"),
        },
        "selected_lora_sha256": selected_sha256,
    }


def _slim_dataset_metadata(paths: ProjectPaths) -> dict[str, Any] | None:
    if not paths.dataset_manifest.is_file():
        return None
    manifest = read_json(paths.dataset_manifest)
    return {
        "trigger_word": manifest.get("trigger_word"),
        "pair_count": manifest.get("pair_count"),
        "accepted_image_extensions": manifest.get("accepted_image_extensions"),
        "dimensions": manifest.get("dimensions"),
        "dataset_fingerprint_sha256": manifest.get("dataset_fingerprint_sha256"),
    }


def _copy_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise BundleValidationError(f"A requested checkpoint file is missing: {source}")
    if source.stat().st_size > MAX_LORA_FILE_BYTES:
        raise BundleValidationError(
            f"The checkpoint exceeds the LoRA size limit and looks like a base model: {source} "
            f"({source.stat().st_size} bytes > {MAX_LORA_FILE_BYTES})."
        )
    shutil.copy2(source, destination)


def create_evaluation_bundle(
    workspace: str | Path,
    run_name: str,
    include_selected_lora: bool = True,
    include_all_checkpoints: bool = False,
    include_images: bool = True,
    include_logs: bool = True,
    include_manifests: bool = True,
) -> ExportBundle:
    paths = ProjectPaths.create(workspace)
    run_manifest_path = paths.run_manifest(run_name)
    if not run_manifest_path.is_file():
        raise ExportError(f"No run manifest was found for run {run_name}.")
    run_manifest = read_json(run_manifest_path)
    inventory = run_manifest.get("checkpoint_inventory", {})
    checkpoints = inventory.get("checkpoints", [])
    active_path = paths.active_checkpoint(run_name)
    selection = (
        read_json(active_path)
        if active_path.is_file()
        else run_manifest.get("active_checkpoint", {})
    )
    if not selection:
        raise ExportError(f"No active checkpoint selection was found for run {run_name}.")

    selected_step = int(selection["checkpoint_step"])
    selected_source = Path(selection["checkpoint_path"])
    if include_selected_lora and not selected_source.is_file():
        raise BundleValidationError(
            f"The selected LoRA checkpoint is missing and cannot be bundled: {selected_source}"
        )

    root = str(paths.root)
    staging = paths.exports / f"_staging_{run_name}_{uuid.uuid4().hex}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    unavailable: list[str] = []
    try:
        selected_sha256 = ""
        if include_selected_lora:
            destination = staging / "checkpoints" / "selected.safetensors"
            _copy_checkpoint(selected_source, destination)
            selected_sha256 = sha256_file(destination)

        bundle_checkpoints: list[dict[str, Any]] = []
        source_records = (
            checkpoints
            if include_all_checkpoints
            else [_find_selected_record(checkpoints, selection)]
        )
        for record in source_records:
            step = int(record["step"])
            archive_path = _bundle_checkpoint_path(step, selected_step)
            if step != selected_step:
                source = Path(record["path"])
                _copy_checkpoint(source, staging / archive_path)
            normalized_record = dict(record)
            normalized_record["path"] = archive_path
            normalized_record["filename"] = PurePosixPath(archive_path).name
            bundle_checkpoints.append(normalized_record)
        bundle_checkpoints.sort(key=lambda item: item["step"])
        available_steps = [item["step"] for item in bundle_checkpoints]

        bundle_inventory = deepcopy(inventory) if inventory else {}
        bundle_inventory["run_directory"] = "checkpoints"
        bundle_inventory["checkpoints"] = bundle_checkpoints
        bundle_inventory["checkpoint_steps"] = available_steps
        bundle_inventory["checkpoint_count"] = len(bundle_checkpoints)
        bundle_active = dict(selection)
        bundle_active["checkpoint_path"] = "checkpoints/selected.safetensors"

        previous_evaluation = None
        if include_manifests:
            manifests_dir = staging / "manifests"
            manifests_dir.mkdir(parents=True, exist_ok=True)
            normalized_run = _scrub_workspace_paths(deepcopy(run_manifest), root)
            normalized_run["checkpoint_inventory"] = bundle_inventory
            normalized_run["active_checkpoint"] = bundle_active
            write_json_atomic(manifests_dir / "run_manifest.json", normalized_run)
            write_json_atomic(
                manifests_dir / "training_config.json", run_manifest.get("training_config", {})
            )
            write_json_atomic(manifests_dir / "checkpoint_inventory.json", bundle_inventory)
            write_json_atomic(manifests_dir / "active_checkpoint.json", bundle_active)
            dataset_metadata = _slim_dataset_metadata(paths)
            if dataset_metadata is not None:
                write_json_atomic(manifests_dir / "dataset_metadata.json", dataset_metadata)
            if paths.dataset_fingerprint.is_file():
                write_json_atomic(
                    manifests_dir / "dataset_fingerprint.json",
                    read_json(paths.dataset_fingerprint),
                )
            if paths.evaluation_manifest(run_name).is_file():
                previous_evaluation = _scrub_workspace_paths(
                    read_json(paths.evaluation_manifest(run_name)), root
                )
                write_json_atomic(
                    manifests_dir / "previous_evaluation_manifest.json", previous_evaluation
                )
                write_json_atomic(
                    manifests_dir / "evaluation_config.json",
                    {
                        "prompts": previous_evaluation.get("prompts"),
                        "seeds": previous_evaluation.get("seeds"),
                        "inference_settings": previous_evaluation.get("inference_settings"),
                        "scale_sweep": previous_evaluation.get("scale_sweep"),
                        "checkpoint_mode": previous_evaluation.get("checkpoint_mode"),
                    },
                )
            yaml_source = paths.run_dir(run_name) / "train_krea2_lora.yaml"
            if yaml_source.is_file():
                config_dir = staging / "config"
                config_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(yaml_source, config_dir / "train_krea2_lora.yaml")
        else:
            unavailable.append("manifests")

        images_included = False
        if include_images:
            image_root = paths.inference / run_name
            if image_root.is_dir():
                for source in sorted(image_root.rglob("*")):
                    if source.is_file() and source.suffix.lower() in _IMAGE_SUFFIXES:
                        destination = staging / "images" / source.relative_to(image_root)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, destination)
                        images_included = True
            if not images_included:
                unavailable.append("evaluation_images")

        logs_included = False
        if include_logs and paths.logs.is_dir():
            for source in sorted(paths.logs.glob("*.log")):
                if source.is_file():
                    destination = staging / "logs" / source.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    logs_included = True
            if not logs_included:
                unavailable.append("logs")

        capabilities = {
            "selected_lora": include_selected_lora,
            "all_checkpoints": include_all_checkpoints and len(available_steps) > 1,
            "checkpoint_sweep": len(available_steps) > 1,
            "previous_evaluation": previous_evaluation is not None,
            "evaluation_images": images_included,
            "logs": logs_included,
        }
        training_config = run_manifest.get("training_config", {})
        lora_rank = selection.get("rank") or training_config.get("lora_rank")
        lora_alpha = training_config.get("lora_alpha", lora_rank)

        file_records = _scan_files(staging)
        _reject_secret_manifests(staging)

        bundle_manifest = {
            "bundle_type": BUNDLE_TYPE,
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "package_version": package_version(),
            "run_name": run_name,
            "trigger_word": run_manifest.get("trigger_word"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selected_checkpoint_step": selected_step,
            "available_checkpoint_steps": available_steps,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "source_git_revision": source_git_revision(),
            "provenance": _build_provenance(paths, run_manifest, selected_sha256),
            "capabilities": capabilities,
            "files": file_records,
        }
        secret_location = find_secret_field(bundle_manifest)
        if secret_location is not None:
            raise ExportError(f"A secret-like field '{secret_location}' was found in the bundle.")
        write_json_atomic(staging / BUNDLE_MANIFEST_NAME, bundle_manifest)

        export_directory = paths.exports / run_name
        export_directory.mkdir(parents=True, exist_ok=True)
        final_zip = export_directory / f"{run_name}_evaluation_bundle.zip"
        temporary_zip = export_directory / f".{run_name}_{uuid.uuid4().hex}.zip"
        _write_bundle_zip(staging, temporary_zip)
        temporary_zip.replace(final_zip)
        _validate_completed_bundle(final_zip)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return ExportBundle(
        archives=[final_zip],
        details={
            "bundle_path": str(final_zip),
            "export_directory": str(export_directory),
            "run_name": run_name,
            "capabilities": capabilities,
            "selected_checkpoint_step": selected_step,
            "available_checkpoint_steps": available_steps,
            "unavailable_categories": unavailable,
            "file_count": len(file_records),
            "packages": [
                {
                    "filename": final_zip.name,
                    "path": str(final_zip),
                    "size_bytes": final_zip.stat().st_size,
                    "sha256": sha256_file(final_zip),
                }
            ],
        },
    )


def _scan_files(staging: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sorted(staging.rglob("*")):
        if not source.is_file():
            continue
        relative = _relpath_posix(source, staging)
        if relative == BUNDLE_MANIFEST_NAME:
            continue
        if relative in seen:
            raise ExportError(f"Duplicate archive path detected: {relative}")
        seen.add(relative)
        records.append(
            {
                "path": relative,
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    records.sort(key=lambda item: item["path"])
    return records


def _reject_secret_manifests(staging: Path) -> None:
    for source in sorted(staging.rglob("*.json")):
        assert_no_secret_fields(read_json(source), _relpath_posix(source, staging))


def _write_bundle_zip(staging: Path, destination: Path) -> None:
    files = sorted(
        (path for path in staging.rglob("*") if path.is_file()),
        key=lambda path: _relpath_posix(path, staging),
    )
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as archive:
        for path in files:
            archive.write(path, arcname=_relpath_posix(path, staging))
    if not destination.is_file() or destination.stat().st_size == 0:
        raise ExportError(f"Bundle archive creation failed: {destination}")


def _validate_completed_bundle(bundle_zip: Path) -> None:
    with zipfile.ZipFile(bundle_zip) as archive:
        names = set(archive.namelist())
        if BUNDLE_MANIFEST_NAME not in names:
            raise ExportError("The completed bundle is missing its manifest.")
        manifest = json.loads(archive.read(BUNDLE_MANIFEST_NAME).decode("utf-8"))
        for record in manifest["files"]:
            if record["path"] not in names:
                raise ExportError(
                    f"The bundle manifest references a missing file: {record['path']}"
                )


def _entry_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return stat.S_ISLNK(mode)


def _entry_is_special(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    if mode == 0:
        return False
    return not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode))


def _validate_zip_entries(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
    if len(infos) > MAX_BUNDLE_FILES:
        raise BundleValidationError(
            f"The bundle declares too many entries: {len(infos)} > {MAX_BUNDLE_FILES}."
        )
    total = 0
    seen: set[str] = set()
    for info in infos:
        name = info.filename
        if name.endswith("/"):
            continue
        if _entry_is_symlink(info):
            raise BundleValidationError(f"The bundle contains a symlink entry: {name}")
        if _entry_is_special(info):
            raise BundleValidationError(f"The bundle contains an unsupported special file: {name}")
        if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
            raise BundleValidationError(f"The bundle contains an absolute path: {name}")
        parts = PurePosixPath(name).parts
        if ".." in parts:
            raise BundleValidationError(f"The bundle contains a parent traversal path: {name}")
        normalized = PurePosixPath(name).as_posix()
        if normalized in seen:
            raise BundleValidationError(f"The bundle contains a duplicate path: {normalized}")
        seen.add(normalized)
        total += info.file_size
        if total > MAX_BUNDLE_UNCOMPRESSED_BYTES:
            raise BundleValidationError(
                f"The bundle uncompressed size exceeds the limit of "
                f"{MAX_BUNDLE_UNCOMPRESSED_BYTES} bytes."
            )


def _require_manifest_fields(manifest: dict[str, Any]) -> None:
    missing = [field for field in _REQUIRED_MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise BundleValidationError(f"The bundle manifest is missing required fields: {missing}.")
    if manifest["bundle_type"] != BUNDLE_TYPE:
        raise BundleValidationError(
            f"Unexpected bundle type. Expected {BUNDLE_TYPE}, found {manifest['bundle_type']}."
        )
    format_version = manifest["bundle_format_version"]
    if not isinstance(format_version, int) or format_version < 1:
        raise BundleValidationError(f"Invalid bundle format version: {format_version}.")
    if format_version > BUNDLE_FORMAT_VERSION:
        raise UnsupportedBundleVersionError(
            f"The bundle format version {format_version} is newer than the supported version "
            f"{BUNDLE_FORMAT_VERSION}. Upgrade the package to import this bundle."
        )


def _verify_declared_files(extract_dir: Path, manifest: dict[str, Any]) -> set[str]:
    declared: set[str] = set()
    for record in manifest["files"]:
        relative = record["path"]
        if relative.startswith("/") or ".." in PurePosixPath(relative).parts:
            raise BundleValidationError(f"The manifest declares an unsafe path: {relative}")
        resolved = extract_dir / relative
        if not resolved.is_file():
            raise BundleValidationError(f"A declared bundle file is missing: {relative}")
        actual_size = resolved.stat().st_size
        if actual_size != record["size_bytes"]:
            raise BundleValidationError(
                f"Size mismatch for {relative}. "
                f"Expected {record['size_bytes']}, found {actual_size}."
            )
        actual_sha = sha256_file(resolved)
        if actual_sha != record["sha256"]:
            raise BundleValidationError(
                f"SHA-256 mismatch for {relative}. Expected {record['sha256']}, found {actual_sha}."
            )
        declared.add(relative)
    return declared


def _reject_undeclared_files(extract_dir: Path, declared: set[str]) -> None:
    for source in sorted(extract_dir.rglob("*")):
        if not source.is_file():
            continue
        relative = _relpath_posix(source, extract_dir)
        if relative == BUNDLE_MANIFEST_NAME:
            continue
        if relative not in declared:
            raise BundleValidationError(f"The bundle contains an undeclared file: {relative}")


def _rebase(path: str, extract_dir: Path) -> str:
    return str(extract_dir / PurePosixPath(path))


def import_evaluation_bundle(workspace: str | Path, zip_path: str | Path) -> ImportedRun:
    paths = ProjectPaths.create(workspace)
    archive_path = Path(zip_path)
    if not archive_path.is_file():
        raise BundleValidationError(f"The evaluation bundle does not exist: {archive_path}")
    if archive_path.suffix.lower() != ".zip":
        raise BundleValidationError("The evaluation bundle must be a single ZIP file.")
    _validate_zip_entries(archive_path)

    temporary = paths.runs / f"_import_tmp_{uuid.uuid4().hex}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temporary)
        manifest_path = temporary / BUNDLE_MANIFEST_NAME
        if not manifest_path.is_file():
            raise BundleValidationError("The bundle is missing its root manifest.")
        manifest = read_json(manifest_path)
        _require_manifest_fields(manifest)
        secret_location = find_secret_field(manifest)
        if secret_location is not None:
            raise BundleValidationError(
                f"The bundle manifest contains a secret-like field '{secret_location}'."
            )
        declared = _verify_declared_files(temporary, manifest)
        _reject_undeclared_files(temporary, declared)
        for source in sorted(temporary.rglob("*.json")):
            location = find_secret_field(read_json(source))
            if location is not None:
                raise BundleValidationError(
                    f"A bundled manifest contains a secret-like field '{location}' in "
                    f"{_relpath_posix(source, temporary)}."
                )
        active = read_json(temporary / "manifests" / "active_checkpoint.json")
        selected_relative = active["checkpoint_path"]
        selected_file = temporary / PurePosixPath(selected_relative)
        if not selected_file.is_file():
            raise BundleValidationError("The selected LoRA checkpoint is missing from the bundle.")
        if selected_file.stat().st_size > MAX_LORA_FILE_BYTES:
            raise BundleValidationError(
                "The selected checkpoint exceeds the LoRA size limit and looks like a base model."
            )
        _inspect_lora(selected_file)

        run_name = manifest["run_name"]
        final_dir = paths.run_dir(run_name) / "imported_bundle"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temporary), str(final_dir))
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    active = read_json(final_dir / "manifests" / "active_checkpoint.json")
    inventory = read_json(final_dir / "manifests" / "checkpoint_inventory.json")
    training_config_path = final_dir / "manifests" / "training_config.json"
    training_config = read_json(training_config_path) if training_config_path.is_file() else {}

    active["checkpoint_path"] = _rebase(active["checkpoint_path"], final_dir)
    inventory["run_directory"] = _rebase(inventory.get("run_directory", "checkpoints"), final_dir)
    for record in inventory.get("checkpoints", []):
        record["path"] = _rebase(record["path"], final_dir)

    reconstructed = {
        "schema_version": 1,
        "run_name": run_name,
        "trigger_word": manifest["trigger_word"],
        "training_config": training_config,
        "checkpoint_inventory": inventory,
        "active_checkpoint": active,
        "provenance": manifest.get("provenance", {}),
        "capabilities": manifest["capabilities"],
        "bundle_format_version": manifest["bundle_format_version"],
        "package_version": manifest.get("package_version"),
        "source_kind": inventory.get("source_kind", "imported"),
        "training_complete": inventory.get("training_complete", False),
        "lora_rank": manifest.get("lora_rank"),
        "lora_alpha": manifest.get("lora_alpha"),
        "imported": True,
        "bundle_source_path": str(archive_path),
        "extraction_directory": str(final_dir),
    }
    write_json_atomic(paths.active_checkpoint(run_name), active)
    write_json_atomic(paths.checkpoint_inventory(run_name), inventory)
    write_json_atomic(paths.run_manifest(run_name), reconstructed)

    return ImportedRun(
        workspace=paths.root,
        run_name=run_name,
        manifest_path=paths.run_manifest(run_name),
        trigger_word=manifest["trigger_word"],
        details=reconstructed,
        bundle_source_path=archive_path,
        extraction_directory=final_dir,
        capabilities=manifest["capabilities"],
        bundle_format_version=manifest["bundle_format_version"],
        provenance=manifest.get("provenance", {}),
    )


def _inspect_lora(path: Path) -> dict[str, Any]:
    from .assets import read_safetensors_header

    _, header = read_safetensors_header(path)
    if not header:
        raise BundleValidationError(f"The selected LoRA has no tensors: {path.name}")
    return {name: info.get("shape") for name, info in header.items()}
