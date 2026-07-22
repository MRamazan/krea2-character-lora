from __future__ import annotations

import math
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ExportError
from .hashing import sha256_file
from .manifests import read_json, write_json_atomic
from .paths import ProjectPaths
from .types import ExportBundle

_CHECKPOINTS_PER_ARCHIVE = 4
_SECRET_TOKENS = ("token", "authorization", "api_key", "secret", "password")


def _write_zip(path: Path, entries: list[tuple[Path, Path]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for source, archive_name in entries:
            archive.write(source, arcname=str(archive_name))
    if not path.is_file() or path.stat().st_size == 0:
        raise ExportError(f"Archive creation failed: {path}")


def _assert_no_secret_fields(payload: Any, source: Path) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if any(token in str(key).lower() for token in _SECRET_TOKENS):
                raise ExportError(f"A secret-like field '{key}' was found in {source}.")
            _assert_no_secret_fields(value, source)
    elif isinstance(payload, list):
        for item in payload:
            _assert_no_secret_fields(item, source)


def _package_record(path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def package_run(
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
    active_path = paths.active_checkpoint(run_name)
    selection = (
        read_json(active_path)
        if active_path.is_file()
        else run_manifest.get("active_checkpoint", {})
    )
    if not selection:
        raise ExportError(f"No active checkpoint selection was found for run {run_name}.")

    export_directory = paths.exports / run_name
    if export_directory.exists():
        shutil.rmtree(export_directory)
    export_directory.mkdir(parents=True, exist_ok=False)

    packages: list[Path] = []

    if include_manifests:
        manifest_sources = [
            run_manifest_path,
            paths.checkpoint_inventory(run_name),
            active_path,
            paths.dataset_manifest,
            paths.dataset_fingerprint,
            paths.training_asset_manifest,
            paths.inference_asset_manifest,
            paths.environment_manifest,
            paths.evaluation_manifest(run_name),
        ]
        manifest_entries: list[tuple[Path, Path]] = []
        for source in manifest_sources:
            if source.is_file():
                _assert_no_secret_fields(read_json(source), source)
                manifest_entries.append((source, Path("manifests") / source.name))
        config_directory = paths.run_dir(run_name)
        for config_file in sorted(config_directory.glob("*.yaml")):
            manifest_entries.append((config_file, Path("config") / config_file.name))
        if manifest_entries:
            archive = export_directory / f"{run_name}_manifests.zip"
            _write_zip(archive, manifest_entries)
            packages.append(archive)

    if include_selected_lora:
        selected_source = Path(selection["checkpoint_path"])
        if not selected_source.is_file():
            raise ExportError(f"The selected checkpoint file is missing: {selected_source}")
        selected_copy = (
            export_directory
            / f"{run_name}_selected_step_{int(selection['checkpoint_step']):08d}.safetensors"
        )
        shutil.copy2(selected_source, selected_copy)
        if sha256_file(selected_source) != sha256_file(selected_copy):
            raise ExportError("The selected checkpoint copy does not match its source.")
        packages.append(selected_copy)

    if include_all_checkpoints:
        checkpoint_records = inventory.get("checkpoints", [])
        for group_index in range(math.ceil(len(checkpoint_records) / _CHECKPOINTS_PER_ARCHIVE)):
            group = checkpoint_records[
                group_index * _CHECKPOINTS_PER_ARCHIVE : (group_index + 1)
                * _CHECKPOINTS_PER_ARCHIVE
            ]
            entries = [
                (Path(record["path"]), Path("checkpoints") / Path(record["path"]).name)
                for record in group
                if Path(record["path"]).is_file()
            ]
            if entries:
                archive = export_directory / f"{run_name}_checkpoints_{group_index + 1:02d}.zip"
                _write_zip(archive, entries)
                packages.append(archive)

    if include_images:
        image_root = paths.inference / run_name
        if image_root.is_dir():
            image_entries = [
                (path, Path("images") / path.relative_to(image_root))
                for path in sorted(image_root.rglob("*"))
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
            if image_entries:
                archive = export_directory / f"{run_name}_images.zip"
                _write_zip(archive, image_entries)
                packages.append(archive)

    if include_logs and paths.logs.is_dir():
        log_entries = [
            (path, Path("logs") / path.relative_to(paths.logs))
            for path in sorted(paths.logs.rglob("*"))
            if path.is_file()
        ]
        if log_entries:
            archive = export_directory / f"{run_name}_logs.zip"
            _write_zip(archive, log_entries)
            packages.append(archive)

    resume_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "project_name": run_manifest.get("project_name"),
        "trigger_word": run_manifest.get("trigger_word"),
        "source_kind": inventory.get("source_kind"),
        "training_complete": inventory.get("training_complete"),
        "final_quality_claim_allowed": inventory.get("final_quality_claim_allowed"),
        "checkpoint_steps": inventory.get("checkpoint_steps"),
        "model_revision": run_manifest.get("model_revision"),
        "vae_revision": run_manifest.get("vae_revision"),
        "source_revision": run_manifest.get("source_revision"),
        "selected_checkpoint": {
            "step": selection["checkpoint_step"],
            "source_path": selection["checkpoint_path"],
            "sha256": selection["checkpoint_sha256"],
        },
        "excluded_reproducible_assets": [
            str(paths.models),
            str(paths.venv),
            str(paths.ai_toolkit),
        ],
        "packages": [_package_record(path) for path in packages],
    }
    manifest_path = export_directory / "resume_manifest.json"
    write_json_atomic(manifest_path, resume_manifest)
    packages.insert(0, manifest_path)
    resume_manifest["packages"] = [_package_record(path) for path in packages]
    write_json_atomic(manifest_path, resume_manifest)

    return ExportBundle(
        archives=packages,
        details={
            "export_directory": str(export_directory),
            "packages": resume_manifest["packages"],
            "run_name": run_name,
        },
    )
