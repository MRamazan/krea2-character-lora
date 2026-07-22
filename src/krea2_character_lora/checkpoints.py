from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from .constants import PRIMARY_ADAPTER_NAME
from .errors import CheckpointError
from .hashing import sha256_file
from .manifests import read_json, write_json_atomic
from .paths import ProjectPaths


def discover_checkpoint_files(run_directory: Path, run_name: str) -> list[tuple[int, Path, bool]]:
    if not run_directory.is_dir():
        return []
    pattern = re.compile(re.escape(run_name) + r"_(\d+)")
    discovered: list[tuple[int, Path, bool]] = []
    for path in sorted(run_directory.glob("*.safetensors")):
        stem = path.stem
        if stem == run_name:
            discovered.append((-1, path, True))
            continue
        match = pattern.fullmatch(stem)
        if match is not None:
            discovered.append((int(match.group(1)), path, False))
    return discovered


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    tensor_count = 0
    parameter_count = 0
    nonfinite_count = 0
    ranks: set[int] = set()
    keys: list[str] = []
    shapes: dict[str, list[int]] = {}
    with safe_open(str(path), framework="numpy") as checkpoint:
        metadata = checkpoint.metadata() or {}
        keys = list(checkpoint.keys())
        for key in keys:
            tensor = checkpoint.get_tensor(key)
            shapes[key] = list(tensor.shape)
            tensor_count += 1
            parameter_count += int(tensor.size)
            if np.issubdtype(tensor.dtype, np.floating):
                nonfinite_count += int(np.count_nonzero(~np.isfinite(tensor)))
            if key.endswith(".lora_A.weight") and tensor.ndim == 2:
                ranks.add(int(tensor.shape[0]))
    if nonfinite_count != 0:
        raise CheckpointError(f"The checkpoint contains non-finite parameters: {path}")
    if len(ranks) != 1:
        raise CheckpointError(f"Unable to infer a single LoRA rank from {path}: {sorted(ranks)}")
    return {
        "path": str(path),
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "nonfinite_parameter_count": nonfinite_count,
        "rank": next(iter(ranks)),
        "keys": keys,
        "shapes": shapes,
        "metadata": metadata,
    }


def inventory_from_directory(
    run_directory: Path,
    run_name: str,
    configured_steps: int,
    source_kind: str,
    process_status: dict[str, Any],
) -> dict[str, Any]:
    discovered = discover_checkpoint_files(run_directory, run_name)
    if not discovered:
        raise CheckpointError(f"No LoRA checkpoints were found in {run_directory}.")
    records: list[dict[str, Any]] = []
    reference_keys: list[str] | None = None
    reference_shapes: dict[str, list[int]] | None = None
    for step, path, is_final in discovered:
        record = inspect_checkpoint(path)
        record["is_final_name"] = is_final
        record["step"] = configured_steps if is_final else step
        if reference_keys is None:
            reference_keys = record["keys"]
            reference_shapes = record["shapes"]
        elif record["keys"] != reference_keys or record["shapes"] != reference_shapes:
            raise CheckpointError(
                f"The checkpoint schema differs from the first checkpoint: {path}"
            )
        records.append(record)
    for record in records:
        record.pop("keys", None)
        record.pop("shapes", None)
    records.sort(key=lambda item: (item["step"], item["is_final_name"]))
    final_records = [record for record in records if record["is_final_name"]]
    numbered_records = [record for record in records if not record["is_final_name"]]
    latest_numbered_step = max((record["step"] for record in numbered_records), default=None)
    production_process_completed = (
        source_kind == "production"
        and process_status.get("status") == "completed_process"
        and process_status.get("process_return_code") == 0
    )
    smoke_process_completed = source_kind == "smoke" and process_status.get("status") == "passed"
    configured_step_reached = (
        latest_numbered_step is not None and latest_numbered_step >= configured_steps
    )
    training_complete = bool(
        configured_step_reached
        or (production_process_completed and final_records)
        or smoke_process_completed
    )
    completion_evidence: list[str] = []
    if configured_step_reached:
        completion_evidence.append("numbered_checkpoint_reached_configured_step")
    if production_process_completed and final_records:
        completion_evidence.append("production_process_returned_zero_with_final_checkpoint")
    if smoke_process_completed:
        completion_evidence.append("smoke_process_passed")
    if not completion_evidence:
        completion_evidence.append("no_independent_completion_evidence")
    excluded_untrusted_final_files: list[str] = []
    trusted_records = records
    if final_records and not training_complete and numbered_records:
        trusted_records = numbered_records
        excluded_untrusted_final_files = [record["path"] for record in final_records]
    step_map: dict[int, dict[str, Any]] = {}
    for record in trusted_records:
        step_map[record["step"]] = record
    unique_records = [step_map[step] for step in sorted(step_map)]
    if not unique_records:
        raise CheckpointError("No trusted checkpoint records remain after validation.")
    return {
        "source_kind": source_kind,
        "run_name": run_name,
        "run_directory": str(run_directory),
        "configured_steps": configured_steps,
        "training_complete": training_complete,
        "completion_evidence": completion_evidence,
        "process_status": process_status,
        "checkpoint_count": len(unique_records),
        "checkpoint_steps": [record["step"] for record in unique_records],
        "checkpoints": unique_records,
        "excluded_untrusted_final_files": excluded_untrusted_final_files,
        "final_quality_claim_allowed": source_kind == "production" and training_complete,
    }


def select_active(
    inventory: dict[str, Any], mode: str = "auto", manual_step: int | None = None
) -> dict[str, Any]:
    checkpoints = inventory["checkpoints"]
    if mode == "manual":
        if manual_step is None:
            raise CheckpointError("A manual checkpoint step is required for manual selection.")
        candidates = [record for record in checkpoints if record["step"] == manual_step]
        if len(candidates) != 1:
            raise CheckpointError(f"The manual checkpoint step {manual_step} was not found once.")
        selected = candidates[0]
    elif mode == "latest":
        selected = checkpoints[-1]
    else:
        final_candidates = [
            record
            for record in checkpoints
            if record["is_final_name"] and inventory["training_complete"]
        ]
        selected = final_candidates[-1] if final_candidates else checkpoints[-1]
    return {
        "selection_mode": mode,
        "source_kind": inventory["source_kind"],
        "training_complete": inventory["training_complete"],
        "completion_evidence": inventory["completion_evidence"],
        "final_quality_claim_allowed": inventory["final_quality_claim_allowed"],
        "checkpoint_step": selected["step"],
        "checkpoint_path": selected["path"],
        "checkpoint_sha256": selected["sha256"],
        "rank": selected["rank"],
        "adapter_name": PRIMARY_ADAPTER_NAME,
        "pipeline_smoke_test_only": not inventory["final_quality_claim_allowed"],
        "permanent_merge_allowed": False,
    }


def select_sweep(
    inventory: dict[str, Any],
    mode: str,
    maximum: int,
    manual_steps: list[int],
    active_step: int,
) -> list[dict[str, Any]]:
    checkpoints = inventory["checkpoints"]
    if mode == "selected":
        return [record for record in checkpoints if record["step"] == active_step]
    if mode == "manual":
        chosen = [record for record in checkpoints if record["step"] in set(manual_steps)]
        if sorted(record["step"] for record in chosen) != sorted(set(manual_steps)):
            raise CheckpointError("At least one requested checkpoint step is missing.")
        return chosen
    if mode == "all" or len(checkpoints) <= maximum:
        return list(checkpoints)
    if maximum <= 1:
        return [checkpoints[-1]]
    indices = sorted(
        {round(index * (len(checkpoints) - 1) / (maximum - 1)) for index in range(maximum)}
    )
    return [checkpoints[index] for index in indices]


def build_and_select(
    paths: ProjectPaths,
    run_name: str,
    configured_production_steps: int,
    smoke_steps: int,
    process_status: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    production_directory = paths.checkpoints_dir(run_name) / run_name
    smoke_directory = paths.smoke_checkpoints_dir(run_name) / f"{run_name}_smoke"
    if list(production_directory.glob("*.safetensors")):
        run_directory = production_directory
        source_kind = "production"
        configured_steps = configured_production_steps
    elif list(smoke_directory.glob("*.safetensors")):
        run_directory = smoke_directory
        source_kind = "smoke"
        configured_steps = smoke_steps
    else:
        raise CheckpointError("No production or smoke-test LoRA checkpoints were found.")
    inventory = inventory_from_directory(
        run_directory,
        run_name if source_kind == "production" else f"{run_name}_smoke",
        configured_steps,
        source_kind,
        process_status,
    )
    selection = select_active(inventory, mode="auto")
    write_json_atomic(paths.checkpoint_inventory(run_name), inventory)
    write_json_atomic(paths.active_checkpoint(run_name), selection)
    return inventory, selection


def apply_manual_selection(workspace: str | Path, run_name: str, step: int) -> dict[str, Any]:
    paths = ProjectPaths.create(workspace)
    inventory_path = paths.checkpoint_inventory(run_name)
    if not inventory_path.is_file():
        raise CheckpointError(f"No checkpoint inventory was found for run {run_name}.")
    inventory = read_json(inventory_path)
    selection = select_active(inventory, mode="manual", manual_step=step)
    write_json_atomic(paths.active_checkpoint(run_name), selection)
    return selection
