from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoints import select_sweep
from .configuration import EvaluationConfig
from .constants import MAX_TEXT_LENGTH, PRIMARY_ADAPTER_NAME
from .errors import EnvironmentPreparationError, EvaluationError
from .manifests import read_json, write_json_atomic
from .paths import ProjectPaths
from .runtime import evaluation_script_path
from .types import EvaluationReport, TrainingRun

_RESULT_MARKER = "__KREA2_RESULT__"


def build_evaluation_request(
    paths: ProjectPaths,
    run: TrainingRun,
    config: EvaluationConfig,
    selection: dict[str, Any],
    sweep_checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    training_config = run.details.get("training_config", {})
    return {
        "run_name": run.run_name,
        "inference_asset_manifest": str(paths.inference_asset_manifest),
        "active_checkpoint": selection,
        "adapter_name": PRIMARY_ADAPTER_NAME,
        "lora_alpha": training_config.get("lora_alpha", 32),
        "training_dtype": training_config.get("training_dtype", "bf16"),
        "max_text_length": MAX_TEXT_LENGTH,
        "prompts": list(config.prompts),
        "seeds": list(config.seeds),
        "width": config.width,
        "height": config.height,
        "inference_steps": config.inference_steps,
        "guidance_scale": config.guidance_scale,
        "negative_prompt": config.negative_prompt,
        "primary_adapter_scale": config.primary_adapter_scale,
        "scale_sweep": list(config.scale_sweep),
        "sweep_checkpoints": [
            {"step": record["step"], "path": record["path"]} for record in sweep_checkpoints
        ],
        "output_root": str(paths.inference / run.run_name),
    }


def _require_environment(paths: ProjectPaths) -> None:
    if not paths.venv_python.is_file():
        raise EnvironmentPreparationError(
            "The isolated environment is missing. Run pipeline.setup and "
            "pipeline.prepare_evaluation_assets first."
        )


def run_evaluation_script(
    paths: ProjectPaths, script_name: str, request_path: Path, log_name: str
) -> dict[str, Any]:
    _require_environment(paths)
    environment = os.environ.copy()
    environment["KREA2_EVAL_REQUEST"] = str(request_path)
    environment["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [str(paths.venv_python), str(evaluation_script_path(script_name))],
        cwd=str(paths.ai_toolkit),
        env=environment,
        capture_output=True,
        text=True,
    )
    log_path = paths.logs / log_name
    log_path.write_text(
        result.stdout + result.stderr + f"\nExit code: {result.returncode}\n", encoding="utf-8"
    )
    if result.returncode != 0:
        raise EvaluationError(f"The {script_name} evaluation failed. Complete log: {log_path}")
    structured = [line for line in result.stdout.splitlines() if line.startswith(_RESULT_MARKER)]
    if len(structured) != 1:
        raise EvaluationError(f"The {script_name} evaluation returned an invalid result.")
    return json.loads(structured[0].removeprefix(_RESULT_MARKER))


def evaluate(paths: ProjectPaths, run: TrainingRun, config: EvaluationConfig) -> EvaluationReport:
    config.validate()
    if not paths.inference_asset_manifest.is_file():
        raise EvaluationError(
            "Inference assets are not prepared. Call pipeline.prepare_evaluation_assets first."
        )
    inventory = run.details.get("checkpoint_inventory", {})
    active_path = paths.active_checkpoint(run.run_name)
    selection = (
        read_json(active_path)
        if active_path.is_file()
        else run.details.get("active_checkpoint", {})
    )
    if not selection:
        raise EvaluationError(f"No active checkpoint selection was found for run {run.run_name}.")
    sweep_checkpoints = select_sweep(
        inventory,
        config.checkpoint_mode,
        config.maximum_checkpoints,
        config.manual_checkpoint_steps,
        selection["checkpoint_step"],
    )
    request = build_evaluation_request(paths, run, config, selection, sweep_checkpoints)
    run_directory = paths.run_dir(run.run_name)
    run_directory.mkdir(parents=True, exist_ok=True)
    request_path = run_directory / "evaluation_request.json"
    write_json_atomic(request_path, request)

    manifest: dict[str, Any] = {
        "run_name": run.run_name,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_checkpoint": selection,
        "prompts": list(config.prompts),
        "seeds": list(config.seeds),
        "inference_settings": {
            "width": config.width,
            "height": config.height,
            "inference_steps": config.inference_steps,
            "guidance_scale": config.guidance_scale,
            "negative_prompt": config.negative_prompt,
        },
        "checkpoint_mode": config.checkpoint_mode,
        "sweep_checkpoint_steps": [record["step"] for record in sweep_checkpoints],
        "scale_sweep": list(config.scale_sweep),
        "automatic_selection": False,
        "permanent_merge_performed": False,
    }
    if config.compare_base_model:
        manifest["base_comparison"] = run_evaluation_script(
            paths, "base", request_path, f"{run.run_name}_base_comparison.log"
        )
    if config.run_checkpoint_sweep:
        manifest["checkpoint_sweep"] = run_evaluation_script(
            paths, "checkpoint_sweep", request_path, f"{run.run_name}_checkpoint_sweep.log"
        )
    if config.run_scale_sweep:
        manifest["scale_sweep"] = run_evaluation_script(
            paths, "scale_sweep", request_path, f"{run.run_name}_scale_sweep.log"
        )
    write_json_atomic(paths.evaluation_manifest(run.run_name), manifest)
    return EvaluationReport(
        workspace=paths.root,
        run_name=run.run_name,
        manifest_path=paths.evaluation_manifest(run.run_name),
        details=manifest,
    )


def load_evaluation(paths: ProjectPaths, run_name: str) -> EvaluationReport:
    manifest_path = paths.evaluation_manifest(run_name)
    if not manifest_path.is_file():
        raise EvaluationError(f"No evaluation manifest was found for run {run_name}.")
    return EvaluationReport(
        workspace=paths.root,
        run_name=run_name,
        manifest_path=manifest_path,
        details=read_json(manifest_path),
    )
