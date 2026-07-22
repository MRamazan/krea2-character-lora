from __future__ import annotations

import copy
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .configuration import TrainingConfig
from .constants import MAX_TEXT_LENGTH, MODEL_ARCHITECTURE
from .errors import EnvironmentPreparationError, TrainingError
from .isolated import isolated_script
from .manifests import read_json, write_json_atomic
from .paths import ProjectPaths
from .types import DatasetResult, TrainingRun


def default_sample_prompts(trigger_word: str) -> list[str]:
    return [
        f"{trigger_word} portrait photograph, soft studio lighting, sharp focus",
        f"{trigger_word} standing outdoors in natural daylight, full body",
        f"{trigger_word} candid close-up, shallow depth of field",
    ]


def build_training_configuration(
    config: TrainingConfig,
    dataset: DatasetResult,
    asset_manifest: dict[str, Any],
    paths: ProjectPaths,
) -> dict[str, Any]:
    training_model = asset_manifest["training_model"]
    text_encoder = asset_manifest["text_encoder"]
    vae = asset_manifest["vae"]
    checkpoint_path = Path(training_model["checkpoint_path"])
    sample_section = {
        "sampler": "flowmatch",
        "sample_every": config.training_sample_every,
        "sample_start_step": 0,
        "width": dataset.details.get("dimensions", {}).get("width", {}).get("median", 1024),
        "height": dataset.details.get("dimensions", {}).get("height", {}).get("median", 1024),
        "prompts": default_sample_prompts(dataset.trigger_word),
        "neg": "",
        "seed": 42,
        "walk_seed": False,
        "guidance_scale": config.raw_sample_guidance,
        "sample_steps": config.raw_sample_steps,
        "network_multiplier": 1.0,
    }
    process = {
        "type": "sd_trainer",
        "training_folder": str(paths.checkpoints_dir(config.run_name)),
        "device": "cuda:0",
        "trigger_word": dataset.trigger_word,
        "network": {
            "type": "lora",
            "linear": config.lora_rank,
            "linear_alpha": config.lora_alpha,
            "transformer_only": True,
            "all_layers": False,
        },
        "save": {
            "dtype": "float16",
            "save_every": config.save_every,
            "max_step_saves_to_keep": config.max_checkpoints_to_keep,
            "save_format": "safetensors",
            "push_to_hub": False,
        },
        "datasets": [
            {
                "folder_path": str(dataset.training_directory),
                "caption_ext": "txt",
                "resolution": list(config.resolutions),
                "buckets": True,
                "bucket_tolerance": 16,
                "num_repeats": config.dataset_repeats,
                "caption_dropout_rate": config.caption_dropout_rate,
                "token_dropout_rate": config.token_dropout_rate,
                "shuffle_tokens": config.shuffle_tokens,
                "keep_tokens": config.keep_tokens,
                "random_crop": False,
                "random_scale": False,
                "flip_x": config.flip_x,
                "flip_y": False,
                "cache_latents": False,
                "cache_latents_to_disk": True,
                "cache_text_embeddings": True,
                "num_workers": 2,
                "prefetch_factor": 2,
            }
        ],
        "train": {
            "batch_size": config.batch_size,
            "steps": config.training_steps,
            "gradient_accumulation": config.gradient_accumulation,
            "train_unet": True,
            "train_text_encoder": False,
            "train_refiner": False,
            "train_turbo": False,
            "gradient_checkpointing": True,
            "noise_scheduler": "flowmatch",
            "timestep_type": "sigmoid",
            "optimizer": config.optimizer,
            "optimizer_params": {"weight_decay": config.weight_decay},
            "lr": config.learning_rate,
            "lr_scheduler": config.lr_scheduler,
            "lr_scheduler_params": {},
            "max_grad_norm": config.max_grad_norm,
            "loss_target": "noise",
            "loss_type": "mse",
            "content_or_style": "balanced",
            "prompt_dropout_prob": 0.0,
            "cache_text_embeddings": True,
            "skip_first_sample": True,
            "disable_sampling": not config.generate_training_samples,
            "merge_network_on_save": False,
            "dtype": config.training_dtype,
        },
        "model": {
            "name_or_path": str(checkpoint_path.parent),
            "arch": MODEL_ARCHITECTURE,
            "dtype": config.training_dtype,
            "vae_dtype": config.training_dtype,
            "te_dtype": config.training_dtype,
            "quantize": False,
            "quantize_te": False,
            "low_vram": False,
            "layer_offloading": False,
            "split_model_over_gpus": False,
            "compile": False,
            "model_kwargs": {
                "checkpoint_filename": training_model["checkpoint_filename"],
                "text_encoder_path": text_encoder["local_directory"],
                "vae_path": vae["normalized_directory"],
                "max_text_length": MAX_TEXT_LENGTH,
            },
        },
        "sample": sample_section,
    }
    return {
        "job": "extension",
        "config": {"name": config.run_name, "process": [process]},
        "meta": {
            "name": "[name]",
            "version": "1.0",
            "project_name": config.project_name,
            "trigger_word": dataset.trigger_word,
            "base_model": training_model["repository"],
            "base_model_revision": training_model["revision"],
            "dataset_fingerprint_sha256": dataset.fingerprint,
        },
    }


def build_smoke_configuration(
    configuration: dict[str, Any], config: TrainingConfig, paths: ProjectPaths, smoke_steps: int
) -> dict[str, Any]:
    smoke = copy.deepcopy(configuration)
    smoke["config"]["name"] = f"{config.run_name}_smoke"
    process = smoke["config"]["process"][0]
    process["training_folder"] = str(paths.smoke_checkpoints_dir(config.run_name))
    process["save"]["save_every"] = 1
    process["save"]["max_step_saves_to_keep"] = max(3, smoke_steps + 1)
    process["train"]["steps"] = smoke_steps
    process["train"]["disable_sampling"] = True
    process["sample"]["sample_every"] = 1_000_000_000
    return smoke


def preflight_configuration(
    configuration: dict[str, Any], training_directory: Path
) -> dict[str, Any]:
    processes = configuration.get("config", {}).get("process", [])
    if not isinstance(processes, list) or len(processes) != 1:
        raise TrainingError("The training configuration must contain exactly one process.")
    process = processes[0]
    if process.get("type") != "sd_trainer":
        raise TrainingError("The training process type must be sd_trainer.")
    model_section = process.get("model", {})
    network_section = process.get("network", {})
    train_section = process.get("train", {})
    dataset_section = process.get("datasets", [])
    if model_section.get("arch") != MODEL_ARCHITECTURE:
        raise TrainingError("The model architecture must be krea2.")
    if network_section.get("type") != "lora":
        raise TrainingError("Only LoRA network training is supported.")
    if not network_section.get("transformer_only"):
        raise TrainingError("Krea 2 LoRA training must target the transformer only.")
    if train_section.get("train_text_encoder"):
        raise TrainingError("Text-encoder training must remain disabled.")
    if not train_section.get("train_unet"):
        raise TrainingError("Transformer training must remain enabled.")
    if train_section.get("merge_network_on_save"):
        raise TrainingError("Permanent LoRA merging must remain disabled.")
    if not isinstance(dataset_section, list) or len(dataset_section) != 1:
        raise TrainingError("Exactly one canonical dataset directory is required.")
    folder = Path(dataset_section[0]["folder_path"])
    images = [path for path in folder.glob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    captions = list(folder.glob("*.txt"))
    if not images or len(images) != len(captions):
        raise TrainingError("The canonical dataset directory lacks matching image-caption pairs.")
    return {
        "run_name": configuration["config"]["name"],
        "dataset_pairs": len(images),
        "network_rank": network_section["linear"],
        "network_alpha": network_section["linear_alpha"],
        "training_steps": train_section["steps"],
        "train_text_encoder": train_section["train_text_encoder"],
        "merge_network_on_save": train_section["merge_network_on_save"],
    }


def write_configurations(
    config: TrainingConfig,
    dataset: DatasetResult,
    asset_manifest: dict[str, Any],
    paths: ProjectPaths,
    smoke_steps: int,
) -> dict[str, Any]:
    configuration = build_training_configuration(config, dataset, asset_manifest, paths)
    smoke = build_smoke_configuration(configuration, config, paths, smoke_steps)
    preflight = preflight_configuration(configuration, dataset.training_directory)
    run_directory = paths.run_dir(config.run_name)
    run_directory.mkdir(parents=True, exist_ok=True)
    production_path = run_directory / "train_krea2_lora.yaml"
    smoke_path = run_directory / "train_krea2_lora_smoke.yaml"
    production_path.write_text(
        yaml.safe_dump(configuration, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    smoke_path.write_text(
        yaml.safe_dump(smoke, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return {
        "configuration": configuration,
        "smoke_configuration": smoke,
        "preflight": preflight,
        "production_path": str(production_path),
        "smoke_path": str(smoke_path),
    }


def preview_resolved_configuration(
    config: TrainingConfig,
    dataset: DatasetResult,
    asset_manifest: dict[str, Any],
    paths: ProjectPaths,
    smoke_steps: int,
) -> dict[str, Any]:
    resolved = write_configurations(config, dataset, asset_manifest, paths, smoke_steps)
    return resolved["preflight"]


def _run_streamed(
    command: list[str], cwd: Path, environment: dict[str, str], log_path: Path
) -> tuple[int, bool]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    interrupted = False
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            if process.stdout is None:
                raise TrainingError("Unable to capture the training process output.")
            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
                log_handle.flush()
        except KeyboardInterrupt:
            interrupted = True
            process.send_signal(signal.SIGINT)
            process.wait()
        return_code = process.wait()
    return return_code, interrupted


def _require_environment(paths: ProjectPaths) -> None:
    if not paths.venv_python.is_file():
        raise EnvironmentPreparationError(
            "The isolated AI Toolkit environment is missing. Run pipeline.setup first. "
            f"Expected interpreter: {paths.venv_python}"
        )


def run_source_preflight(paths: ProjectPaths, production_path: str) -> dict[str, Any]:
    _require_environment(paths)
    environment = os.environ.copy()
    environment["KREA2_TRAINING_CONFIGURATION"] = production_path
    environment["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [str(paths.venv_python), str(isolated_script("preflight.py"))],
        cwd=str(paths.ai_toolkit),
        env=environment,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise TrainingError(f"Training preflight failed.\n{result.stdout}\n{result.stderr}")
    import json

    preflight = json.loads(result.stdout.strip().splitlines()[-1])
    if preflight.get("target_lora_modules") != ["SingleStreamDiT"]:
        raise TrainingError(
            f"Unexpected Krea 2 LoRA target modules: {preflight.get('target_lora_modules')}"
        )
    return preflight


def run_smoke_test(paths: ProjectPaths, run_name: str, smoke_path: str) -> dict[str, Any]:
    _require_environment(paths)
    log_path = paths.logs / f"{run_name}_smoke_training.log"
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    return_code, interrupted = _run_streamed(
        [str(paths.venv_python), "run.py", smoke_path],
        paths.ai_toolkit,
        environment,
        log_path,
    )
    status = {
        "status": "passed" if return_code == 0 and not interrupted else "failed",
        "process_return_code": return_code,
        "interrupted": interrupted,
        "log": str(log_path),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if status["status"] != "passed":
        raise TrainingError(f"The training smoke test failed. Complete log: {log_path}")
    return status


def run_production_training(
    paths: ProjectPaths, run_name: str, production_path: str, resume: str
) -> dict[str, Any]:
    _require_environment(paths)
    production_directory = paths.checkpoints_dir(run_name) / run_name
    existing = (
        list(production_directory.glob("*.safetensors")) if production_directory.is_dir() else []
    )
    if resume == "never" and existing:
        import shutil

        shutil.rmtree(paths.checkpoints_dir(run_name))
    if resume == "required" and not existing:
        raise TrainingError(
            "Resume was required but no existing checkpoints were found for this run."
        )
    log_path = paths.logs / f"{run_name}_production_training.log"
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    return_code, interrupted = _run_streamed(
        [str(paths.venv_python), "run.py", production_path],
        paths.ai_toolkit,
        environment,
        log_path,
    )
    status = {
        "status": "interrupted" if interrupted else "completed_process",
        "process_return_code": return_code,
        "training_complete": False,
        "resume_mode": resume,
        "resumed": bool(existing),
        "log": str(log_path),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if return_code not in {0, 130, -2} and not interrupted:
        raise TrainingError(
            f"Production training failed with exit code {return_code}. Log: {log_path}"
        )
    return status


def write_run_manifest(
    paths: ProjectPaths,
    config: TrainingConfig,
    dataset: DatasetResult,
    asset_manifest: dict[str, Any],
    resolved: dict[str, Any],
    process_status: dict[str, Any],
    inventory: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "run_name": config.run_name,
        "project_name": config.project_name,
        "trigger_word": dataset.trigger_word,
        "training_config": _config_dict(config),
        "dataset_fingerprint_sha256": dataset.fingerprint,
        "dataset_manifest": str(paths.dataset_manifest),
        "model_revision": asset_manifest.get("training_model", {}).get("revision"),
        "vae_revision": asset_manifest.get("vae", {}).get("revision"),
        "source_revision": read_source_revision(paths),
        "production_config": resolved["production_path"],
        "smoke_config": resolved["smoke_path"],
        "preflight": resolved["preflight"],
        "process_status": process_status,
        "status": inventory["source_kind"]
        + "_"
        + ("complete" if inventory["training_complete"] else "incomplete"),
        "training_complete": inventory["training_complete"],
        "source_kind": inventory["source_kind"],
        "checkpoint_inventory": inventory,
        "active_checkpoint": selection,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(paths.run_manifest(config.run_name), manifest)
    return manifest


def _config_dict(config: TrainingConfig) -> dict[str, Any]:
    return {slot: getattr(config, slot) for slot in config.__slots__}


def read_source_revision(paths: ProjectPaths) -> str | None:
    if not paths.environment_manifest.is_file():
        return None
    return read_json(paths.environment_manifest).get("resolved_commit")


def finalize_run(
    paths: ProjectPaths,
    config: TrainingConfig,
    dataset: DatasetResult,
    asset_manifest: dict[str, Any],
    resolved: dict[str, Any],
    process_status: dict[str, Any],
    smoke_steps: int,
) -> TrainingRun:
    from .checkpoints import build_and_select

    inventory, selection = build_and_select(
        paths, config.run_name, config.training_steps, smoke_steps, process_status
    )
    manifest = write_run_manifest(
        paths, config, dataset, asset_manifest, resolved, process_status, inventory, selection
    )
    return TrainingRun(
        workspace=paths.root,
        run_name=config.run_name,
        manifest_path=paths.run_manifest(config.run_name),
        trigger_word=dataset.trigger_word,
        details=manifest,
    )


def load_run(paths: ProjectPaths, run_name: str) -> TrainingRun:
    manifest_path = paths.run_manifest(run_name)
    if not manifest_path.is_file():
        raise TrainingError(
            f"No run manifest was found at {manifest_path}. The run cannot be reloaded."
        )
    manifest = read_json(manifest_path)
    return TrainingRun(
        workspace=paths.root,
        run_name=run_name,
        manifest_path=manifest_path,
        trigger_word=manifest["trigger_word"],
        details=manifest,
    )
