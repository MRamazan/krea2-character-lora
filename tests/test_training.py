import copy
from pathlib import Path

import pytest
import yaml

from krea2_character_lora import training
from krea2_character_lora.configuration import TrainingConfig
from krea2_character_lora.errors import TrainingError
from krea2_character_lora.paths import ProjectPaths
from krea2_character_lora.types import DatasetResult


def _dataset(paths):
    training_directory = paths.dataset_training
    training_directory.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for index in range(4):
        Image.new("RGB", (64, 64), (index * 10, 20, 30)).save(
            training_directory / f"{index:06d}.png"
        )
        (training_directory / f"{index:06d}.txt").write_text("mycharacter", encoding="utf-8")
    return DatasetResult(
        workspace=paths.root,
        manifest_path=paths.dataset_manifest,
        trigger_word="mycharacter",
        details={
            "training_directory": str(training_directory),
            "dataset_fingerprint_sha256": "f" * 64,
            "dimensions": {"width": {"median": 1024}, "height": {"median": 1024}},
            "pair_count": 4,
        },
    )


def _asset_manifest(paths):
    raw = paths.models / "krea_2_raw" / "raw.safetensors"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"placeholder")
    return {
        "training_model": {
            "repository": "krea/Krea-2-Raw",
            "revision": "rawrev",
            "checkpoint_path": str(raw),
            "checkpoint_filename": "raw.safetensors",
        },
        "text_encoder": {"local_directory": str(paths.models / "text_encoder")},
        "vae": {
            "normalized_directory": str(paths.models / "krea2_normalized_vae"),
            "revision": "vaerev",
        },
    }


def test_configuration_is_character_only_and_transformer_lora(tmp_path):
    paths = ProjectPaths.create(tmp_path / "workspace")
    configuration = training.build_training_configuration(
        TrainingConfig(), _dataset(paths), _asset_manifest(paths), paths
    )
    assert "concept_type" not in configuration["meta"]
    process = configuration["config"]["process"][0]
    assert process["model"]["arch"] == "krea2"
    assert process["network"]["transformer_only"] is True
    assert process["train"]["train_text_encoder"] is False
    assert process["train"]["train_unet"] is True
    assert process["train"]["merge_network_on_save"] is False
    assert process["model"]["model_kwargs"]["vae_path"] == str(
        paths.models / "krea2_normalized_vae"
    )


def test_default_sample_prompts_use_trigger():
    prompts = training.default_sample_prompts("mycharacter")
    assert all("mycharacter" in prompt for prompt in prompts)


def test_preflight_rejects_text_encoder_training(tmp_path):
    paths = ProjectPaths.create(tmp_path / "workspace")
    dataset = _dataset(paths)
    configuration = training.build_training_configuration(
        TrainingConfig(), dataset, _asset_manifest(paths), paths
    )
    bad = copy.deepcopy(configuration)
    bad["config"]["process"][0]["train"]["train_text_encoder"] = True
    with pytest.raises(TrainingError):
        training.preflight_configuration(bad, dataset.training_directory)


def test_preflight_rejects_permanent_merge(tmp_path):
    paths = ProjectPaths.create(tmp_path / "workspace")
    dataset = _dataset(paths)
    configuration = training.build_training_configuration(
        TrainingConfig(), dataset, _asset_manifest(paths), paths
    )
    bad = copy.deepcopy(configuration)
    bad["config"]["process"][0]["train"]["merge_network_on_save"] = True
    with pytest.raises(TrainingError):
        training.preflight_configuration(bad, dataset.training_directory)


def test_write_configurations_produces_reduced_smoke(tmp_path):
    paths = ProjectPaths.create(tmp_path / "workspace")
    dataset = _dataset(paths)
    resolved = training.write_configurations(
        TrainingConfig(training_steps=2000), dataset, _asset_manifest(paths), paths, smoke_steps=3
    )
    production = yaml.safe_load(Path(resolved["production_path"]).read_text(encoding="utf-8"))
    smoke = yaml.safe_load(Path(resolved["smoke_path"]).read_text(encoding="utf-8"))
    assert production["config"]["process"][0]["train"]["steps"] == 2000
    assert smoke["config"]["process"][0]["train"]["steps"] == 3
    assert smoke["config"]["process"][0]["train"]["disable_sampling"] is True
    assert smoke["config"]["name"].endswith("_smoke")


def test_finalize_run_writes_manifest_and_reloads(tmp_path, make_lora_checkpoint):
    paths = ProjectPaths.create(tmp_path / "workspace")
    dataset = _dataset(paths)
    asset_manifest = _asset_manifest(paths)
    config = TrainingConfig(run_name="character_v1", training_steps=200)
    resolved = training.write_configurations(config, dataset, asset_manifest, paths, smoke_steps=3)
    production = paths.checkpoints_dir("character_v1") / "character_v1"
    production.mkdir(parents=True, exist_ok=True)
    make_lora_checkpoint(production / "character_v1_000100.safetensors", rank=8)
    make_lora_checkpoint(production / "character_v1_000200.safetensors", rank=8)
    process_status = {"status": "completed_process", "process_return_code": 0}
    run = training.finalize_run(
        paths, config, dataset, asset_manifest, resolved, process_status, smoke_steps=3
    )
    assert run.training_complete is True
    assert paths.run_manifest("character_v1").is_file()
    reloaded = training.load_run(paths, "character_v1")
    assert reloaded.trigger_word == "mycharacter"
    assert reloaded.details["source_kind"] == "production"
    assert reloaded.details["model_revision"] == "rawrev"


def test_load_run_missing_raises(tmp_path):
    paths = ProjectPaths.create(tmp_path / "workspace")
    with pytest.raises(TrainingError):
        training.load_run(paths, "does_not_exist")
