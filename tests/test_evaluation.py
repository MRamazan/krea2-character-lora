import pytest

from krea2_character_lora import evaluation
from krea2_character_lora.checkpoints import inventory_from_directory, select_active
from krea2_character_lora.configuration import EvaluationConfig
from krea2_character_lora.errors import EvaluationError
from krea2_character_lora.manifests import read_json, write_json_atomic
from krea2_character_lora.paths import ProjectPaths
from krea2_character_lora.types import TrainingRun


def _run(tmp_path, make_lora_checkpoint):
    paths = ProjectPaths.create(tmp_path / "workspace")
    run_name = "character_v1"
    production = paths.checkpoints_dir(run_name) / run_name
    production.mkdir(parents=True, exist_ok=True)
    for step in (100, 200, 300):
        make_lora_checkpoint(production / f"{run_name}_{step:09d}.safetensors", rank=8)
    inventory = inventory_from_directory(
        production,
        run_name,
        300,
        "production",
        {"status": "completed_process", "process_return_code": 0},
    )
    selection = select_active(inventory, mode="auto")
    write_json_atomic(paths.active_checkpoint(run_name), selection)
    write_json_atomic(paths.inference_asset_manifest, {"inference_model": {"revision": "turborev"}})
    run = TrainingRun(
        workspace=paths.root,
        run_name=run_name,
        manifest_path=paths.run_manifest(run_name),
        trigger_word="mycharacter",
        details={
            "training_config": {"lora_alpha": 8, "training_dtype": "bf16"},
            "checkpoint_inventory": inventory,
            "active_checkpoint": selection,
        },
    )
    return paths, run


def _config():
    return EvaluationConfig(
        prompts=["mycharacter portrait"],
        seeds=[42],
        checkpoint_mode="auto",
        maximum_checkpoints=2,
    )


def test_build_evaluation_request_structure(tmp_path, make_lora_checkpoint):
    paths, run = _run(tmp_path, make_lora_checkpoint)
    inventory = run.details["checkpoint_inventory"]
    selection = run.details["active_checkpoint"]
    request = evaluation.build_evaluation_request(
        paths, run, _config(), selection, inventory["checkpoints"][:2]
    )
    assert request["adapter_name"] == "character_adapter"
    assert request["lora_alpha"] == 8
    assert request["output_root"].endswith("character_v1")
    assert len(request["sweep_checkpoints"]) == 2


def test_evaluate_assembles_manifest_with_mocked_scripts(
    tmp_path, monkeypatch, make_lora_checkpoint
):
    paths, run = _run(tmp_path, make_lora_checkpoint)

    def fake_script(inner_paths, script_name, request_path, log_name):
        return {"script": script_name, "master_grid": f"/tmp/{script_name}.png"}

    monkeypatch.setattr(evaluation, "run_evaluation_script", fake_script)
    report = evaluation.evaluate(paths, run, _config())
    assert report.details["automatic_selection"] is False
    assert "base_comparison" in report.details
    assert "checkpoint_sweep" in report.details
    assert "scale_sweep" in report.details
    assert paths.evaluation_manifest("character_v1").is_file()


def test_evaluate_requires_inference_assets(tmp_path, make_lora_checkpoint):
    paths, run = _run(tmp_path, make_lora_checkpoint)
    paths.inference_asset_manifest.unlink()
    with pytest.raises(EvaluationError):
        evaluation.evaluate(paths, run, _config())


def test_report_select_checkpoint_updates_active(tmp_path, monkeypatch, make_lora_checkpoint):
    paths, run = _run(tmp_path, make_lora_checkpoint)
    write_json_atomic(
        paths.checkpoint_inventory("character_v1"), run.details["checkpoint_inventory"]
    )

    def fake_script(inner_paths, script_name, request_path, log_name):
        return {"script": script_name}

    monkeypatch.setattr(evaluation, "run_evaluation_script", fake_script)
    report = evaluation.evaluate(paths, run, _config())
    report.select_checkpoint(100)
    selection = read_json(paths.active_checkpoint("character_v1"))
    assert selection["checkpoint_step"] == 100
    assert selection["selection_mode"] == "manual"
