import zipfile

from krea2_character_lora import bundle, evaluation
from krea2_character_lora.configuration import EvaluationConfig
from krea2_character_lora.manifests import read_json, write_json_atomic
from krea2_character_lora.paths import ProjectPaths


def _import(prepared_run, tmp_path, include_all_checkpoints=False, target="target"):
    paths, run_name = prepared_run(steps=(100, 200))
    export = bundle.create_evaluation_bundle(
        paths.root, run_name, include_all_checkpoints=include_all_checkpoints
    )
    target_paths = ProjectPaths.create(tmp_path / target)
    imported = bundle.import_evaluation_bundle(target_paths.root, export.archives[0])
    write_json_atomic(target_paths.inference_asset_manifest, {"inference_model": {"revision": "x"}})
    return target_paths, imported, export.archives[0]


def _config():
    return EvaluationConfig(
        prompts=["mycharacter is a woman in a studio portrait"],
        seeds=[42],
        checkpoint_mode="all",
        maximum_checkpoints=8,
    )


def _mock_scripts(monkeypatch):
    captured = {"scripts": []}

    def fake_script(paths, script_name, request_path, log_name):
        captured["scripts"].append(script_name)
        captured["request"] = read_json(request_path)
        return {"script": script_name, "master_grid": f"/tmp/{script_name}.png"}

    monkeypatch.setattr(evaluation, "run_evaluation_script", fake_script)
    return captured


def test_selected_only_supports_base_scale_and_graceful_sweep(prepared_run, tmp_path, monkeypatch):
    target_paths, imported, _ = _import(prepared_run, tmp_path, include_all_checkpoints=False)
    captured = _mock_scripts(monkeypatch)
    report = evaluation.evaluate(target_paths, imported, _config())
    assert "base" in captured["scripts"]
    assert "scale_sweep" in captured["scripts"]
    assert "checkpoint_sweep" in captured["scripts"]
    assert report.details["single_checkpoint_sweep"] is True
    assert "checkpoint_sweep_notice" in report.details
    assert captured["request"]["include_base_in_checkpoint_grid"] is True


def test_all_checkpoint_supports_multi_checkpoint_sweep(prepared_run, tmp_path, monkeypatch):
    target_paths, imported, _ = _import(prepared_run, tmp_path, include_all_checkpoints=True)
    captured = _mock_scripts(monkeypatch)
    report = evaluation.evaluate(target_paths, imported, _config())
    assert report.details["single_checkpoint_sweep"] is False
    assert report.details["sweep_checkpoint_steps"] == [100, 200]
    assert len(captured["request"]["sweep_checkpoints"]) == 2


def test_request_carries_base_in_grid_flag(prepared_run, tmp_path, monkeypatch):
    target_paths, imported, _ = _import(prepared_run, tmp_path, include_all_checkpoints=True)
    captured = _mock_scripts(monkeypatch)
    config = _config()
    config.include_base_in_checkpoint_grid = False
    evaluation.evaluate(target_paths, imported, config)
    assert captured["request"]["include_base_in_checkpoint_grid"] is False


def test_reexport_is_reimportable_and_not_nested(prepared_run, tmp_path, monkeypatch):
    target_paths, imported, original_zip = _import(
        prepared_run, tmp_path, include_all_checkpoints=False
    )
    _mock_scripts(monkeypatch)
    report = evaluation.evaluate(target_paths, imported, _config())
    reexport = report.export(include_selected_lora=True, include_manifests=True)
    reexport_zip = reexport.archives[0]
    with zipfile.ZipFile(reexport_zip) as archive:
        names = archive.namelist()
    assert not any(name.endswith(".zip") for name in names)
    assert original_zip.name not in names
    reimported = bundle.import_evaluation_bundle(tmp_path / "target2", reexport_zip)
    assert reimported.run_name == imported.run_name
    assert reimported.trigger_word == "mycharacter"


def test_base_column_metadata_is_requested(prepared_run, tmp_path, monkeypatch):
    target_paths, imported, _ = _import(prepared_run, tmp_path, include_all_checkpoints=True)
    _mock_scripts(monkeypatch)
    report = evaluation.evaluate(target_paths, imported, _config())
    assert report.details["include_base_in_checkpoint_grid"] is True
