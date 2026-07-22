import zipfile

import pytest

from krea2_character_lora.checkpoints import inventory_from_directory, select_active
from krea2_character_lora.errors import ExportError
from krea2_character_lora.export import _assert_no_secret_fields, package_run
from krea2_character_lora.manifests import write_json_atomic
from krea2_character_lora.paths import ProjectPaths


def _prepare_run(tmp_path, make_lora_checkpoint):
    paths = ProjectPaths.create(tmp_path / "workspace")
    run_name = "character_v1"
    production = paths.checkpoints_dir(run_name) / run_name
    production.mkdir(parents=True, exist_ok=True)
    make_lora_checkpoint(production / f"{run_name}_000100.safetensors", rank=8)
    make_lora_checkpoint(production / f"{run_name}_000200.safetensors", rank=8)
    inventory = inventory_from_directory(
        production,
        run_name,
        200,
        "production",
        {"status": "completed_process", "process_return_code": 0},
    )
    selection = select_active(inventory, mode="auto")
    write_json_atomic(paths.active_checkpoint(run_name), selection)
    write_json_atomic(
        paths.run_manifest(run_name),
        {
            "run_name": run_name,
            "project_name": "krea2_character_lora",
            "trigger_word": "mycharacter",
            "model_revision": "rawrev",
            "vae_revision": "vaerev",
            "source_revision": "sourcerev",
            "checkpoint_inventory": inventory,
            "active_checkpoint": selection,
        },
    )
    (paths.logs / "production_training.log").write_text("log", encoding="utf-8")
    return paths, run_name


def test_package_run_creates_archives_and_manifest(tmp_path, make_lora_checkpoint):
    paths, run_name = _prepare_run(tmp_path, make_lora_checkpoint)
    bundle = package_run(paths.root, run_name, include_all_checkpoints=True)
    names = {archive.name for archive in bundle.archives}
    assert any(name.endswith("resume_manifest.json") for name in names)
    assert any("selected_step" in name for name in names)
    assert any("checkpoints" in name for name in names)
    for record in bundle.details["packages"]:
        assert len(record["sha256"]) == 64


def test_package_run_excludes_base_weights(tmp_path, make_lora_checkpoint):
    paths, run_name = _prepare_run(tmp_path, make_lora_checkpoint)
    (paths.models / "krea_2_raw").mkdir(parents=True, exist_ok=True)
    (paths.models / "krea_2_raw" / "raw.safetensors").write_bytes(b"x" * 1024)
    bundle = package_run(paths.root, run_name, include_all_checkpoints=True)
    for archive in bundle.archives:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as handle:
                for name in handle.namelist():
                    assert "krea_2_raw" not in name
                    assert "raw.safetensors" not in name


def test_package_run_selected_lora_matches_source(tmp_path, make_lora_checkpoint):
    paths, run_name = _prepare_run(tmp_path, make_lora_checkpoint)
    bundle = package_run(paths.root, run_name, include_selected_lora=True)
    selected = [archive for archive in bundle.archives if "selected_step" in archive.name]
    assert len(selected) == 1
    assert selected[0].is_file()


def test_assert_no_secret_fields_rejects_token():
    with pytest.raises(ExportError):
        _assert_no_secret_fields({"nested": {"hf_token": "secret"}}, __file__)


def test_assert_no_secret_fields_allows_clean_payload():
    _assert_no_secret_fields({"run_name": "x", "packages": [{"sha256": "a"}]}, __file__)
