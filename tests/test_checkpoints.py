import pytest

from krea2_character_lora.checkpoints import (
    apply_manual_selection,
    build_and_select,
    discover_checkpoint_files,
    inspect_checkpoint,
    inventory_from_directory,
    select_active,
    select_sweep,
)
from krea2_character_lora.errors import CheckpointError
from krea2_character_lora.paths import ProjectPaths


def _populate(directory, run_name, steps, make_lora_checkpoint, final=False, rank=4):
    directory.mkdir(parents=True, exist_ok=True)
    for step in steps:
        make_lora_checkpoint(directory / f"{run_name}_{step:09d}.safetensors", rank=rank)
    if final:
        make_lora_checkpoint(directory / f"{run_name}.safetensors", rank=rank)


def test_discover_checkpoint_files_finds_numbered_and_final(tmp_path, make_lora_checkpoint):
    directory = tmp_path / "run"
    _populate(directory, "character_v1", [100, 200], make_lora_checkpoint, final=True)
    discovered = discover_checkpoint_files(directory, "character_v1")
    finals = [item for item in discovered if item[2]]
    numbered = sorted(item[0] for item in discovered if not item[2])
    assert len(finals) == 1
    assert numbered == [100, 200]


def test_inspect_checkpoint_infers_rank(tmp_path, make_lora_checkpoint):
    path = make_lora_checkpoint(tmp_path / "character_v1_000100.safetensors", rank=16)
    record = inspect_checkpoint(path)
    assert record["rank"] == 16
    assert record["nonfinite_parameter_count"] == 0


def test_inspect_checkpoint_rejects_nonfinite(tmp_path, make_lora_checkpoint):
    path = make_lora_checkpoint(tmp_path / "bad.safetensors", rank=4, nonfinite=True)
    with pytest.raises(CheckpointError):
        inspect_checkpoint(path)


def test_inventory_marks_completed_when_configured_step_reached(tmp_path, make_lora_checkpoint):
    directory = tmp_path / "run"
    _populate(directory, "character_v1", [100, 200], make_lora_checkpoint)
    inventory = inventory_from_directory(
        directory, "character_v1", configured_steps=200, source_kind="production", process_status={}
    )
    assert inventory["training_complete"] is True
    assert inventory["checkpoint_steps"] == [100, 200]
    assert inventory["final_quality_claim_allowed"] is True


def test_inventory_interrupted_run_is_incomplete_and_excludes_final(tmp_path, make_lora_checkpoint):
    directory = tmp_path / "run"
    _populate(directory, "character_v1", [100], make_lora_checkpoint, final=True)
    inventory = inventory_from_directory(
        directory,
        "character_v1",
        configured_steps=2000,
        source_kind="production",
        process_status={"status": "interrupted"},
    )
    assert inventory["training_complete"] is False
    assert inventory["checkpoint_steps"] == [100]
    assert inventory["excluded_untrusted_final_files"]
    assert inventory["final_quality_claim_allowed"] is False


def test_inventory_schema_mismatch_raises(tmp_path, make_lora_checkpoint):
    directory = tmp_path / "run"
    make_lora_checkpoint(directory / "character_v1_000100.safetensors", rank=4)
    make_lora_checkpoint(directory / "character_v1_000200.safetensors", rank=4, extra_key=True)
    with pytest.raises(CheckpointError):
        inventory_from_directory(
            directory,
            "character_v1",
            configured_steps=200,
            source_kind="production",
            process_status={},
        )


def test_select_active_modes(tmp_path, make_lora_checkpoint):
    directory = tmp_path / "run"
    _populate(directory, "character_v1", [100, 200], make_lora_checkpoint, final=True)
    inventory = inventory_from_directory(
        directory, "character_v1", configured_steps=200, source_kind="production", process_status={}
    )
    latest = select_active(inventory, mode="latest")
    manual = select_active(inventory, mode="manual", manual_step=100)
    assert latest["checkpoint_step"] == 200
    assert manual["checkpoint_step"] == 100
    with pytest.raises(CheckpointError):
        select_active(inventory, mode="manual", manual_step=999)


def test_select_sweep_modes(tmp_path, make_lora_checkpoint):
    directory = tmp_path / "run"
    _populate(directory, "character_v1", [100, 200, 300, 400], make_lora_checkpoint)
    inventory = inventory_from_directory(
        directory, "character_v1", configured_steps=400, source_kind="production", process_status={}
    )
    assert len(select_sweep(inventory, "all", 8, [], 400)) == 4
    assert len(select_sweep(inventory, "auto", 2, [], 400)) == 2
    assert [record["step"] for record in select_sweep(inventory, "selected", 8, [], 300)] == [300]
    manual = select_sweep(inventory, "manual", 8, [100, 300], 400)
    assert sorted(record["step"] for record in manual) == [100, 300]
    with pytest.raises(CheckpointError):
        select_sweep(inventory, "manual", 8, [100, 999], 400)


def test_build_and_select_and_manual_reload(tmp_path, make_lora_checkpoint):
    paths = ProjectPaths.create(tmp_path / "workspace")
    run_name = "character_v1"
    production = paths.checkpoints_dir(run_name) / run_name
    _populate(production, run_name, [100, 200], make_lora_checkpoint)
    process_status = {"status": "completed_process", "process_return_code": 0}
    inventory, selection = build_and_select(paths, run_name, 200, 3, process_status)
    assert inventory["training_complete"] is True
    assert paths.checkpoint_inventory(run_name).is_file()
    assert paths.active_checkpoint(run_name).is_file()
    updated = apply_manual_selection(paths.root, run_name, 100)
    assert updated["checkpoint_step"] == 100
    assert updated["selection_mode"] == "manual"


def test_build_and_select_prefers_smoke_when_no_production(tmp_path, make_lora_checkpoint):
    paths = ProjectPaths.create(tmp_path / "workspace")
    run_name = "character_v1"
    smoke = paths.smoke_checkpoints_dir(run_name) / f"{run_name}_smoke"
    _populate(smoke, f"{run_name}_smoke", [1, 2, 3], make_lora_checkpoint)
    inventory, selection = build_and_select(paths, run_name, 2000, 3, {"status": "passed"})
    assert inventory["source_kind"] == "smoke"
    assert inventory["training_complete"] is True
    assert selection["final_quality_claim_allowed"] is False
