import json
import zipfile

import pytest

from krea2_character_lora import bundle
from krea2_character_lora.errors import BundleValidationError


def _read_bundle(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read("bundle_manifest.json").decode("utf-8"))
        contents = {name: archive.read(name) for name in names}
    return names, manifest, contents


def test_creates_selected_only_bundle(prepared_run):
    paths, run_name = prepared_run(steps=(100, 200))
    export = bundle.create_evaluation_bundle(paths.root, run_name, include_all_checkpoints=False)
    zip_path = export.archives[0]
    names, manifest, _ = _read_bundle(zip_path)
    assert manifest["bundle_type"] == "krea2_character_lora_evaluation_bundle"
    assert manifest["bundle_format_version"] == 1
    assert manifest["run_name"] == run_name
    assert manifest["trigger_word"] == "mycharacter"
    assert manifest["capabilities"]["selected_lora"] is True
    assert manifest["capabilities"]["all_checkpoints"] is False
    assert manifest["capabilities"]["checkpoint_sweep"] is False
    assert manifest["available_checkpoint_steps"] == [manifest["selected_checkpoint_step"]]
    assert "checkpoints/selected.safetensors" in names


def test_creates_all_checkpoint_bundle(prepared_run):
    paths, run_name = prepared_run(steps=(100, 200))
    export = bundle.create_evaluation_bundle(paths.root, run_name, include_all_checkpoints=True)
    names, manifest, _ = _read_bundle(export.archives[0])
    assert manifest["capabilities"]["all_checkpoints"] is True
    assert manifest["capabilities"]["checkpoint_sweep"] is True
    assert manifest["available_checkpoint_steps"] == [100, 200]
    assert "checkpoints/selected.safetensors" in names
    assert any(name.startswith("checkpoints/step_") for name in names)


def test_writes_deterministic_relative_paths(prepared_run):
    paths, run_name = prepared_run()
    export = bundle.create_evaluation_bundle(paths.root, run_name)
    _, manifest, _ = _read_bundle(export.archives[0])
    file_paths = [record["path"] for record in manifest["files"]]
    assert file_paths == sorted(file_paths)
    for path in file_paths:
        assert not path.startswith("/")
        assert "\\" not in path
        assert ".." not in path.split("/")


def test_writes_hashes_and_sizes(prepared_run):
    paths, run_name = prepared_run()
    export = bundle.create_evaluation_bundle(paths.root, run_name)
    zip_path = export.archives[0]
    with zipfile.ZipFile(zip_path) as archive:
        _, manifest, _ = _read_bundle(zip_path)
        for record in manifest["files"]:
            assert len(record["sha256"]) == 64
            assert record["size_bytes"] == len(archive.read(record["path"]))


def test_excludes_base_vae_text_encoder_and_dataset_images(prepared_run):
    paths, run_name = prepared_run()
    (paths.models / "krea_2_raw").mkdir(parents=True, exist_ok=True)
    (paths.models / "krea_2_raw" / "raw.safetensors").write_bytes(b"x" * 2048)
    (paths.dataset_training).mkdir(parents=True, exist_ok=True)
    (paths.dataset_training / "000001.png").write_bytes(b"img")
    export = bundle.create_evaluation_bundle(paths.root, run_name, include_all_checkpoints=True)
    names, _, _ = _read_bundle(export.archives[0])
    joined = " ".join(names)
    assert "raw.safetensors" not in joined
    assert "diffusion_pytorch_model" not in joined
    assert "models/" not in joined
    assert "dataset" not in joined
    for name in names:
        assert not name.startswith("images/000001")


def test_capability_flags_match_actual_contents(prepared_run):
    paths, run_name = prepared_run(steps=(100, 200), with_images=True, with_eval=True)
    export = bundle.create_evaluation_bundle(
        paths.root, run_name, include_all_checkpoints=True, include_images=True, include_logs=True
    )
    _, manifest, _ = _read_bundle(export.archives[0])
    capabilities = manifest["capabilities"]
    assert capabilities["evaluation_images"] is True
    assert capabilities["logs"] is True
    assert capabilities["previous_evaluation"] is True
    assert capabilities["checkpoint_sweep"] is True


def test_preserves_requested_logs_and_images(prepared_run):
    paths, run_name = prepared_run(with_images=True)
    export = bundle.create_evaluation_bundle(
        paths.root, run_name, include_images=True, include_logs=True
    )
    names, _, _ = _read_bundle(export.archives[0])
    assert any(name.startswith("logs/") and name.endswith(".log") for name in names)
    assert any(name.startswith("images/") and name.endswith(".png") for name in names)


def test_reports_unavailable_optional_categories(prepared_run):
    paths, run_name = prepared_run(with_images=False)
    export = bundle.create_evaluation_bundle(
        paths.root, run_name, include_images=True, include_logs=True
    )
    assert "evaluation_images" in export.details["unavailable_categories"]


def test_rejects_missing_selected_lora(prepared_run):
    paths, run_name = prepared_run()
    from krea2_character_lora.manifests import read_json

    selection = read_json(paths.active_checkpoint(run_name))
    from pathlib import Path

    Path(selection["checkpoint_path"]).unlink()
    with pytest.raises(BundleValidationError):
        bundle.create_evaluation_bundle(paths.root, run_name)


def test_exports_realistic_run_manifest_with_keep_tokens(prepared_run):
    paths, run_name = prepared_run()
    export = bundle.create_evaluation_bundle(paths.root, run_name)
    _, _, contents = _read_bundle(export.archives[0])
    training_config = json.loads(contents["manifests/training_config.json"].decode("utf-8"))
    assert training_config["keep_tokens"] == 1
    assert "shuffle_tokens" in training_config


def test_bundle_manifest_records_provenance(prepared_run):
    paths, run_name = prepared_run()
    export = bundle.create_evaluation_bundle(paths.root, run_name)
    _, manifest, _ = _read_bundle(export.archives[0])
    provenance = manifest["provenance"]
    assert provenance["training_model"]["repository"] == "krea/Krea-2-Raw"
    assert provenance["vae"]["repository"] == "artsyww/KREA2REALVAE"
    assert provenance["vae"]["source_filename"] == "krea2RealVae_v10.safetensors"
    assert len(provenance["selected_lora_sha256"]) == 64
    assert manifest["lora_rank"] == 8
