import json

import pytest

from krea2_character_lora import assets
from krea2_character_lora.errors import VaeValidationError
from krea2_character_lora.paths import ProjectPaths

REFERENCE_SHAPES = {
    "encoder.conv_in.weight": [4, 3, 3, 3],
    "encoder.conv_out.weight": [8, 4, 3, 3],
    "decoder.conv_in.weight": [4, 8, 3, 3],
    "decoder.conv_out.weight": [3, 4, 3, 3],
}


def test_select_vae_safetensors_prefers_diffusers_name():
    files = ["readme.md", "model.safetensors", "vae/diffusion_pytorch_model.safetensors"]
    assert assets.select_vae_safetensors(files) == "vae/diffusion_pytorch_model.safetensors"


def test_select_vae_safetensors_requires_a_file():
    with pytest.raises(assets.AssetError):
        assets.select_vae_safetensors(["config.json"])


def test_detect_removable_prefix():
    custom = ["vae.encoder.weight", "vae.decoder.weight"]
    reference = ["encoder.weight", "decoder.weight"]
    assert assets.detect_removable_prefix(custom, reference) is True
    assert assets.detect_removable_prefix(reference, reference) is False


def test_analyze_state_dict_reports_differences():
    custom = {"a": [2, 2], "b": [4], "c": [1]}
    reference = {"a": [2, 2], "b": [8], "d": [1]}
    analysis = assets.analyze_state_dict(custom, reference)
    assert analysis["missing_keys"] == ["d"]
    assert analysis["unexpected_keys"] == ["c"]
    assert analysis["shape_mismatches"] == ["b: [4] != [8]"]


def test_plan_vae_normalization_compatible_with_prefix():
    custom = {f"vae.{name}": shape for name, shape in REFERENCE_SHAPES.items()}
    plan = assets.plan_vae_normalization(custom, REFERENCE_SHAPES)
    assert plan["remove_prefix"] is True
    assert plan["compatible"] is True


def test_plan_vae_normalization_incompatible():
    custom = dict(REFERENCE_SHAPES)
    custom["encoder.conv_in.weight"] = [9, 9, 9, 9]
    plan = assets.plan_vae_normalization(custom, REFERENCE_SHAPES)
    assert plan["compatible"] is False
    assert plan["analysis"]["shape_mismatches"]


def test_read_safetensors_header_returns_shapes(tmp_path, make_vae_safetensors):
    path = make_vae_safetensors(tmp_path / "vae.safetensors", REFERENCE_SHAPES)
    shapes = assets.state_dict_shapes(path)
    assert shapes["encoder.conv_in.weight"] == [4, 3, 3, 3]


def _install_download_mocks(monkeypatch, tmp_path, custom_shapes):
    def fake_resolve(repo_id, revision=None):
        return f"{repo_id.replace('/', '_')}_rev"

    def fake_list(repo_id, revision=None):
        return ["config.json", "krea2realvae.safetensors"]

    def fake_download(repo_id, filename, revision, local_dir, subfolder=None):
        import numpy as np
        from safetensors.numpy import save_file

        target = local_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                name: np.zeros(tuple(shape), dtype="float32")
                for name, shape in custom_shapes.items()
            },
            str(target),
            metadata={"format": "pt"},
        )
        return target

    def fake_snapshot(repo_id, revision, local_dir, allow_patterns=None):
        import numpy as np
        from safetensors.numpy import save_file

        vae_directory = local_dir / "vae"
        vae_directory.mkdir(parents=True, exist_ok=True)
        (vae_directory / "config.json").write_text(
            json.dumps({"_class_name": "AutoencoderKLQwenImage", "z_dim": 4}), encoding="utf-8"
        )
        save_file(
            {
                name: np.zeros(tuple(shape), dtype="float32")
                for name, shape in REFERENCE_SHAPES.items()
            },
            str(vae_directory / "diffusion_pytorch_model.safetensors"),
            metadata={"format": "pt"},
        )
        return local_dir

    monkeypatch.setattr(assets, "resolve_revision", fake_resolve)
    monkeypatch.setattr(assets, "hf_list_repo_files", fake_list)
    monkeypatch.setattr(assets, "hf_download_file", fake_download)
    monkeypatch.setattr(assets, "hf_snapshot", fake_snapshot)


def test_prepare_custom_vae_success(monkeypatch, tmp_path):
    paths = ProjectPaths.create(tmp_path / "workspace")
    custom_shapes = {f"vae.{name}": shape for name, shape in REFERENCE_SHAPES.items()}
    _install_download_mocks(monkeypatch, tmp_path, custom_shapes)
    manifest = assets.prepare_custom_vae(paths)
    assert manifest["remove_prefix"] is True
    assert manifest["static_compatibility"] is True
    assert manifest["strict_validation"] == "pending"
    normalized = paths.models / "krea2_normalized_vae"
    assert (normalized / "config.json").is_file()
    assert manifest["architecture"] == "AutoencoderKLQwenImage"


def test_prepare_custom_vae_rejects_incompatible_without_fallback(monkeypatch, tmp_path):
    paths = ProjectPaths.create(tmp_path / "workspace")
    custom_shapes = {f"vae.{name}": shape for name, shape in REFERENCE_SHAPES.items()}
    custom_shapes["vae.encoder.conv_in.weight"] = [9, 9, 9, 9]
    custom_shapes["vae.unexpected.weight"] = [2, 2]
    _install_download_mocks(monkeypatch, tmp_path, custom_shapes)
    with pytest.raises(VaeValidationError):
        assets.prepare_custom_vae(paths)
    normalized = paths.models / "krea2_normalized_vae"
    assert not (normalized / "config.json").is_file()


def test_anonymous_download_wrappers_pass_no_token(monkeypatch, tmp_path):
    import huggingface_hub

    captured = {}

    def download_spy(**kwargs):
        captured["download"] = kwargs
        return str(tmp_path / "file")

    def snapshot_spy(**kwargs):
        captured["snapshot"] = kwargs
        return str(tmp_path)

    def list_spy(**kwargs):
        captured["list"] = kwargs
        return []

    class FakeApi:
        def model_info(self, **kwargs):
            captured["model_info"] = kwargs
            return type("Info", (), {"sha": "abc"})()

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download_spy)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_spy)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", list_spy)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    assets.hf_download_file("repo", "file", "rev", tmp_path)
    assets.hf_snapshot("repo", "rev", tmp_path)
    assets.hf_list_repo_files("repo", "rev")
    assets.hf_model_info("repo", "rev")

    assert captured["download"]["token"] is None
    assert captured["snapshot"]["token"] is None
    assert captured["list"]["token"] is None
    assert captured["model_info"]["token"] is None
