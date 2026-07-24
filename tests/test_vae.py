import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from krea2_character_lora import assets
from krea2_character_lora.paths import ProjectPaths

QWEN_CONFIG = {
    "_class_name": "AutoencoderKLQwenImage",
    "z_dim": 16,
    "temperal_downsample": [False, True, True],
}


def _install_vae_snapshot_mock(monkeypatch):
    def fake_resolve(repo_id, revision=None):
        return f"{repo_id.replace('/', '_')}_rev"

    def fake_snapshot(repo_id, revision, local_dir, allow_patterns=None):
        assert repo_id == "Qwen/Qwen-Image"
        vae_directory = local_dir / "vae"
        vae_directory.mkdir(parents=True, exist_ok=True)
        (vae_directory / "config.json").write_text(json.dumps(QWEN_CONFIG), encoding="utf-8")
        save_file(
            {"encoder.conv_in.weight": np.zeros((2,), dtype="float32")},
            str(vae_directory / "diffusion_pytorch_model.safetensors"),
            metadata={"format": "pt"},
        )
        return local_dir

    monkeypatch.setattr(assets, "resolve_revision", fake_resolve)
    monkeypatch.setattr(assets, "hf_snapshot", fake_snapshot)


def test_prepare_vae_uses_krea2_default(monkeypatch, tmp_path):
    paths = ProjectPaths.create(tmp_path / "workspace")
    _install_vae_snapshot_mock(monkeypatch)
    manifest = assets.prepare_vae(paths)
    assert manifest["source"] == "krea2_default"
    assert manifest["repository"] == "Qwen/Qwen-Image"
    assert manifest["subfolder"] == "vae"
    assert manifest["architecture"] == "AutoencoderKLQwenImage"
    assert manifest["weights_filename"] == "diffusion_pytorch_model.safetensors"
    assert len(manifest["weights_sha256"]) == 64
    assert manifest["strict_validation"] == "pending"
    normalized = manifest["normalized_directory"]
    assert normalized.endswith("krea2_vae/vae")
    assert (paths.models / "krea2_vae" / "vae" / "config.json").is_file()
    assert (paths.models / "krea2_vae" / "vae" / "diffusion_pytorch_model.safetensors").is_file()


def test_prepare_vae_requires_config_and_weights(monkeypatch, tmp_path):
    paths = ProjectPaths.create(tmp_path / "workspace")

    def fake_resolve(repo_id, revision=None):
        return "rev"

    def empty_snapshot(repo_id, revision, local_dir, allow_patterns=None):
        (local_dir / "vae").mkdir(parents=True, exist_ok=True)
        return local_dir

    monkeypatch.setattr(assets, "resolve_revision", fake_resolve)
    monkeypatch.setattr(assets, "hf_snapshot", empty_snapshot)
    with pytest.raises(assets.AssetError):
        assets.prepare_vae(paths)


def test_no_custom_vae_repository_is_referenced():
    from krea2_character_lora import constants

    assert not hasattr(constants, "CUSTOM_VAE_REPOSITORY")
    assert constants.VAE_REPOSITORY == "Qwen/Qwen-Image"


def test_anonymous_download_wrappers_pass_no_token(monkeypatch, tmp_path):
    import huggingface_hub

    captured = {}

    def download_spy(**kwargs):
        captured["download"] = kwargs
        return str(tmp_path / "file")

    def snapshot_spy(**kwargs):
        captured["snapshot"] = kwargs
        return str(tmp_path)

    class FakeApi:
        def model_info(self, **kwargs):
            captured["model_info"] = kwargs
            return type("Info", (), {"sha": "abc"})()

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download_spy)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_spy)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    assets.hf_download_file("repo", "file", "rev", tmp_path)
    assets.hf_snapshot("repo", "rev", tmp_path)
    assets.hf_model_info("repo", "rev")

    assert captured["download"]["token"] is None
    assert captured["snapshot"]["token"] is None
    assert captured["model_info"]["token"] is None
