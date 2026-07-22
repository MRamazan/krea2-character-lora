import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from krea2_character_lora import assets
from krea2_character_lora.errors import VaeValidationError
from krea2_character_lora.paths import ProjectPaths

DATA = Path(__file__).parent / "data"
ORIGINAL_SHAPES = json.loads(
    (DATA / "krea2realvae_original_namespace.json").read_text(encoding="utf-8")
)["shapes"]
DIFFUSERS_SHAPES = json.loads(
    (DATA / "qwen_image_diffusers_namespace.json").read_text(encoding="utf-8")
)["shapes"]

QWEN_CONFIG = {
    "_class_name": "AutoencoderKLQwenImage",
    "base_dim": 96,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "temperal_downsample": [False, True, True],
    "z_dim": 16,
}


def _tiny(shapes):
    return {name: [2] for name in shapes}


def _save_tiny(path, shapes):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: np.zeros((2,), dtype="float32") for name in shapes},
        str(path),
        metadata={"format": "pt"},
    )


def test_select_vae_safetensors_prefers_diffusers_name():
    files = ["readme.md", "krea2RealVae_v10.safetensors"]
    assert assets.select_vae_safetensors(files) == "krea2RealVae_v10.safetensors"


def test_select_vae_safetensors_requires_a_file():
    with pytest.raises(assets.AssetError):
        assets.select_vae_safetensors(["config.json"])


def test_detect_vae_format_identifies_original_namespace():
    assert assets.detect_vae_format(list(ORIGINAL_SHAPES)) == "qwen_image_original_vae"


def test_detect_vae_format_identifies_diffusers_namespace():
    assert assets.detect_vae_format(list(DIFFUSERS_SHAPES)) == "diffusers_autoencoder_kl_qwen_image"


def test_detect_vae_format_reports_unknown():
    assert assets.detect_vae_format(["foo.weight", "bar.bias"]) == "unknown"


def test_original_to_diffusers_conversion_is_exact_bijection():
    converted = {
        assets.convert_original_vae_key(name): shape for name, shape in ORIGINAL_SHAPES.items()
    }
    assert len(converted) == len(ORIGINAL_SHAPES)
    assert set(converted) == set(DIFFUSERS_SHAPES)
    for key, shape in converted.items():
        assert shape == DIFFUSERS_SHAPES[key]


def test_conversion_maps_quant_convs():
    assert assets.convert_original_vae_key("conv1.weight") == "quant_conv.weight"
    assert assets.convert_original_vae_key("conv2.bias") == "post_quant_conv.bias"
    assert assets.convert_original_vae_key("decoder.conv1.weight") == "decoder.conv_in.weight"
    assert (
        assets.convert_original_vae_key("decoder.upsamples.3.time_conv.weight")
        == "decoder.up_blocks.0.upsamplers.0.time_conv.weight"
    )
    assert (
        assets.convert_original_vae_key("decoder.upsamples.4.shortcut.weight")
        == "decoder.up_blocks.1.resnets.0.conv_shortcut.weight"
    )


def test_plan_original_format_is_compatible():
    plan = assets.plan_vae_normalization(ORIGINAL_SHAPES, DIFFUSERS_SHAPES)
    assert plan["detected_format"] == "qwen_image_original_vae"
    assert plan["conversion"] == "wan_qwen_original_to_diffusers"
    assert plan["compatible"] is True
    assert plan["converted_key_count"] == len(DIFFUSERS_SHAPES)
    assert not any(plan["analysis"].values())


def test_plan_rejects_missing_key():
    broken = dict(ORIGINAL_SHAPES)
    broken.pop("decoder.conv1.weight")
    plan = assets.plan_vae_normalization(broken, DIFFUSERS_SHAPES)
    assert plan["compatible"] is False
    assert "decoder.conv_in.weight" in plan["analysis"]["missing_keys"]


def test_plan_rejects_unexpected_key():
    extended = dict(ORIGINAL_SHAPES)
    extended["conv1.rogue.weight"] = [2]
    plan = assets.plan_vae_normalization(extended, DIFFUSERS_SHAPES)
    assert plan["compatible"] is False
    assert "quant_conv.rogue.weight" in plan["analysis"]["unexpected_keys"]


def test_plan_rejects_unknown_format():
    plan = assets.plan_vae_normalization({"foo.weight": [2]}, DIFFUSERS_SHAPES)
    assert plan["detected_format"] == "unknown"
    assert plan["compatible"] is False


def test_detect_removable_prefix():
    prefixed = [f"vae.{name}" for name in ORIGINAL_SHAPES]
    assert assets.detect_removable_prefix(prefixed, list(DIFFUSERS_SHAPES)) is True
    assert assets.detect_removable_prefix(list(ORIGINAL_SHAPES), list(DIFFUSERS_SHAPES)) is False


def test_plan_handles_removable_prefix_on_original_format():
    prefixed = {f"vae.{name}": shape for name, shape in ORIGINAL_SHAPES.items()}
    plan = assets.plan_vae_normalization(prefixed, DIFFUSERS_SHAPES)
    assert plan["remove_prefix"] is True
    assert plan["detected_format"] == "qwen_image_original_vae"
    assert plan["compatible"] is True


def test_analyze_state_dict_reports_differences():
    custom = {"a": [2, 2], "b": [4], "c": [1]}
    reference = {"a": [2, 2], "b": [8], "d": [1]}
    analysis = assets.analyze_state_dict(custom, reference)
    assert analysis["missing_keys"] == ["d"]
    assert analysis["unexpected_keys"] == ["c"]
    assert analysis["shape_mismatches"] == ["b: [4] != [8]"]


def test_read_safetensors_header_returns_shapes(tmp_path):
    path = tmp_path / "vae.safetensors"
    _save_tiny(path, {"encoder.conv_in.weight": [2]})
    shapes = assets.state_dict_shapes(path)
    assert shapes["encoder.conv_in.weight"] == [2]


def _install_download_mocks(monkeypatch, custom_shapes):
    def fake_resolve(repo_id, revision=None):
        return f"{repo_id.replace('/', '_')}_rev"

    def fake_list(repo_id, revision=None):
        return ["config.json", "krea2RealVae_v10.safetensors"]

    def fake_download(repo_id, filename, revision, local_dir, subfolder=None):
        target = local_dir / filename
        _save_tiny(target, custom_shapes)
        return target

    def fake_snapshot(repo_id, revision, local_dir, allow_patterns=None):
        vae_directory = local_dir / "vae"
        vae_directory.mkdir(parents=True, exist_ok=True)
        (vae_directory / "config.json").write_text(json.dumps(QWEN_CONFIG), encoding="utf-8")
        _save_tiny(vae_directory / "diffusion_pytorch_model.safetensors", DIFFUSERS_SHAPES)
        return local_dir

    monkeypatch.setattr(assets, "resolve_revision", fake_resolve)
    monkeypatch.setattr(assets, "hf_list_repo_files", fake_list)
    monkeypatch.setattr(assets, "hf_download_file", fake_download)
    monkeypatch.setattr(assets, "hf_snapshot", fake_snapshot)


def test_prepare_custom_vae_converts_original_format(monkeypatch, tmp_path):
    paths = ProjectPaths.create(tmp_path / "workspace")
    _install_download_mocks(monkeypatch, ORIGINAL_SHAPES)
    manifest = assets.prepare_custom_vae(paths)
    assert manifest["detected_format"] == "qwen_image_original_vae"
    assert manifest["conversion"] == "wan_qwen_original_to_diffusers"
    assert manifest["static_compatibility"] is True
    assert manifest["converted_key_count"] == len(DIFFUSERS_SHAPES)
    assert manifest["strict_validation"] == "pending"
    normalized = paths.models / "krea2_normalized_vae"
    assert (normalized / "config.json").is_file()
    mapping_document = json.loads((normalized / "vae_key_mapping.json").read_text(encoding="utf-8"))
    assert mapping_document["conversion"] == "wan_qwen_original_to_diffusers"
    assert len(mapping_document["mapping"]) == len(ORIGINAL_SHAPES)
    assert mapping_document["mapping"]["conv1.weight"] == "quant_conv.weight"


def test_prepare_custom_vae_rejects_incompatible_without_fallback(monkeypatch, tmp_path):
    paths = ProjectPaths.create(tmp_path / "workspace")
    extended = dict(ORIGINAL_SHAPES)
    extended["conv1.rogue.weight"] = [2]
    _install_download_mocks(monkeypatch, extended)
    with pytest.raises(VaeValidationError):
        assets.prepare_custom_vae(paths)
    normalized = paths.models / "krea2_normalized_vae"
    assert not (normalized / "config.json").is_file()
    assert (paths.logs / "custom_vae_validation.json").is_file()


def test_prepare_custom_vae_rejects_unknown_format(monkeypatch, tmp_path):
    paths = ProjectPaths.create(tmp_path / "workspace")
    _install_download_mocks(monkeypatch, {"foo.weight": [2], "bar.bias": [2]})
    with pytest.raises(VaeValidationError):
        assets.prepare_custom_vae(paths)


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
