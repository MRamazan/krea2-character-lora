import json
import zipfile
from pathlib import Path

import pytest

from krea2_character_lora import bundle
from krea2_character_lora.errors import (
    BundleValidationError,
    UnsupportedBundleVersionError,
)
from krea2_character_lora.types import TrainingRun


def _make_bundle(prepared_run, include_all_checkpoints=False, **kwargs):
    paths, run_name = prepared_run(**kwargs)
    export = bundle.create_evaluation_bundle(
        paths.root, run_name, include_all_checkpoints=include_all_checkpoints
    )
    return paths, run_name, export.archives[0]


def _entries(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        return {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
            if not info.filename.endswith("/")
        }


def _write_entries(dest, entries):
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return dest


def _rewrite_manifest(entries, mutate):
    manifest = json.loads(entries["bundle_manifest.json"].decode("utf-8"))
    mutate(manifest)
    entries = dict(entries)
    entries["bundle_manifest.json"] = json.dumps(manifest).encode("utf-8")
    return entries


def test_imports_valid_selected_checkpoint_bundle(prepared_run, tmp_path):
    _, run_name, zip_path = _make_bundle(prepared_run)
    imported = bundle.import_evaluation_bundle(tmp_path / "target", zip_path)
    assert imported.run_name == run_name
    assert imported.trigger_word == "mycharacter"
    assert Path(imported.details["active_checkpoint"]["checkpoint_path"]).is_file()
    assert imported.supports_checkpoint_sweep is False


def test_imports_valid_all_checkpoint_bundle(prepared_run, tmp_path):
    _, _, zip_path = _make_bundle(prepared_run, steps=(100, 200), include_all_checkpoints=True)
    imported = bundle.import_evaluation_bundle(tmp_path / "target", zip_path)
    assert imported.available_checkpoint_steps == [100, 200]
    assert imported.supports_checkpoint_sweep is True
    for record in imported.details["checkpoint_inventory"]["checkpoints"]:
        assert Path(record["path"]).is_file()


def test_imported_run_is_accepted_by_evaluate_type(prepared_run, tmp_path):
    _, _, zip_path = _make_bundle(prepared_run)
    imported = bundle.import_evaluation_bundle(tmp_path / "target", zip_path)
    assert isinstance(imported, TrainingRun)


def test_reconstructs_paths_relative_to_extraction_root(prepared_run, tmp_path):
    _, _, zip_path = _make_bundle(prepared_run, include_all_checkpoints=True)
    imported = bundle.import_evaluation_bundle(tmp_path / "target", zip_path)
    extraction = str(imported.extraction_directory)
    for record in imported.details["checkpoint_inventory"]["checkpoints"]:
        assert record["path"].startswith(extraction)


def test_rejects_tampered_file_content(prepared_run, tmp_path):
    _, _, zip_path = _make_bundle(prepared_run)
    entries = _entries(zip_path)
    entries["checkpoints/selected.safetensors"] = b"tampered-weights"
    tampered = _write_entries(tmp_path / "tampered.zip", entries)
    with pytest.raises(BundleValidationError):
        bundle.import_evaluation_bundle(tmp_path / "target", tampered)


def test_rejects_tampered_manifest_hash(prepared_run, tmp_path):
    _, _, zip_path = _make_bundle(prepared_run)
    entries = _entries(zip_path)

    def mutate(manifest):
        manifest["files"][0]["sha256"] = "0" * 64

    tampered = _write_entries(tmp_path / "tampered.zip", _rewrite_manifest(entries, mutate))
    with pytest.raises(BundleValidationError):
        bundle.import_evaluation_bundle(tmp_path / "target", tampered)


def test_rejects_missing_declared_file(prepared_run, tmp_path):
    _, _, zip_path = _make_bundle(prepared_run)
    entries = _entries(zip_path)
    del entries["checkpoints/selected.safetensors"]
    broken = _write_entries(tmp_path / "broken.zip", entries)
    with pytest.raises(BundleValidationError):
        bundle.import_evaluation_bundle(tmp_path / "target", broken)


def test_rejects_undeclared_file(prepared_run, tmp_path):
    _, _, zip_path = _make_bundle(prepared_run)
    entries = _entries(zip_path)
    entries["evil.py"] = b"print('code')"
    injected = _write_entries(tmp_path / "injected.zip", entries)
    with pytest.raises(BundleValidationError):
        bundle.import_evaluation_bundle(tmp_path / "target", injected)


def test_rejects_unsupported_format_version(prepared_run, tmp_path):
    _, _, zip_path = _make_bundle(prepared_run)
    entries = _rewrite_manifest(
        _entries(zip_path), lambda manifest: manifest.update({"bundle_format_version": 999})
    )
    future = _write_entries(tmp_path / "future.zip", entries)
    with pytest.raises(UnsupportedBundleVersionError):
        bundle.import_evaluation_bundle(tmp_path / "target", future)


def test_rejects_wrong_bundle_type(prepared_run, tmp_path):
    _, _, zip_path = _make_bundle(prepared_run)
    entries = _rewrite_manifest(
        _entries(zip_path), lambda manifest: manifest.update({"bundle_type": "malicious"})
    )
    wrong = _write_entries(tmp_path / "wrong.zip", entries)
    with pytest.raises(BundleValidationError):
        bundle.import_evaluation_bundle(tmp_path / "target", wrong)


def test_rejects_absolute_path(tmp_path):
    absolute = _write_entries(tmp_path / "absolute.zip", {"/etc/passwd": b"x"})
    with pytest.raises(BundleValidationError):
        bundle.import_evaluation_bundle(tmp_path / "target", absolute)


def test_rejects_parent_traversal(tmp_path):
    traversal = _write_entries(tmp_path / "traversal.zip", {"../evil.txt": b"x"})
    with pytest.raises(BundleValidationError):
        bundle.import_evaluation_bundle(tmp_path / "target", traversal)


def test_rejects_nested_zip_slip(tmp_path):
    slip = _write_entries(tmp_path / "slip.zip", {"a/../../evil.txt": b"x"})
    with pytest.raises(BundleValidationError):
        bundle.import_evaluation_bundle(tmp_path / "target", slip)


def test_rejects_symlink_entry(tmp_path):
    destination = tmp_path / "symlink.zip"
    with zipfile.ZipFile(destination, "w") as archive:
        info = zipfile.ZipInfo("link")
        info.external_attr = 0o120777 << 16
        archive.writestr(info, "/etc/passwd")
    with pytest.raises(BundleValidationError):
        bundle.import_evaluation_bundle(tmp_path / "target", destination)


def test_rejects_duplicate_normalized_paths(tmp_path):
    import warnings

    destination = tmp_path / "dup.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("bundle_manifest.json", "{}")
            archive.writestr("manifests/a.json", "{}")
            archive.writestr("manifests/a.json", "{}")
    with pytest.raises(BundleValidationError):
        bundle.import_evaluation_bundle(tmp_path / "target", destination)


def test_rejects_excessive_file_count(prepared_run, tmp_path, monkeypatch):
    _, _, zip_path = _make_bundle(prepared_run)
    monkeypatch.setattr(bundle, "MAX_BUNDLE_FILES", 1)
    with pytest.raises(BundleValidationError):
        bundle.import_evaluation_bundle(tmp_path / "target", zip_path)


def test_rejects_excessive_uncompressed_size(prepared_run, tmp_path, monkeypatch):
    _, _, zip_path = _make_bundle(prepared_run)
    monkeypatch.setattr(bundle, "MAX_BUNDLE_UNCOMPRESSED_BYTES", 1)
    with pytest.raises(BundleValidationError):
        bundle.import_evaluation_bundle(tmp_path / "target", zip_path)


def test_rejects_base_model_sized_selected_checkpoint(prepared_run, tmp_path, monkeypatch):
    _, _, zip_path = _make_bundle(prepared_run)
    monkeypatch.setattr(bundle, "MAX_LORA_FILE_BYTES", 8)
    with pytest.raises(BundleValidationError):
        bundle.import_evaluation_bundle(tmp_path / "target", zip_path)


def test_rejects_secret_like_manifest_field(prepared_run, tmp_path):
    _, _, zip_path = _make_bundle(prepared_run)
    entries = _rewrite_manifest(
        _entries(zip_path), lambda manifest: manifest.update({"api_key": "leak"})
    )
    leaky = _write_entries(tmp_path / "leaky.zip", entries)
    with pytest.raises(BundleValidationError):
        bundle.import_evaluation_bundle(tmp_path / "target", leaky)


def test_cleans_temporary_after_failed_import(prepared_run, tmp_path):
    _, _, zip_path = _make_bundle(prepared_run)
    entries = _rewrite_manifest(
        _entries(zip_path), lambda manifest: manifest.update({"bundle_type": "malicious"})
    )
    wrong = _write_entries(tmp_path / "wrong.zip", entries)
    target = tmp_path / "target"
    with pytest.raises(BundleValidationError):
        bundle.import_evaluation_bundle(target, wrong)
    leftovers = list((target / "runs").glob("_import_tmp_*"))
    assert leftovers == []


def test_does_not_import_uploaded_code(prepared_run, tmp_path):
    import sys

    _, _, zip_path = _make_bundle(prepared_run)
    before = set(sys.modules)
    bundle.import_evaluation_bundle(tmp_path / "target", zip_path)
    added = set(sys.modules) - before
    assert not any("imported_bundle" in name for name in added)
