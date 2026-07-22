import pytest

from krea2_character_lora.configuration import DatasetConfig
from krea2_character_lora.dataset import load_dataset, prepare_dataset
from krea2_character_lora.errors import DatasetError
from krea2_character_lora.paths import ProjectPaths


def _paths(tmp_path):
    return ProjectPaths.create(tmp_path / "workspace")


def _pairs(count, trigger="mycharacter"):
    return [{"name": f"image_{index:03d}", "seed": index} for index in range(count)]


def test_prepare_dataset_discovers_nested_pairs(tmp_path, make_dataset_zip):
    archive = make_dataset_zip(_pairs(6), nested=True)
    config = DatasetConfig(trigger_word="mycharacter", minimum_pair_count=4)
    result = prepare_dataset(archive, config, _paths(tmp_path))
    assert result.pair_count == 6
    assert result.training_directory.is_dir()
    images = sorted(result.training_directory.glob("*.png"))
    captions = sorted(result.training_directory.glob("*.txt"))
    assert len(images) == 6
    assert len(captions) == 6


def test_prepare_dataset_repeated_basenames_are_canonicalized(tmp_path, make_dataset_zip):
    pairs = [{"name": "portrait", "seed": 1}, {"name": "portrait", "seed": 2}]
    archive = make_dataset_zip(pairs, nested=True)
    config = DatasetConfig(trigger_word="mycharacter", minimum_pair_count=2)
    result = prepare_dataset(archive, config, _paths(tmp_path))
    assert result.pair_count == 2
    identifiers = {record["canonical_id"] for record in result.details["records"]}
    assert identifiers == {"000001", "000002"}


def test_missing_caption_is_fatal(tmp_path, make_dataset_zip):
    archive = make_dataset_zip(_pairs(4))
    paths = _paths(tmp_path)
    config = DatasetConfig(trigger_word="mycharacter", minimum_pair_count=2)
    prepared = prepare_dataset(archive, config, paths)
    orphan = prepared.training_directory
    assert orphan.exists()

    pairs = _pairs(4)
    archive2 = make_dataset_zip(pairs)
    import zipfile

    with zipfile.ZipFile(archive2, "a") as handle:
        handle.writestr("only_image.png", b"not-a-real-image")
    with pytest.raises(DatasetError):
        prepare_dataset(archive2, config, _paths(tmp_path / "second"))


def test_orphan_caption_is_fatal(tmp_path, make_dataset_zip):
    archive = make_dataset_zip(_pairs(4))
    import zipfile

    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr("group_extra/orphan.txt", "mycharacter orphan caption")
    config = DatasetConfig(trigger_word="mycharacter", minimum_pair_count=2)
    with pytest.raises(DatasetError):
        prepare_dataset(archive, config, _paths(tmp_path))


def test_empty_caption_is_fatal(tmp_path, make_dataset_zip):
    pairs = _pairs(4)
    pairs[0]["caption"] = ""
    archive = make_dataset_zip(pairs)
    config = DatasetConfig(trigger_word="mycharacter", minimum_pair_count=2)
    with pytest.raises(DatasetError):
        prepare_dataset(archive, config, _paths(tmp_path))


def test_trigger_policy_require_rejects_missing_trigger(tmp_path, make_dataset_zip):
    pairs = [{"name": f"image_{index}", "caption": "portrait without token"} for index in range(4)]
    archive = make_dataset_zip(pairs)
    config = DatasetConfig(
        trigger_word="mycharacter", caption_trigger_policy="require", minimum_pair_count=2
    )
    with pytest.raises(DatasetError):
        prepare_dataset(archive, config, _paths(tmp_path))


def test_trigger_policy_warn_allows_missing_trigger(tmp_path, make_dataset_zip):
    pairs = [{"name": f"image_{index}", "caption": "portrait without token"} for index in range(4)]
    archive = make_dataset_zip(pairs)
    config = DatasetConfig(
        trigger_word="mycharacter", caption_trigger_policy="warn", minimum_pair_count=2
    )
    result = prepare_dataset(archive, config, _paths(tmp_path))
    assert result.details["trigger_audit"]["status"] == "warning"


def test_auto_prefix_creates_backups_and_modifies_captions(tmp_path, make_dataset_zip):
    pairs = [{"name": f"image_{index}", "caption": "portrait without token"} for index in range(4)]
    archive = make_dataset_zip(pairs)
    paths = _paths(tmp_path)
    config = DatasetConfig(
        trigger_word="mycharacter",
        caption_trigger_policy="require",
        auto_prefix_missing_trigger=True,
        minimum_pair_count=2,
    )
    result = prepare_dataset(archive, config, paths)
    audit = result.details["trigger_audit"]
    assert audit["status"] == "passed"
    assert len(audit["modified_captions"]) == 4
    backup_directory = paths.dataset_active / "caption_backups"
    assert backup_directory.is_dir()
    assert len(list(backup_directory.glob("*.txt"))) == 4
    for record in result.details["records"]:
        assert record["text"].startswith("mycharacter")


def test_captions_are_not_modified_without_auto_prefix(tmp_path, make_dataset_zip):
    pairs = [{"name": f"image_{index}", "caption": "mycharacter present"} for index in range(4)]
    archive = make_dataset_zip(pairs)
    paths = _paths(tmp_path)
    config = DatasetConfig(trigger_word="mycharacter", minimum_pair_count=2)
    result = prepare_dataset(archive, config, paths)
    assert not (paths.dataset_active / "caption_backups").exists()
    for record in result.details["records"]:
        assert record["text"] == "mycharacter present"


def test_exact_duplicates_are_fatal_when_configured(tmp_path, make_dataset_zip, solid_image_writer):
    import zipfile

    staging = tmp_path / "dupes"
    staging.mkdir()
    for index in range(4):
        solid_image_writer(staging / f"image_{index}.png", (64, 64), (10, 20, 30))
        (staging / f"image_{index}.txt").write_text("mycharacter", encoding="utf-8")
    archive = tmp_path / "dupes.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(staging.iterdir()):
            handle.write(path, arcname=path.name)
    config = DatasetConfig(
        trigger_word="mycharacter", minimum_pair_count=2, fail_on_exact_duplicates=True
    )
    with pytest.raises(DatasetError):
        prepare_dataset(archive, config, _paths(tmp_path))


def test_near_duplicates_are_reported_as_warning(tmp_path, solid_image_writer):
    import zipfile

    staging = tmp_path / "near"
    staging.mkdir()
    colors = [(10, 20, 30), (11, 21, 31), (200, 10, 10), (10, 200, 10)]
    for index, color in enumerate(colors):
        solid_image_writer(staging / f"image_{index}.png", (64, 64), color)
        (staging / f"image_{index}.txt").write_text("mycharacter", encoding="utf-8")
    archive = tmp_path / "near.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(staging.iterdir()):
            handle.write(path, arcname=path.name)
    config = DatasetConfig(
        trigger_word="mycharacter",
        minimum_pair_count=2,
        fail_on_exact_duplicates=False,
        near_duplicate_hamming_threshold=10,
    )
    result = prepare_dataset(archive, config, _paths(tmp_path))
    kinds = {issue["kind"] for issue in result.details["issues"]}
    assert "near_duplicate" in kinds


def test_fingerprint_is_deterministic(tmp_path, make_dataset_zip):
    archive = make_dataset_zip(_pairs(4))
    config = DatasetConfig(trigger_word="mycharacter", minimum_pair_count=2)
    first = prepare_dataset(archive, config, _paths(tmp_path / "a"))
    second = prepare_dataset(archive, config, _paths(tmp_path / "b"))
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_dataset_reloads_from_manifest(tmp_path, make_dataset_zip):
    archive = make_dataset_zip(_pairs(5))
    paths = _paths(tmp_path)
    config = DatasetConfig(trigger_word="mycharacter", minimum_pair_count=2)
    prepared = prepare_dataset(archive, config, paths)
    reloaded = load_dataset(paths)
    assert reloaded.trigger_word == "mycharacter"
    assert reloaded.pair_count == prepared.pair_count
    assert reloaded.fingerprint == prepared.fingerprint
