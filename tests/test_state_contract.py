from krea2_character_lora.manifests import read_json, write_json_atomic


def test_atomic_manifest_round_trip(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    payload = {"training_kind": "character_lora", "trigger_word": "mycharacter"}
    write_json_atomic(path, payload)
    assert read_json(path) == payload
