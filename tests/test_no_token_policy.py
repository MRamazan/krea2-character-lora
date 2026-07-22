from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = [
    ROOT / "src",
    ROOT / "notebooks" / "cells",
    ROOT / "notebooks" / "evaluation_cells",
]
EXCLUDED = {ROOT / "src" / "krea2_character_lora" / "secrets.py"}

FORBIDDEN_TOKEN_MARKERS = (
    "hf_token",
    "HF_TOKEN",
    "getpass",
    "colab import userdata",
    "userdata.get",
    "whoami",
    "HfFolder",
    "huggingface_hub.login",
)


def test_no_hugging_face_token_handling_in_sources() -> None:
    violations = {}
    for root in SCANNED:
        for path in root.rglob("*.py"):
            if path in EXCLUDED:
                continue
            source = path.read_text(encoding="utf-8")
            hits = [marker for marker in FORBIDDEN_TOKEN_MARKERS if marker in source]
            if hits:
                violations[str(path.relative_to(ROOT))] = hits
    assert not violations


def test_authentication_module_is_removed() -> None:
    assert not (ROOT / "src" / "krea2_character_lora" / "authentication.py").exists()


def test_downloads_pass_token_none() -> None:
    assets_source = (ROOT / "src" / "krea2_character_lora" / "assets.py").read_text(
        encoding="utf-8"
    )
    assert assets_source.count("token=None") >= 4
