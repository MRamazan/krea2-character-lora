import pytest

from krea2_character_lora.errors import ExportError
from krea2_character_lora.secrets import (
    assert_no_secret_fields,
    find_secret_field,
    normalized_field_is_secret,
)

ACCEPTED = [
    "keep_tokens",
    "shuffle_tokens",
    "token_dropout_rate",
    "max_text_length",
    "token_count",
    "caption_tokens",
    "num_tokens",
    "detokenizer",
]

REJECTED = [
    "token",
    "hf_token",
    "huggingface_token",
    "access_token",
    "auth_token",
    "api_key",
    "apikey",
    "password",
    "passwd",
    "secret",
    "client_secret",
    "authorization",
    "cookie",
    "github_access_token",
    "service_api_key",
    "admin_password",
]


@pytest.mark.parametrize("name", ACCEPTED)
def test_accepts_legitimate_fields(name):
    assert normalized_field_is_secret(name) is False


@pytest.mark.parametrize("name", REJECTED)
def test_rejects_secret_fields(name):
    assert normalized_field_is_secret(name) is True


def test_normalizes_hyphen_and_case():
    assert normalized_field_is_secret("HF-Token") is True
    assert normalized_field_is_secret("Keep-Tokens") is False


def test_find_secret_field_scans_dictionaries():
    payload = {"train": {"keep_tokens": 1, "settings": {"api_key": "x"}}}
    assert find_secret_field(payload) == "train.settings.api_key"


def test_find_secret_field_scans_lists():
    payload = {"items": [{"keep_tokens": 1}, {"client_secret": "x"}]}
    assert find_secret_field(payload) == "items[1].client_secret"


def test_find_secret_field_accepts_clean_payload():
    payload = {
        "training_config": {
            "keep_tokens": 1,
            "shuffle_tokens": False,
            "token_dropout_rate": 0.0,
            "max_text_length": 512,
        },
        "checkpoints": [{"step": 100, "token_count": 5}],
    }
    assert find_secret_field(payload) is None


def test_assert_no_secret_fields_raises_on_token():
    with pytest.raises(ExportError):
        assert_no_secret_fields({"nested": {"auth_token": "secret"}}, "manifest")


def test_assert_no_secret_fields_allows_realistic_training_manifest():
    manifest = {
        "run_name": "character_v1",
        "training_config": {
            "keep_tokens": 1,
            "shuffle_tokens": True,
            "token_dropout_rate": 0.1,
            "max_text_length": 512,
        },
    }
    assert_no_secret_fields(manifest, "run_manifest.json")
