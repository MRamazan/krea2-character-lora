from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import ExportError

SECRET_FIELD_NAMES = frozenset(
    {
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
    }
)

SECRET_FIELD_SUFFIXES = (
    "_access_token",
    "_auth_token",
    "_api_key",
    "_password",
    "_passwd",
    "_client_secret",
)


def normalize_field_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def normalized_field_is_secret(name: str) -> bool:
    normalized = normalize_field_name(name)
    if normalized in SECRET_FIELD_NAMES:
        return True
    return any(normalized.endswith(suffix) for suffix in SECRET_FIELD_SUFFIXES)


def find_secret_field(payload: Any, prefix: str = "") -> str | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            location = f"{prefix}.{key_text}" if prefix else key_text
            if normalized_field_is_secret(key_text):
                return location
            found = find_secret_field(value, location)
            if found is not None:
                return found
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            found = find_secret_field(item, f"{prefix}[{index}]")
            if found is not None:
                return found
    return None


def assert_no_secret_fields(payload: Any, source: str | Path) -> None:
    location = find_secret_field(payload)
    if location is not None:
        raise ExportError(f"A secret-like field '{location}' was found in {source}.")
