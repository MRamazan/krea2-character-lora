from __future__ import annotations

from pathlib import Path

from .bundle import create_evaluation_bundle
from .secrets import assert_no_secret_fields, find_secret_field, normalized_field_is_secret
from .types import ExportBundle

_assert_no_secret_fields = assert_no_secret_fields

__all__ = [
    "assert_no_secret_fields",
    "create_evaluation_bundle",
    "find_secret_field",
    "normalized_field_is_secret",
    "package_run",
]


def package_run(
    workspace: str | Path,
    run_name: str,
    include_selected_lora: bool = True,
    include_all_checkpoints: bool = False,
    include_images: bool = True,
    include_logs: bool = True,
    include_manifests: bool = True,
) -> ExportBundle:
    return create_evaluation_bundle(
        workspace,
        run_name,
        include_selected_lora=include_selected_lora,
        include_all_checkpoints=include_all_checkpoints,
        include_images=include_images,
        include_logs=include_logs,
        include_manifests=include_manifests,
    )
