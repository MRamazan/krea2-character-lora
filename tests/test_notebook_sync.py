from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = [
    (ROOT / "notebooks" / "cells", ROOT / "notebooks" / "krea2_character_lora_colab.ipynb", 4),
    (
        ROOT / "notebooks" / "evaluation_cells",
        ROOT / "notebooks" / "krea2_character_lora_evaluation_colab.ipynb",
        3,
    ),
]


@pytest.mark.parametrize(("cell_directory", "notebook_path", "expected"), NOTEBOOKS)
def test_notebook_matches_cell_templates(cell_directory, notebook_path, expected) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    templates = [
        path.read_text(encoding="utf-8").rstrip() for path in sorted(cell_directory.glob("*.py"))
    ]
    assert [cell.rstrip() for cell in code_cells] == templates
    assert len(code_cells) == expected


@pytest.mark.parametrize(("cell_directory", "notebook_path", "expected"), NOTEBOOKS)
def test_cell_directory_defines_expected_cells(cell_directory, notebook_path, expected) -> None:
    assert len(list(cell_directory.glob("*.py"))) == expected
