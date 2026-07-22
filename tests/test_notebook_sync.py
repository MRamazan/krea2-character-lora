from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
CELL_DIRECTORY = ROOT / "notebooks" / "cells"
NOTEBOOK = ROOT / "notebooks" / "krea2_character_lora_colab.ipynb"


def test_notebook_matches_cell_templates() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    templates = [
        path.read_text(encoding="utf-8").rstrip() for path in sorted(CELL_DIRECTORY.glob("*.py"))
    ]
    assert [cell.rstrip() for cell in code_cells] == templates


def test_cell_directory_defines_four_cells() -> None:
    assert len(list(CELL_DIRECTORY.glob("*.py"))) == 4
