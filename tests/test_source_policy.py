import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = [ROOT / "src", ROOT / "scripts", ROOT / "tests"]
NOTEBOOK_CELL_DIRECTORIES = [
    ROOT / "notebooks" / "cells",
    ROOT / "notebooks" / "evaluation_cells",
]
ALLOWED_NOTEBOOK_TITLES = {
    "# Setup",
    "# Dataset",
    "# Training",
    "# Evaluation and export",
    "# Upload and import evaluation bundle",
}


def comment_tokens(path: Path) -> list[str]:
    source = path.read_bytes()
    return [
        token.string
        for token in tokenize.tokenize(io.BytesIO(source).readline)
        if token.type == tokenize.COMMENT
    ]


def test_python_sources_contain_no_comments() -> None:
    violations = {}
    for root in PYTHON_ROOTS:
        for path in root.rglob("*.py"):
            comments = comment_tokens(path)
            if comments:
                violations[str(path.relative_to(ROOT))] = comments
    assert not violations


def test_notebook_comments_are_section_separators() -> None:
    violations = {}
    for directory in NOTEBOOK_CELL_DIRECTORIES:
        for path in directory.glob("*.py"):
            invalid = [
                value
                for value in comment_tokens(path)
                if not re.fullmatch(r"# ={20,}", value) and value not in ALLOWED_NOTEBOOK_TITLES
            ]
            if invalid:
                violations[str(path.relative_to(ROOT))] = invalid
    assert not violations
