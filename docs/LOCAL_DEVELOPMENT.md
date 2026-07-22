# Local Development and GitHub Flow

## Initial setup

```bash
git init
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Open the repository root in Claude Code. Claude Code must read `CLAUDE.md` automatically. Paste `INITIAL_PROMPT.md` as the first task.

## Working loop

1. Change package code and tests locally.
2. Run `ruff check .` and `pytest`.
3. Rebuild the notebook with `python scripts/build_notebook.py`.
4. Run `python scripts/build_notebook.py --check`.
5. Commit locally.
6. Push to GitHub.
7. Point Colab to the exact commit and run the smoke path.
8. Bring Colab logs back to the local repository when runtime-only issues appear.

## Suggested branches

- `main` for stable code
- `refactor/character-pipeline` for the initial migration
- `fix/colab-<issue>` for runtime-specific corrections

## Suggested release sequence

- `v0.1.0-alpha.1` after local tests pass
- `v0.1.0-beta.1` after full Colab smoke validation
- `v0.1.0` after completed training, evaluation, resume, and export validation

## Colab install strategy

During development:

```python
PIPELINE_REVISION = "<exact-commit-sha>"
```

For releases:

```python
PIPELINE_REVISION = "v0.1.0"
```

An exact commit is the strongest reproducibility boundary. A tag is easier to read but should remain immutable.
