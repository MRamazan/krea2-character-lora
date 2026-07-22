# Validation Plan

## Local validation

Run:

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python scripts/build_notebook.py --check
```

Local tests must cover:

- Configuration validation
- Character-only public API
- Absence of concept selectors
- Source comment policy
- Exactly four notebook code cells
- Trigger ownership by the dataset cell
- Absence of all Hugging Face token handling
- Anonymous asset download calls
- Custom VAE file discovery, normalized layout, and strict rejection of incompatible weights
- Dataset manifest round trips
- Run manifest round trips
- Nested dataset pair discovery
- Repeated basename canonicalization
- Trigger audit behavior
- Duplicate inventory behavior
- Checkpoint discovery and conservative selection
- Export exclusion of secret values

## Colab validation

Use a small, valid character dataset and run:

1. Setup from a specific commit.
2. Dataset validation and complete gallery rendering.
3. Three-step smoke training.
4. Interrupted production training after one saved checkpoint.
5. Reload the runtime and restore the run from manifests.
6. Resume training.
7. Prepare Krea 2 Turbo.
8. Run base-versus-LoRA comparison.
9. Run a two-checkpoint sweep.
10. Run a two-value scale sweep.
11. Select a checkpoint manually.
12. Export and inspect archives.

Verify that the base parameter sample hashes remain unchanged and no adapter enters a merged state.
