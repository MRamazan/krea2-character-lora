# Claude Code Project Instructions

## Objective

Refactor the reference notebook into a production-quality, character-only Python package and a compact Google Colab notebook.

## Source of truth

The behavioral source is `reference/universal_pipeline_v2_1.ipynb`. The extracted code cells in `reference/extracted_cells` are navigation aids. Do not modify files under `reference`.

## Product constraints

- The project is entirely in English.
- Character LoRA training is the only supported workflow.
- Do not expose `concept_type`, `CONCEPT_TYPE`, style mode, object mode, product mode, or generic concept training.
- Do not add Google Drive integration.
- Do not add automatic best-checkpoint selection.
- Do not permanently merge LoRA weights into the base model.
- Preserve smoke testing, interrupted-run handling, resume behavior, checkpoint discovery, scale sweeps, export packaging, manifests, hashes, exact revision recording, and conservative completion claims.
- Preserve training on Krea 2 Raw and evaluation on Krea 2 Turbo unless the reference implementation proves a different requirement.
- Keep the text encoder frozen.
- Keep LoRA training transformer-only.

## Notebook contract

The generated notebook must contain exactly four executable code cells and no additional code cells:

1. Setup
2. Dataset
3. Training
4. Evaluation

The setup cell must visibly define the GitHub repository URL and revision, install the package, initialize the pipeline, install and verify the isolated AI Toolkit environment, and prepare the training assets, including Krea 2's default `Qwen/Qwen-Image` VAE. The pipeline uses no Hugging Face authentication; every configured repository is public and downloaded anonymously.

The dataset cell must visibly define `TRIGGER_WORD`, upload exactly one ZIP file, create the dataset configuration, validate image-caption pairs, audit trigger usage, audit duplicates, display a summary, display every image with its caption, and display detected issues.

The training cell must visibly define the run name and important training parameters, preview the resolved training configuration, run an optional smoke test, start or resume production training, display status, and display checkpoint inventory.

The evaluation cell must visibly define prompts, seeds, inference settings, checkpoint sweep settings, and LoRA scale settings. It must prepare inference assets, run base-versus-LoRA comparison, checkpoint sweep, scale sweep, display outputs, package a portable evaluation bundle, and optionally download it.

There is also a separate evaluation-only notebook, `notebooks/krea2_character_lora_evaluation_colab.ipynb`, built from `notebooks/evaluation_cells`, with exactly three executable code cells: setup, upload and import evaluation bundle, and evaluation and export. It calls `pipeline.import_evaluation_bundle(zip_path=...)` to import the portable ZIP produced by the main notebook, derives the trigger word from the imported run, downloads only public evaluation assets, and can export a new re-importable bundle. It uses no Hugging Face token and no training-asset download. Both notebooks are generated and checked by `scripts/build_notebook.py`.

The trigger word must originate in the dataset cell, persist in manifests, and be reused by training and evaluation without a second independent definition. In the evaluation-only notebook the trigger word originates from the imported run.

Notebook comments are allowed only as section separators. Python package code, scripts, and tests must contain no comment tokens.

## Architecture expectations

- Keep a small public API in `api.py` and export it through `__init__.py`.
- Keep configuration dataclasses in `configuration.py`.
- Keep runtime state durable through JSON manifests under the workspace.
- Keep Colab-specific upload and secret retrieval in the notebook when that improves readability.
- Keep heavy installation, subprocess, validation, model download, dataset audit, training, inference, and packaging logic in package modules.
- Do not depend on Python objects surviving a runtime restart. Runs and datasets must be reloadable from manifests.
- Raise focused exception types from `errors.py` with actionable English messages.
- Never log, serialize, hash, or export the Hugging Face token.

## Development rules

- Do not rewrite the pipeline from memory. Migrate behavior from the reference notebook.
- Prefer small, cohesive functions and explicit data structures.
- Avoid hidden global mutable state.
- Do not add abstractions that are not required by the notebook contract.
- Do not push to GitHub.
- Keep all tests passing.
- Rebuild the notebook after changing notebook cell sources.
- Run `ruff check .`, `pytest`, and `python scripts/build_notebook.py --check` before finishing.

## Completion report

At the end, report:

- Files changed
- Behavior preserved from the reference notebook
- Intentional behavior changes
- Tests run and their results
- Colab-only validation steps still required
- Suggested commit sequence
