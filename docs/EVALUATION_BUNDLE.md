# Portable Evaluation Bundle

The portable evaluation bundle is a single self-contained ZIP produced by the main
notebook's export flow and consumed by the evaluation-only notebook. It lets a trained
character LoRA be evaluated on a fresh Colab runtime without repeating training.

## Workflows

Training workflow:

```text
Main notebook
→ train character LoRA
→ evaluate
→ create portable evaluation ZIP
```

Evaluation-only workflow:

```text
New evaluation notebook
→ upload portable ZIP
→ validate and import LoRA
→ download public evaluation assets
→ run base / checkpoint / scale evaluation
→ optionally export a new portable ZIP
```

## Public API

```python
exports = evaluation.export(
    include_selected_lora=True,
    include_all_checkpoints=False,
    include_images=True,
    include_logs=True,
    include_manifests=True,
)
exports.display()

imported_run = pipeline.import_evaluation_bundle(zip_path=EVALUATION_BUNDLE_ZIP)
evaluation = pipeline.evaluate(run=imported_run, config=evaluation_config)
```

The imported object is an `ImportedRun`, a subclass of `TrainingRun`. It exposes
`run_name`, `trigger_word`, `selected_checkpoint_step`, `available_checkpoint_steps`,
`lora_rank`, `lora_alpha`, `provenance`, `capabilities`, `bundle_source_path`,
`extraction_directory`, and `display_summary()`.

## Root manifest

The bundle stores relative POSIX paths and a root `bundle_manifest.json`:

```json
{
  "bundle_type": "krea2_character_lora_evaluation_bundle",
  "bundle_format_version": 1,
  "package_version": "0.1.0",
  "run_name": "character_v1",
  "trigger_word": "mycharacter",
  "created_at": "2026-07-23T00:00:00+00:00",
  "selected_checkpoint_step": 2000,
  "available_checkpoint_steps": [2000],
  "lora_rank": 32,
  "lora_alpha": 32,
  "source_git_revision": null,
  "provenance": {
    "training_model": {"repository": "krea/Krea-2-Raw", "revision": "..."},
    "evaluation_model": {"repository": "krea/Krea-2-Turbo", "revision": "..."},
    "text_encoder": {"repository": "Qwen/Qwen3-VL-4B-Instruct", "revision": "..."},
    "vae": {
      "repository": "Qwen/Qwen-Image",
      "revision": "...",
      "weights_filename": "diffusion_pytorch_model.safetensors",
      "weights_sha256": "..."
    },
    "selected_lora_sha256": "..."
  },
  "capabilities": {
    "selected_lora": true,
    "all_checkpoints": false,
    "checkpoint_sweep": false,
    "previous_evaluation": true,
    "evaluation_images": true,
    "logs": true
  },
  "files": [
    {"path": "checkpoints/selected.safetensors", "size_bytes": 123, "sha256": "..."}
  ]
}
```

## Contents

Included when applicable: the selected LoRA (`checkpoints/selected.safetensors`),
optionally all LoRA checkpoints (`checkpoints/step_XXXXXXXX.safetensors`), normalized
run manifest, training configuration, checkpoint inventory, active checkpoint selection,
dataset metadata, dataset fingerprint, generated AI Toolkit YAML, evaluation
configuration and the previous evaluation manifest when present, evaluation images and
logs when requested, and a per-file size and SHA-256 index.

Never included: Krea base model weights, text encoder weights, VAE weights, Hugging Face
cache files, dataset images, secrets, credentials, absolute operational paths, or
arbitrary workspace files.

## Creation safety

The bundle is staged in a temporary directory. Before the final ZIP is written the
exporter validates all requested files, computes hashes, rejects a missing selected
LoRA, rejects duplicate archive paths, rejects any file larger than the LoRA size limit,
runs the recursive secret-field scan, writes the manifest last, and validates the
completed bundle. Archive paths and file ordering are deterministic.

## Import validation

The importer accepts exactly one ZIP and extracts it into an isolated directory. It
rejects path traversal, absolute paths, `..` components, symlinks and special files,
excessive file counts, excessive uncompressed size, and duplicate normalized paths. It
validates the bundle type and format version, verifies every declared file's existence,
size, and SHA-256, rejects undeclared files, rejects secret-like manifest fields,
inspects the selected LoRA as safetensors, and rejects base-model-sized weights. It never
executes or imports code from the bundle and never trusts absolute paths embedded in
source manifests. The extracted bundle is preserved after a successful import and removed
after a failed one. The checkpoint inventory is reconstructed from validated bundle paths
relative to the extraction root.

## Checkpoint capabilities

- `include_all_checkpoints=True` produces a bundle whose imported run supports the normal
  checkpoint sweep across all bundled checkpoints.
- `include_all_checkpoints=False` produces a selected-checkpoint-only bundle. Its imported
  run still supports base comparison and scale sweep. A requested checkpoint sweep
  evaluates the single selected checkpoint with a clear notice rather than pretending
  multiple checkpoints exist.

All-checkpoint evaluation therefore requires the source bundle to have been exported with
`include_all_checkpoints=True`.

## Base-in-grid comparison

When `INCLUDE_BASE_IN_CHECKPOINT_GRID` (config `include_base_in_checkpoint_grid`) is
enabled, the checkpoint sweep generates one base image per prompt and seed once, with all
adapter scales set to zero, using the same evaluation model, VAE, dimensions, inference
steps, guidance, and negative prompt. That image is prepended as the first `Base` column
of each checkpoint grid and recorded in sweep metadata with `checkpoint_step: null`. The
standalone base-versus-LoRA comparison remains available separately.

## Integration testing

Real GPU behavior (Krea 2 Turbo loading, VAE loading, image generation, and the base
column render) can only be validated on a Colab GPU. Locally, bundle creation and import
validation, path reconstruction, hashing, secret detection, capability flags, and the
evaluation request contract are covered by the unit tests. Re-exports produced by the
evaluation-only notebook use the same bundle format and are re-importable; the original
uploaded ZIP is never nested inside a re-export.
