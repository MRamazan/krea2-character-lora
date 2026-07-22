# Architecture

## Public surface

The notebook imports only these public names:

- `CharacterLoraPipeline`
- `DatasetConfig`
- `TrainingConfig`
- `EvaluationConfig`

The pipeline returns durable report objects:

- `SetupReport`
- `DatasetResult`
- `TrainingRun`
- `EvaluationReport`
- `ExportBundle`
- `ImportedRun`

The pipeline also exposes `import_evaluation_bundle(zip_path=...)`, which returns an
`ImportedRun` (a `TrainingRun` subclass) accepted directly by `evaluate(run=..., config=...)`.
Bundle validation errors are `BundleValidationError`, `BundleImportError`,
`UnsupportedBundleVersionError`, and `BundleIntegrityError`.

## Module responsibilities

### `api.py`

Coordinates public operations and keeps notebook calls readable.

### `configuration.py`

Contains validated character-only configuration dataclasses. It contains no concept selector.

### `paths.py`

Defines canonical workspace paths and creates required directories.

### `environment.py`

Verifies Colab hardware, installs or repairs the isolated environment, resolves AI Toolkit and Diffusers revisions, and verifies source compatibility.

### `assets.py`

Resolves and anonymously downloads training and inference assets, hashes and records them, and normalizes plus strictly validates the custom `AutoencoderKLQwenImage` VAE without any fallback.

### `isolated/`

Self-contained scripts executed by the isolated AI Toolkit interpreter for GPU-bound work: source preflight, custom VAE strict validation, the non-destructive Krea 2 runtime helper, and the base, checkpoint, and scale evaluation passes. These modules are never imported by the outer package.

### `dataset.py`

Extracts ZIP files, canonicalizes nested paths, validates pairs, audits triggers, fingerprints data, finds duplicates, builds galleries, and writes dataset manifests.

### `training.py`

Builds AI Toolkit configurations, performs preflight checks, runs smoke and production training, supports resume, and writes run state.

### `checkpoints.py`

Discovers actual checkpoints, validates metadata, selects defaults conservatively, and applies explicit manual selections.

### `runtime.py`

Contains non-destructive Krea 2 Turbo adapter loading and scale control used by evaluation subprocesses.

### `evaluation.py`

Runs deterministic base comparisons, checkpoint sweeps, and scale sweeps while preserving base weights.

### `export.py`

Thin wrapper that packages the run into a portable evaluation bundle and preserves the
backward-compatible `package_run` entry point.

### `bundle.py`

Creates the portable evaluation bundle and imports it: staging, hashing, capability
detection, manifest normalization to relative POSIX paths, strict import validation, and
reconstruction of the checkpoint inventory relative to the extraction root.

### `secrets.py`

Normalized exact-name and sensitive-suffix secret-field detection used by both bundle
creation and bundle import. Legitimate fields such as `keep_tokens`, `shuffle_tokens`,
and `token_dropout_rate` are accepted.

### `manifests.py`

Reads and writes atomic JSON state with stable schemas.

### `errors.py`

Defines focused user-facing exception types.

## Durable state

The default workspace is `/content/krea2_character_lora`.

```text
workspace/
├── assets/
├── config/
├── datasets/
├── environments/
├── exports/
├── inference/
├── logs/
├── runtime_helpers/
└── runs/
```

Expected manifests include:

```text
config/setup_manifest.json
config/training_asset_manifest.json
config/inference_asset_manifest.json
datasets/active/dataset_manifest.json
datasets/active/dataset_fingerprint.json
runs/<run_name>/run_manifest.json
runs/<run_name>/checkpoint_inventory.json
runs/<run_name>/active_checkpoint.json
runs/<run_name>/evaluation_manifest.json
```

All public reload operations use manifests instead of relying on surviving notebook variables.

## Dependency boundary

The outer package remains lightweight. AI Toolkit, PyTorch, Krea 2 integration dependencies, and model execution run inside the isolated environment created under the workspace. The package communicates with heavy runtime tasks through explicit subprocess commands, environment variables, files, structured JSON output, and logs.
