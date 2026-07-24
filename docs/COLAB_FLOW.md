# Colab Flow

## Cell 1: Setup

The user edits:

- `REPOSITORY_URL`
- `PIPELINE_REVISION`
- `WORKSPACE`

The cell installs the package, initializes the pipeline, installs or repairs the isolated runtime, verifies source compatibility, and prepares the training assets, including Krea 2's default `Qwen/Qwen-Image` VAE. It uses no Hugging Face token; every repository is downloaded anonymously.

## Cell 2: Dataset

The user edits:

- `TRIGGER_WORD`
- Trigger policy
- Duplicate thresholds
- Gallery layout

The cell uploads one ZIP file, validates the dataset, writes manifests, displays the complete gallery, displays caption audit information, and displays issues.

## Cell 3: Training

The user edits:

- Run name
- Steps
- Learning rate
- Batch and accumulation
- Resolutions
- Rank and alpha
- Save cadence
- Dataset behavior
- Smoke test and resume behavior

The cell previews the resolved configuration, runs smoke testing, starts or resumes training, and displays checkpoint state.

## Cell 4: Evaluation

The user edits:

- Prompts
- Seeds
- Dimensions
- Inference steps and guidance
- Checkpoint sweep mode
- Scale sweep values
- Export and download options

The cell prepares Krea 2 Turbo, evaluates the run, displays all comparison grids, packages a portable evaluation bundle, and optionally downloads it.

## Evaluation-only notebook

`notebooks/krea2_character_lora_evaluation_colab.ipynb` has exactly three cells.

### Cell 1: Setup

Installs the package from an explicit revision, initializes `CharacterLoraPipeline`, and
verifies the isolated runtime. It does not download training assets and uses no Hugging
Face token. Krea 2 Turbo, the text encoder, and the VAE are prepared in cell three.

### Cell 2: Upload and import evaluation bundle

Uploads exactly one portable ZIP and calls `pipeline.import_evaluation_bundle(zip_path=...)`.
`imported_run.display_summary()` shows the run name, trigger word, selected checkpoint
step, available checkpoint steps, LoRA rank and alpha, bundle format version, training
model and VAE provenance, the selected LoRA path and SHA-256, checkpoint-sweep
support, and the extraction directory.

### Cell 3: Evaluation and export

Sets `TRIGGER_WORD = imported_run.trigger_word`, exposes prompts, seeds, dimensions,
inference settings, checkpoint mode, scale sweep, base comparison and base-in-grid
options, and export options. It prepares evaluation assets anonymously, evaluates the
imported run, shows the grids, and can export a new re-importable portable bundle.

## Revision policy

During active development, use a branch or exact commit. Before sharing the notebook, create a Git tag and set the notebook revision to that tag or its exact commit SHA. Avoid relying on a moving `main` branch for reproducible training runs.
