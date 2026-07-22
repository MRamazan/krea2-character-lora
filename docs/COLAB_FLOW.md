# Colab Flow

## Cell 1: Setup

The user edits:

- `REPOSITORY_URL`
- `PIPELINE_REVISION`
- `WORKSPACE`

The cell installs the package, initializes the pipeline, installs or repairs the isolated runtime, verifies source compatibility, and prepares the training assets, including the normalized and strictly validated custom VAE. It uses no Hugging Face token; every repository is downloaded anonymously.

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

The cell prepares Krea 2 Turbo, evaluates the run, displays all comparison grids, packages exports, and optionally downloads archives.

## Revision policy

During active development, use a branch or exact commit. Before sharing the notebook, create a Git tag and set the notebook revision to that tag or its exact commit SHA. Avoid relying on a moving `main` branch for reproducible training runs.
