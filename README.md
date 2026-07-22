# Krea 2 Character LoRA

This repository turns the reference Krea 2 LoRA notebook into a character-only Python package with a compact Google Colab interface.

The Colab notebook has exactly four executable cells:

1. Setup
2. Dataset preparation and visualization
3. Training
4. Evaluation and export

The implementation remains in the package. User decisions remain visible in the notebook.

## Character-only scope

The package does not expose a concept type, style mode, object mode, product mode, or generic training mode. Character LoRA training is the only supported workflow. The trigger word is defined in the dataset cell and persisted in dataset and run manifests.

## Anonymous downloads and custom VAE

Every Hugging Face repository used by the pipeline is public and downloaded anonymously. There is no Hugging Face token, `getpass` prompt, Colab Secrets integration, or `.env` support anywhere in the package or notebook.

The default VAE is `artsyww/KREA2REALVAE`. Its published checkpoint (`krea2RealVae_v10.safetensors`) is a complete VAE stored in the original upstream Wan/Qwen-Image key namespace (`conv1`/`conv2`, `encoder.downsamples`, `decoder.upsamples`, RMS-norm `gamma`), not the Diffusers namespace. During setup the pipeline downloads it anonymously, inspects the safetensors header, detects the namespace, applies a deterministic and provably bijective conversion to the Diffusers `AutoencoderKLQwenImage` layout (using the official `Qwen/Qwen-Image` configuration), strictly loads the converted state dict with `AutoencoderKLQwenImage` (rejecting any missing, unexpected, or shape-mismatched key), and runs an encode/decode smoke test. The detected format, conversion, exact key mapping, and validation result are recorded in the VAE manifest. The pipeline never silently falls back to another VAE. The external VAE weights are neither committed to this repository nor included in export archives; review the upstream license before redistributing.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/build_notebook.py
ruff check .
pytest
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Repository workflow

1. Develop and test the package locally.
2. Build the committed notebook with `python scripts/build_notebook.py`.
3. Push the repository to GitHub.
4. Set `REPOSITORY_URL` and `PIPELINE_REVISION` in the setup cell.
5. During development, install a specific commit or branch.
6. For stable Colab runs, install a version tag or immutable commit.

Example:

```python
REPOSITORY_URL = "https://github.com/YOUR_USERNAME/krea2-character-lora.git"
PIPELINE_REVISION = "v0.1.0"
```

## Reference implementation

The original notebook is stored in `reference/universal_pipeline_v2_1.ipynb`. Its code cells are also extracted into `reference/extracted_cells` to make migration and comparison easier.

## Claude Code

Read `CLAUDE.md`, then paste the task from `INITIAL_PROMPT.md` into Claude Code from the repository root.
