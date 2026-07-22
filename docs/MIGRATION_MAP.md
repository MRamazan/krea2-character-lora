# Migration Map

| Reference cell | Target module or notebook cell |
|---|---|
| Central configuration | `configuration.py`, notebook dataset/training/evaluation cells |
| Runtime verification | `environment.py` |
| AI Toolkit installation | `environment.py` |
| Krea 2 implementation verification | `environment.py` |
| Hugging Face authentication | removed entirely; all downloads are anonymous |
| Training asset resolution | `assets.py` |
| Custom VAE normalization and strict validation | `assets.py`, `isolated/validate_vae.py` |
| Dataset ZIP upload | dataset notebook cell |
| Pair extraction and validation | `dataset.py` |
| Trigger audit | `dataset.py` |
| Dataset fingerprint | `dataset.py` |
| Dataset visualization | `dataset.py`, dataset notebook cell |
| Duplicate audit | `dataset.py` |
| Training and smoke configuration | `training.py` |
| Preflight validation | `training.py` |
| Smoke test | `training.py` |
| Production training and resume | `training.py` |
| Checkpoint inventory and selection | `checkpoints.py` |
| Additional LoRA validation | excluded from the first character-only public notebook; internal compatibility may remain only when required by evaluation architecture |
| Inference asset resolution | `assets.py` |
| Runtime helper generation | `runtime.py` |
| Base-versus-LoRA comparison | `evaluation.py` |
| Checkpoint sweep | `evaluation.py`, `checkpoints.py` |
| Manual checkpoint selection | `checkpoints.py`, `EvaluationReport` |
| Scale sweep | `evaluation.py` |
| Session packaging | `export.py` |
| Colab download | evaluation notebook cell or `ExportBundle.download()` |
| Restore utility | public manifest reload methods in `api.py` and `manifests.py` |
