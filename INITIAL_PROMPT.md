Implement the complete refactor described by `CLAUDE.md` and the documents under `docs`.

Start by reading the reference notebook and all extracted source cells. Build a migration inventory that maps every source behavior to a target module. Then implement the package, tests, and generated notebook rather than stopping at a plan.

The result must be character-only. Remove the public and internal `concept_type` setting instead of hardcoding it to `character`. Do not leave dead branches for style, object, product, or generic concept training. Keep only character-relevant behavior and neutral internal naming where required by third-party APIs.

The final Colab notebook must have exactly four executable cells: setup, dataset, training, and evaluation. It must remain readable and expose important user choices. The dataset cell must define the trigger word and show a complete image-and-caption gallery. The setup cell must install the package, prepare the isolated environment, and prepare the training assets, including the normalized and strictly validated custom VAE, using anonymous downloads with no Hugging Face token. The training and evaluation cells must visibly expose their important parameters.

Move fixed implementation details into the package while preserving the reliability features of the reference notebook, including isolated environment setup, exact revision recording, source verification, dataset validation, trigger auditing, duplicate auditing, fingerprints, smoke testing, resume behavior, checkpoint inventory, non-destructive adapter loading, base-versus-LoRA comparison, checkpoint sweeps, scale sweeps, export packaging, hashes, manifests, and conservative completion status.

Do not add Google Drive integration, permanent LoRA merging, text-encoder training, or automatic best-checkpoint selection.

Do not add comment tokens to Python package files, scripts, or tests. Notebook comments may only separate sections. Keep all project text in English.

Use the existing contract tests as requirements, expand them where necessary, and make the full local test suite pass. Do not modify the files under `reference`. Do not push to GitHub.

Complete the implementation, rebuild the notebook, run all checks, and provide the completion report required by `CLAUDE.md`.
