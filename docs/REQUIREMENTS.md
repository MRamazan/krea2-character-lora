# Requirements

## Functional requirements

### Setup

- Install the package from a user-visible GitHub repository URL and immutable revision or branch.
- Use no Hugging Face authentication. All configured repositories are public and downloaded anonymously.
- Verify the GPU and runtime.
- Create or repair the isolated AI Toolkit environment.
- Resolve and record exact source and model revisions.
- Verify the active Krea 2 integration before training.
- Prepare Krea 2 Raw and the text encoder for training.
- Download the custom `artsyww/KREA2REALVAE` VAE anonymously, detect its checkpoint namespace, convert the original Wan/Qwen-Image namespace to a Diffusers-compatible `AutoencoderKLQwenImage` directory using the official `Qwen/Qwen-Image` configuration, strictly load the converted state dict, and run an encode/decode smoke test.
- Record the detected format, conversion, key mapping, and validation result in the asset manifest.
- Never fall back to another VAE when the custom VAE is incompatible; raise a precise error that names the discovered file, detected format, key-mismatch summary, and validation log path instead.
- Defer Krea 2 Turbo download until evaluation.

### Dataset

- Accept exactly one uploaded ZIP file.
- Discover image-caption pairs in nested directories.
- Handle repeated basenames safely.
- Reject unsupported, corrupt, unmatched, or empty inputs with actionable messages.
- Require a non-empty character trigger word.
- Audit trigger presence without silently modifying captions by default.
- Optionally prefix missing triggers only when explicitly enabled and after creating backups.
- Fingerprint the canonical dataset.
- Audit exact and perceptual duplicates.
- Display every image with filename, dimensions, caption, and highlighted trigger word.
- Display a summary and issue report before training.

### Training

- Train a transformer-only LoRA on Krea 2 Raw.
- Keep the text encoder frozen.
- Write production and smoke-test AI Toolkit configurations.
- Validate the generated configuration and source contract before training.
- Support smoke-only, interrupted, completed, and resumed runs.
- Persist enough state to reload a run after a Python runtime restart.
- Discover checkpoints from actual files rather than expected filenames.
- Never label an incomplete run as production-complete.

### Evaluation

- Prepare Krea 2 Turbo only when evaluation begins.
- Load LoRA weights non-destructively.
- Verify representative base parameters remain unchanged.
- Generate deterministic base-versus-LoRA comparisons.
- Compare actual checkpoints with identical prompts, seeds, dimensions, steps, guidance, VAE, and adapter scale.
- Compare editable LoRA scales without merging weights.
- Never choose the best checkpoint automatically.
- Allow explicit manual checkpoint selection after visual review.

### Export

- Export selected LoRA weights, optional checkpoints, images, logs, manifests, resolved revisions, hashes, and package freeze information.
- Package the actual current session state.
- Never include base model weights, the custom VAE weights, or any secret-like field.

## Non-functional requirements

- All project language is English.
- There is no Google Drive integration.
- The notebook has exactly four executable code cells.
- Package code, scripts, and tests contain no comment tokens.
- Notebook comments only separate sections.
- Public errors are actionable and specific.
- Pure Python behavior is locally testable without downloading Krea 2.
- Colab-only behavior has a documented smoke-test procedure.
