# ============================================================
# Setup
# ============================================================

import subprocess
import sys

REPOSITORY_URL = "https://github.com/YOUR_USERNAME/krea2-character-lora.git"
PIPELINE_REVISION = "main"
WORKSPACE = "/content/krea2_character_lora_eval"

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--upgrade",
        f"git+{REPOSITORY_URL}@{PIPELINE_REVISION}",
    ]
)

from krea2_character_lora import CharacterLoraPipeline

pipeline = CharacterLoraPipeline(
    workspace=WORKSPACE,
    repository_revision=PIPELINE_REVISION,
)

setup_report = pipeline.setup(
    verify_environment=True,
    prepare_training_assets=False,
    prepare_inference_assets=False,
)

setup_report.display()
