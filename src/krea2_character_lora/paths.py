from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    root: Path
    assets: Path
    config: Path
    datasets: Path
    environments: Path
    exports: Path
    inference: Path
    logs: Path
    runtime_helpers: Path
    runs: Path

    @classmethod
    def create(cls, root: str | Path) -> "ProjectPaths":
        resolved = Path(root).expanduser().resolve()
        values = {
            "assets": resolved / "assets",
            "config": resolved / "config",
            "datasets": resolved / "datasets",
            "environments": resolved / "environments",
            "exports": resolved / "exports",
            "inference": resolved / "inference",
            "logs": resolved / "logs",
            "runtime_helpers": resolved / "runtime_helpers",
            "runs": resolved / "runs",
        }
        resolved.mkdir(parents=True, exist_ok=True)
        for path in values.values():
            path.mkdir(parents=True, exist_ok=True)
        return cls(root=resolved, **values)

    @property
    def ai_toolkit(self) -> Path:
        return self.environments / "ai-toolkit"

    @property
    def venv(self) -> Path:
        return self.environments / "venv"

    @property
    def venv_python(self) -> Path:
        return self.venv / "bin" / "python"

    @property
    def models(self) -> Path:
        return self.assets / "models"

    @property
    def dataset_active(self) -> Path:
        return self.datasets / "active"

    @property
    def dataset_raw(self) -> Path:
        return self.dataset_active / "raw"

    @property
    def dataset_training(self) -> Path:
        return self.dataset_active / "training"

    @property
    def dataset_audit(self) -> Path:
        return self.dataset_active / "audit"

    @property
    def setup_manifest(self) -> Path:
        return self.config / "setup_manifest.json"

    @property
    def training_asset_manifest(self) -> Path:
        return self.config / "training_asset_manifest.json"

    @property
    def inference_asset_manifest(self) -> Path:
        return self.config / "inference_asset_manifest.json"

    @property
    def environment_manifest(self) -> Path:
        return self.config / "environment_manifest.json"

    @property
    def dataset_manifest(self) -> Path:
        return self.dataset_active / "dataset_manifest.json"

    @property
    def dataset_fingerprint(self) -> Path:
        return self.dataset_active / "dataset_fingerprint.json"

    def run_dir(self, run_name: str) -> Path:
        return self.runs / run_name

    def run_manifest(self, run_name: str) -> Path:
        return self.run_dir(run_name) / "run_manifest.json"

    def checkpoint_inventory(self, run_name: str) -> Path:
        return self.run_dir(run_name) / "checkpoint_inventory.json"

    def active_checkpoint(self, run_name: str) -> Path:
        return self.run_dir(run_name) / "active_checkpoint.json"

    def evaluation_manifest(self, run_name: str) -> Path:
        return self.run_dir(run_name) / "evaluation_manifest.json"

    def checkpoints_dir(self, run_name: str) -> Path:
        return self.run_dir(run_name) / "checkpoints"

    def smoke_checkpoints_dir(self, run_name: str) -> Path:
        return self.run_dir(run_name) / "smoke_checkpoints"
