from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from . import assets, bundle, environment, evaluation, training
from . import dataset as dataset_module
from .configuration import DatasetConfig, EvaluationConfig, TrainingConfig
from .constants import AI_TOOLKIT_REVISION
from .errors import AssetError
from .manifests import read_json, write_json_atomic
from .paths import ProjectPaths
from .types import DatasetResult, EvaluationReport, ImportedRun, SetupReport, TrainingRun


class CharacterLoraPipeline:
    def __init__(
        self,
        workspace: str | Path = "/content/krea2_character_lora",
        repository_revision: str = "main",
        ai_toolkit_revision: str = AI_TOOLKIT_REVISION,
    ) -> None:
        self.paths = ProjectPaths.create(workspace)
        self.repository_revision = repository_revision
        self.ai_toolkit_revision = ai_toolkit_revision

    def setup(
        self,
        prepare_training_assets: bool = True,
        prepare_inference_assets: bool = False,
        verify_environment: bool = True,
        force_reinstall: bool = False,
    ) -> SetupReport:
        details: dict[str, Any] = {
            "workspace": str(self.paths.root),
            "repository_revision": self.repository_revision,
            "ai_toolkit_revision": self.ai_toolkit_revision,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if verify_environment:
            environment_manifest = environment.prepare_environment(
                self.paths, self.ai_toolkit_revision, force_reinstall
            )
            details["environment"] = {
                "resolved_commit": environment_manifest["resolved_commit"],
                "verification": environment_manifest["environment_verification"],
            }
        if prepare_training_assets:
            training_manifest = assets.prepare_training_assets(self.paths)
            details["training_assets"] = {
                "training_model_revision": training_manifest["training_model"]["revision"],
                "vae": {
                    "repository": training_manifest["vae"]["repository"],
                    "revision": training_manifest["vae"]["revision"],
                    "strict_validation": training_manifest["vae"]["strict_validation"],
                },
            }
        if prepare_inference_assets:
            inference_manifest = assets.prepare_inference_assets(self.paths)
            details["inference_assets"] = {
                "inference_model_revision": inference_manifest["inference_model"]["revision"],
            }
        write_json_atomic(self.paths.setup_manifest, details)
        return SetupReport(
            workspace=self.paths.root, manifest_path=self.paths.setup_manifest, details=details
        )

    def prepare_dataset(self, zip_path: str | Path, config: DatasetConfig) -> DatasetResult:
        return dataset_module.prepare_dataset(zip_path, config, self.paths)

    def load_dataset(self) -> DatasetResult:
        return dataset_module.load_dataset(self.paths)

    def preview_training(self, dataset: DatasetResult, config: TrainingConfig) -> dict[str, Any]:
        config.validate()
        asset_manifest = self._training_asset_manifest()
        preflight = training.preview_resolved_configuration(
            config, dataset, asset_manifest, self.paths, smoke_steps=3
        )
        print(f"Run: {preflight['run_name']}")
        print(f"Dataset pairs: {preflight['dataset_pairs']}")
        print(f"LoRA rank/alpha: {preflight['network_rank']}/{preflight['network_alpha']}")
        print(f"Training steps: {preflight['training_steps']}")
        print(f"Text-encoder training: {preflight['train_text_encoder']}")
        print(f"Permanent merge: {preflight['merge_network_on_save']}")
        return preflight

    def train(
        self,
        dataset: DatasetResult,
        config: TrainingConfig,
        run_smoke_test: bool = True,
        smoke_test_steps: int = 3,
        run_production: bool = True,
        resume: Literal["auto", "never", "required"] = "auto",
    ) -> TrainingRun:
        config.validate()
        asset_manifest = self._training_asset_manifest()
        resolved = training.write_configurations(
            config, dataset, asset_manifest, self.paths, smoke_test_steps
        )
        training.run_source_preflight(self.paths, resolved["production_path"])
        smoke_status = None
        if run_smoke_test:
            smoke_status = training.run_smoke_test(
                self.paths, config.run_name, resolved["smoke_path"]
            )
        if run_production:
            process_status = training.run_production_training(
                self.paths, config.run_name, resolved["production_path"], resume
            )
        elif smoke_status is not None:
            process_status = smoke_status
        else:
            raise AssetError("Either production training or the smoke test must run.")
        run = training.finalize_run(
            self.paths,
            config,
            dataset,
            asset_manifest,
            resolved,
            process_status,
            smoke_test_steps,
        )
        if smoke_status is not None:
            run.details["smoke_status"] = smoke_status
            write_json_atomic(self.paths.run_manifest(config.run_name), run.details)
        return run

    def load_run(self, run_name: str) -> TrainingRun:
        return training.load_run(self.paths, run_name)

    def import_evaluation_bundle(self, zip_path: str | Path) -> ImportedRun:
        return bundle.import_evaluation_bundle(self.paths.root, zip_path)

    def prepare_evaluation_assets(self) -> SetupReport:
        inference_manifest = assets.prepare_inference_assets(self.paths)
        details = {
            "inference_model_revision": inference_manifest["inference_model"]["revision"],
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        return SetupReport(
            workspace=self.paths.root,
            manifest_path=self.paths.inference_asset_manifest,
            details=details,
        )

    def evaluate(self, run: TrainingRun, config: EvaluationConfig) -> EvaluationReport:
        return evaluation.evaluate(self.paths, run, config)

    def _training_asset_manifest(self) -> dict[str, Any]:
        if not self.paths.training_asset_manifest.is_file():
            raise AssetError(
                "Training assets are not prepared. Call pipeline.setup with "
                "prepare_training_assets enabled first."
            )
        return read_json(self.paths.training_asset_manifest)
