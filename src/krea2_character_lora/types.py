from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _display_image(path: str | Path) -> None:
    try:
        from IPython.display import display
        from PIL import Image
    except ImportError:
        print(f"Image: {path}")
        return
    with Image.open(path) as opened:
        display(opened.convert("RGB"))


@dataclass(slots=True)
class SetupReport:
    workspace: Path
    manifest_path: Path
    details: dict[str, Any] = field(default_factory=dict)

    def display(self) -> None:
        print(f"Workspace: {self.workspace}")
        print(f"Setup manifest: {self.manifest_path}")
        for key, value in self.details.items():
            print(f"{key}: {value}")


@dataclass(slots=True)
class DatasetResult:
    workspace: Path
    manifest_path: Path
    trigger_word: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def pair_count(self) -> int:
        return int(self.details.get("pair_count", 0))

    @property
    def fingerprint(self) -> str:
        return str(self.details.get("dataset_fingerprint_sha256", ""))

    @property
    def training_directory(self) -> Path:
        return Path(self.details["training_directory"])

    def display_summary(self) -> None:
        from .dataset import render_summary

        print(render_summary(self.details))

    def show_gallery(
        self,
        columns: int | None = None,
        thumbnail_size: int | None = None,
        show_filename: bool = True,
        show_dimensions: bool = True,
        show_caption: bool = True,
        highlight_trigger: bool = True,
        page: int | None = None,
        page_size: int | None = None,
        limit: int | None = None,
    ) -> None:
        from .dataset import render_gallery

        gallery = self.details.get("gallery", {})
        resolved_page_size = page_size or (limit if limit else gallery.get("page_size", 24))
        render_gallery(
            self.details,
            columns=columns or gallery.get("columns", 4),
            thumbnail_size=thumbnail_size or gallery.get("thumbnail_size", 320),
            show_filename=show_filename,
            show_dimensions=show_dimensions,
            show_caption=show_caption,
            highlight=highlight_trigger,
            page_size=resolved_page_size,
            page=page,
        )

    def show_caption_audit(self) -> None:
        from .dataset import render_caption_audit

        print(render_caption_audit(self.details))

    def show_issues(self) -> None:
        from .dataset import render_issues

        print(render_issues(self.details))


@dataclass(slots=True)
class TrainingRun:
    workspace: Path
    run_name: str
    manifest_path: Path
    trigger_word: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def training_complete(self) -> bool:
        return bool(self.details.get("training_complete", False))

    @property
    def status(self) -> str:
        return str(self.details.get("status", "unknown"))

    @property
    def checkpoint_inventory(self) -> dict[str, Any]:
        return dict(self.details.get("checkpoint_inventory", {}))

    def display_summary(self) -> None:
        print(f"Run: {self.run_name}")
        print(f"Status: {self.status}")
        print(f"Training complete: {self.training_complete}")
        print(f"Trigger word: {self.trigger_word}")
        inventory = self.checkpoint_inventory
        if inventory:
            print(f"Source kind: {inventory.get('source_kind')}")
            print(f"Final quality claim allowed: {inventory.get('final_quality_claim_allowed')}")
        for key in ("model_revision", "source_revision"):
            if key in self.details:
                print(f"{key}: {self.details[key]}")

    def show_checkpoints(self) -> None:
        inventory = self.checkpoint_inventory
        checkpoints = inventory.get("checkpoints", [])
        if not checkpoints:
            print("No checkpoints have been discovered yet.")
            return
        print(f"{'step':>8}  {'rank':>5}  {'size_bytes':>12}  sha256")
        for record in checkpoints:
            print(
                f"{record['step']:>8}  {record['rank']:>5}  "
                f"{record['size_bytes']:>12}  {record['sha256'][:16]}"
            )


@dataclass(slots=True)
class ExportBundle:
    archives: list[Path]
    details: dict[str, Any] = field(default_factory=dict)

    def display(self) -> None:
        print(f"Export directory: {self.details.get('export_directory')}")
        for record in self.details.get("packages", []):
            print(f"{record['filename']}  {record['size_bytes']} bytes  {record['sha256'][:16]}")

    def download(self) -> None:
        try:
            from google.colab import files
        except ImportError:
            print("Automatic download is only available inside Google Colab.")
            print("Archives:")
            for archive in self.archives:
                print(f"  {archive}")
            return
        for archive in self.archives:
            files.download(str(archive))


@dataclass(slots=True)
class EvaluationReport:
    workspace: Path
    run_name: str
    manifest_path: Path
    details: dict[str, Any] = field(default_factory=dict)

    def show_base_comparison(self) -> None:
        section = self.details.get("base_comparison")
        if not section or not section.get("grid_path"):
            print("No base-versus-LoRA comparison was produced.")
            return
        print(f"Base parameters unchanged: {section.get('base_parameters_unchanged')}")
        _display_image(section["grid_path"])

    def show_checkpoint_grid(self) -> None:
        section = self.details.get("checkpoint_sweep")
        if not section or not section.get("master_grid"):
            print("No checkpoint sweep was produced.")
            return
        _display_image(section["master_grid"])

    def show_scale_grid(self) -> None:
        section = self.details.get("scale_sweep")
        if not section or not section.get("master_grid"):
            print("No scale sweep was produced.")
            return
        _display_image(section["master_grid"])

    def show_summary(self) -> None:
        print(f"Run: {self.run_name}")
        print(f"Evaluation manifest: {self.manifest_path}")
        selection = self.details.get("active_checkpoint", {})
        if selection:
            print(f"Active checkpoint step: {selection.get('checkpoint_step')}")
            print(f"Final quality claim allowed: {selection.get('final_quality_claim_allowed')}")
        print(
            f"Automatic best-checkpoint selection: {self.details.get('automatic_selection', False)}"
        )

    def select_checkpoint(self, step: int) -> None:
        from .checkpoints import apply_manual_selection

        selection = apply_manual_selection(self.workspace, self.run_name, step)
        self.details["active_checkpoint"] = selection

    def export(
        self,
        include_selected_lora: bool = True,
        include_all_checkpoints: bool = False,
        include_images: bool = True,
        include_logs: bool = True,
        include_manifests: bool = True,
    ) -> ExportBundle:
        from .export import package_run

        return package_run(
            self.workspace,
            self.run_name,
            include_selected_lora=include_selected_lora,
            include_all_checkpoints=include_all_checkpoints,
            include_images=include_images,
            include_logs=include_logs,
            include_manifests=include_manifests,
        )
