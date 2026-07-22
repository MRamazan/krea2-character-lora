import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from safetensors.numpy import save_file


def _write_image(path: Path, size: tuple[int, int] = (64, 64), seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _write_solid_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


@pytest.fixture
def make_dataset_zip(tmp_path):
    def _factory(
        pairs,
        nested=False,
        include_trigger="mycharacter",
        zip_name="dataset.zip",
    ):
        staging = tmp_path / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        for index, pair in enumerate(pairs):
            name = pair["name"]
            folder = staging / f"group_{index % 2}" if nested else staging
            folder.mkdir(parents=True, exist_ok=True)
            image_path = folder / f"{name}.png"
            caption_path = folder / f"{name}.txt"
            _write_image(image_path, seed=pair.get("seed", index))
            caption = pair.get("caption")
            if caption is None:
                caption = f"{include_trigger} portrait number {index}" if include_trigger else ""
            caption_path.write_text(caption, encoding="utf-8")
        archive = tmp_path / zip_name
        with zipfile.ZipFile(archive, "w") as handle:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    handle.write(path, arcname=str(path.relative_to(staging)))
        return archive

    return _factory


@pytest.fixture
def make_lora_checkpoint():
    def _factory(path: Path, rank: int = 4, nonfinite: bool = False, extra_key: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        down = np.zeros((rank, 8), dtype=np.float32)
        up = np.zeros((8, rank), dtype=np.float32)
        if nonfinite:
            up[0, 0] = np.inf
        tensors = {
            "lora_transformer_block.lora_A.weight": down,
            "lora_transformer_block.lora_B.weight": up,
        }
        if extra_key:
            tensors["lora_transformer_block.extra.weight"] = np.zeros((2, 2), dtype=np.float32)
        save_file(tensors, str(path), metadata={"format": "pt"})
        return path

    return _factory


@pytest.fixture
def make_vae_safetensors():
    def _factory(path: Path, shapes: dict[str, list[int]], prefix: str = ""):
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            f"{prefix}{name}": np.zeros(tuple(shape), dtype=np.float32)
            for name, shape in shapes.items()
        }
        save_file(tensors, str(path), metadata={"format": "pt"})
        return path

    return _factory


@pytest.fixture
def solid_image_writer():
    return _write_solid_image


@pytest.fixture
def prepared_run(tmp_path, make_lora_checkpoint):
    from krea2_character_lora.checkpoints import inventory_from_directory, select_active
    from krea2_character_lora.manifests import write_json_atomic
    from krea2_character_lora.paths import ProjectPaths

    def _factory(
        workspace_name="workspace",
        run_name="character_v1",
        steps=(100, 200),
        with_eval=False,
        with_images=False,
        rank=8,
    ):
        paths = ProjectPaths.create(tmp_path / workspace_name)
        production = paths.checkpoints_dir(run_name) / run_name
        for step in steps:
            make_lora_checkpoint(production / f"{run_name}_{step:09d}.safetensors", rank=rank)
        inventory = inventory_from_directory(
            production,
            run_name,
            max(steps),
            "production",
            {"status": "completed_process", "process_return_code": 0},
        )
        selection = select_active(inventory, mode="auto")
        write_json_atomic(paths.active_checkpoint(run_name), selection)
        write_json_atomic(
            paths.run_manifest(run_name),
            {
                "run_name": run_name,
                "project_name": "krea2_character_lora",
                "trigger_word": "mycharacter",
                "model_revision": "rawrev",
                "vae_revision": "vaerev",
                "source_revision": "sourcerev",
                "training_config": {
                    "lora_rank": rank,
                    "lora_alpha": rank,
                    "keep_tokens": 1,
                    "shuffle_tokens": False,
                    "token_dropout_rate": 0.0,
                    "training_dtype": "bf16",
                    "max_text_length": 512,
                },
                "checkpoint_inventory": inventory,
                "active_checkpoint": selection,
            },
        )
        (paths.logs / f"{run_name}_training.log").write_text("training log", encoding="utf-8")
        write_json_atomic(
            paths.training_asset_manifest,
            {
                "training_model": {"repository": "krea/Krea-2-Raw", "revision": "rawrev"},
                "text_encoder": {"repository": "Qwen/Qwen3-VL-4B-Instruct", "revision": "terev"},
                "vae": {
                    "source_repository": "artsyww/KREA2REALVAE",
                    "source_revision": "vaerev",
                    "source_filename": "krea2RealVae_v10.safetensors",
                    "source_sha256": "abc123",
                },
            },
        )
        if with_images:
            image_root = paths.inference / run_name / "checkpoint_sweep"
            image_root.mkdir(parents=True, exist_ok=True)
            _write_solid_image(image_root / "case_01_base.png", (32, 32), (0, 0, 0))
        if with_eval:
            write_json_atomic(
                paths.evaluation_manifest(run_name),
                {
                    "run_name": run_name,
                    "prompts": ["mycharacter is a woman"],
                    "seeds": [42],
                    "inference_settings": {"width": 1024, "height": 1024},
                    "scale_sweep": [1.0],
                    "checkpoint_mode": "auto",
                },
            )
        return paths, run_name

    return _factory
