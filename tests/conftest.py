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
