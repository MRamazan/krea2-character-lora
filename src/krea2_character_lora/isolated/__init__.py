from pathlib import Path

from ..errors import EnvironmentPreparationError

_SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def isolated_script(name: str) -> Path:
    path = _SCRIPT_DIRECTORY / name
    if not path.is_file():
        raise EnvironmentPreparationError(f"The isolated runtime script is missing: {name}")
    return path


def runtime_helper_source() -> str:
    return isolated_script("krea2_runtime.py").read_text(encoding="utf-8")


__all__ = ["isolated_script", "runtime_helper_source"]
