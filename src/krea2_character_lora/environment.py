from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import (
    AI_TOOLKIT_REPOSITORY,
    AI_TOOLKIT_REVISION,
    TORCH_INDEX_URL,
    TORCH_VERSION,
    TORCHAUDIO_VERSION,
    TORCHVISION_VERSION,
    VIRTUALENV_VERSION,
)
from .errors import EnvironmentPreparationError
from .hashing import sha256_file
from .manifests import write_json_atomic
from .paths import ProjectPaths

_REQUIRED_ENTRIES = (
    "run.py",
    "requirements.txt",
    "requirements_base.txt",
    "toolkit",
    "extensions_built_in/diffusion_models/krea2/krea2.py",
)

_VERIFICATION_SCRIPT = """
import importlib.metadata
import json
import sys

import torch
import torchaudio
import torchvision

import diffusers

distribution = importlib.metadata.distribution("diffusers")
direct_url_text = distribution.read_text("direct_url.json")
direct_url = json.loads(direct_url_text) if direct_url_text else {}
print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torchaudio": torchaudio.__version__,
    "cuda_runtime": torch.version.cuda,
    "diffusers": diffusers.__version__,
    "diffusers_commit": direct_url.get("vcs_info", {}).get("commit_id"),
    "cuda_available": torch.cuda.is_available(),
    "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}))
"""


def run_command(
    command: list[str], cwd: Path | None, log_path: Path, allow_failure: bool = False
) -> dict[str, Any]:
    rendered = [str(part) for part in command]
    captured: list[str] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n$ {' '.join(rendered)}\n")
        process = subprocess.Popen(
            rendered,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        if process.stdout is None:
            raise EnvironmentPreparationError(f"Unable to capture command output: {rendered}")
        for line in process.stdout:
            print(line, end="")
            captured.append(line)
            log_handle.write(line)
        return_code = process.wait()
        log_handle.write(f"\nExit code: {return_code}\n")
    if return_code != 0 and not allow_failure:
        raise EnvironmentPreparationError(
            f"Command failed with exit code {return_code}: {' '.join(rendered)}. Log: {log_path}"
        )
    return {"command": rendered, "returncode": return_code, "stdout": "".join(captured)}


def environment_is_usable(venv_python: Path) -> bool:
    if not venv_python.is_file():
        return False
    probe = subprocess.run(
        [str(venv_python), "-c", "import pip, sys; print(sys.executable)"],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def _checkout_ai_toolkit(paths: ProjectPaths, revision: str, log_path: Path) -> str:
    repository = paths.ai_toolkit
    repository.parent.mkdir(parents=True, exist_ok=True)
    if repository.exists() and not (repository / ".git").is_dir():
        shutil.rmtree(repository)
    if not repository.exists():
        run_command(["git", "init", str(repository)], None, log_path)
    remote = run_command(
        ["git", "remote", "get-url", "origin"], repository, log_path, allow_failure=True
    )
    if remote["returncode"] == 0:
        if remote["stdout"].strip() != AI_TOOLKIT_REPOSITORY:
            run_command(
                ["git", "remote", "set-url", "origin", AI_TOOLKIT_REPOSITORY], repository, log_path
            )
    else:
        run_command(["git", "remote", "add", "origin", AI_TOOLKIT_REPOSITORY], repository, log_path)
    if revision in {"main", "master"}:
        run_command(
            ["git", "fetch", "--no-tags", "--depth=1", "origin", revision], repository, log_path
        )
        run_command(["git", "checkout", "--detach", "--force", "FETCH_HEAD"], repository, log_path)
    else:
        direct = run_command(
            ["git", "fetch", "--no-tags", "--depth=1", "origin", revision],
            repository,
            log_path,
            allow_failure=True,
        )
        if direct["returncode"] != 0:
            run_command(
                ["git", "fetch", "--no-tags", "origin", "+refs/heads/*:refs/remotes/origin/*"],
                repository,
                log_path,
            )
        object_check = run_command(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            repository,
            log_path,
            allow_failure=True,
        )
        if object_check["returncode"] != 0:
            raise EnvironmentPreparationError(
                f"The configured AI Toolkit revision could not be fetched: {revision}"
            )
        run_command(["git", "checkout", "--detach", "--force", revision], repository, log_path)
    run_command(["git", "reset", "--hard", "HEAD"], repository, log_path)
    run_command(["git", "clean", "-ffdx"], repository, log_path)
    run_command(["git", "submodule", "sync", "--recursive"], repository, log_path)
    run_command(
        ["git", "submodule", "update", "--init", "--recursive", "--depth=1"], repository, log_path
    )
    return run_command(["git", "rev-parse", "HEAD"], repository, log_path)["stdout"].strip()


def _expected_diffusers_commit(requirements_base: Path) -> str | None:
    text = requirements_base.read_text(encoding="utf-8")
    match = re.search(r"diffusers\.git@([0-9a-fA-F]{40})", text)
    return match.group(1).lower() if match else None


def prepare_environment(
    paths: ProjectPaths, revision: str = AI_TOOLKIT_REVISION, force_reinstall: bool = False
) -> dict[str, Any]:
    log_path = paths.logs / "environment_installation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    resolved_commit = _checkout_ai_toolkit(paths, revision, log_path)
    repository = paths.ai_toolkit
    for entry in _REQUIRED_ENTRIES:
        if not (repository / entry).exists():
            raise EnvironmentPreparationError(f"Required AI Toolkit entry is missing: {entry}")
    requirements = repository / "requirements.txt"
    requirements_base = repository / "requirements_base.txt"
    expected_diffusers = _expected_diffusers_commit(requirements_base)
    venv_python = paths.venv_python
    if force_reinstall and paths.venv.exists():
        shutil.rmtree(paths.venv)
    if paths.venv.exists() and not environment_is_usable(venv_python):
        shutil.rmtree(paths.venv)
    if not environment_is_usable(venv_python):
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--upgrade",
                f"virtualenv=={VIRTUALENV_VERSION}",
            ],
            None,
            log_path,
        )
        run_command(
            [sys.executable, "-m", "virtualenv", "--python", sys.executable, str(paths.venv)],
            None,
            log_path,
        )
    if not environment_is_usable(venv_python):
        raise EnvironmentPreparationError(f"The isolated environment is unusable: {paths.venv}")
    run_command(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        repository,
        log_path,
    )
    run_command(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            f"torchaudio=={TORCHAUDIO_VERSION}",
            "--index-url",
            TORCH_INDEX_URL,
        ],
        repository,
        log_path,
    )
    run_command(
        [str(venv_python), "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements)],
        repository,
        log_path,
    )
    run_command([str(venv_python), "-m", "pip", "check"], repository, log_path)
    verification = run_command([str(venv_python), "-c", _VERIFICATION_SCRIPT], repository, log_path)
    lines = [line for line in verification["stdout"].splitlines() if line.strip()]
    if not lines:
        raise EnvironmentPreparationError("The environment verification returned no output.")
    verification_result = json.loads(lines[-1])
    installed_diffusers = verification_result.get("diffusers_commit")
    if expected_diffusers is not None and installed_diffusers != expected_diffusers:
        raise EnvironmentPreparationError(
            "The installed Diffusers commit does not match the AI Toolkit requirements. "
            f"Expected {expected_diffusers}, received {installed_diffusers}."
        )
    freeze_path = paths.config / "installed_packages.txt"
    freeze = run_command([str(venv_python), "-m", "pip", "freeze"], repository, log_path)["stdout"]
    freeze_path.write_text(freeze, encoding="utf-8")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": AI_TOOLKIT_REPOSITORY,
        "requested_revision": revision,
        "resolved_commit": resolved_commit,
        "repository_path": str(repository),
        "requirements_sha256": sha256_file(requirements),
        "requirements_base_sha256": sha256_file(requirements_base),
        "expected_diffusers_commit": expected_diffusers,
        "venv_python": str(venv_python),
        "environment_verification": verification_result,
        "package_freeze": str(freeze_path),
        "installation_log": str(log_path),
    }
    write_json_atomic(paths.environment_manifest, manifest)
    return manifest
