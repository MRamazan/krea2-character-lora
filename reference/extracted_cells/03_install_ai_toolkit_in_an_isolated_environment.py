import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

repository_path = PATHS["ai_toolkit"]
venv_path = PATHS["venv"]
venv_python = PATHS["venv_python"]
log_path = PATHS["logs"] / "environment_installation.log"
manifest_path = PATHS["config"] / "source_environment_manifest.json"
freeze_path = PATHS["config"] / "installed_packages.txt"

log_path.parent.mkdir(parents=True, exist_ok=True)
log_path.write_text("", encoding="utf-8")


def run_command(command, cwd=None, allow_failure=False):
    command = [str(part) for part in command]
    command_text = " ".join(command)
    print(f"$ {command_text}")
    captured = []
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n$ {command_text}\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        if process.stdout is None:
            raise RuntimeError(f"Unable to capture command output: {command_text}")
        for line in process.stdout:
            print(line, end="")
            captured.append(line)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
        log_handle.write(f"\nExit code: {return_code}\n")
        log_handle.flush()
    result = {
        "command": command,
        "returncode": return_code,
        "stdout": "".join(captured),
    }
    if return_code != 0 and not allow_failure:
        raise RuntimeError(
            f"Command failed with exit code {return_code}: {command_text}\n"
            f"Complete log: {log_path}"
        )
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_is_usable():
    if not venv_python.is_file():
        return False
    result = subprocess.run(
        [str(venv_python), "-c", "import pip, sys; print(sys.executable)"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


if repository_path.exists() and not (repository_path / ".git").is_dir():
    shutil.rmtree(repository_path)

if not repository_path.exists():
    run_command(["git", "init", str(repository_path)])

remote = run_command(
    ["git", "remote", "get-url", "origin"],
    cwd=repository_path,
    allow_failure=True,
)

if remote["returncode"] == 0:
    current_remote = remote["stdout"].strip()
    if current_remote != USER_CONFIG["ai_toolkit_repository"]:
        run_command(
            ["git", "remote", "set-url", "origin", USER_CONFIG["ai_toolkit_repository"]],
            cwd=repository_path,
        )
else:
    run_command(
        ["git", "remote", "add", "origin", USER_CONFIG["ai_toolkit_repository"]],
        cwd=repository_path,
    )

revision = USER_CONFIG["ai_toolkit_revision"]
if revision in {"main", "master"}:
    run_command(
        ["git", "fetch", "--no-tags", "--depth=1", "origin", revision],
        cwd=repository_path,
    )
    run_command(
        ["git", "checkout", "--detach", "--force", "FETCH_HEAD"],
        cwd=repository_path,
    )
else:
    direct_fetch = run_command(
        ["git", "fetch", "--no-tags", "--depth=1", "origin", revision],
        cwd=repository_path,
        allow_failure=True,
    )
    if direct_fetch["returncode"] != 0:
        run_command(
            ["git", "fetch", "--no-tags", "origin", "+refs/heads/*:refs/remotes/origin/*"],
            cwd=repository_path,
        )
    object_check = run_command(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=repository_path,
        allow_failure=True,
    )
    if object_check["returncode"] != 0:
        raise RuntimeError(f"The configured AI Toolkit revision could not be fetched: {revision}")
    run_command(
        ["git", "checkout", "--detach", "--force", revision],
        cwd=repository_path,
    )

run_command(["git", "reset", "--hard", "HEAD"], cwd=repository_path)
run_command(["git", "clean", "-ffdx"], cwd=repository_path)
run_command(["git", "submodule", "sync", "--recursive"], cwd=repository_path)
run_command(
    ["git", "submodule", "update", "--init", "--recursive", "--depth=1"],
    cwd=repository_path,
)
resolved_commit = run_command(
    ["git", "rev-parse", "HEAD"],
    cwd=repository_path,
)["stdout"].strip()

required_entries = [
    repository_path / "run.py",
    repository_path / "requirements.txt",
    repository_path / "requirements_base.txt",
    repository_path / "toolkit",
    repository_path / "extensions_built_in" / "diffusion_models" / "krea2" / "krea2.py",
]
for required_entry in required_entries:
    if not required_entry.exists():
        raise RuntimeError(f"Required AI Toolkit entry is missing: {required_entry}")

requirements_path = repository_path / "requirements.txt"
requirements_base_path = repository_path / "requirements_base.txt"
requirements_hash = sha256_file(requirements_path)
requirements_base_hash = sha256_file(requirements_base_path)
requirements_base_text = requirements_base_path.read_text(encoding="utf-8")
diffusers_match = re.search(
    r"diffusers\.git@([0-9a-fA-F]{40})",
    requirements_base_text,
)
expected_diffusers_commit = diffusers_match.group(1).lower() if diffusers_match else None

if USER_CONFIG["force_reinstall_environment"] and venv_path.exists():
    shutil.rmtree(venv_path)

if venv_path.exists() and not environment_is_usable():
    shutil.rmtree(venv_path)

if not environment_is_usable():
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            f"virtualenv=={USER_CONFIG['virtualenv_version']}",
        ]
    )
    run_command(
        [
            sys.executable,
            "-m",
            "virtualenv",
            "--python",
            sys.executable,
            str(venv_path),
        ]
    )

if not environment_is_usable():
    raise RuntimeError(f"The isolated environment is unusable after creation: {venv_path}")

run_command(
    [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
    ],
    cwd=repository_path,
)
run_command(
    [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        f"torch=={USER_CONFIG['torch_version']}",
        f"torchvision=={USER_CONFIG['torchvision_version']}",
        f"torchaudio=={USER_CONFIG['torchaudio_version']}",
        "--index-url",
        USER_CONFIG["torch_index_url"],
    ],
    cwd=repository_path,
)
run_command(
    [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "-r",
        str(requirements_path),
    ],
    cwd=repository_path,
)
run_command([str(venv_python), "-m", "pip", "check"], cwd=repository_path)

verification_script = r"""
import importlib.metadata
import json
import sys
import torch
import torchvision
import torchaudio
import diffusers

distribution = importlib.metadata.distribution("diffusers")
direct_url_text = distribution.read_text("direct_url.json")
direct_url = json.loads(direct_url_text) if direct_url_text else {}
result = {
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torchaudio": torchaudio.__version__,
    "cuda_runtime": torch.version.cuda,
    "diffusers": diffusers.__version__,
    "diffusers_commit": direct_url.get("vcs_info", {}).get("commit_id"),
    "cuda_available": torch.cuda.is_available(),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(result))
"""
verification_result = run_command(
    [str(venv_python), "-c", verification_script],
    cwd=repository_path,
)
verification_lines = [line for line in verification_result["stdout"].splitlines() if line.strip()]
if not verification_lines:
    raise RuntimeError("The environment verification returned no structured output.")
verification = json.loads(verification_lines[-1])

if expected_diffusers_commit is not None:
    installed_diffusers_commit = verification.get("diffusers_commit")
    if installed_diffusers_commit != expected_diffusers_commit:
        raise RuntimeError(
            "The installed Diffusers commit does not match the active AI Toolkit requirements.\n"
            f"Expected: {expected_diffusers_commit}\n"
            f"Received: {installed_diffusers_commit}"
        )

freeze = run_command(
    [str(venv_python), "-m", "pip", "freeze"],
    cwd=repository_path,
)["stdout"]
freeze_path.write_text(freeze, encoding="utf-8")

manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "repository": USER_CONFIG["ai_toolkit_repository"],
    "requested_revision": revision,
    "resolved_commit": resolved_commit,
    "repository_path": str(repository_path),
    "requirements_path": str(requirements_path),
    "requirements_sha256": requirements_hash,
    "requirements_base_path": str(requirements_base_path),
    "requirements_base_sha256": requirements_base_hash,
    "expected_diffusers_commit": expected_diffusers_commit,
    "virtualenv_version": USER_CONFIG["virtualenv_version"],
    "environment_creation_method": "virtualenv",
    "system_python": sys.executable,
    "venv_python": str(venv_python),
    "environment_verification": verification,
    "package_freeze": str(freeze_path),
    "installation_log": str(log_path),
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(json.dumps(manifest, indent=2))
print("AI Toolkit environment installation passed.")
