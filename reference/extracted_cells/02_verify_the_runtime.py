import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Select a GPU runtime before continuing.")

if not torch.cuda.is_bf16_supported():
    raise RuntimeError("The selected GPU does not support BF16.")

gpu_properties = torch.cuda.get_device_properties(0)
gpu_memory_gib = gpu_properties.total_memory / (1024 ** 3)
disk = shutil.disk_usage(PROJECT_ROOT)

runtime = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_memory_gib": gpu_memory_gib,
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "cuda_runtime": torch.version.cuda,
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "disk_total_gib": disk.total / (1024 ** 3),
    "disk_free_gib": disk.free / (1024 ** 3),
    "project_root": str(PROJECT_ROOT),
    "google_drive_used": False,
}

if USER_CONFIG["strict_hardware_check"] and gpu_memory_gib < USER_CONFIG["minimum_gpu_memory_gib"]:
    raise RuntimeError(
        f"GPU memory is {gpu_memory_gib:.2f} GiB, below the configured minimum of "
        f"{USER_CONFIG['minimum_gpu_memory_gib']} GiB."
    )

runtime_path = PATHS["config"] / "runtime_manifest.json"
runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
print(json.dumps(runtime, indent=2))
print(f"Runtime manifest: {runtime_path}")