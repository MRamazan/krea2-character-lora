import json
import os
import subprocess
from pathlib import Path

if not USER_CONFIG["run_training_smoke_test"]:
    print("Training smoke test skipped by configuration.")
else:
    configuration_path = PATHS["config"] / "train_krea2_lora_smoke.yaml"
    log_path = PATHS["logs"] / "training_smoke_test.log"
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    command = [str(PATHS["venv_python"]), "run.py", str(configuration_path)]
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(PATHS["ai_toolkit"]),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Unable to capture the smoke-test process output.")
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Training smoke test failed with exit code {return_code}. Complete log: {log_path}")
    smoke_run_directory = PATHS["smoke_checkpoints"] / f"{USER_CONFIG['run_name']}_smoke"
    smoke_checkpoints = sorted(smoke_run_directory.glob("*.safetensors")) if smoke_run_directory.is_dir() else []
    if not smoke_checkpoints:
        raise RuntimeError("The smoke test completed without producing a LoRA checkpoint.")
    record = {
        "status": "passed",
        "run_directory": str(smoke_run_directory),
        "checkpoint_count": len(smoke_checkpoints),
        "checkpoints": [str(path) for path in smoke_checkpoints],
        "log": str(log_path),
    }
    record_path = PATHS["config"] / "training_smoke_test_result.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))