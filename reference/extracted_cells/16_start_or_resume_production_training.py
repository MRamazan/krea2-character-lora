import json
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

status_path = PATHS["config"] / "production_training_status.json"
log_path = PATHS["logs"] / "production_training.log"

if not USER_CONFIG["run_production_training"]:
    status = {
        "status": "skipped",
        "reason": "run_production_training is false",
        "training_complete": False,
        "log": str(log_path),
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
else:
    configuration_path = PATHS["config"] / "train_krea2_lora.yaml"
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    command = [str(PATHS["venv_python"]), "run.py", str(configuration_path)]
    interrupted = False
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(PATHS["ai_toolkit"]),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            if process.stdout is None:
                raise RuntimeError("Unable to capture the production-training process output.")
            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
                log_handle.flush()
        except KeyboardInterrupt:
            interrupted = True
            process.send_signal(signal.SIGINT)
            process.wait()
            print("Production training was intentionally interrupted. Saved checkpoints will be inventoried next.")
        return_code = process.wait()
    status = {
        "status": "interrupted" if interrupted else "completed_process",
        "process_return_code": return_code,
        "training_complete": False,
        "started_configuration": str(configuration_path),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "log": str(log_path),
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if return_code not in {0, 130, -2} and not interrupted:
        raise RuntimeError(f"Production training failed with exit code {return_code}. Complete log: {log_path}")
    print(json.dumps(status, indent=2))