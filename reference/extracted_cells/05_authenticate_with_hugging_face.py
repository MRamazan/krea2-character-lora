import getpass
import json
import os
import subprocess

existing_token = os.environ.get("HF_TOKEN", "").strip()
if existing_token:
    token = existing_token
else:
    token = getpass.getpass("Enter a Hugging Face read token: ").strip()

if not token:
    raise RuntimeError("A Hugging Face token is required to access and download the configured assets.")

os.environ["HF_TOKEN"] = token
validation_script = r"""
import json
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
identity = api.whoami()
print(json.dumps({"authenticated": True, "name": identity.get("name"), "type": identity.get("type")}))
"""
result = subprocess.run(
    [str(PATHS["venv_python"]), "-c", validation_script],
    cwd=str(PATHS["ai_toolkit"]),
    env=os.environ.copy(),
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    raise RuntimeError(f"Hugging Face authentication failed.\n{result.stdout}\n{result.stderr}")
record = json.loads(result.stdout.strip().splitlines()[-1])
record_path = PATHS["config"] / "huggingface_authentication.json"
record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))