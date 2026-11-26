import os
import subprocess
from pathlib import Path

import modal

ROOT_DIR = Path(__file__).resolve().parent.parent

app = modal.App("finkyc-gpu-origin")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential")
    .pip_install_from_requirements(str(ROOT_DIR / "requirements.txt"))
    .env({
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/app:/app/src",
        "PORT": "8080",
        "FINKYC_LIVE_GPU": "1",
        "FINKYC_BASE_MODEL": os.getenv("FINKYC_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        "FINKYC_FT_MODEL": os.getenv("FINKYC_FT_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "FINKYC_ADAPTER_PATH": os.getenv("FINKYC_ADAPTER_PATH", "models/adapters/qwen2.5-7b-qlora"),
    })
    .add_local_dir(
        str(ROOT_DIR),
        remote_path="/app",
        ignore=[
            ".venv",
            ".git",
            "__pycache__",
            "logs",
            "checkpoints",
            "cloudflare",
            "finkyc_slurm_package.tar.gz",
            ".DS_Store",
        ],
    )
)

hf_cache = modal.Volume.from_name("finkyc-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=900,
    scaledown_window=300,
    volumes={"/root/.cache/huggingface": hf_cache},
)
@modal.concurrent(max_inputs=8)
@modal.web_server(8080, startup_timeout=180)
def serve():
    subprocess.Popen(["python", "run.py", "serve", "--host", "0.0.0.0", "--port", "8080"], cwd="/app")
