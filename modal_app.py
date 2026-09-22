"""Modal deployment for jev-api.

Runs the existing FastAPI app (jev_api.main:app) on a Modal GPU with a
persistent Volume for the Von model cache, so warm starts skip re-download.

Deploy:   modal deploy modal_app.py
Serve:    modal serve modal_app.py   (temp URL for testing)
"""

import modal

app = modal.App("jev-api")

MODEL_CACHE_DIR = "/data/models"

# Persistent model cache (~1.5 GB) shared across warm starts.
model_cache = modal.Volume.from_name("jev-model-cache", create_if_missing=True)

# Mirror of the repo's Dockerfile deps, but with CUDA torch (the Dockerfile
# pins CPU wheels, so it is not reused here — GPU requires CUDA wheels).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch",
        "transformers>=4.48.0",
        "accelerate>=0.26.0",
        "huggingface_hub>=0.26.0",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "pydantic>=2.9.0",
        "pydantic-settings>=2.6.0",
        "httpx>=0.27.0",
        "von-sdk @ git+https://github.com/wfzyx/von.git",
    )
    .env(
        {
            "PYTHONPATH": "/root/src",
            "VON_BACKEND": "von",
            "JEV_DEVICE": "cuda",
            "MODEL_CACHE_DIR": MODEL_CACHE_DIR,
            "HF_HOME": MODEL_CACHE_DIR,
            "HUGGINGFACE_HUB_CACHE": MODEL_CACHE_DIR,
            "DOWNLOAD_ON_STARTUP": "true",
        }
    )
    # add_local_* must be the LAST build step; copy=True bakes src into the image.
    .add_local_dir("src", remote_path="/root/src", copy=True)
)


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("jev-api")],
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=1800,
    scaledown_window=1200,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app(label="jev-api")
def fastapi_app():
    from jev_api.main import app as fastapi_app

    return fastapi_app
