#!/usr/bin/env python3
"""Optional offline helper: ensure HF snapshot exists under MODEL_CACHE_DIR."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    repo = os.environ.get("HF_REPO", "heman10x/openJev-verdict-2.0")
    token = os.environ.get("HF_TOKEN")
    cache = Path(os.environ.get("MODEL_CACHE_DIR", "./data/models"))
    cache.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo,
        token=token,
        cache_dir=str(cache),
        local_dir=str(cache / "snapshot"),
    )
    print(path)


if __name__ == "__main__":
    main()
