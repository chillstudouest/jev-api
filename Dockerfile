# syntax=docker/dockerfile:1.7

# Base = python 3.12 + build toolchain + git + venv + torch (CPU).
# Built by .github/workflows/base-image.yml from Dockerfile.base and kept on the
# build host (label coolify.managed=true). Bump the tag here AND in that
# workflow whenever python/torch versions change.
FROM ghcr.io/chillstudouest/jev-api-base:py3.12-torch-cpu AS builder

# Layer 2 — SemIf source (changes rarely).
RUN git clone --depth 1 https://github.com/TheoLeeCJ/SemIf.git /opt/semif

# Layer 3 — app code (changes on every deploy). Only this re-runs.
WORKDIR /appbuild
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install .

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_CACHE_DIR=/data/models \
    HF_HOME=/data/models \
    HUGGINGFACE_HUB_CACHE=/data/models \
    VON_BACKEND=von \
    PRELOAD_LAYA=false \
    PRELOAD_SEMIF=false \
    SEMIF_SRC=/opt/semif \
    JEV_DEVICE=cpu \
    HOST=0.0.0.0 \
    PORT=8000 \
    DOWNLOAD_ON_STARTUP=true

# Coolify Dockerfile healthchecks need curl AND wget in the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      wget \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin jev \
    && mkdir -p /data/models \
    && chown -R jev:jev /data

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/semif /opt/semif

WORKDIR /app
COPY --chown=jev:jev src ./src
COPY --chown=jev:jev pyproject.toml README.md ./

ENV PYTHONPATH=/app/src:/opt/semif/src

USER jev
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=10 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "jev_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
