# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN git clone --depth 1 https://github.com/malevrigns/agent-jev.git /opt/agent-jev \
 && git clone --depth 1 https://github.com/TheoLeeCJ/SemIf.git /opt/semif

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU wheels only — no CUDA.
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install .

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_CACHE_DIR=/data/models \
    HF_HOME=/data/models \
    HUGGINGFACE_HUB_CACHE=/data/models \
    VON_BACKEND=von \
    PRELOAD_LAYA=false \
    PRELOAD_AGENTJEV=false \
    PRELOAD_SEMIF=false \
    AGENTJEV_SRC=/opt/agent-jev \
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
COPY --from=builder /opt/agent-jev /opt/agent-jev
COPY --from=builder /opt/semif /opt/semif

WORKDIR /app
COPY --chown=jev:jev src ./src
COPY --chown=jev:jev pyproject.toml README.md ./

ENV PYTHONPATH=/app/src:/opt/agent-jev:/opt/semif/src

USER jev
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=10 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "jev_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
