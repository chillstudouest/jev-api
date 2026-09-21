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

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU wheels only — no CUDA. Von OptionMarker 395M.
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
    VON_BACKEND=option-marker \
    JEV_DEVICE=cpu \
    HOST=0.0.0.0 \
    PORT=8000 \
    DOWNLOAD_ON_STARTUP=true

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin jev \
    && mkdir -p /data/models \
    && chown -R jev:jev /data

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=jev:jev src ./src
COPY --chown=jev:jev pyproject.toml README.md ./

ENV PYTHONPATH=/app/src

USER jev
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=8 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "jev_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
