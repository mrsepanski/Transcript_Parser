# Dev-friendly image for transcript_parser
FROM python:3.12-slim

# System deps for pdf2image (poppler) and OpenCV headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip tooling first
RUN python -m pip install --upgrade pip setuptools wheel

# Copy lightweight files first for better caching of Python deps
COPY pyproject.toml README.md /app/

# Copy source and dev assets
COPY src /app/src
COPY tests /app/tests
COPY tools /app/tools
COPY .pre-commit-config.yaml /app/.pre-commit-config.yaml

# Install package in editable mode with dev extras for tests/hooks
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[dev]"

# Default command: drop into a shell; override in `docker run` to call the CLI
CMD ["bash"]