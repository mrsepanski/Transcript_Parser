# syntax=docker/dockerfile:1

FROM python:3.11-slim

# System deps:
# - poppler-utils: needed by pdf2image to convert PDF -> image
# - libgl1, libglib2.0-0: commonly required by OpenCV (pulled in via paddleocr)
# - build-essential: for any wheels that need compiling fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy minimal metadata first to leverage Docker layer caching
COPY pyproject.toml README.md /app/
# Copy package source
COPY transcript_parser /app/transcript_parser

# Upgrade pip tooling and install project with dev deps (for parity with CI)
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -e ".[dev]"

# Default CLI entrypoint
ENTRYPOINT ["transcript-parser"]
