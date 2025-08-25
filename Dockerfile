# syntax=docker/dockerfile:1.7-labs
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

# System deps: poppler for pdf2image; libs for opencv
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt     apt-get update &&     apt-get install -y --no-install-recommends       poppler-utils       libglib2.0-0       libsm6       libxrender1       libxext6       curl ca-certificates &&     rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency files first for better caching
COPY pyproject.toml README.md /app/

# Install build tools for wheels if needed
RUN python -m pip install --upgrade pip setuptools wheel

# Install project (this will pull paddlepaddle/paddleocr from pyproject deps)
RUN --mount=type=cache,target=/root/.cache/pip     pip install .

# Now copy the rest of the source
COPY src /app/src

# Default command shows CLI help
CMD ["transcript-parser", "--help"]