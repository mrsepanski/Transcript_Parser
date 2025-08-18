# syntax=docker/dockerfile:1
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends     poppler-utils     libgl1     libglib2.0-0     build-essential     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
ENV PYTHONPATH="/app/src"
RUN python -m pip install --upgrade pip setuptools wheel && pip install -e ".[dev]"
ENTRYPOINT ["transcript-parser"]
