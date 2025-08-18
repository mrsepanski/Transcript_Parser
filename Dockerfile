# Dockerfile — ocr fallback Paddle-only (no Tesseract). CPU build.
FROM python:3.11-slim
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# System deps for pdf2image and Paddle (no Tesseract)
RUN apt-get update && apt-get install -y --no-install-recommends     poppler-utils     libgl1     libglib2.0-0     libgomp1     ca-certificates     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Copy tests if present (ok if absent)
COPY tests ./tests

# Install project with Paddle extras
RUN python -m pip install -U pip && pip install -e ".[dev,paddle]"

# Default to CLI help
CMD ["transcript-parser","--help"]
