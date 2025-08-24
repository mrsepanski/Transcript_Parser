
# Dockerfile (CPU, PaddleOCR included by default)
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# System dependencies:
# - poppler-utils: needed by pdf2image (pdftoppm)
# - ghostscript: occasionally useful for PDF normalization
# - libglib2.0-0, libsm6, libxext6, libxrender1: OpenCV runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils ghostscript \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata early for layer caching
COPY pyproject.toml README.md /app/

# Install the package with dev + paddle extras so OCR fallback is always available
RUN python -m pip install --upgrade pip && pip install -e ".[dev,paddle]"

# Copy source last (fewer cache busts)
COPY src /app/src

# Default command shows CLI help; override at runtime to parse files
CMD ["python", "-m", "transcript_parser.parse_transcript", "-h"]
