    # syntax=docker/dockerfile:1
    FROM python:3.11-slim

    ENV PYTHONDONTWRITEBYTECODE=1         PYTHONUNBUFFERED=1         PIP_NO_CACHE_DIR=1

    # System libs:
    # - poppler-utils: for pdf2image (pdftoppm)
    # - libgl1, libglib2.0-0: OpenCV runtime for PaddleOCR
    # - libgomp1: OpenMP runtime required by PaddlePaddle
    RUN apt-get update && apt-get install -y --no-install-recommends         poppler-utils         libgl1         libglib2.0-0         libgomp1         && rm -rf /var/lib/apt/lists/*

    WORKDIR /app

    # Copy project files
    COPY pyproject.toml README.md /app/
    COPY src /app/src

    ENV PYTHONPATH="/app/src"

    # Toolchain & ABI-safe numpy first, then project + OCR stack from official PyPI
    RUN python -m pip install --upgrade pip setuptools wheel &&         pip install --index-url https://pypi.org/simple "numpy<2" &&         pip install -e ".[dev]" &&         pip install --index-url https://pypi.org/simple paddlepaddle==2.6.1 paddleocr==2.7.3 "opencv-python-headless<5"

    # Optional: pre-warm OCR models so first run doesn't download.
    # If this fails, don't fail the build; the first run will download instead.
    RUN python - <<'PY' || true
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
print('PaddleOCR models prewarmed.')
PY

    ENTRYPOINT ["python"]
