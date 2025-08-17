FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests

RUN python -m pip install -U pip && pip install -e ".[dev]"

CMD ["transcript-parser", "--help"]
