# Transcript_Parser

This repository contains a ready-to-develop Python **package** (`transcript_parser`) with:
- Dev tools: `black`, `isort`, `autoflake`, `mypy`, `pytest`
- GitHub Actions CI that runs all required checks on push/PR
- Dockerfile for containerized dev/tests
- CONTRIBUTING and AGENTS guidance

## Quick start (Windows PowerShell)

```powershell
# 1) Create and activate a virtual environment
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1

# 2) Install the package + dev dependencies
python -m pip install -U pip; pip install -e ".[dev]"

# 3) Run required checks
black .; isort .; autoflake --check --recursive src tests; mypy src; pytest -q

# 4) Try the CLI
transcript-parser --name World
```

## Docker

```powershell
docker build -t transcript_parser:dev .; docker run --rm -it -v ${PWD}:/app transcript_parser:dev transcript-parser --name World
# or run tests in the container
docker run --rm -it -v ${PWD}:/app transcript_parser:dev pytest -q
```

## Repo layout
```
src/transcript_parser/   # package code
tests/                   # unit tests
.github/workflows/       # CI
```
