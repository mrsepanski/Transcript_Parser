# Contributing

## Required checks before merging
Run locally and ensure CI passes:
1. `black .`
2. `isort .`
3. `autoflake --check --recursive src tests`
4. `mypy src`
5. `pytest`

## Style guide
- Add docstrings/comments for functions
- Type all inputs/outputs (avoid `Any`)
- Prefer package imports over relative imports
- Use vectorized pandas ops when possible
- Fix all warnings produced by tests
- Consider existing well-supported libraries before reinventing

## Dev install (Windows PowerShell)
```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install -U pip; pip install -e ".[dev]"
```
