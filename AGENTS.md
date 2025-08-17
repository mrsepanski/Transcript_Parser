# AGENTS.md

You are a Python developer working on this repository.

ALWAYS ask for clarification if unsure of the task.

## CONTRIBUTION GUIDES

### STYLE GUIDE
1. ALWAYS add function comments explaining what the function does.
2. ALWAYS type all function inputs/outputs. Avoid `Any`.
3. Use `logger = logging.getLogger(__name__)` when logging is needed.
4. ALWAYS use package imports instead of relative imports.
5. Prefer vectorized pandas operations when possible.
6. Fix ALL warnings produced by tests.
7. BEFORE adding a new dependency, check maintenance status, popularity, and security.

### TESTING
Include positive cases, randomized cases (if possible), edge cases, and negative cases that assert the expected error.

### REQUIRED CHECKS (pre-merge)
1. black
2. isort
3. autoflake
4. mypy
5. pytest
