# Contributing

Contributions are welcome. Please follow these guidelines when opening a PR.

## Prerequisites

- **Python 3.12+** — use `uv` for dependency management:

  ```bash
  uv sync --group dev
  ```

## Running tests

```bash
uv run pytest tests/
```

Enforce the coverage gate during development (PRD requires **≥ 95 % line coverage** on `app/`):

```bash
uv run pytest --cov=app --cov-report=term-missing --cov-full-scan --cov-fail-under=95 tests/
```

## Code quality

Every PR must pass:

```bash
uv run ruff format .
uv run ruff check  .
```

Both CI jobs (lint + coverage) **fail on merge** if these exit non-zero.

## Pull request expectations

1. **Small, focused** — one logical change per PR. Avoid bundling unrelated fixes.
2. **Tests for new code** — any new logic must have tests; coverage cannot drop below 95 %.
3. **Describe the why** — include motivation and context in the PR body, not just what changed.
4. **References** — link to the relevant ticket (e.g. `T5a`) and/or issue in the description.

## Development workflow

```bash
# Create a feature branch
git checkout -b ai-agent/<ticket> main

# Make changes, test locally
uv run pytest --cov=app tests/
uv run ruff check .

# Commit with a descriptive message
git commit -m "feat(<ticket>): ..."
git push origin ai-agent/<ticket>

# Open PR against `main` targeting the relevant ticket
```
