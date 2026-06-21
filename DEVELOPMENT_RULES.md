* If you need, run the code with `uv run python`
* Format and check the code when you're done, and fix the warnings and errors, with:
```
uv run ruff format .
uv run ruff check .
```
* Run coverage with:
```
uv run pytest --cov=app --cov-report=term-missing --cov-fail-under=95 tests
```
