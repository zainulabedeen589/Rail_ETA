# Contributing Guidelines — Rail ETA

Thank you for considering contributing to Rail ETA! This project follows
senior engineering standards, so please read the guidelines below before
submitting changes.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions. Harassment,
discrimination, or toxic behaviour will not be tolerated.

## How to Contribute

### 1. Fork and Clone

```bash
git clone https://github.com/zainulabedeen589/Rail_ETA.git
cd Rail_ETA-main
```

### 2. Create a Branch

Use a descriptive branch name:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix-name
```

### 3. Set Up the Development Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Make Your Changes

- Follow PEP 8 and the project's `pyproject.toml` linting rules.
- Add type hints to all new functions.
- Write tests for new functionality in `tests/`.
- Update documentation in `docs/` if your change affects users.

### 5. Run Quality Checks

Before committing, run:

```bash
make lint          # ruff + black
make typecheck     # mypy
make test          # pytest
make ci            # lint + typecheck + test
```

Or individually:

```bash
ruff check src tests
black --check src tests
mypy src
pytest tests -v
```

### 6. Commit

Write clear, conventional commit messages:

```
feat: add SHAP explanation endpoint
fix: correct midnight-crossing edge case
docs: update API documentation
test: add coverage for predictor batch mode
```

### 7. Push and Open a Pull Request

```bash
git push origin feature/your-feature-name
```

Open a PR against `main`. Describe your changes, link any related issues, and
request review.

## Project Structure

```
src/
├── api_main.py          # FastAPI service
├── base.py              # Model registry & tuning
├── builder.py           # Feature engineering
├── cli.py               # CLI entry point
├── config.py            # Pydantic settings
├── data_loader.py       # CSV loading + validation
├── exceptions.py        # Custom exceptions
├── logger.py            # Structured logging
├── metrics.py           # Regression metrics
├── pipeline.py          # End-to-end orchestration
├── predictor.py         # Single / batch inference
├── training_pipeline.py # Model training logic
├── time_utils.py        # Time parsing & journey duration
└── visuals.py           # Diagnostic charts
```

## Adding a New Model

1. Register a `ModelSpec` in `base.py` (`ModelRegistry._register_defaults`).
2. The trainer (`training_pipeline.py`) automatically picks it up.
3. Add tests in `tests/test_core.py`.
4. Update `docs/METHODOLOGY.md`.

## Reporting Issues

Use the GitHub issue tracker. Include:

- Steps to reproduce.
- Expected vs actual behaviour.
- Python version and OS.
- Relevant logs or tracebacks.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.
