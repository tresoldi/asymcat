# Contributing to ASymCat

Thanks for your interest in improving ASymCat! This document explains how to set
up a development environment and the checks your changes are expected to pass.

## Development setup

ASymCat requires Python 3.10 or newer.

```bash
# Clone your fork and enter the directory
git clone https://github.com/<your-username>/asymcat.git
cd asymcat

# Install the package with all development dependencies
pip install -e ".[dev]"

# (Optional but recommended) install the pre-commit hooks
pre-commit install
```

Installing with the `dev` extra pulls in everything the test suite and the
tooling need: `pytest`, `pytest-cov`, `pytest-xdist`, `ruff`, `mypy`, `bandit`,
`build`, and `twine`. If you only want to run the tests, the lighter
`pip install -e ".[test]"` is enough. The documentation site additionally needs
the `docs` extra (`pip install -e ".[docs]"`).

## Quality checks

All of the checks below run in CI. You can run them locally through the
`Makefile` targets:

```bash
make quality   # ruff format --check, ruff check, mypy
make security  # bandit static security analysis
make test      # run the test suite
make test-cov  # run the test suite with a coverage report (fails under 80%)
```

Or invoke the tools directly:

```bash
ruff format --check .
ruff check .
mypy
bandit -c pyproject.toml -r src/asymcat/
pytest tests/ --cov=asymcat --cov-fail-under=80
```

If you installed the pre-commit hooks, `ruff`, `ruff format`, and `mypy` also run
automatically on every commit.

## Making changes

1. Create a feature branch off `master`.
2. Make your change, adding or updating tests to cover it. New behavior should
   keep overall coverage at or above 80%.
3. Update `CHANGELOG.md` under the `[Unreleased]` section.
4. Ensure `make quality`, `make security`, and `make test-cov` all pass.
5. Open a pull request describing the motivation and the change.

## Project layout

- `src/asymcat/` — the package (`common` for I/O and co-occurrence collection,
  `scorer` for the `CatScorer` measures, `correlation` for sequence helpers).
- `tests/` — `unit/`, `integration/`, and `performance/` suites with shared
  fixtures.
- `docs/` — the MkDocs site (`USER_GUIDE.md`, `TRESOLDI_MEASURE.md`,
  `reference.md`); `docs/archive/` holds the retired Sphinx docs and tutorials.
- `resources/` — sample datasets used by the docs, tests, and examples.

See [ARCHITECTURE.md](ARCHITECTURE.md) for a fuller tour of the codebase.

## Reporting bugs and requesting features

Please use the [issue tracker](https://github.com/tresoldi/asymcat/issues). For
bug reports, include a minimal reproducible example, the ASymCat version, and
your Python version.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License that covers the project.
