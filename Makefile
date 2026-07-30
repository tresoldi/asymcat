# ASymCat Makefile
# POSIX-compatible development commands

.PHONY: help quality security format test test-cov test-fast bump-version build build-release clean install install-dev site site-serve

# Default target: show help
.DEFAULT_GOAL := help

# Python interpreter
PYTHON := python3
PIP := $(PYTHON) -m pip

# Version bump type (patch, minor, major)
TYPE ?= patch

help: ## Show this help message
	@echo "ASymCat Development Commands"
	@echo "============================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Usage examples:"
	@echo "  make quality              # Run all quality checks"
	@echo "  make test-cov             # Run tests with coverage"
	@echo "  make bump-version TYPE=minor  # Bump minor version"
	@echo "  make build-release        # Full release build"

quality: ## Run code quality checks (ruff format --check, ruff check, mypy)
	@echo "==> Checking code formatting..."
	ruff format --check .
	@echo "==> Running ruff linter..."
	ruff check .
	@echo "==> Running mypy type checker..."
	mypy
	@echo "✓ All quality checks passed!"

security: ## Run bandit static security analysis
	@echo "==> Running bandit security scan..."
	bandit -c pyproject.toml -r src/asymcat/
	@echo "✓ Security scan passed!"

format: ## Auto-format code with ruff and apply safe lint fixes
	@echo "==> Formatting code with ruff..."
	ruff format .
	ruff check --fix .
	@echo "✓ Code formatted!"

test: ## Run test suite
	@echo "==> Running tests..."
	pytest tests/
	@echo "✓ Tests passed!"

test-cov: ## Run tests with coverage (HTML report in htmlcov/, fails if <80%)
	@echo "==> Running tests with coverage..."
	pytest --cov=asymcat --cov-report=html --cov-report=term-missing --cov-fail-under=80 tests/
	@echo "✓ Coverage report generated in htmlcov/"

test-fast: ## Run tests in parallel (faster)
	@echo "==> Running tests in parallel..."
	pytest -n auto tests/
	@echo "✓ Tests passed!"

bump-version: ## Bump version (TYPE=patch|minor|major), commit, and tag
	@CURRENT=$$(grep -o "__version__ = \"[^\"]*\"" src/asymcat/__init__.py | cut -d'"' -f2); \
	echo "==> Current version: $$CURRENT"; \
	IFS='.' read -r major minor patch <<< "$$CURRENT"; \
	if [ "$(TYPE)" = "major" ]; then NEW="$$((major + 1)).0.0"; \
	elif [ "$(TYPE)" = "minor" ]; then NEW="$$major.$$((minor + 1)).0"; \
	elif [ "$(TYPE)" = "patch" ]; then NEW="$$major.$$minor.$$((patch + 1))"; \
	else echo "Error: TYPE must be patch, minor, or major"; exit 1; fi; \
	echo "==> Bumping $(TYPE) version to $$NEW..."; \
	sed -i "s/__version__ = \"$$CURRENT\"/__version__ = \"$$NEW\"/" src/asymcat/__init__.py; \
	echo ""; \
	echo "⚠️  Please update CHANGELOG.md manually before committing!"; \
	echo ""; \
	read -p "Press Enter to commit and tag, or Ctrl+C to cancel..."; \
	git add src/asymcat/__init__.py; \
	git commit -m "chore: bump version to $$NEW"; \
	git tag -a "v$$NEW" -m "Release v$$NEW"; \
	echo "✓ Version bumped to $$NEW and tagged!"; \
	echo ""; \
	echo "Next steps:"; \
	echo "  1. Update CHANGELOG.md"; \
	echo "  2. git add CHANGELOG.md && git commit --amend --no-edit"; \
	echo "  3. git push && git push --tags"

build: ## Build package (creates dist/)
	@echo "==> Building package..."
	$(PYTHON) -m build
	@echo "✓ Package built in dist/"

build-release: clean quality test build ## Full release build (clean → quality → test → build)
	@echo "✓ Release build complete!"
	@echo ""
	@echo "Package ready in dist/"
	@ls -lh dist/

clean: ## Remove build artifacts, caches, and coverage reports
	@echo "==> Cleaning build artifacts..."
	rm -rf dist/ build/ *.egg-info src/*.egg-info
	rm -rf .coverage htmlcov/ coverage.xml coverage.lcov site/
	rm -rf .pytest_cache .ruff_cache .mypy_cache .benchmarks
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned!"

install: ## Install package in development mode
	@echo "==> Installing package..."
	$(PIP) install -e .
	@echo "✓ Package installed!"

install-dev: ## Install package with development dependencies (includes all Makefile tools)
	@echo "==> Installing package with dev dependencies..."
	$(PIP) install -e ".[dev,docs]"
	@echo "✓ Package installed with dev dependencies!"
	@echo ""
	@echo "Installed tools for Makefile:"
	@echo "  - pytest, pytest-cov, pytest-xdist (testing)"
	@echo "  - ruff, mypy, bandit (code quality)"
	@echo "  - build, twine (build/release)"
	@echo "  - mkdocs-material, mkdocstrings (docs site)"

site: ## Build the MkDocs documentation site (strict) into site/
	@echo "==> Building documentation site..."
	mkdocs build --strict
	@echo "✓ Site built in site/"

site-serve: ## Serve the docs site locally with live reload
	@echo "==> Serving docs at http://127.0.0.1:8000 ..."
	mkdocs serve
