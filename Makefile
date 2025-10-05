# Makefile for asymcat development
.DEFAULT_GOAL := help
.PHONY: help quality format format-check lint mypy test test-cov test-fast bump-version build build-release clean install install-dev docs docs-clean security

# Variables
PYTHON_BINARY := python3
VIRTUAL_ENV := venv
VIRTUAL_BIN := $(VIRTUAL_ENV)/bin
PYTHON := $(if $(wildcard $(VIRTUAL_BIN)/python),$(VIRTUAL_BIN)/python,$(PYTHON_BINARY))
PIP := $(if $(wildcard $(VIRTUAL_BIN)/pip),$(VIRTUAL_BIN)/pip,pip)
PROJECT_NAME := asymcat
TEST_DIR := tests
TYPE ?= patch  # For version bumping: patch, minor, or major

# Self-documenting help
help: ## Show this help message
	@echo "ASymCat Development Commands"
	@echo "============================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Common workflows:"
	@echo "  make install-dev    # First-time setup"
	@echo "  make quality        # Run all checks before commit"
	@echo "  make test-cov       # Test with coverage report"
	@echo "  make bump-version TYPE=minor  # Bump version and tag"

# Quality checks
quality: ## Run all code quality checks (format-check + lint + typecheck)
	@echo "🔍 Running code quality checks..."
	@echo "1. Checking formatting..."
	$(PYTHON) -m ruff format --check $(PROJECT_NAME)/ $(TEST_DIR)/
	@echo "2. Linting..."
	$(PYTHON) -m ruff check $(PROJECT_NAME)/ $(TEST_DIR)/
	@echo "3. Type checking..."
	$(PYTHON) -m mypy $(PROJECT_NAME)/ $(TEST_DIR)/
	@echo "✅ All quality checks passed!"

format: ## Auto-format code with ruff
	$(PYTHON) -m ruff format $(PROJECT_NAME)/ $(TEST_DIR)/

format-check: ## Check code formatting without modifying
	$(PYTHON) -m ruff format --check $(PROJECT_NAME)/ $(TEST_DIR)/

lint: ## Lint code with ruff
	$(PYTHON) -m ruff check $(PROJECT_NAME)/ $(TEST_DIR)/

ruff-fix: ## Auto-fix ruff issues and format
	$(PYTHON) -m ruff check --fix $(PROJECT_NAME)/ $(TEST_DIR)/
	$(PYTHON) -m ruff format $(PROJECT_NAME)/ $(TEST_DIR)/

mypy: ## Run mypy type checking
	$(PYTHON) -m mypy $(PROJECT_NAME)/ $(TEST_DIR)/

# Testing
test: ## Run test suite
	$(PYTHON) -m pytest

test-cov: ## Run tests with coverage report (HTML + terminal)
	$(PYTHON) -m pytest \
		--cov=$(PROJECT_NAME) \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-fail-under=78

test-fast: ## Run tests in parallel (requires pytest-xdist)
	$(PYTHON) -m pytest -n auto

coverage: test-cov ## Alias for test-cov

# Version management
bump-version: ## Bump version (TYPE=patch|minor|major), commit, and tag
	@CURRENT=$$(grep -o "__version__ = \"[^\"]*\"" $(PROJECT_NAME)/__init__.py | cut -d'"' -f2); \
	echo "==> Current version: $$CURRENT"; \
	IFS='.' read -r major minor patch <<< "$$CURRENT"; \
	if [ "$(TYPE)" = "major" ]; then NEW="$$((major + 1)).0.0"; \
	elif [ "$(TYPE)" = "minor" ]; then NEW="$$major.$$((minor + 1)).0"; \
	elif [ "$(TYPE)" = "patch" ]; then NEW="$$major.$$minor.$$((patch + 1))"; \
	else echo "❌ Error: TYPE must be patch, minor, or major"; exit 1; fi; \
	echo "==> Bumping $(TYPE) version: $$CURRENT → $$NEW"; \
	sed -i "s/__version__ = \"$$CURRENT\"/__version__ = \"$$NEW\"/" $(PROJECT_NAME)/__init__.py; \
	sed -i "s/version = \"$$CURRENT\"/version = \"$$NEW\"/" pyproject.toml; \
	echo ""; \
	echo "⚠️  IMPORTANT: Please update CHANGELOG.md manually:"; \
	echo "   1. Move items from [Unreleased] to [$$NEW]"; \
	echo "   2. Add release date"; \
	echo "   3. Review migration guide if needed"; \
	echo ""; \
	read -p "Press Enter when CHANGELOG is ready, or Ctrl+C to cancel..."; \
	git add $(PROJECT_NAME)/__init__.py pyproject.toml CHANGELOG.md; \
	git commit -m "chore: bump version to $$NEW"; \
	git tag -a "v$$NEW" -m "Release v$$NEW"; \
	echo "✅ Version bumped to $$NEW and tagged!"; \
	echo "📋 Next steps:"; \
	echo "   git push && git push --tags"

# Build
build: ## Build package (creates dist/)
	$(PYTHON) -m build

build-release: clean quality test build ## Full release build (clean → quality → test → build)
	@echo "✅ Release build complete!"
	@ls -lh dist/

# Installation
install: ## Install package in development mode
	$(PYTHON_BINARY) -m venv $(VIRTUAL_ENV)
	$(VIRTUAL_BIN)/pip install -e .

install-dev: ## Install package with all development dependencies
	$(PYTHON_BINARY) -m venv $(VIRTUAL_ENV)
	$(VIRTUAL_BIN)/pip install -e ".[dev]"

# Cleanup
clean: ## Remove build artifacts, caches, and coverage reports
	rm -rf dist/ build/ *.egg-info
	rm -rf .coverage htmlcov/ coverage.xml coverage.lcov
	rm -rf .pytest_cache .ruff_cache .mypy_cache .hypothesis/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete

# Documentation
docs: ## Generate HTML documentation from Nhandu tutorial sources
	@echo "🔄 Generating tutorial documentation..."
	@for f in docs/tutorial_1_basics.py docs/tutorial_2_advanced_measures.py docs/tutorial_3_visualization.py docs/tutorial_4_real_world.py; do \
		echo "  Generating $$(basename $$f .py).html..."; \
		$(PYTHON) -m nhandu "$$f" --format html --working-dir . -o "docs/$$(basename $$f .py).html"; \
	done
	@echo "✅ Documentation generated in docs/"

docs-clean: ## Remove generated HTML documentation
	@echo "🧹 Cleaning generated documentation..."
	rm -f docs/tutorial_1_basics.html docs/tutorial_2_advanced_measures.html docs/tutorial_3_visualization.html docs/tutorial_4_real_world.html
	rm -rf docs/_build/ docs/build/
	@echo "✅ Documentation cleaned!"

# Security
security: ## Run security checks (bandit + safety)
	@echo "🔒 Running security scans..."
	$(PYTHON) -m bandit -r $(PROJECT_NAME)/ || echo "⚠️  Install bandit: pip install bandit"
	$(PYTHON) -m safety check || echo "⚠️  Install safety: pip install safety"
