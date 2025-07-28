PYTHON_BINARY := python3
VIRTUAL_ENV := venv
VIRTUAL_BIN := $(VIRTUAL_ENV)/bin
# Fallback to system python if venv doesn't exist
PYTHON := $(if $(wildcard $(VIRTUAL_BIN)/python),$(VIRTUAL_BIN)/python,$(PYTHON_BINARY))
PIP := $(if $(wildcard $(VIRTUAL_BIN)/pip),$(VIRTUAL_BIN)/pip,pip)
PROJECT_NAME := asymcat
TEST_DIR := tests

## help - Display help about make targets for this Makefile
help:
	@cat Makefile | grep '^## ' --color=never | cut -c4- | sed -e "`printf 's/ - /\t- /;'`" | column -s "`printf '\t'`" -t

## build - Builds the project in preparation for release
build:
	$(PYTHON) -m build

## coverage - Test the project and generate an HTML coverage report
coverage:
	$(PYTHON) -m pytest --cov=$(PROJECT_NAME) --cov-branch --cov-report=html --cov-report=lcov --cov-report=term-missing

## clean - Remove the virtual environment and clear out .pyc files
clean:
	rm -rf $(VIRTUAL_ENV) dist *.egg-info .coverage htmlcov .pytest_cache
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

## format - Auto-format code with ruff
format:
	$(PYTHON) -m ruff format $(PROJECT_NAME)/ $(TEST_DIR)/

## format-check - Check code formatting and linting with ruff
format-check:
	$(PYTHON) -m ruff check $(PROJECT_NAME)/ $(TEST_DIR)/
	$(PYTHON) -m ruff format --check $(PROJECT_NAME)/ $(TEST_DIR)/

## ruff-check - Check code quality with ruff (alias for format-check)
ruff-check: format-check

## install - Install the project locally
install:
	$(PYTHON_BINARY) -m venv $(VIRTUAL_ENV)
	$(VIRTUAL_BIN)/pip install -e ."[dev]"

## lint - Lint the project (ruff check)
lint:
	$(PYTHON) -m ruff check $(PROJECT_NAME)/ $(TEST_DIR)/

## ruff-fix - Auto-fix issues with ruff
ruff-fix:
	$(PYTHON) -m ruff check --fix $(PROJECT_NAME)/ $(TEST_DIR)/
	$(PYTHON) -m ruff format $(PROJECT_NAME)/ $(TEST_DIR)/

## mypy - Run mypy type checking on the project
mypy:
	$(PYTHON) -m mypy $(PROJECT_NAME)/ $(TEST_DIR)/

## test - Test the project
test:
	$(PYTHON) -m pytest

## docs - Build the documentation
docs:
	cd docs && make html

## docs-clean - Clean the documentation build
docs-clean:
	rm -rf docs/build/

## cli-test - Test CLI functionality with sample data
cli-test:
	@echo "Testing CLI with toy dataset..."
	$(PYTHON) -m $(PROJECT_NAME) resources/toy.tsv --scorers mle pmi --verbose
	@echo "\nTesting CLI with JSON output..."
	$(PYTHON) -m $(PROJECT_NAME) resources/toy.tsv --scorers mle --output-format json --top 3
	@echo "\nTesting CLI with smoothing..."
	$(PYTHON) -m $(PROJECT_NAME) resources/toy.tsv --scorers pmi_smoothed --smoothing laplace

## cli-help - Show CLI help
cli-help:
	$(PYTHON) -m $(PROJECT_NAME) --help

## quick-test - Run a quick subset of tests
quick-test:
	$(PYTHON) -m pytest tests/unit/test_data_loading.py tests/unit/test_scoring_measures.py::TestScorerInitialization -v

## security - Run security checks
security:
	$(PYTHON) -m bandit -r $(PROJECT_NAME)/ || echo "Install bandit for security scanning: pip install bandit"
	$(PYTHON) -m safety check || echo "Install safety for vulnerability scanning: pip install safety"

.PHONY: help build coverage clean format format-check ruff-check install lint ruff-fix mypy test docs docs-clean cli-test cli-help quick-test security
