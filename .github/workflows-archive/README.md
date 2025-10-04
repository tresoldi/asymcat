# Archived Workflows

These workflows were archived during the v0.4.0 modernization (October 2024).

## Replaced by

**quality.yml** - Simplified workflow with:
- Ruff formatting check
- Ruff linting
- MyPy type checking
- Pytest with coverage (78% threshold)
- Python 3.10, 3.11, 3.12 matrix testing

## Archived Files

- **build.yml.old** - Complex multi-job workflow with:
  - Lint job (replaced by quality.yml ruff checks)
  - Security job (moved to release.yml)
  - Dependency check job (moved to release.yml)
  - Test job with matrix (replaced by quality.yml)
  - Notebooks job (still runs in quality.yml via pytest)

- **security.yml.old** - Standalone security scanning
  - Bandit, Safety, pip-audit
  - Now integrated into release.yml only

## Why Simplified?

The v0.4.0 modernization focused on:
1. **Simplicity** - One clear workflow for quality checks
2. **Speed** - Fewer jobs, faster feedback
3. **Maintainability** - Easier to understand and update
4. **Template alignment** - Following modern Python project standards

## Running Archived Checks Manually

If you need to run these checks:

```bash
# Security scanning
make security  # or manually: bandit -r asymcat/ && safety check

# All quality checks
make quality  # runs ruff format --check, ruff check, mypy

# Tests with coverage
make test-cov
```

## Restoration

If you need to restore these workflows, copy from this archive and update
dependency installation to use `pip install -e ".[dev]"` instead of the
old granular dependency groups.
