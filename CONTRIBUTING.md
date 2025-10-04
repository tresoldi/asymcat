# Contributing to ASymCat

Thank you for your interest in contributing to ASymCat! This guide will help you get started with development and ensure your contributions align with project standards.

## Table of Contents

- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Quality Standards](#code-quality-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Make (optional but recommended)

### Initial Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/asymcat.git
   cd asymcat
   ```

2. **Install development dependencies:**
   ```bash
   make install-dev
   ```

   This creates a virtual environment and installs the package with all development dependencies.

   Alternatively, without Make:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Verify installation:**
   ```bash
   make quality
   make test-cov
   ```

## Development Workflow

### Common Commands

The Makefile provides convenient commands for development:

```bash
make help           # Show all available commands
make quality        # Run all quality checks (format-check + lint + typecheck)
make test-cov       # Run tests with coverage report
make format         # Auto-format code with ruff
make ruff-fix       # Auto-fix linting issues and format
make mypy           # Run type checking
make test-fast      # Run tests in parallel
```

### Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-new-measure` for new features
- `fix/bug-description` for bug fixes
- `docs/improvement-description` for documentation
- `refactor/component-name` for refactoring

## Code Quality Standards

All code must pass the following checks before being merged:

### 1. Code Formatting (Ruff)

```bash
make format          # Auto-format code
make format-check    # Check formatting without changes
```

Ruff replaces black and isort for unified formatting.

### 2. Linting (Ruff)

```bash
make lint           # Check for linting issues
make ruff-fix       # Auto-fix issues and format
```

### 3. Type Checking (MyPy)

```bash
make mypy
```

All functions should have type hints:
```python
def calculate_score(x: str, y: str, alpha: float = 1.0) -> float:
    """Calculate association score.

    @param x: First category
    @param y: Second category
    @param alpha: Smoothing parameter
    @return: Association score
    """
    ...
```

### 4. Test Coverage

Minimum coverage: **78%** (goal: **80%**)

```bash
make test-cov
```

### Pre-Commit Checklist

Before committing, run:
```bash
make quality && make test-cov
```

This ensures your code passes all checks.

## Testing Guidelines

### Test Organization

Tests are organized by type:
```
tests/
├── unit/              # Fast, isolated unit tests
├── integration/       # Integration tests
└── conftest.py        # Shared fixtures
```

### Writing Tests

Use pytest with descriptive test names:

```python
import pytest
from asymcat import CatScorer

def test_mle_scores_are_asymmetric():
    """Test that MLE scores differ for X→Y vs Y→X."""
    cooccs = {("A", "B"): (10, 5), ("B", "A"): (5, 10)}
    scorer = CatScorer(cooccs)
    scores = scorer.mle()

    xy_score, yx_score = scores[("A", "B")]
    assert xy_score != yx_score

def test_smoothing_prevents_zero_probabilities():
    """Test that smoothing handles zero counts correctly."""
    cooccs = {("A", "B"): (0, 5)}
    scorer = CatScorer(cooccs, smoothing_method="laplace", smoothing_alpha=1.0)
    scores = scorer.mle()

    xy_score, yx_score = scores[("A", "B")]
    assert xy_score > 0.0
```

### Test Markers

Use markers to categorize tests:

```python
@pytest.mark.slow
def test_large_dataset_performance():
    """Test performance with large dataset."""
    ...

@pytest.mark.integration
def test_full_workflow():
    """Test complete analysis workflow."""
    ...
```

Run specific test categories:
```bash
pytest -m "not slow"      # Skip slow tests
pytest -m integration     # Only integration tests
pytest tests/unit/        # Only unit tests
```

### Parametrized Tests

Use parametrization for testing multiple cases:

```python
@pytest.mark.parametrize("smoothing_method,expected_min", [
    ("mle", 0.0),
    ("laplace", 0.01),
    ("lidstone", 0.005),
])
def test_smoothing_methods(smoothing_method, expected_min):
    """Test different smoothing methods."""
    scorer = CatScorer(cooccs, smoothing_method=smoothing_method)
    scores = scorer.mle()
    assert min(s for pair in scores.values() for s in pair) >= expected_min
```

## Submitting Changes

### 1. Ensure Quality

Before submitting a pull request:

```bash
# Run all quality checks
make quality

# Run tests with coverage
make test-cov

# If adding notebooks, execute them
make docs-execute

# Validate notebooks have outputs
make docs-validate
```

### 2. Commit Guidelines

Use conventional commit messages:

```
feat: add Goodman-Kruskal tau measure
fix: correct PMI calculation for zero probabilities
docs: update usage examples in README
test: add coverage for smoothing edge cases
refactor: simplify scorer matrix generation
```

### 3. Pull Request Process

1. **Update documentation** if adding features
2. **Add tests** for new functionality
3. **Update CHANGELOG.md** under `[Unreleased]` section
4. **Create pull request** with clear description

Pull request template:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass (`make test-cov`)
- [ ] Coverage maintained/improved
- [ ] Added tests for new functionality

## Checklist
- [ ] Code follows style guidelines (`make quality`)
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### 4. Code Review

Maintainers will review your PR and may request changes. Common feedback includes:
- Adding type hints
- Improving test coverage
- Clarifying documentation
- Simplifying complex logic

## Release Process

Releases are managed by maintainers following this workflow:

### 1. Version Bumping

```bash
# Patch release (0.4.0 → 0.4.1)
make bump-version TYPE=patch

# Minor release (0.4.0 → 0.5.0)
make bump-version TYPE=minor

# Major release (0.4.0 → 1.0.0)
make bump-version TYPE=major
```

This will:
1. Update `asymcat/__init__.py` and `pyproject.toml`
2. Prompt for CHANGELOG.md update
3. Create commit and git tag
4. Display push instructions

### 2. Update CHANGELOG

Move items from `[Unreleased]` to new version section:

```markdown
## [Unreleased]

## [0.5.0] - 2025-02-15

### Added
- New association measure: Goodman-Kruskal tau

### Fixed
- PMI calculation for zero probabilities
```

### 3. Full Release Build

```bash
# Clean → Quality → Tests → Build
make build-release
```

### 4. Publish to PyPI

```bash
# Upload to PyPI (maintainers only)
python -m twine upload dist/*
```

### 5. Create GitHub Release

1. Push tag: `git push --tags`
2. Create GitHub release from tag
3. Upload dist files
4. Copy CHANGELOG entry to release notes

## Questions or Issues?

- **Bug reports**: [GitHub Issues](https://github.com/tresoldi/asymcat/issues)
- **Feature requests**: [GitHub Discussions](https://github.com/tresoldi/asymcat/discussions)
- **Questions**: Open a discussion or issue

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn and contribute
- Credit others' work appropriately

---

Thank you for contributing to ASymCat! 🎉
