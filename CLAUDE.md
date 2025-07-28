# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASymCat is a Python library for analyzing asymmetric associations between categorical variables. It implements 15+ association measures and provides both Python API and CLI interfaces for categorical data analysis.

**Core Architecture:**
- `asymcat/common.py`: Data loading, preprocessing, and co-occurrence collection
- `asymcat/scorer.py`: CatScorer class with all association measure implementations
- `asymcat/correlation.py`: Correlation analysis utilities
- `asymcat/__main__.py`: Command-line interface implementation

**Key Concepts:**
- **Co-occurrences**: Pairs of categorical values that appear together in sequences
- **Asymmetric measures**: Directional measures where X→Y ≠ Y→X (e.g., MLE, PMI, Theil's U)
- **Symmetric measures**: Non-directional measures (e.g., Chi-square, Cramér's V)
- **Scoring**: Converting co-occurrence counts to association strength measures
- **Smoothing**: Probability estimation methods (MLE, Laplace, Lidstone) for handling sparse data

## Development Commands

For a coding agent, always use the "env\_lingfil" environment from pyenv.

### Environment Setup
```bash
# Create virtual environment and install dev dependencies (legacy)
make install

# Using Hatch (recommended)
pip install hatch
hatch env create

# Install with all optional features
pip install -e ".[all]"
# or with Hatch:
hatch env create all
```

### Testing
```bash
# Run full test suite (legacy)
make test
# or
pytest

# Using Hatch (recommended)
hatch run test
hatch run test-cov  # with coverage

# Run with coverage report (legacy)
make coverage
# or (if pytest-cov is installed)
pytest --cov=asymcat --cov-branch --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m slow          # Slow tests only

# Quick testing (legacy)
make quick-test         # Run essential tests quickly
```

### Code Quality
```bash
# Run all formatting and linting (legacy)
make format-check

# Using Hatch (recommended)
hatch run format-check
hatch run all-checks  # format, lint, typecheck, security, tests

# Individual tools (legacy)
make black-check    # Code formatting check
make isort-check    # Import sorting check
make lint          # Flake8 linting
make mypy          # Type checking

# Using Hatch
hatch run format
hatch run lint
hatch run typecheck
hatch run security

# Auto-fix formatting issues (legacy)
make format        # Runs black, isort, lint, mypy
make black         # Auto-format code
make isort         # Auto-sort imports

# Security checks (legacy)
make security      # Run bandit and safety scans
```

### Build and Release
```bash
# Build package (legacy)
make build

# Using Hatch (recommended)
hatch build

# Build documentation (legacy)
make docs
# Clean docs
make docs-clean

# Using Hatch for docs
hatch run docs:build
hatch run docs:clean
hatch run docs:serve  # Serve locally on port 8000
```

### Local GitHub Actions Testing

**IMPORTANT**: Always test GitHub Actions workflows locally before pushing to ensure they work correctly.

```bash
# Install act (GitHub Actions local runner)
curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | bash -s -- -b ~/.local/bin
export PATH="$PATH:$HOME/.local/bin"

# Run complete build workflow locally (recommended)
./scripts/test-local.sh build

# Run individual jobs
./scripts/test-local.sh lint      # Linting only
./scripts/test-local.sh test      # Tests only  
./scripts/test-local.sh security  # Security scans only
./scripts/test-local.sh notebooks # Notebook execution only

# List all available commands
./scripts/test-local.sh list

# Run specific workflow manually
act -W .github/workflows/build.yml --env-file .env

# Test release workflow (use with caution)
./scripts/test-local.sh release
```

**Configuration Files:**
- `.actrc`: Act configuration matching GitHub Actions environment
- `.env`: Environment variables for local testing
- `scripts/test-local.sh`: Comprehensive local testing script

This setup ensures **exact version and configuration matching** between local testing and GitHub Actions CI/CD.

### CLI Testing
```bash
# Test CLI functionality
python -m asymcat --help
make cli-help  # Alternative using Makefile

# Basic usage examples
python -m asymcat resources/toy.tsv --scorers mle pmi
python -m asymcat resources/galapagos.tsv --format pa-matrix --scorers chi2 fisher

# Advanced features
python -m asymcat resources/toy.tsv --scorers mle pmi_smoothed --smoothing laplace
python -m asymcat resources/toy.tsv --scorers mle --smoothing lidstone --smoothing-alpha 0.5
python -m asymcat resources/toy.tsv --scorers mle --sort-by yx --top 5
python -m asymcat resources/toy.tsv --scorers all --output-format json --output results.json

# Test CLI with Makefile
make cli-test  # Comprehensive CLI testing
```

## Code Architecture

### Data Flow
1. **Input**: Raw categorical data (TSV sequences or presence-absence matrices)
2. **Collection**: `collect_cooccs()` extracts co-occurrence pairs with counts
3. **Scoring**: `CatScorer` class applies various association measures
4. **Output**: Directional scores (X→Y, Y→X) for each category pair

### Core Classes
- **CatScorer**: Main scoring engine with methods for each measure (mle(), pmi(), chi2(), etc.)
  - Constructor: `CatScorer(cooccs, smoothing_method='mle', smoothing_alpha=1.0)`
  - Smoothing methods: 'mle', 'laplace', 'lidstone' (parameterized smoothing)
- **New Methods**:
  - `pmi_smoothed()`: PMI with freqprob smoothing for numerical stability
  - `get_smoothed_probabilities()`: Returns all probability types with smoothing
- Each scoring method returns dict of `{(x,y): (xy_score, yx_score)}`

### Data Formats
- **Sequences**: TSV with aligned sequences (e.g., orthography/phonetics pairs)
- **Presence-Absence**: Binary matrices with entities x features
- **Co-occurrences**: Internal format with (x, y, count) tuples

### Module Organization
- `common.py`: I/O, data preprocessing, n-gram collection
- `scorer.py`: All association measures, matrix operations, score transformations
- `correlation.py`: Higher-level correlation analysis
- `__main__.py`: CLI argument parsing and workflow orchestration

## Testing Structure

**✅ Fully Modernized Test Suite**: Legacy test files have been completely migrated to a modern structure.

Tests are organized in `/tests` with:
- `unit/`: Individual function/method tests (29 test files)
  - `test_data_loading.py`: Data file reading and preprocessing
  - `test_scoring_measures.py`: Individual scoring methods  
  - `test_score_transformations.py`: Scaling, inversion, matrices
  - `test_legacy_compatibility.py`: Validation against original results
- `integration/`: End-to-end workflow tests (11 test files)
  - `test_end_to_end_workflows.py`: Complete analysis pipelines
- `fixtures/`: Shared test data and utilities
  - `data.py`: Test datasets and expected results
  - `assertions.py`: Domain-specific assertion functions
- `conftest.py`: Pytest configuration and fixtures

**Current Coverage**: 75 test functions with comprehensive parametrization
Test markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`

## Key Implementation Notes

- **FreqProb Integration**: Uses freqprob library for robust probability estimation
  - MLE method falls back to simple division for zero co-occurrences (avoids log(0))
  - Laplace/ELE smoothing handles sparse data by adding pseudo-counts
  - Smoothing particularly beneficial for rare category pairs
- All scorers handle missing/zero co-occurrences gracefully
- CLI supports multiple output formats (JSON, CSV, markdown tables)
- CLI smoothing options: `--smoothing {mle,laplace,lidstone}` and `--smoothing-alpha FLOAT`
- New scorer: `pmi_smoothed()` uses freqprob for better numerical stability
- Extensive type hints throughout codebase
- N-gram analysis supported via `collect_ngrams()`
- Score transformation utilities: scaling, inversion, matrix conversion
- Built-in example datasets in `/resources`

## Current Project Status (January 2025)

**Architecture**: Well-structured Python library at version 0.3.0 with comprehensive modernization:
- 15+ association measures (PMI, MLE, Chi-square, Cramér's V, Theil's U, etc.)
- Both CLI interface and Python API
- Recent integration of freqprob 0.3.1 for robust probability estimation and smoothing
- Modern test structure with unit/integration separation

**Recent Major Updates**:
- Comprehensive documentation enhancement with mathematical foundations
- FreqProb integration for numerical stability in probability calculations
- CLI functionality expansion with multiple output formats
- Modern test architecture replacing legacy test structure
- Scientific documentation with visualization examples

**Development Environment**:
- Requires "env_lingfil" pyenv environment
- Dependencies: scipy, numpy, pandas, matplotlib, seaborn, tabulate, freqprob
- Virtual environment setup via `make install` (creates `venv/` directory)

**Known Issues**:
- All major issues have been resolved ✅
- Test suite fully modernized and passing ✅
- Score direction validation fixed ✅
- Data loading edge cases resolved ✅
- Jupyter notebook execution issues fixed ✅

**Recent Achievements**:
1. ✅ Fixed asymmetric measure score direction issues
2. ✅ Resolved all data loading edge cases  
3. ✅ Validated all statistical measure implementations
4. ✅ **Completed test modernization migration**
5. ✅ **Implemented comprehensive Jupyter notebook documentation**
6. ✅ **Fixed all Jupyter notebook execution issues with committed outputs**

## Jupyter Notebook Guidelines

**IMPORTANT**: All Jupyter notebooks in this repository should always be executed with their cell outputs committed to the repository. This ensures that documentation examples are always up-to-date and visible without requiring execution.

### Notebook Structure

The `/docs` directory contains several key notebooks:

1. **`Simple_Examples.ipynb`** ✅ - Basic usage examples with synthetic data (fully working)
2. **`Demo.ipynb`** ✅ - Interactive demonstration with plotting and visualization (fully working)  
3. **`Academic_Analysis_Tutorial.ipynb`** ⚠️ - Comprehensive academic treatment with case studies (complex, needs fixes)
4. **`EXAMPLES_WITH_PLOTS.ipynb`** ⚠️ - Advanced examples with statistical plots (needs data format fixes)

### Executing Notebooks

**Always execute notebooks before committing:**

```bash
# Execute individual notebooks with outputs
jupyter nbconvert --to notebook --execute --inplace docs/Simple_Examples.ipynb
jupyter nbconvert --to notebook --execute --inplace docs/Demo.ipynb

# Execute all notebooks in docs/ directory
for nb in docs/*.ipynb; do
    echo "Executing $nb..."
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done

# Verify notebook outputs are present
ls -la docs/*.ipynb  # Check file sizes (executed notebooks are larger)
```

### Common Issues and Solutions

**Data Format Issues**: Some notebooks incorrectly use `collect_cooccs([data])` when `data` is already co-occurrences. Fix by passing data directly to `CatScorer(data)`.

**Example Fix**:
```python
# Wrong - data is already co-occurrences
test_cooccs = asymcat.collect_cooccs([test_data])
scorer = CatScorer(test_cooccs)

# Correct - pass co-occurrences directly
scorer = CatScorer(test_data)
```

**Matrix Plotting**: The `plot_scorer` function expects matrix dimensions to match index/column parameters:
```python
# xy matrix: rows=alpha_x, cols=alpha_y
plot_scorer(xy, alpha_x, alpha_y, "x->y")

# yx matrix: rows=alpha_y, cols=alpha_x  
plot_scorer(yx, alpha_y, alpha_x, "y->x")
```

### Notebook Maintenance Workflow

1. **Before making changes**: Ensure notebooks execute cleanly
2. **After code changes**: Re-execute all notebooks to update outputs
3. **Before committing**: Verify all cells have outputs and no errors
4. **For large notebooks**: Consider timeout limits for bootstrap/statistical computations

### Working Notebooks Status

- ✅ **Simple_Examples.ipynb**: 278KB, fully executed, demonstrates core functionality
- ✅ **Demo.ipynb**: 221KB, fully executed, includes plotting and visualization
- ✅ **Academic_Analysis_Tutorial.ipynb**: 33KB, executed, comprehensive academic treatment with statistical validation
- ✅ **EXAMPLES_WITH_PLOTS.ipynb**: 39KB, executed, advanced examples with smoothing and visualization

### Quick Verification

```bash
# Check if notebooks have outputs (file size > 50KB indicates execution)
find docs/ -name "*.ipynb" -size +50k -exec echo "✅ {}" \;
find docs/ -name "*.ipynb" -size -50k -exec echo "⚠️  {}" \;
```
