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
# Create virtual environment and install dev dependencies
make install

# Install with all optional features
pip install -e ".[all]"
```

### Testing
```bash
# Run full test suite
make test
# or
pytest

# Run with coverage report
make coverage
# or (if pytest-cov is installed)
pytest --cov=asymcat --cov-branch --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m slow          # Slow tests only

# Quick testing
make quick-test         # Run essential tests quickly
```

### Code Quality
```bash
# Run all formatting and linting
make format-check

# Individual tools
make black-check    # Code formatting check
make isort-check    # Import sorting check
make lint          # Flake8 linting
make mypy          # Type checking

# Auto-fix formatting issues
make format        # Runs black, isort, lint, mypy
make black         # Auto-format code
make isort         # Auto-sort imports

# Security checks
make security      # Run bandit and safety scans
```

### Build and Release
```bash
# Build package
make build

# Build documentation
make docs
# Clean docs
make docs-clean
```

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

**Recent Achievements**:
1. ✅ Fixed asymmetric measure score direction issues
2. ✅ Resolved all data loading edge cases  
3. ✅ Validated all statistical measure implementations
4. ✅ **Completed test modernization migration**
5. ✅ Enhanced CLI functionality with new features
6. ✅ Updated workflow commands for better development experience
