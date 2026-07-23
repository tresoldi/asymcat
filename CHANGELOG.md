# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `correlation.theil_u()` now computes Theil's U (uncertainty coefficient)
  instead of mistakenly returning the raw conditional entropy. The previous
  implementation delegated to `scorer.conditional_entropy`, silently returning
  an unbounded value under the `theil_u` name.

### Added
- Regression tests for the `asymcat.correlation` module (`conditional_entropy`,
  `theil_u`, and `cramers_v` wrappers), bringing the module to full coverage.

### Changed
- Corrected the `asymcat.correlation` module docstring and added per-function
  docstrings clarifying which measures are symmetric vs. directional.

### Removed
- Stale references to non-existent `AUTHORS.md`, `AGENTS.md`, and `DEVELOPER.md`
  files from the sdist packaging config (`pyproject.toml` and `MANIFEST.in`).

## [0.4.0] - 2025-01-XX

### Added
- Coverage threshold enforcement at 78% minimum (with goal of 80%)
- Consolidated dependency structure with simplified `[dev]` group
- Automated version bumping capability via Makefile
- Enhanced Makefile with modern development targets:
  - `help` - Self-documenting default target
  - `quality` - Run all quality checks (ruff + mypy)
  - `test-fast` - Parallel test execution
  - `install-dev` - Install with dev dependencies
- Simplified CI/CD with `quality.yml` workflow (ruff + mypy + tests)
- Test markers for better organization (slow, integration, unit, performance, large_data)
- Coverage reporting with term-missing, HTML, and XML outputs
- Comprehensive pytest configuration in pyproject.toml

### Changed
- **BREAKING**: Simplified optional dependencies - use `pip install asymcat[dev]` instead of multiple groups
- **BREAKING**: Removed CLI tool (`asymcat` command) - library-only package
- Migrated CHANGELOG to Keep a Changelog format with semantic versioning
- Updated freqprob dependency to >=0.4.0 with Lidstone API compatibility
- Consolidated pytest configuration from pytest.ini to pyproject.toml
- Updated CI/CD to simplified quality workflow (faster, clearer)
- Bumped version to 0.4.0 across __init__.py and pyproject.toml
- Enhanced .gitignore with modern Python patterns
- Updated GitHub Actions workflows to use consolidated dependencies

### Removed
- **BREAKING**: CLI entry point (`asymcat.__main__:main`) - reduces complexity, improves coverage
- **BREAKING**: Granular optional dependency groups (test, lint, typecheck, security, dev-tools, docs, jupyter, performance)
- Complex CI/CD workflows (build.yml, security.yml) - archived for reference
- Obsolete dependencies: black, isort, flake8, pre-commit, bump2version
- pytest.ini file (consolidated into pyproject.toml)

### Fixed
- Lidstone API calls for freqprob 0.4.0 compatibility (use `gamma=` parameter)
- Type checking errors with proper type: ignore annotations

### Migration Guide

#### For Users
No changes required. Core functionality unchanged.

#### For Developers

**Dependency Installation:**
```bash
# Before (v0.3.1)
pip install -e ".[test,lint,typecheck,security,dev-tools]"

# After (v0.4.0)
pip install -e ".[dev]"  # All dev tools included
```

**CI/CD Updates:**
```yaml
# Before
- pip install ".[test]"
- pip install ".[security]"

# After
- pip install -e ".[dev]"
```

**CLI Usage:**
The CLI has been removed. Use the library API directly:
```python
# Instead of: asymcat input.tsv --scorers mle pmi
import asymcat
data = asymcat.read_sequences("input.tsv")
cooccs = asymcat.collect_cooccs(data)
scorer = asymcat.scorer.CatScorer(cooccs)
mle_scores = scorer.mle()
pmi_scores = scorer.pmi()
```

## [0.3.1] - 2024-10-04

### Added
- Enhanced documentation with 4 interactive Jupyter notebooks (1.4MB+ examples)
  - `Simple_Examples.ipynb` (278KB) - Perfect starting point
  - `Demo.ipynb` (221KB) - Visualization showcase
  - `Academic_Analysis_Tutorial.ipynb` (44KB) - Research-grade examples
  - `EXAMPLES_WITH_PLOTS.ipynb` (903KB) - Publication-ready analysis
- Publication-ready examples with academic-grade analysis workflows
- Real-world applications: linguistics, ecology, machine learning case studies
- Advanced visualizations with heatmaps and statistical distributions
- All notebooks pre-executed with committed outputs for immediate viewing
- Bootstrap confidence intervals and permutation testing examples
- GitHub Actions workflow for automated notebook execution and validation

### Changed
- Migrated to Ruff for unified linting and formatting (replaced black, isort, flake8)
- Fixed GitHub Actions CI/CD workflows and notebook execution
- Systematically fixed majority of mypy type checking errors
- Migrated build system from setuptools to Hatch
- Enhanced freqprob integration with version compatibility
- Fixed linting issues to ensure all CI workflows pass locally
- Updated Python requirement to 3.10+ across all configurations

### Fixed
- Notebook execution in GitHub Actions with proper timeout and error handling
- Version synchronization between pyproject.toml and __init__.py
- Type checking errors throughout codebase
- Remaining GitHub Actions issues with branch references and versions

## [0.3.0] - 2023-XX-XX

### Changed
- **BREAKING**: Renamed package from `catcoocc` to `asymcat`
  - Better reflects the library's focus on asymmetric categorical association analysis
  - More intuitive and descriptive name for users

### Migration Guide

**Upgrading from catcoocc to asymcat:**

```python
# Before (catcoocc)
import catcoocc

# After (asymcat)
import asymcat
```

All APIs remain the same, only the package name changed.

## [0.2.3] - 2020-06-29

### Added
- Initial PyPI release as `catcoocc`
- Core package infrastructure

## [0.2.2] - 2020-XX-XX

### Added
- Function for inverting a scorer
- Scorer inversion utilities

## [0.2.1] - 2020-XX-XX

### Added
- Basic functions for double series correlation
- Correlation analysis capabilities

## [0.2.0] - 2019-XX-XX

### Added
- Initial public release
- Core asymmetric association measures:
  - Maximum Likelihood Estimation (MLE)
  - Pointwise Mutual Information (PMI)
  - Chi-square test
  - Fisher's exact test
  - Cramér's V
  - Goodman and Kruskal's lambda
  - Jaccard index
  - Mutual information
  - Conditional entropy
  - Theil's U
  - Log-likelihood ratio
- Sequence and presence-absence matrix readers
- N-gram support with configurable window sizes
- Co-occurrence collection from aligned sequences
- Basic visualization utilities
- Scorer transformation and scaling functions

---

## Version History

- **0.4.0** - Modernization: simplified deps, enhanced tooling, Keep a Changelog, removed CLI
- **0.3.1** - Documentation: Jupyter notebooks, Ruff migration, Hatch build, freqprob updates
- **0.3.0** - Renamed from catcoocc to asymcat
- **0.2.3** - First PyPI release (as catcoocc)
- **0.2.2** - Scorer inversion
- **0.2.1** - Double series correlation
- **0.2.0** - Initial release

[Unreleased]: https://github.com/tresoldi/asymcat/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/tresoldi/asymcat/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/tresoldi/asymcat/releases/tag/v0.3.1
[0.3.0]: https://github.com/tresoldi/asymcat/releases/tag/v0.3.0
[0.2.3]: https://github.com/tresoldi/asymcat/releases/tag/catcoocc0.2.3
[0.2.2]: https://github.com/tresoldi/asymcat/releases/tag/v0.2.2
[0.2.1]: https://github.com/tresoldi/asymcat/releases/tag/v0.2.1
[0.2.0]: https://github.com/tresoldi/asymcat/releases/tag/v0.2.0
