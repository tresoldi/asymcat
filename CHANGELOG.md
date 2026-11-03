# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Zenodo DOIs: 10.5281/zenodo.23917212 for version 0.8.0, and 10.5281/zenodo.23917211 for all versions. They are in
  `CITATION.cff`, the README (with a DOI badge), the home page and the How to
  cite page.

### Changed

- **README rewritten as the project's front door**, with the same structure as
  FreqProb's: badges, a short description with the DOI, install, a runnable
  quick start, the selection table, highlights, links into the documentation,
  related projects, citation and licence. Every example now uses inline data
  and shows its real output, and a new test (`tests/unit/test_readme.py`)
  runs them so they cannot drift from the code. All links are absolute, so they
  also work on PyPI. The badges are reduced to version, Python versions, CI,
  docs, DOI and licence.
- **Landing page refreshed.** The masthead shows the version and DOI and a row
  of links into the documentation; four entry cards lead to getting started, the
  measures, the how-to guides and the API reference. The quick start is the
  README's, and a new test (`tests/unit/test_landing_example.py`) runs it. The
  illustrative figure is replaced by a figure computed with ASymCat on the
  bundled UCI Mushroom sample (`P(class | odour)` against `P(odour | class)`).
  The page also gains Related projects and links to How to cite and the
  migration guide.

## [0.8.0] - 2026-11-03

Corrects the direction of `mle()` and `tresoldi()` and prepares ASymCat for
citation. **Results of `mle` and `tresoldi` change**; see `MIGRATION.md`.

### Fixed

- **`mle()` returned its tuple in the reverse of the documented order.** For a
  pair `(x, y)` it gave `(P(x|y), P(y|x))`, while the README, the User Guide,
  and `theil_u`, `cond_entropy` and `goodman_kruskal_lambda` all use the
  `(X→Y, Y→X)` order, i.e. `(P(y|x), P(x|y))`. It now follows that order.
  `tresoldi()`, which takes its exponent from `mle()`, swaps with it. **This
  changes results** for both measures, and for `permutation_pvalue`,
  `bootstrap_ci` and `scorer2matrices` applied to them. The behaviour dated back
  to the original implementation; the legacy expected values in the tests are
  updated, and a new test pins the direction of every measure. See
  `MIGRATION.md`.

### Added

- **References** page on the documentation site, with the published source of
  every measure and a `references.bib` file.
- **How to cite** page with APA and BibTeX (`@software` and `@misc`) entries.
- `resources/README.md` now gives a full citation, source URL and licence for
  every sample dataset, and `resources/LICENSE-cmudict.txt` carries the notice
  CMUdict's licence requires.

### Changed

- **The Tresoldi Measure page is rewritten.** The previous page stated several
  properties the formula does not have: it said high scores require a high
  conditional probability (a conditional probability of 1 gives exactly ±1),
  reversed the monotonicity in the conditional probability, said the PMI term
  needs smoothing (it uses raw frequencies), and listed significance testing as
  an open question (`permutation_pvalue` and `bootstrap_ci` already accept
  `"tresoldi"`). The new page states the definition, the limiting cases and the
  properties with values computed by ASymCat. The User Guide, README and home
  page descriptions are corrected to match.
- `CITATION.cff` adds an ORCID iD and the PyPI artifact; README and home-page
  BibTeX entries now include the year and version and point to the
  documentation site. PyPI's Homepage link now points to the site too.

### Removed

- `docs/archive/` (the pre-0.6 Sphinx site, tutorials and LLM documentation).
  It was excluded from the site and partly outdated; it remains in the git
  history and in the `v0.7.0` tag.

## [0.7.0] - 2026-11-03

Adopts freqprob's bias-corrected entropy estimators and trims the dependencies
to what the package imports. Existing results are unchanged; see `MIGRATION.md`
for the installation changes.

### Added
- `entropy_estimator` option on `CatScorer` (`"plugin"`, the default, plus
  `"chao_shen"` and `"nsb"`) for `theil_u`, `cond_entropy`,
  `mutual_information` and `normalized_mutual_information`, using freqprob's
  bias-corrected estimators for sparse data. `permutation_pvalue` and
  `bootstrap_ci` score their replicates with the same estimator.
- `estimator` argument on the scalar helpers (`conditional_entropy`,
  `compute_theil_u`, `compute_mutual_information`,
  `compute_normalized_mutual_information`) and on the
  `correlation.conditional_entropy` and `correlation.theil_u` wrappers.
- User Guide section on entropy estimation.

### Changed
- Runtime dependencies are now `freqprob>=0.8.0`, `numpy` and `scipy`.
  `matplotlib`, `pandas`, `seaborn` and `tabulate` are no longer installed: the
  package never imported them.
- The `viz` extra installs `matplotlib`, `seaborn` and `pandas`, which the User
  Guide's plotting examples use, instead of the unused `plotly`, `bokeh` and
  `altair`.
- Development tooling: `ruff` 0.17.3 and `mypy` 2.4.1. The unused
  `scikit-learn`, `types-tabulate` and `types-setuptools` were dropped from the
  extras.

### Fixed
- The User Guide and the Tresoldi Measure page pointed at archived files
  (tutorial scripts, `API_REFERENCE.md`, `LLM_DOCUMENTATION.md`) and at a
  stale source path and line range; they now link to the live API Reference.
- `CatScorer`'s docstring listed a nonexistent `'ele'` smoothing method and
  misdescribed `smoothing_alpha`.
- A User Guide plotting example used seaborn's deprecated `shade=` argument.
- The compare links for 0.5.0 and 0.6.0 referenced a `v0.5.0` tag that was
  never created.

## [0.6.0] - 2026-07-30

This release restructures the project to match the layout and tooling of the
sister library [freqprob](https://github.com/tresoldi/freqprob). It is a
packaging-, documentation-, and tooling-level change: the public `asymcat` API
is unchanged, so existing code keeps working without edits (see `MIGRATION.md`).

### Changed
- Moved the importable package to a `src/` layout (`asymcat/` → `src/asymcat/`).
  The public API is re-exported unchanged from `asymcat/__init__.py`.
- Replaced the Sphinx documentation with a MkDocs-Material site: an
  autogenerated API reference via `mkdocstrings`, a custom landing page, a User
  Guide, and MathJax-rendered mathematical notation. The previous Sphinx site
  and rendered tutorials are retained under `docs/archive/`.
- Modernized packaging in `pyproject.toml`: SPDX `license = "MIT"`, dynamic
  version, `viz`/`test`/`docs`/`dev`/`all` extras, and pinned `ruff`/`mypy`.
- Pinned the `freqprob` dependency to `>=0.6.2`.

### Added
- `ARCHITECTURE.md` and `MIGRATION.md`.
- `.pre-commit-config.yaml` mirroring the CI quality checks, and
  `.github/dependabot.yml` for weekly dependency updates.
- freqprob-aligned CI workflows: a Code Quality workflow (ruff, mypy, bandit)
  with an OS matrix (Ubuntu/macOS/Windows × Python 3.10–3.12), a Docs
  build-and-deploy workflow (GitHub Pages), and an OIDC-based PyPI release
  workflow.

### Removed
- The Codecov integration (badge, `codecov.yml`, upload step); coverage is now
  enforced solely through the `--cov-fail-under` gate.
- Vestigial `MANIFEST.in` and `requirements.txt`, superseded by `pyproject.toml`
  and its optional-dependency extras.

### Fixed
- `correlation.cramers_v` passed a NumPy array into a helper annotated for a
  list of lists; it now passes a plain list, resolving a type-check failure
  under newer NumPy stubs.

## [0.5.0] - 2026-07-23

### Fixed
- `correlation.theil_u()` now computes Theil's U (uncertainty coefficient)
  instead of mistakenly returning the raw conditional entropy. The previous
  implementation delegated to `scorer.conditional_entropy`, silently returning
  an unbounded value under the `theil_u` name.
- `read_sequences()` and `read_pa_matrix()` no longer mask their own validation
  errors. A broad `except Exception` was re-wrapping the specific `ValueError`s
  (e.g. "Column not found", "Missing required 'ID' column", "Invalid
  presence-absence value") as a generic `OSError`, contradicting the documented
  behavior; these now propagate as `ValueError` as intended.
- Corrected the stale `version = {0.3.0}` in the README citation to the current
  release.

### Added
- Statistical-significance scorers `CatScorer.chi2_pvalue()`,
  `fisher_pvalue()` and `log_likelihood_ratio_pvalue()`, returning the p-value
  of the corresponding test for each pair in the usual `{(x, y): (p, p)}` shape
  (chi-square and Fisher p-values were already computed by SciPy and discarded;
  the G² p-value uses its chi-square distribution). Also exposes the
  `compute_log_likelihood_ratio_pvalue()` helper.
- `CatScorer.permutation_pvalue(measure, ...)`, a generic permutation test that
  estimates significance for any association measure (including the
  information-theoretic ones with no closed-form null) by shuffling the `x`↔`y`
  pairing while preserving both marginals. Supports `greater` / `less` /
  `two-sided` tails, a reproducible `seed`, and the `(count + 1) / (n + 1)`
  correction. Now practical thanks to the vectorized scorers (~1000
  permutations of `theil_u` run in about a second).
- `CatScorer.bootstrap_ci(measure, ...)`, bootstrap confidence intervals for any
  association measure. It resamples the co-occurrences with replacement and
  returns the percentile interval per pair and direction as
  `{(x, y): ((xy_low, xy_high), (yx_low, yx_high))}`, with configurable
  `confidence_level` and a reproducible `seed`. Shares its resampling loop with
  `permutation_pvalue` via an internal helper.
- Regression tests for the `asymcat.correlation` module (`conditional_entropy`,
  `theil_u`, and `cramers_v` wrappers), bringing the module to full coverage.
- Validation/error-path tests for `asymcat.common` data-loading helpers,
  raising overall coverage above 80%.
- Regression tests pinning the vectorized `theil_u`, `cond_entropy`,
  `mutual_information`, `normalized_mutual_information` and
  `goodman_kruskal_lambda` scorers to their original per-pair algorithms across
  several datasets and degenerate cases.
- Tests for the p-value scorers (parametric and permutation-based) and the
  bootstrap confidence intervals, covering correctness against SciPy, value
  ranges, tails, reproducibility, interval coverage/ordering, the expected
  narrowing with more data, and the response to strong vs. independent
  association.
- Performance benchmark suite (`tests/performance/`) built on `pytest-benchmark`
  measuring the main scoring measures. Benchmarks are opt-in (skipped by
  default) and run with `pytest tests/performance --run-slow --no-cov
  --benchmark-only`.

### Changed
- Vectorized the per-pair scorers `CatScorer.theil_u()`, `cond_entropy()`,
  `mutual_information()`, `normalized_mutual_information()` and
  `goodman_kruskal_lambda()`. Each previously re-scanned the full co-occurrence
  list for every one of the `|X| * |Y|` symbol pairs (`O(|X|*|Y|*n)`); they now
  derive all pairs at once from the global joint-count matrix (`O(|X|*|Y|)`) --
  the entropy-based measures via a shared helper, lambda via an analogous
  mode-based one. Results match the previous implementations to floating-point
  precision (Goodman-Kruskal lambda matches exactly, being integer-count based).
  Measured speed-ups on the benchmark dataset are ~650x for `theil_u` and ~330x
  for `mutual_information`; every measure is now sub-millisecond, with the gap
  widening as the dataset grows.
- Corrected the `asymcat.correlation` module docstring and added per-function
  docstrings clarifying which measures are symmetric vs. directional.
- Raised the enforced test-coverage threshold from 78% to 80% in
  `pyproject.toml` (applies to `pytest` and `make test-cov`).

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

[Unreleased]: https://github.com/tresoldi/asymcat/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/tresoldi/asymcat/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/tresoldi/asymcat/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/tresoldi/asymcat/compare/1867eb9...v0.6.0
[0.5.0]: https://github.com/tresoldi/asymcat/compare/v0.4.0...1867eb9
[0.4.0]: https://github.com/tresoldi/asymcat/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/tresoldi/asymcat/releases/tag/v0.3.1
[0.3.0]: https://github.com/tresoldi/asymcat/releases/tag/v0.3.0
[0.2.3]: https://github.com/tresoldi/asymcat/releases/tag/catcoocc0.2.3
[0.2.2]: https://github.com/tresoldi/asymcat/releases/tag/v0.2.2
[0.2.1]: https://github.com/tresoldi/asymcat/releases/tag/v0.2.1
[0.2.0]: https://github.com/tresoldi/asymcat/releases/tag/v0.2.0
