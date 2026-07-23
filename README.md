# ASymCat: Asymmetric Categorical Association Analysis

[![PyPI version](https://badge.fury.io/py/asymcat.svg)](https://badge.fury.io/py/asymcat)
[![Python versions](https://img.shields.io/pypi/pyversions/asymcat.svg)](https://pypi.org/project/asymcat/)
[![Code Quality](https://github.com/tresoldi/asymcat/workflows/Code%20Quality/badge.svg)](https://github.com/tresoldi/asymcat/actions)
[![codecov](https://codecov.io/gh/tresoldi/asymcat/branch/master/graph/badge.svg)](https://codecov.io/gh/tresoldi/asymcat)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ASymCat is a comprehensive Python library for analyzing **asymmetric associations** between categorical variables. Unlike traditional symmetric measures that treat relationships as bidirectional, ASymCat provides directional measures that reveal which variable predicts which, making it invaluable for understanding causal relationships, dependencies, and information flow in categorical data.

## 🚀 Key Features

- **17+ Association Measures**: From basic MLE to advanced information-theoretic measures
- **Directional Analysis**: X→Y vs Y→X asymmetric relationship quantification
- **Robust Smoothing**: FreqProb integration for numerical stability
- **Multiple Data Formats**: Sequences, presence-absence matrices, n-grams
- **Scalable Architecture**: Optimized for large datasets with efficient algorithms
- **Comprehensive Testing**: 100+ tests with 80%+ coverage ensuring reliability and accuracy

## 🎯 Why Asymmetric Measures Matter

Traditional measures like Pearson's χ² or Cramér's V treat associations as symmetric: the relationship between X and Y is the same as between Y and X. However, many real-world relationships are inherently directional:

- **Linguistics**: Phoneme transitions may be predictable in one direction but not the other
- **Ecology**: Species presence may predict other species asymmetrically  
- **Market Research**: Product purchases may show directional dependencies
- **Medical Analysis**: Symptoms may predict conditions more reliably than vice versa

ASymCat quantifies these directional relationships, revealing hidden patterns that symmetric measures miss.

## 📊 Quick Example

```python
import asymcat

# Load your categorical data
data = asymcat.read_sequences("data.tsv")  # or read_pa_matrix() for binary data

# Collect co-occurrences  
cooccs = asymcat.collect_cooccs(data)

# Create scorer and analyze
scorer = asymcat.CatScorer(cooccs)

# Get asymmetric measures
mle_scores = scorer.mle()           # Maximum likelihood estimation
pmi_scores = scorer.pmi()           # Pointwise mutual information  
chi2_scores = scorer.chi2()         # Chi-square with smoothing
fisher_scores = scorer.fisher()     # Fisher exact test

# Each returns {(x, y): (x→y_score, y→x_score)}
print(f"A→B: {mle_scores[('A', 'B')][0]:.3f}")
print(f"B→A: {mle_scores[('A', 'B')][1]:.3f}")
```

## 🛠️ Installation

### From PyPI (Recommended)
```bash
pip install asymcat
```

### From Source
```bash
git clone https://github.com/tresoldi/asymcat.git
cd asymcat
pip install -e ".[dev]"  # Install with all optional dependencies
```

### Dependencies
- **Core**: numpy, pandas, scipy, matplotlib, seaborn, tabulate, freqprob
- **Development**: pytest, ruff, mypy, jupyter
- **Optional**: plotly, bokeh, altair (for enhanced visualization)

## 📚 Documentation & Resources

ASymCat provides comprehensive documentation organized for different needs:

### Core Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| **[User Guide](docs/USER_GUIDE.md)** | Conceptual foundations, theory, best practices | Everyone - start here |
| **[API Reference](docs/API_REFERENCE.md)** | Complete technical API documentation | Developers |
| **[LLM Documentation](docs/LLM_DOCUMENTATION.md)** | Quick integration and code patterns | AI agents, rapid development |

### Progressive Interactive Tutorials

Learn ASymCat through hands-on Nhandu tutorials with executable code and visualizations:

#### 📘 Tutorial 1: Basics
**Foundation** - Get started with asymmetric analysis
📄 [Python source](docs/tutorial_1_basics.py) | 🌐 [View HTML](https://htmlpreview.github.io/?https://github.com/tresoldi/asymcat/blob/master/docs/tutorial_1_basics.html)
- What are asymmetric associations and why they matter
- Basic workflow: load → collect → score
- Simple measures (MLE, PMI, Jaccard)
- Working with sequences and presence-absence data

#### 📗 Tutorial 2: Advanced Measures
**Depth** - Master all 17+ association measures
📄 [Python source](docs/tutorial_2_advanced_measures.py) | 🌐 [View HTML](https://htmlpreview.github.io/?https://github.com/tresoldi/asymcat/blob/master/docs/tutorial_2_advanced_measures.html)
- Information-theoretic measures (PMI, NPMI, Theil's U)
- Statistical measures (Chi-square, Cramér's V, Fisher)
- Smoothing methods and their effects
- Measure selection decision tree

#### 📙 Tutorial 3: Visualization
**Communication** - Create publication-quality figures
📄 [Python source](docs/tutorial_3_visualization.py) | 🌐 [View HTML](https://htmlpreview.github.io/?https://github.com/tresoldi/asymcat/blob/master/docs/tutorial_3_visualization.html)
- Heatmap visualizations of association matrices
- Score distribution and asymmetry plots
- Matrix transformations (scaling, inversion)
- Multi-measure comparison panels

#### 📕 Tutorial 4: Real-World Applications
**Application** - Complete analysis workflows
📄 [Python source](docs/tutorial_4_real_world.py) | 🌐 [View HTML](https://htmlpreview.github.io/?https://github.com/tresoldi/asymcat/blob/master/docs/tutorial_4_real_world.html)
- **Linguistics**: Grapheme-phoneme correspondence analysis
- **Ecology**: Galápagos finch species co-occurrence patterns
- **Machine Learning**: Feature selection with asymmetric measures
- Interpretation best practices and reporting strategies

> **💡 All tutorials are fully executed with committed outputs** - view the HTML files online via the links above, or run the Python source files locally to explore and modify. Generate fresh documentation with `make docs`.

### Additional Resources
- **[Documentation Index](docs/README.md)**: Complete navigation guide
- **[CHANGELOG](CHANGELOG.md)**: Version history and migration guides

## 🎮 Usage

### Python API

#### Basic Analysis
```python
import asymcat

# Load data (TSV format: tab-separated sequences)
data = asymcat.read_sequences("linguistic_data.tsv")
cooccs = asymcat.collect_cooccs(data)

# Create scorer with smoothing
scorer = asymcat.CatScorer(cooccs, smoothing_method="laplace", smoothing_alpha=1.0)

# Compute multiple measures
results = {
    'mle': scorer.mle(),
    'pmi': scorer.pmi(),
    'chi2': scorer.chi2(),
    'fisher': scorer.fisher(),
    'theil_u': scorer.theil_u(),
}

# Analyze directional relationships
for measure, scores in results.items():
    for (x, y), (xy_score, yx_score) in scores.items():
        if xy_score > yx_score:
            print(f"{measure}: {x}→{y} stronger than {y}→{x}")
```

#### Advanced Features
```python
# N-gram analysis
ngram_cooccs = asymcat.collect_cooccs(data, order=2, pad="#")
ngram_scorer = asymcat.CatScorer(ngram_cooccs)

# Matrix generation for visualization
xy_matrix, yx_matrix, x_labels, y_labels = asymcat.scorer.scorer2matrices(
    ngram_scorer.pmi()
)

# Score transformations
scaled_scores = asymcat.scorer.scale_scorer(scores, method="minmax")
inverted_scores = asymcat.scorer.invert_scorer(scaled_scores)
```


## 📈 Association Measures

ASymCat implements 17+ association measures organized by type:

### Probabilistic Measures
- **MLE**: Maximum Likelihood Estimation - P(X|Y) and P(Y|X)
- **Jaccard Index**: Set overlap with asymmetric interpretation

### Information-Theoretic Measures  
- **PMI**: Pointwise Mutual Information (log P(X,Y)/P(X)P(Y))
- **PMI Smoothed**: Numerically stable PMI with FreqProb smoothing
- **NPMI**: Normalized PMI [-1, 1] range
- **Mutual Information**: Average information shared
- **Conditional Entropy**: Information remaining after observing condition

### Statistical Measures
- **Chi-Square**: Pearson's χ² with optional smoothing
- **Cramér's V**: Normalized chi-square association
- **Fisher Exact**: Exact odds ratios for small samples
- **Log-Likelihood Ratio**: G² statistic

### Specialized Measures
- **Theil's U**: Uncertainty coefficient (entropy-based)
- **Tresoldi**: Custom measure designed for sequence alignment
- **Goodman-Kruskal λ**: Proportional reduction in error

### Statistical Significance (p-values)
Significance counterparts to the statistical tests, returning the same
`{(x, y): (p, p)}` shape (a small value indicates a significant association):
- **`chi2_pvalue()`**: p-value of the chi-square test of independence
- **`fisher_pvalue()`**: p-value of Fisher's exact test
- **`log_likelihood_ratio_pvalue()`**: p-value of the G² test

```python
scorer = asymcat.CatScorer(cooccs)
chi2_scores = scorer.chi2()            # test statistics
chi2_pvals  = scorer.chi2_pvalue()     # matching p-values
```

## 🔬 Scientific Applications

### Linguistics & Language Evolution
```python
# Analyze phoneme transitions
phoneme_data = asymcat.read_sequences("phoneme_alignments.tsv")
cooccs = asymcat.collect_cooccs(phoneme_data)
scorer = asymcat.CatScorer(cooccs)

# Asymmetric sound change analysis
tresoldi_scores = scorer.tresoldi()  # Optimized for linguistic alignment
```

### Ecology & Species Analysis
```python
# Species co-occurrence from presence-absence data
species_data = asymcat.read_pa_matrix("galapagos_species.tsv")
scorer = asymcat.CatScorer(species_data)

# Ecological associations
fisher_scores = scorer.fisher()  # Exact tests for species relationships
```

### Market Research & Business Analytics
```python
# Product purchase associations
purchase_data = asymcat.read_sequences("customer_transactions.tsv")
cooccs = asymcat.collect_cooccs(purchase_data)
scorer = asymcat.CatScorer(cooccs, smoothing_method="lidstone", smoothing_alpha=0.5)

# Market basket analysis
chi2_scores = scorer.chi2()  # Statistical significance testing
```

## 🎯 Data Formats

### Sequence Data (TSV)
```
# linguistic_data.tsv
sound_from	sound_to
p a t a	B A T A
k a t a	G A T A
```

### Presence-Absence Matrix (TSV)
```
# species_data.tsv
site	species_A	species_B	species_C
island_1	1	0	1
island_2	1	1	0
```

### N-gram Support
```python
# Automatic n-gram extraction
bigrams = asymcat.collect_cooccs(data, order=2, pad="#")
trigrams = asymcat.collect_cooccs(data, order=3, pad="#")
```

## 🔧 Development

### Setup Development Environment
```bash
git clone https://github.com/tresoldi/asymcat.git
cd asymcat

# Install development dependencies
make install-dev  # Creates venv and installs with [dev] extras
```

### Common Development Commands
```bash
# Code quality checks (runs all: format-check + lint + typecheck)
make quality

# Auto-format code
make format

# Auto-fix linting issues and format
make ruff-fix

# Type checking
make mypy

# Run tests with coverage report
make test-cov

# Run tests in parallel (faster)
make test-fast

# Generate HTML documentation from tutorials
make docs

# Clean generated documentation
make docs-clean
```

### Testing
```bash
# Full test suite (100+ tests)
pytest

# Specific categories
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
pytest -m slow              # Performance tests
pytest -m "not slow"        # Skip slow tests

# Performance benchmarks (opt-in; skipped by default)
pytest tests/performance --run-slow --no-cov --benchmark-only

# Coverage with threshold enforcement (80%)
make test-cov
```

### Release Process

**Version Bumping:**
```bash
# Bump patch version (0.4.0 → 0.4.1)
make bump-version TYPE=patch

# Bump minor version (0.4.0 → 0.5.0)
make bump-version TYPE=minor

# Bump major version (0.4.0 → 1.0.0)
make bump-version TYPE=major
```

The `bump-version` target will:
1. Update version in `asymcat/__init__.py` and `pyproject.toml`
2. Prompt you to update CHANGELOG.md
3. Create a git commit with the version bump
4. Create a git tag (e.g., `v0.4.1`)
5. Display next steps for pushing changes

**Full Release Build:**
```bash
# Clean → Quality checks → Tests → Build distribution
make build-release
```

### Code Quality Standards

All code must pass:
- **Ruff formatting**: `ruff format --check asymcat/ tests/`
- **Ruff linting**: `ruff check asymcat/ tests/`
- **MyPy type checking**: `mypy asymcat/ tests/`
- **Test coverage**: Minimum 80% coverage

Run all checks before committing:
```bash
make quality && make test-cov
```

## 📚 Documentation

- **[Documentation Index](docs/README.md)**: Complete navigation and quick reference
- **[User Guide](docs/USER_GUIDE.md)**: Conceptual foundations and best practices
- **[API Reference](docs/API_REFERENCE.md)**: Complete technical API documentation
- **[Interactive Tutorials](docs/)**: Four progressive Nhandu tutorials with HTML reports
- **[CHANGELOG](CHANGELOG.md)**: Version history and migration guides

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Setting up the development environment
- Code style guidelines and testing requirements
- Submitting bug reports and feature requests
- Contributing new association measures or improvements

### Quick Start for Contributors
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run quality checks: `make quality && make test-cov`
5. Submit a pull request

## 📖 Citation

If you use ASymCat in your research, please cite:

```bibtex
@software{tresoldi_asymcat_2024,
  title = {ASymCat: Asymmetric Categorical Association Analysis},
  author = {Tresoldi, Tiago},
  year = {2024},
  url = {https://github.com/tresoldi/asymcat},
  version = {0.4.0}
}
```

## 🏆 Acknowledgments

- **FreqProb Library**: Robust probability estimation and smoothing
- **SciPy Community**: Statistical foundations
- **Linguistic Community**: Inspiration from historical linguistics applications

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🚀 What's New in v0.4.0

- ✅ **Simplified Dependencies**: Consolidated to `[viz]` and `[dev]` groups - easier installation
- ✅ **Modern Tooling**: Unified linting/formatting with Ruff, replacing black/isort/flake8
- ✅ **Enhanced CI/CD**: Simplified quality workflow with faster feedback
- ✅ **Coverage Enforcement**: 78% minimum threshold (goal: 80%)
- ✅ **Keep a Changelog**: Semantic versioning with full version history
- ✅ **Developer-Friendly Makefile**: Self-documenting help, automated version bumping
- ✅ **Library-Only Focus**: Removed CLI tool for better coverage and maintainability

**Migration from v0.3.1:**
- Use `pip install asymcat[dev]` instead of multiple dependency groups
- Use library API directly instead of CLI tool (see examples above)
- See [CHANGELOG.md](CHANGELOG.md) for detailed migration guide

## 🔮 Roadmap

- **Statistical Significance**: P-values for the statistical tests (chi-square,
  Fisher, G²) are available; permutation-based p-values for the
  information-theoretic measures are planned
- **Confidence Intervals**: Uncertainty quantification
- **GPU Acceleration**: CUDA support for massive datasets
- **Interactive Dashboards**: Web-based exploration tools
- **Extended Measures**: Additional domain-specific association metrics
- **Nhandu Documentation**: Migration to modern documentation system

---

**[⭐ Star us on GitHub](https://github.com/tresoldi/asymcat)** if you find ASymCat useful!