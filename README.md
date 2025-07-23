# ASymCat: Asymmetric Categorical Association Analysis

[![PyPI version](https://badge.fury.io/py/asymcat.svg)](https://badge.fury.io/py/asymcat)
[![Python versions](https://img.shields.io/pypi/pyversions/asymcat.svg)](https://pypi.org/project/asymcat/)
[![Build Status](https://github.com/tresoldi/asymcat/workflows/build/badge.svg)](https://github.com/tresoldi/asymcat/actions)
[![codecov](https://codecov.io/gh/tresoldi/asymcat/branch/master/graph/badge.svg)](https://codecov.io/gh/tresoldi/asymcat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ASymCat is a comprehensive Python library for analyzing **asymmetric associations** between categorical variables. Unlike traditional symmetric measures that treat relationships as bidirectional, ASymCat provides directional measures that reveal which variable predicts which, making it invaluable for understanding causal relationships, dependencies, and information flow in categorical data.

## 🚀 Key Features

- **17+ Association Measures**: From basic MLE to advanced information-theoretic measures
- **Directional Analysis**: X→Y vs Y→X asymmetric relationship quantification  
- **Robust Smoothing**: FreqProb integration for numerical stability
- **Multiple Data Formats**: Sequences, presence-absence matrices, n-grams
- **CLI Interface**: Production-ready command-line tool with rich output formats
- **Scalable Architecture**: Optimized for large datasets with efficient algorithms
- **Comprehensive Testing**: 75+ tests ensuring reliability and accuracy

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
pip install -e ".[all]"  # Install with all optional dependencies
```

### Dependencies
- **Core**: numpy, pandas, scipy, matplotlib, seaborn, tabulate, freqprob
- **Development**: pytest, black, isort, flake8, mypy
- **Documentation**: sphinx, sphinx-rtd-theme, myst-parser
- **Optional**: jupyter, plotly, bokeh, altair (for enhanced visualization)

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

### Command Line Interface

The CLI provides production-ready access to all functionality:

#### Basic Usage
```bash
# Analyze with multiple measures
asymcat data.tsv --scorers mle pmi chi2 --output results.json

# Presence-absence matrix analysis  
asymcat species_data.tsv --format pa-matrix --scorers fisher theil_u

# Advanced options
asymcat sequences.tsv \
  --scorers all \
  --smoothing laplace \
  --smoothing-alpha 0.5 \
  --sort-by yx \
  --top 10 \
  --output-format csv \
  --output top_associations.csv
```

#### Output Formats
```bash
# JSON output
asymcat data.tsv --scorers mle pmi --output-format json

# CSV for further analysis
asymcat data.tsv --scorers chi2 --output-format csv --precision 6

# Formatted tables
asymcat data.tsv --scorers fisher --table-format markdown
```

#### N-gram Analysis
```bash
# Bigram analysis
asymcat text_data.tsv --ngrams 2 --pad "#" --scorers tresoldi

# Filter by minimum co-occurrence count
asymcat large_dataset.tsv --min-count 5 --scorers mle pmi
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
make install  # Creates venv and installs with [dev] extras

# Run tests
make test
make coverage

# Code quality
make format     # Auto-format with black, isort
make lint       # Check with flake8
make mypy       # Type checking
```

### Testing
```bash
# Full test suite (75+ tests)
pytest

# Specific categories
pytest tests/unit/           # Unit tests only  
pytest tests/integration/    # Integration tests only
pytest -m slow              # Performance tests

# Quick development testing
make quick-test
```

### CLI Development
```bash
# Test CLI functionality
make cli-test
make cli-help

# Security scanning
make security
```

## 📚 Documentation

- **[Developer Guide](DEVELOPER.md)**: Comprehensive guide for contributors
- **[Jupyter Examples](docs/examples/)**: Academic notebooks with detailed analysis
- **[API Documentation](https://asymcat.readthedocs.io/)**: Complete API reference
- **[Mathematical Foundations](docs/MATHEMATICAL_FOUNDATIONS.md)**: Theory and formulas

## 🤝 Contributing

We welcome contributions! Please see our [Developer Guide](DEVELOPER.md) for:

- Setting up the development environment
- Code style guidelines and testing requirements
- Submitting bug reports and feature requests
- Contributing new association measures or improvements

### Quick Start for Contributors
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run the test suite: `make test`
5. Submit a pull request

## 📖 Citation

If you use ASymCat in your research, please cite:

```bibtex
@software{tresoldi_asymcat_2024,
  title = {ASymCat: Asymmetric Categorical Association Analysis},
  author = {Tresoldi, Tiago},
  year = {2024},
  url = {https://github.com/tresoldi/asymcat},
  version = {0.3.0}
}
```

## 🏆 Acknowledgments

- **FreqProb Library**: Robust probability estimation and smoothing
- **SciPy Community**: Statistical foundations
- **Linguistic Community**: Inspiration from historical linguistics applications

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🚀 What's New in v0.3.0

- ✅ **Modern Test Suite**: 75+ comprehensive tests with full parametrization
- ✅ **Enhanced CLI**: New sorting, filtering, and output format options  
- ✅ **FreqProb Integration**: Improved numerical stability and smoothing
- ✅ **Performance Optimizations**: Faster computation for large datasets
- ✅ **Better Documentation**: Comprehensive guides and examples
- ✅ **Type Safety**: Full type annotations throughout codebase

## 🔮 Roadmap

- **Statistical Significance**: P-value calculations for all measures
- **Confidence Intervals**: Uncertainty quantification
- **GPU Acceleration**: CUDA support for massive datasets
- **Interactive Dashboards**: Web-based exploration tools
- **Extended Measures**: Additional domain-specific association metrics

---

**[⭐ Star us on GitHub](https://github.com/tresoldi/asymcat)** if you find ASymCat useful!