# ASymCat Documentation

This directory contains comprehensive documentation for ASymCat, including mathematical foundations, examples, and visualizations.

## 📚 Documentation Files

### Core Documentation
- **[MATHEMATICAL_FOUNDATIONS.md](MATHEMATICAL_FOUNDATIONS.md)** - Complete mathematical basis of asymmetric association measures
- **[EXAMPLES_WITH_PLOTS.ipynb](EXAMPLES_WITH_PLOTS.ipynb)** - Interactive notebook with comprehensive examples and visualizations
- **[CLAUDE.md](../CLAUDE.md)** - Development guide for Claude Code

### Visualizations
- **[generate_readme_plots.py](generate_readme_plots.py)** - Script to generate documentation plots
- **images/** - Generated visualization files

### Legacy Documentation
- **Demo.ipynb** - Original demonstration notebook
- **source/** - Sphinx documentation source files

## 🧮 Mathematical Overview

ASymCat implements **15+ association measures** for categorical co-occurrence analysis:

### Asymmetric Measures (Directional)
| Measure | Range | Interpretation |
|---------|-------|---------------|
| **MLE** | [0,1] | P(Y\|X) - conditional probability |
| **Theil's U** | [0,1] | Uncertainty reduction U(Y\|X) |
| **Lambda** | [0,1] | Prediction error reduction λ(Y\|X) |
| **Tresoldi** | [0,∞) | Combined MLE×PMI measure |

### Symmetric Measures (Bidirectional)
| Measure | Range | Interpretation |
|---------|-------|---------------|
| **PMI** | (-∞,+∞) | Information overlap, 0=independence |
| **Chi-square** | [0,+∞) | Deviation from independence |
| **Jaccard** | [0,1] | Context overlap similarity |
| **Cramér's V** | [0,1] | Normalized chi-square |

## 🔬 Key Features

### Robust Probability Estimation
- **MLE (Maximum Likelihood)**: Direct estimation P(Y|X) = count(X,Y)/count(X)
- **Laplace Smoothing**: Adds pseudo-count of 1 to handle sparse data
- **Lidstone Smoothing**: Parameterized smoothing with adjustable γ parameter

### Comprehensive Data Support
- **Sequential Data**: Aligned sequences (orthography↔phonetics)
- **Presence-Absence**: Binary matrices (species×locations)
- **Categorical Features**: Any categorical co-occurrence data

### Advanced Analytics
- **N-gram Analysis**: Extract patterns from sequential data
- **Matrix Operations**: Convert scores to visualization matrices
- **Score Transformations**: Scaling, normalization, inversion
- **Statistical Tests**: Fisher exact test, log-likelihood ratios

## 📊 Usage Examples

### Basic Analysis
```python
import asymcat

# Load data and compute co-occurrences
data = asymcat.read_sequences("data.tsv")
cooccs = asymcat.collect_cooccs(data)

# Create scorer with smoothing
scorer = asymcat.scorer.CatScorer(cooccs,
                                  smoothing_method='laplace')

# Compute multiple measures
results = {
    'MLE': scorer.mle(),
    'PMI': scorer.pmi(),
    'Theil_U': scorer.theil_u(),
    'PMI_Smoothed': scorer.pmi_smoothed()
}
```

### CLI Analysis
```bash
# Basic analysis
asymcat data.tsv --scorers mle pmi theil_u

# With smoothing for sparse data
asymcat data.tsv --scorers mle pmi_smoothed --smoothing laplace

# Advanced parameterized smoothing
asymcat data.tsv --smoothing lidstone --smoothing-alpha 0.5
```

## 🌍 Real-World Applications

### Computational Linguistics
- **Phoneme-grapheme correspondence**: English orthography → IPA mappings
- **Historical linguistics**: Sound change directionality
- **Morphological analysis**: Affix dependency patterns
- **Syntactic relationships**: Word order asymmetries

### Ecological Analysis
- **Species co-occurrence**: Galápagos finch distribution patterns
- **Habitat associations**: Environmental factor dependencies
- **Predator-prey relationships**: Directional ecological dependencies
- **Biogeographical patterns**: Range overlap asymmetries

### Data Science Applications
- **Feature engineering**: Variable dependency discovery
- **Classification analysis**: Feature → class predictive strength
- **Market research**: Product purchase dependencies
- **Causal inference**: Directional relationship identification

## 🎯 Choosing the Right Measure

### For Prediction Tasks
- **MLE**: When you need interpretable conditional probabilities
- **Theil's U**: When measuring information-theoretic uncertainty reduction
- **Lambda**: When measuring prediction error reduction

### For Association Discovery
- **PMI**: When you want information-theoretic association strength
- **Chi-square**: When testing statistical independence
- **Jaccard**: When measuring context overlap similarity

### For Sparse Data
- **Use smoothing**: Laplace or Lidstone methods
- **pmi_smoothed()**: PMI with numerical stability
- **Higher α/γ values**: For more aggressive smoothing

## 📖 Further Reading

1. **[MATHEMATICAL_FOUNDATIONS.md](MATHEMATICAL_FOUNDATIONS.md)** - Complete mathematical derivations
2. **[EXAMPLES_WITH_PLOTS.ipynb](EXAMPLES_WITH_PLOTS.ipynb)** - Interactive examples with visualizations
3. **Main README** - Quick start and installation guide
4. **API Documentation** - Auto-generated from docstrings

## 🤝 Contributing

See the main repository for contribution guidelines. Documentation improvements are especially welcome!

---

**Mathematical Foundation**: ASymCat implements rigorous statistical and information-theoretic measures with comprehensive numerical stability through freqprob integration.

**Visualization**: All measures can be visualized as heatmaps, networks, and comparative plots for intuitive interpretation.

**Scalability**: Efficient algorithms handle datasets from small examples to large-scale corpus analysis.
