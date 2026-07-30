# ASymCat: LLM Agent Documentation

## Overview

ASymCat is a Python library for analyzing **asymmetric associations** between categorical variables. Unlike traditional symmetric measures (like Pearson's χ² or Cramér's V) that treat relationships as bidirectional, ASymCat quantifies directional dependencies: how well X predicts Y versus how well Y predicts X.

**Key Use Cases:**
- Linguistics: Grapheme-to-phoneme correspondences, phonological transitions
- Ecology: Species co-occurrence patterns, biogeographical relationships
- Market Analysis: Product purchase sequences, customer behavior patterns
- Machine Learning: Feature selection for categorical classification
- Medical Research: Symptom-to-diagnosis directionality

**Installation:**
```bash
pip install asymcat
```

**Optional dependencies:**
```bash
pip install asymcat[viz]    # Plotly, Bokeh, Altair for advanced visualization
pip install asymcat[dev]    # All development tools
pip install asymcat[all]    # Everything
```

**Core Philosophy:**

Symmetric measures answer: "Are X and Y associated?"
Asymmetric measures answer: "How well does X predict Y? How well does Y predict X?"

**Example - Why Asymmetry Matters:**

In medical data, if we see `(fever, flu)` pairs:
- P(flu|fever) might be low (many conditions cause fever)
- P(fever|flu) might be high (flu usually causes fever)
- A symmetric measure would miss this directional relationship!

---

## Quick Start

### Minimal Working Example

```python
import asymcat

# Create co-occurrence data: list of (X, Y) tuples
cooccs = [
    ('a', 'x'), ('a', 'x'), ('a', 'y'),  # 'a' appears with 'x' (67%), 'y' (33%)
    ('b', 'x'),                           # 'b' appears with 'x' (100%)
]

# Create scorer
scorer = asymcat.scorer.CatScorer(cooccs)

# Get conditional probabilities
mle = scorer.mle()
print(mle[('a', 'x')])  # (P(x|a), P(a|x))
# Output: (0.667, 0.667)  # Both directions
```

### Basic Workflow

```python
import asymcat

# Step 1: Load data from file
seqs = asymcat.read_sequences("data/alignments.tsv")
# Or for presence-absence matrices:
# data = asymcat.read_pa_matrix("data/species.tsv")

# Step 2: Collect co-occurrences
cooccs = asymcat.collect_cooccs(seqs)

# Step 3: Create scorer with smoothing
scorer = asymcat.scorer.CatScorer(
    cooccs,
    smoothing_method='laplace',  # Options: 'mle', 'laplace', 'lidstone'
    smoothing_alpha=1.0
)

# Step 4: Compute association measures
mle = scorer.mle()              # Conditional probabilities
pmi = scorer.pmi()              # Pointwise mutual information
chi2 = scorer.chi2()            # Chi-square statistic
theil = scorer.theil_u()        # Theil's uncertainty coefficient

# Step 5: Analyze results
for (x, y), (xy_score, yx_score) in mle.items():
    asymmetry = abs(xy_score - yx_score)
    if asymmetry > 0.3:
        print(f"Strong asymmetry: {x}→{y} = {xy_score:.3f}, {y}→{x} = {yx_score:.3f}")
```

### Common Patterns

**Pattern 1: Find strongest directional associations**
```python
tresoldi_scores = scorer.tresoldi()  # Best for sequence alignments

# Sort by directional strength
sorted_pairs = sorted(
    tresoldi_scores.items(),
    key=lambda x: max(x[1]),  # Max of (xy, yx)
    reverse=True
)

for (x, y), (xy, yx) in sorted_pairs[:10]:
    stronger = "X→Y" if xy > yx else "Y→X"
    print(f"{x} ↔ {y}: {stronger} (scores: {xy:.3f}, {yx:.3f})")
```

**Pattern 2: Compare multiple measures**
```python
measures = {
    'MLE': scorer.mle(),
    'PMI': scorer.pmi(normalized=True),
    'Theil_U': scorer.theil_u(),
    'Chi2': scorer.chi2(),
}

target_pair = ('a', 'x')
for name, scores in measures.items():
    xy, yx = scores[target_pair]
    print(f"{name:12s}: X→Y={xy:.4f}, Y→X={yx:.4f}")
```

### Model Evaluation Basics

```python
# Scale scores to [0, 1] for interpretation
from asymcat.scorer import scale_scorer

scaled_pmi = scale_scorer(pmi, method="minmax", nrange=(0, 1))

# Invert measures where lower = stronger (like entropy)
from asymcat.scorer import invert_scorer

entropy = scorer.cond_entropy()  # Lower = stronger
inverted = invert_scorer(entropy)  # Now higher = stronger

# Convert to matrices for visualization
from asymcat.scorer import scorer2matrices

mat_xy, mat_yx, alph_x, alph_y = scorer2matrices(scaled_pmi)
# mat_xy[i, j] = score from alph_x[i] to alph_y[j]
```

---

## Core Concepts

### Type System

ASymCat uses clear type definitions for safety and clarity:

```python
from typing import Any

# Element: Any hashable type
element_str: Any = "phoneme_a"
element_tuple: Any = ("bigram", "context")
element_int: Any = 42

# CooccDict: Co-occurrence counts
CooccDict = dict[tuple[Any, Any], int]
cooccs: list[tuple[Any, Any]] = [('a', 'x'), ('b', 'y')]

# ScorerDict: Association scores
ScorerDict = dict[tuple[Any, Any], tuple[float, float]]
# Maps (x, y) → (score_xy, score_yx)

# ObservationDict: Contingency table data
ObservationDict = dict[tuple[Any, Any], dict[str, int]]
# Maps (x, y) → {'00': total, '11': both_match, ...}
```

### Data Formats

**1. Co-occurrence Lists**

Direct pairs of observed elements:

```python
cooccs = [
    ('cat', 'meow'),
    ('dog', 'bark'),
    ('cat', 'purr'),
    ('dog', 'bark'),
]

scorer = asymcat.scorer.CatScorer(cooccs)
```

**2. Parallel Sequences**

Aligned sequences from TSV files:

```
# File: grapheme_phoneme.tsv
Orthography             Phonetics
C A T                   k æ t
D O G                   d ɔ g
```

```python
seqs = asymcat.read_sequences("grapheme_phoneme.tsv")
# seqs = [[['C', 'A', 'T'], ['k', 'æ', 't']], ...]

# Collect all element pairs (Cartesian product of each sequence pair)
cooccs = asymcat.collect_cooccs(seqs)
# Yields: [('C', 'k'), ('C', 'æ'), ('C', 't'), ('A', 'k'), ...]
```

**Note:** `collect_cooccs()` creates the Cartesian product of all elements in each sequence pair. If you need position-aligned pairs, sub-windows, or other structures, preprocess your data before calling this function.

**3. Presence-Absence Matrices**

Binary occurrence data:

```
# File: species_data.tsv
ID          Species_A   Species_B   Species_C
Island1     1           1           0
Island2     1           0           1
```

```python
combinations = asymcat.read_pa_matrix("species_data.tsv")
# Returns: [('Species_A', 'Species_B'), ('Species_A', 'Species_C'), ...]

scorer = asymcat.scorer.CatScorer(combinations)
```

### Asymmetric Relationships

The fundamental concept: for each pair (x, y), we compute:

1. **X→Y direction**: How well does X predict Y?
   - `P(Y|X)`: Conditional probability
   - Answers: "Given X, what's the probability of Y?"

2. **Y→X direction**: How well does Y predict X?
   - `P(X|Y)`: Reverse conditional probability
   - Answers: "Given Y, what's the probability of X?"

**All scorer methods return:** `dict[(x, y)] → (score_xy, score_yx)`

```python
mle = scorer.mle()
x_to_y, y_to_x = mle[('doctor', 'treatment')]

# Interpret asymmetry
if x_to_y > y_to_x:
    print("Doctor predicts treatment better than treatment predicts doctor")
```

### Common Parameters

**CatScorer Constructor:**

```python
scorer = asymcat.scorer.CatScorer(
    cooccs,                      # Required: list of (x, y) tuples
    smoothing_method='mle',      # 'mle', 'laplace', 'lidstone'
    smoothing_alpha=1.0          # Smoothing parameter (for laplace/lidstone)
)
```

**Parameter Guidelines:**
- `cooccs`: List of tuples representing observed pairs (must be non-empty)
- `smoothing_method`:
  - `'mle'`: Maximum likelihood (no smoothing) - use with abundant data
  - `'laplace'`: Add-1 smoothing (alpha=1.0) - general purpose
  - `'lidstone'`: Add-α smoothing - tune with `smoothing_alpha`
- `smoothing_alpha`: Only used with `'laplace'` or `'lidstone'` (default: 1.0)

**Common Method Parameters:**

```python
# PMI normalization
pmi = scorer.pmi(normalized=True)   # NPMI in [-1, 1]
pmi = scorer.pmi(normalized=False)  # Raw PMI in (-∞, +∞)

# Contingency table type
chi2 = scorer.chi2(square_ct=True)   # 2x2 table (default)
chi2 = scorer.chi2(square_ct=False)  # 3x2 table with marginals
```

---

## Smoothing Methods

Smoothing prevents zero probabilities and improves generalization to unseen pairs.

### 1. MLE (Maximum Likelihood Estimation)

**Formula:** `P(y|x) = count(x,y) / count(x)`

**When to use:**
- Abundant, dense data
- No unseen pairs expected
- Baseline for comparison

**Limitations:** Assigns zero probability to unseen pairs

```python
scorer = asymcat.scorer.CatScorer(cooccs, smoothing_method='mle')
mle_scores = scorer.mle()
```

### 2. Laplace Smoothing (Add-1)

**Formula:** `P(y|x) = (count(x,y) + 1) / (count(x) + V)`

where V is the vocabulary size.

**When to use:**
- Sparse data with unseen pairs
- General-purpose smoothing
- Small to medium vocabularies

**Effect:** Gives non-zero probability to all pairs

```python
scorer = asymcat.scorer.CatScorer(cooccs, smoothing_method='laplace')
mle_scores = scorer.mle()  # Now uses Laplace-smoothed probabilities
```

### 3. Lidstone Smoothing (Add-α)

**Formula:** `P(y|x) = (count(x,y) + α) / (count(x) + α×V)`

**When to use:**
- Fine-tune smoothing strength
- Very sparse data (small α like 0.1)
- Domain-specific requirements

```python
# Conservative smoothing
scorer = asymcat.scorer.CatScorer(
    cooccs,
    smoothing_method='lidstone',
    smoothing_alpha=0.1  # Small alpha = less smoothing
)

# Aggressive smoothing
scorer = asymcat.scorer.CatScorer(
    cooccs,
    smoothing_method='lidstone',
    smoothing_alpha=2.0  # Large alpha = more smoothing
)
```

### Smoothing Comparison

```python
# Create same data with different smoothing
cooccs = [('a', 'x'), ('a', 'x'), ('b', 'y')]

scorers = {
    'MLE': asymcat.scorer.CatScorer(cooccs, smoothing_method='mle'),
    'Laplace': asymcat.scorer.CatScorer(cooccs, smoothing_method='laplace'),
    'Lidstone_0.1': asymcat.scorer.CatScorer(cooccs, smoothing_method='lidstone', smoothing_alpha=0.1),
    'Lidstone_1.0': asymcat.scorer.CatScorer(cooccs, smoothing_method='lidstone', smoothing_alpha=1.0),
}

# Compare unseen pair ('c', 'z')
for name, scorer in scorers.items():
    probs = scorer.get_smoothed_probabilities()
    p_joint = probs['joint'].get(('c', 'z'), 0.0)
    print(f"{name:15s}: P(c,z) = {p_joint:.6f}")
```

**Output interpretation:**
- MLE will show 0.0 for unseen pairs
- Laplace will show small non-zero probability
- Lidstone probabilities vary with α

---

## Association Measures Guide

### Quick Reference Table

| Measure | Type | Range | Symmetric? | Best For |
|---------|------|-------|------------|----------|
| MLE | Probabilistic | [0, 1] | No | Direct interpretability |
| Jaccard | Probabilistic | [0, 1] | Yes | Context overlap |
| PMI | Information | (-∞, +∞) | Yes | Information content |
| NPMI | Information | [-1, 1] | Yes | Normalized information |
| MI | Information | [0, +∞) | Yes | Statistical dependence |
| Conditional Entropy | Information | [0, +∞) | No | Uncertainty reduction |
| Chi-Square | Statistical | [0, +∞) | Yes | Independence testing |
| Cramér's V | Statistical | [0, 1] | Yes | Normalized chi-square |
| Fisher | Statistical | [0, +∞) | Yes | Small samples, exact test |
| Log-Likelihood | Statistical | [0, +∞) | Yes | Better than chi-square for small expected freq |
| Theil's U | Specialized | [0, 1] | No | Entropy-based prediction |
| Tresoldi | Specialized | Real | No | Sequence alignment |
| Goodman-Kruskal λ | Specialized | [0, 1] | No | Error reduction |

### Measure Selection Guide

**Decision Tree:**

1. **What's your primary goal?**

   **→ Direct probability interpretation:**
   - Use: `mle()` or `get_smoothed_probabilities()`
   - Best when: You need P(Y|X) directly

   **→ Statistical significance testing:**
   - Use: `chi2()` or `fisher()` or `log_likelihood_ratio()`
   - Best when: You need p-values or hypothesis testing

   **→ Information-theoretic analysis:**
   - Use: `pmi()`, `mutual_information()`, or `theil_u()`
   - Best when: Measuring information content or entropy

   **→ Sequence alignment scoring:**
   - Use: `tresoldi()`
   - Best when: Aligning phonetic/orthographic sequences

2. **Data characteristics?**

   **→ Sparse data:**
   - Use smoothing: `smoothing_method='laplace'` or `'lidstone'`
   - Measures: `pmi_smoothed()`, `theil_u()`

   **→ Small sample sizes:**
   - Use: `fisher()` (exact test)
   - Avoid: `chi2()` if expected frequencies < 5

   **→ Large datasets:**
   - Use: `chi2()`, `log_likelihood_ratio()` (faster than Fisher)
   - Avoid: `fisher()` (very slow)

3. **Need asymmetry?**

   **→ Yes, directional predictions:**
   - Use: `mle()`, `theil_u()`, `cond_entropy()`, `goodman_kruskal_lambda()`

   **→ No, symmetric association:**
   - Use: `chi2()`, `cramers_v()`, `pmi()`, `jaccard_index()`
   - Note: Even symmetric measures return (score, score) tuples

### Category-by-Category

#### Probabilistic Measures

**1. MLE (Maximum Likelihood Estimation)**

Direct conditional probabilities.

```python
mle = scorer.mle()
p_y_given_x, p_x_given_y = mle[('x', 'y')]

# Interpretation: P(y|x) = "probability of y when we observe x"
if p_y_given_x > 0.8:
    print("x strongly predicts y")
```

**2. Jaccard Index**

Set overlap measure based on contexts.

```python
jaccard = scorer.jaccard_index()
similarity, _ = jaccard[('x', 'y')]

# Interpretation: How much do x and y share contexts?
# Range [0, 1]: 0 = no overlap, 1 = perfect overlap
```

**3. Smoothed Probabilities (Advanced)**

Complete probability distribution.

```python
probs = scorer.get_smoothed_probabilities()

p_xy = probs['joint'][('x', 'y')]       # P(x, y)
p_y_given_x = probs['yx_given_x'][('x', 'y')]  # P(y|x)
p_x_given_y = probs['xy_given_y'][('x', 'y')]  # P(x|y)
p_x = probs['marginal_x']['x']          # P(x)
p_y = probs['marginal_y']['y']          # P(y)

# Verify: P(y|x) = P(x,y) / P(x)
assert abs(p_y_given_x - (p_xy / p_x)) < 1e-6
```

#### Information-Theoretic Measures

**1. PMI (Pointwise Mutual Information)**

Measures how much more likely x and y co-occur than expected by chance.

```python
pmi = scorer.pmi(normalized=False)
pmi_value, _ = pmi[('x', 'y')]

# Interpretation:
# > 0: x and y co-occur more than expected (attraction)
# = 0: x and y are independent
# < 0: x and y co-occur less than expected (repulsion)

# Normalized version [-1, 1]
npmi = scorer.pmi(normalized=True)
npmi_value, _ = npmi[('x', 'y')]
```

**2. Smoothed PMI (Recommended)**

PMI with numerical stability from FreqProb smoothing.

```python
scorer = asymcat.scorer.CatScorer(cooccs, smoothing_method='lidstone', smoothing_alpha=0.5)
pmi_smooth = scorer.pmi_smoothed(normalized=True)

# Better than pmi() for sparse data
```

**3. Mutual Information**

Average information shared between variables.

```python
mi = scorer.mutual_information()
mi_xy, mi_yx = mi[('x', 'y')]

# Interpretation:
# 0 = independent variables
# Higher = stronger statistical dependence
# Note: Symmetric in theory, computed for both directions
```

**4. Conditional Entropy**

Uncertainty remaining about Y after observing X.

```python
entropy = scorer.cond_entropy()
h_y_given_x, h_x_given_y = entropy[('x', 'y')]

# Interpretation: Lower = stronger association
# H(Y|X) = "How uncertain am I about Y after knowing X?"
# 0 = X perfectly predicts Y
# High = X doesn't help predict Y

# Often better to invert for interpretation
from asymcat.scorer import invert_scorer
inverted = invert_scorer(entropy)  # Now higher = stronger
```

**5. Normalized Mutual Information**

MI normalized by joint entropy.

```python
nmi = scorer.normalized_mutual_information()
nmi_value, _ = nmi[('x', 'y')]

# Range [0, 1]
# 0 = independent
# 1 = perfectly dependent
```

**6. Theil's U (Uncertainty Coefficient)**

Proportional reduction in uncertainty about Y when knowing X.

```python
theil = scorer.theil_u()
u_y_given_x, u_x_given_y = theil[('x', 'y')]

# Interpretation:
# U(Y|X) = "How much does knowing X reduce uncertainty about Y?"
# Range [0, 1]
# 0 = X provides no information about Y
# 1 = X perfectly predicts Y
# Asymmetric: U(Y|X) ≠ U(X|Y) in general
```

#### Statistical Measures

**1. Chi-Square (χ²)**

Test for independence between categorical variables.

```python
chi2 = scorer.chi2(square_ct=True)
chi2_value, _ = chi2[('x', 'y')]

# Interpretation:
# 0 = perfectly independent
# Higher = stronger association
# No upper bound
# Symmetric measure

# Warning: Not valid if expected frequencies < 5
# Use Fisher's exact test instead for small samples
```

**2. Cramér's V**

Normalized chi-square for easier interpretation.

```python
cramers = scorer.cramers_v(square_ct=True)
v_value, _ = cramers[('x', 'y')]

# Range [0, 1]
# 0 = independent
# 1 = perfect association
# Symmetric measure
# Bias-corrected for small samples
```

**3. Fisher's Exact Test**

Exact odds ratio for small samples.

```python
fisher = scorer.fisher()
odds_ratio, _ = fisher[('x', 'y')]

# Interpretation:
# Odds ratio = (n11 × n22) / (n12 × n21)
# > 1: Positive association
# = 1: No association
# < 1: Negative association

# WARNING: Very slow for large contingency tables
# Use chi2() or log_likelihood_ratio() instead
```

**4. Log-Likelihood Ratio (G²)**

Alternative to chi-square, better for small expected frequencies.

```python
llr = scorer.log_likelihood_ratio(square_ct=True)
g2_value, _ = llr[('x', 'y')]

# Similar interpretation to chi-square
# 0 = independent
# Higher = stronger association
# Asymptotically equivalent to chi-square
# More reliable for small expected frequencies
```

#### Specialized Measures

**1. Tresoldi Measure**

Custom measure designed for sequence alignment tasks.

```python
tresoldi = scorer.tresoldi()
score_xy, score_yx = tresoldi[('x', 'y')]

# Combines PMI and MLE:
# T(x,y) = PMI^(1-MLE)  (with sign handling)
# Designed for phonetic/orthographic alignments
# Asymmetric by design
# Recommended for linguistic applications
```

**2. Goodman-Kruskal Lambda (λ)**

Proportional reduction in prediction error.

```python
lambda_scores = scorer.goodman_kruskal_lambda()
lambda_y_given_x, lambda_x_given_y = lambda_scores[('x', 'y')]

# Interpretation:
# λ(Y|X) = "How much does knowing X improve prediction of Y?"
# Range [0, 1]
# 0 = No improvement
# 1 = Perfect prediction
# Asymmetric: λ(Y|X) ≠ λ(X|Y)
```

---

## Common Patterns

### Pattern 1: Loading Different Data Types

**From parallel sequences:**

```python
# TSV file with aligned sequences
seqs = asymcat.read_sequences("alignments.tsv")
cooccs = asymcat.collect_cooccs(seqs)

# With specific columns
seqs = asymcat.read_sequences(
    "data.tsv",
    cols=['Column1', 'Column2'],  # Select specific columns
    col_delim="\t",               # Field delimiter
    elem_delim=" "                # Element delimiter within fields
)
```

**From presence-absence matrix:**

```python
# Binary occurrence data
combinations = asymcat.read_pa_matrix("species.tsv")
scorer = asymcat.scorer.CatScorer(combinations)
```

**From Python data structures:**

```python
# Direct list of tuples
cooccs = [('a', 'x'), ('b', 'y'), ('a', 'x')]

# From pandas DataFrame
import pandas as pd
df = pd.DataFrame({'X': ['a', 'b', 'a'], 'Y': ['x', 'y', 'x']})
cooccs = list(zip(df['X'], df['Y']))

# From numpy arrays
import numpy as np
x_array = np.array(['a', 'b', 'a'])
y_array = np.array(['x', 'y', 'x'])
cooccs = list(zip(x_array, y_array))
```

### Pattern 2: Computing Multiple Measures

**All measures at once:**

```python
scorer = asymcat.scorer.CatScorer(cooccs, smoothing_method='laplace')

# Compute all relevant measures
results = {
    'MLE': scorer.mle(),
    'PMI': scorer.pmi(normalized=True),
    'MI': scorer.mutual_information(),
    'Theil_U': scorer.theil_u(),
    'Chi2': scorer.chi2(),
    'Cramers_V': scorer.cramers_v(),
    'Tresoldi': scorer.tresoldi(),
}

# Analyze specific pair across all measures
target = ('a', 'x')
print(f"Analysis for {target}:")
for measure_name, scores in results.items():
    xy, yx = scores[target]
    print(f"  {measure_name:15s}: X→Y={xy:.4f}, Y→X={yx:.4f}")
```

**Selective computation:**

```python
# Only compute what you need
if need_probabilities:
    mle = scorer.mle()

if need_statistical_test:
    chi2 = scorer.chi2()

if analyzing_sequences:
    tresoldi = scorer.tresoldi()
```

### Pattern 3: Comparing X→Y vs Y→X

**Find asymmetric pairs:**

```python
mle = scorer.mle()

asymmetric_pairs = []
for (x, y), (p_y_given_x, p_x_given_y) in mle.items():
    asymmetry = abs(p_y_given_x - p_x_given_y)

    if asymmetry > 0.3:  # Threshold
        stronger_dir = f"{x}→{y}" if p_y_given_x > p_x_given_y else f"{y}→{x}"
        asymmetric_pairs.append({
            'pair': (x, y),
            'stronger': stronger_dir,
            'asymmetry': asymmetry,
            'xy_score': p_y_given_x,
            'yx_score': p_x_given_y,
        })

# Sort by asymmetry
asymmetric_pairs.sort(key=lambda d: d['asymmetry'], reverse=True)

for item in asymmetric_pairs[:10]:
    print(f"{item['stronger']:10s}: asymmetry={item['asymmetry']:.3f}")
```

**Direction analysis:**

```python
# Count directional preferences
xy_stronger = 0
yx_stronger = 0

for (x, y), (xy_score, yx_score) in mle.items():
    if xy_score > yx_score:
        xy_stronger += 1
    elif yx_score > xy_score:
        yx_stronger += 1

print(f"X→Y stronger: {xy_stronger} pairs")
print(f"Y→X stronger: {yx_stronger} pairs")
```

### Pattern 4: Visualizing Results

**Heatmaps:**

```python
import matplotlib.pyplot as plt
import seaborn as sns
from asymcat.scorer import scorer2matrices, scale_scorer

# Get and scale scores
pmi = scorer.pmi(normalized=True)
scaled_pmi = scale_scorer(pmi, method="minmax", nrange=(0, 1))

# Convert to matrices
mat_xy, mat_yx, alph_x, alph_y = scorer2matrices(scaled_pmi)

# Plot both directions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# X→Y heatmap
sns.heatmap(mat_xy, xticklabels=alph_y, yticklabels=alph_x,
            cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Score'})
ax1.set_title('X→Y Association Scores')
ax1.set_xlabel('Y Elements')
ax1.set_ylabel('X Elements')

# Y→X heatmap
sns.heatmap(mat_yx, xticklabels=alph_x, yticklabels=alph_y,
            cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Score'})
ax2.set_title('Y→X Association Scores')
ax2.set_xlabel('X Elements')
ax2.set_ylabel('Y Elements')

plt.tight_layout()
plt.show()
```

**Score distributions:**

```python
import numpy as np

# Extract all scores
all_xy_scores = [xy for xy, yx in mle.values()]
all_yx_scores = [yx for xy, yx in mle.values()]

plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(all_xy_scores, bins=20, alpha=0.6, label='X→Y', density=True)
plt.hist(all_yx_scores, bins=20, alpha=0.6, label='Y→X', density=True)
plt.xlabel('Score')
plt.ylabel('Density')
plt.title('Score Distribution Comparison')
plt.legend()

# Scatter plot
plt.subplot(1, 2, 2)
plt.scatter(all_xy_scores, all_yx_scores, alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--', label='Perfect symmetry')
plt.xlabel('X→Y Score')
plt.ylabel('Y→X Score')
plt.title('Directional Score Comparison')
plt.legend()

plt.tight_layout()
plt.show()
```

**Bar charts for specific pairs:**

```python
import numpy as np

pairs = list(mle.keys())[:10]  # Top 10 pairs
pair_labels = [f"{x}→{y}" for x, y in pairs]

xy_scores = [mle[p][0] for p in pairs]
yx_scores = [mle[p][1] for p in pairs]

x = np.arange(len(pairs))
width = 0.35

plt.figure(figsize=(14, 6))
plt.bar(x - width/2, xy_scores, width, label='X→Y', alpha=0.8)
plt.bar(x + width/2, yx_scores, width, label='Y→X', alpha=0.8)

plt.xlabel('Pairs')
plt.ylabel('Score')
plt.title('Asymmetric Association Scores')
plt.xticks(x, pair_labels, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Pattern 5: Working with N-grams

**Collect bigrams:**

```python
# From aligned sequences
seqs = asymcat.read_sequences("sequences.tsv")

# Collect bigram co-occurrences
bigrams = asymcat.collect_cooccs(seqs, order=2, pad='#')
# '#' is padding symbol for sequence boundaries

scorer = asymcat.scorer.CatScorer(bigrams)
```

**Collect trigrams:**

```python
trigrams = asymcat.collect_cooccs(seqs, order=3, pad='#')
```

**Generate n-grams from single sequence:**

```python
seq = ['a', 'b', 'c', 'd', 'e']

# Manual n-gram generation
bigrams_gen = asymcat.collect_ngrams(seq, order=2, pad='#')
bigrams = list(bigrams_gen)
# [('#', 'a'), ('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', '#')]

trigrams_gen = asymcat.collect_ngrams(seq, order=3, pad='#')
trigrams = list(trigrams_gen)
# [('#', '#', 'a'), ('#', 'a', 'b'), ('a', 'b', 'c'), ...]
```

**Aligned n-gram pairs:**

```python
# For aligned sequences of equal length
seqs = [
    [['a', 'b', 'c'], ['x', 'y', 'z']],
    [['d', 'e', 'f'], ['p', 'q', 'r']],
]

# Extract aligned bigram pairs
bigram_pairs = asymcat.collect_cooccs(seqs, order=2, pad='#')
# Each position in sequence 1 paired with same position in sequence 2
```

### Pattern 6: Handling Sparse Data

**Check data sparsity:**

```python
obs = asymcat.collect_observations(cooccs)
alphabet_x, alphabet_y = asymcat.collect_alphabets(cooccs)

total_possible = len(alphabet_x) * len(alphabet_y)
total_observed = len(obs)
sparsity = 1 - (total_observed / total_possible)

print(f"Vocabulary: {len(alphabet_x)} × {len(alphabet_y)} = {total_possible} possible pairs")
print(f"Observed: {total_observed} pairs")
print(f"Sparsity: {sparsity:.1%}")
```

**Apply appropriate smoothing:**

```python
if sparsity > 0.7:  # Very sparse
    scorer = asymcat.scorer.CatScorer(cooccs, smoothing_method='lidstone', smoothing_alpha=0.1)
elif sparsity > 0.3:  # Moderately sparse
    scorer = asymcat.scorer.CatScorer(cooccs, smoothing_method='laplace')
else:  # Dense
    scorer = asymcat.scorer.CatScorer(cooccs, smoothing_method='mle')
```

**Compare smoothing effects:**

```python
smoothing_configs = [
    ('MLE', 'mle', 1.0),
    ('Laplace', 'laplace', 1.0),
    ('Lidstone_0.1', 'lidstone', 0.1),
    ('Lidstone_0.5', 'lidstone', 0.5),
]

results = {}
for name, method, alpha in smoothing_configs:
    s = asymcat.scorer.CatScorer(cooccs, smoothing_method=method, smoothing_alpha=alpha)
    mle = s.mle()

    # Count zero probabilities
    zero_count = sum(1 for xy, yx in mle.values() if xy == 0 or yx == 0)
    results[name] = zero_count

print("Zero probability counts by smoothing method:")
for name, count in results.items():
    print(f"  {name:15s}: {count} zero probabilities")
```

---

## Integration Examples

### With pandas DataFrames

**Convert DataFrame to co-occurrences:**

```python
import pandas as pd
import asymcat

# Your data
df = pd.DataFrame({
    'customer': ['Alice', 'Bob', 'Alice', 'Charlie'],
    'product': ['laptop', 'phone', 'tablet', 'laptop'],
    'purchase_count': [1, 2, 1, 1]
})

# Simple: direct conversion
cooccs = list(zip(df['customer'], df['product']))

# Weighted: repeat by counts
cooccs_weighted = []
for _, row in df.iterrows():
    cooccs_weighted.extend([(row['customer'], row['product'])] * int(row['purchase_count']))

scorer = asymcat.scorer.CatScorer(cooccs_weighted)
mle = scorer.mle()
```

**Results to DataFrame:**

```python
# Convert scores to DataFrame
results = []
for (x, y), (xy_score, yx_score) in mle.items():
    results.append({
        'X': x,
        'Y': y,
        'X_to_Y': xy_score,
        'Y_to_X': yx_score,
        'Asymmetry': abs(xy_score - yx_score)
    })

results_df = pd.DataFrame(results)

# Sort by asymmetry
results_df = results_df.sort_values('Asymmetry', ascending=False)
print(results_df.head(10))

# Export to CSV
results_df.to_csv('association_scores.csv', index=False)
```

### With matplotlib/seaborn

**Publication-quality heatmaps:**

```python
import matplotlib.pyplot as plt
import seaborn as sns
from asymcat.scorer import scorer2matrices, scale_scorer

# Configure for publication
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# Get scores
tresoldi = scorer.tresoldi()
scaled = scale_scorer(tresoldi, method="minmax", nrange=(0, 1))
mat_xy, mat_yx, alph_x, alph_y = scorer2matrices(scaled)

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    mat_xy,
    xticklabels=alph_y,
    yticklabels=alph_x,
    cmap='RdYlBu_r',
    center=0.5,
    square=True,
    linewidths=0.5,
    cbar_kws={'label': 'Association Strength'},
    ax=ax
)

ax.set_title('Asymmetric Association: X→Y', fontsize=14, fontweight='bold')
ax.set_xlabel('Y Elements', fontsize=12)
ax.set_ylabel('X Elements', fontsize=12)

plt.tight_layout()
plt.savefig('association_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Multi-panel comparisons:**

```python
measures = ['MLE', 'PMI', 'Theil_U', 'Tresoldi']
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, measure_name in enumerate(measures):
    if measure_name == 'MLE':
        scores = scorer.mle()
    elif measure_name == 'PMI':
        scores = scorer.pmi(normalized=True)
    elif measure_name == 'Theil_U':
        scores = scorer.theil_u()
    elif measure_name == 'Tresoldi':
        scores = scorer.tresoldi()

    scaled = scale_scorer(scores, method="minmax", nrange=(0, 1))
    mat_xy, mat_yx, alph_x, alph_y = scorer2matrices(scaled)

    sns.heatmap(mat_xy, ax=axes[idx], cmap='YlOrRd',
                xticklabels=alph_y, yticklabels=alph_x)
    axes[idx].set_title(f'{measure_name} (X→Y)', fontsize=12)

plt.tight_layout()
plt.savefig('measure_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

### With scikit-learn (Feature Selection)

**Use association scores for feature selection:**

```python
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder
import asymcat

# Your categorical features and target
X_categorical = [
    ['red', 'small'],
    ['blue', 'large'],
    ['red', 'large'],
    ['green', 'small'],
]
y = ['A', 'B', 'A', 'C']

# Encode features
le_features = [LabelEncoder() for _ in range(len(X_categorical[0]))]
le_target = LabelEncoder()

X_encoded = np.array([
    [le.fit_transform([row[i] for row in X_categorical])[idx]
     for i, le in enumerate(le_features)]
    for idx in range(len(X_categorical))
])
y_encoded = le_target.fit_transform(y)

# Compute asymmetric associations for each feature
feature_scores = []

for feature_idx in range(X_encoded.shape[1]):
    # Create co-occurrences: (feature_value, target_value)
    cooccs = list(zip(X_encoded[:, feature_idx], y_encoded))

    scorer = asymcat.scorer.CatScorer(cooccs, smoothing_method='laplace')
    theil = scorer.theil_u()

    # Average Theil's U for this feature
    avg_score = np.mean([max(xy, yx) for xy, yx in theil.values()])
    feature_scores.append(avg_score)

# Select top k features
k = 1
selected_features = np.argsort(feature_scores)[-k:]
print(f"Selected feature indices: {selected_features}")
print(f"Feature scores: {[feature_scores[i] for i in selected_features]}")
```

### Export/Import Workflows

**Save scores to file:**

```python
import json
import pickle

# JSON (human-readable)
mle = scorer.mle()
mle_serializable = {
    f"{x},{y}": [float(xy), float(yx)]
    for (x, y), (xy, yx) in mle.items()
}

with open('scores.json', 'w') as f:
    json.dump(mle_serializable, f, indent=2)

# Pickle (preserves exact types)
with open('scores.pkl', 'wb') as f:
    pickle.dump(mle, f)

# CSV for spreadsheets
import csv

with open('scores.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['X', 'Y', 'X_to_Y', 'Y_to_X', 'Asymmetry'])

    for (x, y), (xy, yx) in mle.items():
        writer.writerow([x, y, xy, yx, abs(xy - yx)])
```

**Load scores from file:**

```python
# From JSON
with open('scores.json', 'r') as f:
    loaded = json.load(f)

# Reconstruct dictionary
mle_loaded = {}
for key, (xy, yx) in loaded.items():
    x, y = key.split(',')
    mle_loaded[(x, y)] = (xy, yx)

# From pickle
with open('scores.pkl', 'rb') as f:
    mle_loaded = pickle.load(f)
```

---

## Troubleshooting

### Common Errors and Solutions

**Error: Empty co-occurrence list**

```python
# Problem
cooccs = []
scorer = asymcat.scorer.CatScorer(cooccs)  # ValueError

# Solution
if not cooccs:
    raise ValueError("No co-occurrences found. Check your data loading.")

# Verify data before processing
print(f"Loaded {len(cooccs)} co-occurrences")
assert len(cooccs) > 0, "Empty co-occurrence list"
```

**Error: Sequences have mismatched lengths**

```python
# Problem
seqs = [
    [['a', 'b'], ['x', 'y', 'z']],  # Different lengths!
]
cooccs = asymcat.collect_cooccs(seqs, order=2)  # ValueError

# Solution: Only use order parameter with aligned sequences
# Option 1: Don't use order (collects all pairs)
cooccs = asymcat.collect_cooccs(seqs)

# Option 2: Ensure sequences are aligned
seqs_aligned = [
    [['a', 'b', 'c'], ['x', 'y', 'z']],  # Same length
]
cooccs = asymcat.collect_cooccs(seqs_aligned, order=2)
```

**Error: File not found**

```python
# Problem
seqs = asymcat.read_sequences("missing_file.tsv")  # FileNotFoundError

# Solution: Check file path and existence
import os

filepath = "data/sequences.tsv"
if not os.path.exists(filepath):
    raise FileNotFoundError(f"File not found: {filepath}")

seqs = asymcat.read_sequences(filepath)
```

**Error: Fisher test too slow**

```python
# Problem
scorer = asymcat.scorer.CatScorer(large_dataset)
fisher = scorer.fisher()  # Takes forever...

# Solution: Use faster alternatives
chi2 = scorer.chi2()  # Much faster
llr = scorer.log_likelihood_ratio()  # Also fast

# Or sample data
import random
sample_size = 1000
sampled_cooccs = random.sample(cooccs, min(sample_size, len(cooccs)))
scorer = asymcat.scorer.CatScorer(sampled_cooccs)
fisher = scorer.fisher()
```

### Performance Tips

**1. Choose appropriate measures:**

```python
# Fast measures (O(n))
mle = scorer.mle()
pmi = scorer.pmi()
chi2 = scorer.chi2()

# Slower measures
fisher = scorer.fisher()  # Very slow for large data
theil = scorer.theil_u()  # Slower than others

# For large datasets:
# Use chi2() or log_likelihood_ratio() instead of fisher()
```

**2. Batch processing:**

```python
# Inefficient: Create new scorer for each subset
for subset in data_subsets:
    scorer = asymcat.scorer.CatScorer(subset)  # Overhead
    scores = scorer.mle()

# Efficient: Process all at once, then filter
all_cooccs = []
for subset in data_subsets:
    all_cooccs.extend(subset)

scorer = asymcat.scorer.CatScorer(all_cooccs)
all_scores = scorer.mle()

# Filter results as needed
subset1_scores = {k: v for k, v in all_scores.items() if k[0] in subset1_elements}
```

**3. Lazy evaluation:**

Scorers cache results, so accessing the same measure twice is fast:

```python
scorer = asymcat.scorer.CatScorer(cooccs)

# First call: computed and cached
mle1 = scorer.mle()  # ~1 second

# Second call: retrieved from cache
mle2 = scorer.mle()  # ~0.001 seconds
```

### Memory Considerations

**Check memory usage:**

```python
import sys

# Size of co-occurrence list
cooccs_size = sys.getsizeof(cooccs)
print(f"Co-occurrences: {cooccs_size / 1024 / 1024:.2f} MB")

# Size of scorer
scorer = asymcat.scorer.CatScorer(cooccs)
scorer_size = sys.getsizeof(scorer)
print(f"Scorer: {scorer_size / 1024 / 1024:.2f} MB")
```

**Reduce memory for large datasets:**

```python
# Process in chunks
chunk_size = 10000
results = []

for i in range(0, len(large_cooccs), chunk_size):
    chunk = large_cooccs[i:i+chunk_size]
    scorer = asymcat.scorer.CatScorer(chunk)
    mle = scorer.mle()
    results.append(mle)

# Merge results
merged = {}
for result in results:
    merged.update(result)
```

### Data Quality Issues

**Check for invalid data:**

```python
# Validate co-occurrences
def validate_cooccs(cooccs):
    if not isinstance(cooccs, list):
        raise TypeError("cooccs must be a list")

    for item in cooccs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"Invalid co-occurrence: {item}")

    return True

validate_cooccs(cooccs)
```

**Handle missing data:**

```python
# Remove None values
cooccs_clean = [(x, y) for x, y in cooccs if x is not None and y is not None]

# Remove empty strings
cooccs_clean = [(x, y) for x, y in cooccs if x != '' and y != '']

# Check for duplicates (usually fine, but verify)
from collections import Counter
pair_counts = Counter(cooccs)
duplicates = {pair: count for pair, count in pair_counts.items() if count > 10}
if duplicates:
    print(f"High-frequency pairs: {duplicates}")
```

**Debug unexpected scores:**

```python
# Investigate specific pair
target = ('a', 'x')

# Get raw observations
obs = asymcat.collect_observations(cooccs)
print(f"Observations for {target}:")
print(obs[target])

# Build contingency table
ct = asymcat.build_ct(obs[target])
print(f"Contingency table:")
print(ct)

# Check smoothed probabilities
probs = scorer.get_smoothed_probabilities()
print(f"P({target[0]}) = {probs['marginal_x'][target[0]]:.4f}")
print(f"P({target[1]}) = {probs['marginal_y'][target[1]]:.4f}")
print(f"P({target}) = {probs['joint'][target]:.4f}")
```

---

## API Quick Reference

### Data Loading Functions

```python
# Read parallel sequences from TSV
asymcat.read_sequences(filename, cols=None, col_delim="\t", elem_delim=" ")

# Read presence-absence matrix
asymcat.read_pa_matrix(filename, delimiter="\t")
```

### Data Processing Functions

```python
# Extract unique alphabets
asymcat.collect_alphabets(cooccs) → (alphabet_x, alphabet_y)

# Generate n-grams with padding
asymcat.collect_ngrams(seq, order, pad) → Generator[tuple]

# Collect co-occurrences from sequences
asymcat.collect_cooccs(seqs, order=None, pad="#") → list[tuple]

# Build observation statistics
asymcat.collect_observations(cooccs) → dict[tuple, dict[str, int]]

# Build contingency table
asymcat.build_ct(observ, square=True) → list[list]
```

### CatScorer Class

```python
# Constructor
scorer = asymcat.scorer.CatScorer(cooccs, smoothing_method='mle', smoothing_alpha=1.0)

# Probabilistic measures
scorer.mle() → ScorerDict
scorer.jaccard_index() → ScorerDict
scorer.get_smoothed_probabilities() → dict[str, dict]
scorer.goodman_kruskal_lambda() → ScorerDict

# Information-theoretic measures
scorer.pmi(normalized=False) → ScorerDict
scorer.pmi_smoothed(normalized=False) → ScorerDict
scorer.mutual_information() → ScorerDict
scorer.normalized_mutual_information() → ScorerDict
scorer.cond_entropy() → ScorerDict
scorer.theil_u() → ScorerDict

# Statistical measures
scorer.chi2(square_ct=True) → ScorerDict
scorer.cramers_v(square_ct=True) → ScorerDict
scorer.fisher() → ScorerDict
scorer.log_likelihood_ratio(square_ct=True) → ScorerDict

# Specialized measures
scorer.tresoldi() → ScorerDict
```

### Utility Functions

```python
from asymcat.scorer import scorer2matrices, scale_scorer, invert_scorer

# Convert scores to matrices
scorer2matrices(scorer) → (mat_xy, mat_yx, alph_x, alph_y)

# Scale scores to range
scale_scorer(scorer, method="minmax", nrange=(0, 1)) → ScorerDict
# method: "minmax", "mean", "stdev"

# Invert scores (higher = stronger)
invert_scorer(scorer) → ScorerDict
```

### Correlation Functions

```python
from asymcat import correlation

# Symmetric correlation measures for series
correlation.cramers_v(series_x, series_y) → float
correlation.conditional_entropy(series_x, series_y) → float
correlation.theil_u(series_x, series_y) → float
```

### Parameter Cheat Sheet

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cooccs` | list[tuple] | Required | Co-occurrence pairs |
| `smoothing_method` | str | 'mle' | 'mle', 'laplace', 'lidstone' |
| `smoothing_alpha` | float | 1.0 | Smoothing parameter (laplace/lidstone) |
| `normalized` | bool | False | Normalize PMI to [-1, 1] |
| `square_ct` | bool | True | Use 2x2 (True) or 3x2 (False) contingency table |
| `order` | int\|None | None | N-gram order (None = full sequences) |
| `pad` | str | "#" | Padding symbol for n-grams |
| `method` | str | "minmax" | Scaling method: "minmax", "mean", "stdev" |
| `nrange` | tuple | (0, 1) | Target range for minmax scaling |

### Return Types Quick Guide

```python
# ScorerDict: All association measures
dict[tuple[Any, Any], tuple[float, float]]
# Maps (x, y) → (score_xy, score_yx)

# Smoothed probabilities
dict[str, dict]
# Keys: 'xy_given_y', 'yx_given_x', 'joint', 'marginal_x', 'marginal_y'

# Matrices
tuple[np.ndarray, np.ndarray, list, list]
# (mat_xy, mat_yx, alphabet_x, alphabet_y)

# Alphabets
tuple[list, list]
# (alphabet_x, alphabet_y) - sorted unique elements

# Observations
dict[tuple[Any, Any], dict[str, int]]
# Maps (x, y) → {'00': count, '11': count, ...}
```

---

## Complete Example: Linguistic Analysis Pipeline

```python
import asymcat
from asymcat.scorer import scale_scorer, scorer2matrices
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 1. Load Data =====
# Grapheme-to-phoneme alignments
seqs = asymcat.read_sequences("grapheme_phoneme.tsv")
print(f"Loaded {len(seqs)} aligned sequences")

# ===== 2. Collect Bigram Co-occurrences =====
bigrams = asymcat.collect_cooccs(seqs, order=2, pad='#')
print(f"Collected {len(bigrams)} bigram co-occurrences")

# ===== 3. Create Scorer with Smoothing =====
scorer = asymcat.scorer.CatScorer(
    bigrams,
    smoothing_method='lidstone',
    smoothing_alpha=0.5
)

# ===== 4. Compute Multiple Measures =====
measures = {
    'MLE': scorer.mle(),
    'Tresoldi': scorer.tresoldi(),  # Best for alignments
    'PMI': scorer.pmi(normalized=True),
    'Theil_U': scorer.theil_u(),
}

# ===== 5. Find Strongest Alignments =====
tresoldi = measures['Tresoldi']
scaled_tresoldi = scale_scorer(tresoldi, method="minmax", nrange=(0, 1))

# Sort by max directional score
sorted_pairs = sorted(
    scaled_tresoldi.items(),
    key=lambda x: max(x[1]),
    reverse=True
)

print("\nTop 10 Grapheme→Phoneme Alignments:")
print("=" * 50)
for (grapheme, phoneme), (g2p, p2g) in sorted_pairs[:10]:
    stronger = "G→P" if g2p > p2g else "P→G"
    print(f"{grapheme} ↔ {phoneme}: {stronger} (scores: {g2p:.3f}, {p2g:.3f})")

# ===== 6. Visualize Results =====
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, (measure_name, scores) in enumerate(measures.items()):
    # Scale to [0, 1]
    scaled = scale_scorer(scores, method="minmax", nrange=(0, 1))

    # Convert to matrix (Grapheme→Phoneme direction)
    mat_xy, mat_yx, graphemes, phonemes = scorer2matrices(scaled)

    # Plot heatmap
    sns.heatmap(
        mat_xy,
        xticklabels=phonemes,
        yticklabels=graphemes,
        cmap='YlOrRd',
        ax=axes[idx],
        cbar_kws={'label': 'Score'}
    )
    axes[idx].set_title(f'{measure_name}: Grapheme→Phoneme', fontsize=12)
    axes[idx].set_xlabel('Phonemes')
    axes[idx].set_ylabel('Graphemes')

plt.tight_layout()
plt.savefig('alignment_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== 7. Export Results =====
import pandas as pd

results = []
for (g, p), (g2p, p2g) in scaled_tresoldi.items():
    results.append({
        'Grapheme': g,
        'Phoneme': p,
        'G_to_P': g2p,
        'P_to_G': p2g,
        'Asymmetry': abs(g2p - p2g),
        'Stronger_Direction': 'G→P' if g2p > p2g else 'P→G'
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Asymmetry', ascending=False)

# Save to CSV
results_df.to_csv('alignment_scores.csv', index=False)
print("\nResults exported to alignment_scores.csv")

# ===== 8. Statistical Summary =====
print("\nStatistical Summary:")
print("=" * 50)
print(f"Total unique pairs: {len(scaled_tresoldi)}")
print(f"Mean G→P score: {results_df['G_to_P'].mean():.3f}")
print(f"Mean P→G score: {results_df['P_to_G'].mean():.3f}")
print(f"Mean asymmetry: {results_df['Asymmetry'].mean():.3f}")
print(f"Max asymmetry: {results_df['Asymmetry'].max():.3f}")

# Pairs with strong asymmetry
strong_asymmetry = results_df[results_df['Asymmetry'] > 0.5]
print(f"\nPairs with strong asymmetry (>0.5): {len(strong_asymmetry)}")
print(strong_asymmetry.head(10))
```

---

## Package Information

**Version:** 0.4.0
**Python:** Requires 3.10+
**Core dependencies:** numpy, pandas, scipy, matplotlib, seaborn, tabulate, freqprob
**License:** MIT
**Repository:** https://github.com/tresoldi/asymcat

### Import Structure

```python
# Main namespace
import asymcat

# Data loading
from asymcat import (
    read_sequences,
    read_pa_matrix,
)

# Data processing
from asymcat import (
    collect_alphabets,
    collect_ngrams,
    collect_cooccs,
    collect_observations,
    build_ct,
)

# Scoring
from asymcat.scorer import (
    CatScorer,
    scorer2matrices,
    scale_scorer,
    invert_scorer,
)

# Correlation (for series)
from asymcat import correlation
from asymcat.correlation import (
    cramers_v,
    conditional_entropy,
    theil_u,
)
```

---

This documentation provides practical guidance for LLM agents to effectively use ASymCat in their projects. For the latest updates, mathematical foundations, and advanced examples, visit the [GitHub repository](https://github.com/tresoldi/asymcat).
