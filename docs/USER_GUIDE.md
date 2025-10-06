# ASymCat User Guide

A comprehensive guide to asymmetric categorical association analysis.

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Background](#mathematical-background)
3. [Association Measures Guide](#association-measures-guide)
4. [Data Preparation](#data-preparation)
5. [Smoothing Methods](#smoothing-methods)
6. [Visualization Techniques](#visualization-techniques)
7. [Best Practices](#best-practices)
8. [Common Use Cases](#common-use-cases)
9. [Interpretation Guidelines](#interpretation-guidelines)
10. [Advanced Topics](#advanced-topics)

## Introduction

### What is Asymmetric Categorical Association?

Asymmetric categorical association measures quantify **directional relationships** between categorical variables. Unlike traditional symmetric measures (Pearson's χ², Cramér's V, etc.) that treat X and Y interchangeably, asymmetric measures distinguish between:

- **P(Y|X)**: The probability of Y given X (how well X predicts Y)
- **P(X|Y)**: The probability of X given Y (how well Y predicts X)

These probabilities are often **not equal**, revealing directional dependencies that symmetric measures cannot capture.

### Why Asymmetric Measures Matter

Many real-world relationships are inherently directional:

**Medical Diagnosis:**
- P(fever|flu) ≈ 0.95 (most flu cases have fever)
- P(flu|fever) ≈ 0.10 (only 10% of fevers are flu)
- **Asymmetry reveals**: Fever is common in flu, but fever doesn't strongly indicate flu

**Linguistics:**
- P(/k/|"c") ≈ 0.70 (letter "c" often sounds like /k/)
- P("c"/k/) ≈ 0.40 (/k/ sound has many spellings: c, k, ck, q, etc.)
- **Asymmetry reveals**: Orthography better predicts phonetics than vice versa

**Ecology:**
- P(species_B|species_A) ≈ 0.80 (B usually found with A)
- P(species_A|species_B) ≈ 0.30 (A found without B often)
- **Asymmetry reveals**: Species B depends on A, but not vice versa

**Market Research:**
- P(product_Y|product_X) ≈ 0.60 (60% who buy X also buy Y)
- P(product_X|product_Y) ≈ 0.20 (20% who buy Y also buy X)
- **Asymmetry reveals**: X is a strong predictor for Y purchases (recommendation opportunity)

### When to Use ASymCat

**Use asymmetric measures when:**
- You care about prediction direction (X→Y vs Y→X)
- Analyzing causal or temporal relationships
- Feature engineering for machine learning
- Understanding dependencies and information flow
- Variables have inherent directionality

**Use symmetric measures when:**
- You only need overall association strength
- Clustering or similarity analysis
- Independence testing without directionality
- Variables are truly interchangeable

### Library Features

ASymCat provides:

- **17+ Association Measures**: Probabilistic, information-theoretic, statistical, and specialized
- **Directional Analysis**: Explicit X→Y vs Y→X quantification
- **Robust Smoothing**: Handle sparse data with FreqProb integration
- **Multiple Data Formats**: Sequences, presence-absence matrices, n-grams
- **Comprehensive Visualization**: Heatmaps, distributions, networks
- **Statistical Validation**: Bootstrap, permutation tests
- **Scalable Architecture**: Optimized for large datasets

### Installation

```bash
# Core installation
pip install asymcat

# With visualization support
pip install asymcat[viz]

# All features
pip install asymcat[all]
```

## Mathematical Background

### Conditional Probabilities

The foundation of asymmetric analysis is the **conditional probability**:

$$P(Y|X) = \frac{P(X, Y)}{P(X)} = \frac{c(X, Y)}{c(X)}$$

Where:
- $P(Y|X)$ = Probability of Y given X
- $P(X, Y)$ = Joint probability of X and Y
- $c(X, Y)$ = Count of co-occurrences
- $c(X)$ = Count of X

**Key Insight:** In general, $P(Y|X) \neq P(X|Y)$

**Example:**
- $c(\text{flu}, \text{fever}) = 95$ (95 flu patients have fever)
- $c(\text{flu}) = 100$ (100 flu patients total)
- $c(\text{fever}) = 1000$ (1000 fever patients total)

Therefore:
- $P(\text{fever}|\text{flu}) = 95/100 = 0.95$
- $P(\text{flu}|\text{fever}) = 95/1000 = 0.095$

**Asymmetry:** $0.95 \neq 0.095$ (huge difference!)

### Information Theory Basics

**Entropy** measures uncertainty:

$$H(X) = -\sum_{x \in X} P(x) \log P(x)$$

- Low entropy: X is predictable
- High entropy: X is random

**Mutual Information** measures shared information:

$$I(X; Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$

- I(X; Y) = 0: X and Y are independent
- I(X; Y) > 0: X and Y share information

**Conditional Entropy** measures remaining uncertainty:

$$H(Y|X) = -\sum_{x,y} P(x,y) \log P(y|x)$$

- H(Y|X) = 0: X completely determines Y
- H(Y|X) = H(Y): X tells us nothing about Y

**Key Asymmetric Property:**

$$H(Y|X) \neq H(X|Y)$$

Example:
- H(phonetics|orthography) might be low (spelling tells us sound)
- H(orthography|phonetics) might be high (sound doesn't tell us spelling)

### Statistical Independence vs Dependence

Two variables are **statistically independent** if:

$$P(X, Y) = P(X) \cdot P(Y)$$

Equivalently:
$$P(Y|X) = P(Y)$$
$$P(X|Y) = P(X)$$

**Departure from independence** can be asymmetric:

- $P(Y|X)$ might be very different from $P(Y)$
- But $P(X|Y)$ might be similar to $P(X)$

This reveals that **Y depends on X, but X doesn't depend on Y**.

### The Asymmetry Concept Mathematically

**Asymmetry Index** (simple difference):

$$A(X, Y) = |P(Y|X) - P(X|Y)|$$

- A = 0: Symmetric relationship
- A → 1: Highly asymmetric

**Theil's Uncertainty Coefficient** (information-theoretic):

$$U(Y|X) = \frac{H(Y) - H(Y|X)}{H(Y)} = \frac{I(X; Y)}{H(Y)}$$

- U(Y|X) = 0: X tells nothing about Y
- U(Y|X) = 1: X completely determines Y
- U(Y|X) ≠ U(X|Y) in general

**Tresoldi Measure** (ASymCat-specific):

$$T(X, Y) = \text{PMI}(X, Y)^{1 - P(Y|X)}$$

- Combines information content with conditional probability
- Designed for sequence alignment applications
- Penalizes low-probability predictions

## Association Measures Guide

ASymCat implements 17+ measures across four categories. This section explains each measure conceptually.

### Probabilistic Measures

These measures are based on probability theory and conditional frequencies.

#### Maximum Likelihood Estimation (MLE)

**Concept:** Direct conditional probability without smoothing.

**Formula:**

$$P_{MLE}(Y|X) = \frac{c(X, Y)}{c(X)}$$

**When to use:**
- Abundant data (no sparse co-occurrences)
- Direct probability interpretation needed
- Baseline for comparison

**Interpretation:**
- Range: [0, 1]
- 0 = Y never occurs with X
- 1 = Y always occurs with X
- Asymmetric by nature: P(Y|X) ≠ P(X|Y)

**Strengths:**
- Intuitive and interpretable
- No hyperparameters
- Direct relationship to data

**Weaknesses:**
- Zero-probability problem for unseen pairs
- Unreliable with sparse data
- No statistical significance testing

**Example Use Case:** Analyzing common word bigrams where all pairs are well-attested.

#### Jaccard Index

**Concept:** Set overlap with asymmetric interpretation.

**Formula:**

$$J(X, Y) = \frac{|X \cap Y|}{|X \cup Y|}$$

For categorical data:
$$J(Y|X) = \frac{c(X, Y)}{c(X) + c(Y) - c(X, Y)}$$

**When to use:**
- Binary or presence-absence data
- Set similarity tasks
- Ecological co-occurrence

**Interpretation:**
- Range: [0, 1]
- 0 = No overlap
- 1 = Perfect overlap
- Can be asymmetric when normalized differently

**Strengths:**
- Intuitive for set operations
- Standard in ecology
- Handles presence-absence naturally

**Weaknesses:**
- Less informative for frequency data
- Limited statistical properties
- Not purely probabilistic

**Example Use Case:** Species co-occurrence on islands (presence/absence data).

#### Goodman-Kruskal λ (Lambda)

**Concept:** Proportional reduction in prediction error.

**Formula:**

$$\lambda(Y|X) = \frac{\sum_x \max_y f(x,y) - \max_y f(y)}{n - \max_y f(y)}$$

Where:
- $f(x,y)$ = frequency of (x,y)
- $n$ = total count

**When to use:**
- Prediction tasks
- Classification performance
- Measuring predictive improvement

**Interpretation:**
- Range: [0, 1]
- 0 = X doesn't help predict Y
- 1 = X perfectly predicts Y
- Asymmetric: λ(Y|X) ≠ λ(X|Y)

**Strengths:**
- Clear predictive interpretation
- Accounts for modal categories
- Standard in social sciences

**Weaknesses:**
- Can be 0 even with association
- Sensitive to distribution shape
- Complex calculation

**Example Use Case:** Feature selection where you want to know how much knowing X improves Y prediction.

### Information-Theoretic Measures

These measures quantify information content and shared knowledge.

#### Pointwise Mutual Information (PMI)

**Concept:** Information gained about Y from observing X.

**Formula:**

$$\text{PMI}(X, Y) = \log \frac{P(X, Y)}{P(X) \cdot P(Y)}$$

**When to use:**
- Collocation detection
- Term association
- Information content analysis

**Interpretation:**
- Range: (-∞, +∞)
- 0 = Independence
- > 0 = Positive association
- < 0 = Negative association
- **Symmetric**: PMI(X, Y) = PMI(Y, X)

**Strengths:**
- Strong theoretical foundation
- Widely used in NLP
- Captures non-linear relationships

**Weaknesses:**
- Biased toward rare events
- Unbounded range
- Not a true probability

**Example Use Case:** Finding word collocations in text (e.g., "strong tea" vs "powerful tea").

#### Normalized PMI (NPMI)

**Concept:** PMI normalized to [-1, 1] range.

**Formula:**

$$\text{NPMI}(X, Y) = \frac{\text{PMI}(X, Y)}{-\log P(X, Y)}$$

**When to use:**
- When comparability across different frequency ranges is needed
- Ranking associations

**Interpretation:**
- Range: [-1, 1]
- -1 = Never co-occur
- 0 = Independence
- 1 = Always co-occur together

**Strengths:**
- Comparable across datasets
- Bounded range
- Less biased than raw PMI

**Weaknesses:**
- Still symmetric
- Undefined when P(X,Y) = 0
- Complex interpretation

**Example Use Case:** Comparing association strengths across different corpora or languages.

#### Mutual Information (MI)

**Concept:** Total information shared between X and Y.

**Formula:**

$$I(X; Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$

**When to use:**
- Feature selection
- Clustering
- Dependence detection

**Interpretation:**
- Range: [0, +∞)
- 0 = Independence
- Higher = More dependence
- **Symmetric**: I(X; Y) = I(Y; X)

**Strengths:**
- Robust theoretical foundation
- Detects non-linear dependencies
- Standard in ML

**Weaknesses:**
- Symmetric (no directionality)
- Sensitive to bin size
- Unbounded

**Example Use Case:** Feature selection in machine learning (which features provide most information about target?).

#### Conditional Entropy

**Concept:** Uncertainty about Y remaining after observing X.

**Formula:**

$$H(Y|X) = -\sum_{x,y} P(x,y) \log P(y|x)$$

**When to use:**
- Prediction uncertainty
- Information flow analysis
- Compression tasks

**Interpretation:**
- Range: [0, H(Y)]
- 0 = X completely determines Y
- H(Y) = X tells nothing about Y
- **Asymmetric**: H(Y|X) ≠ H(X|Y)

**Strengths:**
- Clear uncertainty interpretation
- Asymmetric directionality
- Fundamental to information theory

**Weaknesses:**
- Requires careful normalization
- Not intuitive for non-specialists
- Sensitive to distribution

**Example Use Case:** Analyzing how much uncertainty remains in phonetics after knowing orthography.

#### Theil's U (Uncertainty Coefficient)

**Concept:** Normalized uncertainty reduction.

**Formula:**

$$U(Y|X) = \frac{H(Y) - H(Y|X)}{H(Y)} = \frac{I(X; Y)}{H(Y)}$$

**When to use:**
- Normalized information analysis
- Feature importance ranking
- Comparing different variable pairs

**Interpretation:**
- Range: [0, 1]
- 0 = X provides no information about Y
- 1 = X completely determines Y
- **Asymmetric**: U(Y|X) ≠ U(X|Y)

**Strengths:**
- Normalized to [0, 1]
- Clear interpretation
- Asymmetric directionality
- Robust measure

**Weaknesses:**
- Requires sufficient data
- Can be computationally intensive
- Sensitive to rare categories

**Example Use Case:** Feature selection where you want normalized asymmetric importance scores.

### Statistical Measures

These measures provide statistical significance testing and hypothesis evaluation.

#### Chi-Square (χ²)

**Concept:** Test of independence using squared deviations.

**Formula:**

$$\chi^2 = \sum_{x,y} \frac{(O_{xy} - E_{xy})^2}{E_{xy}}$$

Where:
- $O_{xy}$ = Observed frequency
- $E_{xy}$ = Expected frequency under independence

**When to use:**
- Testing independence
- Categorical data analysis
- Large sample sizes

**Interpretation:**
- Range: [0, +∞)
- 0 = Perfect independence
- Higher = Stronger dependence
- **Symmetric**: χ²(X,Y) = χ²(Y,X)

**Strengths:**
- Well-established statistical test
- P-values available
- Standard in research

**Weaknesses:**
- Symmetric (no directionality)
- Sensitive to sample size
- Assumes large samples

**Example Use Case:** Testing whether two categorical variables are independent.

#### Cramér's V

**Concept:** Normalized chi-square association.

**Formula:**

$$V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1, c-1)}}$$

Where:
- n = sample size
- r, c = number of rows/columns

**When to use:**
- Comparing association across different table sizes
- Normalized effect size

**Interpretation:**
- Range: [0, 1]
- 0 = No association
- 1 = Perfect association
- **Symmetric**: V(X,Y) = V(Y,X)

**Strengths:**
- Normalized to [0, 1]
- Comparable across studies
- Standard effect size

**Weaknesses:**
- Symmetric
- Still sensitive to sample size
- No directionality

**Example Use Case:** Reporting effect sizes in research papers.

#### Fisher Exact Test

**Concept:** Exact test for 2×2 contingency tables.

**Formula:**

$$p = \frac{(a+b)!(c+d)!(a+c)!(b+d)!}{n!a!b!c!d!}$$

For 2×2 table:
```
     Y   ¬Y
X    a    b
¬X   c    d
```

**When to use:**
- Small sample sizes
- 2×2 tables
- Exact p-values needed

**Interpretation:**
- Returns odds ratio and p-value
- Odds ratio: association strength
- P-value: significance

**Strengths:**
- Exact (not approximate)
- Valid for small samples
- Gold standard for 2×2 tables

**Weaknesses:**
- Only for 2×2 tables
- Computationally intensive for large n
- Symmetric

**Example Use Case:** Rare species co-occurrence with few observations.

#### Log-Likelihood Ratio (G²)

**Concept:** Likelihood ratio test statistic.

**Formula:**

$$G^2 = 2\sum_{x,y} O_{xy} \log \frac{O_{xy}}{E_{xy}}$$

**When to use:**
- Alternative to chi-square
- Better for small expected frequencies
- Collocation detection

**Interpretation:**
- Range: [0, +∞)
- 0 = Independence
- Higher = Stronger association
- Follows χ² distribution

**Strengths:**
- Better for sparse data than χ²
- Asymptotically equivalent to χ²
- Standard in NLP

**Weaknesses:**
- Symmetric
- Still requires adequate sample size
- Less intuitive than χ²

**Example Use Case:** Collocation detection in corpora with sparse observations.

### Specialized Measures

#### Tresoldi Measure

**Concept:** PMI weighted by conditional probability.

**Formula:**

$$T(X, Y) = \text{PMI}(X, Y)^{1 - P(Y|X)}$$

**When to use:**
- Sequence alignment
- Historical linguistics
- When both information content and probability matter

**Interpretation:**
- Combines PMI (information) with MLE (probability)
- Penalizes low-probability associations
- **Asymmetric**: T(X,Y) ≠ T(Y,X)

**Strengths:**
- Designed for linguistic alignment
- Balances information and probability
- Asymmetric

**Weaknesses:**
- Less established theoretically
- Domain-specific design
- Complex interpretation

**Example Use Case:** Grapheme-to-phoneme alignment in historical linguistics.

## Data Preparation

### Sequence Data Format

Sequence data represents parallel aligned observations.

**TSV Format (Tab-Separated Values):**

```
grapheme	phoneme
b	B
c	K
c	S
gh	G
gh	F
```

**Requirements:**
- Tab-separated columns
- Header row with column names
- Each row is one co-occurrence
- Can have multiple columns for context

**Loading:**

```python
import asymcat

data = asymcat.read_sequences("data.tsv")
# Returns list of tuples: [('b', 'B'), ('c', 'K'), ...]
```

**With column selection:**

```python
data = asymcat.read_sequences("data.tsv", cols=["grapheme", "phoneme"])
```

**Important:** When you call `collect_cooccs()` on sequence data, it creates the **Cartesian product** of all elements in each sequence pair. This means every element in the first sequence is paired with every element in the second sequence.

If you need specific alignments (position-by-position), sub-windows, or other structures, preprocess your data before calling `collect_cooccs()`—ASymCat is data-agnostic and treats each sequence pair independently.

### Presence-Absence Matrices

Binary data indicating presence (1) or absence (0).

**TSV Format:**

```
site	species_A	species_B	species_C
island_1	1	0	1
island_2	1	1	0
island_3	0	1	1
```

**Requirements:**
- First column: observation ID (e.g., site name)
- Remaining columns: binary (0/1) or boolean
- Header row required

**Loading:**

```python
data = asymcat.read_pa_matrix("islands.tsv")
# Returns list of species pairs present together
```

### N-grams and Padding

N-grams are sequences of N consecutive elements.

**Example: Bigrams (N=2)**

Sequence: `['a', 'b', 'c', 'd']`

Without padding: `[('a', 'b'), ('b', 'c'), ('c', 'd')]`

With padding: `[('#', 'a'), ('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', '#')]`

**Creating n-grams:**

```python
# Bigrams with padding
cooccs = asymcat.collect_cooccs(data, order=2, pad='#')

# Trigrams without padding
cooccs = asymcat.collect_cooccs(data, order=3)
```

**Padding strategies:**
- `pad=None`: No padding (default)
- `pad='#'`: Add boundary markers
- `pad='<s>'`: Start/end tokens

**When to use padding:**
- Sentence/word boundaries matter
- Position-dependent analysis
- Linguistic applications

### Data Quality Considerations

**Sample Size:**
- Minimum: 30-50 observations per category
- Ideal: 100+ observations
- Sparse data: Use smoothing

**Balance:**
- Imbalanced categories okay for asymmetric analysis
- But extreme imbalance (99:1) may need special handling

**Missing Data:**
- Remove incomplete observations
- Or use presence-absence encoding

**Outliers:**
- Rare co-occurrences may dominate some measures (esp. PMI)
- Consider filtering by minimum frequency

**Noise:**
- Annotation errors affect conditional probabilities
- Validate data quality before analysis

## Smoothing Methods

### Why Smoothing Matters

**The Zero-Probability Problem:**

With raw MLE, unseen co-occurrences get P = 0.

**Consequences:**
- Can't compute log probabilities
- Product-based measures fail
- No generalization to new data

**Solution:** Smoothing redistributes probability mass to unseen events.

### MLE (No Smoothing)

**Method:** Direct conditional frequency.

$$P(Y|X) = \frac{c(X,Y)}{c(X)}$$

**When to use:**
- Abundant data (all pairs well-attested)
- No unseen events expected
- Baseline comparison

**Parameters:** None

**Example:**

```python
scorer = asymcat.CatScorer(cooccs, smoothing_method='mle')
```

### Laplace Smoothing (Add-One)

**Method:** Add pseudo-count of 1 to all observations.

$$P(Y|X) = \frac{c(X,Y) + 1}{c(X) + V}$$

Where V = vocabulary size (number of possible Y values).

**When to use:**
- Sparse data with many zero counts
- Need non-zero probabilities everywhere
- General-purpose smoothing

**Parameters:**
- Implicit α = 1

**Example:**

```python
scorer = asymcat.CatScorer(cooccs, smoothing_method='laplace')
```

**Effects:**
- Reduces confidence in rare events
- Increases probability of unseen events
- Can over-smooth with small V

### Lidstone Smoothing (Add-Gamma)

**Method:** Add pseudo-count of γ (gamma).

$$P(Y|X) = \frac{c(X,Y) + \gamma}{c(X) + \gamma \cdot V}$$

**When to use:**
- Fine-tune smoothing strength
- Between MLE and Laplace
- Domain-specific tuning

**Parameters:**
- `smoothing_alpha` = γ (typically 0.1 to 1.0)

**Example:**

```python
scorer = asymcat.CatScorer(cooccs, smoothing_method='lidstone', smoothing_alpha=0.5)
```

**Choosing γ:**
- γ = 0: MLE (no smoothing)
- γ = 0.1-0.5: Light smoothing
- γ = 1.0: Laplace smoothing
- γ > 1.0: Heavy smoothing

### Impact on Rare Events

**Example dataset:**
- Common pair: c(X,Y) = 100
- Rare pair: c(X,Y) = 1
- Unseen pair: c(X,Y) = 0

**MLE:**
- Common: P = 100/150 = 0.667
- Rare: P = 1/150 = 0.007
- Unseen: P = 0/150 = 0.000

**Laplace (V=10):**
- Common: P = 101/160 = 0.631
- Rare: P = 2/160 = 0.013
- Unseen: P = 1/160 = 0.006

**Lidstone (γ=0.1, V=10):**
- Common: P = 100.1/151 = 0.663
- Rare: P = 1.1/151 = 0.007
- Unseen: P = 0.1/151 = 0.001

**Observations:**
- Smoothing reduces common pair probability slightly
- Smoothing increases rare pair probability noticeably
- Smoothing gives non-zero to unseen pairs

## Visualization Techniques

### Heatmap Strategies

**Basic Heatmap:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Get score matrices
xy_matrix, yx_matrix, x_labels, y_labels = asymcat.scorer.scorer2matrices(scores)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(xy_matrix, annot=True, fmt='.2f',
            xticklabels=y_labels, yticklabels=x_labels,
            cmap='RdYlGn', center=0)
plt.title('X → Y Associations')
plt.show()
```

**Diverging Colors:**
- Use `cmap='RdYlGn'` or `'RdBu'` for signed scores (PMI)
- Use `cmap='viridis'` or `'YlOrRd'` for positive scores (MLE, χ²)

**Annotations:**
- `annot=True`: Show values in cells
- `fmt='.2f'`: Two decimal places
- `linewidths=0.5`: Cell borders

**Dual Heatmaps (X→Y vs Y→X):**

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

sns.heatmap(xy_matrix, ax=ax1, annot=True, cmap='YlOrRd')
ax1.set_title('X → Y')

sns.heatmap(yx_matrix, ax=ax2, annot=True, cmap='YlOrRd')
ax2.set_title('Y → X')

plt.tight_layout()
plt.show()
```

### Score Distributions

**Histogram:**

```python
all_scores = [s for pair in scores.values() for s in pair]

plt.hist(all_scores, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title('Distribution of Association Scores')
plt.show()
```

**KDE Plot (Kernel Density Estimation):**

```python
import seaborn as sns

xy_scores = [scores[pair][0] for pair in scores]
yx_scores = [scores[pair][1] for pair in scores]

sns.kdeplot(xy_scores, label='X → Y', shade=True)
sns.kdeplot(yx_scores, label='Y → X', shade=True)
plt.xlabel('Score')
plt.ylabel('Density')
plt.legend()
plt.show()
```

**Box Plots:**

```python
import pandas as pd

df = pd.DataFrame({
    'X→Y': xy_scores,
    'Y→X': yx_scores
})

df.boxplot()
plt.ylabel('Score')
plt.title('Score Distribution Comparison')
plt.show()
```

### Publication-Quality Figures

**Styling:**

```python
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-paper')  # Clean style for papers
plt.rcParams['figure.dpi'] = 300     # High resolution
plt.rcParams['font.size'] = 10       # Readable text
plt.rcParams['font.family'] = 'serif'  # Professional font
```

**Export:**

```python
plt.savefig('figure1.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figure1.png', bbox_inches='tight', dpi=300)
```

**Multi-panel Layout:**

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1
axes[0, 0].bar(...)
axes[0, 0].set_title('(A) MLE Scores')

# Plot 2
axes[0, 1].bar(...)
axes[0, 1].set_title('(B) PMI Scores')

# Plot 3
axes[1, 0].scatter(...)
axes[1, 0].set_title('(C) MLE vs PMI')

# Plot 4
axes[1, 1].hist(...)
axes[1, 1].set_title('(D) Score Distribution')

plt.tight_layout()
plt.savefig('figure_composite.pdf', dpi=300)
```

## Best Practices

### Choosing the Right Measure

**Decision Guide:**

1. **What is your goal?**
   - Prediction → Use MLE, Theil U, λ
   - Information content → Use PMI, MI
   - Independence testing → Use χ², Fisher
   - General association → Use multiple measures

2. **What is your data like?**
   - Abundant → MLE, χ²
   - Sparse → Use smoothing, Fisher
   - Binary → Jaccard, Fisher
   - Multi-category → MLE, Theil U, χ²

3. **Do you need directionality?**
   - Yes → MLE, Theil U, Conditional Entropy, λ, Tresoldi
   - No → PMI, χ², Cramér's V, MI

4. **What range do you prefer?**
   - [0, 1] → MLE, NPMI, Theil U, Cramér's V, λ
   - Unbounded → PMI, χ², G²

**Recommendation:** Start with MLE for interpretability, add Theil U for normalized asymmetry, then explore others as needed.

### Handling Sparse Data

**Symptoms:**
- Many zero co-occurrence counts
- MLE produces many 0.0 and 1.0 scores
- High variance in rare categories

**Solutions:**

1. **Use smoothing:**
   ```python
   scorer = CatScorer(cooccs, smoothing_method='laplace')
   ```

2. **Filter rare events:**
   ```python
   # Remove pairs with count < threshold
   filtered_cooccs = [(x,y) for (x,y) in cooccs if cooccs.count((x,y)) >= 5]
   ```

3. **Aggregate categories:**
   - Combine similar categories
   - Use broader bins

4. **Use exact tests:**
   ```python
   # Fisher's exact test handles small samples better
   fisher_scores = scorer.fisher()
   ```

5. **Bootstrap validation:**
   - Estimate confidence intervals
   - Check stability of results

### Statistical Validation

**Bootstrap Confidence Intervals:**

```python
import numpy as np

def bootstrap_scores(data, n_iterations=1000):
    results = []
    for _ in range(n_iterations):
        # Resample with replacement
        sample = np.random.choice(len(data), size=len(data), replace=True)
        sample_data = [data[i] for i in sample]

        # Compute score
        cooccs = asymcat.collect_cooccs(sample_data)
        scorer = asymcat.CatScorer(cooccs)
        score = scorer.mle()
        results.append(score)

    return results

# Get 95% confidence interval
bootstrap_results = bootstrap_scores(data, n_iterations=1000)
ci_lower = np.percentile(bootstrap_results, 2.5)
ci_upper = np.percentile(bootstrap_results, 97.5)
```

**Permutation Tests:**

```python
def permutation_test(data_x, data_y, n_permutations=1000):
    # Observed score
    observed_cooccs = asymcat.collect_cooccs(list(zip(data_x, data_y)))
    scorer = asymcat.CatScorer(observed_cooccs)
    observed_score = scorer.theil_u()

    # Permuted scores
    permuted_scores = []
    for _ in range(n_permutations):
        shuffled_y = np.random.permutation(data_y)
        perm_cooccs = asymcat.collect_cooccs(list(zip(data_x, shuffled_y)))
        perm_scorer = asymcat.CatScorer(perm_cooccs)
        permuted_scores.append(perm_scorer.theil_u())

    # P-value
    p_value = np.mean([s >= observed_score for s in permuted_scores])
    return p_value
```

**Multiple Testing Correction:**

When testing many pairs, use Bonferroni or FDR correction:

```python
from scipy.stats import false_discovery_control

# If you have p-values for many tests
p_values = [test_pair(x, y) for (x, y) in pairs]

# FDR correction
significant = false_discovery_control(p_values, alpha=0.05)
```

### Performance Optimization

**For large datasets:**

1. **Precompute co-occurrences once:**
   ```python
   cooccs = asymcat.collect_cooccs(data)  # Do once
   scorer = asymcat.CatScorer(cooccs)     # Reuse scorer
   ```

2. **Compute multiple measures at once:**
   ```python
   mle = scorer.mle()
   pmi = scorer.pmi()
   theil = scorer.theil_u()
   # All share same cached computations
   ```

3. **Use appropriate smoothing:**
   - MLE is fastest (no extra computation)
   - Smoothing adds overhead but prevents zeros

4. **Filter before computing:**
   ```python
   # Remove rare pairs first
   filtered = {pair: count for pair, count in cooccs.items() if count >= 5}
   ```

### Reproducibility Guidelines

**Always document:**
1. ASymCat version: `asymcat.__version__`
2. Data source and preprocessing steps
3. Smoothing method and parameters
4. Association measures used
5. Significance thresholds
6. Random seed for bootstrap/permutation

**Example reproducibility block:**

```python
import asymcat
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

print(f"ASymCat version: {asymcat.__version__}")
print(f"Data: CMU dictionary sample (N=1000)")
print(f"Smoothing: Laplace (alpha=1.0)")
print(f"Measures: MLE, PMI, Theil U")
print(f"Significance: α=0.05 (Bonferroni corrected)")
```

## Common Use Cases

### Linguistics: Grapheme-Phoneme Correspondence

**Question:** How predictable are English sound-spelling correspondences?

**Data:** Parallel grapheme-phoneme alignments from CMU dictionary.

**Workflow:**

```python
import asymcat

# Load aligned grapheme-phoneme data
data = asymcat.read_sequences("resources/cmudict.sample1000.tsv")

# Collect co-occurrences
cooccs = asymcat.collect_cooccs(data)

# Create scorer with smoothing (many rare correspondences)
scorer = asymcat.CatScorer(cooccs, smoothing_method='lidstone', smoothing_alpha=0.1)

# Compute directional measures
mle = scorer.mle()
theil = scorer.theil_u()

# Find most asymmetric patterns
for (grapheme, phoneme), (g2p, p2g) in mle.items():
    asymmetry = abs(g2p - p2g)
    if asymmetry > 0.5:
        print(f"{grapheme} → {phoneme}: {g2p:.3f} vs {p2g:.3f} (Δ={asymmetry:.3f})")
```

**Expected Finding:** Grapheme→phoneme often higher than phoneme→grapheme (spelling more predictable from sound).

### Ecology: Species Co-occurrence

**Question:** Do species exhibit directional dependencies?

**Data:** Presence-absence matrix of species across islands.

**Workflow:**

```python
# Load island biogeography data
data = asymcat.read_pa_matrix("resources/galapagos.tsv")

# Collect co-occurrences
cooccs = asymcat.collect_cooccs(data)

# Create scorer
scorer = asymcat.CatScorer(cooccs, smoothing_method='mle')

# Use Jaccard and Fisher (appropriate for binary data)
jaccard = scorer.jaccard_index()
fisher = scorer.fisher()

# Find asymmetric co-occurrence
for (sp_a, sp_b), (a2b, b2a) in jaccard.items():
    if a2b > 0.7 and b2a < 0.3:
        print(f"{sp_a} predicts {sp_b} strongly (J={a2b:.3f})")
        print(f"But {sp_b} doesn't predict {sp_a} (J={b2a:.3f})")
```

**Expected Finding:** Some species only occur with others (dependent), while some are generalists (independent).

### Market Research: Product Recommendations

**Question:** Which products predict future purchases?

**Data:** Customer purchase sequences.

**Workflow:**

```python
# Load transaction data
# Format: [(product_bought_first, product_bought_next), ...]
transactions = asymcat.read_sequences("transactions.tsv")

# Collect co-occurrences
cooccs = asymcat.collect_cooccs(transactions)

# Create scorer
scorer = asymcat.CatScorer(cooccs, smoothing_method='laplace')

# Use MLE for direct prediction strength
mle = scorer.mle()
theil = scorer.theil_u()

# Find strong predictors (X → Y high, Y → X low)
recommendations = []
for (prod_x, prod_y), (x2y, y2x) in mle.items():
    if x2y > 0.6 and y2x < 0.2:
        recommendations.append((prod_x, prod_y, x2y))

# Sort by prediction strength
recommendations.sort(key=lambda x: x[2], reverse=True)

print("Top recommendations:")
for prod_x, prod_y, score in recommendations[:10]:
    print(f"Customers who buy {prod_x} often buy {prod_y} next (P={score:.2f})")
```

**Expected Finding:** Complementary products (chips→salsa) show high asymmetry, substitutes (Coke→Pepsi) show low asymmetry.

### Machine Learning: Feature Selection

**Question:** Which features best predict the target variable?

**Data:** Categorical features and target labels.

**Workflow:**

```python
# Load classification data (e.g., mushroom edibility)
data = asymcat.read_sequences("resources/mushrooms.tsv")

# For each feature, compute association with target
feature_importance = {}

for feature in ['cap_shape', 'cap_color', 'odor', 'gill_size']:
    # Extract feature-target pairs
    pairs = [(row[feature], row['edibility']) for row in data]

    # Compute Theil U (normalized information)
    cooccs = asymcat.collect_cooccs(pairs)
    scorer = asymcat.CatScorer(cooccs)
    theil = scorer.theil_u()

    # U(target|feature) tells us how much feature predicts target
    avg_score = np.mean([s[1] for s in theil.values()])
    feature_importance[feature] = avg_score

# Rank features
ranked = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

print("Feature importance for predicting edibility:")
for feature, score in ranked:
    print(f"{feature}: U={score:.3f}")
```

**Expected Finding:** Some features (e.g., odor) strongly predict edibility, others are weak.

## Interpretation Guidelines

### Reading Asymmetric Scores

**High Asymmetry Example:**

```
P(fever|flu) = 0.95
P(flu|fever) = 0.10
```

**Interpretation:**
- Flu strongly predicts fever (directional)
- Fever weakly predicts flu (not specific)
- **Actionable insight**: Fever is sensitive but not specific to flu

**Low Asymmetry Example:**

```
P(Y|X) = 0.60
P(X|Y) = 0.55
```

**Interpretation:**
- Roughly symmetric relationship
- Both variables moderately predict each other
- **Actionable insight**: Bidirectional association

### Significance Thresholds

**No universal threshold**, but guidelines:

**MLE (probabilities):**
- < 0.1: Weak association
- 0.1-0.3: Moderate association
- 0.3-0.7: Strong association
- > 0.7: Very strong association

**Theil U:**
- < 0.1: Weak information gain
- 0.1-0.3: Moderate information gain
- 0.3-0.7: Strong information gain
- > 0.7: Very strong information gain

**PMI:**
- < 0: Negative association
- 0: Independence
- 0-3: Weak to moderate association
- > 3: Strong association

**Statistical measures (χ², G²):**
- Use p-values (typically p < 0.05)
- Effect size (Cramér's V) for practical significance

### When Asymmetry Matters

**High Asymmetry is informative when:**
- Planning interventions (which direction to influence)
- Building predictive models (which features to use)
- Understanding causality (which variable is upstream)
- Resource allocation (where to invest)

**Low Asymmetry suggests:**
- Symmetric relationship (mutual dependence)
- Spurious correlation (both caused by third variable)
- Balanced interaction

### Common Pitfalls

**1. Confusing correlation with causation**
- Asymmetry ≠ causation (but suggests directionality)
- Always consider confounds

**2. Ignoring sample size**
- Small samples → unreliable estimates
- Use Fisher exact test for small samples

**3. Cherry-picking measures**
- Different measures can give different rankings
- Report multiple measures
- Understand what each measures

**4. Ignoring data quality**
- Measurement error reduces all associations
- Validate data first

**5. Over-interpreting weak associations**
- Statistical significance ≠ practical significance
- Consider effect sizes

### Reporting Results

**Minimal reporting:**
- Measure used
- Smoothing method (if applicable)
- Sample size
- Top results with scores

**Complete reporting:**
- All of the above, plus:
- Confidence intervals
- Multiple measures for robustness
- Full distribution (not just top hits)
- Negative results (what didn't associate)

**Example:**

> We analyzed 1,000 grapheme-phoneme pairs from the CMU dictionary using ASymCat v0.4.0.
> We computed Maximum Likelihood Estimation with Lidstone smoothing (α=0.1) and Theil's U.
> The letter "c" showed strong asymmetry: P(/k/|c)=0.72 but P(c|/k/)=0.38, Theil U(phoneme|grapheme)=0.45.
> This indicates orthography predicts phonetics better than vice versa (95% CI: [0.40, 0.50], bootstrap n=1000).

## Advanced Topics

### Measure Correlations

Different measures often correlate but capture different aspects:

```python
import numpy as np
from scipy.stats import pearsonr

# Compute multiple measures
mle = scorer.mle()
pmi = scorer.pmi()
theil = scorer.theil_u()

# Extract scores for common pairs
pairs = list(mle.keys())
mle_scores = [mle[p][0] for p in pairs]
pmi_scores = [pmi[p][0] for p in pairs]
theil_scores = [theil[p][0] for p in pairs]

# Compute correlations
r_mle_pmi, _ = pearsonr(mle_scores, pmi_scores)
r_mle_theil, _ = pearsonr(mle_scores, theil_scores)
r_pmi_theil, _ = pearsonr(pmi_scores, theil_scores)

print(f"MLE-PMI correlation: {r_mle_pmi:.3f}")
print(f"MLE-Theil correlation: {r_mle_theil:.3f}")
print(f"PMI-Theil correlation: {r_pmi_theil:.3f}")
```

**Typical findings:**
- MLE and Theil U often correlate moderately (both probability-based)
- PMI and Theil U correlate less (different foundations)
- All capture different information (use multiple measures)

### Measure Combinations

**Ensemble scoring:**

```python
from scipy.stats import rankdata

# Get rankings from each measure
mle_ranks = rankdata([mle[p][0] for p in pairs])
theil_ranks = rankdata([theil[p][0] for p in pairs])
pmi_ranks = rankdata([pmi[p][0] for p in pairs])

# Average ranks
combined_ranks = (mle_ranks + theil_ranks + pmi_ranks) / 3

# Find top-ranked pairs
top_indices = np.argsort(combined_ranks)[-10:]
top_pairs = [pairs[i] for i in top_indices]
```

**Consensus scoring:**

Only report associations strong in multiple measures:

```python
strong_in_all = []
for pair in pairs:
    if (mle[pair][0] > 0.5 and
        theil[pair][0] > 0.5 and
        pmi[pair][0] > 3.0):
        strong_in_all.append(pair)
```

### Temporal Analysis

For time-stamped data, analyze how associations change:

```python
# Split data by time period
early_data = [(x, y) for x, y, t in data if t < cutoff]
late_data = [(x, y) for x, y, t in data if t >= cutoff]

# Compute scores for each period
early_scorer = asymcat.CatScorer(asymcat.collect_cooccs(early_data))
late_scorer = asymcat.CatScorer(asymcat.collect_cooccs(late_data))

early_mle = early_scorer.mle()
late_mle = late_scorer.mle()

# Find pairs with changing associations
for pair in set(early_mle.keys()) & set(late_mle.keys()):
    early_score = early_mle[pair][0]
    late_score = late_mle[pair][0]
    change = late_score - early_score

    if abs(change) > 0.3:
        print(f"{pair}: {early_score:.2f} → {late_score:.2f} (Δ={change:.2f})")
```

### Multi-way Associations

Beyond pairwise, analyze three-way associations:

```python
# Collect three-way co-occurrences
triplets = [(x, y, z) for x, y, z in data]

# Analyze conditional associations
# E.g., P(Z|X,Y) vs P(Z|X) vs P(Z|Y)

# This requires custom analysis beyond built-in CatScorer
# But the concepts extend naturally
```

---

**End of User Guide**

For practical examples with code, see the tutorials:
- tutorial_1_basics.py - Core workflow
- tutorial_2_advanced_measures.py - All measures
- tutorial_3_visualization.py - Plotting techniques
- tutorial_4_real_world.py - Complete analyses

For API reference, see API_REFERENCE.md.
For LLM integration, see LLM_DOCUMENTATION.md.
