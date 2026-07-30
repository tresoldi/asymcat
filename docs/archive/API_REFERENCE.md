# ASymCat API Reference

Complete API documentation for all ASymCat classes and functions.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Data Loading Functions](#data-loading-functions)
3. [Data Processing Functions](#data-processing-functions)
4. [Association Measures](#association-measures)
   - [Probabilistic Measures](#probabilistic-measures)
   - [Information-Theoretic Measures](#information-theoretic-measures)
   - [Statistical Measures](#statistical-measures)
   - [Specialized Measures](#specialized-measures)
5. [Utility Functions](#utility-functions)
6. [Correlation Functions](#correlation-functions)
7. [Type Definitions](#type-definitions)

---

## Core Classes

### CatScorer

Main class for computing categorical co-occurrence scores using multiple association measures.

```python
class CatScorer
```

**Constructor:**
```python
CatScorer(cooccs: list[tuple[Any, Any]],
          smoothing_method: str = "mle",
          smoothing_alpha: float = 1.0)
```

**Parameters:**
- `cooccs` - List of co-occurrence tuples representing observed pairs
- `smoothing_method` - Smoothing method for probability estimation. Options: 'mle', 'laplace', 'lidstone' (default: 'mle')
- `smoothing_alpha` - Smoothing parameter alpha for Laplace/Lidstone methods (default: 1.0)

**Properties:**
- `cooccs: list[tuple[Any, Any]]` - Original co-occurrence data
- `obs: dict[tuple[Any, Any], dict[str, int]]` - Dictionary of observations per co-occurrence type
- `alphabet_x: list[Any]` - Sorted list of unique symbols in first position
- `alphabet_y: list[Any]` - Sorted list of unique symbols in second position
- `smoothing_method: str` - Active smoothing method name
- `smoothing_alpha: float` - Smoothing parameter value

**Example:**
```python
cooccs = [('a', 'x'), ('a', 'y'), ('b', 'x'), ('b', 'y')]
scorer = asymcat.scorer.CatScorer(cooccs)
mle_scores = scorer.mle()
print(mle_scores[('a', 'x')])  # (P(x|a), P(a|x))
```

**Notes:**
- All scoring methods return dictionaries mapping pairs to tuples of (score_xy, score_yx)
- The first element of the tuple represents the score in the X→Y direction
- The second element represents the score in the Y→X direction
- Scores are computed lazily and cached for efficiency

---

## Data Loading Functions

### read_sequences

Reads parallel sequences from a tab-delimited file.

```python
def read_sequences(filename: str,
                   cols: list[str] | None = None,
                   col_delim: str = "\t",
                   elem_delim: str = " ") -> list[list[list[str]]]
```

**Parameters:**
- `filename` - Path to the file to be read
- `cols` - List of column names to be collected (default: None, uses first two columns)
- `col_delim` - String used as field delimiter (default: "\t")
- `elem_delim` - String used as element delimiter within fields (default: " ")

**Returns:**
- List of sequence pairs, where each pair contains two lists of elements

**Example:**
```python
seqs = asymcat.read_sequences("data/parallel_sequences.tsv")
# seqs = [[['E', 'X', 'C'], ['ɛ', 'k', 's']], ...]
```

**File Format:**
```
Orthography                 Segments
E X C L A M A T I O N       ɛ k s k l ʌ m eɪ ʃ ʌ n
C L O S E Q U O T E         k l oʊ z k w oʊ t
```

**Raises:**
- `FileNotFoundError` - If the specified file does not exist
- `ValueError` - If file is empty or has invalid format
- `PermissionError` - If file cannot be read due to permissions

---

### read_pa_matrix

Reads a presence-absence matrix and returns observed combinations.

```python
def read_pa_matrix(filename: str,
                   delimiter: str = "\t") -> list[tuple]
```

**Parameters:**
- `filename` - Path to the file to be read
- `delimiter` - String used as field delimiter (default: "\t")

**Returns:**
- List of tuples representing observed combinations from each location

**Example:**
```python
combinations = asymcat.read_pa_matrix("data/presence_absence.tsv")
# combinations = [('taxon1', 'taxon2'), ('taxon1', 'taxon3'), ...]
```

**File Format:**
```
ID          TaxonA  TaxonB  TaxonC
Location1   1       1       0
Location2   1       0       1
```

**Notes:**
- The file must contain an 'ID' column for location identifiers
- Values should be 0 (absent), 1 (present), or empty
- Returns all pairwise combinations of taxa present at each location

**Raises:**
- `FileNotFoundError` - If the specified file does not exist
- `ValueError` - If file format is invalid or ID column is missing

---

## Data Processing Functions

### collect_alphabets

Extract sorted alphabets from a list of co-occurrences.

```python
def collect_alphabets(cooccs: list[tuple]) -> tuple
```

**Parameters:**
- `cooccs` - List of co-occurrence tuples

**Returns:**
- Tuple of (alphabet_x, alphabet_y), each a sorted list of unique symbols

**Example:**
```python
cooccs = [('a', 'x'), ('b', 'y'), ('a', 'y')]
alphabet_x, alphabet_y = asymcat.collect_alphabets(cooccs)
# alphabet_x = ['a', 'b'], alphabet_y = ['x', 'y']
```

**Raises:**
- `ValueError` - If cooccs is empty or contains invalid tuples
- `TypeError` - If cooccs is not a list or contains non-tuple elements

---

### collect_ngrams

Generate n-grams from a sequence with padding.

```python
def collect_ngrams(seq: list[Any] | str,
                   order: int,
                   pad: str) -> Generator[tuple, None, None]
```

**Parameters:**
- `seq` - List or string of elements for n-gram collection
- `order` - N-gram order (must be ≥ 1)
- `pad` - Padding symbol to use at sequence boundaries

**Yields:**
- N-gram tuples of the specified order

**Example:**
```python
seq = ['a', 'b', 'c']
ngrams = list(asymcat.collect_ngrams(seq, order=2, pad='#'))
# ngrams = [('#', 'a'), ('a', 'b'), ('b', 'c'), ('c', '#')]
```

**Notes:**
- Sequences are padded to ensure boundary symbols appear with same frequency as internal symbols
- Padding guarantees sequences shorter than order are still collected

**Raises:**
- `ValueError` - If order is less than 1 or seq is empty
- `TypeError` - If seq is not a list, tuple, or string

---

### collect_cooccs

Collect co-occurring elements from pairs of sequences.

```python
def collect_cooccs(seqs: list[list[list[Any] | str] | tuple],
                   order: int | None = None,
                   pad: str = "#") -> list[tuple]
```

**Parameters:**
- `seqs` - List of sequence pairs (each sequence can be a list or string)
- `order` - N-gram order for collection, None for entire sequences (default: None)
- `pad` - Padding symbol for n-gram collection (default: "#")

**Returns:**
- List of tuples representing all co-occurring element pairs

**Example:**
```python
seqs = [[['a', 'b'], ['x', 'y']], [['c', 'd'], ['z', 'w']]]
# Full sequence co-occurrences
cooccs = asymcat.collect_cooccs(seqs)
# cooccs = [('a', 'x'), ('a', 'y'), ('b', 'x'), ('b', 'y'), ...]

# Bigram co-occurrences (order=2)
cooccs = asymcat.collect_cooccs(seqs, order=2, pad='#')
```

**Notes:**
- Without order: collects Cartesian product of each sequence pair (this is intentional—if you need position-aligned pairs, sub-windows, or other structures, preprocess your data first)
- With order: collects aligned n-gram pairs from sequences of equal length
- Padding symbol must not conflict with symbols in the data
- Pairs containing the padding symbol are automatically removed
- ASymCat is data-agnostic and treats each sequence pair independently

**Raises:**
- `ValueError` - If seqs is empty or sequences have mismatched lengths (when order is specified)
- `TypeError` - If seqs is not a list or contains invalid data types

---

### collect_observations

Build a dictionary of observations for contingency table construction.

```python
def collect_observations(cooccs: list[tuple]) -> dict
```

**Parameters:**
- `cooccs` - List of co-occurrence tuples

**Returns:**
- Dictionary mapping each (x, y) pair to observation counts

**Observation Codes:**
- `"00"`: Total number of co-occurrences
- `"10"`: Count where first element equals x (with any second element)
- `"20"`: Count where first element does not equal x
- `"01"`: Count where second element equals y (with any first element)
- `"02"`: Count where second element does not equal y
- `"11"`: Count where both elements match (x, y)
- `"12"`: Count where first matches x, second does not match y
- `"21"`: Count where first does not match x, second matches y
- `"22"`: Count where neither element matches

**Example:**
```python
cooccs = [('a', 'x'), ('a', 'y'), ('b', 'x')]
obs = asymcat.collect_observations(cooccs)
print(obs[('a', 'x')])
# {'00': 3, '11': 1, '10': 2, '20': 1, '01': 2, '02': 1, '12': 1, '21': 1, '22': 0}
```

**Contingency Table Structure:**
```
               |  y==target  |  y!=target  |
x==target      |  obs["11"]  |  obs["12"]  |
x!=target      |  obs["21"]  |  obs["22"]  |
```

**Raises:**
- `ValueError` - If cooccs is empty or contains invalid tuples
- `TypeError` - If cooccs is not a list

---

### build_ct

Build a contingency table from observations.

```python
def build_ct(observ: dict,
             square: bool = True) -> list
```

**Parameters:**
- `observ` - Dictionary of observations from collect_observations()
- `square` - Whether to return square (2x2) or non-square (3x2) table (default: True)

**Returns:**
- Contingency table as a list of lists (2x2 or 3x2)

**Example:**
```python
obs = observ[('a', 'x')]
ct_square = asymcat.build_ct(obs, square=True)
# [[obs["11"], obs["12"]], [obs["21"], obs["22"]]]

ct_nonsquare = asymcat.build_ct(obs, square=False)
# [[obs["10"], obs["11"], obs["12"]], [obs["20"], obs["21"], obs["22"]]]
```

**Notes:**
- Square tables (2x2) are used for most statistical tests
- Non-square tables (3x2) include marginal totals and are used for some measures

---

## Association Measures

All association measure methods return a dictionary mapping symbol pairs to tuples of (score_xy, score_yx), where:
- `score_xy` represents the association strength from X to Y
- `score_yx` represents the association strength from Y to X

### Probabilistic Measures

#### mle

Maximum Likelihood Estimation of conditional probabilities.

```python
def mle(self) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Returns:**
- Dictionary mapping pairs to (P(y|x), P(x|y)) probabilities

**Mathematical Formula:**

$$P(y|x) = \frac{count(x, y)}{count(x)}$$

**Example:**
```python
scorer = asymcat.scorer.CatScorer(cooccs, smoothing_method='mle')
mle_scores = scorer.mle()
xy_prob, yx_prob = mle_scores[('a', 'x')]
```

**Notes:**
- Uses freqprob library for robust probability estimation
- Supports smoothing via smoothing_method parameter
- Returns conditional probabilities in both directions

---

#### get_smoothed_probabilities

Get comprehensive smoothed probability estimates.

```python
def get_smoothed_probabilities(self) -> dict[str, dict[tuple[Any, Any], float]]
```

**Returns:**
- Dictionary with keys:
  - `'xy_given_y'`: P(X|Y) conditional probabilities
  - `'yx_given_x'`: P(Y|X) conditional probabilities
  - `'joint'`: P(X,Y) joint probabilities
  - `'marginal_x'`: P(X) marginal probabilities
  - `'marginal_y'`: P(Y) marginal probabilities

**Example:**
```python
scorer = asymcat.scorer.CatScorer(cooccs, smoothing_method='laplace')
probs = scorer.get_smoothed_probabilities()
p_xy = probs['joint'][('a', 'x')]
p_x = probs['marginal_x']['a']
```

---

#### jaccard_index

Jaccard similarity coefficient based on context overlap.

```python
def jaccard_index(self) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Returns:**
- Dictionary mapping pairs to Jaccard index scores (range [0, 1])

**Mathematical Formula:**

$$J(X,Y) = \frac{|X \cap Y|}{|X \cup Y|}$$

**Example:**
```python
jaccard_scores = scorer.jaccard_index()
similarity = jaccard_scores[('a', 'x')][0]
```

**Notes:**
- Measures overlap in the contexts where symbols appear
- Symmetric measure (both directions typically equal)
- Range [0, 1] where 1 indicates perfect overlap

---

#### goodman_kruskal_lambda

Goodman-Kruskal's Lambda for asymmetric association.

```python
def goodman_kruskal_lambda(self) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Returns:**
- Dictionary mapping pairs to (λ(Y|X), λ(X|Y)) coefficients

**Mathematical Formula:**

$$\lambda(Y|X) = \frac{\sum_x \max(n_{x,y}) - \max(n_y)}{n - \max(n_y)}$$

**Example:**
```python
lambda_scores = scorer.goodman_kruskal_lambda()
lambda_y_given_x, lambda_x_given_y = lambda_scores[('a', 'x')]
```

**Notes:**
- Measures proportional reduction in prediction error
- Range [0, 1] where 1 indicates perfect prediction
- Asymmetric: λ(Y|X) ≠ λ(X|Y) in general

---

### Information-Theoretic Measures

#### pmi

Pointwise Mutual Information.

```python
def pmi(self, normalized: bool = False) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Parameters:**
- `normalized` - Whether to return normalized PMI (NPMI) in range [-1, 1] (default: False)

**Returns:**
- Dictionary mapping pairs to PMI or NPMI scores

**Mathematical Formula:**

PMI:
$$PMI(x,y) = \log \frac{P(x,y)}{P(x)P(y)}$$

NPMI:
$$NPMI(x,y) = \frac{PMI(x,y)}{-\log P(x,y)}$$

**Example:**
```python
pmi_scores = scorer.pmi(normalized=False)
npmi_scores = scorer.pmi(normalized=True)
pmi_xy, pmi_yx = pmi_scores[('a', 'x')]
```

**Notes:**
- Positive values indicate attraction, negative values repulsion
- NPMI normalizes to [-1, 1] range for easier interpretation
- Uses (1/n)² as limit for unobserved pairs

---

#### pmi_smoothed

PMI with freqprob smoothing for numerical stability.

```python
def pmi_smoothed(self, normalized: bool = False) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Parameters:**
- `normalized` - Whether to return NPMI (default: False)

**Returns:**
- Dictionary mapping pairs to smoothed PMI/NPMI scores

**Example:**
```python
scorer = asymcat.scorer.CatScorer(cooccs, smoothing_method='lidstone', smoothing_alpha=0.5)
pmi_smooth = scorer.pmi_smoothed(normalized=True)
```

**Notes:**
- Recommended over standard pmi() for better handling of sparse data
- Uses configured smoothing method (mle/laplace/lidstone)

---

#### mutual_information

Mutual Information between symbol pairs.

```python
def mutual_information(self) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Returns:**
- Dictionary mapping pairs to (MI(X;Y), MI(Y;X)) scores

**Mathematical Formula:**

$$MI(X;Y) = \sum_x \sum_y P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$

**Example:**
```python
mi_scores = scorer.mutual_information()
mi_xy, mi_yx = mi_scores[('a', 'x')]
```

**Notes:**
- Measures statistical dependence between symbols
- Non-negative: 0 indicates independence
- Symmetric in theory but computed for both directions

---

#### normalized_mutual_information

Normalized Mutual Information.

```python
def normalized_mutual_information(self) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Returns:**
- Dictionary mapping pairs to NMI scores (range [0, 1])

**Mathematical Formula:**

$$NMI(X;Y) = \frac{MI(X;Y)}{H(X,Y)}$$

where H(X,Y) is the joint entropy.

**Example:**
```python
nmi_scores = scorer.normalized_mutual_information()
nmi_xy, nmi_yx = nmi_scores[('a', 'x')]
```

**Notes:**
- Normalized by joint entropy to range [0, 1]
- 1 indicates perfect dependence, 0 indicates independence

---

#### cond_entropy

Conditional entropy.

```python
def cond_entropy(self) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Returns:**
- Dictionary mapping pairs to (H(Y|X), H(X|Y)) entropy values

**Mathematical Formula:**

$$H(Y|X) = \sum_x \sum_y P(x,y) \log \frac{P(x)}{P(x,y)}$$

**Example:**
```python
entropy_scores = scorer.cond_entropy()
h_y_given_x, h_x_given_y = entropy_scores[('a', 'x')]
```

**Notes:**
- Measures remaining uncertainty about Y after observing X
- Lower values indicate stronger association
- Asymmetric: H(Y|X) ≠ H(X|Y) in general

---

#### theil_u

Theil's uncertainty coefficient (Theil's U).

```python
def theil_u(self) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Returns:**
- Dictionary mapping pairs to (U(Y|X), U(X|Y)) coefficients

**Mathematical Formula:**

$$U(Y|X) = \frac{H(Y) - H(Y|X)}{H(Y)}$$

**Example:**
```python
theil_scores = scorer.theil_u()
u_y_given_x, u_x_given_y = theil_scores[('a', 'x')]
```

**Notes:**
- Range [0, 1] where 1 indicates perfect prediction
- Normalized version of mutual information
- Asymmetric measure of association

---

### Statistical Measures

#### chi2

Chi-square statistic for independence testing.

```python
def chi2(self, square_ct: bool = True) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Parameters:**
- `square_ct` - Whether to use square (2x2) or non-square (3x2) contingency table (default: True)

**Returns:**
- Dictionary mapping pairs to χ² statistics

**Mathematical Formula:**

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

**Example:**
```python
chi2_scores = scorer.chi2(square_ct=True)
chi2_value = chi2_scores[('a', 'x')][0]
```

**Notes:**
- Larger values indicate stronger association
- Not suitable for small expected frequencies (<5)
- Symmetric statistic (both directions equal)

---

#### cramers_v

Cramér's V correlation coefficient.

```python
def cramers_v(self, square_ct: bool = True) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Parameters:**
- `square_ct` - Whether to use square or non-square contingency table (default: True)

**Returns:**
- Dictionary mapping pairs to Cramér's V values (range [0, 1])

**Mathematical Formula:**

$$V = \sqrt{\frac{\phi^2}{\min(k-1, r-1)}}$$

where φ² is the chi-square statistic divided by sample size.

**Example:**
```python
cramers_scores = scorer.cramers_v(square_ct=True)
v_value = cramers_scores[('a', 'x')][0]
```

**Notes:**
- Normalized version of chi-square
- Range [0, 1] where 1 indicates perfect association
- Symmetric measure
- Corrected for bias in small samples

---

#### fisher

Fisher's Exact Test odds ratio.

```python
def fisher(self) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Returns:**
- Dictionary mapping pairs to Fisher's exact odds ratios

**Mathematical Formula:**

$$OR = \frac{n_{11} \cdot n_{22}}{n_{12} \cdot n_{21}}$$

**Example:**
```python
fisher_scores = scorer.fisher()
odds_ratio = fisher_scores[('a', 'x')][0]
```

**Notes:**
- Exact test suitable for small sample sizes
- Computes unconditional MLE (differs from R implementation)
- Can be slow for large contingency tables
- Use chi2() for faster approximate results with large samples

**Warning:**
- Very slow for tables with large numbers
- Consider using chi2() or log_likelihood_ratio() as alternatives

---

#### log_likelihood_ratio

Log-Likelihood Ratio (G²) statistic.

```python
def log_likelihood_ratio(self, square_ct: bool = True) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Parameters:**
- `square_ct` - Whether to use square or non-square contingency table (default: True)

**Returns:**
- Dictionary mapping pairs to G² statistics

**Mathematical Formula:**

$$G^2 = 2 \sum_{i,j} O_{ij} \ln\frac{O_{ij}}{E_{ij}}$$

**Example:**
```python
llr_scores = scorer.log_likelihood_ratio(square_ct=True)
g2_value = llr_scores[('a', 'x')][0]
```

**Notes:**
- Alternative to chi-square, works better with small expected frequencies
- Asymptotically equivalent to chi-square for large samples
- Higher values indicate stronger association

---

### Specialized Measures

#### tresoldi

Tresoldi asymmetric uncertainty scorer.

```python
def tresoldi(self) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Returns:**
- Dictionary mapping pairs to asymmetric association scores

**Mathematical Formula:**

$$T(x,y) = \begin{cases}
-(-PMI)^{1-MLE} & \text{if } PMI < 0 \\
PMI^{1-MLE} & \text{if } PMI \geq 0
\end{cases}$$

**Example:**
```python
tresoldi_scores = scorer.tresoldi()
score_xy, score_yx = tresoldi_scores[('a', 'x')]
```

**Notes:**
- Designed specifically for sequence alignment tasks
- Combines PMI and MLE information
- Asymmetric: accounts for directional dependencies
- Preferred measure for phonetic/orthographic alignments

---

## Utility Functions

### scorer2matrices

Convert scorer dictionary to asymmetric matrices.

```python
def scorer2matrices(scorer: dict[tuple[Any, Any], tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, list[Any], list[Any]]
```

**Parameters:**
- `scorer` - Dictionary mapping pairs to (score_xy, score_yx) tuples

**Returns:**
- Tuple of (matrix_xy, matrix_yx, alphabet_x, alphabet_y) where:
  - `matrix_xy`: Y-given-X scoring matrix
  - `matrix_yx`: X-given-Y scoring matrix
  - `alphabet_x`: Alphabet for X dimension
  - `alphabet_y`: Alphabet for Y dimension

**Example:**
```python
from asymcat.scorer import scorer2matrices

mle = scorer.mle()
mat_xy, mat_yx, alph_x, alph_y = scorer2matrices(mle)
# mat_xy[i, j] = P(alph_y[j] | alph_x[i])
```

**Notes:**
- Useful for visualization and matrix operations
- Matrices are indexed consistently with alphabets

---

### scale_scorer

Scale a scorer to a specified range or normalization.

```python
def scale_scorer(scorer: dict[tuple[Any, Any], tuple[float, float]],
                 method: str = "minmax",
                 nrange: tuple[float, float] = (0, 1)) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Parameters:**
- `scorer` - Scoring dictionary to scale
- `method` - Scaling method: "minmax", "mean", or "stdev" (default: "minmax")
- `nrange` - Target range for minmax scaling (default: (0, 1))

**Returns:**
- Scaled version of the scorer

**Scaling Methods:**
- `"minmax"`: Linear scaling to specified range
- `"mean"`: Mean-centered scaling
- `"stdev"`: Standardization (z-score normalization)

**Example:**
```python
from asymcat.scorer import scale_scorer

mle = scorer.mle()
scaled = scale_scorer(mle, method="minmax", nrange=(0, 1))
# All scores now in [0, 1]

standardized = scale_scorer(mle, method="stdev")
# Mean=0, unit variance
```

**Notes:**
- Scaling considers all scores (both xy and yx) together
- Preserves relative ordering of scores

---

### invert_scorer

Invert a scorer so higher values indicate higher affinity.

```python
def invert_scorer(scorer: dict[tuple[Any, Any], tuple[float, float]]) -> dict[tuple[Any, Any], tuple[float, float]]
```

**Parameters:**
- `scorer` - Scoring dictionary to invert

**Returns:**
- Inverted version of the scorer

**Example:**
```python
from asymcat.scorer import invert_scorer

entropy = scorer.cond_entropy()  # Lower = stronger association
inverted = invert_scorer(entropy)  # Higher = stronger association
```

**Notes:**
- Recommended only for scorers in range [0, ∞)
- Inverts by subtracting from maximum: max_score - score
- Useful for measures where lower values indicate stronger association

---

## Correlation Functions

Symmetric correlation measures for categorical series.

### cramers_v

Compute Cramér's V correlation between two series.

```python
def cramers_v(series_x, series_y) -> float
```

**Parameters:**
- `series_x` - First categorical series
- `series_y` - Second categorical series

**Returns:**
- Cramér's V correlation coefficient (range [0, 1])

**Example:**
```python
from asymcat import correlation

series_x = ['a', 'b', 'a', 'c', 'b']
series_y = ['x', 'y', 'x', 'z', 'y']
v = correlation.cramers_v(series_x, series_y)
print(f"Cramér's V: {v:.3f}")
```

**Notes:**
- Symmetric measure (order doesn't matter)
- Based on chi-square statistic
- 1 indicates perfect association, 0 indicates independence

---

### conditional_entropy

Compute conditional entropy between two series.

```python
def conditional_entropy(series_x, series_y) -> float
```

**Parameters:**
- `series_x` - First categorical series
- `series_y` - Second categorical series

**Returns:**
- Conditional entropy H(series_x | series_y)

**Example:**
```python
from asymcat import correlation

h = correlation.conditional_entropy(series_x, series_y)
```

**Notes:**
- Measures uncertainty about series_x given series_y
- Lower values indicate stronger association
- Asymmetric: H(X|Y) ≠ H(Y|X)

---

### theil_u

Compute Theil's U uncertainty coefficient.

```python
def theil_u(series_x, series_y) -> float
```

**Parameters:**
- `series_x` - First categorical series
- `series_y` - Second categorical series

**Returns:**
- Theil's U coefficient

**Example:**
```python
from asymcat import correlation

u = correlation.theil_u(series_x, series_y)
```

**Notes:**
- Normalized version of mutual information
- Asymmetric measure

---

## Type Definitions

### CooccDict

Type alias for co-occurrence dictionaries.

```python
CooccDict = dict[tuple[Any, Any], int]
```

Dictionary mapping co-occurrence pairs to their frequency counts.

---

### ScorerDict

Type alias for scorer output.

```python
ScorerDict = dict[tuple[Any, Any], tuple[float, float]]
```

Dictionary mapping symbol pairs to asymmetric score tuples (score_xy, score_yx).

---

### ObservationDict

Type alias for observation counts.

```python
ObservationDict = dict[tuple[Any, Any], dict[str, int]]
```

Dictionary mapping pairs to their observation statistics with keys:
- `"00"`, `"10"`, `"20"`, `"01"`, `"02"`, `"11"`, `"12"`, `"21"`, `"22"`

---

## Complete Workflow Example

### Basic Co-occurrence Analysis

```python
import asymcat

# Step 1: Load data
seqs = asymcat.read_sequences("data/alignments.tsv")

# Step 2: Collect co-occurrences
cooccs = asymcat.collect_cooccs(seqs, order=2, pad='#')

# Step 3: Create scorer with smoothing
scorer = asymcat.scorer.CatScorer(cooccs, smoothing_method='lidstone', smoothing_alpha=0.5)

# Step 4: Compute multiple association measures
mle = scorer.mle()
pmi = scorer.pmi(normalized=True)
tresoldi = scorer.tresoldi()

# Step 5: Scale scores to [0, 1]
from asymcat.scorer import scale_scorer
scaled_pmi = scale_scorer(pmi, method="minmax", nrange=(0, 1))

# Step 6: Convert to matrices for visualization
from asymcat.scorer import scorer2matrices
mat_xy, mat_yx, alph_x, alph_y = scorer2matrices(scaled_pmi)

print(f"X alphabet: {alph_x}")
print(f"Y alphabet: {alph_y}")
print(f"Matrix shape: {mat_xy.shape}")
```

---

### Presence-Absence Analysis

```python
import asymcat

# Load presence-absence matrix
combinations = asymcat.read_pa_matrix("data/biogeography.tsv")

# Create scorer
scorer = asymcat.scorer.CatScorer(combinations)

# Compute association measures
jaccard = scorer.jaccard_index()
cramers = scorer.cramers_v()
mi = scorer.mutual_information()

# Find strongest associations
for pair, (score_xy, score_yx) in mi.items():
    if score_xy > 2.0:  # Threshold
        print(f"{pair[0]} <-> {pair[1]}: MI = {score_xy:.3f}")
```

---

### N-gram Alignment Scoring

```python
import asymcat
from asymcat.scorer import scale_scorer, scorer2matrices

# Load parallel text sequences
seqs = asymcat.read_sequences("orthography_phonetics.tsv")

# Collect bigram co-occurrences
bigrams = asymcat.collect_cooccs(seqs, order=2, pad='#')

# Create scorer with Laplace smoothing
scorer = asymcat.scorer.CatScorer(bigrams, smoothing_method='laplace')

# Use Tresoldi measure (designed for alignments)
tresoldi_scores = scorer.tresoldi()

# Scale to [0, 1] for easier interpretation
scaled = scale_scorer(tresoldi_scores, method="minmax", nrange=(0, 1))

# Get strongest alignments
sorted_pairs = sorted(scaled.items(), key=lambda x: max(x[1]), reverse=True)
for pair, (xy, yx) in sorted_pairs[:10]:
    print(f"{pair[0]} -> {pair[1]}: {xy:.3f}")
    print(f"{pair[1]} -> {pair[0]}: {yx:.3f}")
```

---

### Comparing Multiple Measures

```python
import asymcat

cooccs = asymcat.collect_cooccs(seqs)
scorer = asymcat.scorer.CatScorer(cooccs)

# Compute various measures
measures = {
    'MLE': scorer.mle(),
    'PMI': scorer.pmi(normalized=True),
    'Chi2': scorer.chi2(),
    'Cramér V': scorer.cramers_v(),
    'MI': scorer.mutual_information(),
    'Tresoldi': scorer.tresoldi()
}

# Analyze a specific pair
target_pair = ('a', 'x')
print(f"Association measures for {target_pair}:")
for name, scores in measures.items():
    xy, yx = scores[target_pair]
    print(f"  {name:12s}: X->Y={xy:.4f}, Y->X={yx:.4f}")
```

---

## Performance Notes

### Computational Complexity

- `collect_cooccs()`: O(n × m) where n is number of sequence pairs, m is sequence length
- `collect_observations()`: O(k × v) where k is number of pairs, v is vocabulary size
- Most association measures: O(k) where k is number of unique pairs
- `theil_u()`: O(k × n) where n is number of observations (slower for large datasets)
- `fisher()`: Can be very slow for large contingency tables

### Memory Considerations

- Co-occurrence lists scale with data size
- Observation dictionaries scale with vocabulary size squared
- All scorers cache results for lazy evaluation
- Use appropriate smoothing for sparse data

### Optimization Tips

1. Use appropriate `order` parameter for n-gram collection
2. Choose smoothing method based on data sparsity
3. Compute only needed association measures
4. Use `chi2()` or `log_likelihood_ratio()` instead of `fisher()` for large datasets
5. Scale/normalize scores after computation, not before

---

## Version Information

This API reference corresponds to ASymCat version 0.4.0.

For tutorials, examples, and additional documentation, see:
- Main documentation: `/docs/`
- Tutorial notebooks: `/notebooks/`
- Example scripts: `/examples/`

---

## Error Handling

### Common Exceptions

- `ValueError`: Invalid parameters, empty data, or mismatched sequence lengths
- `TypeError`: Incorrect data types for parameters
- `FileNotFoundError`: Missing data files
- `KeyError`: Missing required elements in data structures

### Error Prevention

```python
# Validate data before processing
if not cooccs:
    raise ValueError("Empty co-occurrence list")

# Check sequence alignment
if order is not None:
    for seq_a, seq_b in seqs:
        if len(seq_a) != len(seq_b):
            raise ValueError("Sequences must be aligned")

# Handle missing observations
obs = collect_observations(cooccs)
if pair not in obs:
    print(f"Pair {pair} not observed in data")
```

This comprehensive API reference covers all public functions and classes in the ASymCat library. For implementation details and advanced usage, consult the source code and example notebooks.
