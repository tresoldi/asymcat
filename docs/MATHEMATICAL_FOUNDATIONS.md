# Mathematical Foundations of ASymCat

## Overview

ASymCat implements **asymmetric measures of association** between categorical variables. Unlike traditional symmetric measures (where association A→B equals B→A), asymmetric measures capture directional dependencies crucial for understanding causal relationships and information flow.

## Core Mathematical Concepts

### 1. Co-occurrence Matrix

For categorical variables X and Y, we define a **2×2 contingency table** for each pair (x,y):

```
         Y=y    Y≠y    Total
X=x      n₁₁    n₁₀     n₁•
X≠x      n₀₁    n₀₀     n₀•
Total    n•₁    n•₀      n
```

Where:
- `n₁₁`: Count of observations where X=x AND Y=y (joint occurrence)
- `n₁₀`: Count of observations where X=x AND Y≠y
- `n₀₁`: Count of observations where X≠x AND Y=y
- `n₀₀`: Count of observations where X≠x AND Y≠y
- `n`: Total number of observations

### 2. Probability Estimation

#### Maximum Likelihood Estimation (MLE)

The basic probability estimates are:

- **Joint probability**: P(X=x, Y=y) = n₁₁/n
- **Marginal probabilities**: P(X=x) = n₁•/n, P(Y=y) = n•₁/n
- **Conditional probabilities**:
  - P(X=x|Y=y) = n₁₁/n•₁ (X given Y)
  - P(Y=y|X=x) = n₁₁/n₁• (Y given X)

#### Smoothing Methods

For sparse data, ASymCat implements smoothing via the freqprob library:

**Laplace Smoothing (Add-1)**:
```
P(X=x|Y=y) = (n₁₁ + α) / (n•₁ + α|X|)
```

**Expected Likelihood Estimation (ELE)**:
```
P(X=x|Y=y) = (n₁₁ + α·P(X=x)) / (n•₁ + α)
```

Where α is the smoothing parameter and |X| is the size of alphabet X.

### 3. Asymmetric Association Measures

#### Maximum Likelihood Estimation Score
**Asymmetric**: Directional conditional probability
```
MLE(X→Y) = P(Y=y|X=x) = n₁₁/n₁•
MLE(Y→X) = P(X=x|Y=y) = n₁₁/n•₁
```

#### Pointwise Mutual Information (PMI)
**Symmetric**: Measures how much more likely the joint occurrence is compared to independence
```
PMI(X,Y) = log[P(X=x,Y=y) / (P(X=x)·P(Y=y))]
         = log[n₁₁·n / (n₁•·n•₁)]
```

**Normalized PMI (NPMI)**: Scales PMI to [-1, 1] range
```
NPMI(X,Y) = PMI(X,Y) / (-log P(X=x,Y=y))
```

#### Theil's Uncertainty Coefficient (U)
**Asymmetric**: Measures proportional reduction in uncertainty
```
U(X|Y) = [H(X) - H(X|Y)] / H(X)
       = I(X;Y) / H(X)
```

Where:
- H(X) = -Σ P(x) log P(x) (entropy of X)
- H(X|Y) = -Σ P(y) Σ P(x|y) log P(x|y) (conditional entropy)
- I(X;Y) = mutual information

#### Chi-Square Statistic
**Symmetric**: Tests independence hypothesis
```
χ² = n · (n₁₁n₀₀ - n₁₀n₀₁)² / (n₁•n₀•n•₁n•₀)
```

#### Cramér's V
**Symmetric**: Normalized chi-square measure
```
V = √[χ² / (n·min(r-1, c-1))]
```
Where r and c are the number of rows and columns.

#### Jaccard Index
**Symmetric**: Measures overlap in contexts
```
J(X,Y) = |contexts(X) ∩ contexts(Y)| / |contexts(X) ∪ contexts(Y)|
```

#### Goodman-Kruskal Lambda (λ)
**Asymmetric**: Proportional reduction in prediction error
```
λ(X|Y) = [Σ max P(x|y) - max P(x)] / (1 - max P(x))
```

#### Tresoldi Measure
**Asymmetric**: Novel measure combining MLE and PMI
```
Tresoldi(X→Y) = MLE(X→Y) · exp(PMI(X,Y))
```

### 4. Information-Theoretic Measures

#### Mutual Information
**Symmetric**: Measures information shared between variables
```
I(X;Y) = Σ Σ P(x,y) log[P(x,y) / (P(x)P(y))]
```

#### Conditional Entropy
**Asymmetric**: Uncertainty in X given Y
```
H(X|Y) = -Σ P(y) Σ P(x|y) log P(x|y)
```

### 5. Statistical Tests

#### Fisher's Exact Test
**Symmetric**: Exact probability for 2×2 tables
```
P = (n₁•!n₀•!n•₁!n•₀!) / (n!n₁₁!n₁₀!n₀₁!n₀₀!)
```

#### Log-Likelihood Ratio (G²)
**Symmetric**: Alternative to chi-square
```
G² = 2 Σ Σ n_{ij} log(n_{ij} / E_{ij})
```
Where E_{ij} are expected frequencies under independence.

## Asymmetric vs Symmetric: A Mathematical Example

Consider the co-occurrence matrix:

| Observation | X | Y |
|-------------|---|---|
| 1           | A | c |
| 2           | A | d |
| 3           | A | c |
| 4           | B | g |
| 5           | B | g |
| 6           | B | f |

### Contingency Table for (A,c):
```
      Y=c  Y≠c  Total
X=A    2    1     3
X≠A    0    3     3
Total  2    4     6
```

### Asymmetric Analysis:
- **P(X=A|Y=c) = 2/2 = 1.0** (perfect prediction: if Y=c then X=A)
- **P(Y=c|X=A) = 2/3 = 0.67** (uncertain: if X=A then Y might be c or d)

### Symmetric Analysis:
- **PMI(A,c) = log[(2·6)/(3·2)] = log(2) = 0.693** (same for both directions)
- **χ²(A,c) = 6·(2·3-1·0)²/(3·3·2·4) = 2.0** (same for both directions)

This demonstrates how asymmetric measures reveal directional dependencies that symmetric measures cannot capture.

## Implementation Notes

### Numerical Stability
- **Zero co-occurrences**: Handled gracefully with smoothing or direct computation
- **Log-space computation**: PMI and entropy calculations use log-space to prevent underflow
- **Sparse data**: Laplace and ELE smoothing provide robust estimates for rare events

### Computational Complexity
- **Per-pair computation**: O(1) for each measure given contingency table
- **Matrix generation**: O(|X|·|Y|) for full association matrix
- **Memory efficiency**: Lazy evaluation and caching used for large vocabularies

## References

1. **Theil, H.** (1972). *Statistical Decomposition Analysis*. North-Holland.
2. **Goodman, L.A. & Kruskal, W.H.** (1954). Measures of association for cross classifications. *Journal of the American Statistical Association*, 49, 732-764.
3. **Church, K.W. & Hanks, P.** (1990). Word association norms, mutual information, and lexicography. *Computational Linguistics*, 16(1), 22-29.
4. **Manning, C.D. & Schütze, H.** (1999). *Foundations of Statistical Natural Language Processing*. MIT Press.
5. **Tresoldi, T.** (2024). Asymmetric measures for categorical co-occurrence analysis. *GitHub*.
