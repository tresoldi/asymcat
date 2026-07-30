Mathematical Foundations
=======================

This document provides the mathematical foundation for asymmetric categorical association analysis implemented in ASymCat. We present the theoretical framework, formal definitions, and mathematical properties of each measure.

.. contents:: Table of Contents
   :local:
   :depth: 3

Theoretical Framework
---------------------

Basic Definitions
~~~~~~~~~~~~~~~~~

Let **X** and **Y** be two categorical random variables with finite alphabets:

- :math:`\mathcal{A}_X = \{x_1, x_2, \ldots, x_m\}` (alphabet of X)
- :math:`\mathcal{A}_Y = \{y_1, y_2, \ldots, y_n\}` (alphabet of Y)

Given a dataset of observations :math:`\mathcal{D} = \{(x_i, y_j)\}_{i,j}`, we define the **co-occurrence counts**:

.. math::
   c_{ij} = |\{(x, y) \in \mathcal{D} : x = x_i, y = y_j\}|

**Joint Distribution:**

.. math::
   P(X = x_i, Y = y_j) = \frac{c_{ij}}{N}

where :math:`N = \sum_{i,j} c_{ij}` is the total number of observations.

**Marginal Distributions:**

.. math::
   P(X = x_i) = \sum_{j=1}^n P(X = x_i, Y = y_j) = \frac{\sum_{j=1}^n c_{ij}}{N}

.. math::
   P(Y = y_j) = \sum_{i=1}^m P(X = x_i, Y = y_j) = \frac{\sum_{i=1}^m c_{ij}}{N}

**Conditional Distributions (Asymmetric):**

.. math::
   P(Y = y_j | X = x_i) = \frac{P(X = x_i, Y = y_j)}{P(X = x_i)} = \frac{c_{ij}}{\sum_{k=1}^n c_{ik}}

.. math::
   P(X = x_i | Y = y_j) = \frac{P(X = x_i, Y = y_j)}{P(Y = y_j)} = \frac{c_{ij}}{\sum_{k=1}^m c_{kj}}

Information-Theoretic Foundation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Entropy:**

.. math::
   H(X) = -\sum_{i=1}^m P(x_i) \log P(x_i)

.. math::
   H(Y) = -\sum_{j=1}^n P(y_j) \log P(y_j)

**Joint Entropy:**

.. math::
   H(X, Y) = -\sum_{i=1}^m \sum_{j=1}^n P(x_i, y_j) \log P(x_i, y_j)

**Conditional Entropy (Asymmetric):**

.. math::
   H(Y | X) = -\sum_{i=1}^m \sum_{j=1}^n P(x_i, y_j) \log P(y_j | x_i)

.. math::
   H(X | Y) = -\sum_{i=1}^m \sum_{j=1}^n P(x_i, y_j) \log P(x_i | y_j)

**Mutual Information (Symmetric):**

.. math::
   I(X; Y) = H(X) + H(Y) - H(X, Y) = H(Y) - H(Y | X) = H(X) - H(X | Y)

Association Measures
--------------------

Probabilistic Measures
~~~~~~~~~~~~~~~~~~~~~~~

Maximum Likelihood Estimation (MLE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition:**
Direct estimation of conditional probabilities.

**Formula:**

.. math::
   \text{MLE}_{X \to Y}(x_i, y_j) = P(y_j | x_i) = \frac{c_{ij}}{\sum_{k=1}^n c_{ik}}

.. math::
   \text{MLE}_{Y \to X}(x_i, y_j) = P(x_i | y_j) = \frac{c_{ij}}{\sum_{k=1}^m c_{kj}}

**Properties:**
- Range: :math:`[0, 1]`
- Asymmetric: :math:`P(Y|X) \neq P(X|Y)` in general
- Interpretation: Direct conditional probability
- Special case: :math:`P(Y|X) = 1` implies perfect prediction

**Use cases:** Direct probability interpretation, prediction tasks, causal modeling.

Jaccard Index (Asymmetric Interpretation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition:**
Measures overlap between contexts of categories.

**Formula:**

.. math::
   J_{X \to Y}(x_i, y_j) = \frac{|\text{contexts}(x_i) \cap \text{contexts}(y_j)|}{|\text{contexts}(x_i)|}

where :math:`\text{contexts}(x_i)` is the set of categories that co-occur with :math:`x_i`.

**Properties:**
- Range: :math:`[0, 1]`
- Asymmetric interpretation of traditionally symmetric measure
- Interpretation: Proportion of shared contexts

**Use cases:** Context similarity, co-occurrence strength, clustering applications.

Information-Theoretic Measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pointwise Mutual Information (PMI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition:**
Measures the information gained about one variable by observing another.

**Formula:**

.. math::
   \text{PMI}(x_i, y_j) = \log \frac{P(x_i, y_j)}{P(x_i) P(y_j)}

**Properties:**
- Range: :math:`(-\infty, +\infty)`
- Symmetric: :math:`\text{PMI}(x_i, y_j) = \text{PMI}(y_j, x_i)`
- Interpretation: 
  - :math:`\text{PMI} > 0`: Positive association (co-occur more than expected)
  - :math:`\text{PMI} = 0`: Independence
  - :math:`\text{PMI} < 0`: Negative association (co-occur less than expected)

**Smoothed PMI:**
To handle numerical instability with rare events, ASymCat implements smoothed PMI using the FreqProb library:

.. math::
   \text{PMI}_{\text{smooth}}(x_i, y_j) = \log \frac{P_{\text{smooth}}(x_i, y_j)}{P_{\text{smooth}}(x_i) P_{\text{smooth}}(y_j)}

**Use cases:** Information-theoretic analysis, feature selection, natural language processing.

Normalized PMI (NPMI)
^^^^^^^^^^^^^^^^^^^^^^

**Definition:**
PMI normalized to :math:`[-1, 1]` range.

**Formula:**

.. math::
   \text{NPMI}(x_i, y_j) = \frac{\text{PMI}(x_i, y_j)}{-\log P(x_i, y_j)}

**Properties:**
- Range: :math:`[-1, 1]`
- Symmetric
- Interpretation: 
  - :math:`\text{NPMI} = 1`: Perfect association
  - :math:`\text{NPMI} = 0`: Independence
  - :math:`\text{NPMI} = -1`: Perfect negative association

**Use cases:** Normalized information content, comparative analysis across datasets.

Mutual Information
^^^^^^^^^^^^^^^^^^

**Definition:**
Average mutual information between variables.

**Formula:**

.. math::
   I(X; Y) = \sum_{i=1}^m \sum_{j=1}^n P(x_i, y_j) \log \frac{P(x_i, y_j)}{P(x_i) P(y_j)}

**Properties:**
- Range: :math:`[0, \min(H(X), H(Y))]`
- Symmetric: :math:`I(X; Y) = I(Y; X)`
- Interpretation: Total information shared between variables

**Use cases:** Feature selection, dependency detection, information flow analysis.

Conditional Entropy
^^^^^^^^^^^^^^^^^^^

**Definition:**
Uncertainty remaining in one variable after observing another.

**Formula:**

.. math::
   H(Y | X = x_i) = -\sum_{j=1}^n P(y_j | x_i) \log P(y_j | x_i)

**Properties:**
- Range: :math:`[0, \log n]` for Y with n categories
- Asymmetric: :math:`H(Y|X) \neq H(X|Y)` in general
- Interpretation: Lower values indicate better predictability

**Use cases:** Predictability assessment, entropy-based feature selection.

Statistical Measures
~~~~~~~~~~~~~~~~~~~~~

Pearson's Chi-Square
^^^^^^^^^^^^^^^^^^^^

**Definition:**
Tests independence between categorical variables.

**Formula:**

.. math::
   \chi^2 = \sum_{i=1}^m \sum_{j=1}^n \frac{(O_{ij} - E_{ij})^2}{E_{ij}}

where :math:`O_{ij} = c_{ij}` (observed) and :math:`E_{ij} = \frac{\sum_k c_{ik} \sum_k c_{kj}}{N}` (expected under independence).

**Properties:**
- Range: :math:`[0, +\infty)`
- Symmetric
- Interpretation: Deviation from independence; higher values indicate stronger association

**Use cases:** Independence testing, goodness-of-fit tests, categorical data analysis.

Cramér's V
^^^^^^^^^^^

**Definition:**
Normalized chi-square association measure.

**Formula:**

.. math::
   V = \sqrt{\frac{\chi^2}{N \cdot \min(m-1, n-1)}}

**Properties:**
- Range: :math:`[0, 1]`
- Symmetric
- Interpretation: 
  - :math:`V = 0`: Independence
  - :math:`V = 1`: Perfect association

**Use cases:** Effect size measurement, comparative association strength.

Fisher's Exact Test
^^^^^^^^^^^^^^^^^^^

**Definition:**
Exact statistical test for association in 2×2 contingency tables.

**Formula:**
For 2×2 table with cells :math:`a, b, c, d`:

.. math::
   P = \frac{\binom{a+b}{a}\binom{c+d}{c}}{\binom{n}{a+c}}

**Properties:**
- Range: :math:`[0, 1]` (p-value)
- Exact test (no approximation)
- Particularly useful for small sample sizes

**Use cases:** Small sample testing, exact inference, biological applications.

Log-Likelihood Ratio (G²)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition:**
Likelihood ratio test statistic for independence.

**Formula:**

.. math::
   G^2 = 2 \sum_{i=1}^m \sum_{j=1}^n O_{ij} \log \frac{O_{ij}}{E_{ij}}

**Properties:**
- Range: :math:`[0, +\infty)`
- Asymptotically equivalent to :math:`\chi^2`
- Better for sparse data than chi-square

**Use cases:** Likelihood-based inference, sparse contingency tables.

Specialized Measures
~~~~~~~~~~~~~~~~~~~~

Theil's Uncertainty Coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition:**
Measures proportional uncertainty reduction.

**Formula:**

.. math::
   U(Y | X) = \frac{I(X; Y)}{H(Y)} = \frac{H(Y) - H(Y | X)}{H(Y)}

.. math::
   U(X | Y) = \frac{I(X; Y)}{H(X)} = \frac{H(X) - H(X | Y)}{H(X)}

**Properties:**
- Range: :math:`[0, 1]`
- Asymmetric: :math:`U(Y|X) \neq U(X|Y)` in general
- Interpretation: Proportion of uncertainty in Y reduced by knowing X

**Use cases:** Information-theoretic dependency, prediction improvement measurement.

Goodman-Kruskal Lambda
^^^^^^^^^^^^^^^^^^^^^^

**Definition:**
Measures proportional reduction in prediction error.

**Formula:**

.. math::
   \lambda(Y | X) = \frac{\sum_{i=1}^m \max_j c_{ij} - \max_j \sum_{i=1}^m c_{ij}}{N - \max_j \sum_{i=1}^m c_{ij}}

**Properties:**
- Range: :math:`[0, 1]`
- Asymmetric
- Interpretation: Proportional reduction in classification error

**Use cases:** Classification improvement, categorical prediction, error reduction analysis.

Tresoldi Measure
^^^^^^^^^^^^^^^^

**Definition:**
Custom measure designed for sequence alignment applications.

**Formula:**

.. math::
   T(x_i, y_j) = \frac{\text{MLE}(x_i, y_j) \cdot \text{PMI}(x_i, y_j)}{\text{Entropy penalty}}

This measure combines probabilistic and information-theoretic components with domain-specific normalization.

**Properties:**
- Range: Context-dependent
- Asymmetric
- Optimized for linguistic sequence analysis

**Use cases:** Historical linguistics, sequence alignment, phonetic correspondence analysis.

Smoothing Methods
-----------------

Maximum Likelihood Estimation (MLE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**No Smoothing:**

.. math::
   P(y_j | x_i) = \frac{c_{ij}}{\sum_{k=1}^n c_{ik}}

**Problem:** Zero probabilities for unobserved events.

Laplace Smoothing (Add-One)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   P_{\text{Laplace}}(y_j | x_i) = \frac{c_{ij} + 1}{\sum_{k=1}^n c_{ik} + n}

**Properties:**
- Adds pseudo-count of 1 to all events
- Ensures no zero probabilities
- Simple and robust

Lidstone Smoothing (Expected Likelihood Estimation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   P_{\text{ELE}}(y_j | x_i) = \frac{c_{ij} + \alpha}{\sum_{k=1}^n c_{ik} + \alpha n}

where :math:`\alpha > 0` is the smoothing parameter.

**Properties:**
- Generalizes Laplace smoothing (:math:`\alpha = 1`)
- Tunable smoothing strength
- :math:`\alpha < 1`: Conservative smoothing
- :math:`\alpha > 1`: Aggressive smoothing

FreqProb Integration
~~~~~~~~~~~~~~~~~~~

ASymCat integrates the FreqProb library for advanced probability estimation:

.. math::
   P_{\text{FreqProb}}(y_j | x_i) = \text{FreqProb}(c_{ij}, \sum_{k=1}^n c_{ik}, \text{method}, \alpha)

**Available methods:**
- ``'mle'``: Maximum likelihood (no smoothing)
- ``'laplace'``: Laplace smoothing
- ``'lidstone'``: Parameterized smoothing

Mathematical Properties
-----------------------

Symmetry and Asymmetry
~~~~~~~~~~~~~~~~~~~~~~

**Symmetric Measures:**
- :math:`M(X, Y) = M(Y, X)`
- Examples: PMI, χ², Cramér's V, Jaccard (traditional)

**Asymmetric Measures:**
- :math:`M(X \to Y) \neq M(Y \to X)` in general
- Examples: MLE, Theil's U, λ, conditional entropy

Relationship Between Measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Information-Theoretic Relations:**

.. math::
   I(X; Y) = H(X) + H(Y) - H(X, Y)

.. math::
   U(Y | X) = \frac{I(X; Y)}{H(Y)}

.. math::
   U(X | Y) = \frac{I(X; Y)}{H(X)}

**Boundary Conditions:**
- Perfect association: :math:`P(Y|X) = 1` implies :math:`H(Y|X) = 0`
- Independence: :math:`P(X, Y) = P(X)P(Y)` implies :math:`I(X; Y) = 0`

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

**Time Complexity:**
- Most measures: :math:`O(mn)` where :math:`m`, :math:`n` are alphabet sizes
- Matrix operations: :math:`O(mn + m^2 + n^2)`
- Statistical tests: Problem-specific complexity

**Space Complexity:**
- Co-occurrence storage: :math:`O(mn)`
- Probability matrices: :math:`O(mn)`

Numerical Considerations
------------------------

Stability Issues
~~~~~~~~~~~~~~~~

1. **Log of zero:** Use smoothing for PMI-based measures
2. **Division by zero:** Handle empty marginals with smoothing
3. **Underflow:** Use log-space computation for very small probabilities

Precision Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Standard analysis:** 64-bit floating point (Python default)
- **High-precision needs:** Use NumPy's extended precision
- **Sparse data:** Always apply appropriate smoothing

This mathematical foundation provides the theoretical basis for understanding and applying ASymCat's asymmetric association measures. The formal definitions enable rigorous analysis while the computational considerations ensure practical applicability.