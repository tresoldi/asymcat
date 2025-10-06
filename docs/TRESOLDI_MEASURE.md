# The Tresoldi Measure

## Overview

The Tresoldi measure is an asymmetric association measure that combines information content with conditional probability. Originally designed for linguistic alignments with asymmetric categorical associations, it addresses a fundamental challenge: how to balance the **informativeness** of an association (measured by PMI) with its **predictive reliability** (measured by conditional probability).

Unlike traditional measures that capture only one aspect of association—either information content (PMI), uncertainty reduction (Theil's U), or prediction strength (MLE)—the Tresoldi measure synthesizes these perspectives into a unified score optimized for directional associations in categorical data.

## Mathematical Definition

### Formula

For a pair of categorical variables (X, Y), the Tresoldi measure in the X → Y direction is defined as:

```
Tresoldi(X→Y) = sign(PMI) · |PMI(X,Y)|^(1 - P(Y|X))
```

Where:
- **PMI(X,Y)** = Pointwise Mutual Information = log(P(X,Y) / (P(X)·P(Y)))
- **P(Y|X)** = Conditional probability (MLE estimate) = P(X,Y) / P(X)
- **sign(PMI)** preserves the sign of the PMI value
- The exponent **(1 - P(Y|X))** acts as a weighting factor

### Component Breakdown

#### PMI (Pointwise Mutual Information)
```
PMI(X,Y) = log(P(X,Y) / (P(X) · P(Y)))
```

- **Range**: (-∞, +∞)
- **Interpretation**:
  - PMI > 0: Co-occurrence is stronger than expected by independence
  - PMI = 0: Variables are independent
  - PMI < 0: Co-occurrence is weaker than expected (negative association)
- **Symmetric**: PMI(X,Y) = PMI(Y,X)
- **Property**: Sensitive to rare events (can be very large for low-frequency pairs)

#### MLE (Maximum Likelihood Estimate / Conditional Probability)
```
P(Y|X) = P(X,Y) / P(X)
```

- **Range**: [0, 1]
- **Interpretation**: Probability of observing Y given that we observed X
- **Asymmetric**: P(Y|X) ≠ P(X|Y) in general
- **Property**: Insensitive to base rates; focuses on local conditional distribution

### The Weighting Mechanism

The exponent (1 - P(Y|X)) creates a dynamic weighting system:

**When P(Y|X) is high (approaching 1):**
- Exponent approaches 0
- |PMI|^0 → 1
- Tresoldi → sign(PMI) · 1 = ±1
- **Effect**: Strong deterministic associations are capped at ±1, regardless of PMI magnitude

**When P(Y|X) is low (approaching 0):**
- Exponent approaches 1
- |PMI|^1 → |PMI|
- Tresoldi → sign(PMI) · |PMI|
- **Effect**: Weak conditional associations retain their PMI value

**Intermediate cases:**
- The measure smoothly interpolates between these extremes
- High PMI with high P(Y|X): Strong, reliable association
- High PMI with low P(Y|X): Informative but unreliable (down-weighted)
- Low PMI with high P(Y|X): Highly predictable but unsurprising
- Low PMI with low P(Y|X): Weak on both dimensions

## Derivation and Motivation

### Why Combine PMI and MLE?

Consider a linguistic alignment scenario: mapping graphemes (letters) to phonemes (sounds).

**Problem 1: PMI alone**
- Rare letter-sound pairs can have extremely high PMI
- Example: A rare digraph 'ZH' → /ʒ/ might have PMI = 8.5
- But this doesn't capture that it's a **reliable** mapping

**Problem 2: MLE alone**
- Common, expected mappings dominate
- Example: 'E' → /ə/ has high P(/ə/|E) = 0.85
- But this doesn't capture that it's **unsurprising** (also common generally)

**Solution: Tresoldi measure**
- Weights PMI by conditional probability
- High scores require both:
  1. Informative co-occurrence (high |PMI|)
  2. Reliable conditional prediction (high P(Y|X))
- Down-weights spurious high-PMI associations from rare events

### Mathematical Intuition

The exponentiation by (1 - P(Y|X)) can be understood through several lenses:

**1. Uncertainty Weighting**
- (1 - P(Y|X)) represents the uncertainty remaining when predicting Y from X
- High uncertainty (low P(Y|X)) → keep full PMI value
- Low uncertainty (high P(Y|X)) → compress PMI toward ±1

**2. Information Dampening**
- When X strongly predicts Y, the "surprise" value (PMI) is less relevant
- The relationship is more about reliability than information content
- The measure shifts from information-centric to prediction-centric

**3. Geometric Mean Perspective**
- For positive PMI: Tresoldi ≈ PMI^α where α = (1 - P(Y|X))
- This is a weighted geometric mean between PMI (weight α) and 1 (weight 1-α)
- Balances information content with deterministic association

## Properties

### Range and Bounds

**Theoretical Range**: (-∞, +∞) (inherited from PMI)

**Practical Range**: Approximately [-10, +10] for typical data

**Boundary Conditions**:

1. **When P(Y|X) = 0**:
   - Exponent = 1
   - Tresoldi = PMI (original information value)

2. **When P(Y|X) = 1**:
   - Exponent = 0
   - Tresoldi = sign(PMI) · 1 = ±1 (deterministic association)

3. **When PMI = 0**:
   - Tresoldi = 0 (independence, regardless of P(Y|X))

4. **When PMI > 0 and P(Y|X) → 1**:
   - Tresoldi → 1 (strong positive association, capped)

5. **When PMI < 0**:
   - Sign is preserved
   - |PMI| is raised to the exponent
   - Result is negative (negative association preserved)

### Asymmetry

The Tresoldi measure is **fully asymmetric**:

```
Tresoldi(X→Y) ≠ Tresoldi(Y→X)
```

This asymmetry arises from both components:
- PMI(X,Y) is symmetric, but
- P(Y|X) ≠ P(X|Y) in general
- Different exponents create different values

**Example**:
- X = 'C', Y = /k/
- P(/k/|C) = 0.8 (C often becomes /k/)
- P(C|/k/) = 0.3 (/k/ comes from many sources: C, K, QU, etc.)
- Tresoldi(C→/k/) and Tresoldi(/k/→C) will differ substantially

### Monotonicity

**In PMI** (for fixed P(Y|X)):
- Tresoldi is monotonically increasing in PMI
- sign and magnitude both preserved through transformation

**In P(Y|X)** (for fixed PMI > 0):
- Tresoldi increases with P(Y|X) when PMI > 1
- Tresoldi decreases with P(Y|X) when 0 < PMI < 1
- Non-monotonic in general

### Scale Invariance

The measure is **not scale-invariant** in the traditional sense, but has useful properties:
- Multiplying all observations by a constant changes absolute probabilities
- PMI is affected by base rate changes
- The exponent (1 - P(Y|X)) adapts to the local conditional distribution

## Comparison with Other Measures

### vs. PMI (Pointwise Mutual Information)

**PMI Formula**:
```
PMI(X,Y) = log(P(X,Y) / (P(X) · P(Y)))
```

**Key Differences**:

| Aspect | PMI | Tresoldi |
|--------|-----|----------|
| **Symmetry** | Symmetric | Asymmetric |
| **Range** | (-∞, +∞) | (-∞, +∞) but compressed |
| **Sensitivity to rare events** | Very high | Dampened by MLE |
| **Interprets** | Surprise/Information | Surprise weighted by reliability |
| **Best for** | Information content | Directional prediction with reliability |

**When PMI can mislead**:
- Rare co-occurrences get disproportionately high scores
- Doesn't distinguish between reliable and unreliable patterns
- Example: A pair occurring 2/2 times has perfect PMI but low statistical power

**How Tresoldi corrects**:
- Down-weights rare events through the conditional probability component
- Balances "surprisingly common" with "reliably predictable"
- The 2/2 pair would have moderate P(Y|X) and moderate Tresoldi

### vs. Theil's U (Uncertainty Coefficient)

**Theil's U Formula**:
```
U(Y|X) = [H(Y) - H(Y|X)] / H(Y)
```

Where:
- H(Y) = entropy of Y = -Σ P(y) log P(y)
- H(Y|X) = conditional entropy = -Σ P(x) Σ P(y|x) log P(y|x)

**Key Differences**:

| Aspect | Theil's U | Tresoldi |
|--------|-----------|----------|
| **Paradigm** | Information theory (entropy reduction) | Hybrid (information × probability) |
| **Range** | [0, 1] | (-∞, +∞) |
| **Normalized** | Yes | No |
| **Aggregation** | Global (across all Y values) | Local (pairwise) |
| **Interprets** | How much X reduces uncertainty about Y | Association strength weighted by reliability |

**Theil's U perspective**:
- Answers: "How much does knowing X reduce my uncertainty about Y?"
- Global measure: considers entire conditional distribution P(Y|X)
- Always non-negative and bounded

**Tresoldi perspective**:
- Answers: "How strongly and reliably does X associate with this specific Y?"
- Local measure: focuses on individual (X, Y) pair
- Preserves sign (positive/negative association)
- Can exceed 1 for very strong information content

**When to prefer each**:
- **Theil's U**: Overall predictive power, feature selection, normalized comparisons
- **Tresoldi**: Specific association strength, alignment scoring, when direction and sign matter

### vs. MLE (Conditional Probability)

**MLE Formula**:
```
P(Y|X) = P(X,Y) / P(X)
```

**Key Differences**:

| Aspect | MLE | Tresoldi |
|--------|-----|----------|
| **What it measures** | Direct probability | Information content weighted by probability |
| **Range** | [0, 1] | (-∞, +∞) |
| **Base rate sensitivity** | Ignores base rates | Accounts for base rates (via PMI) |
| **Interprets** | "How often Y given X?" | "How surprising and reliable is Y given X?" |

**Example demonstrating the difference**:

Imagine:
- Pair 1: 'E' → /ə/: Occurs 85/100 times when we see 'E'
- Pair 2: 'ZH' → /ʒ/: Occurs 8/10 times when we see 'ZH'

**MLE perspective**:
- P(/ə/|E) = 0.85 (high)
- P(/ʒ/|ZH) = 0.80 (slightly lower)
- MLE prefers Pair 1

**But context matters**:
- /ə/ is the most common vowel in English (high base rate P(/ə/))
- /ʒ/ is rare in English (low base rate P(/ʒ/))

**PMI perspective**:
- PMI(E, /ə/) might be low (expected given base rates)
- PMI(ZH, /ʒ/) might be very high (surprising given rarity)

**Tresoldi perspective**:
- Balances both: Pair 1 is reliable but unsurprising
- Pair 2 is slightly less reliable but highly informative
- Tresoldi can rank them more appropriately for linguistic analysis

### vs. Goodman-Kruskal λ (Lambda)

**Lambda Formula**:
```
λ(Y|X) = (E₀ - E₁) / E₀
```

Where:
- E₀ = prediction error without knowing X
- E₁ = prediction error when knowing X

**Key Differences**:

| Aspect | Lambda | Tresoldi |
|--------|--------|----------|
| **Paradigm** | Error reduction | Information content × probability |
| **Range** | [0, 1] | (-∞, +∞) |
| **Interprets** | Proportional reduction in prediction error | Association strength with reliability weighting |
| **Focus** | Prediction improvement | Information and dependence |

**Lambda measures**:
- How much better you can predict Y if you know X
- Based on modal (most frequent) categories
- Focuses on prediction accuracy

**Tresoldi measures**:
- How informative and reliable the X→Y association is
- Considers the specific (X,Y) pair's information content
- Focuses on association strength, not just prediction

## When to Use the Tresoldi Measure

### Ideal Applications

1. **Linguistic Alignments**
   - Grapheme-to-phoneme correspondence
   - Historical sound changes
   - Morpheme boundary prediction
   - Cross-linguistic cognate detection
   - Where both informativeness and reliability matter

2. **Directional Associations**
   - When X→Y relationship differs from Y→X
   - Causal or temporal precedence
   - Asymmetric dependencies

3. **Balancing Rare and Common Events**
   - When you need to avoid over-emphasizing rare co-occurrences
   - When base rates significantly differ across categories
   - When you want reliability-weighted information content

### When to Consider Alternatives

**Use PMI instead when**:
- You want pure information content
- Symmetry is required
- You're okay with high scores for rare events
- You're doing information-theoretic analysis

**Use Theil's U instead when**:
- You need a normalized measure [0, 1]
- You want global predictive power (not pairwise)
- You're doing feature selection or variable importance analysis
- Uncertainty reduction is the right framework

**Use MLE instead when**:
- You need simple conditional probabilities
- Interpretability is paramount (probabilities are intuitive)
- Base rates don't matter for your question
- You're building a probabilistic model

**Use Goodman-Kruskal λ instead when**:
- Prediction accuracy is the goal
- You want to measure error reduction
- Classification performance matters

## Computational Considerations

### Implementation

```python
import asymcat
from asymcat.scorer import CatScorer

# Load data
data = asymcat.read_sequences("alignment_data.tsv")
cooccs = asymcat.collect_cooccs(data)

# Create scorer
scorer = CatScorer(cooccs, smoothing_method='laplace')

# Compute Tresoldi scores
tresoldi_scores = scorer.tresoldi()

# Access scores
for (x, y), (score_xy, score_yx) in tresoldi_scores.items():
    print(f"{x} → {y}: {score_xy:.3f}")
    print(f"{y} → {x}: {score_yx:.3f}")
```

### Smoothing Considerations

Because the Tresoldi measure depends on both PMI and MLE:

**PMI component**:
- Sensitive to zero probabilities (log(0) is undefined)
- Smoothing is **essential** for sparse data
- Laplace smoothing is recommended

**MLE component**:
- Less sensitive (division by zero only when P(X) = 0)
- But smoothing improves stability
- Helps with rare observations

**Recommended practice**:
```python
# For sparse data (typical in linguistic alignments)
scorer = CatScorer(cooccs, smoothing_method='laplace')

# For denser data
scorer = CatScorer(cooccs, smoothing_method='lidstone', smoothing_alpha=0.1)

# Avoid for Tresoldi (PMI requires smoothing)
# scorer = CatScorer(cooccs, smoothing_method='mle')  # Not recommended
```

### Computational Complexity

- **Time**: O(|alphabet_X| × |alphabet_Y| × |cooccs|)
- **Space**: O(|alphabet_X| × |alphabet_Y|) for caching
- **Caching**: Computed once and cached (lazy evaluation)
- **Dependency**: Requires computing both MLE and PMI first

### Numerical Stability

The implementation handles edge cases:

**Negative PMI**:
```python
if pmi < 0:
    tresoldi = -(abs(pmi) ** (1 - mle))
else:
    tresoldi = pmi ** (1 - mle)
```

This prevents complex number results from raising negative numbers to fractional powers.

**Extreme values**:
- Very high PMI values are compressed by the exponent
- Zero PMI yields zero Tresoldi (independence)
- The measure is numerically stable across the full range

## Interpretation Guide

### Score Magnitude

**Large positive values (> 3)**:
- Strong positive association
- High information content and high reliability
- X strongly predicts this specific Y
- Example: Regular grapheme-phoneme correspondences

**Moderate positive values (1-3)**:
- Meaningful positive association
- Balanced information and reliability
- X reliably predicts Y, but not dramatically
- Example: Common but not universal patterns

**Small positive values (0-1)**:
- Weak positive association
- Either low information or low reliability (or both)
- X and Y co-occur slightly more than expected
- Example: Marginal associations

**Values near zero (-0.5 to +0.5)**:
- Near independence
- X doesn't tell us much about Y
- Co-occurrence close to what's expected by chance

**Negative values (< 0)**:
- Negative association (inverse relationship)
- X and Y co-occur *less* than expected
- Example: Mutually exclusive categories

**Large negative values (< -3)**:
- Strong negative association
- X and Y rarely co-occur when they could
- Reliable *absence* of association

### Comparative Interpretation

**Comparing Tresoldi(X→Y) vs Tresoldi(Y→X)**:
- Reveals asymmetry in the association
- Large difference → strongly directional relationship
- Example: 'QU' → /kw/ is strong, but /kw/ → 'QU' is weak (many sources)

**Comparing across pairs**:
- Higher Tresoldi = stronger, more reliable association
- Can rank associations for alignment or matching
- Useful for rule extraction in linguistic analysis

**Relationship to PMI and MLE**:
- If Tresoldi ≈ PMI: Low conditional probability (exponent near 1)
- If Tresoldi ≈ 1: High conditional probability and positive PMI (exponent near 0)
- If Tresoldi << PMI: High PMI down-weighted by low reliability

## Examples

### Example 1: English Orthography-Phonology

Consider three grapheme-phoneme pairs:

**Pair A: 'C' → /k/**
- Observations: 'C' appears 100 times, /k/ appears 150 times, together 80 times
- P(/k/|C) = 80/100 = 0.80
- P(C, /k/) = 80/total
- PMI = log(P(C,/k/) / (P(C) · P(/k/))) ≈ 1.2
- Tresoldi = 1.2^(1-0.80) = 1.2^0.2 ≈ 1.04

**Pair B: 'GH' → /f/** (as in "rough")
- Observations: 'GH' appears 20 times, /f/ appears 200 times, together 15 times
- P(/f/|GH) = 15/20 = 0.75
- PMI ≈ 2.5 (high because 'GH' → /f/ is surprising given base rates)
- Tresoldi = 2.5^(1-0.75) = 2.5^0.25 ≈ 1.26

**Pair C: 'E' → /ə/** (schwa, most common)
- Observations: 'E' appears 500 times, /ə/ appears 800 times, together 400 times
- P(/ə/|E) = 400/500 = 0.80
- PMI ≈ 0.3 (low because both are common - expected co-occurrence)
- Tresoldi = 0.3^(1-0.80) = 0.3^0.2 ≈ 0.79

**Interpretation**:
- Pair B ('GH' → /f/) scores highest despite lower conditional probability
- It captures a linguistically important but rare pattern
- PMI alone would over-emphasize this; MLE alone would miss it
- Tresoldi balances both perspectives appropriately

### Example 2: Asymmetry in Action

**Forward: 'Q' → /k/**
- P(/k/|Q) ≈ 0.95 (Q almost always produces /k/)
- PMI ≈ 2.0 (Q is rare, so this is informative)
- Tresoldi ≈ 2.0^0.05 ≈ 1.04

**Reverse: /k/ → 'Q'**
- P(Q|/k/) ≈ 0.05 (/k/ comes from many sources: C, K, CK, QU, etc.)
- PMI ≈ 2.0 (same, PMI is symmetric)
- Tresoldi ≈ 2.0^0.95 ≈ 1.90

**Observation**:
- Same PMI, but very different Tresoldi scores
- Q strongly predicts /k/, but /k/ weakly predicts Q
- Asymmetry captured: the relationship is directional
- This is exactly what we want for alignment scoring

### Example 3: Negative Associations

**'GH' → /g/** (silent 'GH' vs. pronounced /g/)
- P(/g/|GH) ≈ 0.05 (usually silent or /f/)
- PMI ≈ -2.5 (negative association)
- Tresoldi = -(|-2.5|^(1-0.05)) = -(2.5^0.95) ≈ -2.38

**Interpretation**:
- Negative Tresoldi indicates 'GH' and /g/ avoid each other
- Linguistically meaningful: 'GH' rarely produces /g/ in modern English
- Magnitude preserved from PMI but slightly compressed
- Useful for identifying systematic gaps or constraints

## Theoretical Foundations

### Information-Theoretic Perspective

The Tresoldi measure can be understood through the lens of weighted information:

**Shannon Information**:
- I(y; x) = -log P(y|x)
- Quantifies "surprise" when Y=y given X=x

**PMI as Mutual Information**:
- PMI(x,y) = log(P(x,y) / (P(x)P(y)))
- Measures how much observing x and y together tells us beyond independence

**The Weighting**:
- Exponent (1 - P(y|x)) represents remaining uncertainty
- High P(y|x) → low uncertainty → compress information value
- Low P(y|x) → high uncertainty → preserve information value

**Interpretation**:
- Tresoldi modulates information content by predictive certainty
- It asks: "How informative is this association, given how deterministic it is?"

### Geometric Interpretation

The exponentiation creates a geometric relationship:

**For positive PMI**:
```
Tresoldi = PMI^α where α = 1 - P(Y|X)
```

This is equivalent to the weighted geometric mean:
```
log(Tresoldi) = α · log(PMI) + (1-α) · log(1)
              = α · log(PMI)
```

**The continuum**:
- α = 1 (P(Y|X) = 0): Tresoldi = PMI (pure information)
- α = 0 (P(Y|X) = 1): Tresoldi = 1 (pure determinism)
- Intermediate α: Smooth interpolation

This geometric mean interpretation explains why:
- Tresoldi is always between 1 and PMI (for PMI > 1)
- Tresoldi is always between PMI and 1 (for 0 < PMI < 1)
- The measure gracefully trades off the two components

### Relationship to Other Frameworks

**Connection to Mutual Information**:
- MI(X;Y) = Σ P(x,y) PMI(x,y)
- Tresoldi for a specific pair is a weighted PMI
- Could be aggregated similarly, though not currently implemented

**Connection to Bayesian Surprise**:
- Bayesian surprise measures information gain
- Tresoldi incorporates prior knowledge (base rates) via PMI
- The conditional probability weights by posterior certainty

**Connection to Association Rules**:
- In data mining: confidence = P(Y|X), support = P(X,Y)
- Tresoldi is like lift (PMI) weighted by confidence
- More sophisticated than either alone

## Limitations and Caveats

### What the Measure Doesn't Capture

1. **Sample Size**:
   - No explicit confidence intervals
   - Rare events can still be overweighted despite MLE component
   - Consider statistical testing separately for small samples

2. **Higher-Order Dependencies**:
   - Pairwise measure only
   - Doesn't capture X → Y → Z chains
   - Ignores context beyond the pair

3. **Causality**:
   - Association ≠ causation
   - Directionality (X→Y) is mathematical, not necessarily causal
   - Requires domain knowledge for causal interpretation

4. **Scale Dependence**:
   - Not normalized to [0, 1] like Theil's U
   - Difficult to compare across vastly different datasets
   - Magnitude depends on base rate distributions

### Potential Misinterpretations

**"High Tresoldi means X causes Y"**:
- No. High Tresoldi means strong, reliable association
- Causality requires experimental or temporal evidence
- Could be spurious correlation or common cause

**"Tresoldi(X→Y) > Tresoldi(Y→X) means X comes before Y"**:
- Not necessarily. Asymmetry indicates different predictive strengths
- Temporal order requires independent evidence
- Example: /k/ → 'C' is weaker, but sound doesn't "come from" letter

**"Negative Tresoldi is bad"**:
- No. Negative associations are meaningful
- Indicates systematic avoidance or mutual exclusivity
- Example: complementary distribution in phonology

**"Tresoldi combines the best of PMI and MLE"**:
- More accurate: combines different perspectives
- Trade-offs exist (not normalized, more complex interpretation)
- "Best" depends on research question

### Comparison Validity

**Cross-dataset comparisons**:
- Valid only if datasets have similar characteristics
- Base rate distributions affect scores significantly
- Normalize or use relative rankings within dataset

**Cross-measure comparisons**:
- Don't directly compare Tresoldi to Theil's U numerically
- Different ranges, different interpretations
- Use rank correlations or conceptual alignment

## Future Directions and Extensions

### Potential Variants

1. **Normalized Tresoldi**:
   - Scale to [0, 1] or [-1, 1]
   - Trade-off: lose information about magnitude
   - Gain: easier cross-dataset comparison

2. **Higher-Order Tresoldi**:
   - Extend to trigrams or longer sequences
   - Tresoldi(X₁X₂→Y) using P(Y|X₁,X₂)
   - Captures longer-range dependencies

3. **Weighted Exponents**:
   - Replace (1 - P(Y|X)) with other weighting functions
   - Could incorporate sample size, entropy, or other factors
   - Domain-specific calibration

4. **Multivariate Tresoldi**:
   - Joint associations: Tresoldi(X,Y→Z)
   - Network analysis applications
   - Computational complexity increases

### Research Applications

**Linguistics**:
- Alignment scoring (current primary use)
- Sound change detection
- Borrowing vs. inheritance discrimination
- Morphological boundary prediction

**Other Domains**:
- Bioinformatics: sequence alignments, motif discovery
- Ecology: species co-occurrence with abundance weighting
- Machine learning: feature interaction strength
- Network analysis: weighted edge strength

### Open Questions

1. **Optimal exponent function**:
   - Is (1 - P(Y|X)) the best weighting?
   - Could other functions better balance the trade-offs?
   - Domain-specific optimization?

2. **Statistical testing**:
   - What is the null distribution of Tresoldi?
   - How to construct confidence intervals?
   - Permutation tests? Bootstrapping?

3. **Aggregation**:
   - How to combine Tresoldi scores across multiple pairs?
   - Weighted average by frequency?
   - Maximum? Minimum? Harmonic mean?

4. **Relationship to other measures**:
   - Can Tresoldi be expressed in other mathematical frameworks?
   - Connection to copulas or dependence measures?
   - Information geometry interpretation?

## References and Further Reading

### Conceptual Foundations

- **Pointwise Mutual Information**:
  - Church, K. W., & Hanks, P. (1990). Word association norms, mutual information, and lexicography. *Computational Linguistics*, 16(1), 22-29.

- **Conditional Probability and MLE**:
  - Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.

- **Theil's Uncertainty Coefficient**:
  - Theil, H. (1970). On the estimation of relationships involving qualitative variables. *American Journal of Sociology*, 76(1), 103-154.

### ASymCat Documentation

- **User Guide**: `docs/USER_GUIDE.md` - Comprehensive usage examples
- **API Reference**: `docs/API_REFERENCE.md` - Technical API documentation
- **Tutorial 2**: `docs/tutorial_2_advanced_measures.py` - Measure comparisons
- **Tutorial 3**: `docs/tutorial_3_visualization.py` - Visualizing Tresoldi scores

### Implementation

The Tresoldi measure implementation can be found in:
- `asymcat/scorer.py`: Line 1014-1041 (`tresoldi()` method)

## Summary

The Tresoldi measure offers a unique perspective on asymmetric categorical associations by combining:

✓ **Information content** (via PMI) - captures surprising co-occurrences
✓ **Conditional reliability** (via MLE) - captures predictive strength
✓ **Directional asymmetry** - X→Y ≠ Y→X
✓ **Balanced weighting** - down-weights unreliable high-information pairs

**Use it when**:
- You need directional association strength
- Both informativeness and reliability matter
- Base rates significantly differ across categories
- You're analyzing linguistic alignments or similar structured data

**Key insight**: Strong associations should be both *surprising* (high PMI) and *reliable* (high conditional probability). The Tresoldi measure captures exactly this combination through its elegant mathematical formulation.
