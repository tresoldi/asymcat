#' # Tutorial 2: Advanced Association Measures
#'
#' **Build on the basics** and explore the full range of ASymCat's association measures.
#'
#' ## What You'll Learn
#'
#' 1. Information-theoretic measures (PMI, NPMI, Mutual Information, Conditional Entropy)
#' 2. Statistical measures (Chi-square, Cramér's V, Fisher exact test)
#' 3. Specialized measures (Theil's U, Goodman-Kruskal λ, Tresoldi)
#' 4. Smoothing methods and their effects
#' 5. Measure selection for different tasks
#' 6. Comparative analysis
#'
#' ## Prerequisites
#'
#' Complete Tutorial 1 first to understand basic workflow and MLE.

# Import required libraries
import asymcat
from asymcat.scorer import CatScorer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 7)

print("ASymCat Advanced Measures Tutorial")
print("=" * 50)

#' ## 1. Review: The Measure Categories
#'
#' ASymCat provides 17+ measures organized into four categories:
#'
#' ### Probabilistic Measures
#' - **MLE**: Maximum likelihood estimation P(Y|X)
#' - **Jaccard**: Set overlap similarity
#' - **Goodman-Kruskal λ**: Prediction error reduction
#'
#' ### Information-Theoretic Measures
#' - **PMI**: Pointwise mutual information
#' - **NPMI**: Normalized PMI
#' - **Mutual Information**: Average information shared
#' - **Conditional Entropy**: Uncertainty remaining
#' - **Theil's U**: Uncertainty coefficient
#'
#' ### Statistical Measures
#' - **Chi-square**: Independence test statistic
#' - **Cramér's V**: Normalized chi-square
#' - **Fisher Exact**: Exact odds ratios
#' - **Log-Likelihood**: G² statistic
#'
#' ### Specialized Measures
#' - **Tresoldi**: Custom measure combining MLE and PMI

#' ## 2. Load Sample Data
#'
#' We'll use English orthography-to-pronunciation data to demonstrate the measures.

# Load data
data = asymcat.read_sequences("resources/english_phonology.tsv")
cooccs = asymcat.collect_cooccs(data)

print(f"Loaded {len(data)} sequence pairs")
print(f"Total co-occurrences: {len(cooccs)}")

# Create scorer with Laplace smoothing
scorer = CatScorer(cooccs, smoothing_method='laplace', smoothing_alpha=1.0)

print("Scorer ready with Laplace smoothing")

#' ## 3. Information-Theoretic Measures
#'
#' These measures quantify the **information content** of associations.

#' ### 3.1 PMI (Pointwise Mutual Information)
#'
#' PMI measures how much information we gain about Y when we observe X:
#'
#' PMI(X,Y) = log₂[P(X,Y) / (P(X)·P(Y))]
#'
#' - PMI > 0: X and Y co-occur more than expected (positive association)
#' - PMI = 0: X and Y are independent
#' - PMI < 0: X and Y co-occur less than expected (negative association)

# Compute PMI
pmi_scores = scorer.pmi()

print("\nPMI (Pointwise Mutual Information):")
print("=" * 60)

# Sort by PMI value and show top associations
sorted_pmi = sorted(pmi_scores.items(), key=lambda x: x[1][0], reverse=True)

print("Top 10 strongest associations (highest PMI):")
for i, ((x, y), (pmi_val, _)) in enumerate(sorted_pmi[:10], 1):
    print(f"{i:2d}. ({x:3s}, {y:3s}): PMI = {pmi_val:6.3f}")

#' ### 3.2 NPMI (Normalized PMI)
#'
#' NPMI normalizes PMI to the range [-1, 1] for easier interpretation:
#'
#' NPMI(X,Y) = PMI(X,Y) / -log₂[P(X,Y)]
#'
#' - NPMI = 1: Perfect co-occurrence (always together)
#' - NPMI = 0: Independent
#' - NPMI = -1: Never co-occur

# Compute NPMI
npmi_scores = scorer.pmi(normalized=True)

print("\n\nNPMI (Normalized PMI):")
print("=" * 60)

sorted_npmi = sorted(npmi_scores.items(), key=lambda x: x[1][0], reverse=True)

print("Top 10 strongest normalized associations:")
for i, ((x, y), (npmi_val, _)) in enumerate(sorted_npmi[:10], 1):
    print(f"{i:2d}. ({x:3s}, {y:3s}): NPMI = {npmi_val:6.3f}")

#' ### 3.3 Theil's U (Uncertainty Coefficient)
#'
#' Theil's U measures uncertainty reduction - how much knowing X reduces
#' uncertainty about Y:
#'
#' U(Y|X) = [H(Y) - H(Y|X)] / H(Y)
#'
#' - U = 0: X tells us nothing about Y
#' - U = 1: X completely determines Y
#' - **Asymmetric**: U(Y|X) ≠ U(X|Y)

# Compute Theil's U
theil_scores = scorer.theil_u()

print("\n\nTheil's U (Uncertainty Coefficient):")
print("=" * 60)

# Show asymmetric nature
sorted_theil = sorted(theil_scores.items(), key=lambda x: x[1][0], reverse=True)

print("Top 10 predictive relationships (highest U(Y|X)):")
for i, ((x, y), (u_y_x, u_x_y)) in enumerate(sorted_theil[:10], 1):
    asymmetry = abs(u_y_x - u_x_y)
    print(f"{i:2d}. {x:3s}→{y:3s}: U={u_y_x:.3f}  |  {y:3s}→{x:3s}: U={u_x_y:.3f}  |  Δ={asymmetry:.3f}")

#' ## 4. Statistical Measures
#'
#' These measures test statistical significance and independence.

#' ### 4.1 Chi-Square Test
#'
#' Chi-square measures deviation from independence:
#'
#' χ²(X,Y) = Σ [(observed - expected)² / expected]
#'
#' Higher values indicate stronger associations.

# Compute chi-square
chi2_scores = scorer.chi2()

print("\n\nChi-Square Statistic:")
print("=" * 60)

sorted_chi2 = sorted(chi2_scores.items(), key=lambda x: x[1][0], reverse=True)

print("Top 10 strongest associations (highest χ²):")
for i, ((x, y), (chi2_val, _)) in enumerate(sorted_chi2[:10], 1):
    print(f"{i:2d}. ({x:3s}, {y:3s}): χ² = {chi2_val:8.2f}")

#' ### 4.2 Cramér's V
#'
#' Cramér's V normalizes chi-square to [0, 1] range:
#'
#' V = √[χ² / (N × min(rows-1, cols-1))]
#'
#' - V = 0: No association
#' - V = 1: Perfect association

# Compute Cramér's V
cramer_scores = scorer.cramers_v()

print("\n\nCramér's V (Normalized Chi-Square):")
print("=" * 60)

sorted_cramer = sorted(cramer_scores.items(), key=lambda x: x[1][0], reverse=True)

print("Top 10 strongest associations:")
for i, ((x, y), (v_val, _)) in enumerate(sorted_cramer[:10], 1):
    print(f"{i:2d}. ({x:3s}, {y:3s}): V = {v_val:6.3f}")

#' ### 4.3 Fisher Exact Test
#'
#' Fisher's exact test computes exact odds ratios, useful for small samples:
#'
#' OR = [P(Y|X) / P(¬Y|X)] / [P(Y|¬X) / P(¬Y|¬X)]

# Compute Fisher exact test
fisher_scores = scorer.fisher()

print("\n\nFisher Exact Test (Odds Ratios):")
print("=" * 60)

sorted_fisher = sorted(fisher_scores.items(), key=lambda x: x[1][0], reverse=True)

print("Top 10 strongest associations:")
for i, ((x, y), (or_xy, or_yx)) in enumerate(sorted_fisher[:10], 1):
    print(f"{i:2d}. ({x:3s}, {y:3s}): OR(Y|X) = {or_xy:8.2f}, OR(X|Y) = {or_yx:8.2f}")

#' ## 5. Specialized Measures
#'
#' These measures are designed for specific types of analysis.

#' ### 5.1 Goodman-Kruskal λ (Lambda)
#'
#' Lambda measures **prediction error reduction** - how much better we can predict
#' Y when we know X compared to guessing without X:
#'
#' λ(Y|X) = [E₀ - E₁] / E₀
#'
#' - λ = 0: Knowing X doesn't help predict Y
#' - λ = 1: Knowing X perfectly predicts Y
#' - **Asymmetric**: λ(Y|X) ≠ λ(X|Y)

# Compute Goodman-Kruskal lambda
lambda_scores = scorer.goodman_kruskal_lambda()

print("\n\nGoodman-Kruskal λ (Prediction Error Reduction):")
print("=" * 60)

sorted_lambda = sorted(lambda_scores.items(), key=lambda x: x[1][0], reverse=True)

print("Top 10 predictive relationships (highest λ(Y|X)):")
for i, ((x, y), (lambda_yx, lambda_xy)) in enumerate(sorted_lambda[:10], 1):
    asymmetry = abs(lambda_yx - lambda_xy)
    print(f"{i:2d}. {x:3s}→{y:3s}: λ={lambda_yx:.3f}  |  {y:3s}→{x:3s}: λ={lambda_xy:.3f}  |  Δ={asymmetry:.3f}")

#' ### 5.2 Tresoldi Measure
#'
#' The Tresoldi measure is a **custom measure for linguistic alignments** that
#' combines conditional probability (MLE) and information content (PMI):
#'
#' Tresoldi(X,Y) = MLE(Y|X) × PMI(X,Y)
#'
#' This balances predictiveness (how often X → Y) with informativeness (how
#' surprising the association is).

# Compute Tresoldi measure
tresoldi_scores = scorer.tresoldi()

print("\n\nTresoldi Measure (MLE × PMI):")
print("=" * 60)

sorted_tresoldi = sorted(tresoldi_scores.items(), key=lambda x: x[1][0], reverse=True)

print("Top 10 strongest associations:")
for i, ((x, y), (tres_xy, tres_yx)) in enumerate(sorted_tresoldi[:10], 1):
    print(f"{i:2d}. ({x:3s}, {y:3s}): Tresoldi(Y|X) = {tres_xy:6.3f}, Tresoldi(X|Y) = {tres_yx:6.3f}")

#' ## 6. Comparing Measures
#'
#' Different measures reveal different aspects of associations. Let's compare them.

# Select a few representative pairs
sample_pairs = list(cooccs)[:20]

# Compute all measures for comparison
measures_dict = {
    'MLE': scorer.mle(),
    'PMI': scorer.pmi(),
    'NPMI': scorer.pmi(normalized=True),
    'Theil_U': scorer.theil_u(),
    'Chi2': scorer.chi2(),
    'Cramer_V': scorer.cramers_v(),
    'Lambda': lambda_scores,
    'Tresoldi': tresoldi_scores,
}

# Create comparison dataframe
comparison_data = []
for pair in sample_pairs:
    if pair in measures_dict['MLE']:
        x, y = pair
        row = {
            'X': x,
            'Y': y,
            'MLE': measures_dict['MLE'][pair][0],
            'PMI': measures_dict['PMI'][pair][0],
            'NPMI': measures_dict['NPMI'][pair][0],
            'Theil_U': measures_dict['Theil_U'][pair][0],
            'Chi2': measures_dict['Chi2'][pair][0],
            'Cramer_V': measures_dict['Cramer_V'][pair][0],
            'Lambda': measures_dict['Lambda'][pair][0],
            'Tresoldi': measures_dict['Tresoldi'][pair][0],
        }
        comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)

print("\n\nMeasure Comparison (First 10 pairs):")
print("=" * 100)
print(comparison_df.head(10).to_string(index=False, float_format=lambda x: f'{x:.3f}'))

#' ### Correlation Between Measures
#'
#' Let's visualize how different measures correlate with each other.

# Compute correlations
corr_matrix = comparison_df[['MLE', 'PMI', 'NPMI', 'Theil_U', 'Chi2', 'Cramer_V', 'Lambda', 'Tresoldi']].corr()

# Plot correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, vmin=-1, vmax=1, square=True, ax=ax,
            cbar_kws={'label': 'Correlation'}, annot_kws={'size': 9})
ax.set_title('Correlation Between Association Measures', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("\n📊 Correlation insights:")
print("  • High correlation = measures capture similar information")
print("  • Low correlation = measures reveal different aspects")

#' ## 6. Smoothing Methods Deep Dive
#'
#' Smoothing is crucial for handling sparse data. Let's compare methods systematically.

# Create a small sparse dataset
sparse_data = [
    ('A', 'X'), ('A', 'X'), ('A', 'X'), ('A', 'X'), ('A', 'X'),
    ('A', 'Y'),
    ('B', 'X'),
    ('C', 'Z'), ('C', 'Z'),
]

print("\n\nSmoothing Methods Comparison:")
print("=" * 60)
print(f"Sparse dataset: {sparse_data}")

# Create scorers with different smoothing
smoothing_configs = [
    ('MLE (no smoothing)', 'mle', None),
    ('Laplace (α=1.0)', 'laplace', 1.0),
    ('Lidstone (α=0.1)', 'lidstone', 0.1),
    ('Lidstone (α=0.5)', 'lidstone', 0.5),
    ('Lidstone (α=2.0)', 'lidstone', 2.0),
]

smoothing_results = []
for name, method, alpha in smoothing_configs:
    if alpha is None:
        s = CatScorer(sparse_data, smoothing_method=method)
    else:
        s = CatScorer(sparse_data, smoothing_method=method, smoothing_alpha=alpha)

    mle = s.mle()
    pmi = s.pmi()

    # Get P(X|A)
    if ('A', 'X') in mle:
        p_x_a = mle[('A', 'X')][0]
        pmi_val = pmi[('A', 'X')][0]
    else:
        p_x_a = 0.0
        pmi_val = 0.0

    smoothing_results.append({
        'Method': name,
        'P(X|A)': f'{p_x_a:.4f}',
        'PMI(A,X)': f'{pmi_val:.3f}',
    })

smoothing_df = pd.DataFrame(smoothing_results)
print("\n")
print(smoothing_df.to_string(index=False))

print("\n📌 Smoothing Trade-offs:")
print("  • MLE: Highest precision, but unstable for rare events")
print("  • Laplace: Conservative, adds equal pseudo-counts")
print("  • Lidstone: Tunable via α - lower α = less smoothing")

#' ## 7. Measure Selection Guide
#'
#' Choosing the right measure depends on your task and data characteristics.

print("\n\n" + "=" * 70)
print("MEASURE SELECTION DECISION TREE")
print("=" * 70)

decision_tree = """
┌─────────────────────────────────────┐
│ What is your primary goal?         │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       │                │
   PREDICTION       DISCOVERY
       │                │
       ▼                ▼
   ┌───────┐      ┌──────────┐
   │  MLE  │      │   PMI    │  ← Information content
   └───────┘      │   NPMI   │  ← Normalized [-1,1]
       │          │  Chi²    │  ← Statistical significance
       │          └──────────┘
       ▼
 ┌─────────────┐
 │ Theil's U   │  ← Uncertainty reduction
 │ Lambda (λ)  │  ← Error reduction
 └─────────────┘

┌─────────────────────────────────────┐
│ Data characteristics?               │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       │                │
    SPARSE           DENSE
       │                │
       ▼                ▼
  Use smoothing    Any measure
  (Laplace/        works well
   Lidstone)
"""

print(decision_tree)

#' ### Detailed Recommendations
#'
#' | Task | Recommended Measure | Why |
#' |------|-------------------|-----|
#' | **Predictive modeling** | MLE, Theil's U | Direct conditional probabilities |
#' | **Feature selection** | Chi², Cramér's V | Statistical significance |
#' | **Information content** | PMI, NPMI | Quantifies surprise/information |
#' | **Rare events** | Fisher, MLE+smoothing | Exact tests for small samples |
#' | **Linguistic alignment** | Tresoldi | Custom measure for sequences |
#' | **Symmetric associations** | Jaccard, PMI | Context overlap |

#' ## 8. Advanced: Custom Measure Combination
#'
#' You can combine multiple measures for richer analysis.

# Combine MLE and PMI for a balanced view
combined_scores = {}
mle = scorer.mle()
pmi = scorer.pmi()

for pair in mle.keys():
    if pair in pmi:
        # Combine: high MLE (predictive) + high PMI (informative)
        mle_xy = mle[pair][0]
        pmi_xy = pmi[pair][0]

        # Geometric mean of normalized values
        combined_xy = np.sqrt(max(0, mle_xy) * max(0, pmi_xy))
        combined_scores[pair] = combined_xy

# Sort by combined score
sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

print("\n\nCombined Score (MLE × PMI):")
print("=" * 60)
print("Top 10 pairs balancing predictiveness and informativeness:")

for i, ((x, y), score) in enumerate(sorted_combined[:10], 1):
    mle_val = mle[(x, y)][0]
    pmi_val = pmi[(x, y)][0]
    print(f"{i:2d}. ({x:3s}, {y:3s}): Combined={score:.3f}  [MLE={mle_val:.3f}, PMI={pmi_val:.3f}]")

#' ## 9. Key Takeaways
#'
#' **You've learned:**
#'
#' ✓ Information-theoretic measures (PMI, NPMI, Theil's U) quantify information content
#' ✓ Statistical measures (Chi², Cramér's V, Fisher) test significance
#' ✓ Different measures reveal different aspects of associations
#' ✓ Smoothing methods handle sparse data (Laplace, Lidstone)
#' ✓ Measure selection depends on task (prediction vs. discovery)
#' ✓ Measures can be combined for richer analysis
#'
#' ## Next Steps
#'
#' Continue to **Tutorial 3: Visualization** to learn:
#' - Creating heatmaps of association matrices
#' - Visualizing score distributions
#' - Comparative measure plots
#' - Publication-quality figures
#'
#' ## Quick Reference
#'
#' ```python
#' scorer = CatScorer(cooccs, smoothing_method='laplace')
#'
#' # Information-theoretic
#' pmi = scorer.pmi()               # Pointwise mutual information
#' npmi = scorer.pmi(normalized=True)  # Normalized PMI [-1,1]
#' theil = scorer.theil_u()         # Uncertainty coefficient
#' mi = scorer.mutual_information() # Average MI
#'
#' # Statistical
#' chi2 = scorer.chi2()             # Chi-square test
#' cramer = scorer.cramers_v()      # Normalized chi-square
#' fisher = scorer.fisher()         # Exact odds ratios
#'
#' # Specialized
#' tresoldi = scorer.tresoldi()     # MLE × PMI combined
#' lambda_scores = scorer.goodman_kruskal_lambda()  # Error reduction
#' ```

print("\n" + "="*60)
print("Tutorial 2 Complete! ✓")
print("="*60)
print("\nYou now understand the full spectrum of association measures!")
print("Proceed to Tutorial 3 to master visualization techniques.")
