#' # Tutorial 3: Visualization Techniques
#'
#' **Master data visualization** for asymmetric categorical associations.
#'
#' ## What You'll Learn
#'
#' 1. Creating heatmaps of association matrices
#' 2. Visualizing score distributions
#' 3. Comparing measures with multi-panel plots
#' 4. Matrix transformations (scaling, inversion)
#' 5. Publication-quality figure customization
#' 6. Interactive exploration patterns
#'
#' ## Prerequisites
#'
#' Complete Tutorials 1-2 to understand measures and their properties.

# Import required libraries
import asymcat
from asymcat.scorer import CatScorer
from asymcat import scorer as scorer_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("ASymCat Visualization Tutorial")
print("=" * 50)

#' ## 1. Basic Heatmap Visualization
#'
#' Heatmaps are ideal for visualizing association matrices, showing relationships
#' between all pairs of categorical values.

# Load sample data
data = asymcat.read_sequences("resources/cmudict.sample100.tsv")
cooccs = asymcat.collect_cooccs(data)

print(f"Loaded {len(data)} sequence pairs")
print(f"Total co-occurrences: {len(cooccs)}")

# Create scorer
scorer = CatScorer(cooccs, smoothing_method='laplace')

# Compute Theil's U
theil_scores = scorer.theil_u()

#' ### Converting Scores to Matrices
#'
#' The `scorer2matrices()` function converts score dictionaries into 2D matrices
#' suitable for visualization.

# Convert to matrices
xy_matrix, yx_matrix, x_labels, y_labels = scorer_utils.scorer2matrices(theil_scores)

print(f"\nMatrix shapes:")
print(f"  X→Y matrix: {xy_matrix.shape} ({len(x_labels)} × {len(y_labels)})")
print(f"  Y→X matrix: {yx_matrix.shape} ({len(y_labels)} × {len(x_labels)})")

#' ### Creating the Heatmap

# Select subset for better visualization
subset_size = 15
xy_subset = xy_matrix[:subset_size, :subset_size]
yx_subset = yx_matrix[:subset_size, :subset_size]
x_labels_subset = x_labels[:subset_size]
y_labels_subset = y_labels[:subset_size]

# X→Y heatmap
fig1, ax1 = plt.subplots(figsize=(10, 10))
sns.heatmap(xy_subset, annot=True, fmt='.2f', linewidths=0.5,
            cmap='YlOrRd', center=0.5, vmin=0, vmax=1,
            xticklabels=y_labels_subset, yticklabels=x_labels_subset,
            ax=ax1, cbar_kws={'label': "Theil's U"})
ax1.set_title("X → Y: U(Y|X)", fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel("Y (Predicted)", fontsize=12)
ax1.set_ylabel("X (Predictor)", fontsize=12)
plt.tight_layout()
plt.show()

# Y→X heatmap
fig2, ax2 = plt.subplots(figsize=(10, 10))
sns.heatmap(yx_subset, annot=True, fmt='.2f', linewidths=0.5,
            cmap='YlOrRd', center=0.5, vmin=0, vmax=1,
            xticklabels=x_labels_subset, yticklabels=y_labels_subset,
            ax=ax2, cbar_kws={'label': "Theil's U"})
ax2.set_title("Y → X: U(X|Y)", fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel("X (Predicted)", fontsize=12)
ax2.set_ylabel("Y (Predictor)", fontsize=12)
plt.tight_layout()
plt.show()

print("\n📊 Heatmaps reveal:")
print("  • Diagonal values: self-prediction (often high)")
print("  • Asymmetry: X→Y matrix ≠ Y→X matrix")
print("  • Hot spots: strongest associations")

#' ## 2. Asymmetry Visualization
#'
#' Directly visualize the asymmetry between X→Y and Y→X directions.

# Compute asymmetry matrix
asymmetry_matrix = np.abs(xy_subset - yx_subset.T)

fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(asymmetry_matrix, annot=True, fmt='.2f', linewidths=0.5,
            cmap='RdYlBu_r', center=0,
            xticklabels=y_labels_subset, yticklabels=x_labels_subset,
            ax=ax, cbar_kws={'label': 'Asymmetry |U(Y|X) - U(X|Y)|'})

ax.set_title("Asymmetry Matrix: |U(Y|X) - U(X|Y)|", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Y", fontsize=12)
ax.set_ylabel("X", fontsize=12)

plt.tight_layout()
plt.show()

print("\n🔍 High asymmetry (red) = strong directional effect")
print("   Low asymmetry (blue) = more symmetric relationship")

#' ## 3. Score Distribution Analysis
#'
#' Understanding the distribution of association scores helps interpret results.

# Compute multiple measures
measures = {
    'MLE': scorer.mle(),
    'PMI': scorer.pmi(),
    'Theil_U': scorer.theil_u(),
    'Chi2': scorer.chi2(),
    'Cramer_V': scorer.cramers_v(),
}

# Extract all scores
score_data = []
for measure_name, scores_dict in measures.items():
    for (x, y), (xy_score, yx_score) in scores_dict.items():
        score_data.append({
            'Measure': measure_name,
            'Direction': 'X→Y',
            'Score': xy_score
        })
        score_data.append({
            'Measure': measure_name,
            'Direction': 'Y→X',
            'Score': yx_score
        })

score_df = pd.DataFrame(score_data)

# Create violin plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, measure_name in enumerate(measures.keys()):
    if idx >= len(axes):
        break

    measure_data = score_df[score_df['Measure'] == measure_name]

    # Filter extreme values for better visualization
    if measure_name == 'Chi2':
        measure_data = measure_data[measure_data['Score'] < 100]
    elif measure_name == 'PMI':
        measure_data = measure_data[measure_data['Score'].abs() < 10]

    sns.violinplot(data=measure_data, x='Direction', y='Score',
                   palette='Set2', ax=axes[idx])

    axes[idx].set_title(f'{measure_name} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Direction')
    axes[idx].set_ylabel('Score Value')
    axes[idx].grid(True, alpha=0.3, axis='y')

# Remove empty subplot if any
if len(measures) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()

print("\n📈 Distribution insights:")
print("  • Shape: skewness indicates outliers")
print("  • Spread: variability in associations")
print("  • Comparison: X→Y vs Y→X differences")

#' ## 4. Scatter Plot Comparisons
#'
#' Compare two measures directly with scatter plots.

# Extract MLE and Theil's U scores
mle_vals = []
theil_vals = []

for pair in measures['MLE'].keys():
    if pair in measures['Theil_U']:
        mle_vals.append(measures['MLE'][pair][0])  # X→Y direction
        theil_vals.append(measures['Theil_U'][pair][0])

# Create scatter plot
fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(mle_vals, theil_vals, alpha=0.5, s=50, c='steelblue', edgecolors='black', linewidths=0.5)

# Add diagonal line
max_val = max(max(mle_vals), max(theil_vals))
ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')

ax.set_xlabel('MLE: P(Y|X)', fontsize=12)
ax.set_ylabel("Theil's U: U(Y|X)", fontsize=12)
ax.set_title("MLE vs. Theil's U Comparison", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)

# Add correlation coefficient
correlation = np.corrcoef(mle_vals, theil_vals)[0, 1]
ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print(f"\n🔗 MLE and Theil's U correlation: {correlation:.3f}")

#' ## 5. Matrix Transformations
#'
#' ASymCat provides utilities to transform score matrices for different analyses.

#' ### Scaling
#'
#' Scale scores to [0, 1] range for normalization.

# Compute PMI (unbounded)
pmi_scores = scorer.pmi()

print("\nOriginal PMI scores (sample):")
for i, (pair, (xy, yx)) in enumerate(list(pmi_scores.items())[:5]):
    print(f"  {pair}: X→Y = {xy:6.2f}, Y→X = {yx:6.2f}")

# Scale using minmax
scaled_pmi = scorer_utils.scale_scorer(pmi_scores, method='minmax')

print("\nScaled PMI scores [0, 1] (same pairs):")
for i, (pair, (xy, yx)) in enumerate(list(scaled_pmi.items())[:5]):
    print(f"  {pair}: X→Y = {xy:6.3f}, Y→X = {yx:6.3f}")

#' ### Inversion
#'
#' Invert scores to reverse their interpretation (high becomes low).

# Invert MLE scores
mle_scores = scorer.mle()
inverted_mle = scorer_utils.invert_scorer(mle_scores)

print("\nOriginal MLE (sample):")
sample_pairs = list(mle_scores.items())[:3]
for pair, (xy, yx) in sample_pairs:
    print(f"  {pair}: X→Y = {xy:.3f}")

print("\nInverted MLE (same pairs):")
for pair, (xy, yx) in sample_pairs:
    print(f"  {pair}: X→Y = {inverted_mle[pair][0]:.3f}")

#' ### Visualization of Transformations

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original PMI
xy_pmi, yx_pmi, x_lab, y_lab = scorer_utils.scorer2matrices(pmi_scores)
pmi_subset = xy_pmi[:10, :10]
sns.heatmap(pmi_subset, annot=False, cmap='RdBu_r', center=0,
            xticklabels=y_lab[:10], yticklabels=x_lab[:10],
            ax=axes[0], cbar_kws={'label': 'PMI'})
axes[0].set_title('Original PMI', fontsize=12, fontweight='bold')

# Scaled PMI
xy_scaled, yx_scaled, _, _ = scorer_utils.scorer2matrices(scaled_pmi)
scaled_subset = xy_scaled[:10, :10]
sns.heatmap(scaled_subset, annot=False, cmap='RdBu_r', center=0.5, vmin=0, vmax=1,
            xticklabels=y_lab[:10], yticklabels=x_lab[:10],
            ax=axes[1], cbar_kws={'label': 'Scaled PMI [0,1]'})
axes[1].set_title('Scaled PMI [0, 1]', fontsize=12, fontweight='bold')

# Inverted MLE
xy_inv, yx_inv, _, _ = scorer_utils.scorer2matrices(inverted_mle)
inv_subset = xy_inv[:10, :10]
sns.heatmap(inv_subset, annot=False, cmap='YlOrRd_r', vmin=0, vmax=1,
            xticklabels=y_lab[:10], yticklabels=x_lab[:10],
            ax=axes[2], cbar_kws={'label': 'Inverted MLE'})
axes[2].set_title('Inverted MLE', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n✨ Transformations enable:")
print("  • Normalization for comparison")
print("  • Reversing interpretation (distance vs. similarity)")
print("  • Standardization across measures")

#' ## 6. Multi-Measure Comparison Panel
#'
#' Create a comprehensive comparison of multiple measures side-by-side.

# Select 6 key measures
comparison_measures = {
    'MLE': scorer.mle(),
    'PMI': scorer.pmi(),
    'NPMI': scorer.pmi(normalized=True),
    'Theil_U': scorer.theil_u(),
    'Chi2': scorer.chi2(),
    'Cramer_V': scorer.cramers_v(),
}

# Create 2×3 grid
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for idx, (measure_name, scores) in enumerate(comparison_measures.items()):
    xy_mat, yx_mat, x_l, y_l = scorer_utils.scorer2matrices(scores)

    # Select subset
    mat_subset = xy_mat[:12, :12]
    x_subset = x_l[:12]
    y_subset = y_l[:12]

    # Determine colormap and normalization
    if measure_name in ['PMI']:
        cmap = 'RdBu_r'
        center = 0
        robust = True
    elif measure_name in ['NPMI']:
        cmap = 'RdBu_r'
        center = 0
        vmin, vmax = -1, 1
        robust = False
    else:
        cmap = 'YlOrRd'
        center = None
        robust = True

    # Plot heatmap
    if measure_name == 'NPMI':
        sns.heatmap(mat_subset, annot=False, cmap=cmap, center=center,
                    vmin=vmin, vmax=vmax, linewidths=0.1,
                    xticklabels=y_subset, yticklabels=x_subset,
                    ax=axes[idx], cbar_kws={'label': measure_name})
    else:
        sns.heatmap(mat_subset, annot=False, cmap=cmap, center=center,
                    robust=robust, linewidths=0.1,
                    xticklabels=y_subset, yticklabels=x_subset,
                    ax=axes[idx], cbar_kws={'label': measure_name})

    axes[idx].set_title(f'{measure_name} (X→Y)', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Y', fontsize=10)
    axes[idx].set_ylabel('X', fontsize=10)

plt.tight_layout()
plt.show()

print("\n🎨 Multi-panel comparison reveals:")
print("  • Different measures highlight different patterns")
print("  • PMI shows information content")
print("  • MLE shows predictive strength")
print("  • Theil's U balances both")

#' ## 7. Publication-Quality Figures
#'
#' Customize plots for publication with fine-grained control.

# Select specific measure
tresoldi_scores = scorer.tresoldi()
xy_tres, yx_tres, x_labels_tres, y_labels_tres = scorer_utils.scorer2matrices(tresoldi_scores)

# Create publication figure
fig = plt.figure(figsize=(14, 6))
gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Subset
subset = 12
xy_sub = xy_tres[:subset, :subset]
yx_sub = yx_tres[:subset, :subset]
x_sub = x_labels_tres[:subset]
y_sub = y_labels_tres[:subset]

# Left panel: X→Y
im1 = ax1.imshow(xy_sub, cmap='viridis', aspect='auto', interpolation='nearest')
ax1.set_xticks(np.arange(len(y_sub)))
ax1.set_yticks(np.arange(len(x_sub)))
ax1.set_xticklabels(y_sub, fontsize=9)
ax1.set_yticklabels(x_sub, fontsize=9)
ax1.set_xlabel('Y (Phoneme)', fontsize=11, fontweight='bold')
ax1.set_ylabel('X (Grapheme)', fontsize=11, fontweight='bold')
ax1.set_title('(A) Tresoldi Measure: X → Y', fontsize=12, fontweight='bold', loc='left')
plt.colorbar(im1, ax=ax1, label='Tresoldi Score')

# Right panel: Y→X
im2 = ax2.imshow(yx_sub, cmap='viridis', aspect='auto', interpolation='nearest')
ax2.set_xticks(np.arange(len(x_sub)))
ax2.set_yticks(np.arange(len(y_sub)))
ax2.set_xticklabels(x_sub, fontsize=9)
ax2.set_yticklabels(y_sub, fontsize=9)
ax2.set_xlabel('X (Grapheme)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Y (Phoneme)', fontsize=11, fontweight='bold')
ax2.set_title('(B) Tresoldi Measure: Y → X', fontsize=12, fontweight='bold', loc='left')
plt.colorbar(im2, ax=ax2, label='Tresoldi Score')

plt.suptitle('Asymmetric Grapheme-Phoneme Associations (Tresoldi Measure)',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.show()

print("\n📄 Publication tips:")
print("  • Use panel labels (A, B, C)")
print("  • Consistent color schemes")
print("  • Clear axis labels and titles")
print("  • Appropriate font sizes (9-12pt)")
print("  • Informative colorbars")

#' ## 8. Key Takeaways
#'
#' **You've learned:**
#'
#' ✓ Creating heatmaps with `scorer2matrices()`
#' ✓ Visualizing asymmetry directly
#' ✓ Score distribution analysis with violin plots
#' ✓ Scatter plots for measure comparison
#' ✓ Matrix transformations (scaling, inversion)
#' ✓ Multi-panel comparison layouts
#' ✓ Publication-quality figure customization
#'
#' ## Next Steps
#'
#' Continue to **Tutorial 4: Real-World Applications** to see:
#' - Complete analysis workflows
#' - Linguistics: grapheme-phoneme correspondence
#' - Ecology: species co-occurrence patterns
#' - Machine learning: feature selection
#' - Interpretation best practices
#'
#' ## Quick Reference
#'
#' ```python
#' # Convert scores to matrices
#' xy, yx, x_labels, y_labels = scorer_utils.scorer2matrices(scores)
#'
#' # Heatmap
#' sns.heatmap(xy[:10, :10], annot=True, cmap='YlOrRd')
#'
#' # Scale scores
#' scaled = scorer_utils.scale_scorer(scores, method='minmax')
#'
#' # Invert scores
#' inverted = scorer_utils.invert_scorer(scores)
#'
#' # Asymmetry matrix
#' asymmetry = np.abs(xy - yx.T)
#' ```

print("\n" + "="*60)
print("Tutorial 3 Complete! ✓")
print("="*60)
print("\nYou can now create compelling visualizations of asymmetric associations!")
print("Proceed to Tutorial 4 for real-world application examples.")
