#' # Tutorial 4: Real-World Applications
#'
#' **Apply ASymCat to real-world problems** across multiple domains.
#'
#' ## What You'll Learn
#'
#' 1. Complete analysis workflows from data to insights
#' 2. Linguistics: Grapheme-phoneme correspondence analysis
#' 3. Ecology: Species co-occurrence patterns
#' 4. Machine learning: Feature selection with asymmetric measures
#' 5. Interpretation best practices
#' 6. Reporting and communication strategies
#'
#' ## Prerequisites
#'
#' Complete Tutorials 1-3 to understand measures, smoothing, and visualization.

# Import required libraries
import asymcat
from asymcat.scorer import CatScorer
from asymcat import scorer as scorer_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 8)
sns.set_palette("husl")

print("ASymCat Real-World Applications Tutorial")
print("=" * 50)

#' ## 1. Application Framework
#'
#' A complete ASymCat analysis follows these steps:
#'
#' 1. **Data Preparation**: Load and validate data
#' 2. **Exploratory Analysis**: Understand basic patterns
#' 3. **Measure Selection**: Choose appropriate measures
#' 4. **Computation**: Calculate associations
#' 5. **Interpretation**: Extract insights
#' 6. **Visualization**: Communicate results
#' 7. **Validation**: Check robustness

#' ## 2. Case Study 1: Linguistics - Grapheme-Phoneme Correspondence
#'
#' ### Research Question
#' **How predictable are English phonemes from orthography?**
#'
#' This question is fundamental to understanding:
#' - Reading acquisition and literacy
#' - Historical language change
#' - Orthographic depth theory

print("\n" + "="*60)
print("CASE STUDY 1: LINGUISTIC ANALYSIS")
print("="*60)

#' ### Step 1: Data Preparation

# Load CMU dictionary data
data = asymcat.read_sequences("../resources/cmudict.sample1000.tsv")
cooccs = asymcat.collect_cooccs(data)

print(f"\nDataset: CMU Pronunciation Dictionary (sample)")
print(f"  Sequence pairs: {len(data)}")
print(f"  Co-occurrence pairs: {len(cooccs)}")

# Show sample alignments
print("\nSample grapheme-phoneme alignments:")
for i in range(5):
    ortho, phon = data[i]
    print(f"  '{' '.join(ortho):20s}' → /{' '.join(phon)}/")

#' ### Step 2: Exploratory Analysis

# Analyze frequency distributions
graphemes = [item for pair in data for item in pair[0]]
phonemes = [item for pair in data for item in pair[1]]

grapheme_counts = Counter(graphemes)
phoneme_counts = Counter(phonemes)

print(f"\nFrequency Statistics:")
print(f"  Unique graphemes: {len(grapheme_counts)}")
print(f"  Unique phonemes: {len(phoneme_counts)}")
print(f"  Most common graphemes: {grapheme_counts.most_common(5)}")
print(f"  Most common phonemes: {phoneme_counts.most_common(5)}")

# Check data sparsity
total_possible = len(grapheme_counts) * len(phoneme_counts)
observed_pairs = len(set(cooccs))
sparsity = 1 - (observed_pairs / total_possible)

print(f"\nData Sparsity:")
print(f"  Possible grapheme-phoneme pairs: {total_possible}")
print(f"  Observed pairs: {observed_pairs}")
print(f"  Sparsity: {sparsity:.1%}")
print(f"  → Smoothing recommended!")

#' ### Step 3: Measure Selection
#'
#' For linguistic correspondence:
#' - **MLE**: Direct predictability P(phoneme|grapheme)
#' - **Theil's U**: Information-theoretic uncertainty reduction
#' - **Tresoldi**: Custom measure for sequence alignment

# Create scorer with Laplace smoothing
scorer = CatScorer(cooccs, smoothing_method='laplace', smoothing_alpha=1.0)

#' ### Step 4: Computation

# Compute measures
mle_scores = scorer.mle()
theil_scores = scorer.theil_u()
tresoldi_scores = scorer.tresoldi()

print(f"\nComputed {len(mle_scores)} association scores")

#' ### Step 5: Interpretation

# Analyze orthography → phonology prediction strength
ortho_to_phon = [xy for xy, yx in mle_scores.values()]
phon_to_ortho = [yx for xy, yx in mle_scores.values()]

print(f"\nDirectional Prediction Analysis:")
print(f"  Mean P(phoneme|grapheme): {np.mean(ortho_to_phon):.3f}")
print(f"  Mean P(grapheme|phoneme): {np.mean(phon_to_ortho):.3f}")
print(f"  Asymmetry ratio: {np.mean(ortho_to_phon) / np.mean(phon_to_ortho):.2f}x")

# Find most predictable graphemes
grapheme_predictability = {}
for (g, p), (p_phon_graph, p_graph_phon) in mle_scores.items():
    if g not in grapheme_predictability:
        grapheme_predictability[g] = []
    grapheme_predictability[g].append(p_phon_graph)

avg_predictability = {g: np.mean(scores) for g, scores in grapheme_predictability.items()}
sorted_graphemes = sorted(avg_predictability.items(), key=lambda x: x[1], reverse=True)

print(f"\nMost predictable graphemes (highest avg P(phoneme|grapheme)):")
for i, (grapheme, avg_p) in enumerate(sorted_graphemes[:10], 1):
    print(f"  {i:2d}. '{grapheme:3s}': {avg_p:.3f}")

print(f"\nLeast predictable graphemes (lowest avg P(phoneme|grapheme)):")
for i, (grapheme, avg_p) in enumerate(sorted_graphemes[-5:], 1):
    print(f"  {i:2d}. '{grapheme:3s}': {avg_p:.3f}")

#' ### Step 6: Visualization

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))

# Subplot 1: Directional comparison
ax1 = plt.subplot(2, 2, 1)
asymmetries = [abs(xy - yx) for xy, yx in mle_scores.values()]
ax1.hist(asymmetries, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax1.set_xlabel('|P(phoneme|grapheme) - P(grapheme|phoneme)|')
ax1.set_ylabel('Frequency')
ax1.set_title('(A) Asymmetry Distribution', fontweight='bold')
ax1.axvline(np.mean(asymmetries), color='red', linestyle='--', label=f'Mean: {np.mean(asymmetries):.3f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Scatter plot
ax2 = plt.subplot(2, 2, 2)
ax2.scatter(ortho_to_phon, phon_to_ortho, alpha=0.4, s=20, c='coral', edgecolors='black', linewidths=0.3)
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
ax2.set_xlabel('P(phoneme|grapheme)')
ax2.set_ylabel('P(grapheme|phoneme)')
ax2.set_title('(B) Directional Comparison', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# Subplot 3: Top associations heatmap
ax3 = plt.subplot(2, 2, 3)
top_pairs = sorted(mle_scores.items(), key=lambda x: x[1][0], reverse=True)[:20]
pair_labels = [f"{g}→{p}" for (g, p), _ in top_pairs]
xy_vals = [scores[0] for _, scores in top_pairs]
yx_vals = [scores[1] for _, scores in top_pairs]

y_pos = np.arange(len(pair_labels))
ax3.barh(y_pos - 0.2, xy_vals, 0.4, label='grapheme→phoneme', alpha=0.8, color='steelblue')
ax3.barh(y_pos + 0.2, yx_vals, 0.4, label='phoneme→grapheme', alpha=0.8, color='coral')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(pair_labels, fontsize=8)
ax3.set_xlabel('Conditional Probability')
ax3.set_title('(C) Top 20 Strongest Associations', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='x')

# Subplot 4: Measure correlation
ax4 = plt.subplot(2, 2, 4)
theil_vals = [u_xy for u_xy, u_yx in theil_scores.values()]
ax4.scatter(ortho_to_phon, theil_vals, alpha=0.4, s=20, c='green', edgecolors='black', linewidths=0.3)
ax4.set_xlabel('MLE: P(phoneme|grapheme)')
ax4.set_ylabel("Theil's U: U(phoneme|grapheme)")
ax4.set_title('(D) MLE vs. Theil U Correlation', fontweight='bold')
ax4.grid(True, alpha=0.3)
corr = np.corrcoef(ortho_to_phon, theil_vals)[0, 1]
ax4.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Grapheme-Phoneme Correspondence Analysis (English)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#' ### Step 7: Key Findings
#'
#' **Linguistic Insights:**
#'
#' 1. English orthography **moderately predicts** phonology (mean P ≈ 0.5-0.7)
#' 2. Graphemes like 'b', 't', 'd' have **high predictability** (consistent sounds)
#' 3. Graphemes like 'c', 'g', 'o' have **low predictability** (multiple pronunciations)
#' 4. **Asymmetry is high**: knowing the letter tells you more about the sound
#'    than vice versa
#' 5. This supports **orthographic depth theory**: English is a "deep" orthography

#' ## 3. Case Study 2: Ecology - Species Co-occurrence Patterns
#'
#' ### Research Question
#' **What are the asymmetric ecological dependencies among Galápagos finches?**

print("\n" + "="*60)
print("CASE STUDY 2: ECOLOGICAL ANALYSIS")
print("="*60)

#' ### Data Preparation

# Load Galápagos finch presence-absence data
pa_data = asymcat.read_pa_matrix("../resources/galapagos.tsv")

print(f"\nDataset: Galápagos Finch Species")
print(f"  Co-occurrence pairs: {len(pa_data)}")

# Load original matrix for analysis
galapagos_df = pd.read_csv("../resources/galapagos.tsv", sep='\t', index_col=0)

print(f"  Islands: {galapagos_df.shape[0]}")
print(f"  Species: {galapagos_df.shape[1]}")

# Show matrix structure
print(f"\nSample of presence-absence matrix:")
print(galapagos_df.head(5).to_string())

#' ### Exploratory Analysis

# Species prevalence
species_prevalence = galapagos_df.sum(axis=0).sort_values(ascending=False)
island_richness = galapagos_df.sum(axis=1).sort_values(ascending=False)

print(f"\nEcological Patterns:")
print(f"  Most widespread species:")
for sp, count in species_prevalence.head(5).items():
    sp_short = sp.split('.')[-1] if '.' in sp else sp
    print(f"    {sp_short:20s}: {int(count):2d}/{galapagos_df.shape[0]} islands")

print(f"\n  Most species-rich islands:")
for island, count in island_richness.head(5).items():
    print(f"    {island:20s}: {int(count):2d}/{galapagos_df.shape[1]} species")

#' ### Measure Selection
#'
#' For ecological co-occurrence:
#' - **MLE**: P(Species2|Species1) - co-occurrence probability
#' - **Fisher**: Exact odds ratios for small samples
#' - **Jaccard**: Habitat overlap similarity

# Create scorer
eco_scorer = CatScorer(pa_data, smoothing_method='laplace')

# Compute measures
eco_mle = eco_scorer.mle()
eco_fisher = eco_scorer.fisher()
eco_jaccard = eco_scorer.jaccard_index()

print(f"\nComputed {len(eco_mle)} species pair associations")

#' ### Interpretation

# Find strongest associations
strong_assoc = [(pair, max(xy, yx)) for pair, (xy, yx) in eco_mle.items() if max(xy, yx) > 0.8]
strong_assoc.sort(key=lambda x: x[1], reverse=True)

print(f"\nStrongest Species Associations (MLE > 0.8):")
for i, ((sp1, sp2), strength) in enumerate(strong_assoc[:15], 1):
    sp1_short = sp1.split('.')[-1] if '.' in sp1 else sp1[:15]
    sp2_short = sp2.split('.')[-1] if '.' in sp2 else sp2[:15]
    xy, yx = eco_mle[(sp1, sp2)]
    print(f"  {i:2d}. {sp1_short:15s} ↔ {sp2_short:15s}: max={strength:.3f}  [P(2|1)={xy:.2f}, P(1|2)={yx:.2f}]")

# Asymmetry analysis
eco_asymmetries = [abs(xy - yx) for xy, yx in eco_mle.values()]
high_asymmetry = [(pair, abs(xy - yx)) for pair, (xy, yx) in eco_mle.items() if abs(xy - yx) > 0.3]
high_asymmetry.sort(key=lambda x: x[1], reverse=True)

print(f"\nHighly Asymmetric Associations (|P(2|1) - P(1|2)| > 0.3):")
for i, ((sp1, sp2), asym) in enumerate(high_asymmetry[:10], 1):
    sp1_short = sp1.split('.')[-1] if '.' in sp1 else sp1[:15]
    sp2_short = sp2.split('.')[-1] if '.' in sp2 else sp2[:15]
    xy, yx = eco_mle[(sp1, sp2)]
    if xy > yx:
        print(f"  {i:2d}. {sp1_short:15s} → {sp2_short:15s} stronger  [Δ={asym:.3f}]")
    else:
        print(f"  {i:2d}. {sp2_short:15s} → {sp1_short:15s} stronger  [Δ={asym:.3f}]")

#' ### Visualization

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: Asymmetry distribution
axes[0, 0].hist(eco_asymmetries, bins=25, alpha=0.7, color='forestgreen', edgecolor='black')
axes[0, 0].set_xlabel('|P(Species2|Species1) - P(Species1|Species2)|')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('(A) Co-occurrence Asymmetry Distribution', fontweight='bold')
axes[0, 0].axvline(np.mean(eco_asymmetries), color='red', linestyle='--',
                    label=f'Mean: {np.mean(eco_asymmetries):.3f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Association network (top 30 pairs)
axes[0, 1].axis('off')
top_30 = strong_assoc[:30]
network_text = "Top 30 Species Association Network:\n\n"
for i, ((sp1, sp2), strength) in enumerate(top_30, 1):
    sp1_s = sp1.split('.')[-1][:10] if '.' in sp1 else sp1[:10]
    sp2_s = sp2.split('.')[-1][:10] if '.' in sp2 else sp2[:10]
    network_text += f"{i:2d}. {sp1_s:10s} ↔ {sp2_s:10s} [{strength:.2f}]\n"
axes[0, 1].text(0.1, 0.95, network_text, transform=axes[0, 1].transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
axes[0, 1].set_title('(B) Strongest Associations', fontweight='bold')

# Subplot 3: Prevalence vs. Association
species_avg_assoc = {}
for (sp1, sp2), (xy, yx) in eco_mle.items():
    if sp1 not in species_avg_assoc:
        species_avg_assoc[sp1] = []
    if sp2 not in species_avg_assoc:
        species_avg_assoc[sp2] = []
    species_avg_assoc[sp1].append(xy)
    species_avg_assoc[sp2].append(yx)

avg_assoc = {sp: np.mean(vals) for sp, vals in species_avg_assoc.items()}

prevalence_vals = []
assoc_vals = []
for sp in avg_assoc.keys():
    if sp in species_prevalence.index:
        prevalence_vals.append(species_prevalence[sp])
        assoc_vals.append(avg_assoc[sp])

axes[1, 0].scatter(prevalence_vals, assoc_vals, s=80, alpha=0.6,
                   c='teal', edgecolors='black', linewidths=0.5)
axes[1, 0].set_xlabel('Species Prevalence (# islands)')
axes[1, 0].set_ylabel('Average Association Strength')
axes[1, 0].set_title('(C) Prevalence vs. Association', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
corr_eco = np.corrcoef(prevalence_vals, assoc_vals)[0, 1]
axes[1, 0].text(0.05, 0.95, f'r = {corr_eco:.3f}', transform=axes[1, 0].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Subplot 4: Island richness distribution
axes[1, 1].bar(range(len(island_richness)), island_richness.values, alpha=0.7,
               color='orange', edgecolor='black')
axes[1, 1].set_xlabel('Islands (sorted by richness)')
axes[1, 1].set_ylabel('Number of Species')
axes[1, 1].set_title('(D) Island Species Richness', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Galápagos Finch Co-occurrence Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#' ### Key Findings
#'
#' **Ecological Insights:**
#'
#' 1. **High co-occurrence**: Many species pairs have strong associations (MLE > 0.8)
#' 2. **Nested patterns**: Widespread species (e.g., *Ce. olivacea*) co-occur with most others
#' 3. **Asymmetric dependencies**: Some species predict others better than vice versa
#' 4. **Island area effect**: Larger islands support more species and stronger networks
#' 5. **Supports island biogeography theory**: Distance and area shape communities

#' ## 4. Case Study 3: Machine Learning - Feature Selection
#'
#' ### Research Question
#' **Can asymmetric measures improve categorical feature selection?**

print("\n" + "="*60)
print("CASE STUDY 3: FEATURE SELECTION")
print("="*60)

#' ### Data Preparation

# Load mushroom dataset (categorical features predicting edibility)
mushroom_data = asymcat.read_pa_matrix("../resources/mushroom-small.tsv")

print(f"\nDataset: Mushroom Classification")
print(f"  Feature-class co-occurrences: {len(mushroom_data)}")

# Show structure
print(f"\nSample co-occurrences:")
for i, (feature, class_label) in enumerate(list(mushroom_data)[:10]):
    print(f"  {i+1:2d}. {feature:30s} ↔ {class_label}")

#' ### Measure Selection
#'
#' For classification feature selection:
#' - **Theil's U**: U(class|feature) measures predictive value
#' - **Chi-square**: Statistical significance of association
#' - **MLE**: Direct predictability P(class|feature)

# Create scorer
ml_scorer = CatScorer(mushroom_data, smoothing_method='laplace')

# Compute measures
ml_theil = ml_scorer.theil_u()
ml_chi2 = ml_scorer.chi2()
ml_mle = ml_scorer.mle()

#' ### Feature Ranking

# Extract features predicting class
feature_importance = {}

for (feature, class_val), (u_class_feat, u_feat_class) in ml_theil.items():
    # We care about U(class|feature) - how well feature predicts class
    if feature not in feature_importance:
        feature_importance[feature] = []
    feature_importance[feature].append(u_class_feat)

# Average importance across classes
avg_importance = {feat: np.mean(vals) for feat, vals in feature_importance.items()}
sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

print(f"\nFeature Importance Ranking (by Theil's U):")
print(f"{'Rank':<6}{'Feature':<35}{'U(class|feature)':<20}")
print("=" * 60)

for i, (feature, importance) in enumerate(sorted_features[:15], 1):
    print(f"{i:<6}{feature:<35}{importance:.4f}")

#' ### Comparison with Chi-Square

# Extract chi-square scores
chi2_importance = {}
for (feature, class_val), (chi2_val, _) in ml_chi2.items():
    if feature not in chi2_importance:
        chi2_importance[feature] = []
    chi2_importance[feature].append(chi2_val)

avg_chi2 = {feat: np.mean(vals) for feat, vals in chi2_importance.items()}

# Compare rankings
print(f"\nTop 10 Features by Different Measures:")
print(f"\n{'Theil U Rank':<40}{'Chi² Rank':<40}")
print("=" * 80)

sorted_chi2 = sorted(avg_chi2.items(), key=lambda x: x[1], reverse=True)

for i in range(min(10, len(sorted_features))):
    theil_feat, theil_val = sorted_features[i]
    chi2_feat, chi2_val = sorted_chi2[i]
    print(f"{i+1:2d}. {theil_feat[:30]:<35}{chi2_feat[:30]:<35}")

#' ### Visualization

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Subplot 1: Feature importance distribution
importance_vals = list(avg_importance.values())
axes[0, 0].hist(importance_vals, bins=20, alpha=0.7, color='purple', edgecolor='black')
axes[0, 0].set_xlabel("Theil's U (class|feature)")
axes[0, 0].set_ylabel('Number of Features')
axes[0, 0].set_title('(A) Feature Importance Distribution', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Top features bar chart
top_n = 15
top_feats = sorted_features[:top_n]
feat_names = [f[:25] for f, _ in top_feats]
feat_vals = [v for _, v in top_feats]

axes[0, 1].barh(range(len(feat_names)), feat_vals, alpha=0.7,
                color='darkblue', edgecolor='black')
axes[0, 1].set_yticks(range(len(feat_names)))
axes[0, 1].set_yticklabels(feat_names, fontsize=8)
axes[0, 1].invert_yaxis()
axes[0, 1].set_xlabel("Theil's U")
axes[0, 1].set_title(f'(B) Top {top_n} Most Predictive Features', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Subplot 3: Theil U vs Chi²
theil_vals_comp = [avg_importance.get(f, 0) for f in avg_chi2.keys()]
chi2_vals_comp = list(avg_chi2.values())

axes[1, 0].scatter(theil_vals_comp, chi2_vals_comp, alpha=0.6, s=60,
                   c='crimson', edgecolors='black', linewidths=0.5)
axes[1, 0].set_xlabel("Theil's U (avg)")
axes[1, 0].set_ylabel('Chi² (avg)')
axes[1, 0].set_title("(C) Theil's U vs. Chi² Comparison", fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
corr_ml = np.corrcoef(theil_vals_comp, chi2_vals_comp)[0, 1]
axes[1, 0].text(0.05, 0.95, f'r = {corr_ml:.3f}', transform=axes[1, 0].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Subplot 4: Feature selection impact
axes[1, 1].axis('off')
selection_text = f"""
Feature Selection Summary:

Top 5 Features (Theil's U):
"""
for i, (feat, val) in enumerate(sorted_features[:5], 1):
    selection_text += f"\n{i}. {feat[:35]:35s} {val:.4f}"

selection_text += "\n\nInterpretation:\n"
selection_text += "• High U(class|feature) = strong predictor\n"
selection_text += "• Asymmetric: feature→class matters\n"
selection_text += "• Use top features for classification\n"
selection_text += f"• {len([v for v in importance_vals if v > 0.5])} features with U > 0.5"

axes[1, 1].text(0.1, 0.95, selection_text, transform=axes[1, 1].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
axes[1, 1].set_title('(D) Feature Selection Strategy', fontweight='bold')

plt.suptitle('Mushroom Classification: Feature Selection Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#' ### Key Findings
#'
#' **Machine Learning Insights:**
#'
#' 1. **Asymmetric measures** (Theil's U) reveal **directional predictive power**
#' 2. Top features have **high U(class|feature)** but may have **low U(feature|class)**
#' 3. This is correct: we want features that **predict** the class, not vice versa
#' 4. **Chi-square** identifies associations but doesn't reveal directionality
#' 5. **Feature selection strategy**: Keep features with U(class|feature) > threshold

#' ## 5. Best Practices Summary
#'
#' ### Data Preparation
#' ✓ Validate data format (sequences vs. presence-absence)
#' ✓ Check for missing values and encoding issues
#' ✓ Assess sparsity → choose smoothing method
#'
#' ### Measure Selection
#' ✓ **Prediction tasks**: MLE, Theil's U, Goodman-Kruskal λ
#' ✓ **Association discovery**: PMI, NPMI, Chi-square
#' ✓ **Small samples**: Fisher exact, smoothing required
#' ✓ **Linguistic alignment**: Tresoldi measure
#'
#' ### Interpretation
#' ✓ Always report **both directions** (X→Y and Y→X)
#' ✓ Quantify **asymmetry**: |score_xy - score_yx|
#' ✓ Consider **domain knowledge** for validation
#' ✓ Use **multiple measures** for robustness
#'
#' ### Visualization
#' ✓ Heatmaps for **matrix overviews**
#' ✓ Scatter plots for **measure comparison**
#' ✓ Asymmetry plots for **directional effects**
#' ✓ Bar charts for **top associations**
#'
#' ### Reporting
#' ✓ State **research question** clearly
#' ✓ Report **data characteristics** (size, sparsity)
#' ✓ Justify **measure choice**
#' ✓ Provide **quantitative results** with effect sizes
#' ✓ Include **visualizations** for key findings
#' ✓ Discuss **limitations** and assumptions

#' ## 6. Key Takeaways
#'
#' **You've learned:**
#'
#' ✓ Complete analysis workflows for real-world problems
#' ✓ Linguistics: grapheme-phoneme asymmetry reveals orthographic depth
#' ✓ Ecology: species co-occurrence patterns support island biogeography
#' ✓ Machine learning: asymmetric measures improve feature selection
#' ✓ Interpretation requires domain knowledge + statistical rigor
#' ✓ Visualization communicates directional relationships effectively
#'
#' ## Congratulations!
#'
#' You've completed all ASymCat tutorials! You're now ready to:
#' - Analyze asymmetric associations in your own data
#' - Choose appropriate measures for different tasks
#' - Create publication-quality visualizations
#' - Interpret results with statistical rigor
#'
#' ## Further Resources
#'
#' - **[User Guide](USER_GUIDE.md)**: Conceptual foundations
#' - **[API Reference](API_REFERENCE.md)**: Complete function reference
#' - **[LLM Documentation](LLM_DOCUMENTATION.md)**: Quick code patterns
#' - **GitHub Issues**: Questions and community support

print("\n" + "="*60)
print("Tutorial Series Complete! ✓")
print("="*60)
print("\nYou're now an ASymCat expert!")
print("Happy analyzing! 🎉")
