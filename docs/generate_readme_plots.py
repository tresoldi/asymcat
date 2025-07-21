#!/usr/bin/env python3
"""
Generate plots for README documentation.
Creates visualizations demonstrating ASymCat's capabilities.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

import asymcat

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def create_mathematical_example_plot():
    """Create visualization of the mathematical example from README."""

    # Simple asymmetric dataset - each sequence pair
    simple_data = [
        [['A'], ['c']],
        [['A'], ['c']],
        [['A'], ['d']],  # A can lead to c or d
        [['B'], ['g']],
        [['B'], ['g']],
        [['B'], ['f']],  # B can lead to g or f
    ]

    cooccs = asymcat.collect_cooccs(simple_data)
    scorer = asymcat.scorer.CatScorer(cooccs)

    # Compute measures
    mle_scores = scorer.mle()
    pmi_scores = scorer.pmi()
    theil_scores = scorer.theil_u()

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Extract data for key pairs
    pairs = [('A', 'c'), ('A', 'd'), ('B', 'g'), ('B', 'f')]
    pair_labels = [f"{p[0]}→{p[1]}" for p in pairs]

    # MLE comparison (X→Y vs Y→X)
    mle_xy = [mle_scores[p][0] if p in mle_scores else 0 for p in pairs]
    mle_yx = [mle_scores[p][1] if p in mle_scores else 0 for p in pairs]

    x = np.arange(len(pairs))
    width = 0.35

    bars1 = axes[0].bar(x - width / 2, mle_xy, width, label='P(Y|X)', alpha=0.8, color='steelblue')
    bars2 = axes[0].bar(x + width / 2, mle_yx, width, label='P(X|Y)', alpha=0.8, color='orange')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=9,
            )

    axes[0].set_xlabel('Variable Pairs')
    axes[0].set_ylabel('Conditional Probability')
    axes[0].set_title('MLE: Asymmetric Relationships\\n(Conditional Probabilities)', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pair_labels, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.1)

    # PMI (symmetric)
    pmi_vals = [pmi_scores[p][0] if p in pmi_scores else 0 for p in pairs]
    bars3 = axes[1].bar(x, pmi_vals, alpha=0.8, color='green')

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        axes[1].annotate(
            f'{height:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9,
        )

    axes[1].set_xlabel('Variable Pairs')
    axes[1].set_ylabel('PMI Score')
    axes[1].set_title('PMI: Symmetric Information\\n(Same for both directions)', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pair_labels, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)

    # Theil's U comparison
    theil_xy = [theil_scores[p][0] if p in theil_scores else 0 for p in pairs]
    theil_yx = [theil_scores[p][1] if p in theil_scores else 0 for p in pairs]

    bars4 = axes[2].bar(x - width / 2, theil_xy, width, label='U(Y|X)', alpha=0.8, color='purple')
    bars5 = axes[2].bar(x + width / 2, theil_yx, width, label='U(X|Y)', alpha=0.8, color='red')

    # Add value labels
    for bars in [bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            axes[2].annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=9,
            )

    axes[2].set_xlabel('Variable Pairs')
    axes[2].set_ylabel('Uncertainty Coefficient')
    axes[2].set_title("Theil's U: Asymmetric\\nUncertainty Reduction", fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(pair_labels, rotation=45, ha='right')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Asymmetric vs Symmetric Association Measures', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('docs/images/mathematical_example.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Created mathematical example plot: docs/images/mathematical_example.png")


def create_smoothing_comparison_plot():
    """Create visualization showing smoothing effects."""

    # Create sparse data with zero co-occurrences
    sparse_data = [
        [['common'], ['frequent']],
        [['common'], ['frequent']],
        [['common'], ['frequent']],
        [['common'], ['rare']],
        [['rare_item'], ['frequent']],
        [['rare_item'], ['very_rare']],
    ]

    sparse_cooccs = asymcat.collect_cooccs(sparse_data)

    # Create scorers with different smoothing
    scorers = {
        'No Smoothing (MLE)': asymcat.scorer.CatScorer(sparse_cooccs, smoothing_method='mle'),
        'Laplace Smoothing': asymcat.scorer.CatScorer(sparse_cooccs, smoothing_method='laplace'),
        'Lidstone (α=0.5)': asymcat.scorer.CatScorer(sparse_cooccs, smoothing_method='lidstone', smoothing_alpha=0.5),
    }

    # Get scores
    smoothing_results = {}
    for method, scorer in scorers.items():
        smoothing_results[method] = scorer.mle()

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Show zero vs non-zero scores
    methods = list(smoothing_results.keys())
    zero_counts = []
    total_counts = []

    for method in methods:
        scores = [max(xy, yx) for xy, yx in smoothing_results[method].values()]
        zero_count = sum(1 for s in scores if s == 0.0)
        zero_counts.append(zero_count)
        total_counts.append(len(scores))

    # Calculate percentages
    zero_percentages = [100 * z / t for z, t in zip(zero_counts, total_counts)]
    nonzero_percentages = [100 - p for p in zero_percentages]

    x = np.arange(len(methods))
    width = 0.6

    bars1 = axes[0].bar(x, zero_percentages, width, label='Zero probabilities', color='lightcoral', alpha=0.8)
    bars2 = axes[0].bar(
        x,
        nonzero_percentages,
        width,
        bottom=zero_percentages,
        label='Non-zero probabilities',
        color='lightblue',
        alpha=0.8,
    )

    # Add percentage labels
    for i, (z_pct, nz_pct) in enumerate(zip(zero_percentages, nonzero_percentages)):
        if z_pct > 5:  # Only show label if percentage is substantial
            axes[0].text(i, z_pct / 2, f'{z_pct:.0f}%', ha='center', va='center', fontweight='bold')
        if nz_pct > 5:
            axes[0].text(i, z_pct + nz_pct / 2, f'{nz_pct:.0f}%', ha='center', va='center', fontweight='bold')

    axes[0].set_xlabel('Smoothing Method')
    axes[0].set_ylabel('Percentage of Pairs')
    axes[0].set_title('Effect of Smoothing on Zero Probabilities', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15, ha='right')
    axes[0].legend()
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3)

    # 2. Score distributions
    colors = ['red', 'green', 'blue']
    for i, method in enumerate(methods):
        scores = [max(xy, yx) for xy, yx in smoothing_results[method].values()]
        # Remove zeros for better visualization
        nonzero_scores = [s for s in scores if s > 0.001]

        if nonzero_scores:
            axes[1].hist(
                nonzero_scores,
                alpha=0.6,
                bins=15,
                label=method.split('(')[0],
                color=colors[i],
                density=True,
                edgecolor='black',
                linewidth=0.5,
            )

    axes[1].set_xlabel('Probability Score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution of Non-Zero Probability Scores', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Smoothing Effects on Sparse Categorical Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('docs/images/smoothing_effects.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Created smoothing comparison plot: docs/images/smoothing_effects.png")


def create_species_cooccurrence_plot():
    """Create species co-occurrence visualization using Galapagos data."""

    try:
        # Load Galápagos data
        galapagos_data = asymcat.read_pa_matrix("resources/galapagos.tsv")
        galapagos_cooccs = asymcat.collect_cooccs(galapagos_data)

        # Create scorer with smoothing
        scorer = asymcat.scorer.CatScorer(galapagos_cooccs, smoothing_method='laplace')

        # Compute measures
        mle_scores = scorer.mle()
        jaccard_scores = scorer.jaccard_index()

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Show original presence-absence matrix
        df = pd.read_csv("resources/galapagos.tsv", sep='\\t', index_col=0)

        # Select subset for better visualization
        selected_species = df.columns[:8]  # First 8 species
        selected_islands = df.index[:12]  # First 12 islands
        df_subset = df.loc[selected_islands, selected_species]

        sns.heatmap(
            df_subset,
            cmap='RdYlBu_r',
            cbar_kws={'label': 'Presence/Absence'},
            xticklabels=True,
            yticklabels=True,
            ax=axes[0],
            linewidths=0.5,
            square=True,
        )
        axes[0].set_title('Galápagos Finch Distribution\\n(Sample: Islands × Species)', fontweight='bold')
        axes[0].set_xlabel('Species')
        axes[0].set_ylabel('Islands')

        # Rotate labels for better readability
        axes[0].set_xticklabels(
            [col.split('.')[0][:12] if '.' in col else col[:12] for col in df_subset.columns], rotation=45, ha='right'
        )
        axes[0].set_yticklabels(df_subset.index, rotation=0)

        # 2. Show strongest associations
        # Get top associations by MLE
        associations = []
        for (sp1, sp2), (xy, yx) in mle_scores.items():
            max_assoc = max(xy, yx)
            if max_assoc > 0.3:  # Only strong associations
                associations.append((sp1, sp2, max_assoc))

        associations.sort(key=lambda x: x[2], reverse=True)
        top_associations = associations[:15]  # Top 15

        if top_associations:
            species_pairs = [f"{a[0][:8]}\\n{a[1][:8]}" for a in top_associations]
            strengths = [a[2] for a in top_associations]

            bars = axes[1].barh(range(len(species_pairs)), strengths, alpha=0.7, color='forestgreen')

            # Add value labels
            for i, (bar, strength) in enumerate(zip(bars, strengths)):
                axes[1].text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{strength:.2f}',
                    ha='left',
                    va='center',
                    fontsize=9,
                )

            axes[1].set_yticks(range(len(species_pairs)))
            axes[1].set_yticklabels(species_pairs, fontsize=9)
            axes[1].set_xlabel('Association Strength (MLE)')
            axes[1].set_title('Strongest Species Co-occurrence\\nAssociations (MLE > 0.3)', fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='x')
            axes[1].set_xlim(0, 1.1)

            # Invert y-axis to show strongest at top
            axes[1].invert_yaxis()

        plt.tight_layout()
        plt.savefig('docs/images/species_cooccurrence.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Created species co-occurrence plot: docs/images/species_cooccurrence.png")

    except Exception as e:
        print(f"⚠️  Could not create species plot: {e}")


def create_linguistic_analysis_plot():
    """Create linguistic analysis visualization using CMU data."""

    try:
        # Load CMU data
        cmu_data = asymcat.read_sequences("resources/cmudict.sample100.tsv")
        cmu_cooccs = asymcat.collect_cooccs(cmu_data)

        scorer = asymcat.scorer.CatScorer(cmu_cooccs)
        mle_scores = scorer.mle()

        # Analyze orthography-phoneme correspondences
        ortho_to_phon = []
        for (x, y), (xy, yx) in mle_scores.items():
            # Assume single letters are orthography, IPA symbols are phonemes
            if len(x) == 1 and x.isalpha() and xy > 0.1:  # Strong correspondences
                ortho_to_phon.append((x.upper(), y, xy))
            elif len(y) == 1 and y.isalpha() and yx > 0.1:
                ortho_to_phon.append((y.upper(), x, yx))

        # Sort by strength and get top correspondences
        ortho_to_phon.sort(key=lambda x: x[2], reverse=True)
        top_correspondences = ortho_to_phon[:20]

        if top_correspondences:
            fig, axes = plt.subplots(1, 2, figsize=(15, 8))

            # 1. Bar chart of top correspondences
            letters = [c[0] for c in top_correspondences]
            phonemes = [c[1] for c in top_correspondences]
            strengths = [c[2] for c in top_correspondences]

            bars = axes[0].bar(
                range(len(letters)), strengths, alpha=0.7, color=plt.cm.viridis([s / max(strengths) for s in strengths])
            )

            axes[0].set_xlabel('Orthography-Phoneme Pairs')
            axes[0].set_ylabel('P(phoneme | letter)')
            axes[0].set_title('Strongest Letter-Sound Correspondences\\n(English Orthography → IPA)', fontweight='bold')

            # Add correspondence labels
            for i, (letter, phoneme, strength) in enumerate(top_correspondences):
                axes[0].text(
                    i, strength + 0.02, f'{letter}→{phoneme}', rotation=90, ha='center', va='bottom', fontsize=8
                )

            axes[0].set_xticks([])
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, max(strengths) * 1.2)

            # 2. Correspondence network/matrix visualization
            # Group by letter
            letter_groups = {}
            for letter, phoneme, strength in top_correspondences:
                if letter not in letter_groups:
                    letter_groups[letter] = []
                letter_groups[letter].append((phoneme, strength))

            # Create matrix-like visualization
            unique_letters = sorted(letter_groups.keys())[:10]  # Top 10 letters
            unique_phonemes = list(set([p for letter in unique_letters for p, s in letter_groups.get(letter, [])]))[:15]

            matrix = np.zeros((len(unique_letters), len(unique_phonemes)))

            for i, letter in enumerate(unique_letters):
                for phoneme, strength in letter_groups.get(letter, []):
                    if phoneme in unique_phonemes:
                        j = unique_phonemes.index(phoneme)
                        matrix[i, j] = strength

            im = axes[1].imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            axes[1].set_xticks(range(len(unique_phonemes)))
            axes[1].set_xticklabels(unique_phonemes, rotation=45, ha='right')
            axes[1].set_yticks(range(len(unique_letters)))
            axes[1].set_yticklabels(unique_letters)
            axes[1].set_xlabel('Phonemes (IPA)')
            axes[1].set_ylabel('Letters')
            axes[1].set_title('Letter-Phoneme Association Matrix\\n(Darker = Stronger Association)', fontweight='bold')

            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1])
            cbar.set_label('Association Strength')

            # Add text annotations for strong associations
            for i in range(len(unique_letters)):
                for j in range(len(unique_phonemes)):
                    if matrix[i, j] > 0.3:
                        axes[1].text(
                            j,
                            i,
                            f'{matrix[i, j]:.2f}',
                            ha='center',
                            va='center',
                            fontsize=8,
                            color='white',
                            fontweight='bold',
                        )

            plt.tight_layout()
            plt.savefig('docs/images/linguistic_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("✅ Created linguistic analysis plot: docs/images/linguistic_analysis.png")

    except Exception as e:
        print(f"⚠️  Could not create linguistic plot: {e}")


def main():
    """Generate all README plots."""

    # Create images directory if it doesn't exist
    os.makedirs('docs/images', exist_ok=True)

    print("🎨 Generating plots for README documentation...")
    print()

    # Generate plots
    create_mathematical_example_plot()
    create_smoothing_comparison_plot()
    create_species_cooccurrence_plot()
    create_linguistic_analysis_plot()

    print()
    print("🎉 All plots generated successfully!")
    print("📁 Images saved in: docs/images/")
    print()
    print("Add these to your README:")
    print("![Mathematical Example](docs/images/mathematical_example.png)")
    print("![Smoothing Effects](docs/images/smoothing_effects.png)")
    print("![Species Co-occurrence](docs/images/species_cooccurrence.png)")
    print("![Linguistic Analysis](docs/images/linguistic_analysis.png)")


if __name__ == "__main__":
    main()
