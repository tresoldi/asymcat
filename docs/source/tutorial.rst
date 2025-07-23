Tutorial: Complete Guide to ASymCat
===================================

This tutorial provides a comprehensive introduction to asymmetric categorical association analysis using ASymCat. We'll cover the theoretical foundations, practical applications, and advanced techniques through real-world examples.

.. contents:: Table of Contents
   :local:
   :depth: 2

Understanding Asymmetric Association
------------------------------------

Traditional Symmetric Measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most association measures treat relationships as bidirectional:

- **Pearson's χ²**: Tests independence assumption
- **Cramér's V**: Normalized association strength
- **PMI**: Pointwise mutual information

These measures assume that the association between X and Y equals the association between Y and X.

The Asymmetric Advantage
~~~~~~~~~~~~~~~~~~~~~~~~

Real-world relationships are often directional:

.. code-block:: python

    import asymcat
    import numpy as np

    # Example: Medicine prescriptions
    # Doctors predict treatments better than treatments predict doctors
    medical_data = [
        ('Dr_Smith', 'Aspirin'), ('Dr_Smith', 'Aspirin'), ('Dr_Smith', 'Ibuprofen'),
        ('Dr_Jones', 'Morphine'), ('Dr_Jones', 'Morphine'), ('Dr_Jones', 'Morphine'),
        ('Dr_Brown', 'Aspirin'), ('Dr_Brown', 'Aspirin')
    ]

    cooccs = asymcat.collect_cooccs([medical_data])
    scorer = asymcat.CatScorer(cooccs)

    # Asymmetric measure: MLE
    mle_scores = scorer.mle()
    doctor_aspirin = mle_scores[('Dr_Smith', 'Aspirin')]
    
    print(f"P(Aspirin|Dr_Smith) = {doctor_aspirin[0]:.3f}")  # Doctor → Drug
    print(f"P(Dr_Smith|Aspirin) = {doctor_aspirin[1]:.3f}")  # Drug → Doctor

This reveals that knowing the doctor predicts the prescription much better than knowing the prescription predicts the doctor.

Core Concepts
-------------

Co-occurrences
~~~~~~~~~~~~~~

ASymCat analyzes **co-occurrences**: pairs of categorical values that appear together in your data.

.. code-block:: python

    # From sequences (aligned data)
    sequence_data = asymcat.read_sequences("orthography_phonetics.tsv")
    cooccs = asymcat.collect_cooccs(sequence_data)

    # From presence-absence matrices
    pa_data = asymcat.read_pa_matrix("species_islands.tsv")
    cooccs = asymcat.collect_cooccs(pa_data)

    # From n-grams
    text = "the quick brown fox"
    bigrams = list(asymcat.collect_ngrams(text.split(), 2, pad="#"))
    cooccs = asymcat.collect_cooccs([bigrams])

Association Measures
~~~~~~~~~~~~~~~~~~~

ASymCat implements 17+ measures organized by mathematical foundation:

**Probabilistic Measures**

.. code-block:: python

    # Maximum Likelihood Estimation
    mle_scores = scorer.mle()
    # Returns P(Y|X) and P(X|Y) for each pair

    # Jaccard Index (with asymmetric interpretation)
    jaccard_scores = scorer.jaccard_index()

**Information-Theoretic Measures**

.. code-block:: python

    # Pointwise Mutual Information (symmetric)
    pmi_scores = scorer.pmi()
    
    # PMI with smoothing (more stable)
    pmi_smooth_scores = scorer.pmi_smoothed()
    
    # Mutual Information
    mi_scores = scorer.mutual_information()
    
    # Conditional Entropy
    entropy_scores = scorer.conditional_entropy()

**Statistical Measures**

.. code-block:: python

    # Chi-square test
    chi2_scores = scorer.chi2()
    
    # Cramér's V
    cramers_scores = scorer.cramers_v()
    
    # Fisher's Exact Test
    fisher_scores = scorer.fisher()

**Specialized Measures**

.. code-block:: python

    # Theil's Uncertainty Coefficient
    theil_scores = scorer.theil_u()
    
    # Goodman-Kruskal Lambda
    lambda_scores = scorer.goodman_kruskal_lambda()
    
    # Tresoldi measure (for sequence alignment)
    tresoldi_scores = scorer.tresoldi()

Working with Real Data
----------------------

Linguistic Analysis Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's analyze grapheme-phoneme correspondences in English:

.. code-block:: python

    import asymcat
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load CMU pronunciation dictionary
    cmu_data = asymcat.read_sequences("cmudict_sample.tsv")
    cooccs = asymcat.collect_cooccs(cmu_data)

    # Create scorer with smoothing for rare correspondences
    scorer = asymcat.CatScorer(cooccs, smoothing_method='laplace')

    # Compute multiple measures
    measures = {
        'MLE': scorer.mle(),
        'PMI': scorer.pmi(),
        'Theil_U': scorer.theil_u()
    }

    # Find strongest orthography → phoneme correspondences
    ortho_phon = []
    for (ortho, phon), (op_score, po_score) in measures['MLE'].items():
        if len(ortho) == 1 and ortho.isalpha():  # Single letter
            ortho_phon.append((ortho, phon, op_score))

    # Sort by strength
    ortho_phon.sort(key=lambda x: x[2], reverse=True)

    print("Strongest Orthography → Phoneme Correspondences:")
    for ortho, phon, score in ortho_phon[:10]:
        print(f"  '{ortho}' → /{phon}/: {score:.3f}")

Ecological Network Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze species co-occurrence patterns:

.. code-block:: python

    # Load species presence-absence data
    species_data = asymcat.read_pa_matrix("galapagos_finches.tsv")
    cooccs = asymcat.collect_cooccs(species_data)

    scorer = asymcat.CatScorer(cooccs, smoothing_method='laplace')

    # Compute ecological measures
    mle_scores = scorer.mle()
    jaccard_scores = scorer.jaccard_index()
    fisher_scores = scorer.fisher()

    # Identify strong associations
    strong_pairs = []
    for (sp1, sp2), (score12, score21) in mle_scores.items():
        max_score = max(score12, score21)
        if max_score > 0.7:  # Strong association threshold
            direction = "→" if score12 > score21 else "←"
            strong_pairs.append((sp1, sp2, max_score, direction))

    strong_pairs.sort(key=lambda x: x[2], reverse=True)

    print("Strong Species Associations:")
    for sp1, sp2, score, direction in strong_pairs:
        print(f"  {sp1} {direction} {sp2}: {score:.3f}")

Advanced Techniques
-------------------

Handling Sparse Data with Smoothing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Categorical data often contains rare co-occurrences. ASymCat provides several smoothing methods:

.. code-block:: python

    # No smoothing (can have zero probabilities)
    scorer_raw = asymcat.CatScorer(cooccs, smoothing_method='mle')

    # Laplace smoothing (add-one smoothing)
    scorer_laplace = asymcat.CatScorer(cooccs, smoothing_method='laplace')

    # Lidstone smoothing (parameterized)
    scorer_lidstone = asymcat.CatScorer(cooccs, 
                                       smoothing_method='lidstone', 
                                       smoothing_alpha=0.5)

    # Compare smoothing effects
    raw_scores = scorer_raw.mle()
    smooth_scores = scorer_laplace.mle()

    print("Smoothing Effect Comparison:")
    for pair in list(raw_scores.keys())[:5]:
        raw_xy, raw_yx = raw_scores[pair]
        smooth_xy, smooth_yx = smooth_scores[pair]
        print(f"  {pair}: Raw=({raw_xy:.3f}, {raw_yx:.3f}), "
              f"Smoothed=({smooth_xy:.3f}, {smooth_yx:.3f})")

Score Transformations and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transform scores for analysis and visualization:

.. code-block:: python

    # Scale scores to [0, 1] range
    scaled_scores = asymcat.scorer.scale_scorer(mle_scores, method="minmax")

    # Standardize scores (zero mean, unit variance)
    standardized_scores = asymcat.scorer.scale_scorer(mle_scores, method="stdev")

    # Invert scores (high becomes low)
    inverted_scores = asymcat.scorer.invert_scorer(scaled_scores)

    # Convert to matrices for heatmap visualization
    xy_matrix, yx_matrix, x_labels, y_labels = asymcat.scorer.scorer2matrices(scaled_scores)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # X→Y heatmap
    sns.heatmap(xy_matrix, xticklabels=y_labels, yticklabels=x_labels,
                annot=True, fmt='.2f', cmap='viridis', ax=ax1)
    ax1.set_title('X → Y Associations')

    # Y→X heatmap
    sns.heatmap(yx_matrix, xticklabels=x_labels, yticklabels=y_labels,
                annot=True, fmt='.2f', cmap='viridis', ax=ax2)
    ax2.set_title('Y → X Associations')

    plt.tight_layout()
    plt.show()

Statistical Validation
~~~~~~~~~~~~~~~~~~~~~~

Validate your results with statistical testing:

.. code-block:: python

    import numpy as np
    from scipy import stats

    def bootstrap_asymmetry(data, n_bootstrap=1000):
        """Bootstrap confidence intervals for asymmetry measures."""
        asymmetries = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            sample = [data[np.random.randint(len(data))] for _ in range(len(data))]
            sample_cooccs = asymcat.collect_cooccs([sample])
            
            if len(sample_cooccs) > 0:
                sample_scorer = asymcat.CatScorer(sample_cooccs, smoothing_method='laplace')
                sample_mle = sample_scorer.mle()
                
                # Calculate mean asymmetry
                pair_asymmetries = [abs(xy - yx) for xy, yx in sample_mle.values()]
                if pair_asymmetries:
                    asymmetries.append(np.mean(pair_asymmetries))
        
        if asymmetries:
            return np.mean(asymmetries), np.percentile(asymmetries, [2.5, 97.5])
        return None, None

    # Test with your data
    mean_asymmetry, ci = bootstrap_asymmetry(medical_data)
    if mean_asymmetry:
        print(f"Mean asymmetry: {mean_asymmetry:.4f}")
        print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

Command-Line Interface Usage
----------------------------

ASymCat provides a powerful CLI for production workflows:

.. code-block:: bash

    # Basic analysis
    asymcat data.tsv --scorers mle pmi chi2

    # With smoothing and output formatting
    asymcat sequences.tsv \
        --scorers all \
        --smoothing laplace \
        --smoothing-alpha 1.0 \
        --sort-by xy \
        --top 20 \
        --output-format csv \
        --output results.csv

    # N-gram analysis
    asymcat text_data.tsv \
        --ngrams 2 \
        --pad "#" \
        --min-count 5 \
        --scorers tresoldi mle

    # Presence-absence matrix analysis
    asymcat species_data.tsv \
        --format pa-matrix \
        --scorers fisher jaccard \
        --table-format markdown

Best Practices
--------------

Data Preparation
~~~~~~~~~~~~~~~

1. **Clean your data**: Remove duplicates and handle missing values
2. **Choose appropriate format**: Sequences for aligned data, PA matrices for binary data
3. **Consider sample size**: Use smoothing for datasets with rare co-occurrences

Measure Selection
~~~~~~~~~~~~~~~~

1. **Start with MLE** for interpretable conditional probabilities
2. **Add information-theoretic measures** (PMI, Theil U) for additional perspectives
3. **Use statistical measures** (Chi², Fisher) for significance testing
4. **Compare multiple measures** to validate findings

Interpretation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Consider domain knowledge** when interpreting asymmetric relationships
2. **Validate with statistical testing** (bootstrap, permutation tests)
3. **Visualize results** to communicate directional patterns
4. **Document methodology** for reproducible analysis

Common Pitfalls
~~~~~~~~~~~~~~~

1. **Ignoring sparsity**: Always consider smoothing for sparse data
2. **Over-interpreting small effects**: Use statistical validation
3. **Conflating association with causation**: Asymmetric ≠ causal
4. **Ignoring multiple comparisons**: Adjust for multiple testing when appropriate

Next Steps
----------

- Explore the :doc:`examples/index` for domain-specific applications
- Check the :doc:`api/index` for detailed function documentation
- Read about :doc:`mathematical-foundations` for theoretical background
- Contribute to the project via the :doc:`contributing` guide

This tutorial provides a solid foundation for using ASymCat effectively. The library's power lies in revealing directional patterns hidden in categorical data, opening new avenues for understanding complex relationships across diverse domains.