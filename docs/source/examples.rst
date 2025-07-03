Examples
========

This section provides detailed examples of using ASymCat for different types of data analysis.

Sequence Analysis
-----------------

Working with aligned sequences such as orthography-phonology pairs:

.. code-block:: python

    import asymcat
    
    # Load sequence data
    data = asymcat.read_sequences("pronunciation_data.tsv")
    cooccs = asymcat.collect_cooccs(data)
    scorer = asymcat.scorer.CatScorer(cooccs)
    
    # Compare different measures
    pmi = scorer.pmi()
    mi = scorer.mutual_information()
    tresoldi = scorer.tresoldi()

Categorical Database Fields
---------------------------

Analyzing associations between categorical variables in a database:

.. code-block:: python

    import asymcat
    
    # Prepare your data as sequence pairs
    # Each row should be [category1_values, category2_values]
    data = [
        [["red", "blue", "green"], ["car", "truck", "car"]],
        [["blue", "red", "red"], ["bike", "car", "car"]],
        # ... more data
    ]
    
    cooccs = asymcat.collect_cooccs(data)
    scorer = asymcat.scorer.CatScorer(cooccs)
    
    # Asymmetric measures
    lambda_scores = scorer.goodman_kruskal_lambda()

Species Co-occurrence
---------------------

Analyzing presence/absence matrices for biological data:

.. code-block:: python

    import asymcat
    
    # Load presence-absence matrix
    combinations = asymcat.read_pa_matrix("species_data.tsv")
    scorer = asymcat.scorer.CatScorer(combinations)
    
    # Symmetric measures for co-occurrence
    jaccard = scorer.jaccard_index()
    chi2 = scorer.chi2()

Working with Matrices
---------------------

Converting scores to matrices for visualization:

.. code-block:: python

    import asymcat
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get scores and convert to matrices
    scores = scorer.mutual_information()
    xy, yx, alpha_x, alpha_y = asymcat.scorer.scorer2matrices(scores)
    
    # Visualize
    plt.figure(figsize=(10, 8))
    sns.heatmap(xy, xticklabels=alpha_x, yticklabels=alpha_y)
    plt.title("Mutual Information (X→Y)")
    plt.show()

Scaling and Transformation
--------------------------

Normalizing scores for comparison:

.. code-block:: python

    # Scale scores to [0,1] range
    scaled = asymcat.scorer.scale_scorer(scores, method="minmax")
    
    # Invert scores (higher = stronger association)
    inverted = asymcat.scorer.invert_scorer(scaled)