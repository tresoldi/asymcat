Quick Start Guide
=================

This guide will get you started with ASymCat quickly.

Basic Concepts
--------------

ASymCat works with the following key concepts:

**Co-occurrences**
    Observations of two categorical elements from two different series.

**Population**
    A collection of co-occurrences which can come from sequences, individual properties, or presence data.

**Scorer**
    A collection of association measures between all possible products of values in alphabet_x and alphabet_y.

Basic Workflow
--------------

1. **Load your data**::

    import asymcat
    data = asymcat.read_sequences("mydata.tsv")

2. **Collect co-occurrences**::

    cooccs = asymcat.collect_cooccs(data)

3. **Create a scorer**::

    scorer = asymcat.scorer.CatScorer(cooccs)

4. **Compute association measures**::

    # Traditional measures
    mle = scorer.mle()
    pmi = scorer.pmi()
    chi2 = scorer.chi2()
    
    # New measures  
    mutual_info = scorer.mutual_information()
    jaccard = scorer.jaccard_index()
    lambda_gk = scorer.goodman_kruskal_lambda()

Example with Real Data
----------------------

Here's a complete example using the included sample data:

.. code-block:: python

    import asymcat
    
    # Load CMU pronunciation data
    data = asymcat.read_sequences("resources/cmudict.sample100.tsv")
    cooccs = asymcat.collect_cooccs(data)
    
    # Create scorer
    scorer = asymcat.scorer.CatScorer(cooccs)
    
    # Get mutual information scores
    mi_scores = scorer.mutual_information()
    
    # Print top associations
    sorted_pairs = sorted(mi_scores.items(), 
                         key=lambda x: max(x[1]), reverse=True)
    
    for pair, scores in sorted_pairs[:10]:
        print(f"{pair}: MI(X→Y)={scores[0]:.3f}, MI(Y→X)={scores[1]:.3f}")

This will show you the strongest associations between letters and phonemes in the pronunciation data.