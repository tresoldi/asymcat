ASymCat: Asymmetric Categorical Association Analysis
==================================================

|PyPI| |Python| |Build| |Coverage| |License|

.. |PyPI| image:: https://badge.fury.io/py/asymcat.svg
   :target: https://badge.fury.io/py/asymcat
.. |Python| image:: https://img.shields.io/pypi/pyversions/asymcat.svg
   :target: https://pypi.org/project/asymcat/
.. |Build| image:: https://github.com/tresoldi/asymcat/workflows/build/badge.svg
   :target: https://github.com/tresoldi/asymcat/actions
.. |Coverage| image:: https://codecov.io/gh/tresoldi/asymcat/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/tresoldi/asymcat
.. |License| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

**ASymCat** is a comprehensive Python library for analyzing **asymmetric associations** between categorical variables. Unlike traditional symmetric measures that treat relationships as bidirectional, ASymCat provides directional measures that reveal which variable predicts which, making it invaluable for understanding causal relationships, dependencies, and information flow in categorical data.

🚀 **Key Features**
------------------

- **17+ Association Measures**: From basic MLE to advanced information-theoretic measures
- **Directional Analysis**: X→Y vs Y→X asymmetric relationship quantification  
- **Robust Smoothing**: FreqProb integration for numerical stability
- **Multiple Data Formats**: Sequences, presence-absence matrices, n-grams
- **CLI Interface**: Production-ready command-line tool with rich output formats
- **Scalable Architecture**: Optimized for large datasets with efficient algorithms
- **Comprehensive Testing**: 75+ tests ensuring reliability and accuracy

🎯 **Why Asymmetric Measures Matter**
------------------------------------

Traditional measures like Pearson's χ² or Cramér's V treat associations as symmetric: the relationship between X and Y is the same as between Y and X. However, many real-world relationships are inherently directional:

- **Linguistics**: Phoneme transitions may be predictable in one direction but not the other
- **Ecology**: Species presence may predict other species asymmetrically  
- **Market Research**: Product purchases may show directional dependencies
- **Medical Analysis**: Symptoms may predict conditions more reliably than vice versa

ASymCat quantifies these directional relationships, revealing hidden patterns that symmetric measures miss.

📊 **Quick Example**
-------------------

.. code-block:: python

    import asymcat

    # Load your categorical data
    data = asymcat.read_sequences("data.tsv")  # or read_pa_matrix() for binary data

    # Collect co-occurrences  
    cooccs = asymcat.collect_cooccs(data)

    # Create scorer and analyze
    scorer = asymcat.CatScorer(cooccs)

    # Get asymmetric measures
    mle_scores = scorer.mle()           # Maximum likelihood estimation
    pmi_scores = scorer.pmi()           # Pointwise mutual information  
    chi2_scores = scorer.chi2()         # Chi-square with smoothing
    fisher_scores = scorer.fisher()     # Fisher exact test

    # Each returns {(x, y): (x→y_score, y→x_score)}
    print(f"A→B: {mle_scores[('A', 'B')][0]:.3f}")
    print(f"B→A: {mle_scores[('A', 'B')][1]:.3f}")

📚 **Documentation Structure**
-----------------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorial

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user-guide/data-formats
   user-guide/association-measures
   user-guide/smoothing-methods
   user-guide/cli-usage
   user-guide/visualization

.. toctree::
   :maxdepth: 2
   :caption: Examples & Applications

   examples
   examples/linguistics
   examples/ecology
   examples/machine-learning
   examples/advanced-usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
   api/common
   api/scorer
   api/correlation

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   mathematical-foundations
   performance
   changelog

.. toctree::
   :maxdepth: 1
   :caption: External Links

   GitHub Repository <https://github.com/tresoldi/asymcat>
   PyPI Package <https://pypi.org/project/asymcat/>
   Issue Tracker <https://github.com/tresoldi/asymcat/issues>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`