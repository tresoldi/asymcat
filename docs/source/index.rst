ASymCat Documentation
=====================

Welcome to ASymCat's documentation!

**ASymCat** is a Python library for obtaining asymmetric measures of association between categorical variables in data exploration and description.

The library is intended for usage on three main types of data:

- Collections of pairwise sequences (both aligned and not)
- Categorical fields in databases  
- Matrices of presence/absence (such as for investigation of species co-occurrence in biology)

More than statistical significance, ASymCat is interested in effect size and strength of association.

Quick Start
-----------

Installation::

    pip install asymcat

Basic usage:

.. code-block:: python

    import asymcat
    
    # Load data and compute co-occurrences
    data = asymcat.read_sequences("mydata.tsv")
    cooccs = asymcat.collect_cooccs(data)
    
    # Create scorer and compute measures
    scorer = asymcat.scorer.CatScorer(cooccs)
    
    # Get different association measures
    mle = scorer.mle()
    pmi = scorer.pmi()
    mutual_info = scorer.mutual_information()

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`