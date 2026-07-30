"""
Convenience wrappers for computing association measures on parallel series.

Unlike :class:`asymcat.scorer.CatScorer`, which returns per-symbol-pair
directional scores from a list of co-occurrences, the helpers in this module
operate on two aligned series (``series_x`` and ``series_y``) and return a
single summary statistic for the pair of variables.

Note that not every measure here is symmetric: :func:`cramers_v` is symmetric
by definition, whereas :func:`conditional_entropy` and :func:`theil_u` are
directional (``x`` given ``y``) and are therefore, in general, asymmetric.
"""

# Import Python standard libraries
from collections import Counter

# Import local modules
from . import common, scorer


def cramers_v(series_x, series_y):
    """
    Compute Cramér's V between two aligned series.

    Cramér's V is a symmetric measure of association derived from Pearson's
    chi-square statistic, ranging from 0 (no association) to 1 (perfect
    association).

    Parameters
    ----------
    series_x : list
        The first series of observed symbols.
    series_y : list
        The second series of observed symbols, aligned with ``series_x``.

    Returns
    -------
    float
        Cramér's V for the two series.
    """
    cooccs = list(zip(series_x, series_y, strict=False))

    # Build a contingency table
    # TODO: use another library? pandas?
    alphabet_x, alphabet_y = common.collect_alphabets(cooccs)
    ct: list[list[float]] = []
    for x_val in alphabet_x:
        counter = Counter([y for x, y in cooccs if x == x_val])
        obs = [float(counter.get(y, 0)) for y in alphabet_y]
        ct.append(obs)

    # Compute Cramér's V and return
    return scorer.compute_cramers_v(ct)


def conditional_entropy(series_x, series_y):
    """
    Compute the conditional entropy of ``series_x`` given ``series_y``.

    This is a directional measure: in general, ``H(x|y) != H(y|x)``.

    Parameters
    ----------
    series_x : list
        The series whose entropy (conditioned on ``series_y``) is computed.
    series_y : list
        The conditioning series, aligned with ``series_x``.

    Returns
    -------
    float
        The conditional entropy ``H(x|y)``.
    """
    return scorer.conditional_entropy(series_x, series_y)


def theil_u(series_x, series_y):
    """
    Compute Theil's U (uncertainty coefficient) of ``series_x`` given ``series_y``.

    Theil's U normalizes the reduction in uncertainty of ``series_x`` obtained
    from knowing ``series_y`` by the entropy of ``series_x``, yielding a value
    in the ``[0, 1]`` range. It is a directional measure and is, in general,
    asymmetric (``U(x|y) != U(y|x)``).

    Parameters
    ----------
    series_x : list
        The series whose uncertainty is being reduced.
    series_y : list
        The conditioning series, aligned with ``series_x``.

    Returns
    -------
    float
        Theil's U for ``series_x`` given ``series_y``, in ``[0, 1]``.
    """
    return scorer.compute_theil_u(series_x, series_y)
