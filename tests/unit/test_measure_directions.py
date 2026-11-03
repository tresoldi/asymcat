"""Pin the direction of every measure's ``(X→Y, Y→X)`` tuple.

Every ``CatScorer`` measure maps a pair ``(x, y)`` to a tuple whose first
element describes ``x`` predicting ``y`` and whose second describes ``y``
predicting ``x``. These tests pin that order against values computed
independently, by hand or with the scalar helpers, on a small table where the
two directions differ.

The data: ``q`` occurs 3 times (twice with ``k``), ``c`` 13 times (8 with
``k``), so ``P(k|q) = 2/3`` while ``P(q|k) = 2/10``.
"""

import math

import pytest

import asymcat
from asymcat.scorer import CatScorer, compute_theil_u, conditional_entropy

PAIRS = [("q", "k")] * 2 + [("c", "k")] * 8 + [("c", "s")] * 5 + [("q", "s")] * 1

# The co-occurrences on which the per-pair information measures are computed:
# those where x == "q" or y == "k".
SUBSET = [pair for pair in PAIRS if pair[0] == "q" or pair[1] == "k"]
SUBSET_X = [x for x, _ in SUBSET]
SUBSET_Y = [y for _, y in SUBSET]


@pytest.fixture(scope="module")
def scorer() -> CatScorer:
    sequences = [[[x], [y]] for x, y in PAIRS]
    return CatScorer(asymcat.collect_cooccs(sequences))


def test_mle_is_p_y_given_x_then_p_x_given_y(scorer):
    assert scorer.mle()[("q", "k")] == pytest.approx((2 / 3, 2 / 10))


def test_smoothed_mle_keeps_the_same_order(scorer):
    sequences = [[[x], [y]] for x, y in PAIRS]
    smoothed = CatScorer(asymcat.collect_cooccs(sequences), smoothing_method="laplace")
    y_given_x, x_given_y = smoothed.mle()[("q", "k")]

    # Laplace shrinks both towards 1/2 but keeps P(k|q) above P(q|k).
    assert y_given_x > x_given_y


def test_tresoldi_uses_p_y_given_x_for_the_x_to_y_direction(scorer):
    pmi = math.log(16 / 15)  # log(P(q,k) / (P(q) P(k))) = log((2/16) / ((3/16)(10/16)))
    expected = (pmi ** (1 - 2 / 3), pmi ** (1 - 2 / 10))
    assert scorer.tresoldi()[("q", "k")] == pytest.approx(expected)


def test_theil_u_is_u_y_given_x_then_u_x_given_y(scorer):
    # compute_theil_u(a, b) is U(a | b), the uncertainty in `a` removed by `b`.
    expected = (compute_theil_u(SUBSET_Y, SUBSET_X), compute_theil_u(SUBSET_X, SUBSET_Y))
    assert scorer.theil_u()[("q", "k")] == pytest.approx(expected)
    assert expected[0] != pytest.approx(expected[1])


def test_cond_entropy_is_h_y_given_x_then_h_x_given_y(scorer):
    # conditional_entropy(a, b) is H(a | b).
    expected = (conditional_entropy(SUBSET_Y, SUBSET_X), conditional_entropy(SUBSET_X, SUBSET_Y))
    assert scorer.cond_entropy()[("q", "k")] == pytest.approx(expected)
    assert expected[0] != pytest.approx(expected[1])


def test_goodman_kruskal_lambda_is_lambda_y_given_x_then_lambda_x_given_y(scorer):
    # On the subset (q,k)x2, (c,k)x8, (q,s)x1:
    # predicting Y: 1 error without X (always guess k), 1 error with X -> 0.
    # predicting X: 3 errors without Y (always guess c), 2 errors with Y -> 1/3.
    assert scorer.goodman_kruskal_lambda()[("q", "k")] == pytest.approx((0.0, 1 / 3))


@pytest.mark.parametrize(
    "measure",
    [
        "pmi",
        "pmi_smoothed",
        "chi2",
        "chi2_pvalue",
        "cramers_v",
        "fisher",
        "fisher_pvalue",
        "log_likelihood_ratio",
        "log_likelihood_ratio_pvalue",
        "mutual_information",
        "normalized_mutual_information",
        "jaccard_index",
    ],
)
def test_symmetric_measures_report_the_same_value_both_ways(scorer, measure):
    for xy, yx in getattr(scorer, measure)().values():
        assert xy == pytest.approx(yx, nan_ok=True)
