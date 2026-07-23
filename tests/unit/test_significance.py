"""
Tests for the statistical-significance (p-value) scorers.

``chi2_pvalue``, ``fisher_pvalue`` and ``log_likelihood_ratio_pvalue`` are the
significance counterparts to the ``chi2``, ``fisher`` and
``log_likelihood_ratio`` statistics. They return the same
``{(x, y): (p, p)}`` mapping shape as the other scorers, with values in
``[0, 1]``, and are validated here against a direct SciPy computation.
"""

import pytest
import scipy.stats as ss  # type: ignore

import asymcat
from asymcat.scorer import (
    CatScorer,
    compute_log_likelihood_ratio,
    compute_log_likelihood_ratio_pvalue,
)

from ..fixtures.data import get_sample_cmu_processed


@pytest.fixture
def scorer():
    """Scorer over the standard sample data."""
    cooccs = asymcat.collect_cooccs(get_sample_cmu_processed())
    return CatScorer(cooccs)


PVALUE_METHODS = ["chi2_pvalue", "fisher_pvalue", "log_likelihood_ratio_pvalue"]


class TestPValueStructure:
    """Shared structural properties of every p-value scorer."""

    @pytest.mark.parametrize("method_name", PVALUE_METHODS)
    def test_values_in_unit_range_and_symmetric(self, scorer, method_name):
        scores = getattr(scorer, method_name)()
        assert len(scores) > 0
        for p_xy, p_yx in scores.values():
            assert 0.0 <= p_xy <= 1.0, f"{method_name}: p-value out of range"
            assert p_xy == p_yx, f"{method_name}: p-value should be symmetric"

    @pytest.mark.parametrize("method_name", PVALUE_METHODS)
    def test_result_is_cached(self, scorer, method_name):
        method = getattr(scorer, method_name)
        assert method() is method()

    @pytest.mark.parametrize("stat_method, pvalue_method", [("chi2", "chi2_pvalue"), ("fisher", "fisher_pvalue")])
    def test_same_pairs_as_statistic(self, scorer, stat_method, pvalue_method):
        assert set(getattr(scorer, pvalue_method)()) == set(getattr(scorer, stat_method)())


class TestPValueCorrectness:
    """The p-values must match a direct SciPy computation."""

    def test_chi2_pvalue_matches_scipy(self, scorer):
        scorer._compute_contingency_table(True)
        pvalues = scorer.chi2_pvalue()
        for pair, ct in scorer._square_ct.items():
            expected = ss.chi2_contingency(ct)[1]
            assert pvalues[pair][0] == pytest.approx(expected)

    def test_chi2_pvalue_nonsquare_matches_scipy(self, scorer):
        scorer._compute_contingency_table(False)
        pvalues = scorer.chi2_pvalue(square_ct=False)
        for pair, ct in scorer._nonsquare_ct.items():
            expected = ss.chi2_contingency(ct)[1]
            assert pvalues[pair][0] == pytest.approx(expected)

    def test_fisher_pvalue_matches_scipy(self, scorer):
        scorer._compute_contingency_table(True)
        pvalues = scorer.fisher_pvalue()
        for pair, ct in scorer._square_ct.items():
            expected = ss.fisher_exact(ct)[1]
            assert pvalues[pair][0] == pytest.approx(expected)

    def test_llr_pvalue_matches_chi2_survival(self, scorer):
        scorer._compute_contingency_table(True)
        pvalues = scorer.log_likelihood_ratio_pvalue()
        for pair, ct in scorer._square_ct.items():
            g2 = compute_log_likelihood_ratio(ct)
            expected = float(ss.chi2.sf(g2, 1))  # 2x2 table -> dof = 1
            assert pvalues[pair][0] == pytest.approx(expected)


class TestPValueBehavior:
    """P-values must respond correctly to strong vs. absent association."""

    def test_strong_association_is_significant(self):
        # 'a' with 'x' and 'b' with 'y', almost perfectly.
        data = [[["a"], ["x"]]] * 50 + [[["b"], ["y"]]] * 50 + [[["a"], ["y"]]] * 2
        scorer = CatScorer(asymcat.collect_cooccs(data))
        for method_name in PVALUE_METHODS:
            p = getattr(scorer, method_name)()[("a", "x")][0]
            assert p < 0.01, f"{method_name}: strong association should be significant, got {p}"

    def test_independent_data_is_not_significant(self):
        # Every x pairs with every y equally: no association.
        data = [[["a", "b"], ["x", "y"]], [["a", "b"], ["y", "x"]]] * 20
        scorer = CatScorer(asymcat.collect_cooccs(data))
        chi2_p = scorer.chi2_pvalue()
        for p_xy, _ in chi2_p.values():
            assert p_xy > 0.05


class TestLLRPValueHelper:
    """Edge cases of the standalone G² p-value helper."""

    def test_no_degrees_of_freedom_returns_one(self):
        # A single-column table has (rows-1) * (cols-1) == 0 degrees of freedom.
        assert compute_log_likelihood_ratio_pvalue([[5.0], [3.0]]) == 1.0

    def test_zero_statistic_returns_one(self):
        # Perfectly balanced table -> G² == 0 -> p-value == 1.
        assert compute_log_likelihood_ratio_pvalue([[5.0, 5.0], [5.0, 5.0]]) == pytest.approx(1.0)
