"""Pin the conventions of `pmi_smoothed`, Goodman-Kruskal lambda and `square_ct`.

The example data is the documentation's: how English spellings map to sounds.
"""

import math

import pytest

from asymcat.scorer import CatScorer, compute_goodman_kruskal_lambda, compute_theil_u

PAIRS = [("q", "k")] * 2 + [("c", "k")] * 6 + [("k", "k")] * 3 + [("c", "s")] * 4


class TestSmoothedProbabilities:
    """The smoothed marginals are the sums of the smoothed joint distribution."""

    @pytest.mark.parametrize("method", ["mle", "laplace", "lidstone"])
    def test_marginals_are_sums_of_the_joint(self, method):
        probs = CatScorer(PAIRS, smoothing_method=method, smoothing_alpha=0.5).get_smoothed_probabilities()
        for x, p_x in probs["marginal_x"].items():
            assert p_x == pytest.approx(sum(p for (a, _), p in probs["joint"].items() if a == x))
        for y, p_y in probs["marginal_y"].items():
            assert p_y == pytest.approx(sum(p for (_, b), p in probs["joint"].items() if b == y))
        assert sum(probs["joint"].values()) == pytest.approx(1.0)

    def test_mle_marginals_are_relative_frequencies(self):
        probs = CatScorer(PAIRS).get_smoothed_probabilities()
        assert probs["marginal_x"]["c"] == pytest.approx(10 / 15)
        assert probs["marginal_y"]["k"] == pytest.approx(11 / 15)

    def test_laplace_pmi_matches_a_hand_computation(self):
        # Six cells, one pseudo-count each: P(q, s) = 1/21, P(q) = 4/21, P(s) = 7/21.
        pmi = CatScorer(PAIRS, smoothing_method="laplace").pmi_smoothed()
        assert pmi[("q", "s")][0] == pytest.approx(math.log((1 / 21) / ((4 / 21) * (7 / 21))))

    @pytest.mark.parametrize("method", ["laplace", "lidstone"])
    def test_unobserved_pairs_score_below_independence(self, method):
        scorer = CatScorer(PAIRS, smoothing_method=method, smoothing_alpha=0.5)
        pmi = scorer.pmi_smoothed()
        for pair, counts in scorer.obs.items():
            if counts["11"] == 0:
                assert pmi[pair][0] < 0


class TestDegenerateSubsetConvention:
    """Theil's U and lambda both score a trivially predictable direction as 1."""

    def test_lambda_matches_theil_u_on_a_degenerate_subset(self):
        scorer = CatScorer(PAIRS)
        # In the subset of ("q", "k") the sound is always /k/.
        assert scorer.goodman_kruskal_lambda()[("q", "k")] == pytest.approx((1.0, 0.0))
        assert scorer.theil_u()[("q", "k")] == pytest.approx((1.0, 0.0), abs=1e-12)

    def test_scalar_lambda_of_a_constant_variable_is_one(self):
        xs, ys = ["a", "b", "c"], ["z", "z", "z"]
        assert compute_goodman_kruskal_lambda(xs, ys, "y_given_x") == 1.0
        assert compute_theil_u(ys, xs) == 1.0


class TestNonSquareTableDeprecation:
    """`square_ct=False` still works, with a DeprecationWarning."""

    @pytest.mark.parametrize(
        "measure", ["chi2", "chi2_pvalue", "cramers_v", "log_likelihood_ratio", "log_likelihood_ratio_pvalue"]
    )
    def test_square_ct_false_warns(self, measure):
        scorer = CatScorer(PAIRS)
        with pytest.deprecated_call(match="square_ct=False is deprecated"):
            getattr(scorer, measure)(square_ct=False)

    def test_default_table_does_not_warn(self, recwarn):
        CatScorer(PAIRS).chi2()
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
