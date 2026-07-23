"""
Tests for the permutation-based significance scorer.

``CatScorer.permutation_pvalue`` builds a null distribution for any association
measure by repeatedly shuffling the pairing between the ``x`` and ``y`` elements
of the co-occurrences (preserving both marginals). These tests cover
reproducibility, the value range and ``(count + 1) / (n + 1)`` bound, key
alignment, the ``alternative`` tails, the response to strong vs. independent
association, and input validation.
"""

import pytest

import asymcat
from asymcat.scorer import CatScorer

# Strongly associated: 'a' almost always with 'x', 'b' almost always with 'y'.
STRONG_DATA = [[["a"], ["x"]]] * 40 + [[["b"], ["y"]]] * 40 + [[["a"], ["y"]]] * 2
# Independent: every x co-occurs with every y equally often.
INDEPENDENT_DATA = [[["a", "b"], ["x", "y"]], [["a", "b"], ["y", "x"]]] * 15


def _strong_scorer():
    return CatScorer(asymcat.collect_cooccs(STRONG_DATA))


def _independent_scorer():
    return CatScorer(asymcat.collect_cooccs(INDEPENDENT_DATA))


class TestPermutationStructure:
    """Reproducibility, ranges, and output shape."""

    def test_is_reproducible_with_seed(self):
        first = _strong_scorer().permutation_pvalue("theil_u", n_permutations=100, seed=42)
        second = _strong_scorer().permutation_pvalue("theil_u", n_permutations=100, seed=42)
        assert first == second

    def test_different_seeds_still_valid(self):
        scorer = _strong_scorer()
        a = scorer.permutation_pvalue("pmi", n_permutations=100, seed=1)
        b = scorer.permutation_pvalue("pmi", n_permutations=100, seed=2)
        assert set(a) == set(b)  # same keys regardless of seed

    @pytest.mark.parametrize("measure", ["mle", "pmi", "theil_u", "mutual_information", "chi2"])
    def test_values_in_valid_range(self, measure):
        n = 100
        pvalues = _strong_scorer().permutation_pvalue(measure, n_permutations=n, seed=0)
        minimum = 1.0 / (n + 1)
        for p_xy, p_yx in pvalues.values():
            assert minimum <= p_xy <= 1.0
            assert minimum <= p_yx <= 1.0

    def test_keys_match_measure_output(self):
        scorer = _strong_scorer()
        pvalues = scorer.permutation_pvalue("theil_u", n_permutations=50, seed=0)
        assert set(pvalues) == set(scorer.theil_u())

    def test_smallest_pvalue_is_bounded_by_permutation_count(self):
        n = 50
        pvalues = _strong_scorer().permutation_pvalue("pmi", n_permutations=n, seed=3)
        flat = [p for pair in pvalues.values() for p in pair]
        assert min(flat) == pytest.approx(1.0 / (n + 1))


class TestPermutationBehavior:
    """The test must react to the presence or absence of association."""

    def test_strong_association_is_significant(self):
        # PMI is symmetric and high for the (a, x) association; upper tail.
        p = _strong_scorer().permutation_pvalue("pmi", n_permutations=200, seed=7)
        assert p[("a", "x")][0] < 0.05

    def test_independent_data_is_not_significant(self):
        p = _independent_scorer().permutation_pvalue("pmi", n_permutations=200, seed=7)
        # No pair should look strongly significant on independent data.
        assert all(p_xy > 0.05 for p_xy, _ in p.values())

    def test_less_tail_for_conditional_entropy(self):
        # Low conditional entropy signals association, so significance lives in
        # the lower tail; the upper tail must then be non-significant.
        scorer = _strong_scorer()
        less = scorer.permutation_pvalue("cond_entropy", n_permutations=200, alternative="less", seed=7)
        greater = scorer.permutation_pvalue("cond_entropy", n_permutations=200, alternative="greater", seed=7)
        assert less[("a", "x")][0] < 0.05
        assert greater[("a", "x")][0] > 0.5

    def test_two_sided_is_at_most_twice_one_sided(self):
        scorer = _strong_scorer()
        greater = scorer.permutation_pvalue("pmi", n_permutations=200, alternative="greater", seed=11)
        two_sided = scorer.permutation_pvalue("pmi", n_permutations=200, alternative="two-sided", seed=11)
        for pair in greater:
            assert two_sided[pair][0] <= min(1.0, 2.0 * greater[pair][0]) + 1e-12


class TestPermutationValidation:
    """Input validation."""

    def test_rejects_unknown_measure(self):
        with pytest.raises(ValueError, match="Unsupported measure"):
            _strong_scorer().permutation_pvalue("not_a_measure")

    def test_rejects_invalid_alternative(self):
        with pytest.raises(ValueError, match="Invalid alternative"):
            _strong_scorer().permutation_pvalue("pmi", alternative="sideways")

    def test_rejects_non_positive_permutations(self):
        with pytest.raises(ValueError, match="n_permutations must be a positive integer"):
            _strong_scorer().permutation_pvalue("pmi", n_permutations=0)

    def test_resamplable_measures_are_real_methods(self):
        scorer = _strong_scorer()
        for measure in CatScorer.RESAMPLABLE_MEASURES:
            assert callable(getattr(scorer, measure))
